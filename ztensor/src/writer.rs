//! `.zt` writing.
//!
//! Append-only: magic, then blobs at aligned offsets, then the manifest blob,
//! then the footer. The default mode produces **canonical form** (spec §6.3):
//! 64 KiB placement, sorted insertion, per-part xxh3 digests, and blob sharing
//! for byte-identical parts.
//!
//! One way in. [`Writer::object`] builds any object — dense or not, one part
//! or several, bytes in hand or streamed a chunk at a time, local or a
//! reference into another file — and [`Writer::add`] is the one-liner over it
//! for the case that is almost all of them.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use xxhash_rust::xxh3::{xxh3_128, xxh3_64, Xxh3};

use crate::cbor;
use crate::error::{Error, Result};
use crate::schema::{
    check_attributes, check_digest, check_name, check_shard_name, BlobRef, DType, Manifest, Object,
    Part, Shard, ALIGN_CANONICAL, ALIGN_FLOOR, FOOTER_LEN, MAGIC, MAX_MANIFEST_LEN, MAX_RANK,
    MIN_FILE_LEN, VERSION,
};
use crate::source::Source;
use crate::vocab::Vocabulary;

/// Writer-side violations of reader rules surface as `InvalidInput` carrying
/// the rule's own message.
fn invalid(e: Error) -> Error {
    match e {
        Error::Reject { detail, .. } => Error::InvalidInput(detail),
        other => other,
    }
}

fn check_alignment(align: u64) -> Result<()> {
    if !align.is_power_of_two() || align < ALIGN_FLOOR {
        return Err(Error::InvalidInput(format!(
            "alignment must be a power of two >= {ALIGN_FLOOR}, got {align}"
        )));
    }
    Ok(())
}

// =======================================================================
// options
// =======================================================================

/// How to write.
pub struct Options {
    canonical: bool,
    align: Option<u64>,
    vocab: Option<Arc<Vocabulary>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            canonical: true,
            align: None,
            vocab: None,
        }
    }
}

impl Options {
    /// Canonical form (the default): 64 KiB placement, ascending insertion,
    /// a digest on every part, raw parts only, single file.
    ///
    /// Turn it off to choose your own placement, insert in any order, encode
    /// parts, or reference other files.
    pub fn canonical(mut self, canonical: bool) -> Self {
        self.canonical = canonical;
        self
    }

    /// Placement alignment: a power of two ≥ 4096.
    pub fn align(mut self, align: u64) -> Self {
        self.align = Some(align);
        self
    }

    pub fn vocabulary(mut self, vocab: &Vocabulary) -> Self {
        self.vocab = Some(Arc::new(vocab.clone()));
        self
    }

    /// Writes to `path` directly.
    pub fn create(self, path: impl AsRef<Path>) -> Result<Writer> {
        self.build(path.as_ref().to_path_buf(), None)
    }

    /// Writes to a sibling partial file and moves it into place on
    /// [`Writer::finish`] — see [`Writer::publish`].
    pub fn publish(self, path: impl AsRef<Path>) -> Result<Writer> {
        let final_path = path.as_ref().to_path_buf();
        let partial = partial_path(&final_path);
        self.build(partial, Some(final_path))
    }

    fn build(self, path: PathBuf, publish_to: Option<PathBuf>) -> Result<Writer> {
        let align = match (self.canonical, self.align) {
            (true, None) => ALIGN_CANONICAL,
            (true, Some(a)) if a == ALIGN_CANONICAL => ALIGN_CANONICAL,
            (true, Some(a)) => {
                return Err(Error::InvalidInput(format!(
                    "canonical form places blobs at {ALIGN_CANONICAL}; got align({a}). \
                     Add .canonical(false) to choose your own alignment."
                )))
            }
            (false, Some(a)) => {
                check_alignment(a)?;
                a
            }
            (false, None) => ALIGN_FLOOR,
        };
        let file = File::create(&path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&MAGIC)?;
        Ok(Writer {
            out: Some(out),
            path,
            publish_to,
            offset: MAGIC.len() as u64,
            align,
            canonical: self.canonical,
            manifest: Manifest::default(),
            dedup: HashMap::new(),
            last_name: None,
            streaming: false,
            vocab: self.vocab.unwrap_or_else(Vocabulary::shared),
        })
    }
}

fn partial_path(final_path: &Path) -> PathBuf {
    let name = final_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out.zt".to_string());
    final_path.with_file_name(format!(".{name}.{}.partial", std::process::id()))
}

// =======================================================================
// writer
// =======================================================================

pub struct Writer {
    out: Option<BufWriter<File>>,
    path: PathBuf,
    /// Where the finished file belongs, when this writer publishes.
    publish_to: Option<PathBuf>,
    offset: u64,
    align: u64,
    canonical: bool,
    manifest: Manifest,
    /// (xxh3_128, length) -> offset of a previously written blob. Hash matches
    /// are confirmed byte-for-byte before sharing (§6.3 requires
    /// *byte-identical* sharing, and hashes can collide).
    dedup: HashMap<(u128, u64), u64>,
    last_name: Option<String>,
    /// Whether a [`Sink`] is open. Every other object-adding method refuses
    /// while one is: their bytes would land in the middle of the streamed part.
    streaming: bool,
    vocab: Arc<Vocabulary>,
}

impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer")
            .field("path", &self.path)
            .field("canonical", &self.canonical)
            .field("align", &self.align)
            .field("objects", &self.manifest.objects.len())
            .finish()
    }
}

impl Writer {
    /// A canonical-form writer over `path`.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Options::default().create(path)
    }

    /// A canonical-form writer that publishes atomically.
    ///
    /// Bytes go to a partial file beside `path`; [`finish`](Self::finish)
    /// flushes, fsyncs, and renames it into place, so a reader never sees a
    /// half-written checkpoint and a crash leaves no file at all. Dropping
    /// without finishing removes the partial.
    pub fn publish(path: impl AsRef<Path>) -> Result<Self> {
        Options::default().publish(path)
    }

    pub fn options() -> Options {
        Options::default()
    }

    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Sets file-level attributes.
    pub fn set_attributes(&mut self, attributes: cbor::Value) {
        self.manifest.attributes = Some(attributes);
    }

    /// The one-liner: a dense tensor with a single raw `"data"` part.
    ///
    /// `data` must be exactly `product(shape) × width(dtype)` bytes of
    /// little-endian elements.
    pub fn add(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Vec<u64>>,
        dtype: DType,
        data: &[u8],
    ) -> Result<()> {
        self.object(name)
            .shape(shape)
            .part("data")
            .dtype(dtype)
            .bytes(data)
            .add()
    }

    /// Begins any object. See [`ObjectBuilder`].
    pub fn object(&mut self, name: impl Into<String>) -> ObjectBuilder<'_, '_> {
        ObjectBuilder {
            writer: self,
            name: name.into(),
            shape: Vec::new(),
            layout: "dense".to_string(),
            attributes: Vec::new(),
            parts: Vec::new(),
            current: None,
            error: None,
        }
    }

    /// Registers an external shard under `name`.
    ///
    /// The name is a label you choose, and the only thing parts will use to
    /// refer to this shard; it never appears on disk as a path. It must match
    /// `[A-Za-z0-9._-]`, not start with `.`, and fit in 64 bytes (spec §7.1),
    /// so that a resolver can spend it as a path component safely.
    ///
    /// The identity is one value, and the two things that produce it —
    /// [`DataShardWriter::finish`] and
    /// [`shard_identity`](crate::shard_identity) — hand back exactly this.
    /// Canonical form is single-file (spec §6.3), so this needs
    /// `.canonical(false)`.
    pub fn add_shard(&mut self, name: impl Into<String>, shard: &Shard) -> Result<()> {
        if self.canonical {
            return Err(Error::InvalidInput(
                "canonical form is single-file; add .canonical(false)".into(),
            ));
        }
        let name = name.into();
        check_shard_name(&name)?;
        check_digest(&shard.digest).map_err(invalid)?;
        if shard.size < MIN_FILE_LEN {
            return Err(Error::InvalidInput(format!(
                "shard size {} below minimum file size",
                shard.size
            )));
        }
        if let Some(existing) = self.manifest.shards.get(&name) {
            if existing != shard {
                return Err(Error::InvalidInput(format!(
                    "shard {name:?} is already registered with a different identity"
                )));
            }
        }
        self.manifest.shards.insert(name, shard.clone());
        Ok(())
    }

    /// Overlay convenience: references every part of `object` (taken from
    /// another file's manifest) through the shard registered as `shard`,
    /// writing nothing. Parts must be local in the source manifest.
    pub fn link(&mut self, name: impl Into<String>, object: &Object, shard: &str) -> Result<()> {
        let name = name.into();
        let mut builder = self
            .object(&name)
            .shape(object.shape.clone())
            .layout(&object.layout);
        if let Some(attrs) = &object.attributes {
            builder = builder.attributes(attrs.clone());
        }
        for (pname, part) in &object.parts {
            if part.blob.shard.is_some() {
                return Err(Error::InvalidInput(format!(
                    "part {pname:?} is itself a foreign reference; only local parts can be linked"
                )));
            }
            let mut linked = part.clone();
            linked.blob.shard = Some(shard.to_string());
            builder = builder.part(pname).external(linked);
        }
        builder.add()
    }

    /// Copies every tensor of a [`Source`] into this file — the universal
    /// conversion path.
    ///
    /// Reads decoded bytes and writes them raw, so a canonical writer turns
    /// *any* source (a foreign format, another `.zt`, a merged snapshot) into
    /// a canonical, bit-reproducible `.zt` file. Tensors arrive in name order,
    /// which is what canonical insertion wants. File attributes are copied
    /// unless already set.
    pub fn ingest(&mut self, source: &Source) -> Result<()> {
        if self.manifest.attributes.is_none() {
            self.manifest.attributes = source.attributes().cloned();
        }
        for tensor in source.tensors() {
            // Owned first: the builder borrows the bytes, and a projection may
            // have to decode them.
            let mut payloads: Vec<(String, DType, Option<String>, Vec<u8>)> = Vec::new();
            for pname in tensor.parts() {
                let part = tensor.part(pname)?;
                payloads.push((
                    pname.to_string(),
                    part.dtype(),
                    part.logical().map(str::to_string),
                    part.bytes()?.into_owned(),
                ));
            }
            let mut builder = self
                .object(tensor.name())
                .shape(tensor.shape().to_vec())
                .layout(tensor.layout());
            if let Some(attrs) = tensor.attributes() {
                builder = builder.attributes(attrs.clone());
            }
            for (pname, dtype, logical, data) in &payloads {
                builder = builder.part(pname).dtype(*dtype).bytes(data);
                if let Some(l) = logical {
                    builder = builder.logical(l);
                }
            }
            builder.add()?;
        }
        Ok(())
    }

    /// Writes the manifest blob and footer, flushes, and — when this writer
    /// publishes — fsyncs and renames into place. Returns the file size.
    pub fn finish(mut self) -> Result<u64> {
        if self.streaming {
            return Err(Error::InvalidInput(
                "a streamed object is still open; close its sink before finishing".into(),
            ));
        }
        let manifest_bytes = cbor::encode(&self.manifest.to_value())?;
        if manifest_bytes.len() as u64 > MAX_MANIFEST_LEN {
            return Err(Error::InvalidInput("manifest exceeds 1 GiB".into()));
        }
        let manifest_offset = align_up(self.offset, self.align)?;
        self.pad_to(manifest_offset)?;
        self.write_bytes(&manifest_bytes)?;

        let mut footer = [0u8; FOOTER_LEN as usize];
        footer[0..8].copy_from_slice(&manifest_offset.to_le_bytes());
        footer[8..16].copy_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
        footer[16..24].copy_from_slice(&xxh3_64(&manifest_bytes).to_le_bytes());
        footer[24..28].copy_from_slice(&VERSION.to_le_bytes());
        // footer[28..32]: reserved, zero
        footer[32..40].copy_from_slice(&MAGIC);
        self.write_bytes(&footer)?;

        let out = self.out.take().expect("writer is open");
        let file = out.into_inner().map_err(|e| Error::Io(e.into_error()))?;
        file.sync_all().or_else(|e| {
            // Durability is only promised by publish(); a plain create() on a
            // filesystem that cannot sync is not a failure to write.
            if self.publish_to.is_some() {
                Err(Error::Io(e))
            } else {
                Ok(())
            }
        })?;
        drop(file);

        if let Some(final_path) = self.publish_to.take() {
            std::fs::rename(&self.path, &final_path)?;
        }
        Ok(self.offset)
    }

    /// Abandons the file: a publishing writer removes its partial, a plain one
    /// removes what it has written.
    pub fn abandon(mut self) {
        self.out = None;
        let _ = std::fs::remove_file(&self.path);
        self.publish_to = None;
    }

    // ---- blob placement ----

    fn out(&mut self) -> Result<&mut BufWriter<File>> {
        self.out
            .as_mut()
            .ok_or_else(|| Error::InvalidInput("writer is already finished".into()))
    }

    /// Writes a blob, or shares an existing one when the bytes are identical.
    /// §6.3 requires *byte-identical* sharing, so a hash match is confirmed by
    /// reading the candidate back — a collision must never alias two different
    /// tensors onto one blob.
    fn write_or_share_blob(&mut self, data: &[u8]) -> Result<u64> {
        let key = (xxh3_128(data), data.len() as u64);
        if let Some(&offset) = self.dedup.get(&key) {
            if self.blob_equals(offset, data)? {
                return Ok(offset);
            }
            // Collision: fall through and write the bytes separately.
        }
        let target = self.write_blob(data)?;
        self.dedup.entry(key).or_insert(target);
        Ok(target)
    }

    /// Compares an already-written blob against `data` (via the file, so no
    /// second copy of every blob is kept in memory).
    fn blob_equals(&mut self, offset: u64, data: &[u8]) -> Result<bool> {
        self.out()?.flush()?;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; data.len()];
        file.read_exact(&mut buf)?;
        Ok(buf == data)
    }

    fn write_blob(&mut self, data: &[u8]) -> Result<u64> {
        let target = align_up(self.offset, self.align)?;
        self.pad_to(target)?;
        self.write_bytes(data)?;
        Ok(target)
    }

    /// Advances to the next aligned offset without writing anything — where a
    /// streamed blob will begin.
    fn reserve_blob(&mut self) -> Result<u64> {
        let target = align_up(self.offset, self.align)?;
        self.pad_to(target)?;
        Ok(target)
    }

    fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.out()?.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(())
    }

    fn pad_to(&mut self, target: u64) -> Result<()> {
        const ZEROS: [u8; 4096] = [0u8; 4096];
        let mut gap = target - self.offset;
        while gap > 0 {
            let n = gap.min(ZEROS.len() as u64) as usize;
            self.out()?.write_all(&ZEROS[..n])?;
            gap -= n as u64;
        }
        self.offset = target;
        Ok(())
    }

    // ---- shared checks ----

    /// Canonical form requires NFC names (spec §6.3 rule 5). ASCII is
    /// trivially NFC; only non-ASCII names pay for the check.
    fn check_canonical_name(&self, name: &str) -> Result<()> {
        if self.canonical && !name.is_ascii() && !unicode_normalization::is_nfc(name) {
            return Err(Error::InvalidInput(format!(
                "canonical form requires NFC-normalized names, got {name:?}"
            )));
        }
        Ok(())
    }

    fn check_new_object(&self, name: &str, shape: &[u64]) -> Result<()> {
        if self.streaming {
            return Err(Error::InvalidInput(format!(
                "object {name:?} cannot be added while a streamed object is open"
            )));
        }
        check_name(name).map_err(invalid)?;
        self.check_canonical_name(name)?;
        if self.manifest.objects.contains_key(name) {
            return Err(Error::InvalidInput(format!("duplicate object {name:?}")));
        }
        if shape.len() > MAX_RANK {
            return Err(Error::InvalidInput(format!(
                "rank {} exceeds {MAX_RANK}",
                shape.len()
            )));
        }
        shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| Error::InvalidInput("shape product overflows u64".into()))?;
        if self.canonical {
            if let Some(last) = &self.last_name {
                if name <= last.as_str() {
                    return Err(Error::InvalidInput(format!(
                        "canonical form requires sorted insertion: {name:?} after {last:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn commit(&mut self, name: String, object: Object) {
        self.manifest.objects.insert(name.clone(), object);
        self.last_name = Some(name);
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        // A publishing writer that never finished leaves nothing behind: the
        // partial file is exactly the half-written state publish() exists to
        // keep out of the world.
        if self.publish_to.is_some() {
            self.out = None;
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn align_up(offset: u64, align: u64) -> Result<u64> {
    offset
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| Error::InvalidInput("file offset overflow".into()))
}

// =======================================================================
// object builder
// =======================================================================

enum Source_<'d> {
    /// Bytes in hand, decoded.
    Bytes(&'d [u8]),
    /// A length to be streamed later.
    Length(u64),
    /// A blob that already exists in another file.
    External(Part),
    Missing,
}

struct PartDraft<'d> {
    name: String,
    dtype: Option<DType>,
    logical: Option<String>,
    encoding: Option<String>,
    source: Source_<'d>,
}

/// Builds one object.
///
/// Parts are declared with [`part`](Self::part) and described by the setters
/// that follow it, which apply to the part most recently named:
///
/// ```no_run
/// # use ztensor::{DType, Writer};
/// # fn f(w: &mut Writer, blocks: &[u8], scales: &[u8]) -> ztensor::Result<()> {
/// w.object("q")
///     .shape([4096u64, 4096])
///     .layout("zt.quant_group/1")
///     .attr("group", 32u64)
///     .part("data").dtype(DType::U8).logical("f4_e2m1").bytes(blocks)
///     .part("scales").dtype(DType::U8).logical("f8_e8m0").bytes(scales)
///     .add()
/// # }
/// ```
///
/// End with [`add`](Self::add) for parts whose bytes are in hand, or
/// [`stream`](Self::stream) for parts declared by length.
pub struct ObjectBuilder<'w, 'd> {
    writer: &'w mut Writer,
    name: String,
    shape: Vec<u64>,
    layout: String,
    attributes: Vec<(cbor::Value, cbor::Value)>,
    parts: Vec<PartDraft<'d>>,
    current: Option<PartDraft<'d>>,
    error: Option<Error>,
}

impl<'w, 'd> ObjectBuilder<'w, 'd> {
    pub fn shape(mut self, shape: impl Into<Vec<u64>>) -> Self {
        self.shape = shape.into();
        self
    }

    /// Layout profile id. Defaults to `"dense"`.
    pub fn layout(mut self, layout: impl Into<String>) -> Self {
        self.layout = layout.into();
        self
    }

    /// One object-level attribute.
    pub fn attr(mut self, key: impl Into<String>, value: impl Into<cbor::Value>) -> Self {
        self.attributes
            .push((cbor::Value::Text(key.into()), value.into()));
        self
    }

    /// Object-level attributes, wholesale. Replaces anything set by
    /// [`attr`](Self::attr).
    pub fn attributes(mut self, attributes: cbor::Value) -> Self {
        match attributes {
            cbor::Value::Map(entries) => self.attributes = entries,
            other => {
                self.attributes = vec![];
                self.fail(Error::InvalidInput(format!(
                    "attributes must be a map, got {other:?}"
                )));
            }
        }
        self
    }

    /// Begins a part. Setters after this one describe it.
    pub fn part(mut self, name: impl Into<String>) -> Self {
        self.flush();
        self.current = Some(PartDraft {
            name: name.into(),
            dtype: None,
            logical: None,
            encoding: None,
            source: Source_::Missing,
        });
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.with_current("dtype", |p| p.dtype = Some(dtype));
        self
    }

    /// Logical type id: `"bool"`, `"f8_e4m3fn"`, `"f4_e2m1"`, ...
    pub fn logical(mut self, logical: impl Into<String>) -> Self {
        let logical = logical.into();
        self.with_current("logical", |p| p.logical = Some(logical));
        self
    }

    /// Stores this part through an encoding profile. Canonical form is raw by
    /// definition, so this needs `.canonical(false)`.
    pub fn encoding(mut self, encoding: impl Into<String>) -> Self {
        let encoding = encoding.into();
        self.with_current("encoding", |p| p.encoding = Some(encoding));
        self
    }

    /// The part's decoded bytes.
    pub fn bytes(mut self, data: &'d [u8]) -> Self {
        self.with_current("bytes", |p| p.source = Source_::Bytes(data));
        self
    }

    /// The part's byte length, to be streamed. See [`stream`](Self::stream).
    pub fn length(mut self, length: u64) -> Self {
        self.with_current("length", |p| p.source = Source_::Length(length));
        self
    }

    /// A blob that already exists in a registered shard: nothing is written
    /// for this part.
    pub fn external(mut self, part: Part) -> Self {
        self.with_current("external", |p| p.source = Source_::External(part));
        self
    }

    fn with_current(&mut self, what: &str, f: impl FnOnce(&mut PartDraft<'d>)) {
        match &mut self.current {
            Some(part) => f(part),
            None => self.fail(Error::InvalidInput(format!(
                "object {:?}: .{what}() applies to a part; call .part(name) first",
                self.name
            ))),
        }
    }

    fn fail(&mut self, error: Error) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    fn flush(&mut self) {
        if let Some(part) = self.current.take() {
            self.parts.push(part);
        }
    }

    /// Validates the object's metadata and produces its parts in name order,
    /// with offsets left at zero.
    fn build(&mut self) -> Result<(Object, Vec<PartDraft<'d>>)> {
        self.flush();
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        let writer = &*self.writer;
        writer.check_new_object(&self.name, &self.shape)?;
        if self.parts.is_empty() {
            return Err(Error::InvalidInput(format!(
                "object {:?} has no parts",
                self.name
            )));
        }
        let attributes = if self.attributes.is_empty() {
            None
        } else {
            let value = cbor::Value::Map(std::mem::take(&mut self.attributes));
            check_attributes(&value).map_err(invalid)?;
            Some(value)
        };

        // Parts in name order: canonical blob order is (object, part).
        let mut drafts = std::mem::take(&mut self.parts);
        drafts.sort_by(|a, b| a.name.cmp(&b.name));

        let mut built: BTreeMap<String, Part> = BTreeMap::new();
        for draft in &drafts {
            check_name(&draft.name).map_err(invalid)?;
            writer.check_canonical_name(&draft.name)?;
            // An external part already states its own dtype and logical type;
            // making the caller repeat them would only create a way to
            // disagree with the file being referenced.
            let external = match &draft.source {
                Source_::External(part) => Some(part),
                _ => None,
            };
            let dtype = draft
                .dtype
                .or_else(|| external.map(|p| p.dtype))
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "object {:?} part {:?}: no dtype",
                        self.name, draft.name
                    ))
                })?;
            let logical = draft
                .logical
                .clone()
                .or_else(|| external.and_then(|p| p.logical.clone()));
            if let Some(logical) = &logical {
                if let Some(required) = writer.vocab.dtype_of(logical) {
                    if dtype != required {
                        return Err(Error::InvalidInput(format!(
                            "part {:?}: type {logical:?} requires dtype {required:?}",
                            draft.name
                        )));
                    }
                }
            }
            let part = match &draft.source {
                Source_::Missing => {
                    return Err(Error::InvalidInput(format!(
                        "object {:?} part {:?}: no bytes, length, or external blob",
                        self.name, draft.name
                    )))
                }
                Source_::External(part) => {
                    let mut part = part.clone();
                    part.dtype = dtype;
                    part.logical = logical.clone();
                    validate_external(&writer.manifest, &draft.name, &part)?;
                    part
                }
                Source_::Length(length) => Part {
                    dtype,
                    logical: logical.clone(),
                    blob: BlobRef {
                        shard: None,
                        offset: 0,
                        length: *length,
                    },
                    encoding: None,
                    decoded_length: None,
                    digest: None, // computed from the streamed bytes
                },
                Source_::Bytes(data) => {
                    let (length, encoding, decoded_length) = match &draft.encoding {
                        None => (data.len() as u64, None, None),
                        Some(id) => {
                            if writer.canonical {
                                return Err(Error::InvalidInput(
                                    "canonical form forbids encoded parts; add .canonical(false)"
                                        .into(),
                                ));
                            }
                            let profile = writer.vocab.encoding(id).ok_or_else(|| {
                                Error::Unsupported(format!(
                                    "encoding profile {id:?} is not registered"
                                ))
                            })?;
                            // Encoded once here to learn the stored length; the
                            // bytes are produced again when written. Encoding is
                            // the rare path, and holding every encoded payload
                            // for a whole checkpoint is the thing to avoid.
                            let stored = profile.encode(data)?;
                            (
                                stored.len() as u64,
                                Some(id.clone()),
                                Some(data.len() as u64),
                            )
                        }
                    };
                    Part {
                        dtype,
                        logical: logical.clone(),
                        blob: BlobRef {
                            shard: None,
                            offset: 0,
                            length,
                        },
                        encoding,
                        decoded_length,
                        digest: Some(format!("xxh3:{:016x}", xxh3_64(data))),
                    }
                }
            };
            if built.insert(draft.name.clone(), part).is_some() {
                return Err(Error::InvalidInput(format!(
                    "duplicate part {:?}",
                    draft.name
                )));
            }
        }

        let object = Object {
            shape: std::mem::take(&mut self.shape),
            layout: std::mem::take(&mut self.layout),
            attributes,
            parts: built,
        };
        if let Some(profile) = writer.vocab.layout(&object.layout) {
            profile
                .validate(&self.name, &object, &writer.vocab)
                .map_err(invalid)?;
        }
        Ok((object, drafts))
    }

    /// Writes the object. Every part must have bytes or be external.
    pub fn add(mut self) -> Result<()> {
        let (mut object, drafts) = self.build()?;
        if drafts
            .iter()
            .any(|d| matches!(d.source, Source_::Length(_)))
        {
            return Err(Error::InvalidInput(format!(
                "object {:?} declares a streamed part; end with .stream() instead of .add()",
                self.name
            )));
        }

        // `object.parts` and `drafts` are both in part-name order.
        for (part, draft) in object.parts.values_mut().zip(&drafts) {
            match &draft.source {
                Source_::Bytes(data) => {
                    part.blob.offset = match (&draft.encoding, &part.encoding) {
                        (Some(id), Some(_)) => {
                            let profile = self
                                .writer
                                .vocab
                                .encoding(id)
                                .expect("checked while building");
                            let stored = profile.encode(data)?;
                            self.writer.write_blob(&stored)?
                        }
                        _ => self.writer.write_or_share_blob(data)?,
                    };
                }
                Source_::External(_) => {} // offsets belong to the other file
                _ => unreachable!("checked above"),
            }
        }
        let name = std::mem::take(&mut self.name);
        self.writer.commit(name, object);
        Ok(())
    }

    /// Opens the object for streaming. Every part must have been declared with
    /// [`length`](Self::length).
    ///
    /// Parts are written in name order, each receiving exactly the byte count
    /// it declared. The returned [`Sink`] is a token, not a borrow: it is
    /// passed back to [`Sink::write`] and consumed by [`Sink::close`], so a
    /// producer driven from outside — one chunk per call, holding both the
    /// writer and the open object in one structure — can exist at all.
    pub fn stream(mut self) -> Result<Sink> {
        let (object, drafts) = self.build()?;
        if !drafts
            .iter()
            .all(|d| matches!(d.source, Source_::Length(_)))
        {
            return Err(Error::InvalidInput(format!(
                "object {:?}: streaming declares every part with .length(); \
                 use .add() for parts whose bytes are in hand",
                self.name
            )));
        }
        self.writer.streaming = true;
        Ok(Sink {
            name: std::mem::take(&mut self.name),
            order: object.parts.keys().cloned().collect(),
            object: Some(object),
            at: 0,
            written: 0,
            hasher: Xxh3::new(),
            started: false,
        })
    }
}

fn validate_external(manifest: &Manifest, pname: &str, part: &Part) -> Result<()> {
    let b = &part.blob;
    let Some(sname) = &b.shard else {
        return Err(Error::InvalidInput(format!(
            "part {pname:?} is an external reference but names no shard"
        )));
    };
    let shard = manifest.shards.get(sname).ok_or_else(|| {
        Error::InvalidInput(format!(
            "part {pname:?} references unregistered shard {sname:?}"
        ))
    })?;
    if !b.offset.is_multiple_of(ALIGN_FLOOR) || b.offset < ALIGN_FLOOR {
        return Err(Error::InvalidInput(format!(
            "part {pname:?}: offset {} violates the {ALIGN_FLOOR} floor",
            b.offset
        )));
    }
    let region_end = shard.size - FOOTER_LEN;
    if b.offset
        .checked_add(b.length)
        .is_none_or(|e| e > region_end)
    {
        return Err(Error::InvalidInput(format!(
            "part {pname:?}: blob outside shard {sname:?}'s data region"
        )));
    }
    if let Some(d) = &part.digest {
        check_digest(d).map_err(invalid)?;
    }
    if part.encoding.is_some() != part.decoded_length.is_some() {
        return Err(Error::InvalidInput(format!(
            "part {pname:?}: decoded_length is required iff encoding is present"
        )));
    }
    Ok(())
}

// =======================================================================
// streaming
// =======================================================================

/// An open streamed object.
///
/// Dropping without [`close`](Self::close) leaves the object out of the
/// manifest and the writer refusing further objects: the bytes already written
/// stay in the file as unreferenced blobs, which the format allows (§2.5) but
/// canonical form does not, so there is no honest way to carry on.
pub struct Sink {
    name: String,
    object: Option<Object>,
    /// Part names, in the order they must be written.
    order: Vec<String>,
    at: usize,
    written: u64,
    hasher: Xxh3,
    started: bool,
}

impl std::fmt::Debug for Sink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sink")
            .field("object", &self.name)
            .field("part", &self.current())
            .field("written", &self.written)
            .finish()
    }
}

impl Sink {
    /// The part currently being written, or `None` when every part is done.
    pub fn current(&self) -> Option<&str> {
        self.order.get(self.at).map(String::as_str)
    }

    /// Bytes written into the current part so far.
    pub fn written(&self) -> u64 {
        self.written
    }

    /// Appends bytes to the current part.
    ///
    /// The first call for a part places it at the next aligned offset. Writing
    /// past a part's declared length is an error rather than a rollover into
    /// the next one: a producer that has miscounted should hear about it where
    /// it happened.
    pub fn write(&mut self, writer: &mut Writer, chunk: &[u8]) -> Result<()> {
        if !writer.streaming {
            return Err(Error::InvalidInput(format!(
                "object {:?} is not open on this writer",
                self.name
            )));
        }
        let Some(part_name) = self.order.get(self.at).cloned() else {
            return Err(Error::InvalidInput(format!(
                "object {:?}: every part is already written",
                self.name
            )));
        };
        let object = self.object.as_mut().expect("object present");
        let declared = object.parts[&part_name].blob.length;

        if !self.started {
            let offset = writer.reserve_blob()?;
            object
                .parts
                .get_mut(&part_name)
                .expect("part present")
                .blob
                .offset = offset;
            self.started = true;
            self.hasher = Xxh3::new();
            self.written = 0;
        }

        let end = self
            .written
            .checked_add(chunk.len() as u64)
            .filter(|&e| e <= declared)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "object {:?} part {part_name:?}: {} bytes written into a part \
                     declared as {declared}",
                    self.name,
                    self.written + chunk.len() as u64
                ))
            })?;

        writer.write_bytes(chunk)?;
        self.hasher.update(chunk);
        self.written = end;

        if self.written == declared {
            let digest = format!("xxh3:{:016x}", self.hasher.digest());
            object
                .parts
                .get_mut(&part_name)
                .expect("part present")
                .digest = Some(digest);
            self.at += 1;
            self.started = false;
        }
        Ok(())
    }

    /// Completes the object and adds it to the manifest.
    pub fn close(mut self, writer: &mut Writer) -> Result<()> {
        if self.at < self.order.len() {
            let part = &self.order[self.at];
            let declared = self.object.as_ref().expect("object present").parts[part]
                .blob
                .length;
            return Err(Error::InvalidInput(format!(
                "object {:?} part {part:?}: {} of {declared} bytes written",
                self.name, self.written
            )));
        }
        let object = self.object.take().expect("object present");
        writer.commit(std::mem::take(&mut self.name), object);
        writer.streaming = false;
        Ok(())
    }
}

// =======================================================================
// data shards
// =======================================================================

/// Writes a data shard (spec §7.2): magic, aligned blobs, and a footer with no
/// manifest. `finish` returns the shard's identity — exactly what
/// [`Writer::add_shard`] wants.
///
/// The whole-file digest is computed while writing, so producing a shard costs
/// one pass.
pub struct DataShardWriter {
    out: BufWriter<File>,
    hasher: Xxh3,
    offset: u64,
    align: u64,
}

impl DataShardWriter {
    /// Creates a data shard with canonical (64 KiB) placement.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Self::create_with_alignment(path, ALIGN_CANONICAL)
    }

    pub fn create_with_alignment(path: impl AsRef<Path>, align: u64) -> Result<Self> {
        check_alignment(align)?;
        let file = File::create(path)?;
        let mut shard = Self {
            out: BufWriter::with_capacity(1 << 20, file),
            hasher: Xxh3::new(),
            offset: 0,
            align,
        };
        shard.put(&MAGIC)?;
        Ok(shard)
    }

    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.out.write_all(bytes)?;
        self.hasher.update(bytes);
        self.offset += bytes.len() as u64;
        Ok(())
    }

    /// Writes one blob at the next aligned offset and returns that offset —
    /// the caller records it for the root's blob references.
    pub fn add_blob(&mut self, data: &[u8]) -> Result<u64> {
        const ZEROS: [u8; 4096] = [0u8; 4096];
        let target = align_up(self.offset, self.align)?;
        let mut gap = target - self.offset;
        while gap > 0 {
            let n = gap.min(ZEROS.len() as u64) as usize;
            self.put(&ZEROS[..n])?;
            gap -= n as u64;
        }
        self.put(data)?;
        Ok(target)
    }

    /// Writes the manifest-less footer and returns the shard's identity.
    pub fn finish(mut self) -> Result<Shard> {
        let mut footer = [0u8; FOOTER_LEN as usize];
        footer[24..28].copy_from_slice(&VERSION.to_le_bytes());
        footer[32..40].copy_from_slice(&MAGIC);
        self.put(&footer)?;
        self.out.flush()?;
        Ok(Shard {
            size: self.offset,
            digest: format!("xxh3:{:016x}", self.hasher.digest()),
        })
    }
}
