//! `.zt` v2 writer.
//!
//! Append-only: magic, then blobs at aligned offsets, then the manifest
//! blob, then the footer. The default mode produces **canonical form**
//! (spec §6.3): 64 KiB placement, sorted insertion, per-part xxh3 digests,
//! and blob sharing for byte-identical parts.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use xxhash_rust::xxh3::{xxh3_128, xxh3_64};

use crate::cbor;
use crate::error::{Error, Result};
use crate::models::{
    check_attributes, check_digest, check_name, registered_dtype, BlobRef, DType, Layout,
    Manifest, Object, Part, Shard, ALIGN_CANONICAL, ALIGN_FLOOR, FOOTER_LEN, MAGIC,
    MAX_MANIFEST_LEN, MAX_RANK, MIN_FILE_LEN, VERSION,
};
use crate::profiles::{encoding_profile, layout_profile};

/// One part of an object handed to [`Writer::add_object`]. `data` is the
/// decoded bytes; `encoding` (if any) names the profile the writer applies.
#[derive(Debug, Clone, Copy)]
pub struct PartDef<'a> {
    pub dtype: DType,
    pub ltype: Option<&'a str>,
    pub encoding: Option<&'a str>,
    pub data: &'a [u8],
}

/// One part of an object written incrementally by
/// [`Writer::stream_object`]. The length is declared up front because the
/// manifest records it and the writer places the next blob after it.
#[derive(Debug, Clone, Copy)]
pub struct StreamPart<'a> {
    pub name: &'a str,
    pub dtype: DType,
    pub ltype: Option<&'a str>,
    pub length: u64,
}

pub struct Writer {
    out: BufWriter<File>,
    path: std::path::PathBuf,
    offset: u64,
    align: u64,
    canonical: bool,
    manifest: Manifest,
    /// (xxh3_128, length) -> offset of a previously written blob. Hash
    /// matches are confirmed byte-for-byte before sharing (§6.3 requires
    /// *byte-identical* sharing, and hashes can collide).
    dedup: HashMap<(u128, u64), u64>,
    last_name: Option<String>,
    /// Whether an [`ObjectWriter`] is open. Every other object-adding method
    /// refuses while one is: their bytes would land in the middle of the
    /// part being streamed.
    streaming: bool,
}

/// Writer-side violations of reader rules surface as `InvalidInput`
/// carrying the rule's own message.
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

impl Writer {
    /// Creates a canonical-form writer (64 KiB placement, sorted insertion
    /// required, digests on every part, identical parts share one blob).
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Self::new(path, ALIGN_CANONICAL, true)
    }

    /// Creates a non-canonical writer with custom placement alignment
    /// (power of two, ≥ 4096). Insertion order is preserved as-is.
    pub fn create_with_alignment(path: impl AsRef<Path>, align: u64) -> Result<Self> {
        check_alignment(align)?;
        Self::new(path, align, false)
    }

    fn new(path: impl AsRef<Path>, align: u64, canonical: bool) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&MAGIC)?;
        Ok(Self {
            out,
            path,
            offset: MAGIC.len() as u64,
            align,
            canonical,
            manifest: Manifest::default(),
            dedup: HashMap::new(),
            last_name: None,
            streaming: false,
        })
    }

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

    /// Shared preamble of every object-adding method: name rules, duplicate
    /// check, shape rules, canonical insertion order.
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

    /// Adds a dense object with a single raw `"data"` part.
    ///
    /// `data` must be exactly `product(shape) × width(dtype)` bytes of
    /// little-endian elements. In canonical mode, objects must be added in
    /// ascending name order, and names should already be NFC-normalized
    /// (the writer does not normalize).
    pub fn add_dense(
        &mut self,
        name: &str,
        shape: &[u64],
        dtype: DType,
        data: &[u8],
    ) -> Result<()> {
        self.add_object(
            name,
            shape,
            "dense",
            &[(
                "data",
                PartDef {
                    dtype,
                    ltype: None,
                    encoding: None,
                    data,
                },
            )],
            None,
        )
    }

    /// Adds an object with arbitrary layout and parts. `data` in each
    /// [`PartDef`] is the *decoded* bytes; when `encoding` names a known
    /// profile the writer encodes them (non-canonical files only —
    /// canonical form is raw, spec §6.3).
    ///
    /// Known layouts are validated before anything is written; unknown
    /// layout ids are accepted as-is (the caller owns their profile spec).
    pub fn add_object(
        &mut self,
        name: &str,
        shape: &[u64],
        layout: &str,
        parts: &[(&str, PartDef)],
        attributes: Option<cbor::Value>,
    ) -> Result<()> {
        self.check_new_object(name, shape)?;
        if parts.is_empty() {
            return Err(Error::InvalidInput(format!("object {name:?} has no parts")));
        }
        if let Some(attrs) = &attributes {
            check_attributes(attrs).map_err(invalid)?;
        }

        // Encode payloads and build part metadata (offsets patched after
        // writing), so known layouts can be validated before any byte is
        // written. Parts are processed in name order — canonical blob order
        // is (object name, part name).
        let mut built: BTreeMap<String, Part> = BTreeMap::new();
        // Raw payloads are borrowed, not copied: a checkpoint's worth of
        // tensor bytes is the one thing worth not duplicating.
        let mut payloads: Vec<std::borrow::Cow<'_, [u8]>> = Vec::with_capacity(parts.len());
        let mut order: Vec<usize> = (0..parts.len()).collect();
        order.sort_by_key(|&i| parts[i].0);
        for &i in &order {
            let (pname, def) = &parts[i];
            check_name(pname).map_err(invalid)?;
            self.check_canonical_name(pname)?;
            if let Some(lt) = def.ltype {
                if let Some(required) = registered_dtype(lt) {
                    if def.dtype != required {
                        return Err(Error::InvalidInput(format!(
                            "part {pname:?}: type {lt:?} requires dtype {required:?}"
                        )));
                    }
                }
            }
            let (stored, encoding, decoded_length) = match def.encoding {
                None => (std::borrow::Cow::Borrowed(def.data), None, None),
                Some(enc) => {
                    if self.canonical {
                        return Err(Error::InvalidInput(
                            "canonical form forbids encoded parts; use create_with_alignment"
                                .into(),
                        ));
                    }
                    let profile = encoding_profile(enc).ok_or_else(|| {
                        Error::Unsupported(format!("unknown encoding profile {enc:?}"))
                    })?;
                    (
                        std::borrow::Cow::Owned(profile.encode(def.data)?),
                        Some(enc.to_string()),
                        Some(def.data.len() as u64),
                    )
                }
            };
            let part = Part {
                dtype: def.dtype,
                ltype: def.ltype.map(str::to_string),
                blob: BlobRef {
                    shard: 0,
                    offset: 0, // patched below
                    length: stored.len() as u64,
                },
                encoding,
                decoded_length,
                digest: Some(format!("xxh3:{:016x}", xxh3_64(def.data))),
            };
            if built.insert(pname.to_string(), part).is_some() {
                return Err(Error::InvalidInput(format!("duplicate part {pname:?}")));
            }
            payloads.push(stored);
        }

        let mut obj = Object {
            shape: shape.to_vec(),
            layout: Layout::from_name(layout),
            attributes,
            parts: built,
        };
        if let Some(profile) = layout_profile(layout) {
            profile.validate(name, &obj).map_err(invalid)?;
        }

        // `obj.parts` and `payloads` are both in part-name order.
        for (part, stored) in obj.parts.values_mut().zip(&payloads) {
            part.blob.offset = if part.encoding.is_none() {
                self.write_or_share_blob(stored)?
            } else {
                self.write_blob(stored)?
            };
        }
        self.manifest.objects.insert(name.to_string(), obj);
        self.last_name = Some(name.to_string());
        Ok(())
    }

    /// Begins an object whose parts are written incrementally.
    ///
    /// The slice-taking [`add_object`](Self::add_object) needs every part's
    /// bytes in memory at once, which a producer streaming a tensor off a
    /// device cannot do — a weight store is tens of gigabytes and the host
    /// should never hold more than a chunk of it. This is the same object,
    /// written a chunk at a time.
    ///
    /// Parts are declared up front, in name order, and written in that order;
    /// the returned [`ObjectWriter`] enforces both. Streamed parts are raw
    /// (an encoding profile needs the whole payload) and are not deduplicated
    /// against earlier blobs, since dedup requires knowing the bytes before
    /// placing them.
    ///
    /// The [`ObjectWriter`] is a token, not a borrow: it is passed back to
    /// [`write_chunk`](Self::write_chunk) and consumed by
    /// [`end_object`](Self::end_object). That lets a caller hold an open
    /// object beside the writer in one structure — which is what a producer
    /// driven from the outside, a chunk per call, has to do.
    pub fn stream_object(
        &mut self,
        name: &str,
        shape: &[u64],
        layout: &str,
        parts: &[StreamPart<'_>],
        attributes: Option<cbor::Value>,
    ) -> Result<ObjectWriter> {
        self.check_new_object(name, shape)?;
        if parts.is_empty() {
            return Err(Error::InvalidInput(format!("object {name:?} has no parts")));
        }
        if let Some(attrs) = &attributes {
            check_attributes(attrs).map_err(invalid)?;
        }

        let mut ordered: Vec<&StreamPart<'_>> = parts.iter().collect();
        ordered.sort_by_key(|p| p.name);
        let mut built: BTreeMap<String, Part> = BTreeMap::new();
        for part in &ordered {
            check_name(part.name).map_err(invalid)?;
            self.check_canonical_name(part.name)?;
            if let Some(lt) = part.ltype {
                if let Some(required) = registered_dtype(lt) {
                    if part.dtype != required {
                        return Err(Error::InvalidInput(format!(
                            "part {:?}: type {lt:?} requires dtype {required:?}",
                            part.name
                        )));
                    }
                }
            }
            let entry = Part {
                dtype: part.dtype,
                ltype: part.ltype.map(str::to_string),
                blob: BlobRef {
                    shard: 0,
                    offset: 0, // patched as each part is written
                    length: part.length,
                },
                encoding: None,
                decoded_length: None,
                digest: None, // computed from the streamed bytes
            };
            if built.insert(part.name.to_string(), entry).is_some() {
                return Err(Error::InvalidInput(format!(
                    "duplicate part {:?}",
                    part.name
                )));
            }
        }

        let object = Object {
            shape: shape.to_vec(),
            layout: Layout::from_name(layout),
            attributes,
            parts: built,
        };
        if let Some(profile) = layout_profile(layout) {
            profile.validate(name, &object).map_err(invalid)?;
        }

        self.streaming = true;
        Ok(ObjectWriter {
            name: name.to_string(),
            object: Some(object),
            order: ordered.iter().map(|p| p.name.to_string()).collect(),
            at: 0,
            written: 0,
            hasher: xxhash_rust::xxh3::Xxh3::new(),
            started: false,
        })
    }

    /// Sets file-level attributes (an arbitrary CBOR map value).
    pub fn set_attributes(&mut self, attributes: cbor::Value) {
        self.manifest.attributes = Some(attributes);
    }

    /// Registers an external shard identity and returns its index (≥ 1).
    /// Canonical form is single-file (spec §6.3), so this requires a
    /// [`Writer::create_with_alignment`] writer.
    pub fn add_shard(&mut self, size: u64, digest: &str) -> Result<u64> {
        if self.canonical {
            return Err(Error::InvalidInput(
                "canonical form is single-file; use create_with_alignment".into(),
            ));
        }
        check_digest(digest).map_err(invalid)?;
        if size < MIN_FILE_LEN {
            return Err(Error::InvalidInput(format!(
                "shard size {size} below minimum file size"
            )));
        }
        let index = self
            .manifest
            .shards
            .last_key_value()
            .map(|(&k, _)| k + 1)
            .unwrap_or(1);
        self.manifest.shards.insert(
            index,
            Shard {
                size,
                digest: digest.to_string(),
            },
        );
        Ok(index)
    }

    /// Adds an object whose parts are pre-existing blob references into
    /// registered shards — nothing is written to this file. This is the
    /// overlay mechanism: a delta model references the base model's blobs.
    pub fn add_external_object(
        &mut self,
        name: &str,
        shape: &[u64],
        layout: &str,
        parts: &[(&str, Part)],
        attributes: Option<cbor::Value>,
    ) -> Result<()> {
        self.check_new_object(name, shape)?;
        if parts.is_empty() {
            return Err(Error::InvalidInput(format!("object {name:?} has no parts")));
        }
        if let Some(attrs) = &attributes {
            check_attributes(attrs).map_err(invalid)?;
        }
        let mut built = BTreeMap::new();
        for (pname, part) in parts {
            check_name(pname).map_err(invalid)?;
            self.check_canonical_name(pname)?;
            if let Some(lt) = &part.ltype {
                if let Some(required) = registered_dtype(lt) {
                    if part.dtype != required {
                        return Err(Error::InvalidInput(format!(
                            "part {pname:?}: type {lt:?} requires dtype {required:?}"
                        )));
                    }
                }
            }
            let b = part.blob;
            let shard = self.manifest.shards.get(&b.shard).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "part {pname:?} references unregistered shard {}",
                    b.shard
                ))
            })?;
            if b.offset % ALIGN_FLOOR != 0 || b.offset < ALIGN_FLOOR {
                return Err(Error::InvalidInput(format!(
                    "part {pname:?}: offset {} violates the 4096 floor",
                    b.offset
                )));
            }
            let region_end = shard.size - FOOTER_LEN;
            if b.offset.checked_add(b.length).is_none_or(|e| e > region_end) {
                return Err(Error::InvalidInput(format!(
                    "part {pname:?}: blob outside shard {}'s data region",
                    b.shard
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
            if built.insert(pname.to_string(), part.clone()).is_some() {
                return Err(Error::InvalidInput(format!("duplicate part {pname:?}")));
            }
        }
        let obj = Object {
            shape: shape.to_vec(),
            layout: Layout::from_name(layout),
            attributes,
            parts: built,
        };
        if let Some(profile) = layout_profile(layout) {
            profile.validate(name, &obj).map_err(invalid)?;
        }
        self.manifest.objects.insert(name.to_string(), obj);
        self.last_name = Some(name.to_string());
        Ok(())
    }

    /// Copies every object of a [`Source`] into this file — the universal
    /// conversion path. Reads decoded bytes tier-1 from the source and
    /// writes them raw, so a canonical writer turns *any* source (a foreign
    /// format projection, another `.zt`, an overlay model) into a
    /// canonical, tier-3, bit-reproducible `.zt` file.
    ///
    /// Objects arrive in name order (the manifest is sorted), satisfying
    /// canonical insertion. File attributes are copied unless already set.
    pub fn ingest(&mut self, src: &dyn crate::Source) -> Result<()> {
        let manifest = src.manifest();
        if self.manifest.attributes.is_none() {
            self.manifest.attributes = manifest.attributes.clone();
        }
        for (name, obj) in &manifest.objects {
            let mut payloads: Vec<(&str, DType, Option<&str>, Vec<u8>)> =
                Vec::with_capacity(obj.parts.len());
            for (pname, part) in &obj.parts {
                payloads.push((
                    pname,
                    part.dtype,
                    part.ltype.as_deref(),
                    src.read(name, pname)?,
                ));
            }
            let defs: Vec<(&str, PartDef)> = payloads
                .iter()
                .map(|(pname, dtype, ltype, data)| {
                    (
                        *pname,
                        PartDef {
                            dtype: *dtype,
                            ltype: *ltype,
                            encoding: None,
                            data,
                        },
                    )
                })
                .collect();
            self.add_object(name, &obj.shape, obj.layout.as_str(), &defs, obj.attributes.clone())?;
        }
        Ok(())
    }

    /// Overlay convenience: references every part of `obj` (an object from
    /// another file's manifest) through registered shard `shard`. Parts
    /// must be local (`shard 0`) in the source manifest.
    pub fn link_object(&mut self, name: &str, obj: &Object, shard: u64) -> Result<()> {
        let mut remapped: Vec<(&str, Part)> = Vec::with_capacity(obj.parts.len());
        for (pname, part) in &obj.parts {
            if part.blob.shard != 0 {
                return Err(Error::InvalidInput(format!(
                    "part {pname:?} is itself a foreign reference; only local parts can be linked"
                )));
            }
            let mut p = part.clone();
            p.blob.shard = shard;
            remapped.push((pname.as_str(), p));
        }
        self.add_external_object(
            name,
            &obj.shape,
            obj.layout.as_str(),
            &remapped,
            obj.attributes.clone(),
        )
    }

    /// Writes a blob, or shares an existing one when the bytes are
    /// identical. §6.3 requires *byte-identical* sharing, so a hash match
    /// is confirmed by reading the candidate back — a hash collision must
    /// never alias two different tensors onto one blob.
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

    /// Compares an already-written blob against `data` (via the file, so
    /// no second copy of every blob is kept in memory).
    fn blob_equals(&mut self, offset: u64, data: &[u8]) -> Result<bool> {
        self.out.flush()?;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; data.len()];
        file.read_exact(&mut buf)?;
        Ok(buf == data)
    }

    /// Advances to the next aligned offset and returns it, without writing
    /// anything — where a streamed blob will begin.
    fn reserve_blob(&mut self) -> Result<u64> {
        let target = align_up(self.offset, self.align)?;
        self.pad_to(target)?;
        Ok(target)
    }

    /// Appends bytes at the current position.
    fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
        self.out.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(())
    }

    fn write_blob(&mut self, data: &[u8]) -> Result<u64> {
        let target = align_up(self.offset, self.align)?;
        self.pad_to(target)?;
        self.out.write_all(data)?;
        self.offset = target + data.len() as u64;
        Ok(target)
    }

    fn pad_to(&mut self, target: u64) -> Result<()> {
        const ZEROS: [u8; 4096] = [0u8; 4096];
        let mut gap = target - self.offset;
        while gap > 0 {
            let n = gap.min(ZEROS.len() as u64) as usize;
            self.out.write_all(&ZEROS[..n])?;
            gap -= n as u64;
        }
        self.offset = target;
        Ok(())
    }

    /// Writes the manifest blob and footer, then flushes. Returns the total
    /// file size in bytes.
    pub fn finish(mut self) -> Result<u64> {
        if self.streaming {
            return Err(Error::InvalidInput(
                "a streamed object is still open; end_object it before finishing".into(),
            ));
        }
        let manifest_bytes = cbor::encode(&self.manifest.to_value())?;
        if manifest_bytes.len() as u64 > MAX_MANIFEST_LEN {
            return Err(Error::InvalidInput("manifest exceeds 1 GiB".into()));
        }
        let manifest_offset = align_up(self.offset, self.align)?;
        self.pad_to(manifest_offset)?;
        self.out.write_all(&manifest_bytes)?;
        self.offset += manifest_bytes.len() as u64;

        let mut footer = [0u8; FOOTER_LEN as usize];
        footer[0..8].copy_from_slice(&manifest_offset.to_le_bytes());
        footer[8..16].copy_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
        footer[16..24].copy_from_slice(&xxh3_64(&manifest_bytes).to_le_bytes());
        footer[24..28].copy_from_slice(&VERSION.to_le_bytes());
        // footer[28..32]: reserved, zero
        footer[32..40].copy_from_slice(&MAGIC);
        self.out.write_all(&footer)?;
        self.offset += FOOTER_LEN;

        self.out.flush()?;
        // No fsync here: per the spec (Appendix D) durable publication is
        // a transport concern — write to a temporary name, fsync, then
        // rename. Syncing unconditionally would tax every caller for a
        // guarantee only publishers need.
        Ok(self.offset)
    }
}

/// Writes one object's parts a chunk at a time.
///
/// Returned by [`Writer::stream_object`]. Parts are written in name order;
/// each must receive exactly the byte count it declared before the next
/// begins, and [`finish`](Self::finish) refuses an object with a part left
/// short. The digest of each part is computed from the bytes as they pass
/// through, so streaming costs no extra read.
///
/// Dropping without calling `end_object` leaves the object out of the
/// manifest and the writer refusing further objects: the bytes already
/// written stay in the file as unreferenced blobs, which the format allows
/// (§2.5) but canonical form does not, so there is no honest way to carry on.
pub struct ObjectWriter {
    name: String,
    object: Option<Object>,
    /// Part names, in the order they must be written.
    order: Vec<String>,
    /// Index into `order` of the part being written.
    at: usize,
    /// Bytes written into the current part.
    written: u64,
    hasher: xxhash_rust::xxh3::Xxh3,
    started: bool,
}

impl ObjectWriter {
    /// The part currently being written, or `None` when every part is done.
    pub fn current(&self) -> Option<&str> {
        self.order.get(self.at).map(String::as_str)
    }
}

impl Writer {
    /// Appends bytes to the current part of an open streamed object.
    ///
    /// The first call for a part places it at the next aligned offset.
    /// Writing past a part's declared length is an error rather than a
    /// rollover into the next one: a producer that has miscounted should
    /// hear about it where it happened.
    pub fn write_chunk(&mut self, object: &mut ObjectWriter, chunk: &[u8]) -> Result<()> {
        if !self.streaming {
            return Err(Error::InvalidInput(format!(
                "object {:?} is not open on this writer",
                object.name
            )));
        }
        let Some(part_name) = object.order.get(object.at).cloned() else {
            return Err(Error::InvalidInput(format!(
                "object {:?}: every part is already written",
                object.name
            )));
        };
        let declared = object.object.as_ref().expect("object present").parts[&part_name]
            .blob
            .length;

        if !object.started {
            let offset = self.reserve_blob()?;
            object
                .object
                .as_mut()
                .expect("object present")
                .parts
                .get_mut(&part_name)
                .expect("part present")
                .blob
                .offset = offset;
            object.started = true;
            object.hasher = xxhash_rust::xxh3::Xxh3::new();
            object.written = 0;
        }

        let end = object
            .written
            .checked_add(chunk.len() as u64)
            .filter(|&e| e <= declared)
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "object {:?} part {part_name:?}: {} bytes written into a part \
                     declared as {declared}",
                    object.name,
                    object.written + chunk.len() as u64
                ))
            })?;

        self.write_bytes(chunk)?;
        object.hasher.update(chunk);
        object.written = end;

        if object.written == declared {
            let digest = format!("xxh3:{:016x}", object.hasher.digest());
            object
                .object
                .as_mut()
                .expect("object present")
                .parts
                .get_mut(&part_name)
                .expect("part present")
                .digest = Some(digest);
            object.at += 1;
            object.started = false;
        }
        Ok(())
    }

    /// Completes a streamed object and adds it to the manifest.
    pub fn end_object(&mut self, mut object: ObjectWriter) -> Result<()> {
        if object.at < object.order.len() {
            let part = &object.order[object.at];
            let declared = object.object.as_ref().expect("object present").parts[part]
                .blob
                .length;
            return Err(Error::InvalidInput(format!(
                "object {:?} part {part:?}: {} of {declared} bytes written",
                object.name, object.written
            )));
        }
        let built = object.object.take().expect("object present");
        self.manifest.objects.insert(object.name.clone(), built);
        self.last_name = Some(object.name);
        self.streaming = false;
        Ok(())
    }
}

fn align_up(offset: u64, align: u64) -> Result<u64> {
    offset
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| Error::InvalidInput("file offset overflow".into()))
}

/// Writes a data shard (spec §7.2): magic, aligned blobs, and a footer
/// with no manifest. `finish` returns the shard's identity — exactly what
/// [`Writer::add_shard`] wants.
///
/// The whole-file digest is computed streamingly while writing, so
/// producing a shard costs one pass.
pub struct DataShardWriter {
    out: BufWriter<File>,
    hasher: xxhash_rust::xxh3::Xxh3,
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
            hasher: xxhash_rust::xxh3::Xxh3::new(),
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

    /// Writes the manifest-less footer and returns `(size, digest)` — the
    /// shard's identity for the root's shard table.
    pub fn finish(mut self) -> Result<(u64, String)> {
        let mut footer = [0u8; FOOTER_LEN as usize];
        footer[24..28].copy_from_slice(&VERSION.to_le_bytes());
        footer[32..40].copy_from_slice(&MAGIC);
        self.put(&footer)?;
        self.out.flush()?;
        // See Writer::finish: durability is the caller's call.
        Ok((self.offset, format!("xxh3:{:016x}", self.hasher.digest())))
    }
}
