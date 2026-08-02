//! `.zt` v2 writer.
//!
//! Append-only: magic, then blobs at aligned offsets, then the manifest
//! blob, then the footer. The default mode produces **canonical form**
//! (spec §6.3): 64 KiB placement, sorted insertion, per-part xxh3 digests,
//! and blob sharing for byte-identical parts.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use xxhash_rust::xxh3::{xxh3_128, xxh3_64};

use crate::cbor;
use crate::error::{Error, Result};
use crate::models::{
    check_digest, check_name, BlobRef, DType, Layout, Manifest, Object, Part, Shard,
    ALIGN_CANONICAL, ALIGN_FLOOR, FOOTER_LEN, MAGIC, MAX_MANIFEST_LEN, MAX_RANK, VERSION,
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

pub struct Writer {
    out: BufWriter<File>,
    offset: u64,
    align: u64,
    canonical: bool,
    manifest: Manifest,
    /// (xxh3_128, length) -> offset. Content-keyed blob sharing; a 128-bit
    /// key makes an accidental collision negligible without a verify read.
    dedup: HashMap<(u128, u64), u64>,
    last_name: Option<String>,
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
        if !align.is_power_of_two() || align < ALIGN_FLOOR {
            return Err(Error::InvalidInput(format!(
                "alignment must be a power of two >= {ALIGN_FLOOR}, got {align}"
            )));
        }
        Self::new(path, align, false)
    }

    fn new(path: impl AsRef<Path>, align: u64, canonical: bool) -> Result<Self> {
        let file = File::create(path)?;
        let mut out = BufWriter::with_capacity(1 << 20, file);
        out.write_all(&MAGIC)?;
        Ok(Self {
            out,
            offset: MAGIC.len() as u64,
            align,
            canonical,
            manifest: Manifest::default(),
            dedup: HashMap::new(),
            last_name: None,
        })
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
        check_name(name).map_err(|_| Error::InvalidInput(format!("invalid name {name:?}")))?;
        if self.manifest.objects.contains_key(name) {
            return Err(Error::InvalidInput(format!("duplicate object {name:?}")));
        }
        if shape.len() > MAX_RANK {
            return Err(Error::InvalidInput(format!(
                "rank {} exceeds {MAX_RANK}",
                shape.len()
            )));
        }
        if self.canonical {
            if let Some(last) = &self.last_name {
                if name <= last.as_str() {
                    return Err(Error::InvalidInput(format!(
                        "canonical form requires sorted insertion: {name:?} after {last:?}"
                    )));
                }
            }
        }

        // Canonical blob order is (object, part) name order: process parts
        // sorted by name.
        let mut order: Vec<usize> = (0..parts.len()).collect();
        order.sort_by_key(|&i| parts[i].0);
        for w in order.windows(2) {
            if parts[w[0]].0 == parts[w[1]].0 {
                return Err(Error::InvalidInput(format!(
                    "duplicate part {:?}",
                    parts[w[0]].0
                )));
            }
        }

        // Encode payloads and build part metadata (dummy offsets), so known
        // layouts can be validated before any byte is written.
        let mut built: Vec<(String, Part, Vec<u8>, bool)> = Vec::with_capacity(parts.len());
        for &i in &order {
            let (pname, def) = &parts[i];
            check_name(pname)
                .map_err(|_| Error::InvalidInput(format!("invalid part name {pname:?}")))?;
            let (stored, encoding, decoded_length) = match def.encoding {
                None => (def.data.to_vec(), None, None),
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
                        profile.encode(def.data)?,
                        Some(enc.to_string()),
                        Some(def.data.len() as u64),
                    )
                }
            };
            let raw = encoding.is_none();
            let part = Part {
                dtype: def.dtype,
                ltype: def.ltype.map(str::to_string),
                blob: BlobRef {
                    shard: 0,
                    offset: 0, // patched after writing
                    length: stored.len() as u64,
                },
                encoding,
                decoded_length,
                digest: Some(format!("xxh3:{:016x}", xxh3_64(def.data))),
            };
            built.push((pname.to_string(), part, stored, raw));
        }

        let mut obj = Object {
            shape: shape.to_vec(),
            layout: Layout::from_name(layout),
            attributes,
            parts: built
                .iter()
                .map(|(n, p, _, _)| (n.clone(), p.clone()))
                .collect(),
        };
        if let Some(profile) = layout_profile(layout) {
            profile
                .validate(name, &obj)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
        }

        for (pname, _, stored, raw) in built {
            let offset = if raw {
                self.write_or_share_blob(&stored)?
            } else {
                self.write_blob(&stored)?
            };
            obj.parts.get_mut(&pname).unwrap().blob.offset = offset;
        }
        self.manifest.objects.insert(name.to_string(), obj);
        self.last_name = Some(name.to_string());
        Ok(())
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
        check_digest(digest)
            .map_err(|_| Error::InvalidInput(format!("malformed digest {digest:?}")))?;
        if size < 48 {
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
        check_name(name).map_err(|_| Error::InvalidInput(format!("invalid name {name:?}")))?;
        if self.manifest.objects.contains_key(name) {
            return Err(Error::InvalidInput(format!("duplicate object {name:?}")));
        }
        if shape.len() > MAX_RANK {
            return Err(Error::InvalidInput(format!(
                "rank {} exceeds {MAX_RANK}",
                shape.len()
            )));
        }
        let mut built = std::collections::BTreeMap::new();
        for (pname, part) in parts {
            check_name(pname)
                .map_err(|_| Error::InvalidInput(format!("invalid part name {pname:?}")))?;
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
                check_digest(d).map_err(|_| {
                    Error::InvalidInput(format!("part {pname:?}: malformed digest"))
                })?;
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
            profile
                .validate(name, &obj)
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
        }
        self.manifest.objects.insert(name.to_string(), obj);
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
        let manifest = src.manifest().clone();
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

    fn write_or_share_blob(&mut self, data: &[u8]) -> Result<u64> {
        let key = (xxh3_128(data), data.len() as u64);
        if let Some(&offset) = self.dedup.get(&key) {
            return Ok(offset);
        }
        let target = self.write_blob(data)?;
        self.dedup.insert(key, target);
        Ok(target)
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
        self.out.get_ref().sync_all()?;
        Ok(self.offset)
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
        if !align.is_power_of_two() || align < ALIGN_FLOOR {
            return Err(Error::InvalidInput(format!(
                "alignment must be a power of two >= {ALIGN_FLOOR}, got {align}"
            )));
        }
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
        self.out.get_ref().sync_all()?;
        Ok((self.offset, format!("xxh3:{:016x}", self.hasher.digest())))
    }
}
