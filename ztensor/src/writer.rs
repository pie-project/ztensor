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
    check_name, BlobRef, DType, Layout, Manifest, Object, Part, ALIGN_CANONICAL, ALIGN_FLOOR,
    FOOTER_LEN, MAGIC, MAX_MANIFEST_LEN, MAX_RANK, VERSION,
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
