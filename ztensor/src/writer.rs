//! `.zt` v2 writer.
//!
//! Append-only: magic, then blobs at aligned offsets, then the manifest
//! blob, then the footer. The default mode produces **canonical form**
//! (spec §6.3): 64 KiB placement, sorted insertion, per-part xxh3 digests,
//! and blob sharing for byte-identical parts.

use std::collections::BTreeMap;
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
        let elems = shape.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d));
        let expected = elems.and_then(|n| n.checked_mul(dtype.width()));
        match expected {
            Some(n) if n == data.len() as u64 => {}
            _ => {
                return Err(Error::InvalidInput(format!(
                    "data length {} does not match shape {:?} x {}",
                    data.len(),
                    shape,
                    dtype.as_str()
                )))
            }
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

        let offset = self.write_or_share_blob(data)?;
        let part = Part {
            dtype,
            ltype: None,
            blob: BlobRef {
                shard: 0,
                offset,
                length: data.len() as u64,
            },
            encoding: None,
            decoded_length: None,
            digest: Some(format!("xxh3:{:016x}", xxh3_64(data))),
        };
        let mut parts = BTreeMap::new();
        parts.insert("data".to_string(), part);
        self.manifest.objects.insert(
            name.to_string(),
            Object {
                shape: shape.to_vec(),
                layout: Layout::Dense,
                attributes: None,
                parts,
            },
        );
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
        let target = align_up(self.offset, self.align)?;
        self.pad_to(target)?;
        self.out.write_all(data)?;
        self.offset = target + data.len() as u64;
        self.dedup.insert(key, target);
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
