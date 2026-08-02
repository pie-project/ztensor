//! `.zt` v2 reader.
//!
//! Opens a file with a read-only shared memory map, runs every validation
//! rule of spec §3.6, and serves structural reads:
//!
//! - [`Reader::view`] — tier 2, zero-copy slice into the map; errors when
//!   zero-copy is impossible (never silently degrades to a copy).
//! - [`Reader::read`] — tier 1, owned bytes.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result, Rule};
use crate::models::{
    check_name, registered_dtype, logical_size, Layout, Manifest, Object, Part, ALIGN_FLOOR,
    FOOTER_LEN, MAGIC, MAX_MANIFEST_LEN, MAX_RANK, VERSION,
};
use crate::cbor;

pub struct Reader {
    mmap: Mmap,
    manifest: Manifest,
    data_shard: bool,
}

impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("len", &self.mmap.len())
            .field("objects", &self.manifest.objects.len())
            .field("data_shard", &self.data_shard)
            .finish()
    }
}

impl Reader {
    /// Opens and fully validates a `.zt` file (spec §8, §3.6).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: read-only shared map; we treat the contents as untrusted
        // bytes and never assume stability beyond the validation snapshot.
        let mmap = unsafe { Mmap::map(&file)? };
        let (manifest, data_shard) = parse_and_validate(&mmap)?;
        Ok(Self {
            mmap,
            manifest,
            data_shard,
        })
    }

    /// True if this file carries no manifest (a data shard, spec §7.2).
    pub fn is_data_shard(&self) -> bool {
        self.data_shard
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Tier 0: iterate objects and their metadata.
    pub fn objects(&self) -> impl Iterator<Item = (&str, &Object)> {
        self.manifest
            .objects
            .iter()
            .map(|(name, obj)| (name.as_str(), obj))
    }

    pub fn get(&self, name: &str) -> Option<&Object> {
        self.manifest.objects.get(name)
    }

    fn part(&self, name: &str, part: &str) -> Result<&Part> {
        let obj = self
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("object {name:?}")))?;
        obj.parts
            .get(part)
            .ok_or_else(|| Error::NotFound(format!("part {name:?}/{part:?}")))
    }

    /// Tier 2: zero-copy view of a part's stored bytes.
    ///
    /// Errors (instead of degrading) when the part is encoded or lives in
    /// another shard. For raw local parts the returned slice is the decoded
    /// content.
    pub fn view(&self, name: &str, part: &str) -> Result<&[u8]> {
        let p = self.part(name, part)?;
        if p.blob.shard != 0 {
            return Err(Error::Unsupported(
                "reading from foreign shards lands in M5".into(),
            ));
        }
        if let Some(enc) = &p.encoding {
            return Err(Error::Unsupported(format!(
                "encoded part (encoding {enc:?}) has no zero-copy view"
            )));
        }
        let start = p.blob.offset as usize;
        let end = start + p.blob.length as usize; // bounds validated at open
        Ok(&self.mmap[start..end])
    }

    /// Tier 1: owned decoded bytes of a part.
    pub fn read(&self, name: &str, part: &str) -> Result<Vec<u8>> {
        self.view(name, part).map(<[u8]>::to_vec)
    }

    /// Verifies a part's stored digest against its decoded bytes.
    /// Returns `Ok(false)` when the part carries no digest.
    pub fn verify(&self, name: &str, part: &str) -> Result<bool> {
        let p = self.part(name, part)?;
        let Some(digest) = p.digest.clone() else {
            return Ok(false);
        };
        let bytes = self.view(name, part)?;
        let Some(hex) = digest.strip_prefix("xxh3:") else {
            return Err(Error::Unsupported(format!(
                "digest algorithm in {digest:?} (only xxh3 in M1)"
            )));
        };
        let expected = u64::from_str_radix(hex, 16)
            .map_err(|_| Error::reject(Rule::Digest, format!("malformed digest {digest:?}")))?;
        if xxh3_64(bytes) != expected {
            return Err(Error::reject(
                Rule::Digest,
                format!("digest mismatch for {name:?}/{part:?}"),
            ));
        }
        Ok(true)
    }
}

// =======================================================================
// Validation (spec §8 reading algorithm + §3.6 validation summary)
// =======================================================================

fn parse_and_validate(buf: &[u8]) -> Result<(Manifest, bool)> {
    let file_len = buf.len() as u64;
    if file_len < 48 {
        return Err(Error::reject(Rule::FileTooSmall, "file shorter than 48 B"));
    }
    if buf[..8] != MAGIC {
        return Err(Error::reject(Rule::HeaderMagic, "bad header magic"));
    }

    let footer = &buf[buf.len() - FOOTER_LEN as usize..];
    let manifest_offset = u64::from_le_bytes(footer[0..8].try_into().unwrap());
    let manifest_length = u64::from_le_bytes(footer[8..16].try_into().unwrap());
    let manifest_hash = u64::from_le_bytes(footer[16..24].try_into().unwrap());
    let version = u32::from_le_bytes(footer[24..28].try_into().unwrap());
    // footer[28..32] is reserved: written as zero, ignored on read.
    if footer[32..40] != MAGIC {
        return Err(Error::reject(Rule::FooterMagic, "bad footer magic"));
    }
    if version != VERSION {
        return Err(Error::reject(
            Rule::Version,
            format!("unsupported version {version}"),
        ));
    }

    if manifest_length == 0 {
        if manifest_offset != 0 || manifest_hash != 0 {
            return Err(Error::reject(
                Rule::ManifestBounds,
                "data shard footer must zero manifest offset and hash",
            ));
        }
        return Ok((Manifest::default(), true));
    }
    if manifest_length > MAX_MANIFEST_LEN {
        return Err(Error::reject(Rule::ManifestTooLarge, "manifest over 1 GiB"));
    }
    if manifest_offset % ALIGN_FLOOR != 0 || manifest_offset < ALIGN_FLOOR {
        return Err(Error::reject(
            Rule::BlobAlignment,
            "manifest blob is misaligned",
        ));
    }
    let data_end = file_len - FOOTER_LEN;
    let manifest_end = manifest_offset
        .checked_add(manifest_length)
        .filter(|&e| e <= data_end)
        .ok_or_else(|| Error::reject(Rule::ManifestBounds, "manifest outside data region"))?;

    let manifest_bytes = &buf[manifest_offset as usize..manifest_end as usize];
    if xxh3_64(manifest_bytes) != manifest_hash {
        return Err(Error::reject(Rule::ManifestHash, "manifest hash mismatch"));
    }

    let manifest = Manifest::from_value(cbor::decode(manifest_bytes)?)?;
    validate_manifest(&manifest, data_end, (manifest_offset, manifest_length))?;
    Ok((manifest, false))
}

fn validate_manifest(
    manifest: &Manifest,
    data_end: u64,
    manifest_blob: (u64, u64),
) -> Result<()> {
    // Local blob references, for the identical-or-disjoint check (§2.4).
    let mut local_refs: Vec<(u64, u64)> = vec![manifest_blob];

    for (name, obj) in &manifest.objects {
        check_name(name)?;
        if obj.shape.len() > MAX_RANK {
            return Err(Error::reject(
                Rule::Shape,
                format!("{name:?}: rank exceeds {MAX_RANK}"),
            ));
        }
        let elems = obj.num_elements()?;

        for (pname, part) in &obj.parts {
            check_name(pname)?;
            let b = part.blob;
            if b.shard != 0 && !manifest.shards.contains_key(&b.shard) {
                return Err(Error::reject(
                    Rule::ShardIndex,
                    format!("{name:?}/{pname:?}: shard {} not in table", b.shard),
                ));
            }
            if b.offset % ALIGN_FLOOR != 0 || b.offset < ALIGN_FLOOR {
                return Err(Error::reject(
                    Rule::BlobAlignment,
                    format!("{name:?}/{pname:?}: offset {} misaligned", b.offset),
                ));
            }
            let region_end = if b.shard == 0 {
                data_end
            } else {
                manifest.shards[&b.shard]
                    .size
                    .checked_sub(FOOTER_LEN)
                    .ok_or_else(|| Error::reject(Rule::BlobBounds, "shard smaller than footer"))?
            };
            b.offset
                .checked_add(b.length)
                .filter(|&e| e <= region_end)
                .ok_or_else(|| {
                    Error::reject(
                        Rule::BlobBounds,
                        format!("{name:?}/{pname:?}: blob outside data region"),
                    )
                })?;
            if b.shard == 0 && b.length > 0 {
                local_refs.push((b.offset, b.length));
            }
        }

        match &obj.layout {
            Layout::Dense => {
                if obj.parts.len() != 1 || !obj.parts.contains_key("data") {
                    return Err(Error::reject(
                        Rule::LayoutRule,
                        format!("{name:?}: dense requires exactly one part named 'data'"),
                    ));
                }
                let part = &obj.parts["data"];
                if let Some(lt) = &part.ltype {
                    if let Some(required) = registered_dtype(lt) {
                        if part.dtype != required {
                            return Err(Error::reject(
                                Rule::Schema,
                                format!("{name:?}: type {lt:?} requires dtype {required:?}"),
                            ));
                        }
                    }
                }
                // Size equation is checkable only when the size function is
                // known; unknown logical types stay structural (§4.2).
                if let Some(expected) = logical_size(part.ltype.as_deref(), part.dtype, elems) {
                    if part.decoded_size() != expected {
                        return Err(Error::reject(
                            Rule::DenseSize,
                            format!(
                                "{name:?}: decoded size {} != expected {expected}",
                                part.decoded_size()
                            ),
                        ));
                    }
                }
            }
            Layout::Other(_) => {} // structural access only; profiles land in M4
        }
    }

    // Identical-or-disjoint (§2.4): sort, drop exact duplicates, then any
    // remaining pair that overlaps is a partial overlap.
    local_refs.sort_unstable();
    local_refs.dedup();
    for w in local_refs.windows(2) {
        let (a_off, a_len) = w[0];
        let (b_off, _) = w[1];
        if a_off + a_len > b_off {
            return Err(Error::reject(
                Rule::BlobOverlap,
                format!("blobs at {a_off} (+{a_len}) and {b_off} partially overlap"),
            ));
        }
    }
    Ok(())
}
