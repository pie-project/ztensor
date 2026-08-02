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
use crate::source::{page_size, Caps, Source};

pub struct Reader {
    mmap: Mmap,
    manifest: Manifest,
    data_shard: bool,
    /// Every occupied byte range in this file — header magic, blobs,
    /// manifest blob, footer — sorted and deduplicated. Basis for the
    /// page-exclusivity capability.
    ranges: Vec<(u64, u64)>,
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
        let (manifest, data_shard, mut ranges) = parse_and_validate(&mmap)?;
        ranges.push((0, MAGIC.len() as u64));
        ranges.push((mmap.len() as u64 - FOOTER_LEN, FOOTER_LEN));
        ranges.sort_unstable();
        ranges.dedup();
        Ok(Self {
            mmap,
            manifest,
            data_shard,
            ranges,
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

    /// Capability report for one part (spec: capability ladder).
    pub fn caps(&self, name: &str, part: &str) -> Result<Caps> {
        let p = self.part(name, part)?;
        let raw_local = p.blob.shard == 0 && p.encoding.is_none();
        let page_exclusive = raw_local
            && is_page_exclusive(&self.ranges, p.blob.offset, p.blob.length, page_size());
        Ok(Caps {
            zero_copy: raw_local,
            alignment: 1u64 << p.blob.offset.trailing_zeros().min(63),
            verifiable: p.digest.is_some(),
            page_exclusive,
        })
    }

    /// Drops the OS page cache for a part's exact page range (weight
    /// streaming eviction). Requires page exclusivity — this call never
    /// touches a page that another blob occupies.
    #[cfg(unix)]
    pub fn evict(&self, name: &str, part: &str) -> Result<()> {
        let p = self.part(name, part)?;
        if p.blob.length == 0 {
            return Ok(());
        }
        let caps = self.caps(name, part)?;
        if !caps.zero_copy {
            return Err(Error::Unsupported(
                "evict applies to raw local parts only".into(),
            ));
        }
        if !caps.page_exclusive {
            return Err(Error::Unsupported(format!(
                "{name:?}/{part:?} shares an OS page with another blob; \
                 eviction would drop a neighbor's cache"
            )));
        }
        let page = page_size();
        let start = p.blob.offset & !(page - 1);
        let end = (p.blob.offset + p.blob.length)
            .div_ceil(page)
            .saturating_mul(page)
            .min(self.mmap.len() as u64);
        // SAFETY: the map is a read-only shared file mapping — DontNeed
        // only drops clean page-cache pages; later accesses re-fault from
        // the file. It cannot discard writes because none exist.
        unsafe {
            self.mmap.unchecked_advise_range(
                memmap2::UncheckedAdvice::DontNeed,
                start as usize,
                (end - start) as usize,
            )?;
        }
        Ok(())
    }

    /// Hints the OS to prefetch a part's pages.
    #[cfg(unix)]
    pub fn prefetch(&self, name: &str, part: &str) -> Result<()> {
        let p = self.part(name, part)?;
        if p.blob.shard != 0 || p.blob.length == 0 {
            return Ok(());
        }
        self.mmap.advise_range(
            memmap2::Advice::WillNeed,
            p.blob.offset as usize,
            p.blob.length as usize,
        )?;
        Ok(())
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

impl Source for Reader {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read(&self, object: &str, part: &str) -> Result<Vec<u8>> {
        Reader::read(self, object, part)
    }

    fn view(&self, object: &str, part: &str) -> Result<&[u8]> {
        Reader::view(self, object, part)
    }

    fn caps(&self, object: &str, part: &str) -> Result<Caps> {
        Reader::caps(self, object, part)
    }
}

/// True iff the page-aligned envelope of `[offset, offset + length)`
/// intersects no *other* occupied range. `ranges` must be sorted, deduped,
/// and pairwise disjoint (guaranteed by validation); zero-length parts are
/// vacuously exclusive.
fn is_page_exclusive(ranges: &[(u64, u64)], offset: u64, length: u64, page: u64) -> bool {
    if length == 0 {
        return true;
    }
    let env_start = offset & !(page - 1);
    let env_end = (offset + length).div_ceil(page).saturating_mul(page);
    let Ok(i) = ranges.binary_search(&(offset, length)) else {
        return false; // not an occupied range we know about
    };
    let prev_clear = i == 0 || {
        let (o, l) = ranges[i - 1];
        o + l <= env_start
    };
    let next_clear = i + 1 >= ranges.len() || ranges[i + 1].0 >= env_end;
    prev_clear && next_clear
}

// =======================================================================
// Validation (spec §8 reading algorithm + §3.6 validation summary)
// =======================================================================

/// (manifest, is_data_shard, sorted local blob ranges incl. the manifest).
type ParsedFile = (Manifest, bool, Vec<(u64, u64)>);

fn parse_and_validate(buf: &[u8]) -> Result<ParsedFile> {
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
        return Ok((Manifest::default(), true, Vec::new()));
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
    let ranges = validate_manifest(&manifest, data_end, (manifest_offset, manifest_length))?;
    Ok((manifest, false, ranges))
}

/// Validates a complete in-memory `.zt` file image and returns its manifest
/// (`None` for a data shard).
///
/// This is [`Reader::open`] minus the file handle: the same §8 reading
/// algorithm and §3.6 validation, driven directly by the conformance corpus
/// and the fuzz targets.
pub fn validate_bytes(buf: &[u8]) -> Result<Option<Manifest>> {
    parse_and_validate(buf).map(|(m, data_shard, _)| if data_shard { None } else { Some(m) })
}

/// Returns the sorted, deduplicated local blob ranges (manifest included).
fn validate_manifest(
    manifest: &Manifest,
    data_end: u64,
    manifest_blob: (u64, u64),
) -> Result<Vec<(u64, u64)>> {
    for (&idx, shard) in &manifest.shards {
        if shard.size < 48 {
            return Err(Error::reject(
                Rule::Schema,
                format!("shard {idx}: size {} below minimum file size", shard.size),
            ));
        }
    }

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
            // Registered logical types pin their storage type (§4.2),
            // regardless of layout.
            if let Some(lt) = &part.ltype {
                if let Some(required) = registered_dtype(lt) {
                    if part.dtype != required {
                        return Err(Error::reject(
                            Rule::Schema,
                            format!("{name:?}/{pname:?}: type {lt:?} requires dtype {required:?}"),
                        ));
                    }
                }
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
    Ok(local_refs)
}

#[cfg(test)]
mod tests {
    use super::is_page_exclusive;

    #[test]
    fn page_exclusivity() {
        // header, two blobs, manifest, footer — a typical 4 KiB-aligned file
        let ranges = [(0u64, 8u64), (4096, 8), (8192, 100), (12288, 340), (12628, 40)];
        // 4 KiB pages: each aligned blob owns its pages
        assert!(is_page_exclusive(&ranges, 4096, 8, 4096));
        assert!(is_page_exclusive(&ranges, 8192, 100, 4096));
        // 64 KiB pages: everything shares page 0
        assert!(!is_page_exclusive(&ranges, 4096, 8, 65536));
        assert!(!is_page_exclusive(&ranges, 8192, 100, 65536));
        // 64 KiB-aligned blobs are exclusive even on 64 KiB pages
        let canonical = [(0u64, 8u64), (65536, 8), (131072, 100), (196608, 380)];
        assert!(is_page_exclusive(&canonical, 65536, 8, 65536));
        assert!(is_page_exclusive(&canonical, 131072, 100, 65536));
        // two ranges inside one page: neither is exclusive
        let packed = [(4096u64, 100u64), (4200, 50)];
        assert!(!is_page_exclusive(&packed, 4096, 100, 4096));
        assert!(!is_page_exclusive(&packed, 4200, 50, 4096));
        // zero-length: vacuously exclusive
        assert!(is_page_exclusive(&ranges, 4096, 0, 65536));
        // unknown range: not exclusive
        assert!(!is_page_exclusive(&ranges, 20480, 8, 4096));
    }
}
