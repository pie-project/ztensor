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
    check_logical_values, check_name, parse_xxh3, registered_dtype, DType, Manifest, Object,
    Part, ALIGN_FLOOR, FOOTER_LEN, MAGIC, MAX_MANIFEST_LEN, MAX_RANK, MIN_FILE_LEN, VERSION,
};
use crate::cbor;
use crate::profiles::{encoding_profile, layout_profile};
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
        self.manifest.part(name, part)
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

    /// Tier 1: owned decoded bytes of a part. Decodes known encoding
    /// profiles; refuses unknown ones (never returns stored bytes as if
    /// they were decoded).
    pub fn read(&self, name: &str, part: &str) -> Result<Vec<u8>> {
        let p = self.part(name, part)?;
        if p.blob.shard != 0 {
            return Err(Error::Unsupported(
                "this part lives in another shard; open the model with Model::open".into(),
            ));
        }
        decode_part(p, self.stored_slice(p))
    }

    /// The stored (possibly encoded) bytes of a local part.
    ///
    /// The `usize` casts are sound on every platform: validation bounded
    /// `offset + length` by the mapped length, which is itself a `usize`,
    /// so neither value can exceed `usize::MAX` by the time we get here.
    pub(crate) fn stored_slice(&self, part: &Part) -> &[u8] {
        let start = part.blob.offset as usize;
        &self.mmap[start..start + part.blob.length as usize]
    }

    /// Capability report for one part (spec: capability ladder).
    pub fn caps(&self, name: &str, part: &str) -> Result<Caps> {
        let p = self.part(name, part)?;
        let raw_local = p.blob.shard == 0 && p.encoding.is_none();
        let page_exclusive = raw_local
            && is_page_exclusive(&self.ranges, p.blob.offset, p.blob.length, page_size());
        Ok(Caps::for_part(p, raw_local, page_exclusive))
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
        if p.blob.shard != 0 || p.encoding.is_some() {
            return Err(Error::Unsupported(
                "evict applies to raw local parts only".into(),
            ));
        }
        if !is_page_exclusive(&self.ranges, p.blob.offset, p.blob.length, page_size()) {
            return Err(Error::Unsupported(format!(
                "{name:?}/{part:?} shares an OS page with another blob; \
                 eviction would drop a neighbor's cache"
            )));
        }
        let (start, end) = page_envelope(p.blob.offset, p.blob.length, page_size());
        let end = end.min(self.mmap.len() as u64);
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

    /// Verifies a part's digest and logical-type content rules.
    /// See the free [`verify`] function.
    pub fn verify(&self, name: &str, part: &str) -> Result<bool> {
        verify(self, name, part)
    }
}

/// Verifies one part of any [`Source`]: its digest (if present) over the
/// decoded bytes, plus the content rules of registered logical types
/// (spec Appendix A — e.g. `bool` bytes must be 0 or 1).
///
/// Returns `Ok(true)` when a digest was checked, `Ok(false)` when the part
/// carries none; content-rule violations are errors either way.
pub fn verify(src: &dyn Source, name: &str, part: &str) -> Result<bool> {
    let manifest = src.manifest();
    let obj = manifest.object(name)?;
    let p = manifest.part(name, part)?;
    let digest = p.digest.clone();
    let ltype = p.ltype.clone();
    let elems = if obj.layout.as_str() == "dense" {
        Some(obj.num_elements()?)
    } else {
        None
    };

    if digest.is_none() && ltype.is_none() {
        return Ok(false);
    }
    // Digests and content rules cover decoded bytes (§3.4): zero-copy when
    // the source can serve a view, decoded read otherwise.
    let owned;
    let bytes: &[u8] = if src.caps(name, part)?.zero_copy {
        src.view(name, part)?
    } else {
        owned = src.read(name, part)?;
        &owned
    };
    if let Some(lt) = &ltype {
        check_logical_values(lt, bytes, elems)?;
    }
    match digest {
        None => Ok(false),
        Some(d) => {
            if xxh3_64(bytes) != parse_xxh3(&d)? {
                return Err(Error::reject(
                    Rule::Digest,
                    format!("digest mismatch for {name:?}/{part:?}"),
                ));
            }
            Ok(true)
        }
    }
}

/// Page-aligned envelope of a byte range.
pub(crate) fn page_envelope(offset: u64, length: u64, page: u64) -> (u64, u64) {
    (
        offset & !(page - 1),
        (offset + length).div_ceil(page).saturating_mul(page),
    )
}

/// An assembled `zt.sparse_csr/1` object with its data-level rules checked.
#[derive(Debug, Clone)]
pub struct Csr {
    pub rows: u64,
    pub cols: u64,
    /// Decoded value bytes, `nnz` elements of `dtype`/`ltype`.
    pub values: Vec<u8>,
    pub dtype: DType,
    pub ltype: Option<String>,
    /// Column index per value, widened to u64.
    pub indices: Vec<u64>,
    /// Row pointers, `rows + 1` entries.
    pub indptr: Vec<u64>,
}

impl Reader {
    /// Reads and assembles a `zt.sparse_csr/1` object. See [`read_csr`].
    pub fn read_csr(&self, name: &str) -> Result<Csr> {
        read_csr(self, name)
    }
}

/// Reads and assembles a `zt.sparse_csr/1` object from any [`Source`],
/// enforcing the profile's data-level MUSTs: `indptr[0] == 0`,
/// non-decreasing, `indptr[rows] == nnz`, per-row strictly increasing
/// indices, and every index `< cols`.
pub fn read_csr(src: &dyn Source, name: &str) -> Result<Csr> {
    let obj = src.manifest().object(name)?;
    if obj.layout.as_str() != "zt.sparse_csr/1" {
        return Err(Error::Unsupported(format!(
            "{name:?} has layout {:?}, not zt.sparse_csr/1",
            obj.layout.as_str()
        )));
    }
    // Re-run the metadata rules so this holds for any Source, including
    // projections that never went through .zt open-time validation.
    crate::profiles::layout_profile("zt.sparse_csr/1")
        .expect("built-in profile")
        .validate(name, obj)?;
    let (rows, cols) = (obj.shape[0], obj.shape[1]);
    let idx_dtype = obj.parts["indices"].dtype;
    let (vdtype, vltype) = {
        let v = &obj.parts["values"];
        (v.dtype, v.ltype.clone())
    };

    let indices = widen_indices(&src.read(name, "indices")?, idx_dtype);
    let indptr = widen_indices(&src.read(name, "indptr")?, idx_dtype);
    let values = src.read(name, "values")?;
    let nnz = indices.len() as u64;

    let bad = |detail: String| Err(Error::reject(Rule::LayoutData, detail));
    if indptr.first() != Some(&0) {
        return bad(format!("{name:?}: indptr must start at 0"));
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) {
        return bad(format!("{name:?}: indptr must be non-decreasing"));
    }
    if indptr.last() != Some(&nnz) {
        return bad(format!("{name:?}: indptr must end at nnz ({nnz})"));
    }
    for r in 0..rows as usize {
        let row = &indices[indptr[r] as usize..indptr[r + 1] as usize];
        if row.windows(2).any(|w| w[0] >= w[1]) {
            return bad(format!("{name:?}: row {r} indices not strictly increasing"));
        }
        if row.last().is_some_and(|&c| c >= cols) {
            return bad(format!("{name:?}: row {r} has an index >= cols ({cols})"));
        }
    }

    Ok(Csr {
        rows,
        cols,
        values,
        dtype: vdtype,
        ltype: vltype,
        indices,
        indptr,
    })
}

/// Decodes a part's stored bytes per its encoding profile (identity for
/// raw). Refuses unknown profiles.
pub(crate) fn decode_part(part: &Part, stored: &[u8]) -> Result<Vec<u8>> {
    match &part.encoding {
        None => Ok(stored.to_vec()),
        Some(enc) => {
            let profile = encoding_profile(enc)
                .ok_or_else(|| Error::Unsupported(format!("unknown encoding profile {enc:?}")))?;
            profile.decode(stored, part.decoded_size())
        }
    }
}

fn widen_indices(bytes: &[u8], dtype: DType) -> Vec<u64> {
    match dtype {
        DType::U32 => bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as u64)
            .collect(),
        _ => bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
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
    let (env_start, env_end) = page_envelope(offset, length, page);
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

/// Checks the frame every container shares — minimum size, header magic,
/// footer magic, supported version — and returns the footer slice.
/// (Reserved footer bytes are written as zero and ignored on read.)
pub(crate) fn check_container(buf: &[u8]) -> Result<&[u8]> {
    if (buf.len() as u64) < MIN_FILE_LEN {
        return Err(Error::reject(
            Rule::FileTooSmall,
            format!("file shorter than {MIN_FILE_LEN} B"),
        ));
    }
    if buf[..8] != MAGIC {
        return Err(Error::reject(Rule::HeaderMagic, "bad header magic"));
    }
    let footer = &buf[buf.len() - FOOTER_LEN as usize..];
    if footer[32..40] != MAGIC {
        return Err(Error::reject(Rule::FooterMagic, "bad footer magic"));
    }
    let version = u32::from_le_bytes(footer[24..28].try_into().unwrap());
    if version != VERSION {
        return Err(Error::reject(
            Rule::Version,
            format!("unsupported version {version}"),
        ));
    }
    Ok(footer)
}

fn parse_and_validate(buf: &[u8]) -> Result<ParsedFile> {
    let file_len = buf.len() as u64;
    let footer = check_container(buf)?;
    let manifest_offset = u64::from_le_bytes(footer[0..8].try_into().unwrap());
    let manifest_length = u64::from_le_bytes(footer[8..16].try_into().unwrap());
    let manifest_hash = u64::from_le_bytes(footer[16..24].try_into().unwrap());

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
        if shard.size < crate::models::MIN_FILE_LEN {
            return Err(Error::reject(
                Rule::Schema,
                format!("shard {idx}: size {} below minimum file size", shard.size),
            ));
        }
    }

    // Blob references grouped by the file they live in, for the per-file
    // identical-or-disjoint check (§2.4 / §3.6 rule 4). Shard 0 also holds
    // the manifest blob.
    let mut refs_by_shard: std::collections::BTreeMap<u64, Vec<(u64, u64)>> =
        std::collections::BTreeMap::new();
    refs_by_shard.insert(0, vec![manifest_blob]);

    for (name, obj) in &manifest.objects {
        check_name(name)?;
        if obj.shape.len() > MAX_RANK {
            return Err(Error::reject(
                Rule::Shape,
                format!("{name:?}: rank exceeds {MAX_RANK}"),
            ));
        }
        obj.num_elements()?; // shape-product overflow check (Rule::Shape)

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
            if b.length > 0 {
                refs_by_shard
                    .entry(b.shard)
                    .or_default()
                    .push((b.offset, b.length));
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

        // Known layout profiles validate their metadata rules; unknown
        // layouts stay structural (§5.2).
        if let Some(profile) = layout_profile(obj.layout.as_str()) {
            profile.validate(name, obj)?;
        }
    }

    // Identical-or-disjoint per referenced file (§2.4): sort, drop exact
    // duplicates, then any remaining pair that overlaps is a partial
    // overlap.
    for (shard, refs) in &mut refs_by_shard {
        refs.sort_unstable();
        refs.dedup();
        for w in refs.windows(2) {
            let (a_off, a_len) = w[0];
            let (b_off, _) = w[1];
            if a_off + a_len > b_off {
                return Err(Error::reject(
                    Rule::BlobOverlap,
                    format!(
                        "blobs at {a_off} (+{a_len}) and {b_off} in shard {shard} \
                         partially overlap"
                    ),
                ));
            }
        }
    }
    Ok(refs_by_shard.remove(&0).unwrap_or_default())
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
