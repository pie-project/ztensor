//! The `.zt` reading algorithm (spec §8) and validation summary (spec §3.6).
//!
//! Two entry points onto the same rules: [`image`] over a complete in-memory
//! file, which is what the conformance corpus and the fuzz targets drive, and
//! an internal one over an opened container, which reads only the footer and
//! the manifest blob, so opening a 100 GB checkpoint to plan against it touches
//! two ranges, not a hundred gigabytes.
//!
//! Canonical form (§6.3) is decided here too, as a pure function of a manifest
//! and of where the file put things. The reader supplies both; nothing here
//! opens a path. [`read::canonical_violations`](crate::read::canonical_violations)
//! is the entry point that does.

use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result, Rule};
use crate::format::cbor;
use crate::format::{
    check_name, check_shard_name, Manifest, ALIGN_CANONICAL, ALIGN_FLOOR, FOOTER_LEN, MAGIC,
    MAX_MANIFEST_LEN, MAX_RANK, MIN_FILE_LEN, VERSION,
};
use crate::provide::store::Store;
use crate::vocab::Vocabulary;

/// What the footer points at. `None` where a footer describes a data shard
/// (spec §7.2): no manifest, and the other fields must be zero.
pub(crate) struct Footer {
    offset: u64,
    length: u64,
    hash: u64,
}

/// A validated `.zt` file: its manifest (absent for a data shard) and every
/// byte range it occupies.
pub(crate) struct Parsed {
    pub manifest: Option<Manifest>,
    pub occupied: Vec<(u64, u64)>,
    /// Where the manifest blob starts.
    ///
    /// `Writer::append` reads it to work out the alignment the file was
    /// written at, and for nothing else — it writes from EOF, because §2.5
    /// forbids moving or overwriting a byte that is already there. The old
    /// manifest stays where it is as an unreferenced blob.
    pub manifest_at: Option<u64>,
}

/// Checks the frame every container shares (minimum size, header magic,
/// footer magic, supported version) and returns the footer slice.
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
    check_footer(footer)?;
    Ok(footer)
}

fn check_footer(footer: &[u8]) -> Result<()> {
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
    Ok(())
}

fn parse_footer(footer: &[u8]) -> Result<Option<Footer>> {
    let offset = u64::from_le_bytes(footer[0..8].try_into().unwrap());
    let length = u64::from_le_bytes(footer[8..16].try_into().unwrap());
    let hash = u64::from_le_bytes(footer[16..24].try_into().unwrap());
    if length == 0 {
        if offset != 0 || hash != 0 {
            return Err(Error::reject(
                Rule::ManifestBounds,
                "data shard footer must zero manifest offset and hash",
            ));
        }
        return Ok(None);
    }
    if length > MAX_MANIFEST_LEN {
        return Err(Error::reject(Rule::ManifestTooLarge, "manifest over 1 GiB"));
    }
    if offset % ALIGN_FLOOR != 0 || offset < ALIGN_FLOOR {
        return Err(Error::reject(
            Rule::BlobAlignment,
            "manifest blob is misaligned",
        ));
    }
    Ok(Some(Footer {
        offset,
        length,
        hash,
    }))
}

/// The manifest blob's bounds within a file of `file_len` bytes.
fn manifest_bounds(footer: &Footer, file_len: u64) -> Result<(u64, u64)> {
    let data_end = file_len - FOOTER_LEN;
    let end = footer
        .offset
        .checked_add(footer.length)
        .filter(|&e| e <= data_end)
        .ok_or_else(|| Error::reject(Rule::ManifestBounds, "manifest outside data region"))?;
    Ok((end, data_end))
}

fn decode_manifest(bytes: &[u8], expected_hash: u64) -> Result<Manifest> {
    if xxh3_64(bytes) != expected_hash {
        return Err(Error::reject(Rule::ManifestHash, "manifest hash mismatch"));
    }
    Manifest::from_value(cbor::decode(bytes)?)
}

/// Frame + magic + footer + occupied ranges, including the header and footer
/// themselves so page exclusivity accounts for them.
fn frame_ranges(mut ranges: Vec<(u64, u64)>, file_len: u64) -> Vec<(u64, u64)> {
    ranges.push((0, MAGIC.len() as u64));
    ranges.push((file_len - FOOTER_LEN, FOOTER_LEN));
    ranges.sort_unstable();
    ranges.dedup();
    ranges
}

/// Validates a complete in-memory `.zt` file image and returns its manifest
/// (`None` for a data shard).
///
/// This is the same §8 reading algorithm and §3.6 validation a reader runs,
/// minus the file handle.
pub fn image(buf: &[u8], vocab: &Vocabulary) -> Result<Option<Manifest>> {
    let footer = check_container(buf)?;
    let Some(footer) = parse_footer(footer)? else {
        return Ok(None);
    };
    let file_len = buf.len() as u64;
    let (manifest_end, data_end) = manifest_bounds(&footer, file_len)?;
    let manifest = decode_manifest(
        &buf[footer.offset as usize..manifest_end as usize],
        footer.hash,
    )?;
    validate_manifest(&manifest, data_end, (footer.offset, footer.length), vocab)?;
    Ok(Some(manifest))
}

/// Where a file put the things canonical form has an opinion about.
///
/// Placement is the half of canonical form that is not in the manifest, so it
/// arrives separately and the rule walk stays a pure function of the two.
pub(crate) struct Placement {
    pub manifest_at: u64,
    pub manifest_len: u64,
    pub file_len: u64,
}

/// Checks a manifest and its placement against canonical form (spec §6.3) and
/// returns every rule they break, in rule order. An empty list means the file
/// is canonical.
///
/// The spec calls canonical form the recommended distribution format, which is
/// only worth saying if the receiver can tell. Nothing is stored in the file to
/// say it is canonical, and nothing needs to be: all six rules are decidable
/// from the bytes.
///
/// Blob sharing (rule 3) is judged by digest and length rather than by
/// comparing payloads, so two parts with the same digest are taken to have the
/// same content. Rule 4 guarantees every part carries one.
pub(crate) fn canonical_violations(manifest: &Manifest, at: &Placement) -> Vec<String> {
    let Placement {
        manifest_at,
        manifest_len,
        file_len,
    } = *at;

    let mut bad = Vec::new();

    // Rule 6 first: a shard table means the parts below may point elsewhere,
    // and the placement walk would be describing the wrong file.
    if !manifest.shards.is_empty() {
        bad.push(format!(
            "rule 6: canonical form is single-file, but the manifest declares {} shard(s)",
            manifest.shards.len()
        ));
    }

    // Rule 5. ASCII is NFC by construction, so only other names pay for this.
    {
        let mut not_nfc = |what: &str, name: &str| {
            if !name.is_ascii() && !unicode_normalization::is_nfc(name) {
                bad.push(format!("rule 5: {what} {name:?} is not in NFC"));
            }
        };
        for (name, object) in &manifest.objects {
            not_nfc("object name", name);
            for pname in object.parts.keys() {
                not_nfc("part name", pname);
            }
        }
    }

    // Rules 1-4, in one walk over the parts in the order rule 3 fixes.
    let mut cursor = MAGIC.len() as u64;
    let mut placed: std::collections::BTreeMap<u64, (String, u64)> = Default::default();
    let mut by_content: std::collections::BTreeMap<(String, u64), u64> = Default::default();
    for (name, object) in &manifest.objects {
        for (pname, part) in &object.parts {
            let label = format!("{name:?}/{pname:?}");
            if part.encoding.is_some() {
                bad.push(format!("rule 4: {label} is stored under an encoding"));
            }
            match &part.digest {
                None => bad.push(format!("rule 4: {label} carries no digest")),
                Some(d) if !d.starts_with("xxh3:") => {
                    bad.push(format!("rule 4: {label} has a {d:?} digest, not xxh3"));
                }
                Some(_) => {}
            }
            if part.blob.shard.is_some() {
                continue; // rule 6 already accounts for this file
            }

            let offset = part.blob.offset;
            if let Some((_, len)) = placed.get(&offset) {
                // Sharing an already-placed blob. Legal, and it must be the
                // same bytes.
                if *len != part.blob.length {
                    bad.push(format!(
                        "rule 3: {label} shares offset {offset} with a blob of a different length"
                    ));
                }
                continue;
            }
            if let Some(d) = &part.digest {
                let key = (d.clone(), part.blob.length);
                if let Some(&first) = by_content.get(&key) {
                    bad.push(format!(
                        "rule 3: {label} repeats the content already at offset {first} \
                         instead of sharing that blob"
                    ));
                } else {
                    by_content.insert(key, offset);
                }
            }

            let expected = match align_up_canonical(cursor) {
                Some(e) => e,
                None => {
                    bad.push(format!("rule 2: {label} places a blob past the end of u64"));
                    break;
                }
            };
            // Each way of missing the expected offset is a different rule, and
            // saying which one tells the reader what to do about it.
            if !offset.is_multiple_of(ALIGN_CANONICAL) {
                bad.push(format!(
                    "rule 2: {label} is at offset {offset}, which is not a multiple of \
                     {ALIGN_CANONICAL}"
                ));
            } else if offset > expected {
                bad.push(format!(
                    "rule 1: {} bytes before {label} at {offset} belong to nothing; \
                     canonical form packs blobs with no gaps",
                    offset - expected
                ));
            } else if offset < expected {
                bad.push(format!(
                    "rule 3: {label} is at {offset}, before the {expected} that its place in \
                     (object, part) order gives it"
                ));
            }
            placed.insert(offset, (label, part.blob.length));
            cursor = offset + part.blob.length;
        }
    }

    // Rule 3's tail, which is also rule 1: the manifest comes straight after
    // the last blob and the footer straight after it. A gap is room for
    // something unreferenced, which is what rule 1 forbids.
    if let Some(expected) = align_up_canonical(cursor) {
        if manifest_at != expected {
            bad.push(format!(
                "rule 1: the manifest is at {manifest_at}, but the blobs end at {expected}; \
                 the space between belongs to nothing"
            ));
        }
    }
    if manifest_at + manifest_len + FOOTER_LEN != file_len {
        bad.push(format!(
            "rule 3: the footer does not immediately follow the manifest \
             (manifest ends at {}, file is {file_len})",
            manifest_at + manifest_len
        ));
    }
    bad
}

fn align_up_canonical(offset: u64) -> Option<u64> {
    offset
        .checked_add(ALIGN_CANONICAL - 1)
        .map(|n| n & !(ALIGN_CANONICAL - 1))
}

/// Opens a `.zt` store: reads the footer and the manifest blob, validates
/// both, and reports every occupied range.
pub(crate) fn store(store: &Store, vocab: &Vocabulary) -> Result<Parsed> {
    let file_len = store.len();
    if file_len < MIN_FILE_LEN {
        return Err(Error::reject(
            Rule::FileTooSmall,
            format!("file shorter than {MIN_FILE_LEN} B"),
        ));
    }
    if store.read(0, MAGIC.len() as u64)? != MAGIC {
        return Err(Error::reject(Rule::HeaderMagic, "bad header magic"));
    }
    let footer_bytes = store.read(file_len - FOOTER_LEN, FOOTER_LEN)?;
    check_footer(&footer_bytes)?;
    let Some(footer) = parse_footer(&footer_bytes)? else {
        return Ok(Parsed {
            manifest: None,
            occupied: frame_ranges(Vec::new(), file_len),
            manifest_at: None,
        });
    };
    let (_, data_end) = manifest_bounds(&footer, file_len)?;
    let manifest = decode_manifest(&store.read(footer.offset, footer.length)?, footer.hash)?;
    let occupied = validate_manifest(&manifest, data_end, (footer.offset, footer.length), vocab)?;
    Ok(Parsed {
        manifest: Some(manifest),
        occupied: frame_ranges(occupied, file_len),
        manifest_at: Some(footer.offset),
    })
}

/// Every metadata rule of §3.6. Returns the blob ranges that live in this
/// file, manifest blob included.
pub(crate) fn validate_manifest(
    manifest: &Manifest,
    data_end: u64,
    manifest_blob: (u64, u64),
    vocab: &Vocabulary,
) -> Result<Vec<(u64, u64)>> {
    for (name, shard) in &manifest.shards {
        check_shard_name(name)?;
        if shard.size < MIN_FILE_LEN {
            return Err(Error::reject(
                Rule::Schema,
                format!(
                    "shard {name:?}: size {} below minimum file size",
                    shard.size
                ),
            ));
        }
    }

    // Blob references grouped by the file they live in, for the per-file
    // identical-or-disjoint check (§2.4 / §3.6 rule 4). `None` is the
    // containing file, which also holds the manifest blob.
    let mut refs_by_shard: std::collections::BTreeMap<Option<&str>, Vec<(u64, u64)>> =
        std::collections::BTreeMap::new();
    refs_by_shard.insert(None, vec![manifest_blob]);

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
            let b = &part.blob;
            let shard = b.shard.as_deref();
            if b.offset % ALIGN_FLOOR != 0 || b.offset < ALIGN_FLOOR {
                return Err(Error::reject(
                    Rule::BlobAlignment,
                    format!("{name:?}/{pname:?}: offset {} misaligned", b.offset),
                ));
            }
            let region_end = match shard {
                None => data_end,
                Some(s) => manifest
                    .shards
                    .get(s)
                    .ok_or_else(|| {
                        Error::reject(
                            Rule::ShardRef,
                            format!("{name:?}/{pname:?}: shard {s:?} not in table"),
                        )
                    })?
                    .size
                    .checked_sub(FOOTER_LEN)
                    .ok_or_else(|| Error::reject(Rule::BlobBounds, "shard smaller than footer"))?,
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
                    .entry(shard)
                    .or_default()
                    .push((b.offset, b.length));
            }
            // Registered logical types pin their storage type (§4.2),
            // regardless of layout.
            if let Some(lt) = &part.logical {
                if let Some(required) = vocab.dtype_of(lt) {
                    if part.dtype != required {
                        return Err(Error::reject(
                            Rule::Schema,
                            format!("{name:?}/{pname:?}: type {lt:?} requires dtype {required:?}"),
                        ));
                    }
                }
            }
        }

        // Registered layouts validate their metadata rules; unregistered
        // layouts stay structural (§5.2).
        if let Some(profile) = vocab.layout(&obj.layout) {
            profile.validate(name, obj, vocab)?;
        }
    }

    // Identical-or-disjoint per referenced file (§2.4): sort, drop exact
    // duplicates, then any remaining pair that overlaps is a partial overlap.
    for (shard, refs) in &mut refs_by_shard {
        refs.sort_unstable();
        refs.dedup();
        for w in refs.windows(2) {
            let (a_off, a_len) = w[0];
            let (b_off, _) = w[1];
            if a_off + a_len > b_off {
                let location = match shard {
                    None => "this file".to_string(),
                    Some(s) => format!("shard {s:?}"),
                };
                return Err(Error::reject(
                    Rule::BlobOverlap,
                    format!(
                        "blobs at {a_off} (+{a_len}) and {b_off} in {location} \
                         partially overlap"
                    ),
                ));
            }
        }
    }
    Ok(refs_by_shard.remove(&None).unwrap_or_default())
}
