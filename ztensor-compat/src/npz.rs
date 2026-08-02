//! NumPy `.npz` (and single `.npy`) → zTensor object model projection.
//!
//! An `.npz` is a ZIP of `.npy` entries. Stored (uncompressed) entries are
//! zero-copy views into the mmap; deflated entries decompress lazily on
//! `read()` and are reported honestly as non-zero-copy.
//!
//! Refusals (never reinterpret): big-endian descrs, `fortran_order: True`
//! (reversing the shape would silently transpose the data), object dtypes.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use memmap2::Mmap;
use ztensor::{
    BlobRef, Caps, DType, Error, Layout, Manifest, Object, Part, Result, Source,
};

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("npz: {}", detail.into()))
}

/// numpy descr → (storage type, logical type). Little-endian or
/// byte-order-free only.
fn map_descr(descr: &str) -> Result<(DType, Option<&'static str>)> {
    Ok(match descr {
        "<f8" | "=f8" => (DType::F64, None),
        "<f4" | "=f4" => (DType::F32, None),
        "<f2" | "=f2" => (DType::F16, None),
        "<i8" | "=i8" => (DType::I64, None),
        "<i4" | "=i4" => (DType::I32, None),
        "<i2" | "=i2" => (DType::I16, None),
        "|i1" => (DType::I8, None),
        "<u8" | "=u8" => (DType::U64, None),
        "<u4" | "=u4" => (DType::U32, None),
        "<u2" | "=u2" => (DType::U16, None),
        "|u1" => (DType::U8, None),
        "|b1" => (DType::U8, Some("bool")),
        other => {
            return Err(Error::Unsupported(format!(
                "numpy descr {other:?} has no projection (big-endian and object dtypes are refused)"
            )))
        }
    })
}

pub(crate) struct NpyHeader {
    pub dtype: DType,
    pub ltype: Option<&'static str>,
    pub shape: Vec<u64>,
    /// Offset of raw data relative to the start of the `.npy` bytes.
    pub data_offset: usize,
}

/// Parses a `.npy` header (magic, version, header dict).
pub(crate) fn parse_npy_header(data: &[u8]) -> Result<NpyHeader> {
    if data.len() < 10 || &data[..6] != b"\x93NUMPY" {
        return Err(bad("not a .npy entry"));
    }
    let major = data[6];
    let (header_len, header_start): (usize, usize) = if major >= 2 {
        if data.len() < 12 {
            return Err(bad("truncated header"));
        }
        (
            u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize,
            12,
        )
    } else {
        (
            u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize,
            10,
        )
    };
    let header_end = header_start
        .checked_add(header_len)
        .filter(|&e| e <= data.len())
        .ok_or_else(|| bad("header extends past entry"))?;
    let header = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|_| bad("header is not UTF-8"))?;

    if find_flag(header, "'fortran_order'")? {
        return Err(Error::Unsupported(
            "npz: fortran_order arrays are refused (reversing the shape would silently \
             transpose the data)"
                .into(),
        ));
    }
    let descr = find_str(header, "'descr'")
        .ok_or_else(|| bad("header missing 'descr'"))?;
    let (dtype, ltype) = map_descr(&descr)?;
    let shape = find_shape(header)?;

    Ok(NpyHeader {
        dtype,
        ltype,
        shape,
        data_offset: header_end,
    })
}

fn find_str(header: &str, key: &str) -> Option<String> {
    let after = header.split(key).nth(1)?.trim_start().strip_prefix(':')?;
    let t = after.trim_start();
    let quote = t.chars().next().filter(|&q| q == '\'' || q == '"')?;
    let inner = &t[1..];
    Some(inner[..inner.find(quote)?].to_string())
}

fn find_flag(header: &str, key: &str) -> Result<bool> {
    let Some(after) = header.split(key).nth(1) else {
        return Ok(false);
    };
    let after = after.trim_start();
    let after = after.strip_prefix(':').unwrap_or(after).trim_start();
    Ok(after.starts_with("True"))
}

fn find_shape(header: &str) -> Result<Vec<u64>> {
    let after = header
        .split("'shape'")
        .nth(1)
        .and_then(|s| s.trim_start().strip_prefix(':'))
        .ok_or_else(|| bad("header missing 'shape'"))?;
    let open = after.find('(').ok_or_else(|| bad("shape missing '('"))?;
    let close = after.find(')').ok_or_else(|| bad("shape missing ')'"))?;
    let mut dims = Vec::new();
    for tok in after[open + 1..close].split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue; // "(3,)" or "()"
        }
        dims.push(
            tok.parse::<u64>()
                .map_err(|_| bad(format!("bad shape dim {tok:?}")))?,
        );
    }
    Ok(dims)
}

enum Location {
    /// Stored entry: absolute range in the file (zero-copy).
    Stored { offset: u64, length: u64 },
    /// Deflated entry: zip index + offset of raw data within the entry.
    Deflated { zip_index: usize, data_offset: usize, data_len: u64 },
}

pub struct Npz {
    mmap: Mmap,
    archive: RefCell<zip::ZipArchive<File>>,
    manifest: Manifest,
    locations: BTreeMap<String, Location>,
}

impl std::fmt::Debug for Npz {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Npz")
            .field("len", &self.mmap.len())
            .field("objects", &self.manifest.objects.len())
            .finish()
    }
}

impl Npz {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // SAFETY: read-only shared map of untrusted bytes.
        let mmap = unsafe { Mmap::map(&file)? };
        let mut archive = zip::ZipArchive::new(File::open(path)?)
            .map_err(|e| bad(format!("not a ZIP archive: {e}")))?;

        let mut objects = BTreeMap::new();
        let mut locations = BTreeMap::new();
        for i in 0..archive.len() {
            let mut entry = archive
                .by_index(i)
                .map_err(|e| bad(format!("ZIP entry {i}: {e}")))?;
            let Some(name) = entry.name().strip_suffix(".npy").map(str::to_string) else {
                continue;
            };

            let stored = entry.compression() == zip::CompressionMethod::Stored;
            let (header, location, blob, encoding) = if stored {
                let start = entry.data_start();
                let size = entry.size();
                let end = start
                    .checked_add(size)
                    .filter(|&e| e <= mmap.len() as u64)
                    .ok_or_else(|| bad(format!("entry {name:?} extends past file")))?;
                let header = parse_npy_header(&mmap[start as usize..end as usize])?;
                let data_off = start + header.data_offset as u64;
                let data_len = size - header.data_offset as u64;
                (
                    header,
                    Location::Stored {
                        offset: data_off,
                        length: data_len,
                    },
                    BlobRef {
                        shard: 0,
                        offset: data_off,
                        length: data_len,
                    },
                    None,
                )
            } else {
                // Parse only the header now; payload decompresses on read().
                let mut head = vec![0u8; 4096.min(entry.size() as usize)];
                entry
                    .read_exact(&mut head)
                    .map_err(|e| bad(format!("entry {name:?}: {e}")))?;
                let header = parse_npy_header(&head)?;
                let data_len = entry.size() - header.data_offset as u64;
                (
                    NpyHeader { ..header },
                    Location::Deflated {
                        zip_index: i,
                        data_offset: 0, // filled below from the header
                        data_len,
                    },
                    BlobRef {
                        shard: 0,
                        offset: entry.data_start(),
                        length: entry.compressed_size(),
                    },
                    Some("npz.deflate/1".to_string()),
                )
            };

            // Size equation: never trust the header blindly.
            let elems = header
                .shape
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| bad(format!("entry {name:?} shape overflows")))?;
            let expected = ztensor::logical_size(header.ltype, header.dtype, elems)
                .ok_or_else(|| bad("size not computable"))?;
            let actual = match &location {
                Location::Stored { length, .. } => *length,
                Location::Deflated { data_len, .. } => *data_len,
            };
            if actual != expected {
                return Err(bad(format!(
                    "entry {name:?} holds {actual} bytes but shape implies {expected}"
                )));
            }

            let mut location = location;
            if let Location::Deflated { data_offset, .. } = &mut location {
                *data_offset = header.data_offset;
            }
            let part = Part {
                dtype: header.dtype,
                ltype: header.ltype.map(str::to_string),
                blob,
                decoded_length: encoding.as_ref().map(|_| expected),
                encoding,
                digest: None,
            };
            let mut parts = BTreeMap::new();
            parts.insert("data".to_string(), part);
            locations.insert(name.clone(), location);
            objects.insert(
                name,
                Object {
                    shape: header.shape,
                    layout: Layout::Dense,
                    attributes: None,
                    parts,
                },
            );
        }

        Ok(Self {
            mmap,
            archive: RefCell::new(archive),
            manifest: Manifest {
                attributes: None,
                shards: BTreeMap::new(),
                objects,
            },
            locations,
        })
    }

    fn location(&self, name: &str, part: &str) -> Result<&Location> {
        if part != "data" {
            return Err(Error::NotFound(format!("part {name:?}/{part:?}")));
        }
        self.locations
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("object {name:?}")))
    }
}

impl Source for Npz {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read(&self, object: &str, part: &str) -> Result<Vec<u8>> {
        match self.location(object, part)? {
            Location::Stored { offset, length } => {
                Ok(self.mmap[*offset as usize..(*offset + *length) as usize].to_vec())
            }
            Location::Deflated {
                zip_index,
                data_offset,
                data_len,
            } => {
                let mut archive = self.archive.borrow_mut();
                let mut entry = archive
                    .by_index(*zip_index)
                    .map_err(|e| bad(format!("ZIP entry: {e}")))?;
                let mut bytes = Vec::with_capacity((*data_offset as u64 + data_len) as usize);
                entry
                    .read_to_end(&mut bytes)
                    .map_err(|e| bad(format!("decompress: {e}")))?;
                if bytes.len() as u64 != *data_offset as u64 + data_len {
                    return Err(bad("decompressed size mismatch"));
                }
                Ok(bytes.split_off(*data_offset))
            }
        }
    }

    fn view(&self, object: &str, part: &str) -> Result<&[u8]> {
        match self.location(object, part)? {
            Location::Stored { offset, length } => {
                Ok(&self.mmap[*offset as usize..(*offset + *length) as usize])
            }
            Location::Deflated { .. } => Err(Error::Unsupported(
                "deflated npz entry has no zero-copy view".into(),
            )),
        }
    }

    fn caps(&self, object: &str, part: &str) -> Result<Caps> {
        let loc = self.location(object, part)?;
        let (zero_copy, alignment) = match loc {
            Location::Stored { offset, .. } => (
                true,
                if *offset == 0 {
                    1
                } else {
                    1u64 << offset.trailing_zeros().min(63)
                },
            ),
            Location::Deflated { .. } => (false, 1),
        };
        Ok(Caps {
            zero_copy,
            alignment,
            verifiable: false,
            page_exclusive: false, // zip packs entries back to back
        })
    }
}
