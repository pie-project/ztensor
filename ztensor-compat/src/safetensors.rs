//! safetensors → zTensor object model projection.
//!
//! The format: `[header_len: u64 LE][JSON header][data]`, where the header
//! maps tensor names to `{dtype, shape, data_offsets: [begin, end]}` with
//! offsets relative to the data section.
//!
//! This is a deliberately strict reader. safetensors headers are JSON, and
//! JSON parsers resolve duplicate keys silently (the classic safetensors
//! aliasing attack); we defuse that class entirely by requiring the tensor
//! ranges to tile the data section exactly — sorted, gap-free,
//! overlap-free, ending at EOF. A file that fails any of it is rejected.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value as Json;
use ztensor::cbor::Value;
use ztensor::{
    BlobRef, Caps, DType, Error, Layout, Manifest, Object, Part, Result, Source,
};

/// Practical cap on the JSON header (matches the reference implementation).
const MAX_HEADER: u64 = 100 << 20;

pub struct Safetensors {
    mmap: Mmap,
    manifest: Manifest,
}

impl std::fmt::Debug for Safetensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Safetensors")
            .field("len", &self.mmap.len())
            .field("objects", &self.manifest.objects.len())
            .finish()
    }
}

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("safetensors: {}", detail.into()))
}

/// safetensors dtype → (storage type, logical type).
fn map_dtype(st: &str) -> Result<(DType, Option<&'static str>)> {
    Ok(match st {
        "F64" => (DType::F64, None),
        "F32" => (DType::F32, None),
        "F16" => (DType::F16, None),
        "BF16" => (DType::BF16, None),
        "I64" => (DType::I64, None),
        "I32" => (DType::I32, None),
        "I16" => (DType::I16, None),
        "I8" => (DType::I8, None),
        "U64" => (DType::U64, None),
        "U32" => (DType::U32, None),
        "U16" => (DType::U16, None),
        "U8" => (DType::U8, None),
        "BOOL" => (DType::U8, Some("bool")),
        "F8_E4M3" => (DType::U8, Some("f8_e4m3fn")),
        "F8_E5M2" => (DType::U8, Some("f8_e5m2")),
        other => {
            return Err(Error::Unsupported(format!(
                "safetensors dtype {other:?} has no registered projection"
            )))
        }
    })
}

impl Safetensors {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: read-only shared map of untrusted bytes.
        let mmap = unsafe { Mmap::map(&file)? };
        let manifest = project(&mmap)?;
        Ok(Self { mmap, manifest })
    }

    fn part(&self, name: &str, part: &str) -> Result<&Part> {
        let obj = self
            .manifest
            .objects
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("object {name:?}")))?;
        obj.parts
            .get(part)
            .ok_or_else(|| Error::NotFound(format!("part {name:?}/{part:?}")))
    }

    pub fn view(&self, name: &str, part: &str) -> Result<&[u8]> {
        let p = self.part(name, part)?;
        let start = p.blob.offset as usize;
        Ok(&self.mmap[start..start + p.blob.length as usize])
    }
}

impl Source for Safetensors {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read(&self, object: &str, part: &str) -> Result<Vec<u8>> {
        self.view(object, part).map(<[u8]>::to_vec)
    }

    fn view(&self, object: &str, part: &str) -> Result<&[u8]> {
        Safetensors::view(self, object, part)
    }

    fn caps(&self, object: &str, part: &str) -> Result<Caps> {
        let p = self.part(object, part)?;
        // Tensors tile the data section back to back (validated at open),
        // so neighbors touch exactly at this part's boundaries: the part is
        // page-exclusive iff both boundaries are page-aligned (or the end
        // is EOF). The header occupies everything before the first tensor,
        // so the start condition also covers it.
        let page = ztensor::page_size();
        let start = p.blob.offset;
        let end = start + p.blob.length;
        let page_exclusive = p.blob.length > 0
            && start % page == 0
            && (end % page == 0 || end == self.mmap.len() as u64);
        Ok(Caps {
            zero_copy: true,
            alignment: if start == 0 {
                1
            } else {
                1u64 << start.trailing_zeros().min(63)
            },
            verifiable: false, // safetensors carries no digests
            page_exclusive,
        })
    }
}

fn project(buf: &[u8]) -> Result<Manifest> {
    if buf.len() < 8 {
        return Err(bad("file shorter than the header length field"));
    }
    let header_len = u64::from_le_bytes(buf[..8].try_into().unwrap());
    if header_len > MAX_HEADER {
        return Err(bad(format!("header length {header_len} exceeds cap")));
    }
    let data_start = header_len
        .checked_add(8)
        .filter(|&s| s <= buf.len() as u64)
        .ok_or_else(|| bad("header extends past end of file"))?;
    let data_len = buf.len() as u64 - data_start;

    let header: Json = serde_json::from_slice(&buf[8..data_start as usize])
        .map_err(|e| bad(format!("header is not valid JSON: {e}")))?;
    let Json::Object(entries) = header else {
        return Err(bad("header root must be a JSON object"));
    };

    let mut attributes: Vec<(Value, Value)> = Vec::new();
    let mut objects = BTreeMap::new();
    // (begin, end, name) for the exact-tiling check.
    let mut ranges: Vec<(u64, u64)> = Vec::new();

    for (name, entry) in entries {
        if name == "__metadata__" {
            let Json::Object(meta) = entry else {
                return Err(bad("__metadata__ must be an object"));
            };
            for (k, v) in meta {
                let Json::String(s) = v else {
                    return Err(bad("__metadata__ values must be strings"));
                };
                attributes.push((Value::Text(k), Value::Text(s)));
            }
            continue;
        }

        let Json::Object(fields) = entry else {
            return Err(bad(format!("tensor {name:?} must be an object")));
        };
        let dtype_str = fields
            .get("dtype")
            .and_then(Json::as_str)
            .ok_or_else(|| bad(format!("tensor {name:?} missing dtype")))?;
        let (dtype, ltype) = map_dtype(dtype_str)?;
        let shape: Vec<u64> = fields
            .get("shape")
            .and_then(Json::as_array)
            .ok_or_else(|| bad(format!("tensor {name:?} missing shape")))?
            .iter()
            .map(|d| d.as_u64())
            .collect::<Option<_>>()
            .ok_or_else(|| bad(format!("tensor {name:?} has non-integer dims")))?;
        let offsets = fields
            .get("data_offsets")
            .and_then(Json::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| bad(format!("tensor {name:?} missing data_offsets")))?;
        let (begin, end) = match (offsets[0].as_u64(), offsets[1].as_u64()) {
            (Some(b), Some(e)) if b <= e && e <= data_len => (b, e),
            _ => return Err(bad(format!("tensor {name:?} has invalid data_offsets"))),
        };

        let elems = shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| bad(format!("tensor {name:?} shape overflows")))?;
        let expected = ztensor::logical_size(ltype, dtype, elems)
            .ok_or_else(|| bad(format!("tensor {name:?} size not computable")))?;
        if end - begin != expected {
            return Err(bad(format!(
                "tensor {name:?} holds {} bytes but shape implies {expected}",
                end - begin
            )));
        }

        ranges.push((begin, end));
        let part = Part {
            dtype,
            ltype: ltype.map(str::to_string),
            blob: BlobRef {
                shard: 0,
                offset: data_start + begin, // absolute file offset
                length: end - begin,
            },
            encoding: None,
            decoded_length: None,
            digest: None,
        };
        let mut parts = BTreeMap::new();
        parts.insert("data".to_string(), part);
        if objects
            .insert(
                name.clone(),
                Object {
                    shape,
                    layout: Layout::Dense,
                    attributes: None,
                    parts,
                },
            )
            .is_some()
        {
            return Err(bad(format!("duplicate tensor name {name:?}")));
        }
    }

    // Exact tiling of the data section: sorted, gap-free, overlap-free,
    // ending at EOF. This forecloses aliasing regardless of how the JSON
    // parser resolved duplicate keys.
    ranges.sort_unstable();
    let mut cursor = 0u64;
    for (begin, end) in &ranges {
        if *begin != cursor {
            return Err(bad(format!(
                "data section not tiled exactly: range starts at {begin}, expected {cursor}"
            )));
        }
        cursor = *end;
    }
    if cursor != data_len {
        return Err(bad(format!(
            "data section is {data_len} bytes but tensors cover {cursor}"
        )));
    }

    Ok(Manifest {
        attributes: if attributes.is_empty() {
            None
        } else {
            Some(Value::Map(attributes))
        },
        shards: BTreeMap::new(),
        objects,
    })
}
