//! GGUF → zTensor object model projection.
//!
//! Layout: `"GGUF"` magic, u32 version (2 or 3, little-endian), u64 tensor
//! count, u64 metadata KV count, the KVs, the tensor infos, then the data
//! section aligned to `general.alignment` (default 32).
//!
//! Projection choices:
//! - Standard element types map to `dense` objects directly.
//! - Quantized tensors keep their **logical shape** and get layout
//!   `gguf.<type>/1` with a single u8 `"data"` part holding the raw
//!   blocks, plus `elems_per_block` / `block_bytes` attributes. Nothing is
//!   dequantized; unknown type ids reject the file (never reinterpret).
//! - All metadata KVs (including tokenizer tables) become file attributes.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use ztensor::cbor::Value;
use ztensor::{
    BlobRef, Caps, DType, Error, Layout, Manifest, Object, Part, Result, Source,
};

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("gguf: {}", detail.into()))
}

/// ggml type table: `(elems_per_block, block_bytes, name)`. Sizes are
/// `sizeof(block_*)` from ggml; a wrong entry here corrupts every offset
/// after it, so entries are kept in sync with ggml-common.h.
fn type_info(id: u32) -> Result<(u64, u64, &'static str)> {
    Ok(match id {
        0 => (1, 4, "f32"),
        1 => (1, 2, "f16"),
        2 => (32, 18, "q4_0"),
        3 => (32, 20, "q4_1"),
        6 => (32, 22, "q5_0"),
        7 => (32, 24, "q5_1"),
        8 => (32, 34, "q8_0"),
        9 => (32, 36, "q8_1"),
        10 => (256, 84, "q2_k"),
        11 => (256, 110, "q3_k"),
        12 => (256, 144, "q4_k"),
        13 => (256, 176, "q5_k"),
        14 => (256, 210, "q6_k"),
        15 => (256, 292, "q8_k"),
        16 => (256, 66, "iq2_xxs"),
        17 => (256, 74, "iq2_xs"),
        18 => (256, 98, "iq3_xxs"),
        19 => (256, 50, "iq1_s"),
        20 => (32, 18, "iq4_nl"),
        21 => (256, 110, "iq3_s"),
        22 => (256, 82, "iq2_s"),
        23 => (256, 136, "iq4_xs"),
        24 => (1, 1, "i8"),
        25 => (1, 2, "i16"),
        26 => (1, 4, "i32"),
        27 => (1, 8, "i64"),
        28 => (1, 8, "f64"),
        29 => (256, 56, "iq1_m"),
        30 => (1, 2, "bf16"),
        39 => (32, 17, "mxfp4"),
        other => {
            return Err(Error::Unsupported(format!(
                "gguf tensor type id {other} has no registered projection"
            )))
        }
    })
}

/// Element types with a direct dense projection.
fn element_dtype(id: u32) -> Option<DType> {
    Some(match id {
        0 => DType::F32,
        1 => DType::F16,
        24 => DType::I8,
        25 => DType::I16,
        26 => DType::I32,
        27 => DType::I64,
        28 => DType::F64,
        30 => DType::BF16,
        _ => return None,
    })
}

pub struct Gguf {
    mmap: Mmap,
    manifest: Manifest,
    /// Sorted occupied ranges for the page-exclusivity check.
    ranges: Vec<(u64, u64)>,
}

impl std::fmt::Debug for Gguf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gguf")
            .field("len", &self.mmap.len())
            .field("objects", &self.manifest.objects.len())
            .finish()
    }
}

// ---- cursor -----------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| bad("unexpected end of file"))?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > self.data.len() as u64 {
            return Err(bad("string length exceeds file"));
        }
        String::from_utf8(self.take(len as usize)?.to_vec())
            .map_err(|_| bad("invalid UTF-8 in string"))
    }

    /// Parses one metadata value into a CBOR attribute value.
    fn meta_value(&mut self, vtype: u32, depth: u32) -> Result<Value> {
        if depth > 32 {
            return Err(bad("metadata nesting too deep"));
        }
        Ok(match vtype {
            0 => Value::Uint(self.take(1)?[0] as u64),
            1 => int_value(self.take(1)?[0] as i8 as i64),
            2 => Value::Uint(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => int_value(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Uint(self.u32()? as u64),
            5 => int_value(i32::from_le_bytes(self.take(4)?.try_into().unwrap()) as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Text(self.string()?),
            9 => {
                let elem_type = self.u32()?;
                let count = self.u64()?;
                // Even a 1-byte element type needs a byte on disk, so a
                // count beyond the remaining bytes is a lie — and the
                // materialized `Value`s are far larger than their encoding,
                // so this bound is what keeps the projection proportional
                // to the file.
                let remaining = (self.data.len() - self.pos) as u64;
                if count > remaining {
                    return Err(bad("array length exceeds remaining bytes"));
                }
                let mut items = Vec::with_capacity(count.min(1 << 16) as usize);
                for _ in 0..count {
                    items.push(self.meta_value(elem_type, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::Uint(self.u64()?),
            11 => int_value(self.u64()? as i64),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => return Err(bad(format!("unknown metadata value type {other}"))),
        })
    }
}

fn int_value(v: i64) -> Value {
    if v < 0 {
        Value::Nint((-1 - v) as u64)
    } else {
        Value::Uint(v as u64)
    }
}

// ---- projection -------------------------------------------------------

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: read-only shared map of untrusted bytes.
        let mmap = unsafe { Mmap::map(&file)? };
        let (manifest, ranges) = project(&mmap)?;
        Ok(Self {
            mmap,
            manifest,
            ranges,
        })
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
}

impl Source for Gguf {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read(&self, object: &str, part: &str) -> Result<Vec<u8>> {
        Source::view(self, object, part).map(<[u8]>::to_vec)
    }

    fn view(&self, object: &str, part: &str) -> Result<&[u8]> {
        let p = self.part(object, part)?;
        crate::safe::slice("gguf tensor", &self.mmap, p.blob.offset, p.blob.length)
    }

    fn caps(&self, object: &str, part: &str) -> Result<Caps> {
        let p = self.part(object, part)?;
        let page = ztensor::page_size();
        let (start, len) = (p.blob.offset, p.blob.length);
        let env_start = start & !(page - 1);
        let env_end = (start + len).div_ceil(page).saturating_mul(page);
        let page_exclusive = len > 0
            && self.ranges.iter().all(|&(o, l)| {
                (o, l) == (start, len) || o + l <= env_start || o >= env_end
            });
        Ok(Caps {
            zero_copy: true,
            alignment: if start == 0 {
                1
            } else {
                1u64 << start.trailing_zeros().min(63)
            },
            verifiable: false,
            page_exclusive,
        })
    }
}

fn project(buf: &[u8]) -> Result<(Manifest, Vec<(u64, u64)>)> {
    let mut c = Cursor { data: buf, pos: 0 };
    if c.take(4)? != b"GGUF" {
        return Err(bad("bad magic"));
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        return Err(Error::Unsupported(format!(
            "gguf version {version} (little-endian v2/v3 only)"
        )));
    }
    let tensor_count = c.u64()?;
    let kv_count = c.u64()?;
    if tensor_count > buf.len() as u64 || kv_count > buf.len() as u64 {
        return Err(bad("header counts exceed file size"));
    }

    // Metadata.
    let mut alignment = 32u64;
    // A KV needs at least a 8-byte length + 4-byte type on disk; a count
    // beyond that is a lie and must not drive the allocation.
    let mut attributes: Vec<(Value, Value)> =
        Vec::with_capacity(crate::safe::capacity(kv_count, 13, buf.len()));
    for _ in 0..kv_count {
        let key = c.string()?;
        let vtype = c.u32()?;
        let value = c.meta_value(vtype, 0)?;
        if key == "general.alignment" {
            alignment = value
                .as_u64()
                .filter(|a| a.is_power_of_two())
                .ok_or_else(|| bad("general.alignment must be a power-of-two uint"))?;
        }
        attributes.push((Value::Text(key), value));
    }

    // Tensor infos.
    struct Info {
        name: String,
        shape: Vec<u64>,
        type_id: u32,
        offset: u64,
    }
    let mut infos = Vec::with_capacity(crate::safe::capacity(tensor_count, 24, buf.len()));
    for _ in 0..tensor_count {
        let name = c.string()?;
        let n_dims = c.u32()?;
        if n_dims > 64 {
            return Err(bad(format!("tensor {name:?} has {n_dims} dims")));
        }
        // ggml stores dims fastest-first; reverse to row-major.
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(c.u64()?);
        }
        shape.reverse();
        let type_id = c.u32()?;
        let offset = c.u64()?;
        infos.push(Info {
            name,
            shape,
            type_id,
            offset,
        });
    }

    let data_start = crate::safe::mul(
        "gguf data section",
        (c.pos as u64).div_ceil(alignment),
        alignment,
    )?;
    if data_start > buf.len() as u64 {
        return Err(bad("aligned data section starts past end of file"));
    }
    let data_len = buf.len() as u64 - data_start;

    let mut objects = BTreeMap::new();
    let mut ranges = vec![(0u64, data_start.min(buf.len() as u64))]; // header region
    for info in infos {
        let (epb, block_bytes, type_name) = type_info(info.type_id)?;
        // Blocks are per-row in ggml: the fastest dim must divide evenly.
        let fastest = info.shape.last().copied().unwrap_or(1);
        if fastest % epb != 0 {
            return Err(bad(format!(
                "tensor {:?}: fastest dim {fastest} not divisible by block size {epb}",
                info.name
            )));
        }
        let elems = crate::safe::product("gguf shape", &info.shape)?;
        let byte_size = crate::safe::mul("gguf tensor size", elems / epb, block_bytes)?;
        let abs = crate::safe::add("gguf tensor offset", data_start, info.offset)?;
        crate::safe::range("gguf tensor", abs, byte_size, buf.len())?;
        let _ = data_len;

        let (layout, attrs, dtype) = match element_dtype(info.type_id) {
            Some(dt) => (Layout::Dense, None, dt),
            None => (
                Layout::Other(format!("gguf.{type_name}/1")),
                Some(Value::Map(vec![
                    (
                        Value::Text("elems_per_block".into()),
                        Value::Uint(epb),
                    ),
                    (Value::Text("block_bytes".into()), Value::Uint(block_bytes)),
                ])),
                DType::U8,
            ),
        };
        let part = Part {
            dtype,
            ltype: None,
            blob: BlobRef {
                shard: 0,
                offset: abs,
                length: byte_size,
            },
            encoding: None,
            decoded_length: None,
            digest: None,
        };
        let mut parts = BTreeMap::new();
        parts.insert("data".to_string(), part);
        ranges.push((abs, byte_size));
        if objects
            .insert(
                info.name.clone(),
                Object {
                    shape: info.shape,
                    layout,
                    attributes: attrs,
                    parts,
                },
            )
            .is_some()
        {
            return Err(bad(format!("duplicate tensor name {:?}", info.name)));
        }
    }

    // Ranges must not overlap (offsets are writer-controlled).
    ranges.sort_unstable();
    ranges.dedup();
    for w in ranges.windows(2) {
        if w[0].0 + w[0].1 > w[1].0 {
            return Err(bad(format!(
                "tensor ranges overlap at {} and {}",
                w[0].0, w[1].0
            )));
        }
    }

    Ok((
        Manifest {
            attributes: if attributes.is_empty() {
                None
            } else {
                Some(Value::Map(attributes))
            },
            shards: BTreeMap::new(),
            objects,
        },
        ranges,
    ))
}
