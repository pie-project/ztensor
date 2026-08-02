//! ONNX → zTensor object model projection.
//!
//! Extracts graph initializers (the model weights) from the protobuf
//! stream with a minimal hand-written wire-format parser — no protobuf
//! dependency. `raw_data` tensors are zero-copy views into the mmap; the
//! typed repeated fields (`float_data`, `int32_data`, ...) are converted
//! to little-endian bytes per the ONNX storage rules (small types are
//! stored one element per int32) and held as owned buffers.
//!
//! Graphs, nodes, and attributes are out of scope: this reads weights,
//! not computation. External data files are refused, not resolved.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use ztensor::{
    BlobRef, Caps, DType, Error, Layout, Manifest, Object, Part, Result, Source,
};

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("onnx: {}", detail.into()))
}

/// ONNX TensorProto.DataType → (storage, logical).
fn map_dtype(id: u64) -> Result<(DType, Option<&'static str>)> {
    Ok(match id {
        1 => (DType::F32, None),
        2 => (DType::U8, None),
        3 => (DType::I8, None),
        4 => (DType::U16, None),
        5 => (DType::I16, None),
        6 => (DType::I32, None),
        7 => (DType::I64, None),
        9 => (DType::U8, Some("bool")),
        10 => (DType::F16, None),
        11 => (DType::F64, None),
        12 => (DType::U32, None),
        13 => (DType::U64, None),
        16 => (DType::BF16, None),
        17 => (DType::U8, Some("f8_e4m3fn")),
        18 => (DType::U8, Some("f8_e4m3fnuz")),
        19 => (DType::U8, Some("f8_e5m2")),
        20 => (DType::U8, Some("f8_e5m2fnuz")),
        other => {
            return Err(Error::Unsupported(format!(
                "onnx data type {other} has no registered projection"
            )))
        }
    })
}

// ---- protobuf wire format ---------------------------------------------

const VARINT: u32 = 0;
const I64BIT: u32 = 1;
const LEN: u32 = 2;
const I32BIT: u32 = 5;

struct Pb<'a> {
    data: &'a [u8],
    pos: usize,
    /// Absolute offset of `data[0]` in the file, for zero-copy ranges.
    base: usize,
}

impl<'a> Pb<'a> {
    fn done(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = *self
                .data
                .get(self.pos)
                .ok_or_else(|| bad("truncated varint"))?;
            self.pos += 1;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift >= 64 {
                return Err(bad("varint too long"));
            }
        }
    }

    fn tag(&mut self) -> Result<(u32, u32)> {
        let v = self.varint()?;
        Ok(((v >> 3) as u32, (v & 7) as u32))
    }

    fn bytes(&mut self) -> Result<(usize, &'a [u8])> {
        let len = self.varint()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| bad("length-delimited field truncated"))?;
        let abs = self.base + self.pos;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok((abs, s))
    }

    fn skip(&mut self, wire: u32) -> Result<()> {
        match wire {
            VARINT => {
                self.varint()?;
            }
            I64BIT => {
                self.pos = self
                    .pos
                    .checked_add(8)
                    .filter(|&e| e <= self.data.len())
                    .ok_or_else(|| bad("truncated fixed64"))?;
            }
            LEN => {
                self.bytes()?;
            }
            I32BIT => {
                self.pos = self
                    .pos
                    .checked_add(4)
                    .filter(|&e| e <= self.data.len())
                    .ok_or_else(|| bad("truncated fixed32"))?;
            }
            other => return Err(bad(format!("unknown wire type {other}"))),
        }
        Ok(())
    }
}

// ---- TensorProto ------------------------------------------------------

enum TensorData {
    Raw { offset: u64, length: u64 },
    Owned(Vec<u8>),
}

struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    data_type: u64,
    data: TensorData,
}

fn parse_tensor(data: &[u8], base: usize) -> Result<TensorInfo> {
    let mut pb = Pb { data, pos: 0, base };
    let mut name = String::new();
    let mut dims = Vec::new();
    let mut data_type = 0u64;
    let mut raw: Option<(u64, u64)> = None;
    // The typed repeated fields, kept as their wire integers until the
    // data type is known.
    let mut i32s: Vec<u32> = Vec::new();
    let mut i64s: Vec<u64> = Vec::new();
    let mut f32s: Vec<u8> = Vec::new();
    let mut f64s: Vec<u8> = Vec::new();
    let mut external = false;

    while !pb.done() {
        let (field, wire) = pb.tag()?;
        match (field, wire) {
            (1, LEN) => {
                let (_, b) = pb.bytes()?;
                let mut sub = Pb { data: b, pos: 0, base: 0 };
                while !sub.done() {
                    dims.push(sub.varint()?);
                }
            }
            (1, VARINT) => dims.push(pb.varint()?),
            (2, VARINT) => data_type = pb.varint()?,
            (4, LEN) => f32s = pb.bytes()?.1.to_vec(), // packed floats: already LE bytes
            (5, LEN) => {
                let (_, b) = pb.bytes()?;
                let mut sub = Pb { data: b, pos: 0, base: 0 };
                while !sub.done() {
                    i32s.push(sub.varint()? as u32);
                }
            }
            (5, VARINT) => i32s.push(pb.varint()? as u32),
            (7, LEN) | (11, LEN) => {
                let (_, b) = pb.bytes()?;
                let mut sub = Pb { data: b, pos: 0, base: 0 };
                while !sub.done() {
                    i64s.push(sub.varint()?);
                }
            }
            (7, VARINT) | (11, VARINT) => i64s.push(pb.varint()?),
            (8, LEN) => {
                name = String::from_utf8(pb.bytes()?.1.to_vec())
                    .map_err(|_| bad("tensor name is not UTF-8"))?;
            }
            (9, LEN) => {
                let (abs, b) = pb.bytes()?;
                raw = Some((abs as u64, b.len() as u64));
            }
            (10, LEN) => f64s = pb.bytes()?.1.to_vec(), // packed doubles
            (13, LEN) => {
                // external_data entries: presence alone means the payload
                // lives in another file.
                external = true;
                pb.bytes()?;
            }
            (14, VARINT) => {
                if pb.varint()? == 1 {
                    external = true;
                }
            }
            (_, w) => pb.skip(w)?,
        }
    }

    if external {
        return Err(Error::Unsupported(format!(
            "onnx: tensor {name:?} uses external data; internalize it first \
             (onnx.load_external_data_for_model)"
        )));
    }

    let (dtype, _ltype) = map_dtype(data_type)?;
    let width = dtype.width() as usize;

    // Assemble owned bytes from typed fields per ONNX storage rules:
    // int32_data carries every type of width ≤ 4 (one element per entry),
    // int64_data carries i64, uint64_data carries u32/u64, float/double
    // are packed IEEE bytes.
    let data = if let Some((offset, length)) = raw {
        TensorData::Raw { offset, length }
    } else if !f32s.is_empty() {
        TensorData::Owned(f32s)
    } else if !f64s.is_empty() {
        TensorData::Owned(f64s)
    } else if !i32s.is_empty() {
        let mut out = Vec::with_capacity(i32s.len() * width);
        for v in &i32s {
            out.extend_from_slice(&v.to_le_bytes()[..width.min(4)]);
        }
        TensorData::Owned(out)
    } else if !i64s.is_empty() {
        let mut out = Vec::with_capacity(i64s.len() * width);
        for v in &i64s {
            out.extend_from_slice(&v.to_le_bytes()[..width.min(8)]);
        }
        TensorData::Owned(out)
    } else {
        TensorData::Owned(Vec::new())
    };

    Ok(TensorInfo {
        name,
        dims,
        data_type,
        data,
    })
}

// ---- reader -----------------------------------------------------------

enum Loc {
    Range { offset: u64, length: u64 },
    Owned(Vec<u8>),
}

pub struct Onnx {
    mmap: Mmap,
    manifest: Manifest,
    locations: BTreeMap<String, Loc>,
}

impl std::fmt::Debug for Onnx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Onnx")
            .field("len", &self.mmap.len())
            .field("objects", &self.manifest.objects.len())
            .finish()
    }
}

impl Onnx {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: read-only shared map of untrusted bytes.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.is_empty() {
            return Err(bad("empty file"));
        }

        // ModelProto.graph is field 7.
        let mut pb = Pb {
            data: &mmap,
            pos: 0,
            base: 0,
        };
        let mut graph: Option<(usize, Vec<u8>)> = None;
        while !pb.done() {
            let (field, wire) = pb.tag()?;
            if field == 7 && wire == LEN {
                let (abs, b) = pb.bytes()?;
                graph = Some((abs, b.to_vec()));
                break;
            }
            pb.skip(wire)?;
        }
        let (graph_base, graph_bytes) =
            graph.ok_or_else(|| bad("no graph field in ModelProto"))?;

        // GraphProto.initializer is field 5.
        let mut objects = BTreeMap::new();
        let mut locations = BTreeMap::new();
        let mut pb = Pb {
            data: &graph_bytes,
            pos: 0,
            base: graph_base,
        };
        while !pb.done() {
            let (field, wire) = pb.tag()?;
            if field != 5 || wire != LEN {
                pb.skip(wire)?;
                continue;
            }
            let (abs, b) = pb.bytes()?;
            let info = parse_tensor(b, abs)?;
            if info.name.is_empty() {
                continue;
            }
            let (dtype, ltype) = map_dtype(info.data_type)?;
            let elems = info
                .dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| bad(format!("tensor {:?} shape overflows", info.name)))?;
            let expected = ztensor::logical_size(ltype, dtype, elems)
                .ok_or_else(|| bad("size not computable"))?;
            let actual = match &info.data {
                TensorData::Raw { length, .. } => *length,
                TensorData::Owned(v) => v.len() as u64,
            };
            if actual != expected {
                return Err(bad(format!(
                    "tensor {:?} holds {actual} bytes but shape implies {expected}",
                    info.name
                )));
            }

            let (loc, blob, encoding) = match info.data {
                TensorData::Raw { offset, length } => (
                    Loc::Range { offset, length },
                    BlobRef {
                        shard: 0,
                        offset,
                        length,
                    },
                    None,
                ),
                TensorData::Owned(v) => (
                    Loc::Owned(v),
                    BlobRef {
                        shard: 0,
                        offset: 0,
                        length: 0,
                    },
                    Some("onnx.typed/1".to_string()),
                ),
            };
            let part = Part {
                dtype,
                ltype: ltype.map(str::to_string),
                blob,
                decoded_length: encoding.as_ref().map(|_| expected),
                encoding,
                digest: None,
            };
            let mut parts = BTreeMap::new();
            parts.insert("data".to_string(), part);
            locations.insert(info.name.clone(), loc);
            if objects
                .insert(
                    info.name.clone(),
                    Object {
                        shape: info.dims,
                        layout: Layout::Dense,
                        attributes: None,
                        parts,
                    },
                )
                .is_some()
            {
                return Err(bad(format!("duplicate initializer {:?}", info.name)));
            }
        }

        Ok(Self {
            mmap,
            manifest: Manifest {
                attributes: None,
                shards: BTreeMap::new(),
                objects,
            },
            locations,
        })
    }

    fn location(&self, name: &str, part: &str) -> Result<&Loc> {
        if part != "data" {
            return Err(Error::NotFound(format!("part {name:?}/{part:?}")));
        }
        self.locations
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("object {name:?}")))
    }
}

impl Source for Onnx {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read(&self, object: &str, part: &str) -> Result<Vec<u8>> {
        Source::view(self, object, part).map(<[u8]>::to_vec)
    }

    fn view(&self, object: &str, part: &str) -> Result<&[u8]> {
        match self.location(object, part)? {
            Loc::Range { offset, length } => {
                Ok(&self.mmap[*offset as usize..(*offset + *length) as usize])
            }
            Loc::Owned(bytes) => Ok(bytes),
        }
    }

    fn caps(&self, object: &str, part: &str) -> Result<Caps> {
        let alignment = match self.location(object, part)? {
            Loc::Range { offset, .. } if *offset > 0 => {
                1u64 << offset.trailing_zeros().min(63)
            }
            _ => 1,
        };
        Ok(Caps {
            zero_copy: true,
            alignment,
            verifiable: false,
            page_exclusive: false,
        })
    }
}
