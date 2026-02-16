//! GGUF format reader.
//!
//! Provides read-only access to `.gguf` files (GGML Universal Format) through the
//! unified `TensorReader` API. Uses memory mapping for zero-copy access.
//!
//! Standard types (F32, F16, BF16, I8, I16, I32, I64, F64) are mapped to native
//! `DType` values. Quantized types (Q4_0, Q8_0, Q2_K, etc.) are exposed as raw
//! `U8` byte arrays with the original GGUF type name stored in tensor attributes.
//!
//! Requires the `gguf` feature.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::error::Error;
use crate::models::{DType, Manifest, Object};
use crate::reader::{TensorData, TensorElement, TensorReader};

// ---- GGUF constants ----

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

// GGUF metadata value types
const GGUF_META_UINT8: u32 = 0;
const GGUF_META_INT8: u32 = 1;
const GGUF_META_UINT16: u32 = 2;
const GGUF_META_INT16: u32 = 3;
const GGUF_META_UINT32: u32 = 4;
const GGUF_META_INT32: u32 = 5;
const GGUF_META_FLOAT32: u32 = 6;
const GGUF_META_BOOL: u32 = 7;
const GGUF_META_STRING: u32 = 8;
const GGUF_META_ARRAY: u32 = 9;
const GGUF_META_UINT64: u32 = 10;
const GGUF_META_INT64: u32 = 11;
const GGUF_META_FLOAT64: u32 = 12;

// ---- GGUF tensor type info ----

/// Returns (block_size_elements, bytes_per_block, type_name) for a GGUF type ID.
fn gguf_type_info(type_id: u32) -> Result<(usize, usize, &'static str), Error> {
    match type_id {
        // Standard types
        0 => Ok((1, 4, "f32")),
        1 => Ok((1, 2, "f16")),
        24 => Ok((1, 1, "i8")),
        25 => Ok((1, 2, "i16")),
        26 => Ok((1, 4, "i32")),
        27 => Ok((1, 8, "i64")),
        28 => Ok((1, 8, "f64")),
        30 => Ok((1, 2, "bf16")),
        // Quantized types
        2 => Ok((32, 18, "q4_0")),
        3 => Ok((32, 20, "q4_1")),
        6 => Ok((32, 22, "q5_0")),
        7 => Ok((32, 24, "q5_1")),
        8 => Ok((32, 34, "q8_0")),
        9 => Ok((32, 40, "q8_1")),
        10 => Ok((256, 32, "q2_k")),
        11 => Ok((256, 48, "q3_k")),
        12 => Ok((256, 64, "q4_k")),
        13 => Ok((256, 80, "q5_k")),
        14 => Ok((256, 96, "q6_k")),
        15 => Ok((256, 128, "q8_k")),
        16 => Ok((256, 32, "iq2_xxs")),
        17 => Ok((256, 40, "iq2_xs")),
        18 => Ok((256, 48, "iq3_xxs")),
        19 => Ok((256, 16, "iq1_s")),
        20 => Ok((32, 20, "iq4_nl")),
        _ => Err(Error::UnsupportedDType(format!(
            "Unknown GGUF type ID: {}",
            type_id
        ))),
    }
}

/// Maps a GGUF type ID to a ztensor DType and optional quantized type name.
/// Standard types return (DType, None). Quantized types return (DType::U8, Some("q4_0")).
fn gguf_type_to_dtype(type_id: u32) -> Result<(DType, Option<&'static str>), Error> {
    match type_id {
        0 => Ok((DType::F32, None)),
        1 => Ok((DType::F16, None)),
        24 => Ok((DType::I8, None)),
        25 => Ok((DType::I16, None)),
        26 => Ok((DType::I32, None)),
        27 => Ok((DType::I64, None)),
        28 => Ok((DType::F64, None)),
        30 => Ok((DType::BF16, None)),
        // All quantized types → U8 raw bytes
        _ => {
            let (_, _, name) = gguf_type_info(type_id)?;
            Ok((DType::U8, Some(name)))
        }
    }
}

/// Calculates the total byte size of a tensor given its element count and GGUF type.
fn gguf_tensor_byte_size(n_elements: u64, type_id: u32) -> Result<usize, Error> {
    let (block_size, bytes_per_block, _) = gguf_type_info(type_id)?;
    let n_elements = usize::try_from(n_elements).map_err(|_| {
        Error::InvalidFileStructure("GGUF tensor element count exceeds platform limit".into())
    })?;
    if block_size == 1 {
        n_elements
            .checked_mul(bytes_per_block)
            .ok_or_else(|| Error::InvalidFileStructure("GGUF tensor byte size overflows".into()))
    } else {
        // Quantized: elements must be divisible by block_size
        let n_blocks = n_elements / block_size;
        n_blocks
            .checked_mul(bytes_per_block)
            .ok_or_else(|| Error::InvalidFileStructure("GGUF tensor byte size overflows".into()))
    }
}

// ---- Binary parser helpers ----

/// Cursor over a byte slice for sequential parsing.
struct ParseCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ParseCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::UnexpectedEof)?;
        if end > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u32_le(&mut self) -> Result<u32, Error> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u64_le(&mut self) -> Result<u64, Error> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String, Error> {
        let len = self.read_u64_le()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| {
            Error::InvalidFileStructure(format!("Invalid UTF-8 in GGUF string: {}", e))
        })
    }

    fn read_gguf_key(&mut self) -> Result<String, Error> {
        let len = self.read_u64_le()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| Error::InvalidFileStructure(format!("Invalid UTF-8 in GGUF key: {}", e)))
    }

    /// Skip a GGUF metadata value of the given type.
    fn skip_meta_value(&mut self, value_type: u32, depth: u32) -> Result<(), Error> {
        if depth > 32 {
            return Err(Error::InvalidFileStructure(
                "GGUF metadata nesting too deep".into(),
            ));
        }
        match value_type {
            GGUF_META_UINT8 | GGUF_META_INT8 | GGUF_META_BOOL => {
                self.read_bytes(1)?;
            }
            GGUF_META_UINT16 | GGUF_META_INT16 => {
                self.read_bytes(2)?;
            }
            GGUF_META_UINT32 | GGUF_META_INT32 | GGUF_META_FLOAT32 => {
                self.read_bytes(4)?;
            }
            GGUF_META_UINT64 | GGUF_META_INT64 | GGUF_META_FLOAT64 => {
                self.read_bytes(8)?;
            }
            GGUF_META_STRING => {
                self.read_string()?;
            }
            GGUF_META_ARRAY => {
                let elem_type = self.read_u32_le()?;
                let count = self.read_u64_le()?;
                for _ in 0..count {
                    self.skip_meta_value(elem_type, depth + 1)?;
                }
            }
            _ => {
                return Err(Error::InvalidFileStructure(format!(
                    "Unknown GGUF metadata value type: {}",
                    value_type
                )));
            }
        }
        Ok(())
    }
}

// ---- GgufReader ----

/// Reader for GGUF (.gguf) files.
///
/// Uses memory mapping for efficient zero-copy access to tensor data.
/// Quantized tensors are exposed as raw `U8` byte arrays with the GGUF
/// type name stored in the tensor's `gguf_type` attribute.
pub struct GgufReader {
    mmap: Mmap,
    pub manifest: Manifest,
    /// Maps tensor name → (byte_offset, byte_length) in the mmap.
    data_ranges: BTreeMap<String, (usize, usize)>,
}

/// Parsed tensor info from the GGUF header.
struct GgufTensorInfo {
    name: String,
    shape: Vec<u64>,
    type_id: u32,
    offset: u64, // relative to tensor data section start
}

impl GgufReader {
    /// Opens a GGUF file using memory mapping.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() < 24 {
            return Err(Error::InvalidFileStructure(
                "File too small to be a valid GGUF file".to_string(),
            ));
        }

        let mut cursor = ParseCursor::new(&mmap);

        // 1. Validate magic
        let magic = cursor.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(Error::InvalidMagicNumber {
                found: magic.to_vec(),
            });
        }

        // 2. Read header
        let version = cursor.read_u32_le()?;
        if version < 2 || version > 3 {
            return Err(Error::InvalidFileStructure(format!(
                "Unsupported GGUF version: {} (expected 2 or 3)",
                version
            )));
        }

        let tensor_count = cursor.read_u64_le()?;
        let metadata_kv_count = cursor.read_u64_le()?;

        // Sanity-check counts against file size to prevent huge allocations.
        let file_len = mmap.len() as u64;
        if tensor_count > file_len || metadata_kv_count > file_len {
            return Err(Error::InvalidFileStructure(
                "GGUF header counts exceed file size".into(),
            ));
        }

        // 3. Parse metadata KV pairs
        let mut alignment: usize = 32; // default GGUF alignment

        for _ in 0..metadata_kv_count {
            let key = cursor.read_gguf_key()?;
            let value_type = cursor.read_u32_le()?;

            if key == "general.alignment" && value_type == GGUF_META_UINT32 {
                alignment = cursor.read_u32_le()? as usize;
            } else {
                cursor.skip_meta_value(value_type, 0)?;
            }
        }

        // 4. Parse tensor info entries
        let mut tensor_infos = Vec::with_capacity(tensor_count as usize);

        for _ in 0..tensor_count {
            let name = cursor.read_gguf_key()?;
            let n_dims = cursor.read_u32_le()?;
            if n_dims > 64 {
                return Err(Error::InvalidFileStructure(format!(
                    "GGUF tensor has {} dimensions (max 64)",
                    n_dims
                )));
            }

            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cursor.read_u64_le()?);
            }
            // GGML stores dimensions in reverse order; reverse to C row-major
            dims.reverse();

            let type_id = cursor.read_u32_le()?;
            let offset = cursor.read_u64_le()?;

            tensor_infos.push(GgufTensorInfo {
                name,
                shape: dims,
                type_id,
                offset,
            });
        }

        // 5. Calculate tensor data section start (aligned)
        let tensor_data_start =
            crate::utils::align_offset_to(cursor.pos as u64, alignment as u64).0 as usize;

        // 6. Build manifest and data ranges
        let mut objects = BTreeMap::new();
        let mut data_ranges = BTreeMap::new();

        for info in &tensor_infos {
            let (dtype, quant_type) = gguf_type_to_dtype(info.type_id)?;

            let n_elements: u64 = if info.shape.is_empty() {
                1
            } else {
                info.shape
                    .iter()
                    .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| {
                        Error::InvalidFileStructure(format!(
                            "Tensor '{}' shape overflows",
                            info.name
                        ))
                    })?
            };
            let byte_size = gguf_tensor_byte_size(n_elements, info.type_id)?;

            let abs_offset = tensor_data_start
                .checked_add(info.offset as usize)
                .ok_or_else(|| {
                    Error::InvalidFileStructure(format!("Tensor '{}' offset overflows", info.name))
                })?;

            // Bounds check
            let end = abs_offset.checked_add(byte_size).ok_or_else(|| {
                Error::InvalidFileStructure(format!("Tensor '{}' extends beyond file", info.name))
            })?;
            if end > mmap.len() {
                return Err(Error::InvalidFileStructure(format!(
                    "Tensor '{}' extends beyond file: offset {} + size {} > file size {}",
                    info.name,
                    abs_offset,
                    byte_size,
                    mmap.len()
                )));
            }

            // For quantized types, the shape represents the raw byte array
            let (obj_shape, obj_dtype) = if quant_type.is_some() {
                (vec![byte_size as u64], DType::U8)
            } else {
                (info.shape.clone(), dtype)
            };

            let mut obj = Object::dense(obj_shape, obj_dtype, abs_offset as u64, byte_size as u64);

            if let Some(qt) = quant_type {
                let mut attrs = BTreeMap::new();
                attrs.insert(
                    "gguf_type".to_string(),
                    ciborium::Value::Text(qt.to_string()),
                );
                attrs.insert(
                    "original_shape".to_string(),
                    ciborium::Value::Array(
                        info.shape
                            .iter()
                            .map(|&d| ciborium::Value::Integer(d.into()))
                            .collect(),
                    ),
                );
                obj.attributes = Some(attrs);
            }

            data_ranges.insert(info.name.clone(), (abs_offset, byte_size));
            objects.insert(info.name.clone(), obj);
        }

        let manifest = Manifest {
            version: format!("gguf-v{}", version),
            attributes: None,
            objects,
        };

        Ok(Self {
            mmap,
            manifest,
            data_ranges,
        })
    }

    /// Gets a zero-copy reference to an object's raw data.
    pub fn view(&self, name: &str) -> Result<&[u8], Error> {
        let (offset, length) = self
            .data_ranges
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;
        Ok(&self.mmap[*offset..*offset + *length])
    }

    /// Gets a typed zero-copy reference to an object's data.
    pub fn view_as<T: TensorElement>(&self, name: &str) -> Result<&[T], Error> {
        let dtype = self
            .manifest
            .objects
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?
            .data_dtype()?;
        if T::DTYPE != dtype {
            return Err(Error::TypeMismatch {
                expected: dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }
        crate::reader::bytes_as_typed(self.view(name)?)
    }

    /// Reads object data as a copy.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, Error> {
        self.view(name).map(|s| s.to_vec())
    }

    /// Reads object data as a typed vector.
    pub fn read_as<T: TensorElement>(&self, name: &str) -> Result<Vec<T>, Error> {
        let dtype = self
            .manifest
            .objects
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?
            .data_dtype()?;
        if T::DTYPE != dtype {
            return Err(Error::TypeMismatch {
                expected: dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }
        crate::reader::bytes_to_typed_vec(self.view(name)?)
    }
}

impl TensorReader for GgufReader {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read_data(&self, name: &str) -> Result<TensorData<'_>, Error> {
        let slice = self.view(name)?;
        Ok(TensorData::Borrowed(slice))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_type_info() {
        // Standard types
        let (bs, bpb, name) = gguf_type_info(0).unwrap();
        assert_eq!((bs, bpb, name), (1, 4, "f32"));

        let (bs, bpb, name) = gguf_type_info(1).unwrap();
        assert_eq!((bs, bpb, name), (1, 2, "f16"));

        // Quantized type
        let (bs, bpb, name) = gguf_type_info(2).unwrap();
        assert_eq!((bs, bpb, name), (32, 18, "q4_0"));

        // Unknown type
        assert!(gguf_type_info(255).is_err());
    }

    #[test]
    fn test_gguf_tensor_byte_size() {
        // F32: 1024 elements × 4 bytes
        assert_eq!(gguf_tensor_byte_size(1024, 0).unwrap(), 4096);

        // F16: 1024 elements × 2 bytes
        assert_eq!(gguf_tensor_byte_size(1024, 1).unwrap(), 2048);

        // Q4_0: 1024 elements / 32 block_size × 18 bytes_per_block
        assert_eq!(gguf_tensor_byte_size(1024, 2).unwrap(), 576);

        // Q8_0: 1024 elements / 32 block_size × 34 bytes_per_block
        assert_eq!(gguf_tensor_byte_size(1024, 8).unwrap(), 1088);
    }

    #[test]
    fn test_gguf_type_to_dtype() {
        // Standard types
        assert_eq!(gguf_type_to_dtype(0).unwrap(), (DType::F32, None));
        assert_eq!(gguf_type_to_dtype(1).unwrap(), (DType::F16, None));
        assert_eq!(gguf_type_to_dtype(24).unwrap(), (DType::I8, None));
        assert_eq!(gguf_type_to_dtype(30).unwrap(), (DType::BF16, None));

        // Quantized types
        assert_eq!(gguf_type_to_dtype(2).unwrap(), (DType::U8, Some("q4_0")));
        assert_eq!(gguf_type_to_dtype(8).unwrap(), (DType::U8, Some("q8_0")));
    }
}
