//! Read-only backwards compatibility with zTensor v0.1.0 format.
//!
//! This module provides legacy support for reading v0.1.0 files.
//! To drop legacy support, simply remove this file and the `pub mod compat;` line from lib.rs.
//!
//! v0.1.0 differences:
//! - Magic: "ZTEN0001" (vs "ZTEN1000" in v1.1)
//! - No footer magic
//! - Manifest is a CBOR array (not map with "objects")
//! - Each tensor has: name, offset, size, dtype, shape, encoding, layout
//! - dtype strings: "float32" vs "f32"

use byteorder::{LittleEndian, ReadBytesExt};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::ZTensorError;
use crate::models::{Component, DType, Encoding, Manifest, Object, ALIGNMENT};
use crate::utils::swap_endianness_in_place;
use crate::reader::Pod;

/// Magic number for v0.1.0 files.
const MAGIC_V01: &[u8; 8] = b"ZTEN0001";

/// v0.1.0 tensor metadata (from CBOR array).
#[derive(Debug, Clone, Deserialize)]
struct LegacyTensorMeta {
    name: String,
    offset: u64,
    size: u64,
    dtype: String,
    shape: Vec<u64>,
    encoding: String,
    #[serde(default)]
    layout: String,
    #[serde(default)]
    checksum: Option<String>,
}

impl LegacyTensorMeta {
    fn to_dtype(&self) -> Result<DType, ZTensorError> {
        match self.dtype.as_str() {
            "float64" => Ok(DType::F64),
            "float32" => Ok(DType::F32),
            "float16" => Ok(DType::F16),
            "bfloat16" => Ok(DType::BF16),
            "int64" => Ok(DType::I64),
            "int32" => Ok(DType::I32),
            "int16" => Ok(DType::I16),
            "int8" => Ok(DType::I8),
            "uint64" => Ok(DType::U64),
            "uint32" => Ok(DType::U32),
            "uint16" => Ok(DType::U16),
            "uint8" => Ok(DType::U8),
            "bool" => Ok(DType::Bool),
            other => Err(ZTensorError::UnsupportedDType(other.to_string())),
        }
    }

    fn to_encoding(&self) -> Encoding {
        match self.encoding.as_str() {
            "zstd" => Encoding::Zstd,
            _ => Encoding::Raw,
        }
    }
}

/// Reader for legacy v0.1.0 zTensor files.
///
/// Converts v0.1.0 format to v1.1 internal representation for unified API.
pub struct LegacyReader<R: Read + Seek> {
    reader: R,
    manifest: Manifest,
}

impl LegacyReader<BufReader<File>> {
    /// Opens a legacy v0.1.0 file from path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::open(path)?;
        Self::new(BufReader::new(file))
    }
}

impl<R: Read + Seek> LegacyReader<R> {
    /// Creates a new legacy reader.
    ///
    /// v0.1.0 parsing:
    /// 1. Read magic "ZTEN0001"
    /// 2. Read last 8 bytes for CBOR array size
    /// 3. Seek to (EOF - 8 - size) for CBOR array
    /// 4. Decode array of tensor metadata
    pub fn new(mut reader: R) -> Result<Self, ZTensorError> {
        // Verify magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if magic != *MAGIC_V01 {
            return Err(ZTensorError::InvalidMagicNumber { found: magic.to_vec() });
        }

        // Read CBOR size from last 8 bytes
        reader.seek(SeekFrom::End(-8))?;
        let cbor_size = reader.read_u64::<LittleEndian>()?;

        // Read CBOR array
        reader.seek(SeekFrom::End(-8 - cbor_size as i64))?;
        let mut cbor_buf = vec![0u8; cbor_size as usize];
        reader.read_exact(&mut cbor_buf)?;

        // Decode as array of tensor metadata
        let tensors: Vec<LegacyTensorMeta> = serde_cbor::from_slice(&cbor_buf)
            .map_err(ZTensorError::CborDeserialize)?;

        // Convert to v1.1 Manifest format
        let mut objects = BTreeMap::new();
        for t in tensors {
            // Only support dense layout
            if !t.layout.is_empty() && t.layout != "dense" {
                continue; // Skip non-dense
            }

            let dtype = t.to_dtype()?;
            let encoding = t.to_encoding();

            let component = Component {
                dtype,
                offset: t.offset,
                length: t.size,
                encoding,
                digest: t.checksum.clone(),
            };

            let mut components = BTreeMap::new();
            components.insert("data".to_string(), component);

            let obj = Object {
                shape: t.shape,
                format: "dense".to_string(),
                attributes: None,
                components,
            };

            objects.insert(t.name, obj);
        }

        let manifest = Manifest {
            version: "0.1.0".to_string(),
            attributes: None,
            objects,
        };

        Ok(Self { reader, manifest })
    }

    /// Lists all objects (dense tensors only).
    pub fn list_objects(&self) -> &BTreeMap<String, Object> {
        &self.manifest.objects
    }

    /// Gets object metadata by name.
    pub fn get_object(&self, name: &str) -> Option<&Object> {
        self.manifest.objects.get(name)
    }

    /// Reads raw byte data of a dense object.
    pub fn read_object(&mut self, name: &str, verify_checksum: bool) -> Result<Vec<u8>, ZTensorError> {
        let obj = self.manifest.objects.get(name)
            .ok_or_else(|| ZTensorError::ObjectNotFound(name.to_string()))?
            .clone();

        let component = obj.components.get("data")
            .ok_or_else(|| ZTensorError::InvalidFileStructure("Missing data component".to_string()))?;

        if component.offset % ALIGNMENT != 0 {
            return Err(ZTensorError::InvalidAlignment {
                offset: component.offset,
                alignment: ALIGNMENT,
            });
        }

        self.reader.seek(SeekFrom::Start(component.offset))?;

        let mut data = match component.encoding {
            Encoding::Zstd => {
                let mut compressed = vec![0u8; component.length as usize];
                self.reader.read_exact(&mut compressed)?;

                if verify_checksum {
                    if let Some(ref digest) = component.digest {
                        verify_checksum_impl(digest, &compressed, name)?;
                    }
                }

                let mut decompressed = Vec::new();
                zstd::stream::copy_decode(std::io::Cursor::new(compressed), &mut decompressed)
                    .map_err(ZTensorError::ZstdDecompression)?;
                decompressed
            }
            Encoding::Raw => {
                let num_elements = obj.num_elements();
                let expected_size = num_elements * component.dtype.byte_size() as u64;
                let mut buf = vec![0u8; expected_size as usize];
                self.reader.read_exact(&mut buf)?;

                if verify_checksum {
                    if let Some(ref digest) = component.digest {
                        verify_checksum_impl(digest, &buf, name)?;
                    }
                }
                buf
            }
        };

        // Handle endianness
        if cfg!(target_endian = "big") && component.dtype.is_multi_byte() {
            swap_endianness_in_place(&mut data, component.dtype.byte_size());
        }

        Ok(data)
    }

    /// Reads object as typed vector.
    pub fn read_object_as<T: Pod>(&mut self, name: &str) -> Result<Vec<T>, ZTensorError> {
        let obj = self.manifest.objects.get(name)
            .ok_or_else(|| ZTensorError::ObjectNotFound(name.to_string()))?;

        let component = obj.components.get("data")
            .ok_or_else(|| ZTensorError::InvalidFileStructure("Missing data".to_string()))?;

        if !T::dtype_matches(&component.dtype) {
            return Err(ZTensorError::TypeMismatch {
                expected: component.dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }

        let bytes = self.read_object(name, true)?;
        let num_elements = bytes.len() / T::SIZE;
        let mut result = vec![T::default(); num_elements];

        unsafe {
            std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, bytes.len())
                .copy_from_slice(&bytes);
        }

        Ok(result)
    }
}

fn verify_checksum_impl(digest: &str, data: &[u8], name: &str) -> Result<(), ZTensorError> {
    if digest.starts_with("crc32c:0x") || digest.starts_with("crc32c:0X") {
        let expected_hex = &digest[9..];
        let expected = u32::from_str_radix(expected_hex, 16).map_err(|_| {
            ZTensorError::ChecksumFormatError(format!("Invalid CRC32C: {}", expected_hex))
        })?;
        let calculated = crc32c::crc32c(data);
        if calculated != expected {
            return Err(ZTensorError::ChecksumMismatch {
                object_name: name.to_string(),
                component_name: "data".to_string(),
                expected: format!("0x{:08X}", expected),
                calculated: format!("0x{:08X}", calculated),
            });
        }
    } else if digest.starts_with("sha256:") {
        let expected_hex = &digest[7..];
        let calculated = crate::utils::sha256_hex(data);
        if calculated != expected_hex.to_lowercase() {
            return Err(ZTensorError::ChecksumMismatch {
                object_name: name.to_string(),
                component_name: "data".to_string(),
                expected: expected_hex.to_string(),
                calculated,
            });
        }
    }
    Ok(())
}

/// Checks if a file is v0.1.0 format by reading magic.
pub fn is_legacy_format<R: Read + Seek>(reader: &mut R) -> Result<bool, ZTensorError> {
    let pos = reader.stream_position()?;
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    reader.seek(SeekFrom::Start(pos))?;
    Ok(magic == *MAGIC_V01)
}

/// Checks if a file at path is v0.1.0 format.
pub fn is_legacy_file(path: impl AsRef<Path>) -> Result<bool, ZTensorError> {
    let mut file = File::open(path)?;
    is_legacy_format(&mut file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_is_legacy_format() {
        let v01_data = b"ZTEN0001";
        let v11_data = b"ZTEN1000";

        let mut cursor = Cursor::new(v01_data.to_vec());
        assert!(is_legacy_format(&mut cursor).unwrap());

        let mut cursor = Cursor::new(v11_data.to_vec());
        assert!(!is_legacy_format(&mut cursor).unwrap());
    }
}
