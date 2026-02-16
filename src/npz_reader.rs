//! NumPy NPZ format reader.
//!
//! Provides read-only access to `.npz` files (ZIP archives of `.npy` arrays)
//! through the unified `TensorReader` API. Uncompressed (STORED) entries use
//! memory mapping for zero-copy access; compressed entries are read into memory.
//!
//! Requires the `npz` feature.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::error::Error;
use crate::models::{DType, Manifest, Object};
use crate::reader::{TensorData, TensorElement, TensorReader};

// ---- NPY format constants ----

const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

// ---- NPY header parsing ----

/// Parsed .npy array header.
#[doc(hidden)]
pub struct NpyHeader {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub fortran_order: bool,
    /// Byte offset where raw data starts (relative to start of .npy data).
    pub data_offset: usize,
}

/// Parse the numpy dtype descriptor string to a DType.
fn parse_npy_descr(descr: &str) -> Result<DType, Error> {
    match descr {
        "<f8" | "=f8" | ">f8" | "float64" => Ok(DType::F64),
        "<f4" | "=f4" | ">f4" | "float32" => Ok(DType::F32),
        "<f2" | "=f2" | ">f2" | "float16" => Ok(DType::F16),
        "<i8" | "=i8" | ">i8" | "int64" => Ok(DType::I64),
        "<i4" | "=i4" | ">i4" | "int32" => Ok(DType::I32),
        "<i2" | "=i2" | ">i2" | "int16" => Ok(DType::I16),
        "|i1" | "int8" => Ok(DType::I8),
        "<u8" | "=u8" | ">u8" | "uint64" => Ok(DType::U64),
        "<u4" | "=u4" | ">u4" | "uint32" => Ok(DType::U32),
        "<u2" | "=u2" | ">u2" | "uint16" => Ok(DType::U16),
        "|u1" | "uint8" => Ok(DType::U8),
        "|b1" | "bool" => Ok(DType::Bool),
        _ => Err(Error::UnsupportedDType(format!(
            "Unsupported numpy dtype: '{}'",
            descr
        ))),
    }
}

/// Extract a quoted string value after a key in the header dict.
/// e.g., from "'descr': '<f4'" extracts "<f4".
fn extract_string_value(header: &str, key: &str) -> Option<String> {
    let key_pos = header.find(key)?;
    let after_key = &header[key_pos + key.len()..];
    // Skip whitespace and colon
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let trimmed = after_colon.trim_start();
    // Extract quoted string (single or double quotes)
    let quote = trimmed.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let inner = &trimmed[1..];
    let end = inner.find(quote)?;
    Some(inner[..end].to_string())
}

/// Extract the shape tuple from the header dict.
/// e.g., from "'shape': (3, 4)" extracts [3, 4].
fn extract_shape(header: &str) -> Result<Vec<u64>, Error> {
    let key = "'shape'";
    let key_pos = header.find(key).ok_or_else(|| {
        Error::InvalidFileStructure("Missing 'shape' in .npy header".to_string())
    })?;
    let after_key = &header[key_pos + key.len()..];
    let after_colon = after_key
        .trim_start()
        .strip_prefix(':')
        .ok_or_else(|| {
            Error::InvalidFileStructure("Malformed shape in .npy header".to_string())
        })?;
    let trimmed = after_colon.trim_start();

    let paren_start = trimmed.find('(').ok_or_else(|| {
        Error::InvalidFileStructure("Missing '(' in shape tuple".to_string())
    })?;
    let paren_end = trimmed.find(')').ok_or_else(|| {
        Error::InvalidFileStructure("Missing ')' in shape tuple".to_string())
    })?;

    let inner = &trimmed[paren_start + 1..paren_end];
    if inner.trim().is_empty() {
        return Ok(vec![]);
    }

    let mut dims = Vec::new();
    for part in inner.split(',') {
        let s = part.trim();
        if s.is_empty() {
            continue; // trailing comma in single-element tuple: (3,)
        }
        let dim: u64 = s.parse().map_err(|_| {
            Error::InvalidFileStructure(format!("Invalid shape dimension: '{}'", s))
        })?;
        dims.push(dim);
    }
    Ok(dims)
}

/// Extract fortran_order boolean from the header dict.
fn extract_fortran_order(header: &str) -> bool {
    if let Some(pos) = header.find("'fortran_order'") {
        let after = &header[pos..];
        after.contains("True")
    } else {
        false
    }
}

/// Parse an .npy header from raw bytes.
#[doc(hidden)]
pub fn parse_npy_header(data: &[u8]) -> Result<NpyHeader, Error> {
    if data.len() < 10 {
        return Err(Error::InvalidFileStructure(
            "Data too small for .npy header".to_string(),
        ));
    }

    if &data[..6] != NPY_MAGIC {
        return Err(Error::InvalidMagicNumber {
            found: data[..6].to_vec(),
        });
    }

    let major = data[6];
    let _minor = data[7];

    let (header_len, header_start) = if major >= 2 {
        // Version 2+: 4-byte LE header length
        if data.len() < 12 {
            return Err(Error::UnexpectedEof);
        }
        let len = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        (len, 12)
    } else {
        // Version 1: 2-byte LE header length
        let len = u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize;
        (len, 10)
    };

    let header_end = header_start + header_len;
    if header_end > data.len() {
        return Err(Error::UnexpectedEof);
    }

    let header_str = std::str::from_utf8(&data[header_start..header_end]).map_err(|_| {
        Error::InvalidFileStructure("Invalid UTF-8 in .npy header".to_string())
    })?;

    let descr = extract_string_value(header_str, "'descr'").ok_or_else(|| {
        Error::InvalidFileStructure("Missing 'descr' in .npy header".to_string())
    })?;

    let dtype = parse_npy_descr(&descr)?;
    let shape = extract_shape(header_str)?;
    let fortran_order = extract_fortran_order(header_str);

    Ok(NpyHeader {
        dtype,
        shape,
        fortran_order,
        data_offset: header_end,
    })
}

// ---- NpzReader ----

/// Data location for an NPZ entry.
enum NpzDataLocation {
    /// Uncompressed: zero-copy from mmap.
    MmapRange { offset: usize, length: usize },
    /// Compressed: owned bytes read into memory.
    Owned(Vec<u8>),
}

/// Reader for NumPy NPZ (.npz) files.
///
/// Uses memory mapping for zero-copy access to uncompressed entries.
/// Compressed entries are read into memory on open.
pub struct NpzReader {
    mmap: Mmap,
    pub manifest: Manifest,
    data_locations: BTreeMap<String, NpzDataLocation>,
}

impl NpzReader {
    /// Opens an NPZ file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Open as ZIP to enumerate entries
        let file2 = File::open(path)?;
        let mut archive = zip::ZipArchive::new(file2).map_err(|e| {
            Error::InvalidFileStructure(format!("Not a valid ZIP/NPZ file: {}", e))
        })?;

        let mut objects = BTreeMap::new();
        let mut data_locations = BTreeMap::new();

        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).map_err(|e| {
                Error::InvalidFileStructure(format!("Cannot read ZIP entry {}: {}", i, e))
            })?;

            let entry_name = entry.name().to_string();

            // Only process .npy files
            if !entry_name.ends_with(".npy") {
                continue;
            }

            // Tensor name = filename without .npy extension
            let tensor_name = entry_name.strip_suffix(".npy").unwrap().to_string();

            let is_stored = entry.compression() == zip::CompressionMethod::Stored;

            if is_stored {
                // Zero-copy path: parse header from mmap to find data offset
                let entry_offset = entry.data_start() as usize;
                let entry_size = entry.size() as usize;

                if entry_offset + entry_size > mmap.len() {
                    return Err(Error::InvalidFileStructure(format!(
                        "NPZ entry '{}' extends beyond file",
                        tensor_name
                    )));
                }

                let npy_data = &mmap[entry_offset..entry_offset + entry_size];
                let header = parse_npy_header(npy_data)?;

                if header.fortran_order {
                    return Err(Error::Other(format!(
                        "Fortran-order arrays not supported: '{}'",
                        tensor_name
                    )));
                }

                let data_start = entry_offset + header.data_offset;
                let data_len = entry_size - header.data_offset;

                let obj = Object::dense(header.shape, header.dtype, data_start as u64, data_len as u64);

                data_locations.insert(
                    tensor_name.clone(),
                    NpzDataLocation::MmapRange {
                        offset: data_start,
                        length: data_len,
                    },
                );
                objects.insert(tensor_name, obj);
            } else {
                // Compressed path: read entire entry into memory
                let mut npy_bytes = Vec::new();
                entry.read_to_end(&mut npy_bytes).map_err(|e| {
                    Error::Io(e)
                })?;

                let header = parse_npy_header(&npy_bytes)?;

                if header.fortran_order {
                    return Err(Error::Other(format!(
                        "Fortran-order arrays not supported: '{}'",
                        tensor_name
                    )));
                }

                let raw_data = npy_bytes[header.data_offset..].to_vec();
                let data_len = raw_data.len();

                let obj = Object::dense(header.shape, header.dtype, 0, data_len as u64);

                data_locations.insert(tensor_name.clone(), NpzDataLocation::Owned(raw_data));
                objects.insert(tensor_name, obj);
            }
        }

        let manifest = Manifest {
            version: "npz".to_string(),
            attributes: None,
            objects,
        };

        Ok(Self {
            mmap,
            manifest,
            data_locations,
        })
    }

    /// Gets a zero-copy reference to a tensor's raw data.
    pub fn view(&self, name: &str) -> Result<&[u8], Error> {
        match self.data_locations.get(name) {
            Some(NpzDataLocation::MmapRange { offset, length }) => {
                Ok(&self.mmap[*offset..*offset + *length])
            }
            Some(NpzDataLocation::Owned(data)) => Ok(data.as_slice()),
            None => Err(Error::ObjectNotFound(name.to_string())),
        }
    }

    /// Gets a typed zero-copy reference to a tensor's data.
    pub fn view_as<T: TensorElement>(&self, name: &str) -> Result<&[T], Error> {
        let dtype = self.manifest.objects.get(name)
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

    /// Reads tensor data as a copy.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, Error> {
        self.view(name).map(|s| s.to_vec())
    }

    /// Reads tensor data as a typed vector.
    pub fn read_as<T: TensorElement>(&self, name: &str) -> Result<Vec<T>, Error> {
        let dtype = self.manifest.objects.get(name)
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

impl TensorReader for NpzReader {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read_data(&self, name: &str) -> Result<TensorData<'_>, Error> {
        match self.data_locations.get(name) {
            Some(NpzDataLocation::MmapRange { offset, length }) => {
                Ok(TensorData::Borrowed(&self.mmap[*offset..*offset + *length]))
            }
            Some(NpzDataLocation::Owned(data)) => Ok(TensorData::Borrowed(data.as_slice())),
            None => Err(Error::ObjectNotFound(name.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_npy_descr() {
        assert_eq!(parse_npy_descr("<f4").unwrap(), DType::F32);
        assert_eq!(parse_npy_descr("<f8").unwrap(), DType::F64);
        assert_eq!(parse_npy_descr("<i4").unwrap(), DType::I32);
        assert_eq!(parse_npy_descr("|u1").unwrap(), DType::U8);
        assert_eq!(parse_npy_descr("|b1").unwrap(), DType::Bool);
        assert!(parse_npy_descr("object").is_err());
    }

    #[test]
    fn test_extract_shape() {
        assert_eq!(
            extract_shape("{'shape': (3, 4), 'other': 1}").unwrap(),
            vec![3, 4]
        );
        assert_eq!(
            extract_shape("{'shape': (10,), 'other': 1}").unwrap(),
            vec![10]
        );
        assert_eq!(
            extract_shape("{'shape': (), 'other': 1}").unwrap(),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn test_extract_string_value() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }";
        assert_eq!(
            extract_string_value(header, "'descr'").unwrap(),
            "<f4"
        );
    }

    #[test]
    fn test_extract_fortran_order() {
        assert!(!extract_fortran_order(
            "{'fortran_order': False, 'shape': (3,)}"
        ));
        assert!(extract_fortran_order(
            "{'fortran_order': True, 'shape': (3,)}"
        ));
    }
}
