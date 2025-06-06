use crate::error::ZTensorError;
use serde::{Deserialize, Serialize};
use serde_cbor::Value as CborValue;
use std::collections::BTreeMap; // Added for DType methods

pub const MAGIC_NUMBER: &[u8; 8] = b"ZTEN0001";
pub const ALIGNMENT: u64 = 64;

/// Data types supported by zTensor 1.0
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Float64,
    Float32,
    Float16,
    BFloat16,
    Int64,
    Int32,
    Int16,
    Int8,
    Uint64,
    Uint32,
    Uint16,
    Uint8,
    Bool,
}

impl DType {
    /// Returns the size of a single element of this data type in bytes.
    /// Returns an error for DType::Unknown.
    pub fn byte_size(&self) -> Result<usize, ZTensorError> {
        match self {
            DType::Float64 | DType::Int64 | DType::Uint64 => Ok(8),
            DType::Float32 | DType::Int32 | DType::Uint32 => Ok(4),
            DType::Float16 | DType::BFloat16 | DType::Int16 | DType::Uint16 => Ok(2),
            DType::Int8 | DType::Uint8 | DType::Bool => Ok(1),
        }
    }

    /// Checks if the data type is multi-byte.
    /// Returns false for DType::Unknown or if byte_size results in an error.
    pub fn is_multi_byte(&self) -> bool {
        self.byte_size().map_or(false, |size| size > 1)
    }

    /// Returns the string representation of the DType.
    pub fn to_string_key(&self) -> String {
        match self {
            DType::Float64 => "float64".to_string(),
            DType::Float32 => "float32".to_string(),
            DType::Float16 => "float16".to_string(),
            DType::BFloat16 => "bfloat16".to_string(),
            DType::Int64 => "int64".to_string(),
            DType::Int32 => "int32".to_string(),
            DType::Int16 => "int16".to_string(),
            DType::Int8 => "int8".to_string(),
            DType::Uint64 => "uint64".to_string(),
            DType::Uint32 => "uint32".to_string(),
            DType::Uint16 => "uint16".to_string(),
            DType::Uint8 => "uint8".to_string(),
            DType::Bool => "bool".to_string(),
        }
    }
}


/// Tensor data encoding types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    Dense,
    SparseCoo,
    SparseCsr,
}


/// Tensor data encoding types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Encoding {
    Raw,
    Zstd,

}

/// Endianness of multi-byte tensor data when `encoding` is `raw`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataEndianness {
    Little,
    Big,
}

impl Default for DataEndianness {
    fn default() -> Self {
        DataEndianness::Little
    }
}

/// Metadata for a single tensor within a zTensor file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub offset: u64,
    pub size: u64, // On-disk size
    pub dtype: DType,
    pub layout: Layout,
    pub shape: Vec<u64>,
    pub encoding: Encoding,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_endianness: Option<DataEndianness>,
    /// Checksum string, e.g., "crc32c:0x1234ABCD".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
    #[serde(flatten)]
    pub custom_fields: BTreeMap<String, CborValue>,
}

impl TensorMetadata {
    /// Calculates the expected number of elements from the shape.
    /// A scalar (empty shape) has 1 element.
    pub fn num_elements(&self) -> u64 {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Calculates the expected size of the raw, uncompressed tensor data in bytes.
    /// Returns an error if the DType is Unknown.
    pub fn uncompressed_data_size(&self) -> Result<u64, ZTensorError> {
        Ok(self.num_elements() * (self.dtype.byte_size()? as u64))
    }
}

/// Specifies the checksum algorithm to be used by the writer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChecksumAlgorithm {
    None,
    Crc32c,
}

#[derive(Debug, Clone)]
pub struct CooTensor<T> {
    pub shape: Vec<u64>,
    pub indices: Vec<Vec<u64>>, // indices[i][j]: j-th index of i-th nonzero
    pub values: Vec<T>,
}

#[derive(Debug, Clone)]
pub struct CsrTensor<T> {
    pub shape: Vec<u64>,
    pub indptr: Vec<u64>,
    pub indices: Vec<u64>,
    pub values: Vec<T>,
}
