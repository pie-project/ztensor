use crate::error::ZTensorError;
use serde::{Deserialize, Serialize};
use serde_cbor::Value as CborValue;
use std::collections::BTreeMap;

pub const MAGIC_NUMBER: &[u8; 8] = b"ZTEN1000";
pub const ALIGNMENT: u64 = 64;

/// Data types supported by zTensor 1.0
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // Use lowercase for serialization as per spec examples
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
    pub fn byte_size(&self) -> Result<usize, ZTensorError> {
        match self {
            DType::Float64 | DType::Int64 | DType::Uint64 => Ok(8),
            DType::Float32 | DType::Int32 | DType::Uint32 => Ok(4),
            DType::Float16 | DType::BFloat16 | DType::Int16 | DType::Uint16 => Ok(2),
            DType::Int8 | DType::Uint8 | DType::Bool => Ok(1),
        }
    }

    /// Checks if the data type is multi-byte.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Encoding {
    Raw,
    Zstd,
}

/// Describes a physical byte range in the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    pub offset: u64,
    pub length: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding: Option<Encoding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
}

/// Logical Tensor definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub format: String, // e.g., "dense", "sparse_csr", "sparse_coo"
    pub components: BTreeMap<String, Component>,
}

impl Tensor {
    /// Calculates the expected number of elements from the shape.
    pub fn num_elements(&self) -> u64 {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }
}

/// The Manifest (Metadata) located at the end of the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: String,
    pub generator: String,
    pub attributes: BTreeMap<String, String>,
    pub tensors: BTreeMap<String, Tensor>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            generator: "zTensor-Rust v0.1.0".to_string(),
            attributes: BTreeMap::new(),
            tensors: BTreeMap::new(),
        }
    }
}

/// Data encoding types for Layout is replaced by String "format" in v1.0.
/// Kept here if we need an enum for internal usage, but the spec says "format": string.
/// We will rely on string matching as per spec.

/// Specifies the checksum algorithm to be used by the writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumAlgorithm {
    None,
    Crc32c,
}

// In-memory representations for Sparse Tensors (helper structs for reading)
// These remain generally useful.

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
    pub indices: Vec<u64>, // Column indices
    pub values: Vec<T>,
}

pub use crate::models::Encoding as ComponentEncoding; // Alias if needed
