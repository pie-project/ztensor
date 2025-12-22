//! Data models for zTensor v1.1 format.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Magic number for zTensor files (header and footer).
pub const MAGIC: &[u8; 8] = b"ZTEN1000";

/// Required alignment for component data (64 bytes for AVX-512).
pub const ALIGNMENT: u64 = 64;

/// Maximum manifest size (1GB) to prevent DoS attacks.
pub const MAX_MANIFEST_SIZE: u64 = 1_073_741_824;

/// Supported data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F64,
    F32,
    F16,
    #[serde(rename = "bf16")]
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}

impl DType {
    /// Returns the size of one element in bytes.
    pub fn byte_size(&self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    /// Returns true if the type is multi-byte (needs endianness handling).
    pub fn is_multi_byte(&self) -> bool {
        self.byte_size() > 1
    }

    /// Returns the string key for this dtype.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::U64 => "u64",
            Self::U32 => "u32",
            Self::U16 => "u16",
            Self::U8 => "u8",
            Self::Bool => "bool",
        }
    }
}

/// Data encoding for components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Encoding {
    #[default]
    Raw,
    Zstd,
}

/// Physical storage location and metadata for a data blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    /// Data type of this component.
    pub dtype: DType,
    /// Absolute file offset (must be 64-byte aligned).
    pub offset: u64,
    /// Size of stored data in bytes.
    pub length: u64,
    /// Encoding method (default: raw).
    #[serde(default, skip_serializing_if = "is_default_encoding")]
    pub encoding: Encoding,
    /// Optional checksum (e.g., "sha256:..." or "crc32c:0x...").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
}

fn is_default_encoding(enc: &Encoding) -> bool {
    *enc == Encoding::Raw
}

/// Logical object (tensor) definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Object {
    /// Logical dimensions (e.g., [1024, 768]).
    pub shape: Vec<u64>,
    /// Layout schema (dense, sparse_csr, sparse_coo, quantized_group, etc.).
    pub format: String,
    /// Object-level attributes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<BTreeMap<String, serde_cbor::Value>>,
    /// Mapping of roles to component definitions.
    pub components: BTreeMap<String, Component>,
}

impl Object {
    /// Calculates the number of elements from the shape.
    pub fn num_elements(&self) -> u64 {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }
}

/// Root manifest structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// zTensor spec version (e.g., "1.1.0").
    pub version: String,
    /// Global metadata key-value pairs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<BTreeMap<String, String>>,
    /// Map of object names to definitions.
    pub objects: BTreeMap<String, Object>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: "1.1.0".to_string(),
            attributes: None,
            objects: BTreeMap::new(),
        }
    }
}

/// Checksum algorithm for writing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChecksumAlgorithm {
    #[default]
    None,
    Crc32c,
    Sha256,
}

/// In-memory COO sparse tensor.
#[derive(Debug, Clone)]
pub struct CooTensor<T> {
    pub shape: Vec<u64>,
    /// indices[i][j]: j-th dimension index of i-th nonzero element
    pub indices: Vec<Vec<u64>>,
    pub values: Vec<T>,
}

/// In-memory CSR sparse tensor.
#[derive(Debug, Clone)]
pub struct CsrTensor<T> {
    pub shape: Vec<u64>,
    pub indptr: Vec<u64>,
    pub indices: Vec<u64>,
    pub values: Vec<T>,
}
