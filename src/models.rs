//! Data models for zTensor v1.2 format.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;

/// Magic number for zTensor files (header and footer).
pub const MAGIC: &[u8; 8] = b"ZTEN1000";

/// Magic number for legacy v0.1.0 files.
pub(crate) const MAGIC_V01: &[u8; 8] = b"ZTEN0001";

/// Required alignment for component data (64 bytes for AVX-512).
pub const ALIGNMENT: u64 = 64;

/// Maximum manifest size (1GB) to prevent DoS attacks.
pub const MAX_MANIFEST_SIZE: u64 = 1_073_741_824;

/// Supported data types for tensor elements.
///
/// Each variant corresponds to a fixed-size numeric or boolean type.
/// The serialized name matches the lowercase variant (e.g., `F32` serializes as `"f32"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    /// 64-bit IEEE 754 floating point.
    F64,
    /// 32-bit IEEE 754 floating point.
    F32,
    /// 16-bit IEEE 754 floating point (half precision).
    F16,
    /// 16-bit brain floating point.
    #[serde(rename = "bf16")]
    BF16,
    /// 64-bit signed integer.
    I64,
    /// 32-bit signed integer.
    I32,
    /// 16-bit signed integer.
    I16,
    /// 8-bit signed integer.
    I8,
    /// 64-bit unsigned integer.
    U64,
    /// 32-bit unsigned integer.
    U32,
    /// 16-bit unsigned integer.
    U16,
    /// 8-bit unsigned integer.
    U8,
    /// Boolean (1 byte per element).
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
    /// Uncompressed data stored as-is.
    #[default]
    Raw,
    /// Zstd-compressed data.
    Zstd,
}

/// Layout format for logical objects.
///
/// Determines how an object's components are interpreted. Dense objects have
/// a single `"data"` component; sparse formats have `"values"` plus index components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Format {
    /// Standard dense tensor — one contiguous `"data"` component.
    Dense,
    /// Compressed Sparse Row — `"values"`, `"indices"`, and `"indptr"` components.
    SparseCsr,
    /// Coordinate list — `"values"` and `"coords"` components.
    SparseCoo,
    /// Quantized weight group — `"data"` and `"scales"` components.
    QuantizedGroup,
    /// Extension format not covered by the built-in variants.
    Other(String),
}

impl Format {
    /// Returns the string representation used in the serialized manifest.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Dense => "dense",
            Self::SparseCsr => "sparse_csr",
            Self::SparseCoo => "sparse_coo",
            Self::QuantizedGroup => "quantized_group",
            Self::Other(s) => s,
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "dense" => Self::Dense,
            "sparse_csr" => Self::SparseCsr,
            "sparse_coo" => Self::SparseCoo,
            "quantized_group" => Self::QuantizedGroup,
            other => Self::Other(other.to_string()),
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Format {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Format {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Format::from_str(&s))
    }
}

/// Physical storage location and metadata for a data blob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    /// Storage type — determines byte size and endianness.
    pub dtype: DType,
    /// Logical type (e.g., "f8_e4m3fn", "complex64"). When absent, equals `dtype`.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    /// Absolute file offset (must be 64-byte aligned).
    pub offset: u64,
    /// Size of stored (on-disk) data in bytes. When encoding is zstd, this is the compressed size.
    pub length: u64,
    /// Original data size before compression. Required when encoding is zstd.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncompressed_length: Option<u64>,
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
    pub format: Format,
    /// Object-level attributes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<BTreeMap<String, ciborium::Value>>,
    /// Mapping of roles to component definitions.
    pub components: BTreeMap<String, Component>,
}

impl Object {
    /// Calculates the number of elements from the shape.
    ///
    /// Returns an error if the product of dimensions overflows `u64`.
    pub fn num_elements(&self) -> Result<u64, crate::error::Error> {
        if self.shape.is_empty() {
            Ok(1)
        } else {
            self.shape.iter().try_fold(1u64, |acc, &d| {
                acc.checked_mul(d).ok_or_else(|| {
                    crate::error::Error::InvalidFileStructure("Shape product overflows u64".into())
                })
            })
        }
    }

    /// Returns the dtype of the "data" component.
    pub fn data_dtype(&self) -> Result<DType, crate::error::Error> {
        self.components.get("data").map(|c| c.dtype).ok_or_else(|| {
            crate::error::Error::InvalidFileStructure("Missing 'data' component".to_string())
        })
    }

    /// Creates a dense object with a single raw "data" component.
    pub fn dense(shape: Vec<u64>, dtype: DType, offset: u64, length: u64) -> Self {
        let component = Component {
            dtype,
            r#type: None,
            offset,
            length,
            uncompressed_length: None,
            encoding: Encoding::Raw,
            digest: None,
        };
        let mut components = BTreeMap::new();
        components.insert("data".to_string(), component);
        Self {
            shape,
            format: Format::Dense,
            attributes: None,
            components,
        }
    }
}

/// Root manifest structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// zTensor spec version (e.g., "1.2.0").
    pub version: String,
    /// Global metadata key-value pairs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<BTreeMap<String, ciborium::Value>>,
    /// Map of object names to definitions.
    pub objects: BTreeMap<String, Object>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: "1.2.0".to_string(),
            attributes: None,
            objects: BTreeMap::new(),
        }
    }
}

/// Checksum algorithm for data integrity verification.
///
/// When writing, the checksum is computed and stored in the component's `digest` field.
/// When reading, the checksum is verified automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Checksum {
    /// No checksum.
    #[default]
    None,
    /// CRC-32C (Castagnoli) — fast hardware-accelerated checksum.
    Crc32c,
    /// SHA-256 — cryptographic hash for strong integrity guarantees.
    Sha256,
}

/// In-memory COO (Coordinate list) sparse tensor.
///
/// Stores nonzero elements as a list of values with their coordinates.
/// Returned by [`Reader::read_coo`](crate::Reader::read_coo).
#[derive(Debug, Clone)]
pub struct CooTensor<T> {
    /// Logical dimensions of the full tensor.
    pub shape: Vec<u64>,
    /// Coordinates of nonzero elements. `indices[i][j]` is the j-th dimension
    /// index of the i-th nonzero element.
    pub indices: Vec<Vec<u64>>,
    /// Nonzero values, one per coordinate entry.
    pub values: Vec<T>,
}

/// In-memory CSR (Compressed Sparse Row) sparse tensor.
///
/// Stores nonzero elements in compressed row format. Returned by
/// [`Reader::read_csr`](crate::Reader::read_csr).
#[derive(Debug, Clone)]
pub struct CsrTensor<T> {
    /// Logical dimensions of the full tensor.
    pub shape: Vec<u64>,
    /// Row pointer array. `indptr[i]..indptr[i+1]` gives the range of
    /// nonzero entries in row `i`.
    pub indptr: Vec<u64>,
    /// Column indices for each nonzero element.
    pub indices: Vec<u64>,
    /// Nonzero values, parallel to `indices`.
    pub values: Vec<T>,
}
