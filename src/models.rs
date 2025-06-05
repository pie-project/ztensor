use crate::error::ZTensorError;
use serde::{Deserialize, Serialize};
use serde_cbor::Value as CborValue;
use std::collections::BTreeMap;

pub const MAGIC_NUMBER: &[u8; 8] = b"ZTEN0001";
pub const ALIGNMENT: u64 = 64;

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
    pub fn byte_size(&self) -> usize {
        match self {
            DType::Float64 | DType::Int64 | DType::Uint64 => 8,
            DType::Float32 | DType::Int32 | DType::Uint32 => 4,
            DType::Float16 | DType::BFloat16 | DType::Int16 | DType::Uint16 => 2,
            DType::Int8 | DType::Uint8 | DType::Bool => 1,
        }
    }

    pub fn is_multi_byte(&self) -> bool {
        self.byte_size() > 1
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Encoding {
    Raw,
    Zstd,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub offset: u64,
    pub size: u64, // On-disk size
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub encoding: Encoding,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_endianness: Option<DataEndianness>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
    #[serde(flatten)]
    pub custom_fields: BTreeMap<String, CborValue>,
}

impl TensorMetadata {
    /// Calculates the expected number of elements from the shape.
    pub fn num_elements(&self) -> u64 {
        if self.shape.is_empty() {
            // Scalar
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Calculates the expected size of the raw, uncompressed tensor data in bytes.
    pub fn uncompressed_data_size(&self) -> u64 {
        self.num_elements() * (self.dtype.byte_size() as u64)
    }
}
