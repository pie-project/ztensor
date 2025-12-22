//! Error types for zTensor operations.

use std::fmt;

/// All errors that can occur when working with zTensor files.
#[derive(Debug)]
pub enum ZTensorError {
    /// I/O error from underlying reader/writer.
    Io(std::io::Error),
    /// CBOR serialization failed.
    CborSerialize(serde_cbor::Error),
    /// CBOR deserialization failed.
    CborDeserialize(serde_cbor::Error),
    /// Zstd compression failed.
    ZstdCompression(std::io::Error),
    /// Zstd decompression failed.
    ZstdDecompression(std::io::Error),
    /// Invalid magic number in file header or footer.
    InvalidMagicNumber { found: Vec<u8> },
    /// Component offset not aligned to 64 bytes.
    InvalidAlignment { offset: u64, alignment: u64 },
    /// Requested object not found in manifest.
    ObjectNotFound(String),
    /// Unsupported data type string.
    UnsupportedDType(String),
    /// Unsupported encoding string.
    UnsupportedEncoding(String),
    /// File structure is invalid.
    InvalidFileStructure(String),
    /// Data conversion failed.
    DataConversionError(String),
    /// Checksum verification failed.
    ChecksumMismatch {
        object_name: String,
        component_name: String,
        expected: String,
        calculated: String,
    },
    /// Checksum string format is invalid.
    ChecksumFormatError(String),
    /// Unexpected end of file.
    UnexpectedEof,
    /// Data size doesn't match expected size.
    InconsistentDataSize { expected: u64, found: u64 },
    /// Type mismatch when reading typed data.
    TypeMismatch {
        expected: String,
        found: String,
        context: String,
    },
    /// Manifest exceeds maximum allowed size (1GB).
    ManifestTooLarge { size: u64 },
    /// Object exceeds maximum allowed size.
    ObjectTooLarge { size: u64, limit: u64 },
    /// Other unspecified error.
    Other(String),
}

impl fmt::Display for ZTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {}", err),
            Self::CborSerialize(err) => write!(f, "CBOR serialization error: {}", err),
            Self::CborDeserialize(err) => write!(f, "CBOR deserialization error: {}", err),
            Self::ZstdCompression(err) => write!(f, "Zstd compression error: {}", err),
            Self::ZstdDecompression(err) => write!(f, "Zstd decompression error: {}", err),
            Self::InvalidMagicNumber { found } => write!(
                f,
                "Invalid magic number. Expected 'ZTEN1000', found {:?}",
                String::from_utf8_lossy(found)
            ),
            Self::InvalidAlignment { offset, alignment } => {
                write!(f, "Offset {} is not aligned to {} bytes", offset, alignment)
            }
            Self::ObjectNotFound(name) => write!(f, "Object not found: {}", name),
            Self::UnsupportedDType(dtype) => write!(f, "Unsupported dtype: {}", dtype),
            Self::UnsupportedEncoding(enc) => write!(f, "Unsupported encoding: {}", enc),
            Self::InvalidFileStructure(msg) => write!(f, "Invalid file structure: {}", msg),
            Self::DataConversionError(msg) => write!(f, "Data conversion error: {}", msg),
            Self::ChecksumMismatch {
                object_name,
                component_name,
                expected,
                calculated,
            } => write!(
                f,
                "Checksum mismatch for '{}/{}'. Expected: {}, Got: {}",
                object_name, component_name, expected, calculated
            ),
            Self::ChecksumFormatError(msg) => write!(f, "Checksum format error: {}", msg),
            Self::UnexpectedEof => write!(f, "Unexpected end of file"),
            Self::InconsistentDataSize { expected, found } => {
                write!(f, "Expected {} bytes, found {} bytes", expected, found)
            }
            Self::TypeMismatch {
                expected,
                found,
                context,
            } => write!(
                f,
                "Type mismatch in {}: expected '{}', found '{}'",
                context, expected, found
            ),
            Self::ManifestTooLarge { size } => {
                write!(f, "Manifest size {} exceeds 1GB limit", size)
            }
            Self::ObjectTooLarge { size, limit } => {
                write!(
                    f,
                    "Object size {} exceeds limit of {} (32GB)",
                    size, limit
                )
            }
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for ZTensorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::CborSerialize(err) => Some(err),
            Self::CborDeserialize(err) => Some(err),
            Self::ZstdCompression(err) => Some(err),
            Self::ZstdDecompression(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ZTensorError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}
