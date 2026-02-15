//! Error types for zTensor operations.

use thiserror::Error;

/// All errors that can occur when working with zTensor files.
#[derive(Debug, Error)]
pub enum Error {
    /// I/O error from underlying reader/writer.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// CBOR serialization failed.
    #[error("CBOR serialization error: {0}")]
    CborSerialize(ciborium::ser::Error<std::io::Error>),

    /// CBOR deserialization failed.
    #[error("CBOR deserialization error: {0}")]
    CborDeserialize(ciborium::de::Error<std::io::Error>),

    /// Zstd compression failed.
    #[error("Zstd compression error: {0}")]
    ZstdCompression(std::io::Error),

    /// Zstd decompression failed.
    #[error("Zstd decompression error: {0}")]
    ZstdDecompression(std::io::Error),

    /// Invalid magic number in file header or footer.
    #[error("Invalid magic number. Expected 'ZTEN1000', found {found:?}")]
    InvalidMagicNumber { found: Vec<u8> },

    /// Component offset not aligned to 64 bytes.
    #[error("Offset {offset} is not aligned to {alignment} bytes")]
    InvalidAlignment { offset: u64, alignment: u64 },

    /// Requested object not found in manifest.
    #[error("Object not found: {0}")]
    ObjectNotFound(String),

    /// Unsupported data type string.
    #[error("Unsupported dtype: {0}")]
    UnsupportedDType(String),

    /// Unsupported encoding string.
    #[error("Unsupported encoding: {0}")]
    UnsupportedEncoding(String),

    /// File structure is invalid.
    #[error("Invalid file structure: {0}")]
    InvalidFileStructure(String),

    /// Data conversion failed.
    #[error("Data conversion error: {0}")]
    DataConversionError(String),

    /// Checksum verification failed.
    #[error("Checksum mismatch for '{object_name}/{component_name}'. Expected: {expected}, Got: {calculated}")]
    ChecksumMismatch {
        object_name: String,
        component_name: String,
        expected: String,
        calculated: String,
    },

    /// Checksum string format is invalid.
    #[error("Checksum format error: {0}")]
    ChecksumFormatError(String),

    /// Unexpected end of file.
    #[error("Unexpected end of file")]
    UnexpectedEof,

    /// Data size doesn't match expected size.
    #[error("Expected {expected} bytes, found {found} bytes")]
    InconsistentDataSize { expected: u64, found: u64 },

    /// Type mismatch when reading typed data.
    #[error("Type mismatch in {context}: expected '{expected}', found '{found}'")]
    TypeMismatch {
        expected: String,
        found: String,
        context: String,
    },

    /// Manifest exceeds maximum allowed size (1GB).
    #[error("Manifest size {size} exceeds 1GB limit")]
    ManifestTooLarge { size: u64 },

    /// Object exceeds maximum allowed size.
    #[error("Object size {size} exceeds limit of {limit} (32GB)")]
    ObjectTooLarge { size: u64, limit: u64 },

    /// Other unspecified error.
    #[error("{0}")]
    Other(String),
}
