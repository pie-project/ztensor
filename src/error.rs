use std::fmt;

#[derive(Debug)]
pub enum ZTensorError {
    Io(std::io::Error),
    CborSerialize(serde_cbor::Error),
    CborDeserialize(serde_cbor::Error),
    ZstdCompression(std::io::Error),
    ZstdDecompression(std::io::Error),
    InvalidMagicNumber {
        found: Vec<u8>,
    },
    InvalidAlignment {
        offset: u64,
        required_alignment: u64,
        actual_offset: u64,
    },
    TensorNotFound(String),
    UnsupportedDType(String),
    UnsupportedEncoding(String),
    InvalidFileStructure(String),
    DataConversionError(String),
    ChecksumMismatch {
        tensor_name: String,
        expected: String,
        calculated: String,
    },
    ChecksumFormatError(String), // New error for checksum parsing
    UnexpectedEof,
    InconsistentDataSize {
        expected: u64,
        found: u64,
    },
    TypeMismatch {
        expected: String,
        found: String,
        context: String,
    }, // For typed data retrieval
    Other(String),
}

impl fmt::Display for ZTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZTensorError::Io(err) => write!(f, "I/O error: {}", err),
            ZTensorError::CborSerialize(err) => write!(f, "CBOR serialization error: {}", err),
            ZTensorError::CborDeserialize(err) => write!(f, "CBOR deserialization error: {}", err),
            ZTensorError::ZstdCompression(err) => write!(f, "Zstd compression error: {}", err),
            ZTensorError::ZstdDecompression(err) => write!(f, "Zstd decompression error: {}", err),
            ZTensorError::InvalidMagicNumber { found } => write!(
                f,
                "Invalid magic number. Expected 'ZTEN0001', found {:?}",
                String::from_utf8_lossy(found)
            ),
            ZTensorError::InvalidAlignment {
                offset,
                required_alignment,
                actual_offset,
            } => write!(
                f,
                "Invalid tensor alignment for offset {}. Must be multiple of {}, but actual start is {}",
                offset, required_alignment, actual_offset
            ),
            ZTensorError::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
            ZTensorError::UnsupportedDType(dtype) => write!(f, "Unsupported DType: {}", dtype),
            ZTensorError::UnsupportedEncoding(encoding) => {
                write!(f, "Unsupported encoding: {}", encoding)
            }
            ZTensorError::InvalidFileStructure(msg) => write!(f, "Invalid file structure: {}", msg),
            ZTensorError::DataConversionError(msg) => write!(f, "Data conversion error: {}", msg),
            ZTensorError::ChecksumMismatch {
                tensor_name,
                expected,
                calculated,
            } => write!(
                f,
                "Checksum mismatch for tensor '{}'. Expected: {}, Calculated: {}",
                tensor_name, expected, calculated
            ),
            ZTensorError::ChecksumFormatError(msg) => write!(f, "Checksum format error: {}", msg),
            ZTensorError::UnexpectedEof => write!(f, "Unexpected end of file"),
            ZTensorError::InconsistentDataSize { expected, found } => write!(
                f,
                "Inconsistent data size. Expected {} bytes, found {} bytes.",
                expected, found
            ),
            ZTensorError::TypeMismatch {
                expected,
                found,
                context,
            } => write!(
                f,
                "Type mismatch for {}. Expected type compatible with zTensor dtype '{}', found incompatible type '{}'",
                context, expected, found
            ),
            ZTensorError::Other(msg) => write!(f, "Other error: {}", msg),
        }
    }
}

// Error trait and From implementations remain largely the same
impl std::error::Error for ZTensorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ZTensorError::Io(err) => Some(err),
            ZTensorError::CborSerialize(err) => Some(err),
            ZTensorError::CborDeserialize(err) => Some(err),
            ZTensorError::ZstdCompression(err) => Some(err),
            ZTensorError::ZstdDecompression(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ZTensorError {
    fn from(err: std::io::Error) -> Self {
        ZTensorError::Io(err)
    }
}

impl From<serde_cbor::Error> for ZTensorError {
    fn from(err: serde_cbor::Error) -> Self {
        ZTensorError::CborDeserialize(err) // Default, context might create CborSerialize
    }
}
