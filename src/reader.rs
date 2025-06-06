use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::ZTensorError;
use crate::models::{ALIGNMENT, DType, Encoding, MAGIC_NUMBER, TensorMetadata};
use crate::utils::{NATIVE_ENDIANNESS, swap_endianness_in_place};

/// Trait for Plain Old Data types that can be safely created from byte sequences.
/// `Copy` is essential for safety if we were to do direct memory transmutation,
/// but here we'll build element by element for better safety and endian handling.
pub trait Pod: Sized + Default + Clone {
    const SIZE: usize = std::mem::size_of::<Self>();
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn from_be_bytes(bytes: &[u8]) -> Self;
    fn dtype_matches(dtype: &DType) -> bool;
}

// Implement Pod for common types
macro_rules! impl_pod {
    ($t:ty, $d:path, $from_le:ident, $from_be:ident) => {
        impl Pod for $t {
            fn from_le_bytes(bytes: &[u8]) -> Self {
                <$t>::$from_le(bytes.try_into().expect("Pod byte slice wrong size"))
            }
            fn from_be_bytes(bytes: &[u8]) -> Self {
                <$t>::$from_be(bytes.try_into().expect("Pod byte slice wrong size"))
            }
            fn dtype_matches(dtype: &DType) -> bool {
                dtype == &$d
            }
        }
    };
}

impl_pod!(f64, DType::Float64, from_le_bytes, from_be_bytes);
impl_pod!(f32, DType::Float32, from_le_bytes, from_be_bytes);
impl_pod!(i64, DType::Int64, from_le_bytes, from_be_bytes);
impl_pod!(i32, DType::Int32, from_le_bytes, from_be_bytes);
impl_pod!(i16, DType::Int16, from_le_bytes, from_be_bytes);
impl_pod!(u64, DType::Uint64, from_le_bytes, from_be_bytes);
impl_pod!(u32, DType::Uint32, from_le_bytes, from_be_bytes);
impl_pod!(u16, DType::Uint16, from_le_bytes, from_be_bytes);

// Simpler Pod impl for u8/i8 (endianness doesn't matter)
macro_rules! impl_pod_byte {
    ($t:ty, $d:path) => {
        impl Pod for $t {
            fn from_le_bytes(bytes: &[u8]) -> Self {
                bytes[0] as $t
            }
            fn from_be_bytes(bytes: &[u8]) -> Self {
                bytes[0] as $t
            }
            fn dtype_matches(dtype: &DType) -> bool {
                dtype == &$d
            }
        }
    };
}
impl_pod_byte!(u8, DType::Uint8);
impl_pod_byte!(i8, DType::Int8);

// Bool needs special handling if it were multi-byte, but spec says 1 byte.
impl Pod for bool {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn from_be_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn dtype_matches(dtype: &DType) -> bool {
        dtype == &DType::Bool
    }
}
// Note: Float16/BFloat16 are not standard Rust types, would need a library and more complex Pod impl.

/// Reads zTensor files.
pub struct ZTensorReader<R: Read + Seek> {
    reader: R,
    pub metadata_list: Vec<TensorMetadata>,
}

impl ZTensorReader<BufReader<File>> {
    /// Opens a zTensor file from the given path and parses its metadata.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::open(path)?;
        Self::new(BufReader::new(file))
    }
}

impl<R: Read + Seek> ZTensorReader<R> {
    /// Creates a new `ZTensorReader` from a `Read + Seek` source and parses metadata.
    pub fn new(mut reader: R) -> Result<Self, ZTensorError> {
        let mut magic_buf = [0u8; 8];
        reader.read_exact(&mut magic_buf)?;
        if magic_buf != *MAGIC_NUMBER {
            return Err(ZTensorError::InvalidMagicNumber {
                found: magic_buf.to_vec(),
            });
        }

        reader.seek(SeekFrom::End(-8))?;
        let cbor_blob_size = reader.read_u64::<LittleEndian>()?;

        let file_size = reader.seek(SeekFrom::End(0))?;
        if cbor_blob_size == 0 && file_size == (MAGIC_NUMBER.len() as u64 + 8) {
            // No CBOR blob, only size field which is 0
            return Ok(Self {
                reader,
                metadata_list: Vec::new(),
            });
        }
        if cbor_blob_size == 1 {
            // Potentially empty array for zero tensors
            if file_size == (MAGIC_NUMBER.len() as u64 + 1 + 8) {
                reader.seek(SeekFrom::Start(MAGIC_NUMBER.len() as u64))?;
                let mut cbor_byte = [0u8; 1];
                reader.read_exact(&mut cbor_byte)?;
                if cbor_byte[0] == 0x80 {
                    // Empty CBOR array
                    return Ok(Self {
                        reader,
                        metadata_list: Vec::new(),
                    });
                }
            }
        }

        if cbor_blob_size > file_size.saturating_sub(MAGIC_NUMBER.len() as u64 + 8) {
            return Err(ZTensorError::InvalidFileStructure(format!(
                "CBOR blob size {} is too large for file size {}",
                cbor_blob_size, file_size
            )));
        }
        if cbor_blob_size == 0 {
            // Valid case: no tensors, CBOR blob size is 0.
            return Ok(Self {
                reader,
                metadata_list: Vec::new(),
            });
        }

        reader.seek(SeekFrom::End(-8 - (cbor_blob_size as i64)))?;
        let mut cbor_buf = vec![0u8; cbor_blob_size as usize];
        reader.read_exact(&mut cbor_buf)?;

        let metadata_list: Vec<TensorMetadata> =
            serde_cbor::from_slice(&cbor_buf).map_err(ZTensorError::CborDeserialize)?;

        for meta in &metadata_list {
            if meta.offset < MAGIC_NUMBER.len() as u64 {
                return Err(ZTensorError::InvalidFileStructure(format!(
                    "Tensor '{}' offset {} is within magic number.",
                    meta.name, meta.offset
                )));
            }
            if meta.offset % ALIGNMENT != 0 {
                return Err(ZTensorError::InvalidAlignment {
                    offset: meta.offset,
                    required_alignment: ALIGNMENT,
                    actual_offset: meta.offset % ALIGNMENT, // Corrected this part
                });
            }
        }
        Ok(Self {
            reader,
            metadata_list,
        })
    }

    /// Lists all tensor metadata entries in the file.
    pub fn list_tensors(&self) -> &Vec<TensorMetadata> {
        &self.metadata_list
    }

    /// Gets metadata for a tensor by its name.
    pub fn get_tensor_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.metadata_list.iter().find(|m| m.name == name)
    }

    /// Reads the raw, processed (decompressed, endian-swapped to native) byte data of a tensor.
    ///
    /// # Arguments
    /// * `name`: The name of the tensor to read.
    pub fn read_raw_tensor_data_by_name(&mut self, name: &str) -> Result<Vec<u8>, ZTensorError> {
        // Find metadata by name, clone it to satisfy borrow checker, as reading requires mutable self.
        // A more optimized way might involve passing index or a non-mutable lookup then a mutable read.
        let metadata = self
            .metadata_list
            .iter()
            .find(|m| m.name == name)
            .cloned() // Clone to avoid lifetime issues with self.reader
            .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;
        self.read_raw_tensor_data(&metadata)
    }

    /// Reads the raw, processed (decompressed, endian-swapped to native) byte data of a tensor
    /// using its metadata.
    ///
    /// # Arguments
    /// * `metadata`: The `TensorMetadata` for the tensor to read.
    pub fn read_raw_tensor_data(
        &mut self,
        metadata: &TensorMetadata,
    ) -> Result<Vec<u8>, ZTensorError> {
        if metadata.offset % ALIGNMENT != 0 {
            // Should have been caught at load time
            return Err(ZTensorError::InvalidAlignment {
                offset: metadata.offset,
                required_alignment: ALIGNMENT,
                actual_offset: metadata.offset,
            });
        }

        self.reader.seek(SeekFrom::Start(metadata.offset))?;
        let mut on_disk_data = vec![0u8; metadata.size as usize];
        self.reader.read_exact(&mut on_disk_data)?;

        // Checksum verification
        if let Some(checksum_str) = &metadata.checksum {
            if checksum_str.starts_with("crc32c:0x") {
                let expected_cs_hex = &checksum_str[9..];
                let expected_cs = u32::from_str_radix(expected_cs_hex, 16).map_err(|_| {
                    ZTensorError::ChecksumFormatError(format!(
                        "Invalid CRC32C hex: {}",
                        expected_cs_hex
                    ))
                })?;

                let calculated_cs = crc32c::crc32c(&on_disk_data);

                if calculated_cs != expected_cs {
                    return Err(ZTensorError::ChecksumMismatch {
                        tensor_name: metadata.name.clone(),
                        expected: format!("0x{:08X}", expected_cs),
                        calculated: format!("0x{:08X}", calculated_cs),
                    });
                }
            } else {
                // Silently ignore unknown checksum formats for now, or return an error/warning
                // return Err(ZTensorError::ChecksumFormatError(format!("Unsupported checksum format: {}", checksum_str)));
                eprintln!(
                    "Warning: Tensor '{}' has an unsupported checksum format: {}",
                    metadata.name, checksum_str
                );
            }
        }

        let mut decoded_data = match &metadata.encoding {
            Encoding::Raw => on_disk_data,
            Encoding::Zstd => zstd::decode_all(std::io::Cursor::new(on_disk_data))
                .map_err(ZTensorError::ZstdDecompression)?,
        };

        let expected_uncompressed_size = metadata.uncompressed_data_size()?; // Now returns Result
        if decoded_data.len() as u64 != expected_uncompressed_size {
            return Err(ZTensorError::InconsistentDataSize {
                expected: expected_uncompressed_size,
                found: decoded_data.len() as u64,
            });
        }

        if metadata.encoding == Encoding::Raw && metadata.dtype.is_multi_byte() {
            let dtype_byte_size = metadata.dtype.byte_size()?;
            let stored_endianness = metadata.data_endianness.clone().unwrap_or_default();
            if stored_endianness != NATIVE_ENDIANNESS {
                swap_endianness_in_place(&mut decoded_data, dtype_byte_size);
            }
        }
        Ok(decoded_data)
    }

    /// Reads tensor data and converts it to a Vec of a specified `Pod` type.
    ///
    /// # Type Parameters
    /// * `T`: The `Pod` type to convert the data into (e.g., `f32`, `u16`).
    ///
    /// # Arguments
    /// * `name`: The name of the tensor to read.
    pub fn read_typed_tensor_data<T: Pod>(&mut self, name: &str) -> Result<Vec<T>, ZTensorError> {
        let metadata = self
            .metadata_list
            .iter()
            .find(|m| m.name == name)
            .cloned()
            .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;

        if !T::dtype_matches(&metadata.dtype) {
            return Err(ZTensorError::TypeMismatch {
                expected: metadata.dtype.to_string_key(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("tensor '{}'", name),
            });
        }

        let raw_bytes = self.read_raw_tensor_data(&metadata)?;

        let element_size = T::SIZE;
        if raw_bytes.len() % element_size != 0 {
            return Err(ZTensorError::DataConversionError(format!(
                "Raw data size {} is not a multiple of element size {} for type {}",
                raw_bytes.len(),
                element_size,
                std::any::type_name::<T>()
            )));
        }

        let num_elements = raw_bytes.len() / element_size;
        let mut typed_data = Vec::with_capacity(num_elements);

        // Data in raw_bytes is already in NATIVE_ENDIANNESS due to read_raw_tensor_data
        for chunk in raw_bytes.chunks_exact(element_size) {
            // Since read_raw_tensor_data ensures native endianness,
            // we can use the native endianness Pod constructor.
            // For Pod types here, from_le_bytes or from_be_bytes are chosen by NATIVE_ENDIANNESS
            // This logic might need refinement based on Pod trait design.
            // Let's assume Pod::from_le_bytes and from_be_bytes are for when data IS le/be.
            // The raw_bytes are already NATIVE.

            // Simplified: if native is little, use from_le, if native is big, use from_be.
            // This assumes T::from_le_bytes correctly interprets LE bytes regardless of host.
            let val = if cfg!(target_endian = "little") {
                T::from_le_bytes(chunk)
            } else {
                T::from_be_bytes(chunk)
            };
            typed_data.push(val);
        }
        Ok(typed_data)
    }
}
