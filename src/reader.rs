use std::fs::File;
use std::io::{Read, Seek, SeekFrom, BufReader};
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt};

use crate::models::{TensorMetadata, MAGIC_NUMBER, ALIGNMENT};
use crate::error::ZTensorError;
use crate::utils::{NATIVE_ENDIANNESS, swap_endianness_in_place};

pub struct ZTensorReader<R: Read + Seek> {
    reader: R,
    pub metadata_list: Vec<TensorMetadata>,
}

impl ZTensorReader<BufReader<File>> {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::open(path)?;
        Self::new(BufReader::new(file))
    }
}

impl<R: Read + Seek> ZTensorReader<R> {
    pub fn new(mut reader: R) -> Result<Self, ZTensorError> {
        // 1. Read Magic Number
        let mut magic_buf = [0u8; 8];
        reader.read_exact(&mut magic_buf)?;
        if magic_buf != *MAGIC_NUMBER {
            return Err(ZTensorError::InvalidMagicNumber { found: magic_buf.to_vec() });
        }

        // 2. Read Total CBOR Blob Size (last 8 bytes)
        reader.seek(SeekFrom::End(-8))?;
        let cbor_blob_size = reader.read_u64::<LittleEndian>()?;

        // Handle zero-tensor case: magic (8) + cbor_blob (1) + size (8) = 17
        let file_size = reader.seek(SeekFrom::End(0))?;
        if file_size == (MAGIC_NUMBER.len() as u64 + 1 + 8) && cbor_blob_size == 1 {
             reader.seek(SeekFrom::Start(MAGIC_NUMBER.len() as u64))?;
             let mut cbor_byte = [0u8;1];
             reader.read_exact(&mut cbor_byte)?;
             if cbor_byte[0] == 0x80 { // Empty CBOR array
                return Ok(Self { reader, metadata_list: Vec::new() });
             } else {
                return Err(ZTensorError::InvalidFileStructure("Invalid CBOR for empty tensor list".to_string()));
             }
        }
        
        if cbor_blob_size > file_size - 8 - 8 { // cbor_blob_size should not be larger than file_size - magic - cbor_size_field
             return Err(ZTensorError::InvalidFileStructure(format!("CBOR blob size {} is too large for file size {}", cbor_blob_size, file_size)));
        }


        // 3. Read CBOR Blob
        reader.seek(SeekFrom::End(-8 - (cbor_blob_size as i64)))?;
        let mut cbor_buf = vec![0u8; cbor_blob_size as usize];
        reader.read_exact(&mut cbor_buf)?;

        let metadata_list: Vec<TensorMetadata> = serde_cbor::from_slice(&cbor_buf)
            .map_err(ZTensorError::CborDeserialize)?;

        // 4. Validate offsets and alignment from metadata (optional, but good practice)
        for meta in &metadata_list {
            if meta.offset < MAGIC_NUMBER.len() as u64 {
                return Err(ZTensorError::InvalidFileStructure(format!(
                    "Tensor '{}' offset {} is within magic number.", meta.name, meta.offset
                )));
            }
            if meta.offset % ALIGNMENT != 0 {
                 return Err(ZTensorError::InvalidAlignment {
                    offset: meta.offset,
                    required_alignment: ALIGNMENT,
                    actual_offset: meta.offset % ALIGNMENT,
                });
            }
        }


        Ok(Self { reader, metadata_list })
    }

    pub fn list_tensors(&self) -> &Vec<TensorMetadata> {
        &self.metadata_list
    }

    pub fn get_tensor_metadata(&self, name: &str) -> Option<&TensorMetadata> {
        self.metadata_list.iter().find(|m| m.name == name)
    }

    pub fn read_raw_tensor_data_by_name(&mut self, name: &str) -> Result<Vec<u8>, ZTensorError> {
        let metadata = self.get_tensor_metadata(name)
            .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?
            .clone(); // Clone to avoid borrowing self mutably twice.
        self.read_raw_tensor_data(&metadata)
    }
    
    pub fn read_raw_tensor_data(&mut self, metadata: &TensorMetadata) -> Result<Vec<u8>, ZTensorError> {
        if metadata.offset % ALIGNMENT != 0 {
            return Err(ZTensorError::InvalidAlignment {
                offset: metadata.offset,
                required_alignment: ALIGNMENT,
                actual_offset: metadata.offset % ALIGNMENT, // This should have been caught at init, but good check
            });
        }

        self.reader.seek(SeekFrom::Start(metadata.offset))?;
        let mut on_disk_data = vec![0u8; metadata.size as usize];
        self.reader.read_exact(&mut on_disk_data)?;

        // TODO: Implement checksum verification if metadata.checksum is Some(...)

        let mut decoded_data = match metadata.encoding {
            crate::models::Encoding::Raw => on_disk_data,
            crate::models::Encoding::Zstd => {
                zstd::decode_all(std::io::Cursor::new(on_disk_data))
                    .map_err(ZTensorError::ZstdDecompression)?
            }
        };

        // Verify uncompressed size
        let expected_uncompressed_size = metadata.uncompressed_data_size();
        if decoded_data.len() as u64 != expected_uncompressed_size {
            return Err(ZTensorError::InconsistentDataSize {
                expected: expected_uncompressed_size,
                found: decoded_data.len() as u64,
            });
        }

        // Handle endianness for raw, multi-byte types
        if metadata.encoding == crate::models::Encoding::Raw && metadata.dtype.is_multi_byte() {
            let stored_endianness = metadata.data_endianness.clone().unwrap_or_default();
            if stored_endianness != NATIVE_ENDIANNESS {
                swap_endianness_in_place(&mut decoded_data, metadata.dtype.byte_size());
            }
        }
        Ok(decoded_data)
    }
}