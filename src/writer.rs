use std::fs::File;
use std::io::{Write, Seek, SeekFrom, BufWriter};
use std::path::Path;
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::BTreeMap;

use crate::models::{TensorMetadata, DType, Encoding, DataEndianness, MAGIC_NUMBER, ALIGNMENT};
use crate::error::ZTensorError;
use crate::utils::{calculate_padding, NATIVE_ENDIANNESS, swap_endianness_in_place};
use serde_cbor::Value as CborValue;


struct PendingTensor {
    metadata: TensorMetadata, // offset will be filled in later
    data: Vec<u8>, // Already encoded/compressed data
}

pub struct ZTensorWriter<W: Write + Seek> {
    writer: W,
    tensors_to_write: Vec<PendingTensor>,
    current_offset: u64,
}

impl ZTensorWriter<BufWriter<File>> {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::create(path)?;
        Self::new(BufWriter::new(file))
    }
}

impl<W: Write + Seek> ZTensorWriter<W> {
    pub fn new(mut writer: W) -> Result<Self, ZTensorError> {
        writer.write_all(MAGIC_NUMBER)?;
        Ok(Self {
            writer,
            tensors_to_write: Vec::new(),
            current_offset: MAGIC_NUMBER.len() as u64,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_tensor(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        encoding: Encoding,
        mut data: Vec<u8>, // This is the raw, uncompressed tensor data in NATIVE_ENDIANNESS
        data_endianness_on_disk: Option<DataEndianness>, // For raw encoding: how to store it
        checksum: Option<String>,
        custom_fields: Option<BTreeMap<String, CborValue>>,
    ) -> Result<(), ZTensorError> {

        // Calculate uncompressed size for validation before potential compression
        let num_elements = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_raw_size = num_elements * dtype.byte_size() as u64;

        if data.len() as u64 != expected_raw_size {
            return Err(ZTensorError::InconsistentDataSize {
                expected: expected_raw_size,
                found: data.len() as u64,
            });
        }

        let final_data_endianness = if encoding == Encoding::Raw && dtype.is_multi_byte() {
            let target_endianness = data_endianness_on_disk.unwrap_or(DataEndianness::Little);
            if target_endianness != NATIVE_ENDIANNESS {
                swap_endianness_in_place(&mut data, dtype.byte_size());
            }
            Some(target_endianness)
        } else {
            None // Not applicable or data is not raw multi-byte
        };

        let (on_disk_data, on_disk_size) = match encoding {
            Encoding::Raw => {
                let size = data.len() as u64;
                (data, size)
            }
            Encoding::Zstd => {
                let compressed_data = zstd::encode_all(std::io::Cursor::new(data), 0) // 0 is default compression level
                    .map_err(ZTensorError::ZstdCompression)?;
                let size = compressed_data.len() as u64;
                (compressed_data, size)
            }
        };
        
        let metadata = TensorMetadata {
            name: name.to_string(),
            offset: 0, // Will be filled during finalize/write_blobs
            size: on_disk_size,
            dtype,
            shape,
            encoding,
            data_endianness: final_data_endianness,
            checksum,
            custom_fields: custom_fields.unwrap_or_default(),
        };

        self.tensors_to_write.push(PendingTensor {
            metadata,
            data: on_disk_data,
        });

        Ok(())
    }
    
    fn write_tensor_blobs(&mut self) -> Result<Vec<TensorMetadata>, ZTensorError> {
        let mut finalized_metadata_list = Vec::new();

        for pending_tensor in self.tensors_to_write.iter_mut() {
            let (aligned_offset, padding_bytes) = calculate_padding(self.current_offset);
            
            if padding_bytes > 0 {
                self.writer.write_all(&vec![0u8; padding_bytes as usize])?;
            }
            
            pending_tensor.metadata.offset = aligned_offset;
            self.writer.write_all(&pending_tensor.data)?;
            
            self.current_offset = aligned_offset + pending_tensor.metadata.size;
            finalized_metadata_list.push(pending_tensor.metadata.clone());
        }
        Ok(finalized_metadata_list)
    }

    pub fn finalize(mut self) -> Result<u64, ZTensorError> {
        // Write all tensor blobs and collect finalized metadata
        let finalized_metadata = self.write_tensor_blobs()?;

        // Serialize metadata to CBOR
        let cbor_blob = serde_cbor::to_vec(&finalized_metadata)
            .map_err(ZTensorError::CborSerialize)?;
        
        // Write CBOR Blob
        self.writer.write_all(&cbor_blob)?;
        
        // Write Total CBOR Blob Size (u64, little-endian)
        let cbor_blob_size = cbor_blob.len() as u64;
        self.writer.write_u64::<LittleEndian>(cbor_blob_size)?;
        
        self.writer.flush()?; // Ensure all buffered data is written.
        
        // Return total file size
        let total_size = self.writer.seek(SeekFrom::Current(0))?;
        Ok(total_size)
    }
}

// Special case for zero-tensor files, if needed as a separate function.
// Otherwise, calling finalize on a ZTensorWriter with no added tensors handles it.
pub fn write_empty_ztensor_file<W: Write + Seek>(mut writer: W) -> Result<u64, ZTensorError> {
    writer.write_all(MAGIC_NUMBER)?;
    let empty_cbor_array = [0x80u8]; // CBOR representation of an empty array
    writer.write_all(&empty_cbor_array)?;
    writer.write_u64::<LittleEndian>(1u64)?; // Size of empty_cbor_array
    writer.flush()?;
    Ok((MAGIC_NUMBER.len() + 1 + 8) as u64)
}