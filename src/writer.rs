use std::fs::File;
use std::io::{Write, Seek, SeekFrom, BufWriter};
use std::path::Path;
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::BTreeMap;

// Updated models import
use crate::models::{TensorMetadata, DType, Encoding, DataEndianness, MAGIC_NUMBER, ALIGNMENT, ChecksumAlgorithm};
use crate::error::ZTensorError;
use crate::utils::{calculate_padding, NATIVE_ENDIANNESS, swap_endianness_in_place};
use serde_cbor::Value as CborValue;

struct PendingTensor {
    metadata: TensorMetadata,
    data_to_write: Vec<u8>, // This is the final on-disk data (e.g. compressed)
}

/// Writes zTensor files.
pub struct ZTensorWriter<W: Write + Seek> {
    writer: W,
    tensors_to_write: Vec<PendingTensor>,
    current_offset: u64,
}

impl ZTensorWriter<BufWriter<File>> {
    /// Creates a new `ZTensorWriter` for the given file path.
    /// The file will be created or truncated.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::create(path)?;
        Self::new(BufWriter::new(file))
    }
}

impl<W: Write + Seek> ZTensorWriter<W> {
    /// Creates a new `ZTensorWriter` using the provided `Write + Seek` object.
    /// Writes the zTensor magic number to the beginning of the writer.
    pub fn new(mut writer: W) -> Result<Self, ZTensorError> {
        writer.write_all(MAGIC_NUMBER)?;
        Ok(Self {
            writer,
            tensors_to_write: Vec::new(),
            current_offset: MAGIC_NUMBER.len() as u64,
        })
    }

    /// Adds a tensor to the zTensor file.
    ///
    /// The tensor data will be processed (endianness swap, compression) according to the parameters
    /// and prepared for writing. The actual writing to disk happens during `finalize`.
    ///
    /// # Arguments
    /// * `name`: Name of the tensor.
    /// * `shape`: Shape (dimensions) of the tensor.
    /// * `dtype`: Data type of the tensor elements.
    /// * `encoding`: Encoding to use for storing the tensor data (e.g., Raw, Zstd).
    /// * `raw_native_data`: Raw tensor data as bytes, in the host's native endianness.
    /// * `store_endianness`: If `encoding` is `Raw` and `dtype` is multi-byte, this specifies
    ///                       the endianness to store the data in. Defaults to Little Endian.
    /// * `checksum_algo`: Algorithm to use for calculating a checksum of the on-disk data.
    /// * `custom_fields`: Optional BTreeMap for custom metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn add_tensor(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        encoding: Encoding,
        mut raw_native_data: Vec<u8>, // Input data is raw, in native host endianness
        store_endianness: Option<DataEndianness>, // For raw encoding: how to store it. Defaults to Little.
        checksum_algo: ChecksumAlgorithm,
        custom_fields: Option<BTreeMap<String, CborValue>>,
    ) -> Result<(), ZTensorError> {
      

        let num_elements = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_raw_size = num_elements * dtype.byte_size()? as u64;

        if raw_native_data.len() as u64 != expected_raw_size {
            return Err(ZTensorError::InconsistentDataSize {
                expected: expected_raw_size,
                found: raw_native_data.len() as u64,
            });
        }

        // Handle endianness for raw data before compression or direct storage
        let final_data_endianness_field = if encoding == Encoding::Raw && dtype.is_multi_byte() {
            let target_endianness = store_endianness.unwrap_or_default(); // Default to Little
            if target_endianness != NATIVE_ENDIANNESS {
                swap_endianness_in_place(&mut raw_native_data, dtype.byte_size()?);
            }
            Some(target_endianness)
        } else {
            None
        };
        
        // Process data for on-disk storage (compression)
        let on_disk_data = match encoding {
            Encoding::Raw => raw_native_data, // Already endian-swapped if needed
            Encoding::Zstd => zstd::encode_all(std::io::Cursor::new(raw_native_data), 0)
                .map_err(ZTensorError::ZstdCompression)?,
        };
        
        let on_disk_size = on_disk_data.len() as u64;

        // Calculate checksum if requested
        let checksum_str = match checksum_algo {
            ChecksumAlgorithm::None => None,
            ChecksumAlgorithm::Crc32c => {
                // Checksum is calculated on the *on-disk* data (potentially compressed)
                let cs_val = crc32c::crc32c(&on_disk_data);
                Some(format!("crc32c:0x{:08X}", cs_val))
            }
        };
        
        let metadata = TensorMetadata {
            name: name.to_string(),
            offset: 0, // Placeholder, filled during finalize
            size: on_disk_size,
            dtype,
            shape,
            encoding,
            data_endianness: final_data_endianness_field,
            checksum: checksum_str,
            custom_fields: custom_fields.unwrap_or_default(),
        };

        self.tensors_to_write.push(PendingTensor {
            metadata,
            data_to_write: on_disk_data,
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
            self.writer.write_all(&pending_tensor.data_to_write)?;
            
            self.current_offset = aligned_offset + pending_tensor.metadata.size;
            finalized_metadata_list.push(pending_tensor.metadata.clone());
        }
        Ok(finalized_metadata_list)
    }

    /// Finalizes the zTensor file.
    /// This writes all pending tensor data (with padding), then the CBOR metadata index,
    /// and finally the size of the CBOR index.
    ///
    /// Returns the total size of the file written.
    pub fn finalize(mut self) -> Result<u64, ZTensorError> {
        let finalized_metadata = self.write_tensor_blobs()?;

        let cbor_blob = serde_cbor::to_vec(&finalized_metadata)
            .map_err(ZTensorError::CborSerialize)?;
        
        self.writer.write_all(&cbor_blob)?;
        
        let cbor_blob_size = cbor_blob.len() as u64;
        self.writer.write_u64::<LittleEndian>(cbor_blob_size)?;
        
        self.writer.flush()?;
        
        self.writer.seek(SeekFrom::Current(0)).map_err(ZTensorError::Io)
    }
}

/// Writes a zTensor file with zero tensors.
/// This is a convenience function. The same result can be achieved by creating a `ZTensorWriter`
/// and calling `finalize` without adding any tensors.
pub fn write_empty_ztensor_file<W: Write + Seek>(mut writer: W) -> Result<u64, ZTensorError> {
    writer.write_all(MAGIC_NUMBER)?;
    let empty_cbor_array = [0x80u8]; 
    writer.write_all(&empty_cbor_array)?;
    writer.write_u64::<LittleEndian>(1u64)?; 
    writer.flush()?;
    Ok((MAGIC_NUMBER.len() + 1 + 8) as u64)
}