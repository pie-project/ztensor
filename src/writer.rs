use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use crate::error::ZTensorError;
use crate::models::{
    ChecksumAlgorithm, Component, DType, Encoding, Manifest, Tensor, MAGIC_NUMBER,
};
use crate::utils::align_offset;

/// Writes zTensor files (v1.0).
pub struct ZTensorWriter<W: Write + Seek> {
    writer: W,
    manifest: Manifest,
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
            manifest: Manifest::default(),
            current_offset: MAGIC_NUMBER.len() as u64,
        })
    }

    /// Adds a dense tensor to the zTensor file.
    ///
    /// This is a high-level helper for the strict "dense" format where there is a single
    /// "data" component.
    ///
    /// # Arguments
    /// * `name`: Name of the tensor.
    /// * `shape`: Shape (dimensions) of the tensor.
    /// * `dtype`: Data type of the tensor elements.
    /// * `encoding`: Encoding to use for storing the tensor data (e.g., Raw, Zstd).
    /// * `raw_native_data`: Raw tensor data as bytes, in the host's native endianness.
    /// * `checksum_algo`: Algorithm to use for calculating a checksum.
    #[allow(clippy::too_many_arguments)]
    pub fn add_tensor(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        encoding: Encoding,
        raw_native_data: Vec<u8>,
        checksum_algo: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        // Validation: Check size
        let num_elements: u64 = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_raw_size = num_elements * dtype.byte_size()? as u64;

        if raw_native_data.len() as u64 != expected_raw_size {
            return Err(ZTensorError::InconsistentDataSize {
                expected: expected_raw_size,
                found: raw_native_data.len() as u64,
            });
        }

        let component = self.write_component(raw_native_data, dtype, encoding, checksum_algo)?;

        let mut components = BTreeMap::new();
        components.insert("data".to_string(), component);

        let tensor = Tensor {
            dtype,
            shape,
            format: "dense".to_string(),
            components,
        };

        self.manifest.tensors.insert(name.to_string(), tensor);

        Ok(())
    }

    /// Adds a sparse CSR tensor to the zTensor file.
    #[allow(clippy::too_many_arguments)]
    pub fn add_csr_tensor(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values_native: Vec<u8>,
        indices: Vec<u64>,
        indptr: Vec<u64>,
        encoding: Encoding,
        checksum_algo: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        let values_comp = self.write_component(values_native, dtype, encoding, checksum_algo)?;
        
        let indices_bytes = crate::utils::u64_vec_to_bytes(indices);
        let indices_comp = self.write_component(indices_bytes, DType::Uint64, encoding, checksum_algo)?;
        
        let indptr_bytes = crate::utils::u64_vec_to_bytes(indptr);
        let indptr_comp = self.write_component(indptr_bytes, DType::Uint64, encoding, checksum_algo)?;

        let mut components = BTreeMap::new();
        components.insert("values".to_string(), values_comp);
        components.insert("indices".to_string(), indices_comp);
        components.insert("indptr".to_string(), indptr_comp);

        let tensor = Tensor {
            dtype,
            shape,
            format: "sparse_csr".to_string(),
            components,
        };

        self.manifest.tensors.insert(name.to_string(), tensor);
        Ok(())
    }

    /// Adds a sparse COO tensor to the zTensor file.
    #[allow(clippy::too_many_arguments)]
    pub fn add_coo_tensor(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values_native: Vec<u8>,
        indices: Vec<u64>,
        encoding: Encoding,
        checksum_algo: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        let values_comp = self.write_component(values_native, dtype, encoding, checksum_algo)?;
        
        let indices_bytes = crate::utils::u64_vec_to_bytes(indices);
        let coords_comp = self.write_component(indices_bytes, DType::Uint64, encoding, checksum_algo)?;

        let mut components = BTreeMap::new();
        components.insert("values".to_string(), values_comp);
        components.insert("coords".to_string(), coords_comp);

        let tensor = Tensor {
            dtype,
            shape,
            format: "sparse_coo".to_string(),
            components,
        };

        self.manifest.tensors.insert(name.to_string(), tensor);
        Ok(())
    }

    /// Backward compatibility alias for add_csr_tensor
    #[deprecated(since = "0.2.0", note = "use add_csr_tensor instead")]
    #[allow(clippy::too_many_arguments)]
    pub fn add_sparse_csr_tensor(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values_native: Vec<u8>,
        indices: Vec<u64>,
        indptr: Vec<u64>,
        encoding: Encoding,
        checksum_algo: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        self.add_csr_tensor(name, shape, dtype, values_native, indices, indptr, encoding, checksum_algo)
    }

    /// Backward compatibility alias for add_coo_tensor
    #[deprecated(since = "0.2.0", note = "use add_coo_tensor instead")]
    #[allow(clippy::too_many_arguments)]
    pub fn add_sparse_coo_tensor(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values_native: Vec<u8>,
        indices: Vec<u64>,
        encoding: Encoding,
        checksum_algo: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        self.add_coo_tensor(name, shape, dtype, values_native, indices, encoding, checksum_algo)
    }

    fn write_component(
        &mut self,
        mut data: Vec<u8>,
        dtype: DType,
        encoding: Encoding,
        checksum_algo: ChecksumAlgorithm
    ) -> Result<Component, ZTensorError> {
        // 1. Handle Endianness: convert from native to little-endian if needed
        if !crate::utils::is_little_endian() && dtype.byte_size()? > 1 {
             crate::utils::swap_endianness_in_place(&mut data, dtype.byte_size()?);
        }

        // 2. Compression
        let (on_disk_data, stored_encoding) = match encoding {
            Encoding::Raw => (data, None), 
            Encoding::Zstd => {
                let compressed = zstd::encode_all(std::io::Cursor::new(data), 0)
                    .map_err(ZTensorError::ZstdCompression)?;
                (compressed, Some(Encoding::Zstd))
            }
        };

        // 3. Checksum
        let digest_str = match checksum_algo {
            ChecksumAlgorithm::None => None,
            ChecksumAlgorithm::Crc32c => {
                let cs_val = crc32c::crc32c(&on_disk_data);
                Some(format!("crc32c:0x{:08X}", cs_val))
            }
        };

        // 4. Write
        let (offset, length) = self.write_component_data(&on_disk_data)?;

        Ok(Component {
            offset,
            length,
            encoding: stored_encoding,
            digest: digest_str,
        })
    }

    /// Internal helper to write a byte blob to the file with alignment.
    /// Returns (offset, length).
    fn write_component_data(&mut self, data: &[u8]) -> Result<(u64, u64), ZTensorError> {
        let (aligned_offset, padding_bytes) = align_offset(self.current_offset);

        if padding_bytes > 0 {
            self.writer.write_all(&vec![0u8; padding_bytes as usize])?;
        }

        self.writer.write_all(data)?;
        let length = data.len() as u64;

        self.current_offset = aligned_offset + length;
        Ok((aligned_offset, length))
    }

    /// Finalizes the zTensor file.
    /// Writes the Manifest (CBOR) and the Manifest Size.
    pub fn finalize(mut self) -> Result<u64, ZTensorError> {
        let cbor_blob =
            serde_cbor::to_vec(&self.manifest).map_err(ZTensorError::CborSerialize)?;

        self.writer.write_all(&cbor_blob)?;

        let cbor_blob_size = cbor_blob.len() as u64;
        self.writer.write_u64::<LittleEndian>(cbor_blob_size)?;

        self.writer.flush()?;

        // Current offset is at the end of the file now
        Ok(self.current_offset + cbor_blob_size + 8)
    }
}
