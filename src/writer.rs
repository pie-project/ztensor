//! zTensor file writer.

use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use crate::error::ZTensorError;
use crate::models::{ChecksumAlgorithm, Component, DType, Encoding, Manifest, Object, MAGIC};
use crate::utils::{align_offset, is_little_endian, sha256_hex, swap_endianness_in_place, u64_vec_to_bytes};

/// Writer for zTensor v1.1 files.
pub struct ZTensorWriter<W: Write + Seek> {
    writer: W,
    manifest: Manifest,
    current_offset: u64,
}

impl ZTensorWriter<BufWriter<File>> {
    /// Creates a new writer for the given file path.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::create(path)?;
        Self::new(BufWriter::new(file))
    }
}

impl<W: Write + Seek> ZTensorWriter<W> {
    /// Creates a new writer from a Write + Seek source.
    pub fn new(mut writer: W) -> Result<Self, ZTensorError> {
        writer.write_all(MAGIC)?;
        Ok(Self {
            writer,
            manifest: Manifest::default(),
            current_offset: MAGIC.len() as u64,
        })
    }

    /// Sets global attributes on the manifest.
    pub fn set_attributes(&mut self, attrs: BTreeMap<String, String>) {
        self.manifest.attributes = Some(attrs);
    }

    /// Adds a dense object (tensor) to the file.
    #[allow(clippy::too_many_arguments)]
    pub fn add_object(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        encoding: Encoding,
        raw_data: Vec<u8>,
        checksum: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        // Validate size
        let num_elements: u64 = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_size = num_elements * dtype.byte_size() as u64;

        if raw_data.len() as u64 != expected_size {
            return Err(ZTensorError::InconsistentDataSize {
                expected: expected_size,
                found: raw_data.len() as u64,
            });
        }

        let component = self.write_component(raw_data, dtype, encoding, checksum)?;

        let mut components = BTreeMap::new();
        components.insert("data".to_string(), component);

        let obj = Object {
            shape,
            format: "dense".to_string(),
            attributes: None,
            components,
        };

        self.manifest.objects.insert(name.to_string(), obj);
        Ok(())
    }

    /// Adds a CSR sparse object.
    #[allow(clippy::too_many_arguments)]
    pub fn add_csr_object(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: Vec<u8>,
        indices: Vec<u64>,
        indptr: Vec<u64>,
        encoding: Encoding,
        checksum: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        let values_comp = self.write_component(values, dtype, encoding, checksum)?;
        let indices_bytes = u64_vec_to_bytes(indices);
        let indices_comp = self.write_component(indices_bytes, DType::U64, encoding, checksum)?;
        let indptr_bytes = u64_vec_to_bytes(indptr);
        let indptr_comp = self.write_component(indptr_bytes, DType::U64, encoding, checksum)?;

        let mut components = BTreeMap::new();
        components.insert("values".to_string(), values_comp);
        components.insert("indices".to_string(), indices_comp);
        components.insert("indptr".to_string(), indptr_comp);

        let obj = Object {
            shape,
            format: "sparse_csr".to_string(),
            attributes: None,
            components,
        };

        self.manifest.objects.insert(name.to_string(), obj);
        Ok(())
    }

    /// Adds a COO sparse object.
    #[allow(clippy::too_many_arguments)]
    pub fn add_coo_object(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: Vec<u8>,
        coords: Vec<u64>,
        encoding: Encoding,
        checksum: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        let values_comp = self.write_component(values, dtype, encoding, checksum)?;
        let coords_bytes = u64_vec_to_bytes(coords);
        let coords_comp = self.write_component(coords_bytes, DType::U64, encoding, checksum)?;

        let mut components = BTreeMap::new();
        components.insert("values".to_string(), values_comp);
        components.insert("coords".to_string(), coords_comp);

        let obj = Object {
            shape,
            format: "sparse_coo".to_string(),
            attributes: None,
            components,
        };

        self.manifest.objects.insert(name.to_string(), obj);
        Ok(())
    }

    fn write_component(
        &mut self,
        mut data: Vec<u8>,
        dtype: DType,
        encoding: Encoding,
        checksum: ChecksumAlgorithm,
    ) -> Result<Component, ZTensorError> {
        // Convert to little-endian if needed
        if !is_little_endian() && dtype.byte_size() > 1 {
            swap_endianness_in_place(&mut data, dtype.byte_size());
        }

        // Compress if requested
        let (on_disk_data, stored_encoding) = match encoding {
            Encoding::Raw => (data, Encoding::Raw),
            Encoding::Zstd => {
                let compressed = zstd::encode_all(std::io::Cursor::new(data), 0)
                    .map_err(ZTensorError::ZstdCompression)?;
                (compressed, Encoding::Zstd)
            }
        };

        // Calculate checksum
        let digest = match checksum {
            ChecksumAlgorithm::None => None,
            ChecksumAlgorithm::Crc32c => {
                let cs = crc32c::crc32c(&on_disk_data);
                Some(format!("crc32c:0x{:08X}", cs))
            }
            ChecksumAlgorithm::Sha256 => {
                let hash = sha256_hex(&on_disk_data);
                Some(format!("sha256:{}", hash))
            }
        };

        // Write with alignment
        let (offset, length) = self.write_aligned_data(&on_disk_data)?;

        Ok(Component {
            dtype,
            offset,
            length,
            encoding: stored_encoding,
            digest,
        })
    }

    fn write_aligned_data(&mut self, data: &[u8]) -> Result<(u64, u64), ZTensorError> {
        let (aligned_offset, padding) = align_offset(self.current_offset);

        if padding > 0 {
            self.writer.write_all(&vec![0u8; padding as usize])?;
        }

        self.writer.write_all(data)?;
        let length = data.len() as u64;

        self.current_offset = aligned_offset + length;
        Ok((aligned_offset, length))
    }

    /// Finalizes the file by writing manifest and footer.
    ///
    /// File structure:
    /// [MAGIC 8B] [BLOBS...] [CBOR MANIFEST] [MANIFEST SIZE 8B] [MAGIC 8B]
    pub fn finalize(mut self) -> Result<u64, ZTensorError> {
        let cbor = serde_cbor::to_vec(&self.manifest)
            .map_err(ZTensorError::CborSerialize)?;

        self.writer.write_all(&cbor)?;

        let cbor_size = cbor.len() as u64;
        self.writer.write_u64::<LittleEndian>(cbor_size)?;

        // Write footer magic
        self.writer.write_all(MAGIC)?;

        self.writer.flush()?;

        Ok(self.current_offset + cbor_size + 8 + 8)
    }
}
