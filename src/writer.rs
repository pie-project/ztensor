//! zTensor file writer.

use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use crate::error::ZTensorError;
use crate::models::{ChecksumAlgorithm, Component, DType, Encoding, Manifest, Object, MAGIC};
use crate::utils::{align_offset, is_little_endian, swap_endianness_in_place, DigestWriter};
use crate::reader::Pod;

/// Compression settings for writing.
#[derive(Debug, Clone, Copy)]
pub enum Compression {
    /// No compression.
    Raw,
    /// Zstd compression with specified level (default is 3, 0 means default).
    Zstd(i32),
}

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

    /// Adds a dense object from raw bytes (FFI/unsafe usage).
    ///
    /// The caller must ensure `data` contains valid LE bytes for the given `dtype`.
    /// Endianness swapping will be performed if `dtype` is multi-byte and host is BE.
    pub fn add_object_bytes(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        compression: Compression,
        data: &[u8],
        checksum: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        let num_elements: u64 = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_size = num_elements * dtype.byte_size() as u64;

        if data.len() as u64 != expected_size {
             return Err(ZTensorError::InconsistentDataSize {
                expected: expected_size,
                found: data.len() as u64,
            });
        }

        let component = self.write_component(data, dtype, compression, checksum)?;
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

    /// Adds a dense object (tensor) to the file.
    #[allow(clippy::too_many_arguments)]
    pub fn add_object<T: Pod + bytemuck::Pod>(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        compression: Compression,
        data: &[T],
        checksum: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        // Validation
        if !T::dtype_matches(&dtype) {
             return Err(ZTensorError::TypeMismatch {
                expected: dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("add_object '{}'", name),
            });
        }
        
        // Safe cast to bytes (requires T: bytemuck::Pod)
        let bytes = bytemuck::cast_slice(data);
        self.add_object_bytes(name, shape, dtype, compression, bytes, checksum)
    }

    /// Adds a CSR sparse object from raw bytes.
    pub fn add_csr_object_bytes(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: &[u8],
        indices: &[u64],
        indptr: &[u64],
        compression: Compression,
        checksum: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        let indices_bytes = bytemuck::cast_slice(indices);
        let indptr_bytes = bytemuck::cast_slice(indptr);

        let values_comp = self.write_component(values, dtype, compression, checksum)?;
        let indices_comp = self.write_component(indices_bytes, DType::U64, compression, checksum)?;
        let indptr_comp = self.write_component(indptr_bytes, DType::U64, compression, checksum)?;

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

    /// Adds a CSR sparse object.
    #[allow(clippy::too_many_arguments)]
    pub fn add_csr_object<T: Pod + bytemuck::Pod>(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: &[T],
        indices: &[u64],
        indptr: &[u64],
        compression: Compression,
        checksum: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        let values_bytes = bytemuck::cast_slice(values);
        self.add_csr_object_bytes(name, shape, dtype, values_bytes, indices, indptr, compression, checksum)
    }

    /// Adds a COO sparse object from raw bytes.
    pub fn add_coo_object_bytes(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: &[u8],
        coords: &[u64],
        compression: Compression,
        checksum: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        let coords_bytes = bytemuck::cast_slice(coords);
        
        let values_comp = self.write_component(values, dtype, compression, checksum)?;
        let coords_comp = self.write_component(coords_bytes, DType::U64, compression, checksum)?;

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

    /// Adds a COO sparse object.
    #[allow(clippy::too_many_arguments)]
    pub fn add_coo_object<T: Pod + bytemuck::Pod>(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: &[T],
        coords: &[u64],
        compression: Compression,
        checksum: ChecksumAlgorithm,
    ) -> Result<(), ZTensorError> {
        let values_bytes = bytemuck::cast_slice(values);
        self.add_coo_object_bytes(name, shape, dtype, values_bytes, coords, compression, checksum)
    }

    fn write_component(
        &mut self,
        data: &[u8],
        dtype: DType,
        compression: Compression,
        checksum: ChecksumAlgorithm,
    ) -> Result<Component, ZTensorError> {
        // 1. Align
        let (aligned_offset, padding) = align_offset(self.current_offset);
        if padding > 0 {
            self.writer.write_all(&vec![0u8; padding as usize])?;
        }
        self.current_offset = aligned_offset;

        // 2. Setup Writers
        struct CountingWriter<W> {
            inner: W,
            count: u64,
        }
        impl<W: Write> Write for CountingWriter<W> {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let n = self.inner.write(buf)?;
                self.count += n as u64;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.inner.flush()
            }
        }

        let mut counting_writer = CountingWriter {
            inner: &mut self.writer,
            count: 0,
        };
        
        let mut digest_writer = DigestWriter::new(&mut counting_writer, checksum);

        let stored_encoding = match compression {
            Compression::Raw => {
                // Write data
                // Uses streaming chunks to perform endianness swap if needed
                Self::write_data(&mut digest_writer, data, dtype)?;
                Encoding::Raw
            }
            Compression::Zstd(level) => {
                {
                    let mut encoder = zstd::stream::write::Encoder::new(&mut digest_writer, level)
                        .map_err(ZTensorError::ZstdCompression)?;
                    Self::write_data(&mut encoder, data, dtype)?;
                    encoder.finish().map_err(ZTensorError::ZstdCompression)?;
                }
                Encoding::Zstd
            }
        };

        // Finalize digest
        let digest = digest_writer.finalize();
        
        // Update offset
        let length = counting_writer.count;
        self.current_offset += length;

        Ok(Component {
            dtype,
            offset: aligned_offset,
            length,
            encoding: stored_encoding,
            digest,
        })
    }

    fn write_data<Output: Write>(
        writer: &mut Output,
        data: &[u8],
        dtype: DType,
    ) -> Result<(), ZTensorError> {
        let is_native_safe = is_little_endian() || !dtype.is_multi_byte();
        
        if is_native_safe {
            writer.write_all(data)?;
        } else {
            // Swap in chunks
            const CHUNK_SIZE: usize = 4096;
            let mut buffer = Vec::with_capacity(CHUNK_SIZE);
            
            // Iterate over chunks of size CHUNK_SIZE
            // Ensure we don't split multi-byte elements
            // Since CHUNK_SIZE=4096 is divisible by 1,2,4,8, we are safe.
            for chunk in data.chunks(CHUNK_SIZE) {
                buffer.clear();
                buffer.extend_from_slice(chunk);
                
                swap_endianness_in_place(&mut buffer, dtype.byte_size());
                
                writer.write_all(&buffer)?;
            }
        }
        Ok(())
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
