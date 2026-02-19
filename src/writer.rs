//! zTensor file writer.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::Error;
use crate::models::{Checksum, Component, DType, Encoding, Format, Manifest, Object, MAGIC};
use crate::reader::{Tensor, TensorElement};
use crate::utils::{align_offset, is_little_endian, swap_endianness_in_place, DigestWriter};

/// Zero-filled padding buffer (avoids heap allocation per tensor).
const ZERO_PAD: [u8; 64] = [0u8; 64];

/// Compression settings for writing.
///
/// # Examples
///
/// ```
/// use ztensor::writer::Compression;
///
/// let raw = Compression::Raw;
/// let fast = Compression::Zstd(1);
/// let balanced = Compression::Zstd(3);
/// let high = Compression::Zstd(19);
/// ```
#[derive(Debug, Clone, Copy)]
pub enum Compression {
    /// No compression.
    Raw,
    /// Zstd compression with the specified level (1-22, 0 means library default).
    Zstd(i32),
}

/// Writer for zTensor files.
///
/// Writes tensors sequentially, then finalizes the file with a CBOR manifest
/// and footer. Tensors are 64-byte aligned for zero-copy reads.
///
/// # Examples
///
/// ```no_run
/// use ztensor::{Writer, Checksum};
/// use ztensor::writer::Compression;
///
/// let mut writer = Writer::create("model.zt")?;
///
/// let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
/// writer.add("weights", &[2, 2], &weights)?;
///
/// writer.add_with("bias", &[4], &weights)
///     .compress(Compression::Zstd(3))
///     .checksum(Checksum::Crc32c)
///     .write()?;
///
/// let total_bytes = writer.finish()?;
/// # Ok::<(), ztensor::Error>(())
/// ```
pub struct Writer<W: Write + Seek> {
    writer: W,
    manifest: Manifest,
    current_offset: u64,
}

impl Writer<BufWriter<File>> {
    /// Creates a new writer for the given file path.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::Writer;
    ///
    /// let mut writer = Writer::create("output.zt")?;
    /// # writer.finish()?;
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn create(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::create(path)?;
        Self::new(BufWriter::with_capacity(256 * 1024, file))
    }

    /// Opens an existing `.zt` file for appending new tensors.
    ///
    /// Reads the existing manifest, truncates the manifest/footer, and
    /// positions the writer to append new tensor data. Existing tensors
    /// and their data are preserved unchanged.
    ///
    /// After appending tensors with [`add`](Writer::add) or
    /// [`add_with`](Writer::add_with), call [`finish`](Writer::finish)
    /// to write the updated manifest and footer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::Writer;
    ///
    /// let mut writer = Writer::append("model.zt")?;
    /// writer.add("new_tensor", &[4], &[1.0f32, 2.0, 3.0, 4.0])?;
    /// writer.finish()?;
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn append(path: impl AsRef<Path>) -> Result<Self, Error> {
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Read footer: last 16 bytes = [manifest_size: u64 LE] [MAGIC: 8B]
        file.seek(SeekFrom::End(-16))?;
        let mut size_buf = [0u8; 8];
        file.read_exact(&mut size_buf)?;
        let manifest_size = u64::from_le_bytes(size_buf);

        let mut footer_magic = [0u8; 8];
        file.read_exact(&mut footer_magic)?;
        if footer_magic != *MAGIC {
            return Err(Error::InvalidMagicNumber {
                found: footer_magic.to_vec(),
            });
        }

        // Read manifest
        let file_size = file.seek(SeekFrom::End(0))?;
        let manifest_start = file_size - 16 - manifest_size;
        file.seek(SeekFrom::Start(manifest_start))?;
        let mut cbor_buf = vec![0u8; manifest_size as usize];
        file.read_exact(&mut cbor_buf)?;

        let manifest: Manifest = ciborium::from_reader(std::io::Cursor::new(&cbor_buf))
            .map_err(Error::CborDeserialize)?;

        // Truncate file at manifest start (remove old manifest+footer)
        file.set_len(manifest_start)?;
        file.seek(SeekFrom::Start(manifest_start))?;

        Ok(Self {
            writer: BufWriter::with_capacity(256 * 1024, file),
            manifest,
            current_offset: manifest_start,
        })
    }
}

impl<W: Write + Seek> Writer<W> {
    /// Creates a new writer from a Write + Seek source.
    pub fn new(mut writer: W) -> Result<Self, Error> {
        writer.write_all(MAGIC)?;
        Ok(Self {
            writer,
            manifest: Manifest::default(),
            current_offset: MAGIC.len() as u64,
        })
    }

    /// Sets global attributes on the manifest.
    pub fn set_attributes(&mut self, attrs: BTreeMap<String, ciborium::Value>) {
        self.manifest.attributes = Some(attrs);
    }

    /// Adds a dense object from raw bytes (FFI/unsafe usage).
    ///
    /// The caller must ensure `data` contains valid LE bytes for the given `dtype`.
    /// Endianness swapping will be performed if `dtype` is multi-byte and host is BE.
    pub fn add_bytes(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        compression: Compression,
        data: &[u8],
        checksum: Checksum,
    ) -> Result<(), Error> {
        let num_elements: u64 = if shape.is_empty() {
            1
        } else {
            shape.iter().try_fold(1u64, |acc, &d| {
                acc.checked_mul(d).ok_or_else(|| {
                    Error::InvalidFileStructure("Shape product overflows u64".into())
                })
            })?
        };
        let expected_size = num_elements
            .checked_mul(dtype.byte_size() as u64)
            .ok_or_else(|| Error::InvalidFileStructure("Tensor byte size overflows u64".into()))?;

        if data.len() as u64 != expected_size {
            return Err(Error::InconsistentDataSize {
                expected: expected_size,
                found: data.len() as u64,
            });
        }

        if self.manifest.objects.contains_key(name) {
            return Err(Error::Other(format!("Duplicate tensor name: '{}'", name)));
        }

        let component = self.write_component(data, dtype, compression, checksum)?;
        let mut components = BTreeMap::new();
        components.insert("data".to_string(), component);

        let obj = Object {
            shape,
            format: Format::Dense,
            attributes: None,
            components,
        };

        self.manifest.objects.insert(name.to_string(), obj);
        Ok(())
    }

    /// Adds a dense tensor to the file.
    ///
    /// DType is inferred from `T`. Data is stored uncompressed with no checksum.
    /// Use [`add_with`](Writer::add_with) for compression/checksum control.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::Writer;
    ///
    /// let mut writer = Writer::create("model.zt")?;
    /// writer.add("weights", &[3, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// writer.add("ids", &[4], &[0u64, 1, 2, 3])?;
    /// writer.finish()?;
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn add<T: TensorElement + bytemuck::Pod>(
        &mut self,
        name: &str,
        shape: &[u64],
        data: &[T],
    ) -> Result<(), Error> {
        let bytes = bytemuck::cast_slice(data);
        self.add_bytes(
            name,
            shape.to_vec(),
            T::DTYPE,
            Compression::Raw,
            bytes,
            Checksum::None,
        )
    }

    /// Adds a dense tensor with builder options for compression and checksum.
    ///
    /// Returns an [`AddBuilder`] for chaining `.compress()` and `.checksum()`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::{Writer, Checksum};
    /// use ztensor::writer::Compression;
    ///
    /// let data: Vec<f32> = vec![0.0; 1024];
    /// let mut writer = Writer::create("model.zt")?;
    /// writer.add_with("weights", &[32, 32], &data)
    ///     .compress(Compression::Zstd(3))
    ///     .checksum(Checksum::Sha256)
    ///     .write()?;
    /// writer.finish()?;
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn add_with<'a, T: TensorElement + bytemuck::Pod>(
        &'a mut self,
        name: &str,
        shape: &[u64],
        data: &'a [T],
    ) -> AddBuilder<'a, W, T> {
        AddBuilder {
            writer: self,
            name: name.to_string(),
            shape: shape.to_vec(),
            data,
            compression: Compression::Raw,
            checksum: Checksum::None,
        }
    }

    /// Adds a CSR sparse object from raw byte buffers.
    ///
    /// For a typed API, use [`add_csr`](Writer::add_csr) instead.
    #[allow(clippy::too_many_arguments)]
    pub fn add_csr_bytes(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: &[u8],
        indices: &[u64],
        indptr: &[u64],
        compression: Compression,
        checksum: Checksum,
    ) -> Result<(), Error> {
        if self.manifest.objects.contains_key(name) {
            return Err(Error::Other(format!("Duplicate tensor name: '{}'", name)));
        }

        let indices_bytes = bytemuck::cast_slice(indices);
        let indptr_bytes = bytemuck::cast_slice(indptr);

        let values_comp = self.write_component(values, dtype, compression, checksum)?;
        let indices_comp =
            self.write_component(indices_bytes, DType::U64, compression, checksum)?;
        let indptr_comp = self.write_component(indptr_bytes, DType::U64, compression, checksum)?;

        let mut components = BTreeMap::new();
        components.insert("values".to_string(), values_comp);
        components.insert("indices".to_string(), indices_comp);
        components.insert("indptr".to_string(), indptr_comp);

        let obj = Object {
            shape,
            format: Format::SparseCsr,
            attributes: None,
            components,
        };

        self.manifest.objects.insert(name.to_string(), obj);
        Ok(())
    }

    /// Adds a CSR (Compressed Sparse Row) object.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::{Writer, Checksum, DType};
    /// use ztensor::writer::Compression;
    ///
    /// // 3x3 identity matrix in CSR format
    /// let values = vec![1.0f32, 1.0, 1.0];
    /// let indices = vec![0u64, 1, 2];      // column indices
    /// let indptr = vec![0u64, 1, 2, 3];    // row pointers
    ///
    /// let mut writer = Writer::create("sparse.zt")?;
    /// writer.add_csr(
    ///     "identity", vec![3, 3], DType::F32,
    ///     &values, &indices, &indptr,
    ///     Compression::Raw, Checksum::None,
    /// )?;
    /// writer.finish()?;
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn add_csr<T: TensorElement + bytemuck::Pod>(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: &[T],
        indices: &[u64],
        indptr: &[u64],
        compression: Compression,
        checksum: Checksum,
    ) -> Result<(), Error> {
        let values_bytes = bytemuck::cast_slice(values);
        self.add_csr_bytes(
            name,
            shape,
            dtype,
            values_bytes,
            indices,
            indptr,
            compression,
            checksum,
        )
    }

    /// Adds a COO sparse object from raw byte buffers.
    ///
    /// For a typed API, use [`add_coo`](Writer::add_coo) instead.
    #[allow(clippy::too_many_arguments)]
    pub fn add_coo_bytes(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: &[u8],
        coords: &[u64],
        compression: Compression,
        checksum: Checksum,
    ) -> Result<(), Error> {
        if self.manifest.objects.contains_key(name) {
            return Err(Error::Other(format!("Duplicate tensor name: '{}'", name)));
        }

        let coords_bytes = bytemuck::cast_slice(coords);

        let values_comp = self.write_component(values, dtype, compression, checksum)?;
        let coords_comp = self.write_component(coords_bytes, DType::U64, compression, checksum)?;

        let mut components = BTreeMap::new();
        components.insert("values".to_string(), values_comp);
        components.insert("coords".to_string(), coords_comp);

        let obj = Object {
            shape,
            format: Format::SparseCoo,
            attributes: None,
            components,
        };

        self.manifest.objects.insert(name.to_string(), obj);
        Ok(())
    }

    /// Adds a COO (Coordinate list) sparse object.
    ///
    /// Coordinates are stored in SoA (struct-of-arrays) layout: all row indices
    /// first, then all column indices, etc.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::{Writer, Checksum, DType};
    /// use ztensor::writer::Compression;
    ///
    /// // 3x3 matrix with values at (0,1) and (2,0)
    /// let values = vec![5.0f32, 3.0];
    /// // SoA coords: [row0, row1, col0, col1]
    /// let coords = vec![0u64, 2, 1, 0];
    ///
    /// let mut writer = Writer::create("sparse.zt")?;
    /// writer.add_coo(
    ///     "matrix", vec![3, 3], DType::F32,
    ///     &values, &coords,
    ///     Compression::Raw, Checksum::None,
    /// )?;
    /// writer.finish()?;
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn add_coo<T: TensorElement + bytemuck::Pod>(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: DType,
        values: &[T],
        coords: &[u64],
        compression: Compression,
        checksum: Checksum,
    ) -> Result<(), Error> {
        let values_bytes = bytemuck::cast_slice(values);
        self.add_coo_bytes(
            name,
            shape,
            dtype,
            values_bytes,
            coords,
            compression,
            checksum,
        )
    }

    /// Adds a generic object with arbitrary components.
    ///
    /// This is the general-purpose method that supports any format (dense, sparse,
    /// quantized, custom). Each component is specified as `(name, dtype, logical_type, bytes)`.
    ///
    /// # Arguments
    /// * `name` — Object name (must be unique).
    /// * `shape` — Logical shape dimensions.
    /// * `format` — Layout format (dense, sparse_csr, etc.).
    /// * `component_data` — Slice of `(component_name, dtype, logical_type, raw_bytes)`.
    /// * `attributes` — Optional per-object attributes.
    /// * `compression` — Compression method for all components.
    /// * `checksum` — Checksum algorithm for all components.
    pub fn add_object(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        format: Format,
        component_data: &[(&str, DType, Option<&str>, &[u8])],
        attributes: Option<BTreeMap<String, ciborium::Value>>,
        compression: Compression,
        checksum: Checksum,
    ) -> Result<(), Error> {
        if self.manifest.objects.contains_key(name) {
            return Err(Error::Other(format!("Duplicate tensor name: '{}'", name)));
        }

        let mut components = BTreeMap::new();
        for (comp_name, dtype, logical_type, bytes) in component_data {
            let mut component = self.write_component(bytes, *dtype, compression, checksum)?;
            if let Some(lt) = logical_type {
                component.r#type = Some(lt.to_string());
            }
            components.insert(comp_name.to_string(), component);
        }

        let obj = Object {
            shape,
            format,
            attributes,
            components,
        };

        self.manifest.objects.insert(name.to_string(), obj);
        Ok(())
    }

    /// Writes a [`Tensor`] object (from [`TensorReader::read_object`]) to the file.
    ///
    /// This is the write counterpart of `read_object()`, enabling generic copy
    /// of any tensor (dense, sparse, quantized, custom) between files.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::{TensorReader, Writer, Compression, Checksum};
    ///
    /// let reader = ztensor::open("input.zt")?;
    /// let mut writer = Writer::create("output.zt")?;
    /// for name in reader.keys() {
    ///     let tensor = reader.read_object(name)?;
    ///     writer.write_object(name, &tensor, Compression::Raw, Checksum::None)?;
    /// }
    /// writer.finish()?;
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn write_object(
        &mut self,
        name: &str,
        tensor: &Tensor<'_>,
        compression: Compression,
        checksum: Checksum,
    ) -> Result<(), Error> {
        let comp_data: Vec<(&str, DType, Option<&str>, &[u8])> = tensor
            .components
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str(),
                    v.dtype,
                    v.logical_type.as_deref(),
                    v.data.as_slice(),
                )
            })
            .collect();
        self.add_object(
            name,
            tensor.shape.clone(),
            tensor.format.clone(),
            &comp_data,
            tensor.attributes.clone(),
            compression,
            checksum,
        )
    }

    fn write_component(
        &mut self,
        data: &[u8],
        dtype: DType,
        compression: Compression,
        checksum: Checksum,
    ) -> Result<Component, Error> {
        // 1. Align
        let (aligned_offset, padding) = align_offset(self.current_offset);
        if padding > 0 {
            self.writer.write_all(&ZERO_PAD[..padding as usize])?;
        }
        self.current_offset = aligned_offset;

        // 2. Setup digest writer (also counts bytes)
        let mut digest_writer = DigestWriter::new(&mut self.writer, checksum);

        let stored_encoding = match compression {
            Compression::Raw => {
                Self::write_data(&mut digest_writer, data, dtype)?;
                Encoding::Raw
            }
            Compression::Zstd(level) => {
                {
                    let mut encoder = zstd::stream::write::Encoder::new(&mut digest_writer, level)
                        .map_err(Error::ZstdCompression)?;
                    Self::write_data(&mut encoder, data, dtype)?;
                    encoder.finish().map_err(Error::ZstdCompression)?;
                }
                Encoding::Zstd
            }
        };

        // Finalize digest and get byte count
        let length = digest_writer.bytes_written;
        let digest = digest_writer.finalize();
        self.current_offset += length;

        Ok(Component {
            dtype,
            r#type: None,
            offset: aligned_offset,
            length,
            uncompressed_length: match stored_encoding {
                Encoding::Zstd => Some(data.len() as u64),
                Encoding::Raw => None,
            },
            encoding: stored_encoding,
            digest,
        })
    }

    fn write_data<Output: Write>(
        writer: &mut Output,
        data: &[u8],
        dtype: DType,
    ) -> Result<(), Error> {
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

    /// Finalizes the file by writing the CBOR manifest and footer.
    ///
    /// Returns the total file size in bytes. Must be called after all tensors
    /// have been added.
    ///
    /// File layout: `[MAGIC 8B] [BLOBS...] [CBOR MANIFEST] [MANIFEST SIZE 8B] [MAGIC 8B]`
    pub fn finish(mut self) -> Result<u64, Error> {
        let mut cbor = Vec::new();
        ciborium::into_writer(&self.manifest, &mut cbor).map_err(Error::CborSerialize)?;

        self.writer.write_all(&cbor)?;

        let cbor_size = cbor.len() as u64;
        self.writer.write_all(&cbor_size.to_le_bytes())?;

        // Write footer magic
        self.writer.write_all(MAGIC)?;

        self.writer.flush()?;

        Ok(self.current_offset + cbor_size + 8 + 8)
    }
}

/// Builder for adding a tensor with compression and checksum options.
///
/// Created by [`Writer::add_with`]. Call [`.write()`](AddBuilder::write) to
/// finalize.
///
/// # Examples
///
/// ```no_run
/// use ztensor::{Writer, Checksum};
/// use ztensor::writer::Compression;
///
/// let data = vec![1.0f32; 100];
/// let mut writer = Writer::create("model.zt")?;
/// writer.add_with("tensor", &[10, 10], &data)
///     .compress(Compression::Zstd(3))
///     .checksum(Checksum::Crc32c)
///     .write()?;
/// # writer.finish()?;
/// # Ok::<(), ztensor::Error>(())
/// ```
pub struct AddBuilder<'a, W: Write + Seek, T: TensorElement + bytemuck::Pod> {
    writer: &'a mut Writer<W>,
    name: String,
    shape: Vec<u64>,
    data: &'a [T],
    compression: Compression,
    checksum: Checksum,
}

impl<'a, W: Write + Seek, T: TensorElement + bytemuck::Pod> AddBuilder<'a, W, T> {
    /// Sets the compression method.
    pub fn compress(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Sets the checksum algorithm.
    pub fn checksum(mut self, checksum: Checksum) -> Self {
        self.checksum = checksum;
        self
    }

    /// Writes the tensor to the file.
    pub fn write(self) -> Result<(), Error> {
        let bytes = bytemuck::cast_slice(self.data);
        self.writer.add_bytes(
            &self.name,
            self.shape,
            T::DTYPE,
            self.compression,
            bytes,
            self.checksum,
        )
    }
}
