//! zTensor file reader.

use byteorder::{LittleEndian, ReadBytesExt};
use half::{bf16, f16};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::ZTensorError;
use crate::models::{Component, CooTensor, CsrTensor, DType, Encoding, Manifest, Object, ALIGNMENT, MAGIC, MAX_MANIFEST_SIZE};
use crate::utils::swap_endianness_in_place;

/// Max object size (32GB).
const MAX_OBJECT_SIZE: u64 = 32 * 1024 * 1024 * 1024;

/// Trait for types that can be safely read from bytes.
pub trait Pod: Sized + Default + Clone {
    const SIZE: usize = std::mem::size_of::<Self>();
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn dtype_matches(dtype: &DType) -> bool;
}

macro_rules! impl_pod {
    ($t:ty, $d:path) => {
        impl Pod for $t {
            fn from_le_bytes(bytes: &[u8]) -> Self {
                <$t>::from_le_bytes(bytes.try_into().expect("Pod byte slice wrong size"))
            }
            fn dtype_matches(dtype: &DType) -> bool {
                dtype == &$d
            }
        }
    };
}

impl_pod!(f64, DType::F64);
impl_pod!(f32, DType::F32);
impl_pod!(i64, DType::I64);
impl_pod!(i32, DType::I32);
impl_pod!(i16, DType::I16);
impl_pod!(u64, DType::U64);
impl_pod!(u32, DType::U32);
impl_pod!(u16, DType::U16);

impl Pod for u8 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bytes[0]
    }
    fn dtype_matches(dtype: &DType) -> bool {
        dtype == &DType::U8
    }
}

impl Pod for i8 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bytes[0] as i8
    }
    fn dtype_matches(dtype: &DType) -> bool {
        dtype == &DType::I8
    }
}

impl Pod for bool {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn dtype_matches(dtype: &DType) -> bool {
        dtype == &DType::Bool
    }
}

impl Pod for f16 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f16::from_le_bytes(bytes.try_into().expect("f16 byte slice wrong size"))
    }
    fn dtype_matches(dtype: &DType) -> bool {
        dtype == &DType::F16
    }
}

impl Pod for bf16 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bf16::from_le_bytes(bytes.try_into().expect("bf16 byte slice wrong size"))
    }
    fn dtype_matches(dtype: &DType) -> bool {
        dtype == &DType::BF16
    }
}

/// Context for error messages.
#[derive(Clone)]
struct ReadContext<'a> {
    object_name: &'a str,
    component_name: &'a str,
}

impl<'a> ReadContext<'a> {
    fn new(object_name: &'a str, component_name: &'a str) -> Self {
        Self { object_name, component_name }
    }

    fn unknown() -> Self {
        Self { object_name: "unknown", component_name: "unknown" }
    }
}

/// Reader for zTensor v1.1 files.
pub struct ZTensorReader<R: Read + Seek> {
    pub reader: R,
    pub manifest: Manifest,
}

impl ZTensorReader<BufReader<File>> {
    /// Opens a zTensor file from path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::open(path)?;
        Self::new(BufReader::new(file))
    }
}

impl ZTensorReader<Cursor<Mmap>> {
    /// Opens a zTensor file using memory mapping.
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::new(Cursor::new(mmap))
    }

    /// Gets a zero-copy reference to an object's data.
    ///
    /// This is only available for:
    /// - Memory-mapped readers
    /// - Dense objects
    /// - Raw encoding (not compressed)
    pub fn get_object_slice(&self, name: &str) -> Result<&[u8], ZTensorError> {
        let obj = self.manifest.objects.get(name)
            .ok_or_else(|| ZTensorError::ObjectNotFound(name.to_string()))?;

        if obj.format != "dense" {
            return Err(ZTensorError::TypeMismatch {
                expected: "dense".to_string(),
                found: obj.format.clone(),
                context: format!("object '{}'", name),
            });
        }

        let component = obj.components.get("data").ok_or_else(|| {
            ZTensorError::InvalidFileStructure(format!(
                "Dense object '{}' missing 'data' component",
                name
            ))
        })?;

        if component.encoding != Encoding::Raw {
            return Err(ZTensorError::Other(format!(
                "Zero-copy not supported for compressed component in '{}'", name
            )));
        }

        // Validate alignment
        if component.offset % ALIGNMENT != 0 {
            return Err(ZTensorError::InvalidAlignment {
                offset: component.offset,
                alignment: ALIGNMENT,
            });
        }

        let mmap = self.reader.get_ref();
        let start = component.offset as usize;
        let end = start + component.length as usize;

        if end > mmap.len() {
            return Err(ZTensorError::InvalidFileStructure(format!(
                "Component '{}' out of bounds (end={} > file_len={})",
                name, end, mmap.len()
            )));
        }

        Ok(&mmap[start..end])
    }

    /// Gets a typed zero-copy reference to an object's data.
    pub fn get_object_slice_as<T: Pod>(&self, name: &str) -> Result<&[T], ZTensorError> {
        let obj = self.manifest.objects.get(name)
            .ok_or_else(|| ZTensorError::ObjectNotFound(name.to_string()))?;
        
        let component = obj.components.get("data").ok_or_else(|| {
            ZTensorError::InvalidFileStructure(format!("Missing 'data' component for {}", name))
        })?;

        if !T::dtype_matches(&component.dtype) {
             return Err(ZTensorError::TypeMismatch {
                expected: component.dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }

        let bytes = self.get_object_slice(name)?;
        
        // Safety checks for casting
        if bytes.len() % T::SIZE != 0 {
            return Err(ZTensorError::InvalidFileStructure(format!(
                "Byte length {} is not a multiple of type size {}",
                bytes.len(), T::SIZE
            )));
        }
        
        let ptr = bytes.as_ptr();
        if (ptr as usize) % std::mem::align_of::<T>() != 0 {
            return Err(ZTensorError::Other(format!(
                "Memory not aligned for type {}", std::any::type_name::<T>()
            )));
        }

        let len = bytes.len() / T::SIZE;
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const T, len) })
    }
}

impl<R: Read + Seek> ZTensorReader<R> {
    /// Creates a new reader and parses the manifest.
    ///
    /// Parsing algorithm per spec §4:
    /// 1. Read Footer: EOF - 16
    /// 2. Verify Magic: last 8 bytes == ZTEN1000
    /// 3. Read Size: first 8 bytes as u64 LE
    /// 4. Safety Check: abort if > 1GB
    /// 5. Read Manifest: EOF - 16 - manifest_size
    /// 6. Decode CBOR
    pub fn new(mut reader: R) -> Result<Self, ZTensorError> {
        // Verify header magic
        let mut header_magic = [0u8; 8];
        reader.read_exact(&mut header_magic)?;
        if header_magic != *MAGIC {
            return Err(ZTensorError::InvalidMagicNumber {
                found: header_magic.to_vec(),
            });
        }

        // Read footer (last 16 bytes)
        reader.seek(SeekFrom::End(-16))?;
        let manifest_size = reader.read_u64::<LittleEndian>()?;
        
        let mut footer_magic = [0u8; 8];
        reader.read_exact(&mut footer_magic)?;
        if footer_magic != *MAGIC {
            return Err(ZTensorError::InvalidMagicNumber {
                found: footer_magic.to_vec(),
            });
        }

        // Safety check: manifest size limit
        if manifest_size > MAX_MANIFEST_SIZE {
            return Err(ZTensorError::ManifestTooLarge { size: manifest_size });
        }

        // Read and decode manifest
        reader.seek(SeekFrom::End(-16 - manifest_size as i64))?;
        let mut cbor_buf = vec![0u8; manifest_size as usize];
        reader.read_exact(&mut cbor_buf)?;

        let manifest: Manifest = serde_cbor::from_slice(&cbor_buf)
            .map_err(ZTensorError::CborDeserialize)?;

        Ok(Self { reader, manifest })
    }

    /// Lists all objects in the file.
    pub fn list_objects(&self) -> &BTreeMap<String, Object> {
        &self.manifest.objects
    }

    /// Gets metadata for an object by name.
    pub fn get_object(&self, name: &str) -> Option<&Object> {
        self.manifest.objects.get(name)
    }

    // =========================================================================
    // COMPONENT READING
    // =========================================================================

    fn read_component_into(
        &mut self,
        component: &Component,
        dst: &mut [u8],
        ctx: &ReadContext,
        verify_checksum: bool,
    ) -> Result<(), ZTensorError> {
        if component.offset % ALIGNMENT != 0 {
            return Err(ZTensorError::InvalidAlignment {
                offset: component.offset,
                alignment: ALIGNMENT,
            });
        }

        self.reader.seek(SeekFrom::Start(component.offset))?;

        match component.encoding {
            Encoding::Zstd => {
                let mut compressed = vec![0u8; component.length as usize];
                self.reader.read_exact(&mut compressed)?;

                if verify_checksum {
                    if let Some(ref digest) = component.digest {
                        Self::verify_checksum(digest, &compressed, ctx)?;
                    }
                }

                zstd::stream::copy_decode(Cursor::new(compressed), &mut *dst)
                    .map_err(ZTensorError::ZstdDecompression)?;
            }
            Encoding::Raw => {
                if dst.len() as u64 != component.length {
                    return Err(ZTensorError::InvalidFileStructure(format!(
                        "Component length mismatch for {}/{}: expected {}, got {}",
                        ctx.object_name, ctx.component_name, component.length, dst.len()
                    )));
                }
                self.reader.read_exact(dst)?;

                if verify_checksum {
                    if let Some(ref digest) = component.digest {
                        Self::verify_checksum(digest, dst, ctx)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Reads raw component data.
    pub fn read_component(&mut self, component: &Component) -> Result<Vec<u8>, ZTensorError> {
        let ctx = ReadContext::unknown();

        match component.encoding {
            Encoding::Zstd => {
                self.reader.seek(SeekFrom::Start(component.offset))?;
                let mut compressed = vec![0u8; component.length as usize];
                self.reader.read_exact(&mut compressed)?;

                if let Some(ref digest) = component.digest {
                    Self::verify_checksum(digest, &compressed, &ctx)?;
                }

                let mut decompressed = Vec::new();
                zstd::stream::copy_decode(Cursor::new(compressed), &mut decompressed)
                    .map_err(ZTensorError::ZstdDecompression)?;
                Ok(decompressed)
            }
            Encoding::Raw => {
                let mut data = vec![0u8; component.length as usize];
                self.read_component_into(component, &mut data, &ctx, true)?;
                Ok(data)
            }
        }
    }

    fn verify_checksum(digest: &str, data: &[u8], ctx: &ReadContext) -> Result<(), ZTensorError> {
        if digest.starts_with("crc32c:0x") || digest.starts_with("crc32c:0X") {
            let expected_hex = &digest[9..];
            let expected = u32::from_str_radix(expected_hex, 16).map_err(|_| {
                ZTensorError::ChecksumFormatError(format!("Invalid CRC32C hex: {}", expected_hex))
            })?;
            let calculated = crc32c::crc32c(data);
            if calculated != expected {
                return Err(ZTensorError::ChecksumMismatch {
                    object_name: ctx.object_name.to_string(),
                    component_name: ctx.component_name.to_string(),
                    expected: format!("0x{:08X}", expected),
                    calculated: format!("0x{:08X}", calculated),
                });
            }
        } else if digest.starts_with("sha256:") {
            let expected_hex = &digest[7..];
            let calculated = crate::utils::sha256_hex(data);
            if calculated != expected_hex.to_lowercase() {
                return Err(ZTensorError::ChecksumMismatch {
                    object_name: ctx.object_name.to_string(),
                    component_name: ctx.component_name.to_string(),
                    expected: expected_hex.to_string(),
                    calculated,
                });
            }
        }
        Ok(())
    }

    // =========================================================================
    // DENSE OBJECT READING
    // =========================================================================

    fn read_dense_impl(
        &mut self,
        name: &str,
        obj: &Object,
        verify_checksum: bool,
    ) -> Result<Vec<u8>, ZTensorError> {
        if obj.format != "dense" {
            return Err(ZTensorError::TypeMismatch {
                expected: "dense".to_string(),
                found: obj.format.clone(),
                context: format!("object '{}'", name),
            });
        }

        let component = obj.components.get("data").ok_or_else(|| {
            ZTensorError::InvalidFileStructure(format!(
                "Dense object '{}' missing 'data' component",
                name
            ))
        })?;

        let num_elements = obj.num_elements();
        let expected_size = num_elements * component.dtype.byte_size() as u64;

        if expected_size > MAX_OBJECT_SIZE {
            return Err(ZTensorError::ObjectTooLarge {
                size: expected_size,
                limit: MAX_OBJECT_SIZE,
            });
        }

        let mut data = vec![0u8; expected_size as usize];
        let ctx = ReadContext::new(name, "data");
        self.read_component_into(component, &mut data, &ctx, verify_checksum)?;

        if cfg!(target_endian = "big") && component.dtype.is_multi_byte() {
            swap_endianness_in_place(&mut data, component.dtype.byte_size());
        }

        Ok(data)
    }

    /// Reads raw byte data of a dense object.
    pub fn read_object(&mut self, name: &str, verify_checksum: bool) -> Result<Vec<u8>, ZTensorError> {
        let obj = self.manifest.objects.get(name)
            .ok_or_else(|| ZTensorError::ObjectNotFound(name.to_string()))?
            .clone();
        self.read_dense_impl(name, &obj, verify_checksum)
    }

    /// Reads multiple objects in batch.
    pub fn read_objects(&mut self, names: &[&str], verify_checksum: bool) -> Result<Vec<Vec<u8>>, ZTensorError> {
        let mut results = Vec::with_capacity(names.len());
        for name in names {
            results.push(self.read_object(name, verify_checksum)?);
        }
        Ok(results)
    }

    // =========================================================================
    // TYPED READING
    // =========================================================================

    /// Reads object data as a typed vector.
    pub fn read_object_as<T: Pod>(&mut self, name: &str) -> Result<Vec<T>, ZTensorError> {
        let obj = self.manifest.objects.get(name)
            .ok_or_else(|| ZTensorError::ObjectNotFound(name.to_string()))?;

        let component = obj.components.get("data").ok_or_else(|| {
            ZTensorError::InvalidFileStructure(format!("Missing 'data' component for {}", name))
        })?;

        if !T::dtype_matches(&component.dtype) {
            return Err(ZTensorError::TypeMismatch {
                expected: component.dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }

        if obj.format != "dense" {
            return Err(ZTensorError::TypeMismatch {
                expected: "dense".to_string(),
                found: obj.format.clone(),
                context: name.to_string(),
            });
        }

        let component = component.clone();
        let num_elements = if obj.shape.is_empty() { 1 } else { obj.shape.iter().product::<u64>() as usize };

        let byte_len = num_elements * T::SIZE;

        if byte_len as u64 > MAX_OBJECT_SIZE {
            return Err(ZTensorError::ObjectTooLarge {
                size: byte_len as u64,
                limit: MAX_OBJECT_SIZE,
            });
        }

        let mut typed_data = vec![T::default(); num_elements];
        let output_slice = unsafe {
            std::slice::from_raw_parts_mut(typed_data.as_mut_ptr() as *mut u8, byte_len)
        };

        let ctx = ReadContext::new(name, "data");
        self.read_component_into(&component, output_slice, &ctx, true)?;

        Ok(typed_data)
    }

    // =========================================================================
    // SPARSE READING
    // =========================================================================

    fn read_component_as<T: Pod>(
        &mut self,
        component: &Component,
        ctx: &ReadContext,
    ) -> Result<Vec<T>, ZTensorError> {
        match component.encoding {
            Encoding::Zstd => {
                let bytes = self.read_component(component)?;
                if bytes.len() as u64 > MAX_OBJECT_SIZE {
                    return Err(ZTensorError::ObjectTooLarge {
                        size: bytes.len() as u64,
                        limit: MAX_OBJECT_SIZE,
                    });
                }
                let num_elements = bytes.len() / T::SIZE;
                let mut values = vec![T::default(); num_elements];
                unsafe {
                    std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, bytes.len())
                        .copy_from_slice(&bytes);
                }
                Ok(values)
            }
            Encoding::Raw => {
                if component.length > MAX_OBJECT_SIZE {
                    return Err(ZTensorError::ObjectTooLarge {
                        size: component.length,
                        limit: MAX_OBJECT_SIZE,
                    });
                }
                let num_elements = component.length as usize / T::SIZE;
                let mut values = vec![T::default(); num_elements];
                let byte_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        component.length as usize,
                    )
                };
                self.read_component_into(component, byte_slice, ctx, true)?;
                Ok(values)
            }
        }
    }

    fn read_u64_component(
        &mut self,
        component: &Component,
        ctx: &ReadContext,
    ) -> Result<Vec<u64>, ZTensorError> {
        let bytes = match component.encoding {
            Encoding::Zstd => self.read_component(component)?,
            Encoding::Raw => {
                if component.length > MAX_OBJECT_SIZE {
                    return Err(ZTensorError::ObjectTooLarge {
                        size: component.length,
                        limit: MAX_OBJECT_SIZE,
                    });
                }
                let mut buf = vec![0u8; component.length as usize];
                self.read_component_into(component, &mut buf, ctx, true)?;
                buf
            }
        };



        Ok(bytes
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    /// Reads a COO sparse object.
    pub fn read_coo_object<T: Pod>(&mut self, name: &str) -> Result<CooTensor<T>, ZTensorError> {
        let obj = self.get_object(name)
            .ok_or_else(|| ZTensorError::ObjectNotFound(name.to_string()))?;

        if obj.format != "sparse_coo" {
            return Err(ZTensorError::TypeMismatch {
                expected: "sparse_coo".to_string(),
                found: obj.format.clone(),
                context: format!("object '{}'", name),
            });
        }

        let shape = obj.shape.clone();
        let val_comp = obj.components.get("values")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'values'".to_string()))?.clone();
        let coords_comp = obj.components.get("coords")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'coords'".to_string()))?.clone();

        let val_ctx = ReadContext::new(name, "values");
        let mut values: Vec<T> = self.read_component_as(&val_comp, &val_ctx)?;

        if cfg!(target_endian = "big") && val_comp.dtype.is_multi_byte() {
            let byte_len = values.len() * T::SIZE;
            let val_slice = unsafe {
                std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len)
            };
            swap_endianness_in_place(val_slice, val_comp.dtype.byte_size());
        }

        let coords_ctx = ReadContext::new(name, "coords");
        let all_coords = self.read_u64_component(&coords_comp, &coords_ctx)?;

        let nnz = values.len();
        let ndim = shape.len();

        if all_coords.len() != nnz * ndim {
            return Err(ZTensorError::DataConversionError(
                "COO coords size mismatch".to_string(),
            ));
        }

        let mut indices = Vec::with_capacity(nnz);
        for i in 0..nnz {
            let mut idx = Vec::with_capacity(ndim);
            for d in 0..ndim {
                idx.push(all_coords[d * nnz + i]);
            }
            indices.push(idx);
        }

        Ok(CooTensor { shape, indices, values })
    }

    /// Reads a CSR sparse object.
    pub fn read_csr_object<T: Pod>(&mut self, name: &str) -> Result<CsrTensor<T>, ZTensorError> {
        let obj = self.get_object(name)
            .ok_or_else(|| ZTensorError::ObjectNotFound(name.to_string()))?;

        if obj.format != "sparse_csr" {
            return Err(ZTensorError::TypeMismatch {
                expected: "sparse_csr".to_string(),
                found: obj.format.clone(),
                context: format!("object '{}'", name),
            });
        }

        let shape = obj.shape.clone();
        let val_comp = obj.components.get("values")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'values'".to_string()))?.clone();
        let idx_comp = obj.components.get("indices")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'indices'".to_string()))?.clone();
        let ptr_comp = obj.components.get("indptr")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'indptr'".to_string()))?.clone();

        let val_ctx = ReadContext::new(name, "values");
        let mut values: Vec<T> = self.read_component_as(&val_comp, &val_ctx)?;

        if cfg!(target_endian = "big") && val_comp.dtype.is_multi_byte() {
            let byte_len = values.len() * T::SIZE;
            let val_slice = unsafe {
                std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len)
            };
            swap_endianness_in_place(val_slice, val_comp.dtype.byte_size());
        }

        let idx_ctx = ReadContext::new(name, "indices");
        let indices = self.read_u64_component(&idx_comp, &idx_ctx)?;

        let ptr_ctx = ReadContext::new(name, "indptr");
        let indptr = self.read_u64_component(&ptr_comp, &ptr_ctx)?;

        Ok(CsrTensor { shape, indptr, indices, values })
    }
}
