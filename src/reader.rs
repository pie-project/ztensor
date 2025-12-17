use byteorder::{LittleEndian, ReadBytesExt};
use half::{bf16, f16};
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use memmap2::Mmap;

use crate::error::ZTensorError;
use crate::models::{
    ALIGNMENT, Component, DType, Encoding, Manifest, Tensor, MAGIC_NUMBER,
};
use crate::utils::swap_endianness_in_place;

/// Trait for Plain Old Data types that can be safely created from byte sequences.
pub trait Pod: Sized + Default + Clone {
    const SIZE: usize = std::mem::size_of::<Self>();
    fn from_le_bytes(bytes: &[u8]) -> Self;
    fn dtype_matches(dtype: &DType) -> bool;
}

// Implement Pod for common types
macro_rules! impl_pod {
    ($t:ty, $d:path, $from_le:ident) => {
        impl Pod for $t {
            fn from_le_bytes(bytes: &[u8]) -> Self {
                <$t>::$from_le(bytes.try_into().expect("Pod byte slice wrong size"))
            }
            fn dtype_matches(dtype: &DType) -> bool {
                dtype == &$d
            }
        }
    };
}

impl_pod!(f64, DType::Float64, from_le_bytes);
impl_pod!(f32, DType::Float32, from_le_bytes);
impl_pod!(i64, DType::Int64, from_le_bytes);
impl_pod!(i32, DType::Int32, from_le_bytes);
impl_pod!(i16, DType::Int16, from_le_bytes);
impl_pod!(u64, DType::Uint64, from_le_bytes);
impl_pod!(u32, DType::Uint32, from_le_bytes);
impl_pod!(u16, DType::Uint16, from_le_bytes);

macro_rules! impl_pod_byte {
    ($t:ty, $d:path) => {
        impl Pod for $t {
            fn from_le_bytes(bytes: &[u8]) -> Self {
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
        dtype == &DType::Float16
    }
}

impl Pod for bf16 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bf16::from_le_bytes(bytes.try_into().expect("bf16 byte slice wrong size"))
    }
    fn dtype_matches(dtype: &DType) -> bool {
        dtype == &DType::BFloat16
    }
}

/// Context for error messages when reading components.
#[derive(Clone)]
struct ReadContext<'a> {
    tensor_name: &'a str,
    component_name: &'a str,
}

impl<'a> ReadContext<'a> {
    fn new(tensor_name: &'a str, component_name: &'a str) -> Self {
        Self { tensor_name, component_name }
    }
    
    fn unknown() -> Self {
        Self { tensor_name: "unknown", component_name: "unknown" }
    }
}

/// Reads zTensor files (v1.0).
pub struct ZTensorReader<R: Read + Seek> {
    pub reader: R,
    pub manifest: Manifest,
}

impl ZTensorReader<BufReader<File>> {
    /// Opens a zTensor file from the given path and parses its metadata.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::open(path)?;
        Self::new(BufReader::new(file))
    }
}

impl ZTensorReader<Cursor<Mmap>> {
    /// Opens a zTensor file using memory mapping.
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self, ZTensorError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| ZTensorError::Io(e))? };
        Self::new(Cursor::new(mmap))
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
        let manifest_size = reader.read_u64::<LittleEndian>()?;
        let file_size = reader.seek(SeekFrom::End(0))?;
        
        if manifest_size > file_size.saturating_sub(MAGIC_NUMBER.len() as u64 + 8) {
            return Err(ZTensorError::InvalidFileStructure(format!(
                "Manifest size {} is too large for file size {}",
                manifest_size, file_size
            )));
        }

        reader.seek(SeekFrom::End(-8 - (manifest_size as i64)))?;
        let mut cbor_buf = vec![0u8; manifest_size as usize];
        reader.read_exact(&mut cbor_buf)?;

        let manifest: Manifest =
            serde_cbor::from_slice(&cbor_buf).map_err(ZTensorError::CborDeserialize)?;

        Ok(Self { reader, manifest })
    }

    /// Lists all tensors in the file.
    pub fn list_tensors(&self) -> &std::collections::BTreeMap<String, Tensor> {
        &self.manifest.tensors
    }

    /// Gets metadata for a tensor by its name.
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.manifest.tensors.get(name)
    }

    // =========================================================================
    // COMPONENT READING
    // =========================================================================

    /// Reads component data into a destination buffer.
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

        match &component.encoding {
            Some(Encoding::Zstd) => {
                let mut compressed_buf = vec![0u8; component.length as usize];
                self.reader.read_exact(&mut compressed_buf)?;
                
                if verify_checksum {
                    if let Some(checksum_str) = &component.digest {
                        Self::verify_checksum(checksum_str, &compressed_buf, ctx)?;
                    }
                }

                zstd::stream::copy_decode(std::io::Cursor::new(compressed_buf), &mut *dst)
                    .map_err(ZTensorError::ZstdDecompression)?;
            }
            Some(Encoding::Raw) | None => {
                if dst.len() as u64 != component.length {
                    return Err(ZTensorError::InvalidFileStructure(format!(
                        "Component length mismatch for {}/{}: expected {}, got dst len {}", 
                        ctx.tensor_name, ctx.component_name, component.length, dst.len()
                    )));
                }
                self.reader.read_exact(dst)?;
                
                if verify_checksum {
                    if let Some(checksum_str) = &component.digest {
                        Self::verify_checksum(checksum_str, dst, ctx)?;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Reads raw component data, allocating a new vector.
    pub fn read_component(&mut self, component: &Component) -> Result<Vec<u8>, ZTensorError> {
        let ctx = ReadContext::unknown();
        
        match &component.encoding {
            Some(Encoding::Zstd) => {
                self.reader.seek(SeekFrom::Start(component.offset))?;
                let mut compressed_buf = vec![0u8; component.length as usize];
                self.reader.read_exact(&mut compressed_buf)?;
                
                if let Some(checksum_str) = &component.digest {
                    Self::verify_checksum(checksum_str, &compressed_buf, &ctx)?;
                }

                let mut decompressed = Vec::new();
                zstd::stream::copy_decode(std::io::Cursor::new(compressed_buf), &mut decompressed)
                    .map_err(ZTensorError::ZstdDecompression)?;
                Ok(decompressed)
            }
            Some(Encoding::Raw) | None => {
                let mut data = vec![0u8; component.length as usize];
                self.read_component_into(component, &mut data, &ctx, true)?;
                Ok(data)
            }
        }
    }

    fn verify_checksum(checksum_str: &str, data: &[u8], ctx: &ReadContext) -> Result<(), ZTensorError> {
        if checksum_str.starts_with("crc32c:0x") {
            let expected_cs_hex = &checksum_str[9..];
            let expected_cs = u32::from_str_radix(expected_cs_hex, 16).map_err(|_| {
                ZTensorError::ChecksumFormatError(format!("Invalid CRC32C hex: {}", expected_cs_hex))
            })?;
            let calculated_cs = crc32c::crc32c(data);
            if calculated_cs != expected_cs {
                return Err(ZTensorError::ChecksumMismatch {
                    tensor_name: ctx.tensor_name.to_string(),
                    component_name: ctx.component_name.to_string(),
                    expected: format!("0x{:08X}", expected_cs),
                    calculated: format!("0x{:08X}", calculated_cs),
                });
            }
        }
        Ok(())
    }

    // =========================================================================
    // DENSE TENSOR READING
    // =========================================================================

    /// Core implementation for reading dense tensor data.
    fn read_tensor_impl(&mut self, name: &str, tensor: &Tensor, verify_checksum: bool) -> Result<Vec<u8>, ZTensorError> {
        if tensor.format != "dense" {
            return Err(ZTensorError::TypeMismatch {
                expected: "dense".to_string(),
                found: tensor.format.clone(),
                context: format!("tensor '{}'", name),
            });
        }

        let component = tensor.components.get("data").ok_or_else(|| {
            ZTensorError::InvalidFileStructure(format!("Dense tensor '{}' missing 'data' component", name))
        })?;

        let num_elements: u64 = if tensor.shape.is_empty() { 1 } else { tensor.shape.iter().product() };
        let expected_size = num_elements * tensor.dtype.byte_size()? as u64;
        
        let mut data = vec![0u8; expected_size as usize];
        let ctx = ReadContext::new(name, "data");
        self.read_component_into(component, &mut data, &ctx, verify_checksum)?;

        if cfg!(target_endian = "big") && tensor.dtype.is_multi_byte() {
            swap_endianness_in_place(&mut data, tensor.dtype.byte_size()?);
        }

        Ok(data)
    }

    /// Reads the raw byte data of a dense tensor.
    /// 
    /// # Arguments
    /// * `name` - Name of the tensor to read
    /// * `verify_checksum` - Whether to verify checksums (slower but safer)
    pub fn read_tensor(&mut self, name: &str, verify_checksum: bool) -> Result<Vec<u8>, ZTensorError> {
        let tensor = self.manifest.tensors.get(name)
            .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?
            .clone();
        self.read_tensor_impl(name, &tensor, verify_checksum)
    }

    /// Reads multiple tensors in batch.
    /// 
    /// More efficient than calling `read_tensor` repeatedly as it avoids 
    /// redundant error handling setup per call.
    /// 
    /// # Arguments
    /// * `names` - Names of tensors to read
    /// * `verify_checksum` - Whether to verify checksums
    /// 
    /// # Returns
    /// Vector of raw byte data in same order as input names
    pub fn read_tensors(&mut self, names: &[&str], verify_checksum: bool) -> Result<Vec<Vec<u8>>, ZTensorError> {
        let mut results = Vec::with_capacity(names.len());
        for name in names {
            results.push(self.read_tensor(name, verify_checksum)?);
        }
        Ok(results)
    }

    // =========================================================================
    // TYPED TENSOR READING
    // =========================================================================

    /// Reads tensor data into a typed vector.
    /// 
    /// # Type Parameters
    /// * `T` - The Pod type to read. Must match the tensor's dtype.
    /// 
    /// # Examples
    /// ```ignore
    /// let floats: Vec<f32> = reader.read_tensor_as("weights")?;
    /// let halfs: Vec<half::f16> = reader.read_tensor_as("embeddings")?;
    /// ```
    pub fn read_tensor_as<T: Pod>(&mut self, name: &str) -> Result<Vec<T>, ZTensorError> {
        let tensor = self.manifest.tensors.get(name)
            .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;
        
        if !T::dtype_matches(&tensor.dtype) {
            return Err(ZTensorError::TypeMismatch {
                expected: tensor.dtype.to_string_key(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("tensor '{}'", name),
            });
        }
        
        if tensor.format != "dense" {
            return Err(ZTensorError::TypeMismatch {
                expected: "dense".to_string(),
                found: tensor.format.clone(),
                context: name.to_string(),
            });
        }

        let component = tensor.components.get("data")
            .ok_or(ZTensorError::InvalidFileStructure(format!("missing data for {}", name)))?
            .clone();

        let num_elements: usize = if tensor.shape.is_empty() { 1 } else { tensor.shape.iter().product::<u64>() as usize };
        let mut typed_data = vec![T::default(); num_elements];
        
        let byte_len = num_elements * T::SIZE;
        let output_slice = unsafe {
            std::slice::from_raw_parts_mut(typed_data.as_mut_ptr() as *mut u8, byte_len)
        };
        
        let ctx = ReadContext::new(name, "data");
        self.read_component_into(&component, output_slice, &ctx, true)?;
        
        Ok(typed_data)
    }

    // =========================================================================
    // SPARSE TENSOR READING
    // =========================================================================

    fn read_component_as<T: Pod>(&mut self, component: &Component, ctx: &ReadContext) -> Result<Vec<T>, ZTensorError> {
        match &component.encoding {
            Some(Encoding::Zstd) => {
                let bytes = self.read_component(component)?;
                let num_elements = bytes.len() / T::SIZE;
                let mut values = vec![T::default(); num_elements];
                unsafe {
                    std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, bytes.len())
                        .copy_from_slice(&bytes);
                }
                Ok(values)
            }
            Some(Encoding::Raw) | None => {
                let num_elements = component.length as usize / T::SIZE;
                let mut values = vec![T::default(); num_elements];
                let byte_slice = unsafe {
                    std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, component.length as usize)
                };
                self.read_component_into(component, byte_slice, ctx, true)?;
                Ok(values)
            }
        }
    }

    fn read_u64_component(&mut self, component: &Component, ctx: &ReadContext) -> Result<Vec<u64>, ZTensorError> {
        let mut bytes = match &component.encoding {
            Some(Encoding::Zstd) => self.read_component(component)?,
            Some(Encoding::Raw) | None => {
                let mut buf = vec![0u8; component.length as usize];
                self.read_component_into(component, &mut buf, ctx, true)?;
                buf
            }
        };

        if cfg!(target_endian = "big") {
            swap_endianness_in_place(&mut bytes, 8);
        }
        
        Ok(bytes.chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    /// Reads a COO format sparse tensor.
    pub fn read_coo_tensor<T: Pod>(&mut self, name: &str) -> Result<crate::models::CooTensor<T>, ZTensorError> {
        let tensor = self.get_tensor(name)
            .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;
        
        if tensor.format != "sparse_coo" {
            return Err(ZTensorError::TypeMismatch {
                expected: "sparse_coo".to_string(),
                found: tensor.format.clone(),
                context: format!("tensor '{}'", name),
            });
        }
        
        let dtype = tensor.dtype;
        let shape = tensor.shape.clone();
        let val_comp = tensor.components.get("values")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'values'".to_string()))?.clone();
        let coords_comp = tensor.components.get("coords")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'coords'".to_string()))?.clone();
        
        let val_ctx = ReadContext::new(name, "values");
        let mut values: Vec<T> = self.read_component_as(&val_comp, &val_ctx)?;
        
        if cfg!(target_endian = "big") && dtype.is_multi_byte() {
            let byte_len = values.len() * T::SIZE;
            let val_slice = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len) };
            swap_endianness_in_place(val_slice, dtype.byte_size()?);
        }

        let coords_ctx = ReadContext::new(name, "coords");
        let all_coords = self.read_u64_component(&coords_comp, &coords_ctx)?;
        
        let nnz = values.len();
        let ndim = shape.len();
        
        if all_coords.len() != nnz * ndim {
            return Err(ZTensorError::DataConversionError("COO coords size mismatch".to_string()));
        }

        let mut indices = Vec::with_capacity(nnz);
        for i in 0..nnz {
            let mut idx = Vec::with_capacity(ndim);
            for d in 0..ndim {
                idx.push(all_coords[d * nnz + i]);
            }
            indices.push(idx);
        }

        Ok(crate::models::CooTensor { shape, indices, values })
    }

    /// Reads a CSR format sparse tensor.
    pub fn read_csr_tensor<T: Pod>(&mut self, name: &str) -> Result<crate::models::CsrTensor<T>, ZTensorError> {
        let tensor = self.get_tensor(name)
            .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;
        
        if tensor.format != "sparse_csr" {
            return Err(ZTensorError::TypeMismatch {
                expected: "sparse_csr".to_string(),
                found: tensor.format.clone(),
                context: format!("tensor '{}'", name),
            });
        }
        
        let dtype = tensor.dtype;
        let shape = tensor.shape.clone();
        let val_comp = tensor.components.get("values")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'values'".to_string()))?.clone();
        let idx_comp = tensor.components.get("indices")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'indices'".to_string()))?.clone();
        let ptr_comp = tensor.components.get("indptr")
            .ok_or(ZTensorError::InvalidFileStructure("Missing 'indptr'".to_string()))?.clone();
        
        let val_ctx = ReadContext::new(name, "values");
        let mut values: Vec<T> = self.read_component_as(&val_comp, &val_ctx)?;
        
        if cfg!(target_endian = "big") && dtype.is_multi_byte() {
            let byte_len = values.len() * T::SIZE;
            let val_slice = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len) };
            swap_endianness_in_place(val_slice, dtype.byte_size()?);
        }

        let idx_ctx = ReadContext::new(name, "indices");
        let indices = self.read_u64_component(&idx_comp, &idx_ctx)?;
        
        let ptr_ctx = ReadContext::new(name, "indptr");
        let indptr = self.read_u64_component(&ptr_comp, &ptr_ctx)?;

        Ok(crate::models::CsrTensor { shape, indptr, indices, values })
    }
}
