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

// Simpler Pod impl for u8/i8 (endianness doesn't matter)
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

// Bool needs special handling if it were multi-byte, but spec says 1 byte.
impl Pod for bool {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn dtype_matches(dtype: &DType) -> bool {
        dtype == &DType::Bool
    }
}

// Half-precision floating point types (using the `half` crate)
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

/// Reads zTensor files (v1.0).
pub struct ZTensorReader<R: Read + Seek> {
    /// The underlying reader - public for advanced use cases like FFI
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
        // SAFETY: We are mapping the file. Standard mmap caveats apply.
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
        
         // Basic validity check
        if manifest_size > file_size.saturating_sub(MAGIC_NUMBER.len() as u64 + 8) {
             return Err(ZTensorError::InvalidFileStructure(format!(
                "Manifest size {} is too large for file size {}",
                manifest_size, file_size
            )));
        }

        // Read Manifest
        reader.seek(SeekFrom::End(-8 - (manifest_size as i64)))?;
        let mut cbor_buf = vec![0u8; manifest_size as usize];
        reader.read_exact(&mut cbor_buf)?;

        let manifest: Manifest =
            serde_cbor::from_slice(&cbor_buf).map_err(ZTensorError::CborDeserialize)?;

        Ok(Self {
            reader,
            manifest,
        })
    }

    /// Lists all tensors in the file.
    pub fn list_tensors(&self) -> &std::collections::BTreeMap<String, Tensor> {
        &self.manifest.tensors
    }

    /// Gets metadata for a tensor by its name.
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.manifest.tensors.get(name)
    }

    /// Reads component data with context for error messages.
    fn read_component_with_context(
        &mut self, 
        component: &Component, 
        tensor_name: &str,
        component_name: &str
    ) -> Result<Vec<u8>, ZTensorError> {
         if component.offset % ALIGNMENT != 0 {
             return Err(ZTensorError::InvalidAlignment {
                offset: component.offset,
                alignment: ALIGNMENT,
            });
        }

        self.reader.seek(SeekFrom::Start(component.offset))?;
        let mut buffer = vec![0u8; component.length as usize];
        self.reader.read_exact(&mut buffer)?;

        // Verify Checksum
         if let Some(checksum_str) = &component.digest {
            if checksum_str.starts_with("crc32c:0x") {
                 let expected_cs_hex = &checksum_str[9..];
                 let expected_cs = u32::from_str_radix(expected_cs_hex, 16).map_err(|_| {
                    ZTensorError::ChecksumFormatError(format!(
                        "Invalid CRC32C hex: {}",
                        expected_cs_hex
                    ))
                })?;
                let calculated_cs = crc32c::crc32c(&buffer);
                if calculated_cs != expected_cs {
                     return Err(ZTensorError::ChecksumMismatch {
                        tensor_name: tensor_name.to_string(),
                        component_name: component_name.to_string(),
                        expected: format!("0x{:08X}", expected_cs),
                        calculated: format!("0x{:08X}", calculated_cs),
                    });
                }
            }
        }
        
        // Decompress
        match component.encoding {
            Some(Encoding::Zstd) => {
                let mut decompressed: Vec<u8> = Vec::new(); 
                 let _ = zstd::stream::copy_decode(std::io::Cursor::new(buffer), &mut decompressed)
                    .map_err(ZTensorError::ZstdDecompression)?;
                 Ok(decompressed)
            }
            Some(Encoding::Raw) | None => Ok(buffer),
        }
    }

    /// Reads raw component data (public for FFI access).
    /// Note: Uses "unknown" for tensor context in error messages.
    pub fn read_component(&mut self, component: &Component) -> Result<Vec<u8>, ZTensorError> {
        self.read_component_with_context(component, "unknown", "unknown")
    }

    /// Reads the raw, processed (decompressed, endian-swapped to native) byte data of a dense tensor.
    pub fn read_tensor(&mut self, name: &str) -> Result<Vec<u8>, ZTensorError> {
        // Get tensor metadata, extracting only the data we need to avoid borrow issues
        let (dtype, shape, data_component) = {
            let tensor = self.manifest.tensors.get(name)
                .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;
            
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
            
            (tensor.dtype, tensor.shape.clone(), component.clone())
        };

        let mut data = self.read_component_with_context(&data_component, name, "data")?;

        // Endianness handling:
        // Spec: "All integers and multi-byte data types must be written in LE."
        // We read raw bytes. If we are on Big Endian system, we must swap.
        if cfg!(target_endian = "big") && dtype.is_multi_byte() {
            swap_endianness_in_place(&mut data, dtype.byte_size()?);
        }

        // Validity check size
        let num_elements: u64 = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_size = num_elements * dtype.byte_size()? as u64;
        if data.len() as u64 != expected_size {
             return Err(ZTensorError::InconsistentDataSize {
                expected: expected_size,
                found: data.len() as u64,
            });
        }

        Ok(data)
    }

    /// Reads tensor data into a typed vector.
    /// 
    /// # Type Parameters
    /// * `T` - The Pod type to read the tensor data as. Must match the tensor's dtype.
    /// 
    /// # Examples
    /// ```ignore
    /// let floats: Vec<f32> = reader.read_tensor_as("weights")?;
    /// let halfs: Vec<half::f16> = reader.read_tensor_as("embeddings")?;
    /// ```
    pub fn read_tensor_as<T: Pod>(&mut self, name: &str) -> Result<Vec<T>, ZTensorError> {
        // Get dtype for validation before reading
        let dtype = {
            let tensor = self.get_tensor(name)
                .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;
            tensor.dtype
        };
        
        if !T::dtype_matches(&dtype) {
             return Err(ZTensorError::TypeMismatch {
                expected: dtype.to_string_key(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("tensor '{}'", name),
            });
        }
        
        let data_bytes = self.read_tensor(name)?;
        
        let element_size = T::SIZE;
        let num_elements = data_bytes.len() / element_size;
        
        let mut typed_data = vec![T::default(); num_elements];
         // Safety: T is Pod.
        let output_slice = unsafe {
             std::slice::from_raw_parts_mut(
                typed_data.as_mut_ptr() as *mut u8,
                num_elements * element_size
            )
        };
        output_slice.copy_from_slice(&data_bytes);
        
        Ok(typed_data)
    }

    /// Backward compatibility alias for read_tensor_as
    #[deprecated(since = "0.2.0", note = "use read_tensor_as instead")]
    pub fn read_typed_tensor_data<T: Pod>(&mut self, name: &str) -> Result<Vec<T>, ZTensorError> {
        self.read_tensor_as(name)
    }

    // --- Sparse tensor reading ---

    /// Reads a COO format sparse tensor.
    pub fn read_coo_tensor<T: Pod>(&mut self, name: &str) -> Result<crate::models::CooTensor<T>, ZTensorError> {
        // Extract tensor metadata
        let (dtype, shape, val_comp, coords_comp) = {
            let tensor = self.get_tensor(name)
                .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;
            
            if tensor.format != "sparse_coo" {
                return Err(ZTensorError::TypeMismatch {
                    expected: "sparse_coo".to_string(),
                    found: tensor.format.clone(),
                    context: format!("tensor '{}'", name),
                });
            }
            
            let val_comp = tensor.components.get("values")
                .ok_or(ZTensorError::InvalidFileStructure("Missing 'values'".to_string()))?
                .clone();
            let coords_comp = tensor.components.get("coords")
                .ok_or(ZTensorError::InvalidFileStructure("Missing 'coords'".to_string()))?
                .clone();
            
            (tensor.dtype, tensor.shape.clone(), val_comp, coords_comp)
        };
        
        // 1. values
        let mut val_bytes = self.read_component_with_context(&val_comp, name, "values")?;
        if cfg!(target_endian = "big") && dtype.is_multi_byte() {
             swap_endianness_in_place(&mut val_bytes, dtype.byte_size()?);
        }
        // Convert to T
        let mut values = vec![T::default(); val_bytes.len() / T::SIZE];
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, val_bytes.len()).copy_from_slice(&val_bytes); }

        // 2. coords (Matrix ndim x nnz)
        let mut coords_bytes = self.read_component_with_context(&coords_comp, name, "coords")?;
        if cfg!(target_endian = "big") {
            swap_endianness_in_place(&mut coords_bytes, 8);
        }
        
        let u64_size = 8;
        let total_u64s = coords_bytes.len() / u64_size;
        let nnz = values.len();
        let ndim = shape.len();
        
        if total_u64s != nnz * ndim {
             return Err(ZTensorError::DataConversionError("COO coords size mismatch".to_string()));
        }

        let all_coords: Vec<u64> = coords_bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
        
        let mut indices = Vec::with_capacity(nnz);
        for i in 0..nnz {
            let mut idx = Vec::with_capacity(ndim);
            for d in 0..ndim {
                // all_coords[d * nnz + i]
                idx.push(all_coords[d * nnz + i]);
            }
            indices.push(idx);
        }

        Ok(crate::models::CooTensor {
            shape,
            indices,
            values,
        })
    }

    /// Reads a CSR format sparse tensor.
    pub fn read_csr_tensor<T: Pod>(&mut self, name: &str) -> Result<crate::models::CsrTensor<T>, ZTensorError> {
        // Extract tensor metadata
        let (dtype, shape, val_comp, idx_comp, ptr_comp) = {
            let tensor = self.get_tensor(name)
                .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;
            
            if tensor.format != "sparse_csr" {
                return Err(ZTensorError::TypeMismatch {
                    expected: "sparse_csr".to_string(),
                    found: tensor.format.clone(),
                    context: format!("tensor '{}'", name),
                });
            }
            
            let val_comp = tensor.components.get("values")
                .ok_or(ZTensorError::InvalidFileStructure("Missing 'values'".to_string()))?
                .clone();
            let idx_comp = tensor.components.get("indices")
                .ok_or(ZTensorError::InvalidFileStructure("Missing 'indices'".to_string()))?
                .clone();
            let ptr_comp = tensor.components.get("indptr")
                .ok_or(ZTensorError::InvalidFileStructure("Missing 'indptr'".to_string()))?
                .clone();
            
            (tensor.dtype, tensor.shape.clone(), val_comp, idx_comp, ptr_comp)
        };
        
        // values
        let mut val_bytes = self.read_component_with_context(&val_comp, name, "values")?;
        if cfg!(target_endian = "big") && dtype.is_multi_byte() { 
            swap_endianness_in_place(&mut val_bytes, dtype.byte_size()?); 
        }
        let mut values = vec![T::default(); val_bytes.len() / T::SIZE];
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, val_bytes.len()).copy_from_slice(&val_bytes); }

         // indices
        let mut idx_bytes = self.read_component_with_context(&idx_comp, name, "indices")?;
        if cfg!(target_endian = "big") { swap_endianness_in_place(&mut idx_bytes, 8); }
        let indices: Vec<u64> = idx_bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
        
        // indptr
        let mut ptr_bytes = self.read_component_with_context(&ptr_comp, name, "indptr")?;
        if cfg!(target_endian = "big") { swap_endianness_in_place(&mut ptr_bytes, 8); }
        let indptr: Vec<u64> = ptr_bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();

        Ok(crate::models::CsrTensor {
            shape,
            indptr,
            indices,
            values,
        })
    }
}
