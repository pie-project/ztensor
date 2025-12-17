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

    /// Reads component data directly into a destination buffer with optional checksum verification.
    fn read_component_into_opts(
        &mut self, 
        component: &Component, 
        dst: &mut [u8],
        tensor_name: &str,
        component_name: &str,
        verify_checksum: bool,
    ) -> Result<(), ZTensorError> {
         if component.offset % ALIGNMENT != 0 {
             return Err(ZTensorError::InvalidAlignment {
                offset: component.offset,
                alignment: ALIGNMENT,
            });
        }

        self.reader.seek(SeekFrom::Start(component.offset))?;

        // Case 1: Raw (No Compression) - Read directly into dst
        if matches!(component.encoding, Some(Encoding::Raw) | None) {
             if dst.len() as u64 != component.length {
                 return Err(ZTensorError::InvalidFileStructure(format!(
                     "Component length mismatch for {}/{}: expected {}, got dst len {}", 
                     tensor_name, component_name, component.length, dst.len()
                 )));
             }
             self.reader.read_exact(dst)?;
             
             // Verify Checksum on dst (only if requested)
             if verify_checksum {
                 if let Some(checksum_str) = &component.digest {
                    Self::verify_checksum(checksum_str, dst, tensor_name, component_name)?;
                }
            }
        } 
        // Case 2: Compression (Zstd) - Read to temp buf, decompress to dst
        else if matches!(component.encoding, Some(Encoding::Zstd)) {
            let mut compressed_buf = vec![0u8; component.length as usize];
            self.reader.read_exact(&mut compressed_buf)?;
            
            // Verify Checksum on compressed data (only if requested)
            if verify_checksum {
                if let Some(checksum_str) = &component.digest {
                    Self::verify_checksum(checksum_str, &compressed_buf, tensor_name, component_name)?;
                }
            }

            // Decompress
            let _ = zstd::stream::copy_decode(std::io::Cursor::new(compressed_buf), &mut *dst)
                   .map_err(ZTensorError::ZstdDecompression)?;
        }
        
        Ok(())
    }

    /// Reads component data directly into a destination buffer (with checksum verification).
    fn read_component_into(
        &mut self, 
        component: &Component, 
        dst: &mut [u8],
        tensor_name: &str,
        component_name: &str
    ) -> Result<(), ZTensorError> {
        self.read_component_into_opts(component, dst, tensor_name, component_name, true)
    }

    fn verify_checksum(checksum_str: &str, data: &[u8], tensor_name: &str, component_name: &str) -> Result<(), ZTensorError> {
        if checksum_str.starts_with("crc32c:0x") {
             let expected_cs_hex = &checksum_str[9..];
             let expected_cs = u32::from_str_radix(expected_cs_hex, 16).map_err(|_| {
                ZTensorError::ChecksumFormatError(format!(
                    "Invalid CRC32C hex: {}",
                    expected_cs_hex
                ))
            })?;
            let calculated_cs = crc32c::crc32c(data);
            if calculated_cs != expected_cs {
                 return Err(ZTensorError::ChecksumMismatch {
                    tensor_name: tensor_name.to_string(),
                    component_name: component_name.to_string(),
                    expected: format!("0x{:08X}", expected_cs),
                    calculated: format!("0x{:08X}", calculated_cs),
                });
            }
        }
        Ok(())
    }

    /// Reads raw component data (public for FFI access).
    /// Note: Uses "unknown" for tensor context in error messages.
    /// Warning: This allocates a new vector. Use read_tensor_as for typed efficient reading.
    pub fn read_component(&mut self, component: &Component) -> Result<Vec<u8>, ZTensorError> {
        // We need to know the decompressed size to allocate the vector.
        // For Raw, it is component.length.
        // For Zstd, the Manifest SHOULD arguably store decompressed_length if we strictly want to pre-allocate,
        // but current Component struct might not have it if it wasn't added.
        // CHECK: Does Component have decompressed_size?
        // Looking at models.rs (implied): usually just length (stored size).
        // If Zstd, we might not know output size easily without streaming or an extra field.
        // However, for Tensors, we DO know the shape and dtype, so we know expected size.
        // 'read_component' is a raw access tool. 
        
        // If Raw/None:
        if matches!(component.encoding, Some(Encoding::Raw) | None) {
            let mut vec = vec![0u8; component.length as usize];
            self.read_component_into(component, &mut vec, "unknown", "unknown")?;
            Ok(vec)
        } else {
             // Fallback for compressed generic component read where we might not know size:
             // We can use the old read logic or just standard decompression growing vec.
             self.read_component_alloc(component, "unknown", "unknown")
        }
    }
    
    // Fallback/Legacy logic for allocating read when size is unknown (mostly for compressed generic blobs)
    fn read_component_alloc(
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

        if let Some(checksum_str) = &component.digest {
             Self::verify_checksum(checksum_str, &buffer, tensor_name, component_name)?;
        }
        
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

        // Validity check size
        let num_elements: u64 = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_size = num_elements * dtype.byte_size()? as u64;
        
        let mut data = vec![0u8; expected_size as usize];
        self.read_component_into(&data_component, &mut data, name, "data")?;

        // Endianness handling:
        // Spec: "All integers and multi-byte data types must be written in LE."
        // We read raw bytes. If we are on Big Endian system, we must swap.
        if cfg!(target_endian = "big") && dtype.is_multi_byte() {
            swap_endianness_in_place(&mut data, dtype.byte_size()?);
        }

        Ok(data)
    }

    /// Reads tensor data using provided metadata, skipping the BTreeMap lookup.
    /// This is an optimization for cases where we already have the metadata (e.g. FFI).
    pub fn read_tensor_by_info(&mut self, name: &str, tensor: &Tensor) -> Result<Vec<u8>, ZTensorError> {
        self.read_tensor_by_info_opts(name, tensor, true)
    }

    /// Fast tensor read that skips checksum verification.
    /// Use this for maximum performance when data integrity is verified elsewhere or not critical.
    pub fn read_tensor_by_info_fast(&mut self, name: &str, tensor: &Tensor) -> Result<Vec<u8>, ZTensorError> {
        self.read_tensor_by_info_opts(name, tensor, false)
    }

    /// Internal method with configurable checksum verification.
    fn read_tensor_by_info_opts(&mut self, name: &str, tensor: &Tensor, verify_checksum: bool) -> Result<Vec<u8>, ZTensorError> {
        let (dtype, shape, data_component) = {
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

        // Validity check size
        let num_elements: u64 = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_size = num_elements * dtype.byte_size()? as u64;
        
        let mut data = vec![0u8; expected_size as usize];
        self.read_component_into_opts(&data_component, &mut data, name, "data", verify_checksum)?;

        // Endianness handling:
        if cfg!(target_endian = "big") && dtype.is_multi_byte() {
            swap_endianness_in_place(&mut data, dtype.byte_size()?);
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
        
        // Calculate required size
        // We trust the manifest shape/dtype for allocation key.

        // Check tensor again or just access? We need the component actually to pass to read.
        // Re-get everything cleanly.
         let (_dtype, shape, data_component) = {
            let tensor = self.manifest.tensors.get(name)
                .ok_or_else(|| ZTensorError::TensorNotFound(name.to_string()))?;
            
             if tensor.format != "dense" {
                return Err(ZTensorError::TypeMismatch { expected: "dense".to_string(), found: tensor.format.clone(), context: name.to_string() });
            }
            let component = tensor.components.get("data").ok_or(ZTensorError::InvalidFileStructure(format!("missing data for {}", name)))?;
            (tensor.dtype, tensor.shape.clone(), component.clone())
        };

        let num_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product::<u64>() as usize };
        let mut typed_data = vec![T::default(); num_elements];
        
        let element_size = T::SIZE;
        let byte_len = num_elements * element_size;

        // Create byte view of the typed vector
        let output_slice = unsafe {
             std::slice::from_raw_parts_mut(
                typed_data.as_mut_ptr() as *mut u8,
                byte_len
            )
        };
        
        // Read directly into the slice
        self.read_component_into(&data_component, output_slice, name, "data")?;
        
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
        // We know nnz from the component length ideally? 
        // Typically COO "values" is nnz * T::SIZE (raw) or compressed.
        // We can't know nnz easily if compressed without metadata.
        // But for "Raw", we know component.length.
        // If "Zstd", we generally rely on implicit sizing or we need to decomp into temporary then cast.
        // However, we implemented `read_component_into` which REQUIRES `dst` to be sized.
        
        // Strategy: 
        // If Raw, use component.length to size the alloc.
        // If Zstd, we rely on `read_component_alloc` (the fallback) because we don't know the uncompressed size 
        // unless we store it or assume it from somewhere (like shape, but sparse doesn't imply nnz from shape).
        // WAIT: Sparse tensors usually store nnz in metadata? ztensor spec doesn't strictly have 'nnz' field at top level?
        // Ah, checked `models.rs`, Tensor struct has `shape` and `dtype`. 
        // It DOES NOT store `nnz` explicitly.
        // So for sparse tensors, if compressed, we DO NOT know the output size a priori.
        
        // For Raw, we know: size = component.length.
        // So for Raw we can optimize. For Zstd we fallback.
        
        let mut values: Vec<T>;
        
        if matches!(val_comp.encoding, Some(Encoding::Raw) | None) {
             let byte_len = val_comp.length as usize;
             let num_vals = byte_len / T::SIZE;
             values = vec![T::default(); num_vals];
             let val_slice = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len) };
             self.read_component_into(&val_comp, val_slice, name, "values")?;
        } else {
             // Fallback for compressed sparse where we don't know NNZ
             let val_bytes = self.read_component_alloc(&val_comp, name, "values")?;
             values = vec![T::default(); val_bytes.len() / T::SIZE];
             unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, val_bytes.len()).copy_from_slice(&val_bytes); }
        }
        
        if cfg!(target_endian = "big") && dtype.is_multi_byte() {
             // We need to swap the Typed Vec... since we cast it to bytes above during read, it's now LE bytes in T memory.
             // Wait, if we read directly into T memory as bytes, it's LE. We need to swap bytes if host is BE.
             // Impl_pod trait doesn't expose byte swapping logic on T directly easily without iteratively.
             // But we can cast back to u8 slice and swap.
             let byte_len = values.len() * T::SIZE;
             let val_slice = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len) };
             swap_endianness_in_place(val_slice, dtype.byte_size()?);
        }

        // 2. coords (Matrix ndim x nnz)
        let mut coords_bytes: Vec<u8>;
        // Similar logic for coords (u64)
         if matches!(coords_comp.encoding, Some(Encoding::Raw) | None) {
             coords_bytes = vec![0u8; coords_comp.length as usize];
             self.read_component_into(&coords_comp, &mut coords_bytes, name, "coords")?;
         } else {
             coords_bytes = self.read_component_alloc(&coords_comp, name, "coords")?;
         }

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
        let mut values: Vec<T>;
        if matches!(val_comp.encoding, Some(Encoding::Raw) | None) {
             let byte_len = val_comp.length as usize;
             values = vec![T::default(); byte_len / T::SIZE];
             let val_slice = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len) };
             self.read_component_into(&val_comp, val_slice, name, "values")?;
        } else {
             let val_bytes = self.read_component_alloc(&val_comp, name, "values")?;
             values = vec![T::default(); val_bytes.len() / T::SIZE];
             unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, val_bytes.len()).copy_from_slice(&val_bytes); }
        }
        if cfg!(target_endian = "big") && dtype.is_multi_byte() { 
             let byte_len = values.len() * T::SIZE;
             let val_slice = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len) };
             swap_endianness_in_place(val_slice, dtype.byte_size()?); 
        }

        // indices
        let mut idx_bytes: Vec<u8>;
        if matches!(idx_comp.encoding, Some(Encoding::Raw) | None) {
             idx_bytes = vec![0u8; idx_comp.length as usize];
             self.read_component_into(&idx_comp, &mut idx_bytes, name, "indices")?;
        } else {
             idx_bytes = self.read_component_alloc(&idx_comp, name, "indices")?;
        }
        
        if cfg!(target_endian = "big") { swap_endianness_in_place(&mut idx_bytes, 8); }
        let indices: Vec<u64> = idx_bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect();
        
        // indptr
        let mut ptr_bytes: Vec<u8>;
        if matches!(ptr_comp.encoding, Some(Encoding::Raw) | None) {
             ptr_bytes = vec![0u8; ptr_comp.length as usize];
             self.read_component_into(&ptr_comp, &mut ptr_bytes, name, "indptr")?;
        } else {
             ptr_bytes = self.read_component_alloc(&ptr_comp, name, "indptr")?;
        }
        
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
