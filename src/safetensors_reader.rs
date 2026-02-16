//! SafeTensors format reader.
//!
//! Provides read-only access to `.safetensors` files through a unified API
//! that mirrors `Reader`. Uses memory mapping for zero-copy access.
//!
//! Requires the `safetensors` feature.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;

use crate::error::Error;
use crate::models::{DType, Manifest, Object};
use crate::reader::{TensorData, TensorElement, TensorReader};

/// Reader for SafeTensors (.safetensors) files.
///
/// Uses memory mapping for efficient zero-copy access to tensor data.
pub struct SafeTensorsReader {
    mmap: Mmap,
    pub manifest: Manifest,
    /// Maps tensor name -> (byte_offset, byte_length) in the mmap.
    data_ranges: BTreeMap<String, (usize, usize)>,
}

fn convert_dtype(dtype: safetensors::Dtype) -> Result<DType, Error> {
    match dtype {
        safetensors::Dtype::F64 => Ok(DType::F64),
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::I64 => Ok(DType::I64),
        safetensors::Dtype::I32 => Ok(DType::I32),
        safetensors::Dtype::I16 => Ok(DType::I16),
        safetensors::Dtype::I8 => Ok(DType::I8),
        safetensors::Dtype::U64 => Ok(DType::U64),
        safetensors::Dtype::U32 => Ok(DType::U32),
        safetensors::Dtype::U16 => Ok(DType::U16),
        safetensors::Dtype::U8 => Ok(DType::U8),
        safetensors::Dtype::BOOL => Ok(DType::Bool),
        other => Err(Error::UnsupportedDType(format!("{:?}", other))),
    }
}

impl SafeTensorsReader {
    /// Opens a SafeTensors file using memory mapping.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Parse the safetensors header to build our manifest
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| Error::InvalidFileStructure(format!("SafeTensors parse error: {}", e)))?;

        let mut objects = BTreeMap::new();
        let mut data_ranges = BTreeMap::new();

        for (name, tensor) in st.tensors() {
            let dtype = convert_dtype(tensor.dtype())?;
            let shape: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();

            let data = tensor.data();
            let data_ptr = data.as_ptr() as usize;
            let mmap_ptr = mmap.as_ptr() as usize;
            let offset = data_ptr.checked_sub(mmap_ptr).ok_or_else(|| {
                Error::InvalidFileStructure(
                    "SafeTensors tensor data pointer is before mmap base".to_string(),
                )
            })?;
            let length = data.len();

            let obj = Object::dense(shape, dtype, offset as u64, length as u64);
            data_ranges.insert(name.clone(), (offset, length));
            objects.insert(name, obj);
        }

        let manifest = Manifest {
            version: "safetensors".to_string(),
            attributes: None,
            objects,
        };

        Ok(Self {
            mmap,
            manifest,
            data_ranges,
        })
    }

    /// Gets a zero-copy reference to an object's raw data.
    pub fn view(&self, name: &str) -> Result<&[u8], Error> {
        let (offset, length) = self
            .data_ranges
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;
        Ok(&self.mmap[*offset..*offset + *length])
    }

    /// Gets a typed zero-copy reference to an object's data.
    pub fn view_as<T: TensorElement>(&self, name: &str) -> Result<&[T], Error> {
        let dtype = self
            .manifest
            .objects
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?
            .data_dtype()?;
        if T::DTYPE != dtype {
            return Err(Error::TypeMismatch {
                expected: dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }
        crate::reader::bytes_as_typed(self.view(name)?)
    }

    /// Reads object data as a copy (for API compatibility with Reader).
    pub fn read(&self, name: &str) -> Result<Vec<u8>, Error> {
        self.view(name).map(|s| s.to_vec())
    }

    /// Reads object data as a typed vector.
    pub fn read_as<T: TensorElement>(&self, name: &str) -> Result<Vec<T>, Error> {
        let dtype = self
            .manifest
            .objects
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?
            .data_dtype()?;
        if T::DTYPE != dtype {
            return Err(Error::TypeMismatch {
                expected: dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }
        crate::reader::bytes_to_typed_vec(self.view(name)?)
    }
}

impl TensorReader for SafeTensorsReader {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read_data(&self, name: &str) -> Result<TensorData<'_>, Error> {
        let slice = self.view(name)?;
        Ok(TensorData::Borrowed(slice))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use tempfile::NamedTempFile;

    #[test]
    fn test_safetensors_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        // Create a safetensors file
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_bytes = bytemuck::cast_slice(&data);

        let tensors = vec![(
            "test_tensor".to_string(),
            TensorView::new(safetensors::Dtype::F32, vec![2, 3], data_bytes)?,
        )];

        let serialized = safetensors::tensor::serialize(tensors, &None)?;

        let mut file = NamedTempFile::new()?;
        std::io::Write::write_all(&mut file, &serialized)?;
        let path = file.path().to_path_buf();

        // Read it back
        let reader = SafeTensorsReader::open(&path)?;

        assert_eq!(reader.tensors().len(), 1);
        let obj = reader.get("test_tensor").unwrap();
        assert_eq!(obj.shape, vec![2, 3]);

        let slice = reader.view("test_tensor")?;
        assert_eq!(slice.len(), 24); // 6 * 4 bytes

        let typed: Vec<f32> = reader.read_as("test_tensor")?;
        assert_eq!(typed, data);

        Ok(())
    }

    #[test]
    fn test_safetensors_multiple_tensors() -> Result<(), Box<dyn std::error::Error>> {
        let f32_data: Vec<f32> = vec![1.0, 2.0];
        let i32_data: Vec<i32> = vec![10, 20, 30];

        let tensors = vec![
            (
                "float_tensor".to_string(),
                TensorView::new(
                    safetensors::Dtype::F32,
                    vec![2],
                    bytemuck::cast_slice(&f32_data),
                )?,
            ),
            (
                "int_tensor".to_string(),
                TensorView::new(
                    safetensors::Dtype::I32,
                    vec![3],
                    bytemuck::cast_slice(&i32_data),
                )?,
            ),
        ];

        let serialized = safetensors::tensor::serialize(tensors, &None)?;

        let mut file = NamedTempFile::new()?;
        std::io::Write::write_all(&mut file, &serialized)?;

        let reader = SafeTensorsReader::open(file.path())?;
        assert_eq!(reader.tensors().len(), 2);

        let f32_result: Vec<f32> = reader.read_as("float_tensor")?;
        assert_eq!(f32_result, f32_data);

        let i32_result: Vec<i32> = reader.read_as("int_tensor")?;
        assert_eq!(i32_result, i32_data);

        // Type mismatch
        match reader.read_as::<i32>("float_tensor") {
            Err(Error::TypeMismatch { .. }) => {}
            _ => panic!("Expected TypeMismatch"),
        }

        Ok(())
    }
}
