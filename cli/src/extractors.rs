//! Tensor extractors for different file formats

use anyhow::{Context, Result, bail};
use safetensors::tensor::SafeTensors;
use serde_pickle::Value as PickleValue;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::pickle;
use crate::utils::{dtype_size, safetensor_dtype_to_ztensor, gguf_dtype_to_ztensor};

/// Standardized intermediate structure for extracted tensors
pub struct ExtractedTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: ztensor::DType,
    pub data: Vec<u8>,
}

impl Clone for ExtractedTensor {
    fn clone(&self) -> Self {
        ExtractedTensor {
            name: self.name.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            data: self.data.clone(),
        }
    }
}

/// Trait for format-specific tensor extraction
pub trait TensorExtractor {
    fn extract_tensors(&self, path: &str) -> Result<Vec<ExtractedTensor>>;
}

// ----------------------------------------------------------------------------
// SafeTensor Extractor
// ----------------------------------------------------------------------------

pub struct SafeTensorExtractor;

impl TensorExtractor for SafeTensorExtractor {
    fn extract_tensors(&self, path: &str) -> Result<Vec<ExtractedTensor>> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open SafeTensor file '{}'", path))?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("Failed to deserialize SafeTensor file '{}'", path))?;
        
        let mut result = Vec::new();
        for (name, tensor) in st.tensors() {
            let dtype = safetensor_dtype_to_ztensor(&tensor.dtype())
                .ok_or_else(|| anyhow::anyhow!("Unsupported dtype: {:?}", tensor.dtype()))?;
            let shape: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();
            let data = tensor.data().to_vec();
            result.push(ExtractedTensor { name, shape, dtype, data });
        }
        Ok(result)
    }
}

// ----------------------------------------------------------------------------
// GGUF Extractor
// ----------------------------------------------------------------------------

pub struct GgufExtractor;

impl TensorExtractor for GgufExtractor {
    fn extract_tensors(&self, path: &str) -> Result<Vec<ExtractedTensor>> {
        let mut container = gguf_rs::get_gguf_container(path)
            .with_context(|| format!("Failed to open GGUF file '{}'", path))?;
        let model = container.decode()
            .with_context(|| format!("Failed to decode GGUF model from '{}'", path))?;
        let mut f = File::open(path)?;
        
        let mut result = Vec::new();
        for tensor_info in model.tensors() {
            let ggml_dtype: gguf_rs::GGMLType = tensor_info
                .kind
                .try_into()
                .map_err(|e| anyhow::anyhow!("Unsupported GGUF dtype: {}", e))?;
            
            if let Some(dtype) = gguf_dtype_to_ztensor(&ggml_dtype) {
                let shape: Vec<u64> = tensor_info
                    .shape
                    .iter()
                    .map(|&d| d as u64)
                    .filter(|&d| d > 0)
                    .collect();
                let numel: u64 = shape.iter().product();
                let type_size = dtype_size(dtype);
                let data_size = numel * type_size;
                let mut data = vec![0u8; data_size as usize];
                f.seek(SeekFrom::Start(tensor_info.offset))?;
                f.read_exact(&mut data)?;
                
                result.push(ExtractedTensor {
                    name: tensor_info.name.clone(),
                    shape,
                    dtype,
                    data,
                });
            } else {
                eprintln!(
                    "Warning: Skipping tensor '{}' with unsupported GGUF dtype: {:?}",
                    tensor_info.name, ggml_dtype
                );
            }
        }
        Ok(result)
    }
}

// ----------------------------------------------------------------------------
// Pickle Extractor
// ----------------------------------------------------------------------------

pub struct PickleExtractor;

impl TensorExtractor for PickleExtractor {
    fn extract_tensors(&self, path: &str) -> Result<Vec<ExtractedTensor>> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open Pickle file '{}'", path))?;
        let reader = BufReader::new(file);
        
        let pkl_value: PickleValue = serde_pickle::value_from_reader(reader, Default::default())
            .map_err(|e| anyhow::anyhow!(
                "Failed to deserialize Pickle file '{}': {}\n\
                This often happens with PyTorch .pt/.bin files containing custom objects or large tensors.\n\
                For best results, save your tensors in Python as a dict of numpy arrays or lists, e.g.:\n    \
                import torch, pickle\n    \
                tensors = {{k: v.cpu().numpy() for k, v in torch.load('pytorch_model.bin', map_location='cpu').items()}}\n    \
                with open('model_simple.pkl', 'wb') as f: pickle.dump(tensors, f)\n\
                Then use this tool on 'model_simple.pkl'.",
                path, e
            ))?;
        
        let mut found_tensors: Vec<(String, Vec<u64>, ztensor::DType, Vec<u8>)> = Vec::new();
        pickle::pickle_value_to_ztensor_components("", &pkl_value, &mut found_tensors)
            .map_err(|e| anyhow::anyhow!("Error processing pickle structure from '{}': {}", path, e))?;
        
        if found_tensors.is_empty() {
            bail!("No compatible (numpy/pytorch-like) tensors found in '{}'.", path);
        }
        
        Ok(found_tensors
            .into_iter()
            .map(|(name, shape, dtype, data)| {
                let cleaned_name = if name.is_empty() { "unnamed_tensor".to_string() } else { name };
                ExtractedTensor { name: cleaned_name, shape, dtype, data }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extracted_tensor_clone() {
        let tensor = ExtractedTensor {
            name: "test".to_string(),
            shape: vec![2, 3],
            dtype: ztensor::DType::F32,
            data: vec![1, 2, 3, 4],
        };
        let cloned = tensor.clone();
        assert_eq!(cloned.name, tensor.name);
        assert_eq!(cloned.shape, tensor.shape);
        assert_eq!(cloned.data, tensor.data);
    }
}
