//! Utility functions for the zTensor CLI

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};

/// Get the byte size of a dtype
pub fn dtype_size(dtype: ztensor::DType) -> u64 {
    match dtype {
        ztensor::DType::Float64 | ztensor::DType::Int64 | ztensor::DType::Uint64 => 8,
        ztensor::DType::Float32 | ztensor::DType::Int32 | ztensor::DType::Uint32 => 4,
        ztensor::DType::BFloat16 | ztensor::DType::Float16 | ztensor::DType::Int16 | ztensor::DType::Uint16 => 2,
        ztensor::DType::Int8 | ztensor::DType::Uint8 | ztensor::DType::Bool => 1,
    }
}

/// Convert SafeTensor dtype to zTensor dtype
pub fn safetensor_dtype_to_ztensor(dtype: &safetensors::tensor::Dtype) -> Option<ztensor::DType> {
    match dtype {
        safetensors::tensor::Dtype::F64 => Some(ztensor::DType::Float64),
        safetensors::tensor::Dtype::F32 => Some(ztensor::DType::Float32),
        safetensors::tensor::Dtype::F16 => Some(ztensor::DType::Float16),
        safetensors::tensor::Dtype::BF16 => Some(ztensor::DType::BFloat16),
        safetensors::tensor::Dtype::I64 => Some(ztensor::DType::Int64),
        safetensors::tensor::Dtype::I32 => Some(ztensor::DType::Int32),
        safetensors::tensor::Dtype::I16 => Some(ztensor::DType::Int16),
        safetensors::tensor::Dtype::I8 => Some(ztensor::DType::Int8),
        safetensors::tensor::Dtype::U64 => Some(ztensor::DType::Uint64),
        safetensors::tensor::Dtype::U32 => Some(ztensor::DType::Uint32),
        safetensors::tensor::Dtype::U16 => Some(ztensor::DType::Uint16),
        safetensors::tensor::Dtype::U8 => Some(ztensor::DType::Uint8),
        safetensors::tensor::Dtype::BOOL => Some(ztensor::DType::Bool),
        _ => None,
    }
}

/// Convert GGUF dtype to zTensor dtype
pub fn gguf_dtype_to_ztensor(dtype: &gguf_rs::GGMLType) -> Option<ztensor::DType> {
    match dtype {
        gguf_rs::GGMLType::F64 => Some(ztensor::DType::Float64),
        gguf_rs::GGMLType::F32 => Some(ztensor::DType::Float32),
        gguf_rs::GGMLType::F16 => Some(ztensor::DType::Float16),
        gguf_rs::GGMLType::BF16 => Some(ztensor::DType::BFloat16),
        gguf_rs::GGMLType::I64 => Some(ztensor::DType::Int64),
        gguf_rs::GGMLType::I32 => Some(ztensor::DType::Int32),
        gguf_rs::GGMLType::I16 => Some(ztensor::DType::Int16),
        gguf_rs::GGMLType::I8 => Some(ztensor::DType::Int8),
        _ => None,
    }
}

/// Create a standard progress bar style
pub fn create_progress_style() -> Result<ProgressStyle> {
    ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .map_err(|e| anyhow::anyhow!("Failed to create progress style: {}", e))
        .map(|s| s.progress_chars("#>-"))
}

/// Create a new progress bar with standard styling
pub fn create_progress_bar(len: u64) -> Result<ProgressBar> {
    let pb = ProgressBar::new(len);
    pb.set_style(create_progress_style()?);
    Ok(pb)
}

/// Supported input formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    SafeTensor,
    GGUF,
    Pickle,
}

/// Detect input format from file extension
pub fn detect_format_from_extension(input: &str) -> Option<InputFormat> {
    let input = input.to_ascii_lowercase();
    if input.ends_with(".safetensor") || input.ends_with(".safetensors") {
        Some(InputFormat::SafeTensor)
    } else if input.ends_with(".gguf") {
        Some(InputFormat::GGUF)
    } else if input.ends_with(".pkl") || input.ends_with(".pickle") {
        Some(InputFormat::Pickle)
    } else {
        None
    }
}

/// Detect input format for a list of inputs
pub fn detect_format_for_inputs(inputs: &[String], format: &str) -> Option<InputFormat> {
    if format == "auto" {
        let mut detected: Option<InputFormat> = None;
        for input in inputs {
            let this_format = detect_format_from_extension(input);
            if let Some(fmt) = this_format {
                if let Some(prev) = detected {
                    if prev != fmt {
                        return None; // Mixed formats
                    }
                } else {
                    detected = Some(fmt);
                }
            } else {
                return None;
            }
        }
        detected
    } else {
        match format {
            "safetensor" => Some(InputFormat::SafeTensor),
            "gguf" => Some(InputFormat::GGUF),
            "pickle" => Some(InputFormat::Pickle),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size_64bit() {
        assert_eq!(dtype_size(ztensor::DType::Float64), 8);
        assert_eq!(dtype_size(ztensor::DType::Int64), 8);
        assert_eq!(dtype_size(ztensor::DType::Uint64), 8);
    }

    #[test]
    fn test_dtype_size_32bit() {
        assert_eq!(dtype_size(ztensor::DType::Float32), 4);
        assert_eq!(dtype_size(ztensor::DType::Int32), 4);
        assert_eq!(dtype_size(ztensor::DType::Uint32), 4);
    }

    #[test]
    fn test_dtype_size_16bit() {
        assert_eq!(dtype_size(ztensor::DType::Float16), 2);
        assert_eq!(dtype_size(ztensor::DType::BFloat16), 2);
        assert_eq!(dtype_size(ztensor::DType::Int16), 2);
        assert_eq!(dtype_size(ztensor::DType::Uint16), 2);
    }

    #[test]
    fn test_dtype_size_8bit() {
        assert_eq!(dtype_size(ztensor::DType::Int8), 1);
        assert_eq!(dtype_size(ztensor::DType::Uint8), 1);
        assert_eq!(dtype_size(ztensor::DType::Bool), 1);
    }

    #[test]
    fn test_detect_format_safetensor() {
        assert_eq!(detect_format_from_extension("model.safetensor"), Some(InputFormat::SafeTensor));
        assert_eq!(detect_format_from_extension("model.safetensors"), Some(InputFormat::SafeTensor));
        assert_eq!(detect_format_from_extension("MODEL.SAFETENSOR"), Some(InputFormat::SafeTensor));
    }

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(detect_format_from_extension("model.gguf"), Some(InputFormat::GGUF));
        assert_eq!(detect_format_from_extension("MODEL.GGUF"), Some(InputFormat::GGUF));
    }

    #[test]
    fn test_detect_format_pickle() {
        assert_eq!(detect_format_from_extension("model.pkl"), Some(InputFormat::Pickle));
        assert_eq!(detect_format_from_extension("model.pickle"), Some(InputFormat::Pickle));
    }

    #[test]
    fn test_detect_format_unknown() {
        assert_eq!(detect_format_from_extension("model.pt"), None);
        assert_eq!(detect_format_from_extension("model.bin"), None);
    }

    #[test]
    fn test_detect_format_for_inputs_explicit() {
        let inputs = vec!["a.txt".to_string()];
        assert_eq!(detect_format_for_inputs(&inputs, "safetensor"), Some(InputFormat::SafeTensor));
        assert_eq!(detect_format_for_inputs(&inputs, "gguf"), Some(InputFormat::GGUF));
        assert_eq!(detect_format_for_inputs(&inputs, "pickle"), Some(InputFormat::Pickle));
    }

    #[test]
    fn test_detect_format_for_inputs_mixed() {
        let inputs = vec!["a.safetensors".to_string(), "b.gguf".to_string()];
        assert_eq!(detect_format_for_inputs(&inputs, "auto"), None);
    }

    #[test]
    fn test_safetensor_dtype_to_ztensor() {
        assert_eq!(safetensor_dtype_to_ztensor(&safetensors::tensor::Dtype::F32), Some(ztensor::DType::Float32));
        assert_eq!(safetensor_dtype_to_ztensor(&safetensors::tensor::Dtype::I64), Some(ztensor::DType::Int64));
    }

    #[test]
    fn test_gguf_dtype_to_ztensor() {
        assert_eq!(gguf_dtype_to_ztensor(&gguf_rs::GGMLType::F32), Some(ztensor::DType::Float32));
        assert_eq!(gguf_dtype_to_ztensor(&gguf_rs::GGMLType::I64), Some(ztensor::DType::Int64));
    }
}
