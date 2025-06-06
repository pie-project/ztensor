// Format detection and dtype/shape helpers

pub fn detect_format_from_extension(input: &str) -> Option<&'static str> {
    let input = input.to_ascii_lowercase();
    if input.ends_with(".safetensor") || input.ends_with(".safetensors") {
        Some("safetensor")
    } else if input.ends_with(".gguf") {
        Some("gguf")
    } else if input.ends_with(".pkl") || input.ends_with(".pickle") {
        Some("pickle")
    } else {
        None
    }
}

pub fn dtype_size(dtype: &ztensor::DType) -> Option<usize> {
    use ztensor::DType::*;
    Some(match dtype {
        Float64 | Int64 | Uint64 => 8,
        Float32 | Int32 | Uint32 => 4,
        BFloat16 | Float16  | Int16 | Uint16 => 2,
        Int8 | Uint8 | Bool => 1,
        _ => return None,
    })
}

pub fn shape_numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}
