//! Pickle parsing helpers for the zTensor CLI

use serde_pickle::Value as PickleValue;

/// Maximum recursion depth for pickle parsing to prevent stack overflow
pub const MAX_PICKLE_DEPTH: usize = 50;

/// Convert PickleValue keys to strings for tensor naming
pub fn key_to_string(key_val: &serde_pickle::HashableValue) -> String {
    use serde_pickle::HashableValue;
    match key_val {
        HashableValue::String(s) => s.clone(),
        HashableValue::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
        HashableValue::I64(i) => i.to_string(),
        HashableValue::F64(f) => f.to_string(),
        HashableValue::Bool(b) => b.to_string(),
        HashableValue::Tuple(items) => items
            .iter()
            .map(key_to_string)
            .collect::<Vec<_>>()
            .join("_"),
        _ => format!("key_{:?}", key_val).replace(|c: char| !c.is_alphanumeric() && c != '_', "_"),
    }
}

/// Try to parse a PickleValue as a potential tensor
pub fn try_parse_pickle_tensor(
    value: &PickleValue,
) -> Result<Option<(Vec<u64>, ztensor::DType, Vec<u8>)>, String> {
    let mut data_bytes = Vec::new();
    let mut inferred_dtype: Option<ztensor::DType> = None;
    let mut shape = Vec::new();

    fn ensure_shape_depth(shape: &Vec<u64>, depth: usize) -> Result<(), String> {
        if shape.len() != depth {
            Err(format!(
                "Dimension mismatch: expected scalar at depth {}, but tensor shape is deeper. Shape: {:?}, Depth: {}",
                depth, shape, depth
            ))
        } else {
            Ok(())
        }
    }

    fn collect_elements(
        current_value: &PickleValue,
        current_depth: usize,
        shape: &mut Vec<u64>,
        data_bytes: &mut Vec<u8>,
        dtype: &mut Option<ztensor::DType>,
    ) -> Result<(), String> {
        // Depth limit to prevent stack overflow on malicious/deeply nested files
        if current_depth > super::pickle::MAX_PICKLE_DEPTH {
            return Err(format!(
                "Maximum recursion depth ({}) exceeded while parsing pickle tensor",
                super::pickle::MAX_PICKLE_DEPTH
            ));
        }
        
        match current_value {
            PickleValue::List(items) | PickleValue::Tuple(items) => {
                if current_depth == 0 && items.is_empty() {
                    shape.push(0);
                    if dtype.is_none() {
                        *dtype = Some(ztensor::DType::F32);
                    }
                    return Ok(());
                }
                if items.is_empty() {
                    return Err("Empty list/tuple found within tensor structure.".to_string());
                }

                if shape.len() == current_depth {
                    shape.push(items.len() as u64);
                } else if shape[current_depth] != items.len() as u64 {
                    return Err(format!(
                        "Jagged tensor: dimension mismatch at depth {}. Expected {}, got {}. Shape: {:?}",
                        current_depth,
                        shape[current_depth],
                        items.len(),
                        shape
                    ));
                }

                for item in items {
                    collect_elements(item, current_depth + 1, shape, data_bytes, dtype)?;
                }
                Ok(())
            }
            PickleValue::F64(f) => {
                if dtype.is_none() {
                    *dtype = Some(ztensor::DType::F64);
                }
                if *dtype != Some(ztensor::DType::F64) {
                    return Err("Mixed data types: expected Float64".to_string());
                }
                ensure_shape_depth(shape, current_depth)?;
                data_bytes.extend_from_slice(&f.to_le_bytes());
                Ok(())
            }
            PickleValue::I64(i) => {
                if dtype.is_none() {
                    *dtype = Some(ztensor::DType::I64);
                }
                if *dtype != Some(ztensor::DType::I64) {
                    return Err("Mixed data types: expected Int64". to_string());
                }
                ensure_shape_depth(shape, current_depth)?;
                data_bytes.extend_from_slice(&i.to_le_bytes());
                Ok(())
            }
            PickleValue::Bool(b) => {
                if dtype.is_none() {
                    *dtype = Some(ztensor::DType::Bool);
                }
                if *dtype != Some(ztensor::DType::Bool) {
                    return Err("Mixed data types: expected Bool".to_string());
                }
                ensure_shape_depth(shape, current_depth)?;
                data_bytes.push(if *b { 1 } else { 0 });
                Ok(())
            }
            _ => Err(format!(
                "Unsupported data type in tensor: {:?}",
                current_value
            )),
        }
    }

    match collect_elements(value, 0, &mut shape, &mut data_bytes, &mut inferred_dtype) {
        Ok(()) => {
            if let Some(dt) = inferred_dtype {
                Ok(Some((shape, dt, data_bytes)))
            } else {
                Ok(None)
            }
        }
        Err(_) => Ok(None),
    }
}

/// Recursively traverse PickleValue, identify tensors, and flatten names
pub fn pickle_value_to_ztensor_components(
    prefix: &str,
    value: &PickleValue,
    tensors: &mut Vec<(String, Vec<u64>, ztensor::DType, Vec<u8>)>,
) -> Result<(), String> {
    match value {
        PickleValue::Dict(items) => {
            for (key_val, sub_value) in items {
                let key_str = key_to_string(key_val);
                let new_prefix = if prefix.is_empty() {
                    key_str
                } else {
                    format!("{}.{}", prefix, key_str)
                };
                pickle_value_to_ztensor_components(&new_prefix, sub_value, tensors)?;
            }
        }
        _ => {
            if let Some((shape, dtype, data)) = try_parse_pickle_tensor(value)? {
                if !data.is_empty() || shape == vec![0] {
                    let tensor_name = if prefix.is_empty() {
                        "tensor".to_string()
                    } else {
                        prefix.to_string()
                    };
                    tensors.push((tensor_name, shape, dtype, data));
                }
            } else {
                if let PickleValue::List(sub_items) | PickleValue::Tuple(sub_items) = value {
                    for (idx, item) in sub_items.iter().enumerate() {
                        let new_prefix = if prefix.is_empty() {
                            format!("item_{}", idx)
                        } else {
                            format!("{}.{}", prefix, idx)
                        };
                        pickle_value_to_ztensor_components(&new_prefix, item, tensors)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_max_pickle_depth_constant() {
        assert_eq!(MAX_PICKLE_DEPTH, 50);
    }

    #[test]
    fn test_key_to_string_string() {
        let key = serde_pickle::HashableValue::String("test_key".to_string());
        assert_eq!(key_to_string(&key), "test_key");
    }

    #[test]
    fn test_key_to_string_int() {
        let key = serde_pickle::HashableValue::I64(42);
        assert_eq!(key_to_string(&key), "42");
    }

    #[test]
    fn test_key_to_string_bool() {
        let key = serde_pickle::HashableValue::Bool(true);
        assert_eq!(key_to_string(&key), "true");
    }

    #[test]
    fn test_key_to_string_tuple() {
        let key = serde_pickle::HashableValue::Tuple(vec![
            serde_pickle::HashableValue::String("a".to_string()),
            serde_pickle::HashableValue::I64(1),
        ]);
        assert_eq!(key_to_string(&key), "a_1");
    }

    #[test]
    fn test_try_parse_pickle_tensor_float64_list() {
        let value = PickleValue::List(vec![
            PickleValue::F64(1.0),
            PickleValue::F64(2.0),
            PickleValue::F64(3.0),
        ]);
        let result = try_parse_pickle_tensor(&value).unwrap();
        assert!(result.is_some());
        let (shape, dtype, data) = result.unwrap();
        assert_eq!(shape, vec![3]);
        assert_eq!(dtype, ztensor::DType::F64);
        assert_eq!(data.len(), 24);
    }

    #[test]
    fn test_try_parse_pickle_tensor_bool() {
        let value = PickleValue::List(vec![
            PickleValue::Bool(true),
            PickleValue::Bool(false),
        ]);
        let result = try_parse_pickle_tensor(&value).unwrap();
        assert!(result.is_some());
        let (shape, dtype, data) = result.unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(dtype, ztensor::DType::Bool);
        assert_eq!(data, vec![1, 0]);
    }

    #[test]
    fn test_try_parse_pickle_tensor_empty() {
        let value = PickleValue::List(vec![]);
        let result = try_parse_pickle_tensor(&value).unwrap();
        assert!(result.is_some());
        let (shape, dtype, data) = result.unwrap();
        assert_eq!(shape, vec![0]);
        assert_eq!(dtype, ztensor::DType::F32);
        assert!(data.is_empty());
    }

    #[test]
    fn test_try_parse_pickle_tensor_mixed_types() {
        let value = PickleValue::List(vec![
            PickleValue::F64(1.0),
            PickleValue::I64(2),
        ]);
        let result = try_parse_pickle_tensor(&value).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_pickle_value_to_ztensor_components_dict() {
        let mut dict = BTreeMap::new();
        dict.insert(
            serde_pickle::HashableValue::String("tensor1".to_string()),
            PickleValue::List(vec![PickleValue::F64(1.0), PickleValue::F64(2.0)]),
        );
        let value = PickleValue::Dict(dict);
        
        let mut tensors = Vec::new();
        pickle_value_to_ztensor_components("", &value, &mut tensors).unwrap();
        
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].0, "tensor1");
    }
}
