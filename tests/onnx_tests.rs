#![cfg(feature = "onnx")]

use std::io::Write;

use half::f16;
use tempfile::NamedTempFile;

use ztensor::{DType, Error, OnnxReader, TensorReader};

mod common;
use common::data_generators::*;
use common::onnx_builder::*;

#[test]
fn onnx_f32_raw_data() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let raw = bytemuck::cast_slice::<f32, u8>(&data);
    let tensor = build_tensor_proto("weight", 1, &[2, 2], raw);
    let file = build_onnx_file(vec![tensor]);

    let reader = OnnxReader::open(file.path()).unwrap();
    assert_eq!(reader.keys(), vec!["weight"]);

    let obj = reader.get("weight").unwrap();
    assert_eq!(obj.shape, vec![2, 2]);
    let comp = obj.components.get("data").unwrap();
    assert_eq!(comp.dtype, DType::F32);

    let result: Vec<f32> = reader.read_as("weight").unwrap();
    assert_eq!(result, data);
}

#[test]
fn onnx_multiple_tensors() {
    let f32_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let i64_data: Vec<i64> = vec![10, -20, 30, 40];
    let u8_data: Vec<u8> = vec![0, 128, 255];

    let t1 = build_tensor_proto("fc.weight", 1, &[3], bytemuck::cast_slice(&f32_data));
    let t2 = build_tensor_proto("embedding", 7, &[4], bytemuck::cast_slice(&i64_data));
    let t3 = build_tensor_proto("mask", 2, &[3], &u8_data);
    let file = build_onnx_file(vec![t1, t2, t3]);

    let reader = OnnxReader::open(file.path()).unwrap();
    assert_eq!(reader.keys().len(), 3);

    assert_eq!(reader.read_as::<f32>("fc.weight").unwrap(), f32_data);
    assert_eq!(reader.read_as::<i64>("embedding").unwrap(), i64_data);
    assert_eq!(reader.read_as::<u8>("mask").unwrap(), u8_data);
}

#[test]
fn onnx_float_data_field() {
    // Test that typed data fields (not raw_data) work
    let data: Vec<f32> = vec![1.5, -2.5, 3.0];
    let tensor = build_tensor_proto_float_data("bias", &[3], &data);
    let file = build_onnx_file(vec![tensor]);

    let reader = OnnxReader::open(file.path()).unwrap();
    let result: Vec<f32> = reader.read_as("bias").unwrap();
    assert_eq!(result, data);
}

#[test]
fn onnx_empty_model() {
    // Model with no initializers
    let file = build_onnx_file(vec![]);
    let reader = OnnxReader::open(file.path()).unwrap();
    assert!(reader.keys().is_empty());
}

#[test]
fn onnx_f16_raw_data() {
    let f16_data = make_f16_data(6);
    let raw = bytemuck::cast_slice::<f16, u8>(&f16_data);
    let tensor = build_tensor_proto("half_weight", 10, &[2, 3], raw);
    let file = build_onnx_file(vec![tensor]);

    let reader = OnnxReader::open(file.path()).unwrap();
    let result: Vec<f16> = reader.read_as("half_weight").unwrap();
    assert_eq!(result, f16_data);
}

#[test]
fn onnx_open_dispatch() {
    let data: Vec<f32> = vec![1.0, 2.0];
    let raw = bytemuck::cast_slice::<f32, u8>(&data);
    let tensor = build_tensor_proto("x", 1, &[2], raw);
    let file = build_onnx_file(vec![tensor]);

    // Use ztensor::open() which dispatches by extension
    let reader = ztensor::open(file.path()).unwrap();
    assert_eq!(reader.keys(), vec!["x"]);
    let td = reader.read_data("x").unwrap();
    let bytes = td.as_slice();
    let result: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result, data);
}

#[test]
fn onnx_error_not_found() {
    let tensor = build_tensor_proto("a", 1, &[1], &[0, 0, 0, 0]);
    let file = build_onnx_file(vec![tensor]);
    let reader = OnnxReader::open(file.path()).unwrap();
    match reader.read_as::<f32>("nonexistent") {
        Err(Error::ObjectNotFound(_)) => {}
        other => panic!("Expected ObjectNotFound, got {:?}", other),
    }
}

#[test]
fn onnx_large_tensor() {
    let data = make_f32_data(4096);
    let raw = bytemuck::cast_slice::<f32, u8>(&data);
    let tensor = build_tensor_proto("big", 1, &[64, 64], raw);
    let file = build_onnx_file(vec![tensor]);

    let reader = OnnxReader::open(file.path()).unwrap();
    let obj = reader.get("big").unwrap();
    assert_eq!(obj.shape, vec![64, 64]);
    let result: Vec<f32> = reader.read_as("big").unwrap();
    assert_eq!(result, data);
}

// ----- Robustness tests -----

#[test]
fn onnx_error_truncated_file() {
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&[0x08, 0x07]).unwrap(); // varint tag + value but no tensors
    tmp.flush().unwrap();
    // Truncated file: opens with empty manifest or returns error, must not panic.
    if let Ok(reader) = OnnxReader::open(tmp.path()) {
        assert!(reader.manifest.objects.is_empty());
    }
}

#[test]
fn onnx_error_empty_file() {
    let tmp = NamedTempFile::new().unwrap();
    // Empty file: opens with empty manifest or returns error, must not panic.
    if let Ok(reader) = OnnxReader::open(tmp.path()) {
        assert!(reader.manifest.objects.is_empty());
    }
}
