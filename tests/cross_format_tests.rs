#![cfg(feature = "all-formats")]

use std::collections::BTreeMap;
use std::io::{Cursor, Seek, SeekFrom};

use half::f16;

use ztensor::{DType, PyTorchReader, Reader, SafeTensorsReader, TensorReader, Writer};

mod common;
use common::data_generators::*;
use common::pytorch_builder::*;
use common::safetensors_builder::*;

/// Write identical data to all three formats, read back, verify bit-exact match.
fn cross_format_verify_f32(name: &str, shape_zt: Vec<u64>, shape_st: Vec<usize>, data: &[f32]) {
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
    let n: usize = data.len();

    // ZTensor
    let mut zt_buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut zt_buf).unwrap();
    w.add(name, &shape_zt, data).unwrap();
    w.finish().unwrap();
    zt_buf.seek(SeekFrom::Start(0)).unwrap();
    let zt_reader = Reader::new(&mut zt_buf).unwrap();
    let zt_data: Vec<f32> = zt_reader.read_as(name).unwrap();

    // SafeTensors
    let st_file = build_safetensors_file(vec![(
        name.into(),
        safetensors::Dtype::F32,
        shape_st,
        raw_bytes.clone(),
    )]);
    let st_reader = SafeTensorsReader::open(st_file.path()).unwrap();
    let st_data: Vec<f32> = st_reader.read_as(name).unwrap();

    // PyTorch
    let shape_pt: Vec<usize> = zt_data
        .len()
        .min(n)
        .max(1)
        .min(n)
        .checked_div(1) // just use [n] for simplicity
        .map(|_| vec![n])
        .unwrap();
    let stride_pt = compute_strides(&shape_pt);
    let specs = vec![PtTensorSpec {
        name: name.into(),
        storage_type: "FloatStorage".into(),
        storage_key: "0".into(),
        shape: shape_pt,
        stride: stride_pt,
        storage_offset: 0,
        numel: n,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let pt_file = build_pytorch_zip(&specs, &storage);
    let pt_reader = PyTorchReader::open(pt_file.path()).unwrap();
    let pt_data: Vec<f32> = pt_reader.read_as(name).unwrap();

    // Verify all three match
    assert_eq!(zt_data, data, "ZTensor data mismatch");
    assert_eq!(st_data, data, "SafeTensors data mismatch");
    assert_eq!(pt_data, data, "PyTorch data mismatch");
}

#[test]
fn cross_f32_2d() {
    let data = make_f32_data(64 * 128);
    cross_format_verify_f32("matrix", vec![64, 128], vec![64, 128], &data);
}

#[test]
fn cross_f16_all_formats() {
    let data = make_f16_data(256);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

    // ZTensor
    let mut zt_buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut zt_buf).unwrap();
    w.add("t", &[256], &data).unwrap();
    w.finish().unwrap();
    zt_buf.seek(SeekFrom::Start(0)).unwrap();
    let zt_reader = Reader::new(&mut zt_buf).unwrap();
    let zt_result: Vec<f16> = zt_reader.read_as("t").unwrap();

    // SafeTensors
    let st_file = build_safetensors_file(vec![(
        "t".into(),
        safetensors::Dtype::F16,
        vec![256],
        raw_bytes.clone(),
    )]);
    let st_reader = SafeTensorsReader::open(st_file.path()).unwrap();
    let st_result: Vec<f16> = st_reader.read_as("t").unwrap();

    // PyTorch
    let specs = vec![PtTensorSpec {
        name: "t".into(),
        storage_type: "HalfStorage".into(),
        storage_key: "0".into(),
        shape: vec![256],
        stride: vec![1],
        storage_offset: 0,
        numel: 256,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let pt_file = build_pytorch_zip(&specs, &storage);
    let pt_reader = PyTorchReader::open(pt_file.path()).unwrap();
    let pt_result: Vec<f16> = pt_reader.read_as("t").unwrap();

    assert_eq!(zt_result, data);
    assert_eq!(st_result, data);
    assert_eq!(pt_result, data);
}

#[test]
fn cross_i32_all_formats() {
    let data = make_i32_data(128);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

    // ZTensor
    let mut zt_buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut zt_buf).unwrap();
    w.add("t", &[128], &data).unwrap();
    w.finish().unwrap();
    zt_buf.seek(SeekFrom::Start(0)).unwrap();
    let zt_reader = Reader::new(&mut zt_buf).unwrap();
    let zt_result: Vec<i32> = zt_reader.read_as("t").unwrap();

    // SafeTensors
    let st_file = build_safetensors_file(vec![(
        "t".into(),
        safetensors::Dtype::I32,
        vec![128],
        raw_bytes.clone(),
    )]);
    let st_reader = SafeTensorsReader::open(st_file.path()).unwrap();
    let st_result: Vec<i32> = st_reader.read_as("t").unwrap();

    // PyTorch
    let specs = vec![PtTensorSpec {
        name: "t".into(),
        storage_type: "IntStorage".into(),
        storage_key: "0".into(),
        shape: vec![128],
        stride: vec![1],
        storage_offset: 0,
        numel: 128,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let pt_file = build_pytorch_zip(&specs, &storage);
    let pt_reader = PyTorchReader::open(pt_file.path()).unwrap();
    let pt_result: Vec<i32> = pt_reader.read_as("t").unwrap();

    assert_eq!(zt_result, data);
    assert_eq!(st_result, data);
    assert_eq!(pt_result, data);
}

#[test]
fn cross_large_tensor() {
    let data = make_f32_data(512 * 512);
    cross_format_verify_f32("big", vec![512, 512], vec![512, 512], &data);
}

#[test]
fn cross_manifest_shapes() {
    let data = make_f32_data(8 * 16);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

    // ZTensor
    let mut zt_buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut zt_buf).unwrap();
    w.add("t", &[8, 16], &data).unwrap();
    w.finish().unwrap();
    zt_buf.seek(SeekFrom::Start(0)).unwrap();
    let zt_reader = Reader::new(&mut zt_buf).unwrap();

    // SafeTensors
    let st_file = build_safetensors_file(vec![(
        "t".into(),
        safetensors::Dtype::F32,
        vec![8, 16],
        raw_bytes.clone(),
    )]);
    let st_reader = SafeTensorsReader::open(st_file.path()).unwrap();

    // PyTorch
    let specs = vec![PtTensorSpec {
        name: "t".into(),
        storage_type: "FloatStorage".into(),
        storage_key: "0".into(),
        shape: vec![8, 16],
        stride: vec![16, 1],
        storage_offset: 0,
        numel: 128,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let pt_file = build_pytorch_zip(&specs, &storage);
    let pt_reader = PyTorchReader::open(pt_file.path()).unwrap();

    // All three should agree on shape and dtype
    let zt_obj = zt_reader.get("t").unwrap();
    let st_obj = st_reader.get("t").unwrap();
    let pt_obj = pt_reader.get("t").unwrap();

    assert_eq!(zt_obj.shape, vec![8, 16]);
    assert_eq!(st_obj.shape, vec![8, 16]);
    assert_eq!(pt_obj.shape, vec![8, 16]);

    assert_eq!(zt_obj.components.get("data").unwrap().dtype, DType::F32);
    assert_eq!(st_obj.components.get("data").unwrap().dtype, DType::F32);
    assert_eq!(pt_obj.components.get("data").unwrap().dtype, DType::F32);
}

#[test]
fn cross_model_checkpoint() {
    // 10-layer model: each layer has weight [32, 32] and bias [32]
    for layer_idx in 0..10 {
        let w_name = format!("layer.{}.weight", layer_idx);
        let b_name = format!("layer.{}.bias", layer_idx);
        let w_data = make_f32_data(32 * 32);
        let b_data = make_f32_data(32);

        let w_bytes: Vec<u8> = bytemuck::cast_slice(&w_data).to_vec();
        let b_bytes: Vec<u8> = bytemuck::cast_slice(&b_data).to_vec();

        // ZTensor
        let mut zt_buf = Cursor::new(Vec::new());
        let mut w = Writer::new(&mut zt_buf).unwrap();
        w.add(&w_name, &[32, 32], &w_data).unwrap();
        w.add(&b_name, &[32], &b_data).unwrap();
        w.finish().unwrap();
        zt_buf.seek(SeekFrom::Start(0)).unwrap();
        let zt_reader = Reader::new(&mut zt_buf).unwrap();

        // SafeTensors
        let st_file = build_safetensors_file(vec![
            (
                w_name.clone(),
                safetensors::Dtype::F32,
                vec![32, 32],
                w_bytes.clone(),
            ),
            (
                b_name.clone(),
                safetensors::Dtype::F32,
                vec![32],
                b_bytes.clone(),
            ),
        ]);
        let st_reader = SafeTensorsReader::open(st_file.path()).unwrap();

        // PyTorch
        let specs = vec![
            PtTensorSpec {
                name: w_name.clone(),
                storage_type: "FloatStorage".into(),
                storage_key: "0".into(),
                shape: vec![32, 32],
                stride: vec![32, 1],
                storage_offset: 0,
                numel: 1024,
            },
            PtTensorSpec {
                name: b_name.clone(),
                storage_type: "FloatStorage".into(),
                storage_key: "1".into(),
                shape: vec![32],
                stride: vec![1],
                storage_offset: 0,
                numel: 32,
            },
        ];
        let mut storage = BTreeMap::new();
        storage.insert("0".into(), w_bytes);
        storage.insert("1".into(), b_bytes);
        let pt_file = build_pytorch_zip(&specs, &storage);
        let pt_reader = PyTorchReader::open(pt_file.path()).unwrap();

        // Verify all three formats produce identical data
        let zt_w: Vec<f32> = zt_reader.read_as(&w_name).unwrap();
        let st_w: Vec<f32> = st_reader.read_as(&w_name).unwrap();
        let pt_w: Vec<f32> = pt_reader.read_as(&w_name).unwrap();
        assert_eq!(zt_w, w_data, "layer {} weight ZT mismatch", layer_idx);
        assert_eq!(st_w, w_data, "layer {} weight ST mismatch", layer_idx);
        assert_eq!(pt_w, w_data, "layer {} weight PT mismatch", layer_idx);

        let zt_b: Vec<f32> = zt_reader.read_as(&b_name).unwrap();
        let st_b: Vec<f32> = st_reader.read_as(&b_name).unwrap();
        let pt_b: Vec<f32> = pt_reader.read_as(&b_name).unwrap();
        assert_eq!(zt_b, b_data, "layer {} bias ZT mismatch", layer_idx);
        assert_eq!(st_b, b_data, "layer {} bias ST mismatch", layer_idx);
        assert_eq!(pt_b, b_data, "layer {} bias PT mismatch", layer_idx);
    }
}

#[test]
fn cross_mixed_dtypes() {
    let f32_data = make_f32_data(64);
    let i64_data = make_i64_data(32);
    let u8_data = make_u8_data(128);

    let f32_bytes: Vec<u8> = bytemuck::cast_slice(&f32_data).to_vec();
    let i64_bytes: Vec<u8> = bytemuck::cast_slice(&i64_data).to_vec();

    // ZTensor
    let mut zt_buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut zt_buf).unwrap();
    w.add("f32", &[64], &f32_data).unwrap();
    w.add("i64", &[32], &i64_data).unwrap();
    w.add("u8", &[128], &u8_data).unwrap();
    w.finish().unwrap();
    zt_buf.seek(SeekFrom::Start(0)).unwrap();
    let zt_reader = Reader::new(&mut zt_buf).unwrap();

    // SafeTensors
    let st_file = build_safetensors_file(vec![
        (
            "f32".into(),
            safetensors::Dtype::F32,
            vec![64],
            f32_bytes.clone(),
        ),
        (
            "i64".into(),
            safetensors::Dtype::I64,
            vec![32],
            i64_bytes.clone(),
        ),
        (
            "u8".into(),
            safetensors::Dtype::U8,
            vec![128],
            u8_data.clone(),
        ),
    ]);
    let st_reader = SafeTensorsReader::open(st_file.path()).unwrap();

    // PyTorch
    let specs = vec![
        PtTensorSpec {
            name: "f32".into(),
            storage_type: "FloatStorage".into(),
            storage_key: "0".into(),
            shape: vec![64],
            stride: vec![1],
            storage_offset: 0,
            numel: 64,
        },
        PtTensorSpec {
            name: "i64".into(),
            storage_type: "LongStorage".into(),
            storage_key: "1".into(),
            shape: vec![32],
            stride: vec![1],
            storage_offset: 0,
            numel: 32,
        },
        PtTensorSpec {
            name: "u8".into(),
            storage_type: "ByteStorage".into(),
            storage_key: "2".into(),
            shape: vec![128],
            stride: vec![1],
            storage_offset: 0,
            numel: 128,
        },
    ];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), f32_bytes);
    storage.insert("1".into(), i64_bytes);
    storage.insert("2".into(), u8_data.clone());
    let pt_file = build_pytorch_zip(&specs, &storage);
    let pt_reader = PyTorchReader::open(pt_file.path()).unwrap();

    // Verify
    assert_eq!(zt_reader.read_as::<f32>("f32").unwrap(), f32_data);
    assert_eq!(st_reader.read_as::<f32>("f32").unwrap(), f32_data);
    assert_eq!(pt_reader.read_as::<f32>("f32").unwrap(), f32_data);

    assert_eq!(zt_reader.read_as::<i64>("i64").unwrap(), i64_data);
    assert_eq!(st_reader.read_as::<i64>("i64").unwrap(), i64_data);
    assert_eq!(pt_reader.read_as::<i64>("i64").unwrap(), i64_data);

    assert_eq!(zt_reader.read_as::<u8>("u8").unwrap(), u8_data);
    assert_eq!(st_reader.read_as::<u8>("u8").unwrap(), u8_data);
    assert_eq!(pt_reader.read_as::<u8>("u8").unwrap(), u8_data);
}
