#![cfg(feature = "pickle")]

use std::collections::BTreeMap;
use std::io::Write;

use half::f16;
use tempfile::NamedTempFile;

use ztensor::{DType, Error, PyTorchReader, TensorReader};

mod common;
use common::data_generators::*;
use common::pytorch_builder::*;

// ----- Storage type tests (macro-generated) -----

macro_rules! pt_storage_test {
    ($name:ident, $storage:expr, $dtype:expr, $t:ty, $make:expr, $n:expr) => {
        #[test]
        fn $name() {
            let data: Vec<$t> = $make($n);
            let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
            let shape = vec![$n];
            let stride = compute_strides(&shape);
            let specs = vec![PtTensorSpec {
                name: "tensor".into(),
                storage_type: $storage.into(),
                storage_key: "0".into(),
                shape: shape.clone(),
                stride,
                storage_offset: 0,
                numel: $n,
            }];
            let mut storage = BTreeMap::new();
            storage.insert("0".into(), raw_bytes);
            let file = build_pytorch_zip(&specs, &storage);

            let reader = PyTorchReader::open(file.path()).unwrap();
            assert_eq!(reader.tensors().len(), 1);
            let obj = reader.get("tensor").unwrap();
            assert_eq!(obj.shape, vec![$n as u64]);
            assert_eq!(obj.components.get("data").unwrap().dtype, $dtype);
            let result: Vec<$t> = reader.read_as("tensor").unwrap();
            assert_eq!(result, data);
        }
    };
}

pt_storage_test!(
    pt_float_storage,
    "FloatStorage",
    DType::F32,
    f32,
    make_f32_data,
    12
);
pt_storage_test!(
    pt_double_storage,
    "DoubleStorage",
    DType::F64,
    f64,
    make_f64_data,
    8
);
pt_storage_test!(
    pt_half_storage,
    "HalfStorage",
    DType::F16,
    f16,
    make_f16_data,
    16
);
pt_storage_test!(
    pt_long_storage,
    "LongStorage",
    DType::I64,
    i64,
    make_i64_data,
    6
);
pt_storage_test!(
    pt_int_storage,
    "IntStorage",
    DType::I32,
    i32,
    make_i32_data,
    10
);
pt_storage_test!(
    pt_byte_storage,
    "ByteStorage",
    DType::U8,
    u8,
    make_u8_data,
    20
);

#[test]
fn pt_bool_storage() {
    let data = make_bool_data(8);
    let shape = vec![8];
    let stride = compute_strides(&shape);
    let specs = vec![PtTensorSpec {
        name: "tensor".into(),
        storage_type: "BoolStorage".into(),
        storage_key: "0".into(),
        shape: shape.clone(),
        stride,
        storage_offset: 0,
        numel: 8,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), data.clone());
    let file = build_pytorch_zip(&specs, &storage);

    let reader = PyTorchReader::open(file.path()).unwrap();
    let result: Vec<bool> = reader.read_as("tensor").unwrap();
    let expected: Vec<bool> = data.iter().map(|&b| b != 0).collect();
    assert_eq!(result, expected);
}

// ----- Shape tests -----

#[test]
fn pt_1d() {
    let data = make_f32_data(32);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let specs = vec![PtTensorSpec {
        name: "v".into(),
        storage_type: "FloatStorage".into(),
        storage_key: "0".into(),
        shape: vec![32],
        stride: vec![1],
        storage_offset: 0,
        numel: 32,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let file = build_pytorch_zip(&specs, &storage);
    let reader = PyTorchReader::open(file.path()).unwrap();
    assert_eq!(reader.get("v").unwrap().shape, vec![32]);
    assert_eq!(reader.read_as::<f32>("v").unwrap(), data);
}

#[test]
fn pt_2d() {
    let data = make_f32_data(8 * 16);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let specs = vec![PtTensorSpec {
        name: "w".into(),
        storage_type: "FloatStorage".into(),
        storage_key: "0".into(),
        shape: vec![8, 16],
        stride: vec![16, 1],
        storage_offset: 0,
        numel: 128,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let file = build_pytorch_zip(&specs, &storage);
    let reader = PyTorchReader::open(file.path()).unwrap();
    assert_eq!(reader.get("w").unwrap().shape, vec![8, 16]);
    assert_eq!(reader.read_as::<f32>("w").unwrap(), data);
}

#[test]
fn pt_3d() {
    let data = make_f32_data(2 * 3 * 4);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let specs = vec![PtTensorSpec {
        name: "t".into(),
        storage_type: "FloatStorage".into(),
        storage_key: "0".into(),
        shape: vec![2, 3, 4],
        stride: vec![12, 4, 1],
        storage_offset: 0,
        numel: 24,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let file = build_pytorch_zip(&specs, &storage);
    let reader = PyTorchReader::open(file.path()).unwrap();
    assert_eq!(reader.get("t").unwrap().shape, vec![2, 3, 4]);
    assert_eq!(reader.read_as::<f32>("t").unwrap(), data);
}

// ----- State dict patterns -----

#[test]
fn pt_multi_tensor() {
    let w_data = make_f32_data(4 * 8);
    let b_data = make_f32_data(8);
    let e_data = make_f32_data(10 * 4);

    let w_bytes: Vec<u8> = bytemuck::cast_slice(&w_data).to_vec();
    let b_bytes: Vec<u8> = bytemuck::cast_slice(&b_data).to_vec();
    let e_bytes: Vec<u8> = bytemuck::cast_slice(&e_data).to_vec();

    let specs = vec![
        PtTensorSpec {
            name: "weight".into(),
            storage_type: "FloatStorage".into(),
            storage_key: "0".into(),
            shape: vec![4, 8],
            stride: vec![8, 1],
            storage_offset: 0,
            numel: 32,
        },
        PtTensorSpec {
            name: "bias".into(),
            storage_type: "FloatStorage".into(),
            storage_key: "1".into(),
            shape: vec![8],
            stride: vec![1],
            storage_offset: 0,
            numel: 8,
        },
        PtTensorSpec {
            name: "embed".into(),
            storage_type: "FloatStorage".into(),
            storage_key: "2".into(),
            shape: vec![10, 4],
            stride: vec![4, 1],
            storage_offset: 0,
            numel: 40,
        },
    ];

    let mut storage = BTreeMap::new();
    storage.insert("0".into(), w_bytes);
    storage.insert("1".into(), b_bytes);
    storage.insert("2".into(), e_bytes);
    let file = build_pytorch_zip(&specs, &storage);

    let reader = PyTorchReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 3);
    assert_eq!(reader.read_as::<f32>("weight").unwrap(), w_data);
    assert_eq!(reader.read_as::<f32>("bias").unwrap(), b_data);
    assert_eq!(reader.read_as::<f32>("embed").unwrap(), e_data);
}

#[test]
fn pt_nested_keys() {
    let data = make_f32_data(16);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let specs = vec![PtTensorSpec {
        name: "model.layers.0.self_attn.q_proj.weight".into(),
        storage_type: "FloatStorage".into(),
        storage_key: "0".into(),
        shape: vec![4, 4],
        stride: vec![4, 1],
        storage_offset: 0,
        numel: 16,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let file = build_pytorch_zip(&specs, &storage);

    let reader = PyTorchReader::open(file.path()).unwrap();
    let obj = reader
        .get("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(obj.shape, vec![4, 4]);
    assert_eq!(
        reader
            .read_as::<f32>("model.layers.0.self_attn.q_proj.weight")
            .unwrap(),
        data
    );
}

#[test]
fn pt_shared_storage() {
    // Two tensors sharing the same storage with different offsets
    let full_data: Vec<f32> = make_f32_data(12);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&full_data).to_vec();

    let specs = vec![
        PtTensorSpec {
            name: "first_half".into(),
            storage_type: "FloatStorage".into(),
            storage_key: "0".into(),
            shape: vec![6],
            stride: vec![1],
            storage_offset: 0,
            numel: 12, // total storage elements
        },
        PtTensorSpec {
            name: "second_half".into(),
            storage_type: "FloatStorage".into(),
            storage_key: "0".into(),
            shape: vec![6],
            stride: vec![1],
            storage_offset: 6, // skip 6 elements = 24 bytes
            numel: 12,
        },
    ];

    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let file = build_pytorch_zip(&specs, &storage);

    let reader = PyTorchReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 2);

    let first: Vec<f32> = reader.read_as("first_half").unwrap();
    assert_eq!(first, &full_data[..6]);

    let second: Vec<f32> = reader.read_as("second_half").unwrap();
    assert_eq!(second, &full_data[6..]);
}

#[test]
fn pt_nonzero_offset() {
    // Tensor starting at an offset in storage
    let full_data: Vec<f32> = make_f32_data(20);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&full_data).to_vec();

    let specs = vec![PtTensorSpec {
        name: "offset_tensor".into(),
        storage_type: "FloatStorage".into(),
        storage_key: "0".into(),
        shape: vec![2, 5],
        stride: vec![5, 1],
        storage_offset: 10, // skip first 10 elements
        numel: 20,
    }];

    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let file = build_pytorch_zip(&specs, &storage);

    let reader = PyTorchReader::open(file.path()).unwrap();
    let result: Vec<f32> = reader.read_as("offset_tensor").unwrap();
    assert_eq!(result, &full_data[10..]);
}

// ----- Error tests -----

#[test]
fn pt_error_not_found() {
    let data = make_f32_data(4);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let specs = vec![PtTensorSpec {
        name: "exists".into(),
        storage_type: "FloatStorage".into(),
        storage_key: "0".into(),
        shape: vec![4],
        stride: vec![1],
        storage_offset: 0,
        numel: 4,
    }];
    let mut storage = BTreeMap::new();
    storage.insert("0".into(), raw_bytes);
    let file = build_pytorch_zip(&specs, &storage);

    let reader = PyTorchReader::open(file.path()).unwrap();
    match reader.read("missing") {
        Err(Error::ObjectNotFound(_)) => {}
        other => panic!("Expected ObjectNotFound, got {:?}", other),
    }
}

#[test]
fn pt_error_invalid_zip() {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"this is not a zip file").unwrap();
    file.flush().unwrap();

    match PyTorchReader::open(file.path()) {
        Err(Error::InvalidFileStructure(msg)) => {
            assert!(msg.contains("ZIP") || msg.contains("zip"), "msg={}", msg);
        }
        Err(e) => panic!("Expected InvalidFileStructure, got {:?}", e),
        Ok(_) => panic!("Expected error"),
    }
}

#[test]
fn pt_error_no_pickle() {
    let mut file = NamedTempFile::new().unwrap();
    {
        let mut zip = zip::ZipWriter::new(&mut file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file("archive/data/0", options).unwrap();
        zip.write_all(&[0u8; 16]).unwrap();
        zip.finish().unwrap();
    }

    match PyTorchReader::open(file.path()) {
        Err(Error::InvalidFileStructure(msg)) => {
            assert!(msg.contains("pickle") || msg.contains("pkl"), "msg={}", msg);
        }
        Err(e) => panic!("Expected InvalidFileStructure about pickle, got {:?}", e),
        Ok(_) => panic!("Expected error"),
    }
}

#[test]
fn pt_error_empty_state_dict() {
    // Pickle that produces an empty dict
    let mut pickle = Vec::new();
    pickle.push(0x80);
    pickle.push(2); // PROTO 2
    pickle.push(0x7d); // EMPTY_DICT
    pickle.push(0x2e); // STOP

    let mut file = NamedTempFile::new().unwrap();
    {
        let mut zip = zip::ZipWriter::new(&mut file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        zip.start_file("archive/data.pkl", options).unwrap();
        zip.write_all(&pickle).unwrap();
        zip.finish().unwrap();
    }

    match PyTorchReader::open(file.path()) {
        Err(Error::InvalidFileStructure(msg)) => {
            assert!(msg.contains("No tensors"), "msg={}", msg);
        }
        Err(e) => panic!(
            "Expected InvalidFileStructure about no tensors, got {:?}",
            e
        ),
        Ok(_) => panic!("Expected error"),
    }
}
