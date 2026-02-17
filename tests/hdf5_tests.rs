#![cfg(feature = "hdf5")]

use std::io::Write;
use std::path::PathBuf;

use tempfile::NamedTempFile;

use ztensor::{DType, Hdf5Reader, TensorReader};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

// ---- Error tests ----

#[test]
fn hdf5_rejects_bad_magic() {
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(b"NOT_HDF5_MAGIC_BYTES").unwrap();
    tmp.flush().unwrap();
    assert!(Hdf5Reader::open(tmp.path()).is_err());
}

#[test]
fn hdf5_rejects_truncated_superblock() {
    let mut tmp = NamedTempFile::new().unwrap();
    // Real HDF5 signature but truncated
    tmp.write_all(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a])
        .unwrap();
    tmp.flush().unwrap();
    assert!(Hdf5Reader::open(tmp.path()).is_err());
}

// ---- Contiguous tests ----

#[test]
fn hdf5_contiguous_simple() {
    let reader = Hdf5Reader::open(fixture("contiguous_simple.h5")).unwrap();
    let keys = reader.keys();
    assert_eq!(keys, vec!["data"]);

    let obj = reader.get("data").unwrap();
    assert_eq!(obj.shape, vec![5]);
    let comp = obj.components.get("data").unwrap();
    assert_eq!(comp.dtype, DType::F32);

    let result: Vec<f32> = reader.read_as("data").unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

// ---- Chunked tests ----

#[test]
fn hdf5_chunked_uncompressed() {
    let reader = Hdf5Reader::open(fixture("chunked_uncompressed.h5")).unwrap();
    let keys = reader.keys();
    assert_eq!(keys, vec!["weight"]);

    let obj = reader.get("weight").unwrap();
    assert_eq!(obj.shape, vec![6, 4]);
    let comp = obj.components.get("data").unwrap();
    assert_eq!(comp.dtype, DType::F32);

    let result: Vec<f32> = reader.read_as("weight").unwrap();
    let expected: Vec<f32> = (0..24).map(|i| i as f32).collect();
    assert_eq!(result, expected);
}

#[test]
fn hdf5_chunked_gzip() {
    let reader = Hdf5Reader::open(fixture("chunked_gzip.h5")).unwrap();

    let obj = reader.get("weight").unwrap();
    assert_eq!(obj.shape, vec![6, 4]);

    let result: Vec<f32> = reader.read_as("weight").unwrap();
    let expected: Vec<f32> = (0..24).map(|i| i as f32).collect();
    assert_eq!(result, expected);
}

#[test]
fn hdf5_chunked_shuffle_gzip() {
    let reader = Hdf5Reader::open(fixture("chunked_shuffle_gzip.h5")).unwrap();

    let obj = reader.get("weight").unwrap();
    assert_eq!(obj.shape, vec![6, 4]);

    let result: Vec<f32> = reader.read_as("weight").unwrap();
    let expected: Vec<f32> = (0..24).map(|i| i as f32).collect();
    assert_eq!(result, expected);
}

#[test]
fn hdf5_chunked_1d() {
    let reader = Hdf5Reader::open(fixture("chunked_1d.h5")).unwrap();

    let obj = reader.get("bias").unwrap();
    assert_eq!(obj.shape, vec![100]);

    let result: Vec<f32> = reader.read_as("bias").unwrap();
    let expected: Vec<f32> = (0..100).map(|i| i as f32).collect();
    assert_eq!(result, expected);
}

// ---- Mixed layout tests ----

#[test]
fn hdf5_mixed_contiguous_and_chunked() {
    let reader = Hdf5Reader::open(fixture("mixed_layouts.h5")).unwrap();

    let mut keys = reader.keys();
    keys.sort();
    assert_eq!(keys, vec!["chunked", "contiguous"]);

    // Contiguous tensor
    let cont_obj = reader.get("contiguous").unwrap();
    assert_eq!(cont_obj.shape, vec![3]);
    let cont: Vec<f32> = reader.read_as("contiguous").unwrap();
    assert_eq!(cont, vec![1.0, 2.0, 3.0]);

    // Chunked + gzip tensor
    let chunk_obj = reader.get("chunked").unwrap();
    assert_eq!(chunk_obj.shape, vec![3, 4]);
    let chunked: Vec<f32> = reader.read_as("chunked").unwrap();
    let expected: Vec<f32> = (0..12).map(|i| i as f32).collect();
    assert_eq!(chunked, expected);
}

// ---- Keras-like nested group tests ----

#[test]
fn hdf5_keras_like_groups() {
    let reader = Hdf5Reader::open(fixture("keras_like.h5")).unwrap();

    let mut keys = reader.keys();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "dense_1.bias",
            "dense_1.kernel",
            "dense_2.bias",
            "dense_2.kernel"
        ]
    );

    // dense_1/kernel: 4x3 of ones
    let obj = reader.get("dense_1.kernel").unwrap();
    assert_eq!(obj.shape, vec![4, 3]);
    let kernel: Vec<f32> = reader.read_as("dense_1.kernel").unwrap();
    assert!(kernel.iter().all(|&v| v == 1.0));

    // dense_1/bias: 3 zeros
    let bias: Vec<f32> = reader.read_as("dense_1.bias").unwrap();
    assert_eq!(bias, vec![0.0, 0.0, 0.0]);

    // dense_2/kernel: 3x2 of 0.5
    let kernel2: Vec<f32> = reader.read_as("dense_2.kernel").unwrap();
    assert!(kernel2.iter().all(|&v| v == 0.5));

    // dense_2/bias: 2 zeros
    let bias2: Vec<f32> = reader.read_as("dense_2.bias").unwrap();
    assert_eq!(bias2, vec![0.0, 0.0]);
}

// ---- Open dispatch test ----

#[test]
fn hdf5_open_dispatch() {
    let reader = ztensor::open(fixture("contiguous_simple.h5")).unwrap();
    let keys = reader.keys();
    assert_eq!(keys, vec!["data"]);

    let obj = reader.get("data").unwrap();
    assert_eq!(obj.shape, vec![5]);

    let data = reader.read_data("data").unwrap();
    let floats: &[f32] = bytemuck::cast_slice(data.as_slice());
    assert_eq!(floats, &[1.0, 2.0, 3.0, 4.0, 5.0]);
}
