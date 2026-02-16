#![cfg(feature = "npz")]

use std::io::{Cursor, Write};

use tempfile::NamedTempFile;

use ztensor::{DType, Error, NpzReader, TensorReader};

mod common;
use common::data_generators::*;
use common::npz_builder::*;

#[test]
fn npz_f32_1d() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&data);
    let file = build_npz_file(vec![("weights", "<f4", &[4], bytes)]);
    let reader = NpzReader::open(file.path()).unwrap();

    assert_eq!(reader.keys(), vec!["weights"]);
    let obj = reader.get("weights").unwrap();
    assert_eq!(obj.shape, vec![4]);
    let comp = obj.components.get("data").unwrap();
    assert_eq!(comp.dtype, DType::F32);

    let result: Vec<f32> = reader.read_as("weights").unwrap();
    assert_eq!(result, data);
}

#[test]
fn npz_f32_2d() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&data);
    let file = build_npz_file(vec![("mat", "<f4", &[2, 3], bytes)]);
    let reader = NpzReader::open(file.path()).unwrap();

    let obj = reader.get("mat").unwrap();
    assert_eq!(obj.shape, vec![2, 3]);
    let result: Vec<f32> = reader.read_as("mat").unwrap();
    assert_eq!(result, data);
}

#[test]
fn npz_multiple_dtypes() {
    let f32_data: Vec<f32> = vec![1.0, 2.5, -3.0];
    let i32_data: Vec<i32> = vec![10, -20, 30];
    let f64_data: Vec<f64> = vec![1.5, 2.5];
    let u8_data: Vec<u8> = vec![0, 128, 255];

    let file = build_npz_file(vec![
        ("floats", "<f4", &[3], bytemuck::cast_slice(&f32_data)),
        ("ints", "<i4", &[3], bytemuck::cast_slice(&i32_data)),
        ("doubles", "<f8", &[2], bytemuck::cast_slice(&f64_data)),
        ("bytes", "|u1", &[3], &u8_data),
    ]);
    let reader = NpzReader::open(file.path()).unwrap();

    assert_eq!(reader.keys().len(), 4);
    assert_eq!(reader.read_as::<f32>("floats").unwrap(), f32_data);
    assert_eq!(reader.read_as::<i32>("ints").unwrap(), i32_data);
    assert_eq!(reader.read_as::<f64>("doubles").unwrap(), f64_data);
    assert_eq!(reader.read_as::<u8>("bytes").unwrap(), u8_data);
}

#[test]
fn npz_i64_large() {
    let data = make_i64_data(1024);
    let bytes = bytemuck::cast_slice::<i64, u8>(&data);
    let file = build_npz_file(vec![("big", "<i8", &[1024], bytes)]);
    let reader = NpzReader::open(file.path()).unwrap();

    let result: Vec<i64> = reader.read_as("big").unwrap();
    assert_eq!(result, data);
}

#[test]
fn npz_compressed() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&data);
    let npy_data = build_npy("<f4", &[4], bytes);

    let mut file = NamedTempFile::with_suffix(".npz").unwrap();
    {
        let mut zip = zip::ZipWriter::new(&mut file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("arr.npy", options).unwrap();
        std::io::Write::write_all(&mut zip, &npy_data).unwrap();
        zip.finish().unwrap();
    }
    file.flush().unwrap();

    let reader = NpzReader::open(file.path()).unwrap();
    let result: Vec<f32> = reader.read_as("arr").unwrap();
    assert_eq!(result, data);
}

#[test]
fn npz_open_dispatch() {
    let data: Vec<f32> = vec![1.0, 2.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&data);
    let file = build_npz_file(vec![("x", "<f4", &[2], bytes)]);

    // Use ztensor::open() which dispatches by extension
    let reader = ztensor::open(file.path()).unwrap();
    assert_eq!(reader.keys(), vec!["x"]);
    let td = reader.read_data("x").unwrap();
    let raw = td.as_slice();
    let result: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(result, data);
}

#[test]
fn npz_error_not_found() {
    let file = build_npz_file(vec![("a", "<f4", &[1], &[0, 0, 0, 0])]);
    let reader = NpzReader::open(file.path()).unwrap();
    match reader.read_as::<f32>("nonexistent") {
        Err(Error::ObjectNotFound(_)) => {}
        other => panic!("Expected ObjectNotFound, got {:?}", other),
    }
}

// ----- Robustness tests -----

#[test]
fn npz_rejects_invalid_npy_header() {
    // Create a valid ZIP containing an invalid .npy file
    let buf = Vec::new();
    let cursor = Cursor::new(buf);
    let mut zip = zip::ZipWriter::new(cursor);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file("arr_0.npy", options).unwrap();
    zip.write_all(b"NOT_A_NUMPY_ARRAY").unwrap();
    let cursor = zip.finish().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(cursor.get_ref()).unwrap();
    tmp.flush().unwrap();
    assert!(NpzReader::open(tmp.path()).is_err());
}
