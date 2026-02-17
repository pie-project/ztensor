use std::io::{Cursor, Seek, SeekFrom, Write};

use half::{bf16, f16};
use tempfile::NamedTempFile;

use ztensor::writer::Compression;
use ztensor::{Checksum, DType, Encoding, Error, Format, Reader, Writer};

mod common;
use common::data_generators::*;

// ----- Shape tests -----

#[test]
fn zt_dense_f32_1d() {
    let data = make_f32_data(1024);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("v", &[1024], &data).unwrap();
    w.finish().unwrap();
    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    assert_eq!(r.read_as::<f32>("v").unwrap(), data);
}

#[test]
fn zt_dense_f32_2d() {
    let data = make_f32_data(64 * 128);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("w", &[64, 128], &data).unwrap();
    w.finish().unwrap();
    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    let obj = r.get("w").unwrap();
    assert_eq!(obj.shape, vec![64, 128]);
    assert_eq!(r.read_as::<f32>("w").unwrap(), data);
}

#[test]
fn zt_dense_f32_3d() {
    let data = make_f32_data(3 * 3 * 64);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("k", &[3, 3, 64], &data).unwrap();
    w.finish().unwrap();
    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    assert_eq!(r.get("k").unwrap().shape, vec![3, 3, 64]);
    assert_eq!(r.read_as::<f32>("k").unwrap(), data);
}

#[test]
fn zt_dense_f32_4d() {
    let data = make_f32_data(8 * 3 * 32 * 32);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("x", &[8, 3, 32, 32], &data).unwrap();
    w.finish().unwrap();
    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    assert_eq!(r.get("x").unwrap().shape, vec![8, 3, 32, 32]);
    assert_eq!(r.read_as::<f32>("x").unwrap(), data);
}

#[test]
fn zt_dense_scalar() {
    let data: Vec<f32> = vec![42.0];
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("s", &[], &data).unwrap();
    w.finish().unwrap();
    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    let obj = r.get("s").unwrap();
    assert!(obj.shape.is_empty());
    assert_eq!(r.read_as::<f32>("s").unwrap(), vec![42.0]);
}

#[test]
fn zt_dense_large() {
    let data = make_f32_data(1024 * 1024);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("big", &[1024, 1024], &data).unwrap();
    w.finish().unwrap();
    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    assert_eq!(r.read_as::<f32>("big").unwrap(), data);
}

// ----- All dtype tests (macro-generated) -----

macro_rules! zt_dtype_test {
    ($name:ident, $t:ty, $make:expr, $n:expr) => {
        #[test]
        fn $name() {
            let data: Vec<$t> = $make($n);
            let mut buf = Cursor::new(Vec::new());
            let mut w = Writer::new(&mut buf).unwrap();
            w.add("tensor", &[$n as u64], &data).unwrap();
            w.finish().unwrap();
            buf.seek(SeekFrom::Start(0)).unwrap();
            let r = Reader::new(&mut buf).unwrap();
            let result: Vec<$t> = r.read_as("tensor").unwrap();
            assert_eq!(result, data);
        }
    };
}

zt_dtype_test!(zt_dtype_f64, f64, make_f64_data, 64);
zt_dtype_test!(zt_dtype_f32, f32, make_f32_data, 64);
zt_dtype_test!(zt_dtype_f16, f16, make_f16_data, 64);
zt_dtype_test!(zt_dtype_bf16, bf16, make_bf16_data, 64);
zt_dtype_test!(zt_dtype_i64, i64, make_i64_data, 64);
zt_dtype_test!(zt_dtype_i32, i32, make_i32_data, 64);
zt_dtype_test!(zt_dtype_i16, i16, make_i16_data, 64);
zt_dtype_test!(zt_dtype_i8, i8, make_i8_data, 64);
zt_dtype_test!(zt_dtype_u64, u64, make_u64_data, 64);
zt_dtype_test!(zt_dtype_u32, u32, make_u32_data, 64);
zt_dtype_test!(zt_dtype_u16, u16, make_u16_data, 64);
zt_dtype_test!(zt_dtype_u8, u8, make_u8_data, 64);

#[test]
fn zt_dtype_bool() {
    let data = make_bool_data(64);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add_bytes(
        "tensor",
        vec![64],
        DType::Bool,
        Compression::Raw,
        &data,
        Checksum::None,
    )
    .unwrap();
    w.finish().unwrap();
    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    let result: Vec<bool> = r.read_as("tensor").unwrap();
    let expected: Vec<bool> = data.iter().map(|&b| b != 0).collect();
    assert_eq!(result, expected);
}

// ----- Multi-object tests -----

#[test]
fn zt_multi_object_small() {
    let weight = make_f32_data(64 * 32);
    let bias = make_f32_data(32);
    let embed = make_f32_data(100 * 16);

    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("weight", &[64, 32], &weight).unwrap();
    w.add("bias", &[32], &bias).unwrap();
    w.add("embed", &[100, 16], &embed).unwrap();
    w.finish().unwrap();

    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    assert_eq!(r.tensors().len(), 3);
    assert_eq!(r.read_as::<f32>("weight").unwrap(), weight);
    assert_eq!(r.read_as::<f32>("bias").unwrap(), bias);
    assert_eq!(r.read_as::<f32>("embed").unwrap(), embed);
}

#[test]
fn zt_multi_object_mixed_dtypes() {
    let f32_data = make_f32_data(16);
    let i64_data = make_i64_data(8);
    let u8_data = make_u8_data(32);
    let f16_data = make_f16_data(24);

    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("f32", &[16], &f32_data).unwrap();
    w.add("i64", &[8], &i64_data).unwrap();
    w.add("u8", &[32], &u8_data).unwrap();
    w.add("f16", &[24], &f16_data).unwrap();
    w.finish().unwrap();

    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    assert_eq!(r.tensors().len(), 4);
    assert_eq!(r.read_as::<f32>("f32").unwrap(), f32_data);
    assert_eq!(r.read_as::<i64>("i64").unwrap(), i64_data);
    assert_eq!(r.read_as::<u8>("u8").unwrap(), u8_data);
    assert_eq!(r.read_as::<f16>("f16").unwrap(), f16_data);
}

#[test]
fn zt_multi_object_many() {
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();

    let mut all_data = Vec::new();
    for i in 0..50 {
        let data = make_f32_data(4 * 4);
        w.add(&format!("layer_{:02}.weight", i), &[4, 4], &data)
            .unwrap();
        all_data.push(data);
    }
    w.finish().unwrap();

    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    assert_eq!(r.tensors().len(), 50);
    for (i, data) in all_data.iter().enumerate() {
        let name = format!("layer_{:02}.weight", i);
        assert_eq!(r.read_as::<f32>(&name).unwrap(), *data);
    }
}

// ----- Compression tests -----

#[test]
fn zt_compressed_zstd() {
    let data = make_f32_data(4096);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add_with("c", &[4096], &data)
        .compress(Compression::Zstd(3))
        .write()
        .unwrap();
    w.finish().unwrap();

    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    let obj = r.get("c").unwrap();
    let comp = obj.components.get("data").unwrap();
    assert_eq!(comp.encoding, Encoding::Zstd);
    // Compressed should be smaller than raw
    assert!(comp.length < 4096 * 4);
    assert_eq!(r.read_as::<f32>("c").unwrap(), data);
}

#[test]
fn zt_compressed_vs_raw() {
    let data = make_f32_data(2048);

    let mut raw_buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut raw_buf).unwrap();
    w.add("t", &[2048], &data).unwrap();
    w.finish().unwrap();

    let mut zstd_buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut zstd_buf).unwrap();
    w.add_with("t", &[2048], &data)
        .compress(Compression::Zstd(3))
        .write()
        .unwrap();
    w.finish().unwrap();

    raw_buf.seek(SeekFrom::Start(0)).unwrap();
    zstd_buf.seek(SeekFrom::Start(0)).unwrap();
    let r1 = Reader::new(&mut raw_buf).unwrap();
    let r2 = Reader::new(&mut zstd_buf).unwrap();

    let d1: Vec<f32> = r1.read_as("t").unwrap();
    let d2: Vec<f32> = r2.read_as("t").unwrap();
    assert_eq!(d1, d2, "compressed and raw data must be bit-identical");
}

// ----- Checksum tests -----

#[test]
fn zt_checksum_crc32c() {
    let data = make_u8_data(256);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add_with("c", &[256], &data)
        .checksum(Checksum::Crc32c)
        .write()
        .unwrap();
    w.finish().unwrap();

    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    let obj = r.get("c").unwrap();
    let digest = obj.components.get("data").unwrap().digest.as_ref().unwrap();
    assert!(digest.starts_with("crc32c:0x"), "digest={}", digest);

    // Reading with verification should succeed
    assert_eq!(r.read("c", true).unwrap(), data);
}

#[test]
fn zt_checksum_sha256() {
    let data = make_u8_data(256);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add_with("s", &[256], &data)
        .checksum(Checksum::Sha256)
        .write()
        .unwrap();
    w.finish().unwrap();

    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    let obj = r.get("s").unwrap();
    let digest = obj.components.get("data").unwrap().digest.as_ref().unwrap();
    assert!(digest.starts_with("sha256:"), "digest={}", digest);
    assert_eq!(digest.len(), 7 + 64); // "sha256:" + 64 hex chars
    assert_eq!(r.read("s", true).unwrap(), data);
}

// ----- I/O mode tests -----

#[test]
fn zt_mmap_read() {
    let data = make_f32_data(128);
    let mut file = NamedTempFile::new().unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("t", &[128], &data).unwrap();
        w.finish().unwrap();
    }

    let r = Reader::open_mmap(file.path()).unwrap();
    assert_eq!(r.read_as::<f32>("t").unwrap(), data);

    // Zero-copy slice access
    let slice = r.view("t").unwrap();
    assert_eq!(slice.len(), 128 * 4);
    let typed = r.view_as::<f32>("t").unwrap();
    assert_eq!(typed, &data[..]);
}

#[test]
fn zt_stream_read() {
    let data = make_f32_data(64);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("t", &[64], &data).unwrap();
    w.finish().unwrap();

    let bytes = buf.into_inner();
    let reader = Reader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(reader.read_as::<f32>("t").unwrap(), data);
}

#[test]
fn zt_open_any_v1() {
    let data = make_f32_data(32);
    let mut file = NamedTempFile::new().unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("t", &[32], &data).unwrap();
        w.finish().unwrap();
    }

    let r = Reader::open_any_path(file.path()).unwrap();
    assert_eq!(r.manifest.version, "1.2.0");
    assert_eq!(r.read_as::<f32>("t").unwrap(), data);
}

// ----- Manifest correctness -----

#[test]
fn zt_manifest_shapes_and_dtypes() {
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("a", &[10, 20], &make_f32_data(200)).unwrap();
    w.add("b", &[5], &make_i64_data(5)).unwrap();
    w.finish().unwrap();

    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();

    let a = r.get("a").unwrap();
    assert_eq!(a.shape, vec![10, 20]);
    assert_eq!(a.format, Format::Dense);
    assert_eq!(a.components.get("data").unwrap().dtype, DType::F32);

    let b = r.get("b").unwrap();
    assert_eq!(b.shape, vec![5]);
    assert_eq!(b.components.get("data").unwrap().dtype, DType::I64);
}

#[test]
fn zt_manifest_offsets_aligned() {
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    // Write multiple objects to check alignment
    w.add("a", &[7], &make_u8_data(7)).unwrap();
    w.add("b", &[13], &make_u8_data(13)).unwrap();
    w.add("c", &[1], &[42.0f32]).unwrap();
    w.finish().unwrap();

    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    for (_, obj) in r.tensors() {
        for (_, comp) in &obj.components {
            assert_eq!(
                comp.offset % 64,
                0,
                "offset {} is not 64-byte aligned",
                comp.offset
            );
        }
    }
}

// ----- Error tests -----

#[test]
fn zt_error_object_not_found() {
    let mut buf = Cursor::new(Vec::new());
    let w = Writer::new(&mut buf).unwrap();
    w.finish().unwrap();
    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    match r.read("missing", true) {
        Err(Error::ObjectNotFound(name)) => assert_eq!(name, "missing"),
        other => panic!("Expected ObjectNotFound, got {:?}", other),
    }
}

#[test]
fn zt_error_type_mismatch() {
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("f", &[4], &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    w.finish().unwrap();
    buf.seek(SeekFrom::Start(0)).unwrap();
    let r = Reader::new(&mut buf).unwrap();
    match r.read_as::<i32>("f") {
        Err(Error::TypeMismatch { .. }) => {}
        other => panic!("Expected TypeMismatch, got {:?}", other),
    }
}

#[test]
fn zt_error_invalid_file() {
    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03];
    let mut buf = Cursor::new(garbage);
    match Reader::new(&mut buf) {
        Err(Error::InvalidMagicNumber { .. }) => {}
        Err(e) => panic!("Expected InvalidMagicNumber, got {:?}", e),
        Ok(_) => panic!("Expected error"),
    }
}

#[test]
fn zt_error_empty_file() {
    let mut buf = Cursor::new(Vec::new());
    match Reader::new(&mut buf) {
        Err(_) => {} // Should fail (UnexpectedEof or similar)
        Ok(_) => panic!("Expected error for empty file"),
    }
}

// ----- Writer overflow tests -----

#[test]
fn writer_rejects_shape_overflow() {
    let mut buf = Cursor::new(Vec::new());
    let mut writer = Writer::new(&mut buf).unwrap();
    let data = vec![0u8; 8];
    let result = writer.add_bytes(
        "t",
        vec![u64::MAX, 2],
        DType::U8,
        Compression::Raw,
        &data,
        Checksum::None,
    );
    assert!(result.is_err(), "shape product should overflow");
}

#[test]
fn writer_rejects_byte_size_overflow() {
    let mut buf = Cursor::new(Vec::new());
    let mut writer = Writer::new(&mut buf).unwrap();
    let data = vec![0u8; 8];
    // Shape product fits u64, but num_elements * dtype_size overflows
    let result = writer.add_bytes(
        "t",
        vec![u64::MAX / 2],
        DType::F64,
        Compression::Raw,
        &data,
        Checksum::None,
    );
    assert!(result.is_err(), "byte size should overflow");
}

// ----- Append tests -----

#[test]
fn zt_append_single_tensor() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[3], &[1.0f32, 2.0, 3.0]).unwrap();
        w.add("b", &[2], &[4.0f32, 5.0]).unwrap();
        w.finish().unwrap();
    }

    {
        let mut w = Writer::append(file.path()).unwrap();
        w.add("c", &[2], &[6.0f32, 7.0]).unwrap();
        w.finish().unwrap();
    }

    let r = Reader::open(file.path()).unwrap();
    assert_eq!(r.tensors().len(), 3);
    assert_eq!(r.read_as::<f32>("a").unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(r.read_as::<f32>("b").unwrap(), vec![4.0, 5.0]);
    assert_eq!(r.read_as::<f32>("c").unwrap(), vec![6.0, 7.0]);
}

#[test]
fn zt_append_preserves_existing() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let original_data = make_f32_data(1024);
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("weights", &[1024], &original_data).unwrap();
        w.finish().unwrap();
    }

    {
        let mut w = Writer::append(file.path()).unwrap();
        w.add("bias", &[4], &[0.1f32, 0.2, 0.3, 0.4]).unwrap();
        w.finish().unwrap();
    }

    let r = Reader::open(file.path()).unwrap();
    assert_eq!(r.read_as::<f32>("weights").unwrap(), original_data);
    assert_eq!(r.read_as::<f32>("bias").unwrap(), vec![0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn zt_append_duplicate_name_errors() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[2], &[1.0f32, 2.0]).unwrap();
        w.finish().unwrap();
    }

    let mut w = Writer::append(file.path()).unwrap();
    let result = w.add("a", &[2], &[3.0f32, 4.0]);
    assert!(result.is_err(), "duplicate name should error");
}

#[test]
fn zt_append_compressed() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let data1 = make_f32_data(256);
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add_with("t1", &[256], &data1)
            .compress(Compression::Zstd(3))
            .write()
            .unwrap();
        w.finish().unwrap();
    }

    let data2 = make_f32_data(128);
    {
        let mut w = Writer::append(file.path()).unwrap();
        w.add_with("t2", &[128], &data2)
            .compress(Compression::Zstd(3))
            .write()
            .unwrap();
        w.finish().unwrap();
    }

    let r = Reader::open(file.path()).unwrap();
    assert_eq!(r.read_as::<f32>("t1").unwrap(), data1);
    assert_eq!(r.read_as::<f32>("t2").unwrap(), data2);
}

#[test]
fn zt_append_mmap_readable() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[4], &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        w.finish().unwrap();
    }

    {
        let mut w = Writer::append(file.path()).unwrap();
        w.add("b", &[3], &[5.0f32, 6.0, 7.0]).unwrap();
        w.finish().unwrap();
    }

    // Verify mmap reader also works on appended file
    let r = Reader::open_mmap(file.path()).unwrap();
    assert_eq!(r.read_as::<f32>("a").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(r.read_as::<f32>("b").unwrap(), vec![5.0, 6.0, 7.0]);
}

// ----- Remove tests -----

#[test]
fn zt_remove_single_tensor() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let output = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[2], &[1.0f32, 2.0]).unwrap();
        w.add("b", &[3], &[3.0f32, 4.0, 5.0]).unwrap();
        w.add("c", &[1], &[6.0f32]).unwrap();
        w.finish().unwrap();
    }

    ztensor::remove_tensors(file.path(), output.path(), &["b"]).unwrap();

    let r = Reader::open(output.path()).unwrap();
    assert_eq!(r.tensors().len(), 2);
    assert_eq!(r.read_as::<f32>("a").unwrap(), vec![1.0, 2.0]);
    assert_eq!(r.read_as::<f32>("c").unwrap(), vec![6.0]);
    assert!(r.get("b").is_none());
}

#[test]
fn zt_remove_multiple_tensors() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let output = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[2], &[1.0f32, 2.0]).unwrap();
        w.add("b", &[2], &[3.0f32, 4.0]).unwrap();
        w.add("c", &[2], &[5.0f32, 6.0]).unwrap();
        w.finish().unwrap();
    }

    ztensor::remove_tensors(file.path(), output.path(), &["a", "c"]).unwrap();

    let r = Reader::open(output.path()).unwrap();
    assert_eq!(r.tensors().len(), 1);
    assert_eq!(r.read_as::<f32>("b").unwrap(), vec![3.0, 4.0]);
}

#[test]
fn zt_remove_nonexistent_errors() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let output = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[2], &[1.0f32, 2.0]).unwrap();
        w.finish().unwrap();
    }

    match ztensor::remove_tensors(file.path(), output.path(), &["nonexistent"]) {
        Err(Error::ObjectNotFound(name)) => assert_eq!(name, "nonexistent"),
        other => panic!("Expected ObjectNotFound, got {:?}", other),
    }
}

// ----- Replace tests -----

#[test]
fn zt_replace_basic() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let original = [1.0f32, 2.0, 3.0, 4.0];
    let replacement = [10.0f32, 20.0, 30.0, 40.0];
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("t", &[4], &original).unwrap();
        w.finish().unwrap();
    }

    ztensor::replace_tensor(file.path(), "t", bytemuck::cast_slice(&replacement)).unwrap();

    let r = Reader::open(file.path()).unwrap();
    assert_eq!(r.read_as::<f32>("t").unwrap(), replacement);
}

#[test]
fn zt_replace_preserves_other_tensors() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let data_a = make_f32_data(64);
    let data_b = make_f32_data(32);
    let new_b: Vec<f32> = (0..32).map(|i| i as f32 * 100.0).collect();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[64], &data_a).unwrap();
        w.add("b", &[32], &data_b).unwrap();
        w.finish().unwrap();
    }

    ztensor::replace_tensor(file.path(), "b", bytemuck::cast_slice(&new_b)).unwrap();

    let r = Reader::open(file.path()).unwrap();
    assert_eq!(
        r.read_as::<f32>("a").unwrap(),
        data_a,
        "other tensor should be unchanged"
    );
    assert_eq!(r.read_as::<f32>("b").unwrap(), new_b);
}

#[test]
fn zt_replace_recomputes_crc32c() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let original = [1.0f32, 2.0, 3.0];
    let replacement = [4.0f32, 5.0, 6.0];
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add_with("t", &[3], &original)
            .checksum(Checksum::Crc32c)
            .write()
            .unwrap();
        w.finish().unwrap();
    }

    let old_digest = {
        let r = Reader::open(file.path()).unwrap();
        r.get("t")
            .unwrap()
            .components
            .get("data")
            .unwrap()
            .digest
            .clone()
    };

    ztensor::replace_tensor(file.path(), "t", bytemuck::cast_slice(&replacement)).unwrap();

    let r = Reader::open(file.path()).unwrap();
    let new_digest = r
        .get("t")
        .unwrap()
        .components
        .get("data")
        .unwrap()
        .digest
        .clone();
    assert_ne!(old_digest, new_digest, "checksum should be recomputed");
    assert!(new_digest.unwrap().starts_with("crc32c:0x"));
    // Verify data reads correctly (checksum verification is automatic)
    assert_eq!(r.read_as::<f32>("t").unwrap(), replacement);
}

#[test]
fn zt_replace_recomputes_sha256() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let original = [1.0f32, 2.0, 3.0];
    let replacement = [4.0f32, 5.0, 6.0];
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add_with("t", &[3], &original)
            .checksum(Checksum::Sha256)
            .write()
            .unwrap();
        w.finish().unwrap();
    }

    ztensor::replace_tensor(file.path(), "t", bytemuck::cast_slice(&replacement)).unwrap();

    let r = Reader::open(file.path()).unwrap();
    let digest = r
        .get("t")
        .unwrap()
        .components
        .get("data")
        .unwrap()
        .digest
        .as_ref()
        .unwrap();
    assert!(digest.starts_with("sha256:"));
    assert_eq!(r.read_as::<f32>("t").unwrap(), replacement);
}

#[test]
fn zt_replace_size_mismatch_errors() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("t", &[4], &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        w.finish().unwrap();
    }

    // Too short
    let result = ztensor::replace_tensor(file.path(), "t", &[0u8; 8]);
    assert!(result.is_err(), "wrong byte size should error");

    // Too long
    let result = ztensor::replace_tensor(file.path(), "t", &[0u8; 32]);
    assert!(result.is_err(), "wrong byte size should error");
}

#[test]
fn zt_replace_compressed_errors() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let data = make_f32_data(256);
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add_with("t", &[256], &data)
            .compress(Compression::Zstd(3))
            .write()
            .unwrap();
        w.finish().unwrap();
    }

    let new_data = vec![0u8; 256 * 4];
    let result = ztensor::replace_tensor(file.path(), "t", &new_data);
    assert!(
        result.is_err(),
        "compressed tensor should not be replaceable"
    );
}

#[test]
fn zt_replace_nonexistent_errors() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[2], &[1.0f32, 2.0]).unwrap();
        w.finish().unwrap();
    }

    match ztensor::replace_tensor(file.path(), "missing", &[0u8; 8]) {
        Err(Error::ObjectNotFound(name)) => assert_eq!(name, "missing"),
        other => panic!("Expected ObjectNotFound, got {:?}", other),
    }
}

#[test]
fn zt_replace_mmap_readable() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    let original = [1.0f32, 2.0, 3.0];
    let replacement = [7.0f32, 8.0, 9.0];
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("t", &[3], &original).unwrap();
        w.finish().unwrap();
    }

    ztensor::replace_tensor(file.path(), "t", bytemuck::cast_slice(&replacement)).unwrap();

    // Verify mmap reader works on replaced file
    let r = Reader::open_mmap(file.path()).unwrap();
    assert_eq!(r.view_as::<f32>("t").unwrap(), &replacement[..]);
}

#[test]
fn zt_replace_after_append() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[3], &[1.0f32, 2.0, 3.0]).unwrap();
        w.finish().unwrap();
    }
    {
        let mut w = Writer::append(file.path()).unwrap();
        w.add("b", &[2], &[4.0f32, 5.0]).unwrap();
        w.finish().unwrap();
    }

    // Replace the appended tensor
    let new_b = [40.0f32, 50.0];
    ztensor::replace_tensor(file.path(), "b", bytemuck::cast_slice(&new_b)).unwrap();

    let r = Reader::open(file.path()).unwrap();
    assert_eq!(r.read_as::<f32>("a").unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(r.read_as::<f32>("b").unwrap(), new_b);
}

#[test]
fn zt_replace_multiple_tensors() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("a", &[2], &[1.0f32, 2.0]).unwrap();
        w.add("b", &[2], &[3.0f32, 4.0]).unwrap();
        w.add("c", &[2], &[5.0f32, 6.0]).unwrap();
        w.finish().unwrap();
    }

    let new_a = [10.0f32, 20.0];
    let new_c = [50.0f32, 60.0];
    ztensor::replace_tensor(file.path(), "a", bytemuck::cast_slice(&new_a)).unwrap();
    ztensor::replace_tensor(file.path(), "c", bytemuck::cast_slice(&new_c)).unwrap();

    let r = Reader::open(file.path()).unwrap();
    assert_eq!(r.read_as::<f32>("a").unwrap(), new_a);
    assert_eq!(
        r.read_as::<f32>("b").unwrap(),
        vec![3.0, 4.0],
        "untouched tensor unchanged"
    );
    assert_eq!(r.read_as::<f32>("c").unwrap(), new_c);
}

#[test]
fn zt_replace_same_tensor_twice() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add("t", &[3], &[1.0f32, 2.0, 3.0]).unwrap();
        w.finish().unwrap();
    }

    let v1 = [10.0f32, 20.0, 30.0];
    let v2 = [100.0f32, 200.0, 300.0];
    ztensor::replace_tensor(file.path(), "t", bytemuck::cast_slice(&v1)).unwrap();
    ztensor::replace_tensor(file.path(), "t", bytemuck::cast_slice(&v2)).unwrap();

    let r = Reader::open(file.path()).unwrap();
    assert_eq!(r.read_as::<f32>("t").unwrap(), v2);
}

#[test]
fn zt_replace_sparse_errors() {
    let mut file = NamedTempFile::with_suffix(".zt").unwrap();
    {
        let mut w = Writer::new(&mut file).unwrap();
        w.add_csr(
            "sparse",
            vec![2, 3],
            DType::F32,
            &[1.0f32, 2.0],
            &[1u64, 2],
            &[0u64, 1, 2],
            Compression::Raw,
            Checksum::None,
        )
        .unwrap();
        w.finish().unwrap();
    }

    let result = ztensor::replace_tensor(file.path(), "sparse", &[0u8; 8]);
    assert!(result.is_err(), "sparse tensor should not be replaceable");
}
