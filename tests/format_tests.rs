//! Comprehensive integration tests for all tensor formats: ZTensor, SafeTensors, PyTorch.
//!
//! Run with: `cargo test --features all-formats --test format_tests`

#![cfg(feature = "all-formats")]

use std::collections::BTreeMap;
use std::io::{Cursor, Seek, SeekFrom, Write};

use half::{bf16, f16};
use tempfile::NamedTempFile;

use ztensor::writer::Compression;
use ztensor::{
    Checksum, DType, Encoding, Error, Format, GgufReader, Hdf5Reader, PyTorchReader,
    SafeTensorsReader, Reader, TensorReader, Writer, NpzReader, OnnxReader,
};

// =========================================================================
// Section 1: Shared Test Helpers
// =========================================================================

fn make_f64_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 * 0.1).collect()
}
fn make_f32_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.1).collect()
}
fn make_f16_data(n: usize) -> Vec<f16> {
    (0..n).map(|i| f16::from_f32(i as f32 * 0.1)).collect()
}
fn make_bf16_data(n: usize) -> Vec<bf16> {
    (0..n).map(|i| bf16::from_f32(i as f32 * 0.1)).collect()
}
fn make_i64_data(n: usize) -> Vec<i64> {
    (0..n).map(|i| i as i64 - (n / 2) as i64).collect()
}
fn make_i32_data(n: usize) -> Vec<i32> {
    (0..n).map(|i| i as i32 - (n / 2) as i32).collect()
}
fn make_i16_data(n: usize) -> Vec<i16> {
    (0..n).map(|i| (i as i16).wrapping_mul(7)).collect()
}
fn make_i8_data(n: usize) -> Vec<i8> {
    (0..n).map(|i| (i % 128) as i8).collect()
}
fn make_u64_data(n: usize) -> Vec<u64> {
    (0..n).map(|i| i as u64 * 3).collect()
}
fn make_u32_data(n: usize) -> Vec<u32> {
    (0..n).map(|i| i as u32 * 5).collect()
}
fn make_u16_data(n: usize) -> Vec<u16> {
    (0..n).map(|i| i as u16 * 7).collect()
}
fn make_u8_data(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 256) as u8).collect()
}
fn make_bool_data(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 2) as u8).collect()
}

// ----- SafeTensors file builder -----

fn build_safetensors_file(
    tensors: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>,
) -> NamedTempFile {
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                name.clone(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();

    let serialized = safetensors::tensor::serialize(views, &None).unwrap();
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(&serialized).unwrap();
    file.flush().unwrap();
    file
}

// ----- PyTorch pickle builder -----

struct PtTensorSpec {
    name: String,
    storage_type: String,
    storage_key: String,
    shape: Vec<usize>,
    stride: Vec<usize>,
    storage_offset: usize,
    numel: usize,
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn push_int(p: &mut Vec<u8>, val: usize) {
    if val <= 255 {
        p.push(0x4b); // BININT1
        p.push(val as u8);
    } else if val <= 65535 {
        p.push(0x4d); // BININT2
        p.extend_from_slice(&(val as u16).to_le_bytes());
    } else {
        p.push(0x4a); // BININT
        p.extend_from_slice(&(val as i32).to_le_bytes());
    }
}

fn push_int_tuple(p: &mut Vec<u8>, vals: &[usize]) {
    match vals.len() {
        0 => p.push(0x29), // EMPTY_TUPLE
        1 => {
            push_int(p, vals[0]);
            p.push(0x85); // TUPLE1
        }
        2 => {
            push_int(p, vals[0]);
            push_int(p, vals[1]);
            p.push(0x86); // TUPLE2
        }
        3 => {
            push_int(p, vals[0]);
            push_int(p, vals[1]);
            push_int(p, vals[2]);
            p.push(0x87); // TUPLE3
        }
        _ => {
            p.push(0x28); // MARK
            for &v in vals {
                push_int(p, v);
            }
            p.push(0x74); // TUPLE
        }
    }
}

fn push_global(p: &mut Vec<u8>, module: &str, name: &str) {
    p.push(0x63); // GLOBAL
    p.extend_from_slice(module.as_bytes());
    p.push(b'\n');
    p.extend_from_slice(name.as_bytes());
    p.push(b'\n');
}

fn push_short_binunicode(p: &mut Vec<u8>, s: &str) {
    assert!(s.len() <= 255);
    p.push(0x8c); // SHORT_BINUNICODE
    p.push(s.len() as u8);
    p.extend_from_slice(s.as_bytes());
}

fn build_pickle_state_dict(specs: &[PtTensorSpec]) -> Vec<u8> {
    let mut p = Vec::new();

    // PROTO 2
    p.push(0x80);
    p.push(2);

    // EMPTY_DICT
    p.push(0x7d);

    // MARK for SETITEMS
    p.push(0x28);

    for spec in specs {
        // Key: tensor name
        push_short_binunicode(&mut p, &spec.name);

        // Value: _rebuild_tensor_v2(storage, offset, shape, stride, False, OrderedDict())
        push_global(&mut p, "torch._utils", "_rebuild_tensor_v2");

        // MARK for args tuple
        p.push(0x28);

        // arg0: storage via BINPERSID
        p.push(0x28); // MARK for persistent_id tuple
        push_short_binunicode(&mut p, "storage");
        push_global(&mut p, "torch", &spec.storage_type);
        push_short_binunicode(&mut p, &spec.storage_key);
        push_short_binunicode(&mut p, "cpu");
        push_int(&mut p, spec.numel);
        p.push(0x74); // TUPLE
        p.push(0x51); // BINPERSID

        // arg1: storage_offset
        push_int(&mut p, spec.storage_offset);

        // arg2: shape tuple
        push_int_tuple(&mut p, &spec.shape);

        // arg3: stride tuple
        push_int_tuple(&mut p, &spec.stride);

        // arg4: requires_grad = False
        p.push(0x89); // NEWFALSE

        // arg5: OrderedDict()
        push_global(&mut p, "collections", "OrderedDict");
        p.push(0x29); // EMPTY_TUPLE
        p.push(0x52); // REDUCE

        p.push(0x74); // TUPLE (closes args)
        p.push(0x52); // REDUCE
    }

    // SETITEMS
    p.push(0x75);

    // STOP
    p.push(0x2e);

    p
}

fn build_pytorch_zip(
    specs: &[PtTensorSpec],
    storage_data: &BTreeMap<String, Vec<u8>>,
) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    {
        let mut zip = zip::ZipWriter::new(&mut file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);

        let pickle = build_pickle_state_dict(specs);
        zip.start_file("archive/data.pkl", options).unwrap();
        zip.write_all(&pickle).unwrap();

        for (key, data) in storage_data {
            zip.start_file(format!("archive/data/{}", key), options)
                .unwrap();
            zip.write_all(data).unwrap();
        }

        zip.finish().unwrap();
    }
    file
}

// =========================================================================
// Section 2: ZTensor Format Tests
// =========================================================================

// ----- Shape tests -----

#[test]
fn zt_dense_f32_1d() {
    let data = make_f32_data(1024);
    let mut buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut buf).unwrap();
    w.add("v", &[1024], &data)
        .unwrap();
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
    w.add("w", &[64, 128], &data)
        .unwrap();
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
    w.add("k", &[3, 3, 64], &data)
        .unwrap();
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
    w.add("x", &[8, 3, 32, 32], &data)
        .unwrap();
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
    w.add("s", &[], &data)
        .unwrap();
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
    w.add("big", &[1024, 1024], &data)
        .unwrap();
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
            w.add(
                "tensor",
                &[$n as u64],
                &data,
            )
            .unwrap();
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
    w.add_bytes("tensor", vec![64], DType::Bool, Compression::Raw, &data, Checksum::None)
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
        w.add(
            &format!("layer_{:02}.weight", i),
            &[4, 4],
            &data,
        )
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
    w.add_with("c", &[4096], &data).compress(Compression::Zstd(3)).write()
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
    w.add("t", &[2048], &data)
        .unwrap();
    w.finish().unwrap();

    let mut zstd_buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut zstd_buf).unwrap();
    w.add_with("t", &[2048], &data).compress(Compression::Zstd(3)).write()
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
    w.add_with("c", &[256], &data).checksum(Checksum::Crc32c).write()
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
    w.add_with("s", &[256], &data).checksum(Checksum::Sha256).write()
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
        w.add("t", &[128], &data)
            .unwrap();
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
    w.add("t", &[64], &data)
        .unwrap();
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
        w.add("t", &[32], &data)
            .unwrap();
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

// =========================================================================
// Section 3: SafeTensors Format Tests
// =========================================================================

#[test]
fn st_f32_roundtrip() {
    let data = make_f32_data(24);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "tensor".into(),
        safetensors::Dtype::F32,
        vec![4, 6],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 1);
    let obj = reader.get("tensor").unwrap();
    assert_eq!(obj.shape, vec![4, 6]);
    assert_eq!(reader.read_as::<f32>("tensor").unwrap(), data);
}

#[test]
fn st_f16_roundtrip() {
    let data = make_f16_data(12);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "tensor".into(),
        safetensors::Dtype::F16,
        vec![3, 4],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.read_as::<f16>("tensor").unwrap(), data);
}

#[test]
fn st_bf16_roundtrip() {
    let data = make_bf16_data(12);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "tensor".into(),
        safetensors::Dtype::BF16,
        vec![12],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.read_as::<bf16>("tensor").unwrap(), data);
}

#[test]
fn st_all_dtypes() {
    let f64_data = make_f64_data(4);
    let f32_data = make_f32_data(4);
    let f16_data = make_f16_data(4);
    let bf16_data = make_bf16_data(4);
    let i64_data = make_i64_data(4);
    let i32_data = make_i32_data(4);
    let i16_data = make_i16_data(4);
    let i8_data = make_i8_data(4);
    let u64_data = make_u64_data(4);
    let u32_data = make_u32_data(4);
    let u16_data = make_u16_data(4);
    let u8_data = make_u8_data(4);
    let bool_data = make_bool_data(4);

    let file = build_safetensors_file(vec![
        ("f64".into(), safetensors::Dtype::F64, vec![4], bytemuck::cast_slice(&f64_data).to_vec()),
        ("f32".into(), safetensors::Dtype::F32, vec![4], bytemuck::cast_slice(&f32_data).to_vec()),
        ("f16".into(), safetensors::Dtype::F16, vec![4], bytemuck::cast_slice(&f16_data).to_vec()),
        ("bf16".into(), safetensors::Dtype::BF16, vec![4], bytemuck::cast_slice(&bf16_data).to_vec()),
        ("i64".into(), safetensors::Dtype::I64, vec![4], bytemuck::cast_slice(&i64_data).to_vec()),
        ("i32".into(), safetensors::Dtype::I32, vec![4], bytemuck::cast_slice(&i32_data).to_vec()),
        ("i16".into(), safetensors::Dtype::I16, vec![4], bytemuck::cast_slice(&i16_data).to_vec()),
        ("i8".into(), safetensors::Dtype::I8, vec![4], bytemuck::cast_slice(&i8_data).to_vec()),
        ("u64".into(), safetensors::Dtype::U64, vec![4], bytemuck::cast_slice(&u64_data).to_vec()),
        ("u32".into(), safetensors::Dtype::U32, vec![4], bytemuck::cast_slice(&u32_data).to_vec()),
        ("u16".into(), safetensors::Dtype::U16, vec![4], bytemuck::cast_slice(&u16_data).to_vec()),
        ("u8".into(), safetensors::Dtype::U8, vec![4], u8_data.clone()),
        ("bool".into(), safetensors::Dtype::BOOL, vec![4], bool_data.clone()),
    ]);

    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 13);

    assert_eq!(reader.read_as::<f64>("f64").unwrap(), f64_data);
    assert_eq!(reader.read_as::<f32>("f32").unwrap(), f32_data);
    assert_eq!(reader.read_as::<f16>("f16").unwrap(), f16_data);
    assert_eq!(reader.read_as::<bf16>("bf16").unwrap(), bf16_data);
    assert_eq!(reader.read_as::<i64>("i64").unwrap(), i64_data);
    assert_eq!(reader.read_as::<i32>("i32").unwrap(), i32_data);
    assert_eq!(reader.read_as::<i16>("i16").unwrap(), i16_data);
    assert_eq!(reader.read_as::<i8>("i8").unwrap(), i8_data);
    assert_eq!(reader.read_as::<u64>("u64").unwrap(), u64_data);
    assert_eq!(reader.read_as::<u32>("u32").unwrap(), u32_data);
    assert_eq!(reader.read_as::<u16>("u16").unwrap(), u16_data);
    assert_eq!(reader.read_as::<u8>("u8").unwrap(), u8_data);
    let bools: Vec<bool> = reader.read_as("bool").unwrap();
    let expected_bools: Vec<bool> = bool_data.iter().map(|&b| b != 0).collect();
    assert_eq!(bools, expected_bools);
}

#[test]
fn st_1d_vector() {
    let data = make_f32_data(100);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "v".into(),
        safetensors::Dtype::F32,
        vec![100],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.get("v").unwrap().shape, vec![100]);
    assert_eq!(reader.read_as::<f32>("v").unwrap(), data);
}

#[test]
fn st_2d_matrix() {
    let data = make_f32_data(8 * 16);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "m".into(),
        safetensors::Dtype::F32,
        vec![8, 16],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.get("m").unwrap().shape, vec![8, 16]);
}

#[test]
fn st_high_rank_5d() {
    let data = make_f32_data(2 * 3 * 4 * 5 * 6);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "t5".into(),
        safetensors::Dtype::F32,
        vec![2, 3, 4, 5, 6],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.get("t5").unwrap().shape, vec![2, 3, 4, 5, 6]);
    assert_eq!(reader.read_as::<f32>("t5").unwrap(), data);
}

#[test]
fn st_multi_tensor_10() {
    let mut tensors = Vec::new();
    let mut expected = Vec::new();
    for i in 0..10 {
        let n = (i + 1) * 8;
        let data = make_f32_data(n);
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        tensors.push((
            format!("tensor_{}", i),
            safetensors::Dtype::F32,
            vec![n],
            bytes,
        ));
        expected.push((format!("tensor_{}", i), data));
    }
    let file = build_safetensors_file(tensors);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 10);
    for (name, data) in &expected {
        assert_eq!(reader.read_as::<f32>(name).unwrap(), *data);
    }
}

#[test]
fn st_model_like() {
    let embed = make_f32_data(100 * 64);
    let q_proj = make_f32_data(64 * 64);
    let k_proj = make_f32_data(64 * 64);
    let v_proj = make_f32_data(64 * 64);
    let gate = make_f16_data(256 * 64);
    let up = make_f16_data(256 * 64);
    let down = make_f16_data(64 * 256);
    let ln = make_f32_data(64);
    let lm_head = make_f32_data(100 * 64);

    let file = build_safetensors_file(vec![
        ("embed_tokens.weight".into(), safetensors::Dtype::F32, vec![100, 64], bytemuck::cast_slice(&embed).to_vec()),
        ("layers.0.self_attn.q_proj.weight".into(), safetensors::Dtype::F32, vec![64, 64], bytemuck::cast_slice(&q_proj).to_vec()),
        ("layers.0.self_attn.k_proj.weight".into(), safetensors::Dtype::F32, vec![64, 64], bytemuck::cast_slice(&k_proj).to_vec()),
        ("layers.0.self_attn.v_proj.weight".into(), safetensors::Dtype::F32, vec![64, 64], bytemuck::cast_slice(&v_proj).to_vec()),
        ("layers.0.mlp.gate_proj.weight".into(), safetensors::Dtype::F16, vec![256, 64], bytemuck::cast_slice(&gate).to_vec()),
        ("layers.0.mlp.up_proj.weight".into(), safetensors::Dtype::F16, vec![256, 64], bytemuck::cast_slice(&up).to_vec()),
        ("layers.0.mlp.down_proj.weight".into(), safetensors::Dtype::F16, vec![64, 256], bytemuck::cast_slice(&down).to_vec()),
        ("layers.0.input_layernorm.weight".into(), safetensors::Dtype::F32, vec![64], bytemuck::cast_slice(&ln).to_vec()),
        ("lm_head.weight".into(), safetensors::Dtype::F32, vec![100, 64], bytemuck::cast_slice(&lm_head).to_vec()),
    ]);

    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 9);

    // Verify shapes and dtypes
    let e = reader.get("embed_tokens.weight").unwrap();
    assert_eq!(e.shape, vec![100, 64]);
    assert_eq!(e.components.get("data").unwrap().dtype, DType::F32);

    let g = reader.get("layers.0.mlp.gate_proj.weight").unwrap();
    assert_eq!(g.shape, vec![256, 64]);
    assert_eq!(g.components.get("data").unwrap().dtype, DType::F16);

    // Verify data
    assert_eq!(reader.read_as::<f32>("embed_tokens.weight").unwrap(), embed);
    assert_eq!(reader.read_as::<f16>("layers.0.mlp.gate_proj.weight").unwrap(), gate);
    assert_eq!(reader.read_as::<f32>("lm_head.weight").unwrap(), lm_head);
}

#[test]
fn st_zero_copy_slice() {
    let data = make_f32_data(16);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "t".into(),
        safetensors::Dtype::F32,
        vec![4, 4],
        raw_bytes.clone(),
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    let slice = reader.view("t").unwrap();
    assert_eq!(slice, &raw_bytes[..]);
}

#[test]
fn st_zero_copy_typed() {
    let data = make_f32_data(16);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "t".into(),
        safetensors::Dtype::F32,
        vec![16],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    let typed = reader.view_as::<f32>("t").unwrap();
    assert_eq!(typed, &data[..]);
}

#[test]
fn st_error_not_found() {
    let file = build_safetensors_file(vec![(
        "exists".into(),
        safetensors::Dtype::F32,
        vec![1],
        vec![0; 4],
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    match reader.read("missing") {
        Err(Error::ObjectNotFound(_)) => {}
        other => panic!("Expected ObjectNotFound, got {:?}", other),
    }
}

// =========================================================================
// Section 4: PyTorch Format Tests
// =========================================================================

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
            assert_eq!(
                obj.components.get("data").unwrap().dtype,
                $dtype
            );
            let result: Vec<$t> = reader.read_as("tensor").unwrap();
            assert_eq!(result, data);
        }
    };
}

pt_storage_test!(pt_float_storage, "FloatStorage", DType::F32, f32, make_f32_data, 12);
pt_storage_test!(pt_double_storage, "DoubleStorage", DType::F64, f64, make_f64_data, 8);
pt_storage_test!(pt_half_storage, "HalfStorage", DType::F16, f16, make_f16_data, 16);
pt_storage_test!(pt_long_storage, "LongStorage", DType::I64, i64, make_i64_data, 6);
pt_storage_test!(pt_int_storage, "IntStorage", DType::I32, i32, make_i32_data, 10);
pt_storage_test!(pt_byte_storage, "ByteStorage", DType::U8, u8, make_u8_data, 20);

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
            assert!(
                msg.contains("pickle") || msg.contains("pkl"),
                "msg={}",
                msg
            );
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
        Err(e) => panic!("Expected InvalidFileStructure about no tensors, got {:?}", e),
        Ok(_) => panic!("Expected error"),
    }
}

// =========================================================================
// Section 5: Cross-Format Verification Tests
// =========================================================================

/// Write identical data to all three formats, read back, verify bit-exact match.
fn cross_format_verify_f32(name: &str, shape_zt: Vec<u64>, shape_st: Vec<usize>, data: &[f32]) {
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
    let n: usize = data.len();

    // ZTensor
    let mut zt_buf = Cursor::new(Vec::new());
    let mut w = Writer::new(&mut zt_buf).unwrap();
    w.add(name, &shape_zt, data)
        .unwrap();
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
    let shape_pt: Vec<usize> = zt_data.len().min(n).max(1).min(n)
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
    w.add("t", &[256], &data)
        .unwrap();
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
    w.add("t", &[128], &data)
        .unwrap();
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
    w.add("t", &[8, 16], &data)
        .unwrap();
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
            (w_name.clone(), safetensors::Dtype::F32, vec![32, 32], w_bytes.clone()),
            (b_name.clone(), safetensors::Dtype::F32, vec![32], b_bytes.clone()),
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
        ("f32".into(), safetensors::Dtype::F32, vec![64], f32_bytes.clone()),
        ("i64".into(), safetensors::Dtype::I64, vec![32], i64_bytes.clone()),
        ("u8".into(), safetensors::Dtype::U8, vec![128], u8_data.clone()),
    ]);
    let st_reader = SafeTensorsReader::open(st_file.path()).unwrap();

    // PyTorch
    let specs = vec![
        PtTensorSpec { name: "f32".into(), storage_type: "FloatStorage".into(), storage_key: "0".into(), shape: vec![64], stride: vec![1], storage_offset: 0, numel: 64 },
        PtTensorSpec { name: "i64".into(), storage_type: "LongStorage".into(), storage_key: "1".into(), shape: vec![32], stride: vec![1], storage_offset: 0, numel: 32 },
        PtTensorSpec { name: "u8".into(), storage_type: "ByteStorage".into(), storage_key: "2".into(), shape: vec![128], stride: vec![1], storage_offset: 0, numel: 128 },
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

// =========================================================================
// Section 7: NPZ Format Tests
// =========================================================================

/// Build a .npy file in memory: magic + version + header + raw data.
fn build_npy(descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {} }}",
        descr, shape_str
    );
    // Pad header to 64-byte alignment (v1: 10 bytes preamble)
    let preamble_len = 10;
    let total = preamble_len + header_dict.len() + 1; // +1 for newline
    let pad = ((total + 63) / 64) * 64 - total;
    let padded_header = format!("{}{}\n", header_dict, " ".repeat(pad));

    let mut npy = Vec::new();
    npy.extend_from_slice(b"\x93NUMPY");
    npy.push(1); // major version
    npy.push(0); // minor version
    let header_len = padded_header.len() as u16;
    npy.extend_from_slice(&header_len.to_le_bytes());
    npy.extend_from_slice(padded_header.as_bytes());
    npy.extend_from_slice(data);
    npy
}

/// Build an .npz file (ZIP of .npy entries) with STORED compression.
fn build_npz_file(entries: Vec<(&str, &str, &[usize], &[u8])>) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".npz").unwrap();
    {
        let mut zip = zip::ZipWriter::new(&mut file);
        for (name, descr, shape, data) in &entries {
            let npy_data = build_npy(descr, shape, data);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file(format!("{}.npy", name), options).unwrap();
            std::io::Write::write_all(&mut zip, &npy_data).unwrap();
        }
        zip.finish().unwrap();
    }
    file.flush().unwrap();
    file
}

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
    let result: Vec<f32> = raw.chunks_exact(4)
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

// =========================================================================
// Section 8: ONNX Format Tests
// =========================================================================

// ---- Minimal protobuf encoder helpers ----

fn pb_varint(val: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut v = val;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
    buf
}

fn pb_tag(field: u32, wire_type: u32) -> Vec<u8> {
    pb_varint(((field as u64) << 3) | (wire_type as u64))
}

fn pb_length_delimited(field: u32, data: &[u8]) -> Vec<u8> {
    let mut out = pb_tag(field, 2);
    out.extend(pb_varint(data.len() as u64));
    out.extend_from_slice(data);
    out
}

fn pb_varint_field(field: u32, val: u64) -> Vec<u8> {
    let mut out = pb_tag(field, 0);
    out.extend(pb_varint(val));
    out
}

/// Build a TensorProto with raw_data.
fn build_tensor_proto(name: &str, data_type: u64, dims: &[i64], raw_data: &[u8]) -> Vec<u8> {
    let mut proto = Vec::new();

    // field 1: dims (packed repeated int64)
    let mut packed_dims = Vec::new();
    for &d in dims {
        packed_dims.extend(pb_varint(d as u64));
    }
    proto.extend(pb_length_delimited(1, &packed_dims));

    // field 2: data_type (varint)
    proto.extend(pb_varint_field(2, data_type));

    // field 8: name (string)
    proto.extend(pb_length_delimited(8, name.as_bytes()));

    // field 9: raw_data (bytes)
    proto.extend(pb_length_delimited(9, raw_data));

    proto
}

/// Build a TensorProto with float_data (field 4, packed).
fn build_tensor_proto_float_data(name: &str, dims: &[i64], float_data: &[f32]) -> Vec<u8> {
    let mut proto = Vec::new();

    // field 1: dims
    let mut packed_dims = Vec::new();
    for &d in dims {
        packed_dims.extend(pb_varint(d as u64));
    }
    proto.extend(pb_length_delimited(1, &packed_dims));

    // field 2: data_type = 1 (FLOAT)
    proto.extend(pb_varint_field(2, 1));

    // field 8: name
    proto.extend(pb_length_delimited(8, name.as_bytes()));

    // field 4: float_data (packed repeated float = wire type 2 with packed fixed32)
    let mut packed_floats = Vec::new();
    for &f in float_data {
        packed_floats.extend_from_slice(&f.to_le_bytes());
    }
    proto.extend(pb_length_delimited(4, &packed_floats));

    proto
}

/// Build an ONNX ModelProto: model -> graph -> initializer[].
fn build_onnx_file(tensors: Vec<Vec<u8>>) -> NamedTempFile {
    // Build GraphProto
    let mut graph = Vec::new();
    for tensor in &tensors {
        // field 5: initializer (TensorProto)
        graph.extend(pb_length_delimited(5, tensor));
    }

    // Build ModelProto
    let mut model = Vec::new();
    // field 7: graph (GraphProto)
    model.extend(pb_length_delimited(7, &graph));

    let mut file = NamedTempFile::with_suffix(".onnx").unwrap();
    std::io::Write::write_all(&mut file, &model).unwrap();
    file.flush().unwrap();
    file
}

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
    let result: Vec<f32> = bytes.chunks_exact(4)
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

// =========================================================================
// Section: Robustness / Error-Path Tests
// =========================================================================

// --- Writer overflow tests ---

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

// --- GGUF corrupt file tests ---

#[test]
fn gguf_rejects_truncated_header() {
    let mut tmp = NamedTempFile::new().unwrap();
    // Just the magic, no version/counts
    tmp.write_all(b"GGUF").unwrap();
    tmp.flush().unwrap();
    assert!(GgufReader::open(tmp.path()).is_err());
}

#[test]
fn gguf_rejects_invalid_version() {
    let mut tmp = NamedTempFile::new().unwrap();
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&99u32.to_le_bytes()); // invalid version
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
    tmp.write_all(&data).unwrap();
    tmp.flush().unwrap();
    assert!(GgufReader::open(tmp.path()).is_err());
}

#[test]
fn gguf_rejects_huge_tensor_count() {
    let mut tmp = NamedTempFile::new().unwrap();
    let mut data = Vec::new();
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&u64::MAX.to_le_bytes()); // absurd tensor_count
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
    tmp.write_all(&data).unwrap();
    tmp.flush().unwrap();
    assert!(GgufReader::open(tmp.path()).is_err());
}

// --- ONNX corrupt file tests ---

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

// --- HDF5 corrupt file tests ---

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
    tmp.write_all(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]).unwrap();
    tmp.flush().unwrap();
    assert!(Hdf5Reader::open(tmp.path()).is_err());
}

// --- NPZ corrupt file tests ---

#[test]
fn npz_rejects_invalid_npy_header() {
    // Create a valid ZIP containing an invalid .npy file
    let buf = Vec::new();
    let cursor = Cursor::new(buf);
    let mut zip = zip::ZipWriter::new(cursor);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored);
    zip.start_file("arr_0.npy", options).unwrap();
    zip.write_all(b"NOT_A_NUMPY_ARRAY").unwrap();
    let cursor = zip.finish().unwrap();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(cursor.get_ref()).unwrap();
    tmp.flush().unwrap();
    assert!(NpzReader::open(tmp.path()).is_err());
}

