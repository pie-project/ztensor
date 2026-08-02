//! M1 exit criteria: round-trip, canonical determinism, tied-weight dedup,
//! and a first slice of must-reject cases.

use std::fs;
use std::path::PathBuf;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::cbor::Value;
use ztensor::{cbor, DType, Error, Reader, Rule, Writer, ALIGN_CANONICAL, MAGIC};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn roundtrip_dense() {
    let path = tmp("roundtrip.zt");
    let a = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b: Vec<u8> = (0..10).collect(); // 5 bf16 elements
    let c: Vec<u8> = vec![7; 7];

    let mut w = Writer::create(&path).unwrap();
    w.add_dense("a.weight", &[2, 3], DType::F32, &a).unwrap();
    w.add_dense("b.bias", &[5], DType::BF16, &b).unwrap();
    w.add_dense("c.mask", &[7], DType::U8, &c).unwrap();
    w.finish().unwrap();

    let r = Reader::open(&path).unwrap();
    assert!(!r.is_data_shard());
    assert_eq!(r.objects().count(), 3);
    assert_eq!(r.get("a.weight").unwrap().shape, vec![2, 3]);
    assert_eq!(r.view("a.weight", "data").unwrap(), &a[..]);
    assert_eq!(r.view("b.bias", "data").unwrap(), &b[..]);
    assert_eq!(r.read("c.mask", "data").unwrap(), c);
    assert!(r.verify("a.weight", "data").unwrap());

    // Canonical placement: every blob at a 64 KiB boundary.
    for (_, obj) in r.objects() {
        for part in obj.parts.values() {
            assert_eq!(part.blob.offset % ALIGN_CANONICAL, 0);
        }
    }
}

#[test]
fn canonical_is_deterministic() {
    let write = |path: &PathBuf| {
        let mut w = Writer::create(path).unwrap();
        w.add_dense("x", &[4], DType::F32, &f32_bytes(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        w.add_dense("y", &[2], DType::U8, &[9, 9]).unwrap();
        w.finish().unwrap();
    };
    let p1 = tmp("det1.zt");
    let p2 = tmp("det2.zt");
    write(&p1);
    write(&p2);
    assert_eq!(fs::read(&p1).unwrap(), fs::read(&p2).unwrap());
}

#[test]
fn tied_weights_share_one_blob() {
    let path = tmp("tied.zt");
    let data = f32_bytes(&[42.0; 256]);
    let mut w = Writer::create(&path).unwrap();
    w.add_dense("embed", &[16, 16], DType::F32, &data).unwrap();
    w.add_dense("lm_head", &[16, 16], DType::F32, &data).unwrap();
    w.finish().unwrap();

    let r = Reader::open(&path).unwrap();
    let e = r.get("embed").unwrap().parts["data"].blob;
    let l = r.get("lm_head").unwrap().parts["data"].blob;
    assert_eq!(e.offset, l.offset, "identical parts must share one blob");
    assert_eq!(r.view("embed", "data").unwrap(), &data[..]);
}

#[test]
fn zero_length_tensor() {
    let path = tmp("zero.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add_dense("empty", &[0, 8], DType::F32, &[]).unwrap();
    w.finish().unwrap();
    let r = Reader::open(&path).unwrap();
    assert_eq!(r.view("empty", "data").unwrap().len(), 0);
}

#[test]
fn canonical_requires_sorted_insertion() {
    let path = tmp("unsorted.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add_dense("b", &[1], DType::U8, &[0]).unwrap();
    let err = w.add_dense("a", &[1], DType::U8, &[0]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
}

// =======================================================================
// Must-reject cases (M2 grows these into the conformance corpus)
// =======================================================================

fn write_small(path: &PathBuf) {
    let mut w = Writer::create(path).unwrap();
    w.add_dense("t", &[2], DType::U8, &[1, 2]).unwrap();
    w.finish().unwrap();
}

fn expect_reject(path: &PathBuf, rule: Rule) {
    match Reader::open(path) {
        Err(Error::Reject { rule: got, .. }) => assert_eq!(got, rule),
        other => panic!("expected Reject({rule:?}), got {other:?}"),
    }
}

#[test]
fn reject_bad_footer_magic() {
    let path = tmp("badfooter.zt");
    write_small(&path);
    let mut bytes = fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 0xff;
    fs::write(&path, &bytes).unwrap();
    expect_reject(&path, Rule::FooterMagic);
}

#[test]
fn reject_corrupt_manifest() {
    let path = tmp("badmanifest.zt");
    write_small(&path);
    let mut bytes = fs::read(&path).unwrap();
    let n = bytes.len();
    let m_off = u64::from_le_bytes(bytes[n - 40..n - 32].try_into().unwrap()) as usize;
    bytes[m_off] ^= 0xff;
    fs::write(&path, &bytes).unwrap();
    expect_reject(&path, Rule::ManifestHash);
}

#[test]
fn reject_truncated() {
    let path = tmp("truncated.zt");
    write_small(&path);
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..bytes.len() - 10]).unwrap();
    let err = Reader::open(&path).unwrap_err();
    assert!(matches!(err, Error::Reject { .. }));
}

/// Assembles a file by hand: magic, blobs, a caller-supplied manifest value,
/// and a correct footer. Lets tests express structurally hostile manifests
/// that the writer would refuse to produce.
fn assemble(path: &PathBuf, data_len: u64, manifest: &Value) {
    let manifest_bytes = cbor::encode(manifest).unwrap();
    let m_off = (8 + data_len).div_ceil(4096) * 4096;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MAGIC);
    bytes.resize(m_off as usize, 0);
    // fill "data" region with a marker so blobs have content
    for b in bytes.iter_mut().take(m_off as usize).skip(8) {
        *b = 0xab;
    }
    bytes.extend_from_slice(&manifest_bytes);
    let mut footer = [0u8; 40];
    footer[0..8].copy_from_slice(&m_off.to_le_bytes());
    footer[8..16].copy_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
    footer[16..24].copy_from_slice(&xxh3_64(&manifest_bytes).to_le_bytes());
    footer[24..28].copy_from_slice(&2u32.to_le_bytes());
    footer[32..40].copy_from_slice(&MAGIC);
    bytes.extend_from_slice(&footer);
    fs::write(path, &bytes).unwrap();
}

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

fn dense_obj(dtype: &str, shape: &[u64], offset: u64, length: u64) -> Value {
    Value::Map(vec![
        (
            text("shape"),
            Value::Array(shape.iter().map(|&d| Value::Uint(d)).collect()),
        ),
        (text("layout"), text("dense")),
        (
            text("parts"),
            Value::Map(vec![(
                text("data"),
                Value::Map(vec![
                    (text("dtype"), text(dtype)),
                    (
                        text("blob"),
                        Value::Array(vec![
                            Value::Uint(0),
                            Value::Uint(offset),
                            Value::Uint(length),
                        ]),
                    ),
                ]),
            )]),
        ),
    ])
}

fn manifest_of(objs: Vec<(&str, Value)>) -> Value {
    Value::Map(vec![(
        text("objects"),
        Value::Map(objs.into_iter().map(|(n, o)| (text(n), o)).collect()),
    )])
}

#[test]
fn reject_partial_overlap() {
    let path = tmp("overlap.zt");
    // blob A: [4096, 12288), blob B: [8192, 8200) — inside A.
    let m = manifest_of(vec![
        ("a", dense_obj("f32", &[2048], 4096, 8192)),
        ("b", dense_obj("f32", &[2], 8192, 8)),
    ]);
    assemble(&path, 12288 - 8, &m);
    expect_reject(&path, Rule::BlobOverlap);
}

#[test]
fn identical_refs_are_legal() {
    let path = tmp("aliased.zt");
    let m = manifest_of(vec![
        ("a", dense_obj("f32", &[2], 4096, 8)),
        ("b", dense_obj("f32", &[2], 4096, 8)),
    ]);
    assemble(&path, 4096 + 8 - 8, &m);
    let r = Reader::open(&path).unwrap();
    assert_eq!(r.view("a", "data").unwrap(), r.view("b", "data").unwrap());
}

#[test]
fn reject_misaligned_blob() {
    let path = tmp("misaligned.zt");
    let m = manifest_of(vec![("a", dense_obj("f32", &[2], 4100, 8))]);
    assemble(&path, 8192, &m);
    expect_reject(&path, Rule::BlobAlignment);
}

#[test]
fn reject_dense_size_mismatch() {
    let path = tmp("badsize.zt");
    // f32 x [3] = 12 bytes, but blob claims 8.
    let m = manifest_of(vec![("a", dense_obj("f32", &[3], 4096, 8))]);
    assemble(&path, 8192, &m);
    expect_reject(&path, Rule::DenseSize);
}

#[test]
fn reject_unknown_dtype() {
    let path = tmp("baddtype.zt");
    let m = manifest_of(vec![("a", dense_obj("f4", &[2], 4096, 1))]);
    assemble(&path, 8192, &m);
    expect_reject(&path, Rule::Schema);
}
