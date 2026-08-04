//! Assembling `zt.sparse_csr/1`, which is a layout profile and so lives here
//! rather than in the core crate.
//!
//! The core writer still validates the layout's *metadata* rules, because those
//! are vocabulary and the vocabulary is core's. What is here is the data-level
//! half: reading three parts back and checking the invariants that only the
//! bytes can violate.

use std::path::PathBuf;

use ztensor::{DType, Error, Source, Writer};
use ztensor_compat::csr;

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn le_u64s(vals: &[u64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn csr_roundtrip() {
    // [[1.0, 0, 2.0], [0, 3.0, 0]] as CSR
    let path = tmp("csr.zt");
    let values = f32s(&[1.0, 2.0, 3.0]);
    let indices = le_u64s(&[0, 2, 1]);
    let indptr = le_u64s(&[0, 2, 3]);

    let mut w = Writer::create(&path).unwrap();
    w.object("m", |o| {
        o.shape([2u64, 3])
            .layout("zt.sparse_csr/1")
            .part("values", |p| p.dtype(DType::F32).bytes(&values))
            .part("indices", |p| p.dtype(DType::U64).bytes(&indices))
            .part("indptr", |p| p.dtype(DType::U64).bytes(&indptr))
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let tensor = src.tensor("m").unwrap();
    let csr = csr::read(&tensor).unwrap();
    assert_eq!((csr.rows, csr.cols), (2, 3));
    assert_eq!(csr.indices, vec![0, 2, 1]);
    assert_eq!(csr.indptr, vec![0, 2, 3]);
    assert_eq!(csr.values, values);
    assert_eq!(csr.dtype, DType::F32);
    // The parts are ordinary mappable ranges too.
    assert_eq!(tensor.part("values").unwrap().map().unwrap(), &values[..]);
}

/// A tensor that is not CSR at all is refused rather than misread. The core
/// crate has the matching test that such a layout still *reads* structurally.
#[test]
fn a_foreign_layout_is_not_assembled_as_csr() {
    let path = tmp("csr-not.zt");
    let mut w = Writer::create(&path).unwrap();
    w.object("q", |o| {
        o.shape([64u64])
            .layout("pie.custom/1")
            .part("scales", |p| p.dtype(DType::U8).bytes(&[1u8; 64]))
            .part("weights", |p| p.dtype(DType::U8).bytes(&[2u8; 32]))
    })
    .unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let tensor = src.tensor("q").unwrap();
    assert!(matches!(csr::read(&tensor), Err(Error::Unsupported(_))));
}
