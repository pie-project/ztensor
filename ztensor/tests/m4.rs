//! M4 exit criteria: layout/encoding profile framework — sparse CSR
//! round-trip with data validation, zstd-seekable encoding round-trip.

use std::path::PathBuf;

use ztensor::{DType, Error, PartDef, Reader, Rule, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn le_u64s(vals: &[u64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn raw(dtype: DType, data: &[u8]) -> PartDef<'_> {
    PartDef {
        dtype,
        ltype: None,
        encoding: None,
        data,
    }
}

#[test]
fn csr_roundtrip() {
    // [[1.0, 0, 2.0], [0, 3.0, 0]] as CSR
    let path = tmp("csr.zt");
    let values = f32s(&[1.0, 2.0, 3.0]);
    let indices = le_u64s(&[0, 2, 1]);
    let indptr = le_u64s(&[0, 2, 3]);

    let mut w = Writer::create(&path).unwrap();
    w.add_object(
        "m",
        &[2, 3],
        "zt.sparse_csr/1",
        &[
            ("values", raw(DType::F32, &values)),
            ("indices", raw(DType::U64, &indices)),
            ("indptr", raw(DType::U64, &indptr)),
        ],
        None,
    )
    .unwrap();
    w.finish().unwrap();

    let r = Reader::open(&path).unwrap();
    let csr = r.read_csr("m").unwrap();
    assert_eq!((csr.rows, csr.cols), (2, 3));
    assert_eq!(csr.indices, vec![0, 2, 1]);
    assert_eq!(csr.indptr, vec![0, 2, 3]);
    assert_eq!(csr.values, values);
    assert_eq!(csr.dtype, DType::F32);
    // parts are also plain tier-2 views
    assert_eq!(r.view("m", "values").unwrap(), &values[..]);
}

#[test]
fn writer_rejects_invalid_csr_metadata() {
    let path = tmp("csr-bad.zt");
    let mut w = Writer::create(&path).unwrap();
    // indptr holds 2 entries; rank-2 [2, 3] requires rows + 1 = 3
    let err = w
        .add_object(
            "m",
            &[2, 3],
            "zt.sparse_csr/1",
            &[
                ("values", raw(DType::F32, &f32s(&[1.0]))),
                ("indices", raw(DType::U64, &le_u64s(&[0]))),
                ("indptr", raw(DType::U64, &le_u64s(&[0, 1]))),
            ],
            None,
        )
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}

#[test]
fn unknown_layout_is_written_and_structural() {
    let path = tmp("custom-layout.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add_object(
        "q",
        &[64],
        "pie.custom/1",
        &[
            ("scales", raw(DType::U8, &[1u8; 64])),
            ("weights", raw(DType::U8, &[2u8; 32])),
        ],
        None,
    )
    .unwrap();
    w.finish().unwrap();

    let r = Reader::open(&path).unwrap();
    assert_eq!(r.get("q").unwrap().parts.len(), 2);
    assert_eq!(r.read("q", "weights").unwrap(), vec![2u8; 32]);
    assert!(matches!(r.read_csr("q"), Err(Error::Unsupported(_))));
}

#[cfg(feature = "zstd")]
mod zstd_seekable {
    use super::*;

    const ENC: &str = "zt.zstd-seekable/1";

    fn encoded(dtype: DType, data: &[u8]) -> PartDef<'_> {
        PartDef {
            dtype,
            ltype: None,
            encoding: Some(ENC),
            data,
        }
    }

    #[test]
    fn encoded_dense_roundtrip() {
        let path = tmp("zstd.zt");
        // > 1 MiB so the stream has multiple frames
        let data: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();

        let mut w = Writer::create_with_alignment(&path, 4096).unwrap();
        w.add_object("t", &[3_000_000], "dense", &[("data", encoded(DType::U8, &data))], None)
            .unwrap();
        w.finish().unwrap();

        let r = Reader::open(&path).unwrap();
        let part = &r.get("t").unwrap().parts["data"];
        assert_eq!(part.encoding.as_deref(), Some(ENC));
        assert!(part.blob.length < data.len() as u64, "should compress");

        assert_eq!(r.read("t", "data").unwrap(), data);
        assert!(r.verify("t", "data").unwrap()); // digest over decoded bytes
        assert!(matches!(r.view("t", "data"), Err(Error::Unsupported(_))));
        let caps = r.caps("t", "data").unwrap();
        assert!(!caps.zero_copy);
        assert_eq!(caps.tier(), 1);
    }

    #[test]
    fn encoded_empty_part() {
        let path = tmp("zstd-empty.zt");
        let mut w = Writer::create_with_alignment(&path, 4096).unwrap();
        w.add_object("e", &[0], "dense", &[("data", encoded(DType::U8, &[]))], None)
            .unwrap();
        w.finish().unwrap();
        let r = Reader::open(&path).unwrap();
        assert_eq!(r.read("e", "data").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn canonical_forbids_encoding() {
        let path = tmp("zstd-canonical.zt");
        let mut w = Writer::create(&path).unwrap();
        let err = w
            .add_object("t", &[4], "dense", &[("data", encoded(DType::U8, &[1, 2, 3, 4]))], None)
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn corrupt_stream_rejected_not_zero_filled() {
        let path = tmp("zstd-corrupt.zt");
        let data = vec![9u8; 100_000];
        let mut w = Writer::create_with_alignment(&path, 4096).unwrap();
        w.add_object("t", &[100_000], "dense", &[("data", encoded(DType::U8, &data))], None)
            .unwrap();
        w.finish().unwrap();

        // Flip a byte inside the compressed frame body.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4096 + 10] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();

        let r = Reader::open(&path).unwrap(); // manifest untouched: opens fine
        let err = r.read("t", "data").unwrap_err();
        assert!(
            matches!(err, Error::Reject { rule: Rule::Encoding, .. }),
            "{err:?}"
        );
    }
}
