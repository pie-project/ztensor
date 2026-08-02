//! Writing an object a chunk at a time.
//!
//! The property that matters: a streamed write and a slice write produce the
//! same file. A producer that cannot hold a tensor in memory — one copying a
//! weight off a device in chunks — should not thereby produce a different
//! artifact.

use std::path::PathBuf;

use ztensor::{DType, Error, Reader, StreamPart, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn payload(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

#[test]
fn a_streamed_object_matches_a_slice_written_one() {
    let a = payload(1, 300_000);
    let b = payload(2, 64);

    let sliced = tmp("sliced.zt");
    {
        let mut w = Writer::create(&sliced).unwrap();
        w.add_dense("t.a", &[300_000], DType::U8, &a).unwrap();
        w.add_dense("t.b", &[64], DType::U8, &b).unwrap();
        w.finish().unwrap();
    }

    let streamed = tmp("streamed.zt");
    {
        let mut w = Writer::create(&streamed).unwrap();
        for (name, bytes) in [("t.a", &a), ("t.b", &b)] {
            let mut object = w
                .stream_object(
                    name,
                    &[bytes.len() as u64],
                    "dense",
                    &[StreamPart {
                        name: "data",
                        dtype: DType::U8,
                        ltype: None,
                        length: bytes.len() as u64,
                    }],
                    None,
                )
                .unwrap();
            // Deliberately uneven chunks: the file must not depend on how the
            // producer happened to slice its copies.
            for chunk in bytes.chunks(7919) {
                w.write_chunk(&mut object, chunk).unwrap();
            }
            w.end_object(object).unwrap();
        }
        w.finish().unwrap();
    }

    assert_eq!(
        std::fs::read(&sliced).unwrap(),
        std::fs::read(&streamed).unwrap(),
        "a streamed write produced a different file"
    );

    // And the digests it computed on the fly verify.
    let r = Reader::open(&streamed).unwrap();
    assert!(r.verify("t.a", "data").unwrap());
    assert!(r.verify("t.b", "data").unwrap());
    assert_eq!(r.read("t.a", "data").unwrap(), a);
}

#[test]
fn a_multi_part_object_streams_in_name_order() {
    let path = tmp("multipart.zt");
    let data = payload(3, 512);
    let scales = payload(4, 16);

    let mut w = Writer::create_with_alignment(&path, 4096).unwrap();
    let mut object = w
        .stream_object(
            "w",
            &[1024],
            "zt.mx/1",
            &[
                StreamPart {
                    name: "data",
                    dtype: DType::U8,
                    ltype: Some("f4_e2m1"),
                    length: 512,
                },
                StreamPart {
                    name: "scales",
                    dtype: DType::U8,
                    ltype: Some("f8_e8m0"),
                    length: 16,
                },
            ],
            None,
        )
        .unwrap();

    assert_eq!(object.current(), Some("data"));
    w.write_chunk(&mut object, &data).unwrap();
    assert_eq!(object.current(), Some("scales"));
    w.write_chunk(&mut object, &scales).unwrap();
    assert_eq!(object.current(), None);
    w.end_object(object).unwrap();
    w.finish().unwrap();

    let r = Reader::open(&path).unwrap();
    assert_eq!(r.read("w", "data").unwrap(), data);
    assert_eq!(r.read("w", "scales").unwrap(), scales);
    assert!(r.verify("w", "scales").unwrap());
}

#[test]
fn writing_past_a_declared_length_is_an_error() {
    let path = tmp("overrun.zt");
    let mut w = Writer::create(&path).unwrap();
    let mut object = w
        .stream_object(
            "t",
            &[16],
            "dense",
            &[StreamPart {
                name: "data",
                dtype: DType::U8,
                ltype: None,
                length: 16,
            }],
            None,
        )
        .unwrap();
    w.write_chunk(&mut object, &[0u8; 8]).unwrap();
    let err = w.write_chunk(&mut object, &[0u8; 9]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}

#[test]
fn finishing_a_short_part_is_an_error() {
    let path = tmp("short.zt");
    let mut w = Writer::create(&path).unwrap();
    let mut object = w
        .stream_object(
            "t",
            &[16],
            "dense",
            &[StreamPart {
                name: "data",
                dtype: DType::U8,
                ltype: None,
                length: 16,
            }],
            None,
        )
        .unwrap();
    w.write_chunk(&mut object, &[0u8; 8]).unwrap();
    let err = w.end_object(object).unwrap_err();
    assert!(
        format!("{err}").contains("8 of 16 bytes"),
        "expected a short-part error, got {err}"
    );
}

/// The writer has one blob cursor, so bytes from a second object written while
/// a stream is open would land inside the part being streamed.
#[test]
fn nothing_else_may_be_written_while_a_stream_is_open() {
    let path = tmp("interleaved.zt");
    let mut w = Writer::create(&path).unwrap();
    let mut object = w
        .stream_object(
            "a",
            &[16],
            "dense",
            &[StreamPart {
                name: "data",
                dtype: DType::U8,
                ltype: None,
                length: 16,
            }],
            None,
        )
        .unwrap();

    let err = w.add_dense("b", &[4], DType::U8, &[0u8; 4]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");

    w.write_chunk(&mut object, &[0u8; 16]).unwrap();

    // Nor may the file be closed around an object that is still open.
    let err = w.finish().unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}

/// A layout's metadata rules are checked when the object is declared, before
/// any bytes move — a producer streaming gigabytes should learn its object is
/// malformed at the start, not at the end.
#[test]
fn layout_rules_are_checked_before_the_first_chunk() {
    let path = tmp("badlayout.zt");
    let mut w = Writer::create_with_alignment(&path, 4096).unwrap();
    let err = w
        .stream_object(
            "m",
            &[2, 3],
            "zt.sparse_csr/1",
            &[
                // indptr must hold rows + 1 = 3 entries; this declares 2.
                StreamPart {
                    name: "indptr",
                    dtype: DType::U64,
                    ltype: None,
                    length: 16,
                },
                StreamPart {
                    name: "indices",
                    dtype: DType::U64,
                    ltype: None,
                    length: 24,
                },
                StreamPart {
                    name: "values",
                    dtype: DType::F32,
                    ltype: None,
                    length: 12,
                },
            ],
            None,
        )
        .err()
        .expect("a malformed CSR object must be refused up front");
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}
