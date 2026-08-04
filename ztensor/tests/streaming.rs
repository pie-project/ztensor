//! Writing an object a chunk at a time.
//!
//! The property that matters: a streamed write and a slice write produce the
//! same file. A producer that cannot hold a tensor in memory, such as one
//! copying a weight off a device in chunks, should not produce a different
//! artifact.
//!
//! The sink is a token rather than a borrow of the writer, which is what lets
//! a producer driven from outside hold both in one structure.

use std::path::PathBuf;

use ztensor::{DType, Error, Source, Writer};

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
        w.add("t.a", [300_000u64], DType::U8, &a).unwrap();
        w.add("t.b", [64u64], DType::U8, &b).unwrap();
        w.finish().unwrap();
    }

    let streamed = tmp("streamed.zt");
    {
        let mut w = Writer::create(&streamed).unwrap();
        for (name, bytes) in [("t.a", &a), ("t.b", &b)] {
            let mut sink = w
                .stream(name, |o| {
                    o.shape([bytes.len() as u64])
                        .part("data", |p| p.dtype(DType::U8).length(bytes.len() as u64))
                })
                .unwrap();
            // Deliberately uneven chunks: the file must not depend on how the
            // producer happened to slice its copies.
            for chunk in bytes.chunks(7919) {
                sink.write(&mut w, chunk).unwrap();
            }
            sink.close(&mut w).unwrap();
        }
        w.finish().unwrap();
    }

    assert_eq!(
        std::fs::read(&sliced).unwrap(),
        std::fs::read(&streamed).unwrap(),
        "a streamed write produced a different file"
    );

    // And the digests it computed on the fly verify.
    let src = Source::open(&streamed).unwrap();
    assert!(src.tensor("t.a").unwrap().verify().unwrap().is_checked());
    assert!(src.tensor("t.b").unwrap().verify().unwrap().is_checked());
    assert_eq!(
        &*src.tensor("t.a").unwrap().data().unwrap().bytes().unwrap(),
        &a[..]
    );
}

#[test]
fn a_multi_part_object_streams_in_name_order() {
    let path = tmp("multipart.zt");
    let data = payload(3, 512);
    let scales = payload(4, 16);

    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&path)
        .unwrap();
    let mut sink = w
        .stream("w", |o| {
            o.shape([1024u64])
                .layout("zt.mx/1")
                .part("data", |p| {
                    p.dtype(DType::U8).logical("f4_e2m1").length(512)
                })
                .part("scales", |p| {
                    p.dtype(DType::U8).logical("f8_e8m0").length(16)
                })
        })
        .unwrap();

    assert_eq!(sink.current(), Some("data"));
    sink.write(&mut w, &data).unwrap();
    assert_eq!(sink.current(), Some("scales"));
    sink.write(&mut w, &scales).unwrap();
    assert_eq!(sink.current(), None);
    sink.close(&mut w).unwrap();
    w.finish().unwrap();

    let src = Source::open(&path).unwrap();
    let tensor = src.tensor("w").unwrap();
    assert_eq!(&*tensor.part("data").unwrap().bytes().unwrap(), &data[..]);
    assert_eq!(
        &*tensor.part("scales").unwrap().bytes().unwrap(),
        &scales[..]
    );
    assert!(tensor
        .part("scales")
        .unwrap()
        .verify()
        .unwrap()
        .is_checked());
}

#[test]
fn writing_past_a_declared_length_is_an_error() {
    let path = tmp("overrun.zt");
    let mut w = Writer::create(&path).unwrap();
    let mut sink = w
        .stream("t", |o| {
            o.shape([16u64])
                .part("data", |p| p.dtype(DType::U8).length(16))
        })
        .unwrap();
    sink.write(&mut w, &[0u8; 8]).unwrap();
    let err = sink.write(&mut w, &[0u8; 9]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}

#[test]
fn closing_a_short_part_is_an_error() {
    let path = tmp("short.zt");
    let mut w = Writer::create(&path).unwrap();
    let mut sink = w
        .stream("t", |o| {
            o.shape([16u64])
                .part("data", |p| p.dtype(DType::U8).length(16))
        })
        .unwrap();
    sink.write(&mut w, &[0u8; 8]).unwrap();
    let err = sink.close(&mut w).unwrap_err();
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
    let mut sink = w
        .stream("a", |o| {
            o.shape([16u64])
                .part("data", |p| p.dtype(DType::U8).length(16))
        })
        .unwrap();

    let err = w.add("b", [4u64], DType::U8, &[0u8; 4]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");

    sink.write(&mut w, &[0u8; 16]).unwrap();

    // Nor may the file be closed around an object that is still open.
    let err = w.finish().unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}

/// Mixing the two ways of giving a part its bytes is refused rather than half
/// honoured: an object is written from slices or streamed, not both.
#[test]
fn bytes_and_length_do_not_mix() {
    let path = tmp("mixed.zt");
    let mut w = Writer::create(&path).unwrap();
    let err = w
        .stream("t", |o| {
            o.shape([8u64])
                .part("data", |p| p.dtype(DType::U8).length(4))
                .part("scales", |p| p.dtype(DType::U8).bytes(&[1u8; 4]))
        })
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");

    let err = w
        .object("t", |o| {
            o.shape([4u64])
                .part("data", |p| p.dtype(DType::U8).length(4))
        })
        .unwrap_err();
    assert!(format!("{err}").contains("stream"), "{err}");
}

/// A layout's metadata rules are checked when the object is declared, before
/// any bytes move, so a producer streaming gigabytes learns its object is
/// malformed at the start, not at the end.
#[test]
fn layout_rules_are_checked_before_the_first_chunk() {
    let path = tmp("badlayout.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&path)
        .unwrap();
    let err = w
        .stream("m", |o| {
            o.shape([2u64, 3])
                .layout("zt.sparse_csr/1")
                // indptr must hold rows + 1 = 3 entries; this declares 2.
                .part("indptr", |p| p.dtype(DType::U64).length(16))
                .part("indices", |p| p.dtype(DType::U64).length(24))
                .part("values", |p| p.dtype(DType::F32).length(12))
        })
        .expect_err("a malformed CSR object must be refused up front");
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}

/// A sink drives the writer that opened it, and no other.
///
/// The check used to be "is *some* object open on that writer", which any
/// writer mid-stream satisfies. A sink handed the wrong writer appended its
/// bytes to whatever blob that writer had open: two files quietly wrong, and
/// the sink believing it had written a part it never wrote.
#[test]
fn a_sink_refuses_a_writer_that_did_not_open_it() {
    let a = tmp("sink-owner-a.zt");
    let b = tmp("sink-owner-b.zt");
    let mut wa = Writer::options().canonical(false).create(&a).unwrap();
    let mut wb = Writer::options().canonical(false).create(&b).unwrap();

    let mut sa = open_sink(&mut wa, "from_a");
    let mut sb = open_sink(&mut wb, "from_b");

    // Both writers are streaming, which is exactly when the old check passed.
    let err = sa.write(&mut wb, &[0xAA; 8]).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");

    // Each still works with its own, and the crossed write left no trace.
    sa.write(&mut wa, &[0xAA; 8]).unwrap();
    sa.close(&mut wa).unwrap();
    sb.write(&mut wb, &[0xBB; 8]).unwrap();
    sb.close(&mut wb).unwrap();
    wa.finish().unwrap();
    wb.finish().unwrap();

    for (path, name, byte) in [(&a, "from_a", 0xAAu8), (&b, "from_b", 0xBB)] {
        let src = Source::open(path).unwrap();
        assert_eq!(src.names().collect::<Vec<_>>(), vec![name]);
        assert_eq!(
            &*src.tensor(name).unwrap().data().unwrap().bytes().unwrap(),
            &[byte; 8]
        );
    }
}

/// `close` is checked too: committing an object onto the wrong writer would
/// put it in the wrong manifest.
#[test]
fn closing_onto_the_wrong_writer_is_refused() {
    let a = tmp("sink-close-a.zt");
    let b = tmp("sink-close-b.zt");
    let mut wa = Writer::options().canonical(false).create(&a).unwrap();
    let mut wb = Writer::options().canonical(false).create(&b).unwrap();

    let mut sa = open_sink(&mut wa, "from_a");
    let sb = open_sink(&mut wb, "from_b");
    sa.write(&mut wa, &[0xAA; 8]).unwrap();

    assert!(sb.close(&mut wa).is_err(), "b's object onto a's writer");

    // `wa` is untouched: its own sink still finishes, and `from_b` is absent.
    sa.close(&mut wa).unwrap();
    wa.finish().unwrap();
    let src = Source::open(&a).unwrap();
    assert_eq!(src.names().collect::<Vec<_>>(), vec!["from_a"]);

    wb.abandon();
}

/// One eight-byte u8 part, streamed.
fn open_sink(w: &mut Writer, name: &str) -> ztensor::Sink {
    w.stream(name, |o| {
        o.shape([8u64])
            .part("data", |p| p.dtype(DType::U8).length(8))
    })
    .unwrap()
}
