//! The vocabulary: layouts, encodings, and adding your own.
//!
//! The spec calls L2 registry-managed, so the registry is a value. These tests
//! cover the profiles this crate ships and, just as importantly, one it does
//! not: a layout registered by the caller has to be validated exactly like a
//! built-in, and the same file read without it has to stay readable and
//! unchecked.

use std::path::PathBuf;

use ztensor::format as schema;
use ztensor::vocab::Layout;
use ztensor::{DType, Error, Rule, Source, Vocabulary, Writer};

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
fn writer_rejects_invalid_csr_metadata() {
    let path = tmp("csr-bad.zt");
    let mut w = Writer::create(&path).unwrap();
    // A description borrows its bytes, so they have to outlive it: the payloads
    // are bound here rather than built inline in the closure.
    let (values, indices, indptr) = (f32s(&[1.0]), le_u64s(&[0]), le_u64s(&[0, 1]));
    // indptr holds 2 entries; rank-2 [2, 3] requires rows + 1 = 3
    let err = w
        .object("m", |o| {
            o.shape([2u64, 3])
                .layout("zt.sparse_csr/1")
                .part("values", |p| p.dtype(DType::F32).bytes(&values))
                .part("indices", |p| p.dtype(DType::U64).bytes(&indices))
                .part("indptr", |p| p.dtype(DType::U64).bytes(&indptr))
        })
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
}

#[test]
fn an_unregistered_layout_is_written_and_stays_structural() {
    let path = tmp("custom-layout.zt");
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
    assert_eq!(tensor.parts().count(), 2);
    assert_eq!(
        &*tensor.part("weights").unwrap().bytes().unwrap(),
        &[2u8; 32]
    );
}

// =======================================================================
// registering a profile of your own
// =======================================================================

/// A layout that insists on a `"scales"` part alongside the data, the shape a
/// downstream quantization profile has.
struct Grouped;

impl Layout for Grouped {
    fn id(&self) -> &str {
        "pie.custom/1"
    }

    fn validate(&self, name: &str, obj: &schema::Object, _v: &Vocabulary) -> ztensor::Result<()> {
        if !obj.parts.contains_key("scales") {
            return Err(Error::reject(
                Rule::LayoutRule,
                format!("{name:?}: pie.custom/1 requires a 'scales' part"),
            ));
        }
        Ok(())
    }
}

#[test]
fn a_registered_layout_is_checked_like_a_built_in() {
    let vocab = Vocabulary::standard().with_layout(Grouped);

    // Writing: the profile refuses the object before a byte is written.
    let path = tmp("registered-layout.zt");
    let mut w = Writer::options().vocabulary(&vocab).create(&path).unwrap();
    let err = w
        .object("q", |o| {
            o.shape([32u64])
                .layout("pie.custom/1")
                .part("data", |p| p.dtype(DType::U8).bytes(&[1u8; 32]))
        })
        .unwrap_err();
    assert!(format!("{err}").contains("scales"), "{err}");

    // Written properly it round-trips.
    w.object("q", |o| {
        o.shape([32u64])
            .layout("pie.custom/1")
            .part("data", |p| p.dtype(DType::U8).bytes(&[1u8; 32]))
            .part("scales", |p| p.dtype(DType::U8).bytes(&[2u8; 4]))
    })
    .unwrap();
    w.finish().unwrap();

    // Reading with the vocabulary validates; reading without it is structural,
    // so an old reader still gets the bytes it can address.
    Source::options().vocabulary(&vocab).open(&path).unwrap();
    let plain = Source::open(&path).unwrap();
    assert_eq!(plain.tensor("q").unwrap().parts().count(), 2);
}

#[test]
fn a_registered_layout_rejects_a_file_that_violates_it() {
    // Written by someone who did not have the profile...
    let path = tmp("violates-registered-layout.zt");
    let mut w = Writer::create(&path).unwrap();
    w.object("q", |o| {
        o.shape([32u64])
            .layout("pie.custom/1")
            .part("data", |p| p.dtype(DType::U8).bytes(&[1u8; 32]))
    })
    .unwrap();
    w.finish().unwrap();

    // ...is refused by a reader that does.
    let vocab = Vocabulary::standard().with_layout(Grouped);
    let err = Source::options()
        .vocabulary(&vocab)
        .open(&path)
        .unwrap_err();
    assert_eq!(err.rule(), Some(Rule::LayoutRule), "{err}");
    // And is perfectly readable without it.
    assert!(Source::open(&path).is_ok());
}

#[cfg(feature = "zstd")]
mod zstd_seekable {
    use super::*;

    const ENC: &str = "zt.zstd-seekable/1";

    fn writer(path: &PathBuf) -> Writer {
        Writer::options()
            .canonical(false)
            .align(4096)
            .create(path)
            .unwrap()
    }

    #[test]
    fn encoded_dense_roundtrip() {
        let path = tmp("zstd.zt");
        // > 1 MiB so the stream has multiple frames
        let data: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();

        let mut w = writer(&path);
        w.object("t", |o| {
            o.shape([3_000_000u64])
                .part("data", |p| p.dtype(DType::U8).encoding(ENC).bytes(&data))
        })
        .unwrap();
        w.finish().unwrap();

        let src = Source::open(&path).unwrap();
        let tensor = src.tensor("t").unwrap();
        let stored = src.provenance().root().unwrap().part("t", "data").unwrap();
        assert_eq!(stored.encoding.as_deref(), Some(ENC));
        assert!(stored.blob.length < data.len() as u64, "should compress");

        assert_eq!(&*tensor.bytes().unwrap(), &data[..]);
        assert!(tensor.verify().unwrap().checked()); // digest over decoded bytes

        // An encoded part has no address and no borrow: the stored range is
        // not the tensor, and the message says so.
        let caps = tensor.caps().unwrap();
        assert!(!caps.map && !caps.locate);
        assert!(matches!(tensor.map(), Err(Error::Unsupported(_))));
        assert!(matches!(tensor.locate(), Err(Error::Unsupported(_))));
        assert!(!tensor.bytes().unwrap().is_mapped());
    }

    #[test]
    fn encoded_empty_part() {
        let path = tmp("zstd-empty.zt");
        let mut w = writer(&path);
        w.object("e", |o| {
            o.shape([0u64])
                .part("data", |p| p.dtype(DType::U8).encoding(ENC).bytes(&[]))
        })
        .unwrap();
        w.finish().unwrap();
        let src = Source::open(&path).unwrap();
        assert_eq!(&*src.tensor("e").unwrap().bytes().unwrap(), &[] as &[u8]);
    }

    #[test]
    fn canonical_forbids_encoding() {
        let path = tmp("zstd-canonical.zt");
        let mut w = Writer::create(&path).unwrap();
        let err = w
            .object("t", |o| {
                o.shape([4u64]).part("data", |p| {
                    p.dtype(DType::U8).encoding(ENC).bytes(&[1, 2, 3, 4])
                })
            })
            .unwrap_err();
        assert!(format!("{err}").contains("canonical(false)"), "{err}");
    }

    #[test]
    fn an_unregistered_encoding_is_refused_not_guessed() {
        let path = tmp("zstd-unknown-encoding.zt");
        let data = vec![9u8; 4096];
        let mut w = writer(&path);
        w.object("t", |o| {
            o.shape([4096u64])
                .part("data", |p| p.dtype(DType::U8).encoding(ENC).bytes(&data))
        })
        .unwrap();
        w.finish().unwrap();

        // A reader without the profile can open the file and see the tensor,
        // but must not hand back the stored bytes as if they were the tensor.
        let bare = Vocabulary::empty();
        let src = Source::options().vocabulary(&bare).open(&path).unwrap();
        let err = src.tensor("t").unwrap().bytes().unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    }

    #[test]
    fn corrupt_stream_rejected_not_zero_filled() {
        let path = tmp("zstd-corrupt.zt");
        let data = vec![9u8; 100_000];
        let mut w = writer(&path);
        w.object("t", |o| {
            o.shape([100_000u64])
                .part("data", |p| p.dtype(DType::U8).encoding(ENC).bytes(&data))
        })
        .unwrap();
        w.finish().unwrap();

        // Flip a byte inside the compressed frame body.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4096 + 10] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();

        // The manifest is untouched, so the file opens; the bytes are refused.
        let src = Source::open(&path).unwrap();
        let err = src.tensor("t").unwrap().bytes().unwrap_err();
        assert_eq!(err.rule(), Some(Rule::Encoding), "{err:?}");
    }
}
