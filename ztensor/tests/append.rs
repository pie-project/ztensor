//! Adding to a finished `.zt` without rewriting it.
//!
//! The interesting cases are not "does it round-trip" but the ones where the
//! file on disk could end up subtly wrong: a new manifest smaller than the one
//! it replaced, an append onto an append, and a name that is already taken.

use std::fs;
use std::path::PathBuf;

use ztensor::{shard_identity, DType, Error, Source, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The blobs already in the file are neither moved nor rewritten.
#[test]
fn appending_leaves_the_existing_bytes_where_they_were() {
    let path = tmp("append-basic.zt");
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    w.add("a", [4u64], DType::F32, &a).unwrap();
    w.finish().unwrap();

    let before = Source::open(&path).unwrap();
    let a_at = before.tensor("a").unwrap().locate().unwrap();
    let (a_off, a_len) = (a_at.offset, a_at.len);
    drop(before);
    let head = fs::read(&path).unwrap()[..(a_off + a_len) as usize].to_vec();

    let b = f32s(&[9.0; 8]);
    let mut w = Writer::append(&path).unwrap();
    w.add("b", [8u64], DType::F32, &b).unwrap();
    w.finish().unwrap();

    // Every byte up to the end of the original data region is untouched.
    assert_eq!(&fs::read(&path).unwrap()[..head.len()], &head[..]);

    let src = Source::open(&path).unwrap();
    assert_eq!(src.names().collect::<Vec<_>>(), vec!["a", "b"]);
    assert_eq!(src.tensor("a").unwrap().map().unwrap(), &a[..]);
    assert_eq!(src.tensor("b").unwrap().map().unwrap(), &b[..]);
    assert_eq!(src.tensor("a").unwrap().locate().unwrap().offset, a_off);
}

/// The footer ends the file, whatever the manifest did.
///
/// An append rewrites the manifest in place over the old one. That is only
/// safe while the new manifest is at least as long as the old, which holds
/// today because adding to a CBOR map only grows it. This pins the property
/// the reader actually depends on, so a change that broke the assumption
/// would fail here rather than in the field.
#[test]
fn the_footer_still_ends_the_file() {
    let path = tmp("append-shorter.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    // Many long names make the first manifest large.
    for i in 0..64 {
        let name = format!("a.very.long.tensor.name.that.pads.the.manifest.{i:04}");
        w.add(&name, [2u64], DType::F32, &f32s(&[1.0, 2.0]))
            .unwrap();
    }
    w.finish().unwrap();
    let big_manifest = fs::metadata(&path).unwrap().len();

    let mut w = Writer::append(&path).unwrap();
    w.add("z", [1u64], DType::F32, &f32s(&[3.0])).unwrap();
    w.finish().unwrap();

    let bytes = fs::read(&path).unwrap();
    assert_eq!(
        &bytes[bytes.len() - 8..],
        &ztensor::MAGIC,
        "the footer must end the file"
    );
    let src = Source::open(&path).unwrap();
    assert_eq!(src.len(), 65);
    assert!(bytes.len() as u64 > big_manifest - 4096);
}

/// Appending twice: the second append reads what the first wrote.
#[test]
fn appends_compose() {
    let path = tmp("append-twice.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    w.add("one", [1u64], DType::F32, &f32s(&[1.0])).unwrap();
    w.finish().unwrap();

    for name in ["two", "three"] {
        let mut w = Writer::append(&path).unwrap();
        w.add(name, [1u64], DType::F32, &f32s(&[2.0])).unwrap();
        w.finish().unwrap();
    }

    let src = Source::open(&path).unwrap();
    assert_eq!(src.names().collect::<Vec<_>>(), vec!["one", "three", "two"]);
    for name in ["one", "two", "three"] {
        src.tensor(name).unwrap().verify().unwrap();
    }
}

/// A shard table survives the trip, and a shard can be added by appending.
#[test]
fn a_shard_can_be_added_to_a_finished_file() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let shard = dir.join("append-shard-data.zt");
    let mut w = Writer::create(&shard).unwrap();
    w.add("borrowed", [4u64], DType::F32, &f32s(&[7.0; 4]))
        .unwrap();
    w.finish().unwrap();
    let id = shard_identity(&shard).unwrap();
    let object = ztensor::validate::manifest_of(&shard)
        .unwrap()
        .unwrap()
        .object("borrowed")
        .unwrap()
        .clone();

    let root = dir.join("append-root.zt");
    let mut w = Writer::options().canonical(false).create(&root).unwrap();
    w.add("local", [1u64], DType::F32, &f32s(&[1.0])).unwrap();
    w.finish().unwrap();

    let mut w = Writer::append(&root).unwrap();
    w.add_shard("data", &id).unwrap();
    w.link("borrowed", &object, "data").unwrap();
    w.finish().unwrap();

    let s = Source::options()
        .resolver({
            let shard = shard.clone();
            move |_: &str, _: &ztensor::Shard| Ok(shard.clone())
        })
        .open(&root)
        .unwrap();
    assert_eq!(s.names().collect::<Vec<_>>(), vec!["borrowed", "local"]);
    assert_eq!(
        s.tensor("borrowed").unwrap().map().unwrap(),
        &f32s(&[7.0; 4])[..]
    );
    s.verify_shards().unwrap();
}

#[test]
fn a_name_already_in_the_file_is_refused() {
    let path = tmp("append-dup.zt");
    let mut w = Writer::options().canonical(false).create(&path).unwrap();
    w.add("t", [1u64], DType::F32, &f32s(&[1.0])).unwrap();
    w.finish().unwrap();

    let mut w = Writer::append(&path).unwrap();
    let err = w.add("t", [1u64], DType::F32, &f32s(&[2.0])).unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err}");
    w.abandon();

    // The failed attempt did not damage the file.
    assert_eq!(Source::open(&path).unwrap().len(), 1);
}

#[test]
fn canonical_form_cannot_be_appended_to() {
    let path = tmp("append-canonical.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("t", [1u64], DType::F32, &f32s(&[1.0])).unwrap();
    w.finish().unwrap();

    let err = Writer::options().append(&path).unwrap_err();
    let message = format!("{err}");
    assert!(message.contains("canonical(false)"), "{message}");
}

#[test]
fn a_data_shard_has_nothing_to_append_to() {
    let path = tmp("append-datashard.zt");
    let mut ds = ztensor::DataShardWriter::create(&path).unwrap();
    ds.add_blob(&[1u8; 64]).unwrap();
    ds.finish().unwrap();

    let err = Writer::append(&path).unwrap_err();
    assert!(format!("{err}").contains("no manifest"), "{err}");
}
