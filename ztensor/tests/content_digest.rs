//! The content digest (spec §6.4): identity that layout cannot reach.
//!
//! The property worth testing is not "it returns a string" but that genuinely
//! different files agree, and that files with different tensors do not. Every
//! test here writes the same model two ways and demands one answer.

use std::path::PathBuf;

use ztensor::{shard_identity, DType, DigestAlgorithm, Source, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn digest_of(path: &PathBuf) -> String {
    ztensor::manifest_of(path)
        .unwrap()
        .unwrap()
        .content_digest(DigestAlgorithm::Sha256)
        .unwrap()
}

/// Alignment is layout, so it must not reach the digest.
#[test]
fn placement_does_not_change_the_content_digest() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let b = vec![9u8; 300];

    let canonical = tmp("cd-canonical.zt");
    let mut w = Writer::create(&canonical).unwrap();
    w.add("a", [4u64], DType::F32, &a).unwrap();
    w.add("b", [300u64], DType::U8, &b).unwrap();
    w.finish().unwrap();

    let floor = tmp("cd-floor.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&floor)
        .unwrap();
    // Also inserted in the other order, which is another thing layout decides.
    w.add("b", [300u64], DType::U8, &b).unwrap();
    w.add("a", [4u64], DType::F32, &a).unwrap();
    w.finish().unwrap();

    // The files really are different.
    assert_ne!(
        std::fs::read(&canonical).unwrap(),
        std::fs::read(&floor).unwrap()
    );
    assert_eq!(
        digest_of(&canonical),
        digest_of(&floor),
        "64 KiB and 4 KiB placement describe the same model"
    );
}

/// Splitting a model across files must not change what the model is.
///
/// This is the property that makes a canonical multi-file profile unnecessary:
/// the reason to pin a shard-partition policy was to keep identity stable, and
/// identity is stable without one.
#[test]
fn sharding_does_not_change_the_content_digest() {
    let payload = f32s(&[7.0; 64]);

    let single = tmp("cd-single.zt");
    let mut w = Writer::create(&single).unwrap();
    w.add("w", [64u64], DType::F32, &payload).unwrap();
    w.finish().unwrap();
    let expected = digest_of(&single);

    // The same tensor, now living in a shard that a root points at.
    let shard = tmp("cd-shard.zt");
    let mut w = Writer::create(&shard).unwrap();
    w.add("w", [64u64], DType::F32, &payload).unwrap();
    w.finish().unwrap();
    let id = shard_identity(&shard, DigestAlgorithm::Sha256).unwrap();
    let object = ztensor::manifest_of(&shard)
        .unwrap()
        .unwrap()
        .object("w")
        .unwrap()
        .clone();

    let root = tmp("cd-root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root)
        .unwrap();
    w.add_shard("part", &id).unwrap();
    w.link("w", &object, "part").unwrap();
    w.finish().unwrap();

    assert_eq!(
        digest_of(&root),
        expected,
        "one file and a root plus a shard hold the same model"
    );
}

/// Different tensors must give different digests, or the whole thing says
/// nothing.
#[test]
fn different_models_differ() {
    let base = tmp("cd-base.zt");
    let mut w = Writer::create(&base).unwrap();
    w.add("w", [4u64], DType::F32, &f32s(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    w.finish().unwrap();
    let expected = digest_of(&base);

    // One different value.
    let changed = tmp("cd-changed.zt");
    let mut w = Writer::create(&changed).unwrap();
    w.add("w", [4u64], DType::F32, &f32s(&[1.0, 2.0, 3.0, 4.5]))
        .unwrap();
    w.finish().unwrap();
    assert_ne!(digest_of(&changed), expected, "different bytes");

    // Same bytes, different name.
    let renamed = tmp("cd-renamed.zt");
    let mut w = Writer::create(&renamed).unwrap();
    w.add("v", [4u64], DType::F32, &f32s(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    w.finish().unwrap();
    assert_ne!(digest_of(&renamed), expected, "different tensor name");

    // Same bytes, different shape.
    let reshaped = tmp("cd-reshaped.zt");
    let mut w = Writer::create(&reshaped).unwrap();
    w.add("w", [2u64, 2], DType::F32, &f32s(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    w.finish().unwrap();
    assert_ne!(digest_of(&reshaped), expected, "different shape");
}

/// A part with no digest has nothing to stand for its content, so the answer
/// is "undefined", not a number that looks right.
#[test]
fn a_part_without_a_digest_has_no_content_digest() {
    use xxhash_rust::xxh3::xxh3_64;
    use ztensor::cbor::{self, Value};

    let text = |s: &str| Value::Text(s.to_string());
    let manifest = Value::Map(vec![(
        text("objects"),
        Value::Map(vec![(
            text("t"),
            Value::Map(vec![
                (text("shape"), Value::Array(vec![Value::Uint(8)])),
                (text("layout"), text("dense")),
                (
                    text("parts"),
                    Value::Map(vec![(
                        text("data"),
                        Value::Map(vec![
                            (text("dtype"), text("u8")),
                            (
                                text("blob"),
                                Value::Array(vec![Value::Uint(4096), Value::Uint(8)]),
                            ),
                        ]),
                    )]),
                ),
            ]),
        )]),
    )]);
    let encoded = cbor::encode(&manifest).unwrap();
    let mut bytes = vec![0u8; 8192];
    bytes[..8].copy_from_slice(&ztensor::MAGIC);
    bytes.extend_from_slice(&encoded);
    let mut footer = [0u8; 40];
    footer[0..8].copy_from_slice(&8192u64.to_le_bytes());
    footer[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
    footer[16..24].copy_from_slice(&xxh3_64(&encoded).to_le_bytes());
    footer[24..28].copy_from_slice(&2u32.to_le_bytes());
    footer[32..40].copy_from_slice(&ztensor::MAGIC);
    bytes.extend_from_slice(&footer);
    let path = tmp("cd-nodigest.zt");
    std::fs::write(&path, &bytes).unwrap();

    let manifest = ztensor::manifest_of(&path).unwrap().unwrap();
    let err = manifest
        .content_digest(DigestAlgorithm::Sha256)
        .unwrap_err();
    assert!(matches!(err, ztensor::Error::Unsupported(_)), "{err}");
    let _ = Source::open(&path).unwrap();
}
