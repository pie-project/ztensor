//! The spec §8 reading algorithm, spelled out step by step.
//!
//! This example deliberately avoids the `Reader` type: it demonstrates that
//! a functional `.zt` reader needs nothing beyond a CBOR decoder and XXH3.
//!
//! Usage: `cargo run --example minimal_reader -- model.zt`

use std::env;
use std::fs;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::cbor::{self, Value};

fn main() {
    let path = env::args().nth(1).expect("usage: minimal_reader <file.zt>");
    let buf = fs::read(&path).expect("read file");

    // 1. Minimum size.
    assert!(buf.len() >= 48, "not a .zt file: too small");

    // 2. Footer: last 40 bytes.
    let magic = [0x89, b'Z', b'T', b'2', 0x0d, 0x0a, 0x1a, 0x0a];
    let footer = &buf[buf.len() - 40..];
    assert_eq!(&footer[32..40], &magic, "bad footer magic");
    let version = u32::from_le_bytes(footer[24..28].try_into().unwrap());
    assert_eq!(version, 2, "unsupported version");

    // 3. Data shard?
    let m_off = u64::from_le_bytes(footer[0..8].try_into().unwrap()) as usize;
    let m_len = u64::from_le_bytes(footer[8..16].try_into().unwrap()) as usize;
    let m_hash = u64::from_le_bytes(footer[16..24].try_into().unwrap());
    if m_len == 0 {
        println!("{path}: data shard (no manifest)");
        return;
    }

    // 4. Manifest bytes + hash.
    let manifest_bytes = &buf[m_off..m_off + m_len];
    assert_eq!(xxh3_64(manifest_bytes), m_hash, "manifest hash mismatch");

    // 5. Deterministic CBOR decode.
    let root = cbor::decode(manifest_bytes).expect("manifest CBOR");

    // 6. Walk the objects.
    let objects = map_get(&root, "objects").expect("manifest missing 'objects'");
    for (key, obj) in objects.as_map().unwrap() {
        let name = key.as_text().unwrap();
        let shape: Vec<u64> = map_get(obj, "shape")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        let layout = map_get(obj, "layout").and_then(Value::as_text).unwrap_or("?");
        let parts = map_get(obj, "parts").and_then(Value::as_map).unwrap_or(&[]);
        let dtype = parts
            .first()
            .and_then(|(_, p)| map_get(p, "dtype"))
            .and_then(Value::as_text)
            .unwrap_or("?");
        println!("{name}: {layout} {dtype} {shape:?} ({} part(s))", parts.len());
    }
}

fn map_get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.as_map()?
        .iter()
        .find(|(k, _)| k.as_text() == Some(key))
        .map(|(_, val)| val)
}
