#![cfg(feature = "gguf")]

use std::io::Write;

use tempfile::NamedTempFile;

use ztensor::GgufReader;

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
