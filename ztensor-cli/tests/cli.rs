//! End-to-end smoke tests: the `zt` binary against real files.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn zt(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_zt"))
        .args(args)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

fn st_file(name: &str) -> PathBuf {
    let header = br#"{"a.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b.bias":{"dtype":"U8","shape":[4],"data_offsets":[16,20]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0u8; 16]);
    bytes.extend_from_slice(&[1, 2, 3, 4]);
    let path = tmp(name);
    fs::write(&path, &bytes).unwrap();
    path
}

#[test]
fn convert_ls_verify_diff() {
    let st = st_file("cli.safetensors");
    let zt_out = tmp("cli.zt");

    // convert
    let (ok, out) = zt(&["convert", st.to_str().unwrap(), zt_out.to_str().unwrap()]);
    assert!(ok, "{out}");
    assert!(out.contains("canonical"), "{out}");

    // ls
    let (ok, out) = zt(&["ls", zt_out.to_str().unwrap()]);
    assert!(ok, "{out}");
    assert!(
        out.contains("a.weight") && out.contains("f32 [2,2]"),
        "{out}"
    );

    // verify (digests were added by conversion)
    let (ok, out) = zt(&["verify", zt_out.to_str().unwrap(), "--deep"]);
    assert!(ok, "{out}");
    assert!(out.contains("digest-verified"), "{out}");

    // diff: source vs converted -> identical content
    let (ok, out) = zt(&["diff", st.to_str().unwrap(), zt_out.to_str().unwrap()]);
    assert!(ok, "{out}");
    assert!(out.contains("identical"), "{out}");
}

#[test]
fn diff_detects_changes() {
    let a = st_file("diff-a.safetensors");
    let b_path = tmp("diff-b.safetensors");
    let mut bytes = fs::read(&a).unwrap();
    let n = bytes.len();
    bytes[n - 1] ^= 0xff; // change one data byte of b.bias
    fs::write(&b_path, &bytes).unwrap();

    let (ok, out) = zt(&["diff", a.to_str().unwrap(), b_path.to_str().unwrap()]);
    assert!(!ok, "diff of different files must be nonzero: {out}");
    assert!(out.contains("b.bias"), "{out}");
    assert!(out.contains("1 changed"), "{out}");
}

#[test]
fn verify_catches_corruption() {
    let st = st_file("corrupt.safetensors");
    let zt_out = tmp("corrupt.zt");
    let (ok, _) = zt(&["convert", st.to_str().unwrap(), zt_out.to_str().unwrap()]);
    assert!(ok);

    // Corrupt one data byte (not the manifest): open still succeeds,
    // digest verification must fail.
    let mut bytes = fs::read(&zt_out).unwrap();
    bytes[65536] ^= 0xff;
    fs::write(&zt_out, &bytes).unwrap();

    let (ok, out) = zt(&["verify", zt_out.to_str().unwrap()]);
    assert!(!ok, "{out}");
    assert!(out.contains("digest mismatch"), "{out}");
}

#[test]
fn usage_and_unknown_format() {
    let (ok, out) = zt(&[]);
    assert!(!ok);
    assert!(out.contains("USAGE"), "{out}");

    let junk = tmp("junk.bin");
    fs::write(&junk, b"garbage").unwrap();
    let (ok, out) = zt(&["ls", junk.to_str().unwrap()]);
    assert!(!ok);
    assert!(out.contains("cannot detect"), "{out}");
}
