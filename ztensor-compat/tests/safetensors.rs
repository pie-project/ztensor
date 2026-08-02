//! safetensors projection: strict open, honest caps, and the conversion
//! path to canonical `.zt`.

use std::fs;
use std::path::PathBuf;

use ztensor_compat::Safetensors;
use ztensor::{DType, Error, Reader, Source, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// Builds a safetensors file. Offsets are assigned in the given tensor
/// order; `meta` becomes `__metadata__`.
fn st_bytes(tensors: &[(&str, &str, &[u64], &[u8])], meta: &[(&str, &str)]) -> Vec<u8> {
    let mut entries = Vec::new();
    if !meta.is_empty() {
        let kv: Vec<String> = meta
            .iter()
            .map(|(k, v)| format!("\"{k}\":\"{v}\""))
            .collect();
        entries.push(format!("\"__metadata__\":{{{}}}", kv.join(",")));
    }
    let mut cursor = 0usize;
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let dims: Vec<String> = shape.iter().map(u64::to_string).collect();
        entries.push(format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            dims.join(","),
            cursor,
            cursor + bytes.len()
        ));
        cursor += bytes.len();
        data.extend_from_slice(bytes);
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

fn st_file(name: &str, tensors: &[(&str, &str, &[u64], &[u8])], meta: &[(&str, &str)]) -> PathBuf {
    let path = tmp(name);
    fs::write(&path, st_bytes(tensors, meta)).unwrap();
    path
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn open_and_read() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let b = vec![7u8; 8]; // 4 bf16 elements
    let path = st_file(
        "basic.safetensors",
        &[
            ("a.weight", "F32", &[2, 2], &a),
            ("b.weight", "BF16", &[4], &b),
        ],
        &[("format", "pt")],
    );

    let st = Safetensors::open(&path).unwrap();
    assert_eq!(st.manifest().objects.len(), 2);
    let obj = &st.manifest().objects["a.weight"];
    assert_eq!(obj.shape, vec![2, 2]);
    assert_eq!(obj.parts["data"].dtype, DType::F32);
    assert!(st.manifest().attributes.is_some());

    assert_eq!(st.read("a.weight", "data").unwrap(), a);
    assert_eq!(Source::view(&st, "b.weight", "data").unwrap(), &b[..]);

    let caps = st.caps("a.weight", "data").unwrap();
    assert!(caps.zero_copy);
    assert!(!caps.verifiable);
    assert!(caps.tier() >= 2);
}

#[test]
fn dtype_projections() {
    let path = st_file(
        "dtypes.safetensors",
        &[
            ("mask", "BOOL", &[4], &[0, 1, 0, 1]),
            ("fp8", "F8_E4M3", &[4], &[1, 2, 3, 4]),
        ],
        &[],
    );
    let st = Safetensors::open(&path).unwrap();
    let mask = &st.manifest().objects["mask"].parts["data"];
    assert_eq!((mask.dtype, mask.ltype.as_deref()), (DType::U8, Some("bool")));
    let fp8 = &st.manifest().objects["fp8"].parts["data"];
    assert_eq!(
        (fp8.dtype, fp8.ltype.as_deref()),
        (DType::U8, Some("f8_e4m3fn"))
    );
}

#[test]
fn unknown_dtype_refused() {
    let path = st_file("f4.safetensors", &[("t", "F4", &[2], &[0x21])], &[]);
    assert!(matches!(
        Safetensors::open(&path),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_bad_geometry() {
    // size mismatch: F32 [2,2] needs 16 bytes
    let path = st_file("short.safetensors", &[("t", "F32", &[2, 2], &[0u8; 12])], &[]);
    assert!(Safetensors::open(&path).is_err());

    // overlap / hole: hand-build offsets that don't tile
    let mut bytes = st_bytes(&[("a", "U8", &[8], &[1u8; 8])], &[]);
    // corrupt data_offsets [0,8] -> [0,4]: shape mismatch aside, the data
    // section now has a trailing hole
    let needle = b"[0,8]";
    let pos = bytes.windows(5).position(|w| w == needle).unwrap();
    bytes[pos..pos + 5].copy_from_slice(b"[0,4]");
    let path = tmp("hole.safetensors");
    fs::write(&path, &bytes).unwrap();
    assert!(Safetensors::open(&path).is_err());

    // truncated header
    let path = st_file("trunc.safetensors", &[("t", "U8", &[4], &[9u8; 4])], &[]);
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..9]).unwrap();
    assert!(Safetensors::open(&path).is_err());
}

/// The conversion path: HF checkpoint in, canonical tier-3 `.zt` out —
/// bit-reproducibly.
#[test]
fn convert_to_canonical_zt() {
    let a = f32s(&[1.0, 2.0, 3.0, 4.0]);
    let b = vec![3u8; 8];
    let st_path = st_file(
        "convert.safetensors",
        &[("b.weight", "BF16", &[4], &b), ("a.weight", "F32", &[2, 2], &a)],
        &[("format", "pt")],
    );
    let st = Safetensors::open(&st_path).unwrap();

    let convert = |out: &PathBuf| {
        let mut w = Writer::create(out).unwrap();
        w.ingest(&st).unwrap();
        w.finish().unwrap();
    };
    let zt1 = tmp("converted1.zt");
    let zt2 = tmp("converted2.zt");
    convert(&zt1);
    convert(&zt2);

    // Bit-reproducible: same source, identical canonical output.
    assert_eq!(fs::read(&zt1).unwrap(), fs::read(&zt2).unwrap());

    let r = Reader::open(&zt1).unwrap();
    assert_eq!(r.read("a.weight", "data").unwrap(), a);
    assert_eq!(r.read("b.weight", "data").unwrap(), b);
    assert!(r.verify("a.weight", "data").unwrap()); // digests added
    assert!(r.manifest().attributes.is_some()); // metadata carried over

    // Upgrade to tier 3 (on ≤64K page hosts).
    let caps = r.caps("a.weight", "data").unwrap();
    assert!(caps.verifiable && caps.zero_copy);
    if ztensor::page_size() <= ztensor::ALIGN_CANONICAL {
        assert_eq!(caps.tier(), 3);
    }
}
