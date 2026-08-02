//! gguf / npz / pt projections: hand-built files, strict opens, honest
//! refusals, and conversion through `Writer::ingest`.

use std::fs;
use std::path::PathBuf;

use ztensor::{DType, Error, Source};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn f32s(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// =======================================================================
// gguf
// =======================================================================

#[cfg(feature = "gguf")]
mod gguf {
    use super::*;
    use ztensor_compat::Gguf;

    fn gstr(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }

    /// One f32 tensor `[2, 4]` and one q8_0 tensor `[2, 64]`, with a
    /// string KV.
    fn gguf_bytes() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes()); // tensors
        b.extend(1u64.to_le_bytes()); // kvs
        // kv: general.name = "test"
        gstr(&mut b, "general.name");
        b.extend(8u32.to_le_bytes());
        gstr(&mut b, "test");
        // tensor 0: "dense" f32, logical [2, 4] -> ne fastest-first [4, 2]
        gstr(&mut b, "dense");
        b.extend(2u32.to_le_bytes());
        b.extend(4u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(0u32.to_le_bytes()); // F32
        b.extend(0u64.to_le_bytes()); // offset in data section
        // tensor 1: "quant" q8_0, logical [2, 64] -> ne [64, 2],
        // 128 elems / 32 per block * 34 = 136 bytes, at offset 32 (aligned)
        gstr(&mut b, "quant");
        b.extend(2u32.to_le_bytes());
        b.extend(64u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(8u32.to_le_bytes()); // Q8_0
        b.extend(64u64.to_le_bytes());
        // data section: aligned to 32
        while b.len() % 32 != 0 {
            b.push(0);
        }
        let data_start = b.len();
        b.extend(f32s(&[0.5; 8])); // dense: 32 bytes at 0
        b.resize(data_start + 64, 0); // pad to quant offset
        b.extend(vec![7u8; 136]); // quant blocks
        b
    }

    #[test]
    fn open_and_read() {
        let path = tmp("basic.gguf");
        fs::write(&path, gguf_bytes()).unwrap();
        let g = Gguf::open(&path).unwrap();

        let dense = &g.manifest().objects["dense"];
        assert_eq!(dense.shape, vec![2, 4]); // reversed from ne
        assert_eq!(dense.parts["data"].dtype, DType::F32);
        assert_eq!(g.read("dense", "data").unwrap(), f32s(&[0.5; 8]));

        let quant = &g.manifest().objects["quant"];
        assert_eq!(quant.layout.as_str(), "gguf.q8_0/1");
        assert_eq!(quant.shape, vec![2, 64]); // logical shape preserved
        assert_eq!(quant.parts["data"].blob.length, 136);
        assert_eq!(g.read("quant", "data").unwrap(), vec![7u8; 136]);

        assert!(g.manifest().attributes.is_some()); // KVs preserved
        assert!(g.caps("dense", "data").unwrap().zero_copy);
    }

    #[test]
    fn unknown_type_id_refused() {
        let mut b = gguf_bytes();
        // type id of tensor 0 lives right after its dims; corrupt via
        // rebuild with type 99 instead.
        let needle = 0u32.to_le_bytes();
        // find "dense" tensor's type field: after name+ndims+2 dims
        let name_pos = b.windows(5).position(|w| w == b"dense").unwrap();
        let type_pos = name_pos + 5 + 4 + 16;
        b[type_pos..type_pos + 4].copy_from_slice(&99u32.to_le_bytes());
        let _ = needle;
        let path = tmp("badtype.gguf");
        fs::write(&path, &b).unwrap();
        assert!(matches!(Gguf::open(&path), Err(Error::Unsupported(_))));
    }

    #[test]
    fn ingest_quant_preserves_layout() {
        let path = tmp("ingest.gguf");
        fs::write(&path, gguf_bytes()).unwrap();
        let g = Gguf::open(&path).unwrap();

        let zt = tmp("from-gguf.zt");
        let mut w = ztensor::Writer::create(&zt).unwrap();
        w.ingest(&g).unwrap();
        w.finish().unwrap();

        let r = ztensor::Reader::open(&zt).unwrap();
        let quant = r.get("quant").unwrap();
        assert_eq!(quant.layout.as_str(), "gguf.q8_0/1");
        assert_eq!(r.read("quant", "data").unwrap(), vec![7u8; 136]);
        assert!(r.verify("quant", "data").unwrap());
    }
}

// =======================================================================
// npz
// =======================================================================

#[cfg(feature = "npz")]
mod npz {
    use super::*;
    use std::io::Write;
    use ztensor_compat::Npz;

    fn npy_bytes(descr: &str, shape: &str, fortran: bool, data: &[u8]) -> Vec<u8> {
        let dict = format!(
            "{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shape}, }}",
            if fortran { "True" } else { "False" }
        );
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((dict.len() as u16).to_le_bytes());
        out.extend(dict.as_bytes());
        out.extend(data);
        out
    }

    fn write_npz(name: &str, entries: &[(&str, Vec<u8>, bool)]) -> PathBuf {
        let path = tmp(name);
        let mut z = zip::ZipWriter::new(fs::File::create(&path).unwrap());
        for (entry_name, bytes, compress) in entries {
            let method = if *compress {
                zip::CompressionMethod::Deflated
            } else {
                zip::CompressionMethod::Stored
            };
            let opts = zip::write::SimpleFileOptions::default().compression_method(method);
            z.start_file(format!("{entry_name}.npy"), opts).unwrap();
            z.write_all(bytes).unwrap();
        }
        z.finish().unwrap();
        path
    }

    #[test]
    fn stored_and_deflated() {
        let a = f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = vec![9u8; 4];
        let path = write_npz(
            "basic.npz",
            &[
                ("a", npy_bytes("<f4", "(2, 3)", false, &a), false),
                ("b", npy_bytes("|u1", "(4,)", false, &b), true),
            ],
        );
        let n = Npz::open(&path).unwrap();

        assert_eq!(n.manifest().objects["a"].shape, vec![2, 3]);
        assert_eq!(n.read("a", "data").unwrap(), a);
        assert!(n.view("a", "data").is_ok()); // stored: zero-copy
        assert!(n.caps("a", "data").unwrap().zero_copy);

        assert_eq!(n.read("b", "data").unwrap(), b); // deflated: lazy read
        assert!(matches!(n.view("b", "data"), Err(Error::Unsupported(_))));
        assert!(!n.caps("b", "data").unwrap().zero_copy);
    }

    #[test]
    fn refusals() {
        // fortran order: reversing the shape would transpose the data
        let path = write_npz(
            "fortran.npz",
            &[("t", npy_bytes("<f4", "(2, 3)", true, &f32s(&[0.0; 6])), false)],
        );
        assert!(matches!(Npz::open(&path), Err(Error::Unsupported(_))));

        // big-endian descr
        let path = write_npz(
            "be.npz",
            &[("t", npy_bytes(">f4", "(2,)", false, &f32s(&[0.0; 2])), false)],
        );
        assert!(matches!(Npz::open(&path), Err(Error::Unsupported(_))));

        // size mismatch
        let path = write_npz(
            "short.npz",
            &[("t", npy_bytes("<f4", "(4,)", false, &f32s(&[0.0; 2])), false)],
        );
        assert!(Npz::open(&path).is_err());
    }

    #[test]
    fn bool_maps_to_logical() {
        let path = write_npz(
            "bool.npz",
            &[("m", npy_bytes("|b1", "(3,)", false, &[0, 1, 1]), false)],
        );
        let n = Npz::open(&path).unwrap();
        let part = &n.manifest().objects["m"].parts["data"];
        assert_eq!((part.dtype, part.ltype.as_deref()), (DType::U8, Some("bool")));
    }
}

// =======================================================================
// pt (pickle)
// =======================================================================

#[cfg(feature = "pickle")]
mod pt {
    use super::*;
    use std::io::Write;
    use ztensor_compat::Pt;

    /// Emits the pickle stream torch writes for `{'w': tensor}` with the
    /// given shape/stride over storage key "0" (FloatStorage).
    fn state_dict_pickle(shape: &[u8], stride: &[u8]) -> Vec<u8> {
        let mut p = vec![0x80, 0x02, 0x7d]; // PROTO 2, EMPTY_DICT
        p.extend([0x8c, 0x01]);
        p.extend(b"w"); // key "w"
        // GLOBAL torch._utils _rebuild_tensor_v2
        p.push(0x63);
        p.extend(b"torch._utils\n_rebuild_tensor_v2\n");
        p.push(0x28); // MARK (args)
        {
            // persistent id tuple
            p.push(0x28); // MARK
            p.extend([0x8c, 0x07]);
            p.extend(b"storage");
            p.push(0x63);
            p.extend(b"torch\nFloatStorage\n");
            p.extend([0x8c, 0x01]);
            p.extend(b"0"); // key
            p.extend([0x8c, 0x03]);
            p.extend(b"cpu");
            p.extend([0x4b, 0x04]); // numel 4
            p.push(0x74); // TUPLE
            p.push(0x51); // BINPERSID
        }
        p.extend([0x4b, 0x00]); // storage_offset 0
        // shape tuple
        for &d in shape {
            p.extend([0x4b, d]);
        }
        p.push(0x86); // TUPLE2
        // stride tuple
        for &s in stride {
            p.extend([0x4b, s]);
        }
        p.push(0x86); // TUPLE2
        p.push(0x89); // requires_grad = False
        p.push(0x7d); // backward hooks placeholder
        p.push(0x74); // TUPLE -> args
        p.push(0x52); // REDUCE
        p.push(0x73); // SETITEM
        p.push(0x2e); // STOP
        p
    }

    fn write_pt(name: &str, pickle: &[u8], storage: &[u8]) -> PathBuf {
        let path = tmp(name);
        let mut z = zip::ZipWriter::new(fs::File::create(&path).unwrap());
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        z.start_file("archive/data.pkl", opts).unwrap();
        z.write_all(pickle).unwrap();
        z.start_file("archive/data/0", opts).unwrap();
        z.write_all(storage).unwrap();
        z.finish().unwrap();
        path
    }

    #[test]
    fn state_dict_roundtrip() {
        let data = f32s(&[1.0, 2.0, 3.0, 4.0]);
        let path = write_pt(
            "basic.pt",
            &state_dict_pickle(&[2, 2], &[2, 1]), // contiguous
            &data,
        );
        let pt = Pt::open(&path).unwrap();
        let obj = &pt.manifest().objects["w"];
        assert_eq!(obj.shape, vec![2, 2]);
        assert_eq!(obj.parts["data"].dtype, DType::F32);
        assert_eq!(pt.read("w", "data").unwrap(), data);
        assert!(pt.view("w", "data").is_ok()); // stored zip entry
        assert!(pt.caps("w", "data").unwrap().zero_copy);
    }

    #[test]
    fn non_contiguous_refused_loudly() {
        let path = write_pt(
            "transposed.pt",
            &state_dict_pickle(&[2, 2], &[1, 2]), // transposed stride
            &f32s(&[0.0; 4]),
        );
        let err = Pt::open(&path).unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(ref m) if m.contains("contiguous")),
            "{err:?}"
        );
    }

    #[test]
    fn ingest_to_canonical() {
        let data = f32s(&[5.0, 6.0, 7.0, 8.0]);
        let path = write_pt("ingest.pt", &state_dict_pickle(&[4, 1], &[1, 1]), &data);
        let pt = Pt::open(&path).unwrap();

        let zt = tmp("from-pt.zt");
        let mut w = ztensor::Writer::create(&zt).unwrap();
        w.ingest(&pt).unwrap();
        w.finish().unwrap();

        let r = ztensor::Reader::open(&zt).unwrap();
        assert_eq!(r.read("w", "data").unwrap(), data);
        assert!(r.verify("w", "data").unwrap());
    }
}
