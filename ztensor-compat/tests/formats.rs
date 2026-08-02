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
// hdf5
// =======================================================================

#[cfg(feature = "hdf5")]
mod hdf5 {
    use super::*;
    use ztensor_compat::Hdf5;

    /// A minimal HDF5 file: superblock v0, one contiguous f32 dataset "w"
    /// of shape [4] in the root group. Offsets laid out by hand.
    fn h5_bytes(vals: &[f32]) -> Vec<u8> {
        assert_eq!(vals.len(), 4);
        let mut b = vec![0u8; 368];
        let undef = [0xffu8; 8];
        // superblock v0 @0
        b[0..8].copy_from_slice(b"\x89HDF\r\n\x1a\n");
        b[13] = 8; // offset size
        b[14] = 8; // length size
        b[16..18].copy_from_slice(&4u16.to_le_bytes()); // leaf k
        b[18..20].copy_from_slice(&16u16.to_le_bytes()); // internal k
        b[32..40].copy_from_slice(&undef); // free space
        b[40..48].copy_from_slice(&368u64.to_le_bytes()); // eof
        b[48..56].copy_from_slice(&undef); // driver info
        // root symbol table entry @56
        b[72..76].copy_from_slice(&1u32.to_le_bytes()); // cache type: group
        b[80..88].copy_from_slice(&96u64.to_le_bytes()); // btree
        b[88..96].copy_from_slice(&144u64.to_le_bytes()); // heap
        // group B-tree @96
        b[96..100].copy_from_slice(b"TREE");
        b[102..104].copy_from_slice(&1u16.to_le_bytes()); // entries
        b[104..112].copy_from_slice(&undef);
        b[112..120].copy_from_slice(&undef);
        b[128..136].copy_from_slice(&192u64.to_le_bytes()); // SNOD addr
        // local heap @144
        b[144..148].copy_from_slice(b"HEAP");
        b[152..160].copy_from_slice(&16u64.to_le_bytes()); // data seg size
        b[168..176].copy_from_slice(&176u64.to_le_bytes()); // data seg addr
        // heap data @176: name "w" at heap offset 8
        b[184] = b'w';
        // SNOD @192
        b[192..196].copy_from_slice(b"SNOD");
        b[196] = 1;
        b[198..200].copy_from_slice(&1u16.to_le_bytes()); // one symbol
        b[200..208].copy_from_slice(&8u64.to_le_bytes()); // link name offset
        b[208..216].copy_from_slice(&248u64.to_le_bytes()); // object header
        // object header v1 @248
        b[248] = 1;
        b[250..252].copy_from_slice(&3u16.to_le_bytes()); // messages
        b[252..256].copy_from_slice(&1u32.to_le_bytes()); // ref count
        b[256..260].copy_from_slice(&88u32.to_le_bytes()); // header size
        // dataspace message @264: v1, 1 dim of 4
        b[264..266].copy_from_slice(&0x0001u16.to_le_bytes());
        b[266..268].copy_from_slice(&16u16.to_le_bytes());
        b[272] = 1; // version
        b[273] = 1; // ndims
        b[280..288].copy_from_slice(&4u64.to_le_bytes());
        // datatype message @288: f32
        b[288..290].copy_from_slice(&0x0003u16.to_le_bytes());
        b[290..292].copy_from_slice(&24u16.to_le_bytes());
        b[296..304].copy_from_slice(&[0x11, 0x20, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00]);
        // layout message @320: v3 contiguous @352, 16 bytes
        b[320..322].copy_from_slice(&0x0008u16.to_le_bytes());
        b[322..324].copy_from_slice(&24u16.to_le_bytes());
        b[328] = 3; // version
        b[329] = 1; // contiguous
        b[330..338].copy_from_slice(&352u64.to_le_bytes());
        b[338..346].copy_from_slice(&16u64.to_le_bytes());
        // data @352
        b[352..368].copy_from_slice(&f32s(vals));
        b
    }

    #[test]
    fn contiguous_dataset() {
        let vals = [1.5f32, 2.5, 3.5, 4.5];
        let path = tmp("basic.h5");
        fs::write(&path, h5_bytes(&vals)).unwrap();
        let h = Hdf5::open(&path).unwrap();
        assert!(h.skipped().is_empty());
        let obj = &h.manifest().objects["w"];
        assert_eq!(obj.shape, vec![4]);
        assert_eq!(obj.parts["data"].dtype, DType::F32);
        assert_eq!(h.read("w", "data").unwrap(), f32s(&vals));
        assert!(h.caps("w", "data").unwrap().zero_copy);
    }

    #[test]
    fn size_lie_rejected() {
        // Claim 24 bytes of data for a 4-element f32 dataset.
        let mut b = h5_bytes(&[0.0; 4]);
        b[338..346].copy_from_slice(&24u64.to_le_bytes());
        let path = tmp("badsize.h5");
        fs::write(&path, &b).unwrap();
        assert!(Hdf5::open(&path).is_err());
    }
}

// =======================================================================
// onnx
// =======================================================================

#[cfg(feature = "onnx")]
mod onnx {
    use super::*;
    use ztensor_compat::Onnx;

    fn len_field(field: u32, body: &[u8]) -> Vec<u8> {
        let mut out = vec![(field << 3 | 2) as u8];
        assert!(body.len() < 128);
        out.push(body.len() as u8);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn raw_data_initializer() {
        let data = f32s(&[1.0, 2.0, 3.0, 4.0]);
        // TensorProto: dims 2,2 / dtype F32 / name "w" / raw_data
        let mut tensor = vec![0x08, 2, 0x08, 2, 0x10, 1];
        tensor.extend(len_field(8, b"w"));
        tensor.extend(len_field(9, &data));
        let graph = len_field(5, &tensor);
        let model = len_field(7, &graph);
        let path = tmp("basic.onnx");
        fs::write(&path, &model).unwrap();

        let o = Onnx::open(&path).unwrap();
        let obj = &o.manifest().objects["w"];
        assert_eq!(obj.shape, vec![2, 2]);
        assert_eq!(obj.parts["data"].dtype, DType::F32);
        assert_eq!(o.read("w", "data").unwrap(), data);
        assert!(o.caps("w", "data").unwrap().zero_copy);
    }

    /// f16 stored in int32_data: one element per int32 (v1 assembled these
    /// as 4-byte ints — silently wrong sizes).
    #[test]
    fn f16_in_int32_data() {
        // two f16 1.0 values (0x3c00) as packed varints
        let mut tensor = vec![0x08, 2, 0x10, 10];
        tensor.extend(len_field(8, b"h"));
        tensor.extend(len_field(5, &[0x80, 0x78, 0x80, 0x78]));
        let graph = len_field(5, &tensor);
        let model = len_field(7, &graph);
        let path = tmp("f16.onnx");
        fs::write(&path, &model).unwrap();

        let o = Onnx::open(&path).unwrap();
        assert_eq!(o.read("h", "data").unwrap(), vec![0x00, 0x3c, 0x00, 0x3c]);
    }

    #[test]
    fn external_data_refused() {
        let mut tensor = vec![0x08, 2, 0x10, 1, 0x70, 1]; // data_location = 1
        tensor.extend(len_field(8, b"x"));
        let graph = len_field(5, &tensor);
        let model = len_field(7, &graph);
        let path = tmp("external.onnx");
        fs::write(&path, &model).unwrap();
        assert!(matches!(Onnx::open(&path), Err(Error::Unsupported(_))));
    }
}

// =======================================================================
// open_any detection
// =======================================================================

#[cfg(all(feature = "safetensors", feature = "gguf"))]
mod detect {
    use super::*;
    use ztensor_compat::open_any;

    #[test]
    fn detects_zt_and_foreign() {
        // .zt
        let zt = tmp("detect.zt");
        let mut w = ztensor::Writer::create(&zt).unwrap();
        w.add_dense("t", &[2], DType::U8, &[1, 2]).unwrap();
        w.finish().unwrap();
        let src = open_any(&zt).unwrap();
        assert_eq!(src.read("t", "data").unwrap(), vec![1, 2]);

        // safetensors
        let st = tmp("detect.safetensors");
        let header = br#"{"t":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[3, 4]);
        fs::write(&st, &bytes).unwrap();
        let src = open_any(&st).unwrap();
        assert_eq!(src.read("t", "data").unwrap(), vec![3, 4]);

        // garbage
        let junk = tmp("detect.junk");
        fs::write(&junk, b"not a tensor file at all").unwrap();
        assert!(matches!(open_any(&junk), Err(Error::Unsupported(_))));
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
