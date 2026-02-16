use std::collections::BTreeMap;
use std::io::Write;
use tempfile::NamedTempFile;

pub struct PtTensorSpec {
    pub name: String,
    pub storage_type: String,
    pub storage_key: String,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub storage_offset: usize,
    pub numel: usize,
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn push_int(p: &mut Vec<u8>, val: usize) {
    if val <= 255 {
        p.push(0x4b); // BININT1
        p.push(val as u8);
    } else if val <= 65535 {
        p.push(0x4d); // BININT2
        p.extend_from_slice(&(val as u16).to_le_bytes());
    } else {
        p.push(0x4a); // BININT
        p.extend_from_slice(&(val as i32).to_le_bytes());
    }
}

fn push_int_tuple(p: &mut Vec<u8>, vals: &[usize]) {
    match vals.len() {
        0 => p.push(0x29), // EMPTY_TUPLE
        1 => {
            push_int(p, vals[0]);
            p.push(0x85); // TUPLE1
        }
        2 => {
            push_int(p, vals[0]);
            push_int(p, vals[1]);
            p.push(0x86); // TUPLE2
        }
        3 => {
            push_int(p, vals[0]);
            push_int(p, vals[1]);
            push_int(p, vals[2]);
            p.push(0x87); // TUPLE3
        }
        _ => {
            p.push(0x28); // MARK
            for &v in vals {
                push_int(p, v);
            }
            p.push(0x74); // TUPLE
        }
    }
}

fn push_global(p: &mut Vec<u8>, module: &str, name: &str) {
    p.push(0x63); // GLOBAL
    p.extend_from_slice(module.as_bytes());
    p.push(b'\n');
    p.extend_from_slice(name.as_bytes());
    p.push(b'\n');
}

fn push_short_binunicode(p: &mut Vec<u8>, s: &str) {
    assert!(s.len() <= 255);
    p.push(0x8c); // SHORT_BINUNICODE
    p.push(s.len() as u8);
    p.extend_from_slice(s.as_bytes());
}

fn build_pickle_state_dict(specs: &[PtTensorSpec]) -> Vec<u8> {
    let mut p = Vec::new();

    // PROTO 2
    p.push(0x80);
    p.push(2);

    // EMPTY_DICT
    p.push(0x7d);

    // MARK for SETITEMS
    p.push(0x28);

    for spec in specs {
        // Key: tensor name
        push_short_binunicode(&mut p, &spec.name);

        // Value: _rebuild_tensor_v2(storage, offset, shape, stride, False, OrderedDict())
        push_global(&mut p, "torch._utils", "_rebuild_tensor_v2");

        // MARK for args tuple
        p.push(0x28);

        // arg0: storage via BINPERSID
        p.push(0x28); // MARK for persistent_id tuple
        push_short_binunicode(&mut p, "storage");
        push_global(&mut p, "torch", &spec.storage_type);
        push_short_binunicode(&mut p, &spec.storage_key);
        push_short_binunicode(&mut p, "cpu");
        push_int(&mut p, spec.numel);
        p.push(0x74); // TUPLE
        p.push(0x51); // BINPERSID

        // arg1: storage_offset
        push_int(&mut p, spec.storage_offset);

        // arg2: shape tuple
        push_int_tuple(&mut p, &spec.shape);

        // arg3: stride tuple
        push_int_tuple(&mut p, &spec.stride);

        // arg4: requires_grad = False
        p.push(0x89); // NEWFALSE

        // arg5: OrderedDict()
        push_global(&mut p, "collections", "OrderedDict");
        p.push(0x29); // EMPTY_TUPLE
        p.push(0x52); // REDUCE

        p.push(0x74); // TUPLE (closes args)
        p.push(0x52); // REDUCE
    }

    // SETITEMS
    p.push(0x75);

    // STOP
    p.push(0x2e);

    p
}

pub fn build_pytorch_zip(
    specs: &[PtTensorSpec],
    storage_data: &BTreeMap<String, Vec<u8>>,
) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    {
        let mut zip = zip::ZipWriter::new(&mut file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);

        let pickle = build_pickle_state_dict(specs);
        zip.start_file("archive/data.pkl", options).unwrap();
        zip.write_all(&pickle).unwrap();

        for (key, data) in storage_data {
            zip.start_file(format!("archive/data/{}", key), options)
                .unwrap();
            zip.write_all(data).unwrap();
        }

        zip.finish().unwrap();
    }
    file
}
