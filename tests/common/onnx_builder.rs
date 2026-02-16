use std::io::Write;
use tempfile::NamedTempFile;

pub fn pb_varint(val: u64) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut v = val;
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
    buf
}

pub fn pb_tag(field: u32, wire_type: u32) -> Vec<u8> {
    pb_varint(((field as u64) << 3) | (wire_type as u64))
}

pub fn pb_length_delimited(field: u32, data: &[u8]) -> Vec<u8> {
    let mut out = pb_tag(field, 2);
    out.extend(pb_varint(data.len() as u64));
    out.extend_from_slice(data);
    out
}

pub fn pb_varint_field(field: u32, val: u64) -> Vec<u8> {
    let mut out = pb_tag(field, 0);
    out.extend(pb_varint(val));
    out
}

/// Build a TensorProto with raw_data.
pub fn build_tensor_proto(name: &str, data_type: u64, dims: &[i64], raw_data: &[u8]) -> Vec<u8> {
    let mut proto = Vec::new();

    // field 1: dims (packed repeated int64)
    let mut packed_dims = Vec::new();
    for &d in dims {
        packed_dims.extend(pb_varint(d as u64));
    }
    proto.extend(pb_length_delimited(1, &packed_dims));

    // field 2: data_type (varint)
    proto.extend(pb_varint_field(2, data_type));

    // field 8: name (string)
    proto.extend(pb_length_delimited(8, name.as_bytes()));

    // field 9: raw_data (bytes)
    proto.extend(pb_length_delimited(9, raw_data));

    proto
}

/// Build a TensorProto with float_data (field 4, packed).
pub fn build_tensor_proto_float_data(name: &str, dims: &[i64], float_data: &[f32]) -> Vec<u8> {
    let mut proto = Vec::new();

    // field 1: dims
    let mut packed_dims = Vec::new();
    for &d in dims {
        packed_dims.extend(pb_varint(d as u64));
    }
    proto.extend(pb_length_delimited(1, &packed_dims));

    // field 2: data_type = 1 (FLOAT)
    proto.extend(pb_varint_field(2, 1));

    // field 8: name
    proto.extend(pb_length_delimited(8, name.as_bytes()));

    // field 4: float_data (packed repeated float = wire type 2 with packed fixed32)
    let mut packed_floats = Vec::new();
    for &f in float_data {
        packed_floats.extend_from_slice(&f.to_le_bytes());
    }
    proto.extend(pb_length_delimited(4, &packed_floats));

    proto
}

/// Build an ONNX ModelProto: model -> graph -> initializer[].
pub fn build_onnx_file(tensors: Vec<Vec<u8>>) -> NamedTempFile {
    // Build GraphProto
    let mut graph = Vec::new();
    for tensor in &tensors {
        // field 5: initializer (TensorProto)
        graph.extend(pb_length_delimited(5, tensor));
    }

    // Build ModelProto
    let mut model = Vec::new();
    // field 7: graph (GraphProto)
    model.extend(pb_length_delimited(7, &graph));

    let mut file = NamedTempFile::with_suffix(".onnx").unwrap();
    std::io::Write::write_all(&mut file, &model).unwrap();
    file.flush().unwrap();
    file
}
