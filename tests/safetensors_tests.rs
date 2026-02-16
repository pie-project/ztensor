#![cfg(feature = "safetensors")]

use half::{bf16, f16};

use ztensor::{DType, Error, SafeTensorsReader, TensorReader};

mod common;
use common::data_generators::*;
use common::safetensors_builder::*;

#[test]
fn st_f32_roundtrip() {
    let data = make_f32_data(24);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "tensor".into(),
        safetensors::Dtype::F32,
        vec![4, 6],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 1);
    let obj = reader.get("tensor").unwrap();
    assert_eq!(obj.shape, vec![4, 6]);
    assert_eq!(reader.read_as::<f32>("tensor").unwrap(), data);
}

#[test]
fn st_f16_roundtrip() {
    let data = make_f16_data(12);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "tensor".into(),
        safetensors::Dtype::F16,
        vec![3, 4],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.read_as::<f16>("tensor").unwrap(), data);
}

#[test]
fn st_bf16_roundtrip() {
    let data = make_bf16_data(12);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "tensor".into(),
        safetensors::Dtype::BF16,
        vec![12],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.read_as::<bf16>("tensor").unwrap(), data);
}

#[test]
fn st_all_dtypes() {
    let f64_data = make_f64_data(4);
    let f32_data = make_f32_data(4);
    let f16_data = make_f16_data(4);
    let bf16_data = make_bf16_data(4);
    let i64_data = make_i64_data(4);
    let i32_data = make_i32_data(4);
    let i16_data = make_i16_data(4);
    let i8_data = make_i8_data(4);
    let u64_data = make_u64_data(4);
    let u32_data = make_u32_data(4);
    let u16_data = make_u16_data(4);
    let u8_data = make_u8_data(4);
    let bool_data = make_bool_data(4);

    let file = build_safetensors_file(vec![
        (
            "f64".into(),
            safetensors::Dtype::F64,
            vec![4],
            bytemuck::cast_slice(&f64_data).to_vec(),
        ),
        (
            "f32".into(),
            safetensors::Dtype::F32,
            vec![4],
            bytemuck::cast_slice(&f32_data).to_vec(),
        ),
        (
            "f16".into(),
            safetensors::Dtype::F16,
            vec![4],
            bytemuck::cast_slice(&f16_data).to_vec(),
        ),
        (
            "bf16".into(),
            safetensors::Dtype::BF16,
            vec![4],
            bytemuck::cast_slice(&bf16_data).to_vec(),
        ),
        (
            "i64".into(),
            safetensors::Dtype::I64,
            vec![4],
            bytemuck::cast_slice(&i64_data).to_vec(),
        ),
        (
            "i32".into(),
            safetensors::Dtype::I32,
            vec![4],
            bytemuck::cast_slice(&i32_data).to_vec(),
        ),
        (
            "i16".into(),
            safetensors::Dtype::I16,
            vec![4],
            bytemuck::cast_slice(&i16_data).to_vec(),
        ),
        (
            "i8".into(),
            safetensors::Dtype::I8,
            vec![4],
            bytemuck::cast_slice(&i8_data).to_vec(),
        ),
        (
            "u64".into(),
            safetensors::Dtype::U64,
            vec![4],
            bytemuck::cast_slice(&u64_data).to_vec(),
        ),
        (
            "u32".into(),
            safetensors::Dtype::U32,
            vec![4],
            bytemuck::cast_slice(&u32_data).to_vec(),
        ),
        (
            "u16".into(),
            safetensors::Dtype::U16,
            vec![4],
            bytemuck::cast_slice(&u16_data).to_vec(),
        ),
        (
            "u8".into(),
            safetensors::Dtype::U8,
            vec![4],
            u8_data.clone(),
        ),
        (
            "bool".into(),
            safetensors::Dtype::BOOL,
            vec![4],
            bool_data.clone(),
        ),
    ]);

    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 13);

    assert_eq!(reader.read_as::<f64>("f64").unwrap(), f64_data);
    assert_eq!(reader.read_as::<f32>("f32").unwrap(), f32_data);
    assert_eq!(reader.read_as::<f16>("f16").unwrap(), f16_data);
    assert_eq!(reader.read_as::<bf16>("bf16").unwrap(), bf16_data);
    assert_eq!(reader.read_as::<i64>("i64").unwrap(), i64_data);
    assert_eq!(reader.read_as::<i32>("i32").unwrap(), i32_data);
    assert_eq!(reader.read_as::<i16>("i16").unwrap(), i16_data);
    assert_eq!(reader.read_as::<i8>("i8").unwrap(), i8_data);
    assert_eq!(reader.read_as::<u64>("u64").unwrap(), u64_data);
    assert_eq!(reader.read_as::<u32>("u32").unwrap(), u32_data);
    assert_eq!(reader.read_as::<u16>("u16").unwrap(), u16_data);
    assert_eq!(reader.read_as::<u8>("u8").unwrap(), u8_data);
    let bools: Vec<bool> = reader.read_as("bool").unwrap();
    let expected_bools: Vec<bool> = bool_data.iter().map(|&b| b != 0).collect();
    assert_eq!(bools, expected_bools);
}

#[test]
fn st_1d_vector() {
    let data = make_f32_data(100);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "v".into(),
        safetensors::Dtype::F32,
        vec![100],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.get("v").unwrap().shape, vec![100]);
    assert_eq!(reader.read_as::<f32>("v").unwrap(), data);
}

#[test]
fn st_2d_matrix() {
    let data = make_f32_data(8 * 16);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "m".into(),
        safetensors::Dtype::F32,
        vec![8, 16],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.get("m").unwrap().shape, vec![8, 16]);
}

#[test]
fn st_high_rank_5d() {
    let data = make_f32_data(2 * 3 * 4 * 5 * 6);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "t5".into(),
        safetensors::Dtype::F32,
        vec![2, 3, 4, 5, 6],
        bytes,
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.get("t5").unwrap().shape, vec![2, 3, 4, 5, 6]);
    assert_eq!(reader.read_as::<f32>("t5").unwrap(), data);
}

#[test]
fn st_multi_tensor_10() {
    let mut tensors = Vec::new();
    let mut expected = Vec::new();
    for i in 0..10 {
        let n = (i + 1) * 8;
        let data = make_f32_data(n);
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
        tensors.push((
            format!("tensor_{}", i),
            safetensors::Dtype::F32,
            vec![n],
            bytes,
        ));
        expected.push((format!("tensor_{}", i), data));
    }
    let file = build_safetensors_file(tensors);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 10);
    for (name, data) in &expected {
        assert_eq!(reader.read_as::<f32>(name).unwrap(), *data);
    }
}

#[test]
fn st_model_like() {
    let embed = make_f32_data(100 * 64);
    let q_proj = make_f32_data(64 * 64);
    let k_proj = make_f32_data(64 * 64);
    let v_proj = make_f32_data(64 * 64);
    let gate = make_f16_data(256 * 64);
    let up = make_f16_data(256 * 64);
    let down = make_f16_data(64 * 256);
    let ln = make_f32_data(64);
    let lm_head = make_f32_data(100 * 64);

    let file = build_safetensors_file(vec![
        (
            "embed_tokens.weight".into(),
            safetensors::Dtype::F32,
            vec![100, 64],
            bytemuck::cast_slice(&embed).to_vec(),
        ),
        (
            "layers.0.self_attn.q_proj.weight".into(),
            safetensors::Dtype::F32,
            vec![64, 64],
            bytemuck::cast_slice(&q_proj).to_vec(),
        ),
        (
            "layers.0.self_attn.k_proj.weight".into(),
            safetensors::Dtype::F32,
            vec![64, 64],
            bytemuck::cast_slice(&k_proj).to_vec(),
        ),
        (
            "layers.0.self_attn.v_proj.weight".into(),
            safetensors::Dtype::F32,
            vec![64, 64],
            bytemuck::cast_slice(&v_proj).to_vec(),
        ),
        (
            "layers.0.mlp.gate_proj.weight".into(),
            safetensors::Dtype::F16,
            vec![256, 64],
            bytemuck::cast_slice(&gate).to_vec(),
        ),
        (
            "layers.0.mlp.up_proj.weight".into(),
            safetensors::Dtype::F16,
            vec![256, 64],
            bytemuck::cast_slice(&up).to_vec(),
        ),
        (
            "layers.0.mlp.down_proj.weight".into(),
            safetensors::Dtype::F16,
            vec![64, 256],
            bytemuck::cast_slice(&down).to_vec(),
        ),
        (
            "layers.0.input_layernorm.weight".into(),
            safetensors::Dtype::F32,
            vec![64],
            bytemuck::cast_slice(&ln).to_vec(),
        ),
        (
            "lm_head.weight".into(),
            safetensors::Dtype::F32,
            vec![100, 64],
            bytemuck::cast_slice(&lm_head).to_vec(),
        ),
    ]);

    let reader = SafeTensorsReader::open(file.path()).unwrap();
    assert_eq!(reader.tensors().len(), 9);

    // Verify shapes and dtypes
    let e = reader.get("embed_tokens.weight").unwrap();
    assert_eq!(e.shape, vec![100, 64]);
    assert_eq!(e.components.get("data").unwrap().dtype, DType::F32);

    let g = reader.get("layers.0.mlp.gate_proj.weight").unwrap();
    assert_eq!(g.shape, vec![256, 64]);
    assert_eq!(g.components.get("data").unwrap().dtype, DType::F16);

    // Verify data
    assert_eq!(reader.read_as::<f32>("embed_tokens.weight").unwrap(), embed);
    assert_eq!(
        reader
            .read_as::<f16>("layers.0.mlp.gate_proj.weight")
            .unwrap(),
        gate
    );
    assert_eq!(reader.read_as::<f32>("lm_head.weight").unwrap(), lm_head);
}

#[test]
fn st_zero_copy_slice() {
    let data = make_f32_data(16);
    let raw_bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![(
        "t".into(),
        safetensors::Dtype::F32,
        vec![4, 4],
        raw_bytes.clone(),
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    let slice = reader.view("t").unwrap();
    assert_eq!(slice, &raw_bytes[..]);
}

#[test]
fn st_zero_copy_typed() {
    let data = make_f32_data(16);
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let file = build_safetensors_file(vec![("t".into(), safetensors::Dtype::F32, vec![16], bytes)]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    let typed = reader.view_as::<f32>("t").unwrap();
    assert_eq!(typed, &data[..]);
}

#[test]
fn st_error_not_found() {
    let file = build_safetensors_file(vec![(
        "exists".into(),
        safetensors::Dtype::F32,
        vec![1],
        vec![0; 4],
    )]);
    let reader = SafeTensorsReader::open(file.path()).unwrap();
    match reader.read("missing") {
        Err(Error::ObjectNotFound(_)) => {}
        other => panic!("Expected ObjectNotFound, got {:?}", other),
    }
}
