use std::collections::BTreeMap;
use std::env;
use std::process;
use std::path::Path;
use ztensor::{ChecksumAlgorithm, ZTensorReader, ZTensorError};
use safetensors::SafeTensorError;
use safetensors::tensor::SafeTensors;
use std::fs::File;
use std::io::{BufWriter, Write, Seek, SeekFrom};
use ztensor::{ZTensorWriter, DType, Layout, Encoding, DataEndianness};

fn print_tensor_metadata(meta: &ztensor::TensorMetadata) {
    println!("  Name: {}", meta.name);
    println!("  DType: {:?}", meta.dtype);
    println!("  Encoding: {:?}", meta.encoding);
    println!("  Layout: {:?}", meta.layout);
    println!("  Shape: {:?}", meta.shape);
    println!("  Offset: {}", meta.offset);
    println!("  Size (on-disk): {} bytes", meta.size);
    if let Some(endian) = &meta.data_endianness {
        println!("  Data Endianness: {:?}", endian);
    }
    if let Some(cs) = &meta.checksum {
        println!("  Checksum: {}", cs);
    }
    if !meta.custom_fields.is_empty() {
        println!("  Custom fields: {:?}", meta.custom_fields);
    }
}

fn safetensor_dtype_to_ztensor(dtype: &safetensors::tensor::Dtype) -> Option<ztensor::DType> {
    match dtype {
        safetensors::tensor::Dtype::F64 => Some(ztensor::DType::Float64),
        safetensors::tensor::Dtype::F32 => Some(ztensor::DType::Float32),
        safetensors::tensor::Dtype::F16 => Some(ztensor::DType::Float16),
        safetensors::tensor::Dtype::BF16 => Some(ztensor::DType::BFloat16),
        safetensors::tensor::Dtype::I64 => Some(ztensor::DType::Int64),
        safetensors::tensor::Dtype::I32 => Some(ztensor::DType::Int32),
        safetensors::tensor::Dtype::I16 => Some(ztensor::DType::Int16),
        safetensors::tensor::Dtype::I8 => Some(ztensor::DType::Int8),
        safetensors::tensor::Dtype::U64 => Some(ztensor::DType::Uint64),
        safetensors::tensor::Dtype::U32 => Some(ztensor::DType::Uint32),
        safetensors::tensor::Dtype::U16 => Some(ztensor::DType::Uint16),
        safetensors::tensor::Dtype::U8 => Some(ztensor::DType::Uint8),
        safetensors::tensor::Dtype::BOOL => Some(ztensor::DType::Bool),
        _ => None,
    }
}

fn convert_safetensor_to_ztensor(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    let mut writer = ZTensorWriter::create(output)?;
    for (name, tensor) in st.tensors() {
        let dtype = safetensor_dtype_to_ztensor(&tensor.dtype())
            .ok_or_else(|| format!("Unsupported dtype: {:?}", tensor.dtype()))?;
        let shape: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();
        let data = tensor.data().to_vec();
        writer.add_tensor(
            &name,
            shape,
            dtype,
            Layout::Dense,
            Encoding::Raw,
            data,
            Some(DataEndianness::Little),
            ChecksumAlgorithm::None,
            Some(BTreeMap::new()),
        )?;
    }
    writer.finalize()?;
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 4 && args[1] == "convert" && args[2].ends_with(".safetensor") && args[3].ends_with(".zt") {
        match convert_safetensor_to_ztensor(&args[2], &args[3]) {
            Ok(()) => {
                println!("Successfully converted {} to {}", args[2], args[3]);
            }
            Err(e) => {
                eprintln!("Conversion failed: {}", e);
                process::exit(1);
            }
        }
        return;
    }
    if args.len() < 2 {
        eprintln!("Usage: {} <ztensor-file>", args[0]);
        process::exit(1);
    }
    let file_path = &args[1];
    if !Path::new(file_path).exists() {
        eprintln!("File not found: {}", file_path);
        process::exit(1);
    }
    match ZTensorReader::open(file_path) {
        Ok(reader) => {
            let tensors = reader.list_tensors();
            println!("zTensor file: {}", file_path);
            println!("Number of tensors: {}", tensors.len());
            for (i, meta) in tensors.iter().enumerate() {
                println!("\n--- Tensor {} ---", i);
                print_tensor_metadata(meta);
            }
        }
        Err(e) => {
            eprintln!("Failed to open zTensor file: {}", e);
            process::exit(1);
        }
    }
}
