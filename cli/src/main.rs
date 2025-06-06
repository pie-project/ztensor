use std::collections::BTreeMap;
use std::env;
use std::process;
use std::path::Path;
use ztensor::{ChecksumAlgorithm, ZTensorReader, ZTensorError};
use safetensors::SafeTensorError;
use safetensors::tensor::SafeTensors;
use std::fs::File;
use std::io::{BufWriter, Write, Seek, SeekFrom, Read};
use ztensor::{ZTensorWriter, DType, Layout, Encoding, DataEndianness};
use clap::{Parser, Subcommand};
use clap::CommandFactory;
use comfy_table::{Table, Row, Cell, presets::UTF8_FULL, ContentArrangement};
use humansize::{format_size, DECIMAL};

#[derive(Parser)]
#[command(name = "ztensor", version, about = "zTensor CLI: inspect and convert tensor files", author, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Show metadata and stats for a zTensor file
    Info {
        /// Path to the .zt file
        file: String,
    },
    /// Convert a SafeTensor file to zTensor format
    Convert {
        /// Input .safetensor file
        input: String,
        /// Output .zt file
        output: String,
        /// Compress tensor data with zstd
        #[arg(long)]
        compress: bool,
    },
    /// Convert a GGUF file to zTensor format
    ConvertGGUF {
        /// Input .gguf file
        input: String,
        /// Output .zt file
        output: String,
        /// Compress tensor data with zstd
        #[arg(long)]
        compress: bool,
    },
    /// Compress an uncompressed zTensor file (raw encoding) to zstd encoding
    Compress {
        /// Input .zt file (uncompressed)
        input: String,
        /// Output .zt file (compressed)
        output: String,
    },
    /// Decompress a compressed zTensor file (zstd encoding) to raw encoding
    Decompress {
        /// Input .zt file (compressed)
        input: String,
        /// Output .zt file (uncompressed)
        output: String,
    },
}

fn print_tensor_metadata(meta: &ztensor::TensorMetadata) {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Field", "Value"]);
    table.add_row(Row::from(vec![Cell::new("Name"), Cell::new(&meta.name)]));
    table.add_row(Row::from(vec![Cell::new("DType"), Cell::new(format!("{:?}", meta.dtype))]));
    table.add_row(Row::from(vec![Cell::new("Encoding"), Cell::new(format!("{:?}", meta.encoding))]));
    table.add_row(Row::from(vec![Cell::new("Layout"), Cell::new(format!("{:?}", meta.layout))]));
    let shape_str = if meta.shape.len() == 0 {
        "(scalar)".to_string()
    } else {
        format!("[{}]", meta.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
    };
    table.add_row(Row::from(vec![Cell::new("Shape"), Cell::new(shape_str)]));
    table.add_row(Row::from(vec![Cell::new("Offset"), Cell::new(meta.offset.to_string())]));
    table.add_row(Row::from(vec![Cell::new("Size (on-disk)"), Cell::new(format_size(meta.size, DECIMAL))]));
    if let Some(endian) = &meta.data_endianness {
        table.add_row(Row::from(vec![Cell::new("Data Endianness"), Cell::new(format!("{:?}", endian))]));
    }
    if let Some(cs) = &meta.checksum {
        table.add_row(Row::from(vec![Cell::new("Checksum"), Cell::new(cs)]));
    }
    if !meta.custom_fields.is_empty() {
        table.add_row(Row::from(vec![Cell::new("Custom fields"), Cell::new(format!("{:?}", meta.custom_fields))]));
    }
    println!("{}", table);
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

fn gguf_dtype_to_ztensor(dtype: &gguf_rs::GGMLType) -> Option<ztensor::DType> {
    match dtype {
        gguf_rs::GGMLType::F64 => Some(ztensor::DType::Float64),
        gguf_rs::GGMLType::F32 => Some(ztensor::DType::Float32),
        gguf_rs::GGMLType::F16 => Some(ztensor::DType::Float16),
        gguf_rs::GGMLType::BF16 => Some(ztensor::DType::BFloat16),
        gguf_rs::GGMLType::I64 => Some(ztensor::DType::Int64),
        gguf_rs::GGMLType::I32 => Some(ztensor::DType::Int32),
        gguf_rs::GGMLType::I16 => Some(ztensor::DType::Int16),
        gguf_rs::GGMLType::I8 => Some(ztensor::DType::Int8),
        // gguf_rs::GGMLType::BOOL => Some(ztensor::DType::Bool), // GGUF does not have a BOOL type
        _ => None,
    }
}

fn convert_safetensor_to_ztensor(input: &str, output: &str, compress: bool) -> Result<(), Box<dyn std::error::Error>> {
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
            if compress { Encoding::Zstd } else { Encoding::Raw },
            data,
            Some(DataEndianness::Little),
            ChecksumAlgorithm::None,
            Some(BTreeMap::new()),
        )?;
    }
    writer.finalize()?;
    Ok(())
}

fn convert_gguf_to_ztensor(input: &str, output: &str, compress: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut container = gguf_rs::get_gguf_container(input)?;
    let model = container.decode()?;
    let mut writer = ZTensorWriter::create(output)?;

    let mut f = File::open(input)?;

    for tensor_info in model.tensors() {
        let ggml_dtype: gguf_rs::GGMLType = tensor_info.kind.try_into()
            .map_err(|e| format!("Unsupported GGUF dtype: {}", e))?;

        if let Some(dtype) = gguf_dtype_to_ztensor(&ggml_dtype) {
            let shape: Vec<u64> = tensor_info.shape.iter().map(|&d| d as u64).filter(|&d| d > 0).collect();
            let numel: u64 = shape.iter().product();
            let type_size = match dtype {
                ztensor::DType::Float64 | ztensor::DType::Int64 | ztensor::DType::Uint64 => 8,
                ztensor::DType::Float32 | ztensor::DType::Int32 | ztensor::DType::Uint32 => 4,
                ztensor::DType::BFloat16 | ztensor::DType::Float16  | ztensor::DType::Int16 | ztensor::DType::Uint16 => 2,
                ztensor::DType::Int8 | ztensor::DType::Uint8 | ztensor::DType::Bool => 1,
                _ => return Err(format!("Unsupported ztensor dtype for size calculation: {:?}", dtype).into()),
            };
            let data_size = numel * type_size;
            let mut data = vec![0u8; data_size as usize];

            f.seek(SeekFrom::Start(tensor_info.offset))?;
            f.read_exact(&mut data)?;

            writer.add_tensor(
                &tensor_info.name,
                shape,
                dtype,
                Layout::Dense,
                if compress { Encoding::Zstd } else { Encoding::Raw },
                data,
                Some(DataEndianness::Little), // GGUF is typically little endian, adjust if needed
                ChecksumAlgorithm::None,
                Some(BTreeMap::new()),
            )?;
        } else {
            eprintln!("Warning: Skipping tensor '{}' with unsupported GGUF dtype: {:?}", tensor_info.name, ggml_dtype);
        }
    }
    writer.finalize()?;
    Ok(())
}

fn compress_ztensor(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = ZTensorReader::open(input)?;
    let mut writer = ZTensorWriter::create(output)?;
    let metas = reader.list_tensors().clone();
    for meta in metas {
        let data = reader.read_raw_tensor_data(&meta)?;
        writer.add_tensor(
            &meta.name,
            meta.shape.clone(),
            meta.dtype.clone(),
            meta.layout.clone(),
            Encoding::Zstd,
            data,
            meta.data_endianness.clone(),
            ChecksumAlgorithm::None,
            Some(meta.custom_fields.clone()),
        )?;
    }
    writer.finalize()?;
    Ok(())
}

fn decompress_ztensor(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = ZTensorReader::open(input)?;
    let mut writer = ZTensorWriter::create(output)?;
    let metas = reader.list_tensors().clone();
    for meta in metas {
        let data = reader.read_raw_tensor_data(&meta)?;
        writer.add_tensor(
            &meta.name,
            meta.shape.clone(),
            meta.dtype.clone(),
            meta.layout.clone(),
            Encoding::Raw,
            data,
            meta.data_endianness.clone(),
            ChecksumAlgorithm::None,
            Some(meta.custom_fields.clone()),
        )?;
    }
    writer.finalize()?;
    Ok(())
}

fn dtype_size(dtype: &ztensor::DType) -> Option<usize> {
    use ztensor::DType::*;
    Some(match dtype {
        Float64 | Int64 | Uint64 => 8,
        Float32 | Int32 | Uint32 => 4,
        BFloat16 | Float16  | Int16 | Uint16 => 2,
        Int8 | Uint8 | Bool => 1,
        _ => return None,
    })
}

fn shape_numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

fn print_tensors_table(tensors: &[ztensor::TensorMetadata]) {
    let mut table = Table::new();
    table.load_preset(comfy_table::presets::UTF8_HORIZONTAL_ONLY)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec![
        "#", "Name", "Shape", "DType", "Encoding", "Layout", "Offset", "Size", "On-disk Size"
    ]);
    for (i, meta) in tensors.iter().enumerate() {
        let shape_str = if meta.shape.is_empty() {
            "(scalar)".to_string()
        } else {
            format!("[{}]", meta.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
        };
        let dtype_str = format!("{:?}", meta.dtype);
        let encoding_str = format!("{:?}", meta.encoding);
        let layout_str = format!("{:?}", meta.layout);
        let offset_str = meta.offset.to_string();
        let numel = shape_numel(&meta.shape);
        let size = dtype_size(&meta.dtype).map(|s| s as u64 * numel);
        let size_str = size.map(|s| format_size(s, DECIMAL)).unwrap_or("?".to_string());
        let on_disk_str = format_size(meta.size, DECIMAL);
        table.add_row(Row::from(vec![
            Cell::new(i),
            Cell::new(&meta.name),
            Cell::new(shape_str),
            Cell::new(dtype_str),
            Cell::new(encoding_str),
            Cell::new(layout_str),
            Cell::new(offset_str),
            Cell::new(size_str),
            Cell::new(on_disk_str),
        ]));
    }
    println!("{}", table);
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Some(Commands::Convert { input, output, compress }) => {
            match convert_safetensor_to_ztensor(input, output, *compress) {
                Ok(()) => println!("Successfully converted {} to {}", input, output),
                Err(e) => {
                    eprintln!("Conversion failed: {}", e);
                    process::exit(1);
                }
            }
        }
        Some(Commands::ConvertGGUF { input, output, compress }) => {
            match convert_gguf_to_ztensor(input, output, *compress) {
                Ok(()) => println!("Successfully converted GGUF {} to zTensor {}", input, output),
                Err(e) => {
                    eprintln!("GGUF to zTensor conversion failed: {}", e);
                    process::exit(1);
                }
            }
        }
        Some(Commands::Compress { input, output }) => {
            match compress_ztensor(input, output) {
                Ok(()) => println!("Successfully compressed {} to {}", input, output),
                Err(e) => {
                    eprintln!("Compression failed: {}", e);
                    process::exit(1);
                }
            }
        }
        Some(Commands::Decompress { input, output }) => {
            match decompress_ztensor(input, output) {
                Ok(()) => println!("Successfully decompressed {} to {}", input, output),
                Err(e) => {
                    eprintln!("Decompression failed: {}", e);
                    process::exit(1);
                }
            }
        }
        Some(Commands::Info { file }) => {
            if !Path::new(file).exists() {
                eprintln!("File not found: {}", file);
                process::exit(1);
            }
            match ZTensorReader::open(file) {
                Ok(reader) => {
                    let tensors = reader.list_tensors();
                    println!("zTensor file: {}", file);
                    println!("Number of tensors: {}", tensors.len());
                    print_tensors_table(&tensors);
                }
                Err(e) => {
                    eprintln!("Failed to open zTensor file: {}", e);
                    process::exit(1);
                }
            }
        }
        None => {
            Cli::command().print_help().unwrap();
            println!();
            process::exit(1);
        }
    }
}
