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
use clap::{Parser, Subcommand};
use clap::CommandFactory;
use comfy_table::{Table, Row, Cell, presets::UTF8_FULL, ContentArrangement};
use humansize::{format_size, DECIMAL};

#[derive(Parser)]
#[command(name = "ztensor", version, about = "zTensor CLI: inspect and convert tensor files", author)]
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
    table.load_preset(UTF8_FULL)
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
