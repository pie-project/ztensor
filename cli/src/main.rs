use clap::CommandFactory;
use clap::{Parser, Subcommand};
use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};
use humansize::{DECIMAL, format_size};
use safetensors::SafeTensorError;
use safetensors::tensor::SafeTensors;
use serde_pickle::Value as PickleValue;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process;
use ztensor::{ChecksumAlgorithm, ZTensorError, ZTensorReader};
use ztensor::{DType, DataEndianness, Encoding, Layout, ZTensorWriter};

#[derive(Parser)]
#[command(
    name = "ztensor",
    version,
    about = "zTensor CLI: inspect, convert, and compress tensor files",
    long_about = "zTensor CLI is a tool for inspecting, converting, and compressing tensor files.\n\nSupported formats:\n  - SafeTensor (.safetensor, .safetensors)\n  - GGUF (.gguf)\n  - Pickle (.pkl, .pickle)\n  - zTensor (.zt)\n\nYou can convert between formats, compress/decompress zTensor files, and view metadata/stats.",
    author,
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert a tensor file (SafeTensor, GGUF, Pickle) to zTensor format
    #[command(
        about = "Convert a tensor file (SafeTensor, GGUF, Pickle) to zTensor format.",
        long_about = "Convert a tensor file to zTensor format.\n\nSupported input formats: SafeTensor (.safetensor), GGUF (.gguf), Pickle (.pkl, .pickle).\n\nExamples:\n  ztensor convert model.safetensor model.zt\n  ztensor convert --format gguf model.gguf model.zt --compress\n  ztensor convert --format pickle model.pkl model.zt\n"
    )]
    Convert {
        /// Input file (.safetensor, .gguf, .pkl, etc)
        #[arg(help = "Path to the input tensor file (SafeTensor, GGUF, Pickle)")]
        input: String,
        /// Output .zt file
        #[arg(help = "Path to the output .zt file")]
        output: String,
        /// Compress tensor data with zstd
        #[arg(
            long,
            help = "Compress tensor data using zstd (smaller file size, slower read)"
        )]
        compress: bool,
        /// Input format: auto, safetensor, gguf, pickle
        #[arg(
            long,
            value_name = "FORMAT",
            default_value = "auto",
            help = "Input format: auto, safetensor, gguf, pickle. Default: auto-detect from extension."
        )]
        format: String,
    },
    /// Compress an uncompressed zTensor file (raw encoding) to zstd encoding
    #[command(
        about = "Compress an uncompressed zTensor file (raw encoding) to zstd encoding.",
        long_about = "Compress a zTensor file that is currently uncompressed (raw encoding) to zstd encoding.\n\nExample:\n  ztensor compress model_raw.zt model_compressed.zt\n"
    )]
    Compress {
        /// Input .zt file (uncompressed)
        #[arg(help = "Path to the input .zt file (must be uncompressed)")]
        input: String,
        /// Output .zt file (compressed)
        #[arg(help = "Path to the output .zt file (compressed)")]
        output: String,
    },
    /// Decompress a compressed zTensor file (zstd encoding) to raw encoding
    #[command(
        about = "Decompress a compressed zTensor file (zstd encoding) to raw encoding.",
        long_about = "Decompress a zTensor file that is compressed (zstd encoding) to raw encoding.\n\nExample:\n  ztensor decompress model_compressed.zt model_raw.zt\n"
    )]
    Decompress {
        /// Input .zt file (compressed)
        #[arg(help = "Path to the input .zt file (must be compressed)")]
        input: String,
        /// Output .zt file (uncompressed)
        #[arg(help = "Path to the output .zt file (uncompressed)")]
        output: String,
    },
    /// Show metadata and stats for a zTensor file
    #[command(
        about = "Show metadata and stats for a zTensor file.",
        long_about = "Display metadata and statistics for a zTensor file, including tensor names, shapes, data types, encodings, and sizes.\n\nExample:\n  ztensor info model.zt\n"
    )]
    Info {
        /// Path to the .zt file
        #[arg(help = "Path to the .zt file to inspect")]
        file: String,
    },
}

fn print_tensor_metadata(meta: &ztensor::TensorMetadata) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Field", "Value"]);
    table.add_row(Row::from(vec![Cell::new("Name"), Cell::new(&meta.name)]));
    table.add_row(Row::from(vec![
        Cell::new("DType"),
        Cell::new(format!("{:?}", meta.dtype)),
    ]));
    table.add_row(Row::from(vec![
        Cell::new("Encoding"),
        Cell::new(format!("{:?}", meta.encoding)),
    ]));
    table.add_row(Row::from(vec![
        Cell::new("Layout"),
        Cell::new(format!("{:?}", meta.layout)),
    ]));
    let shape_str = if meta.shape.len() == 0 {
        "(scalar)".to_string()
    } else {
        format!(
            "[{}]",
            meta.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    table.add_row(Row::from(vec![Cell::new("Shape"), Cell::new(shape_str)]));
    table.add_row(Row::from(vec![
        Cell::new("Offset"),
        Cell::new(meta.offset.to_string()),
    ]));
    table.add_row(Row::from(vec![
        Cell::new("Size (on-disk)"),
        Cell::new(format_size(meta.size, DECIMAL)),
    ]));
    if let Some(endian) = &meta.data_endianness {
        table.add_row(Row::from(vec![
            Cell::new("Data Endianness"),
            Cell::new(format!("{:?}", endian)),
        ]));
    }
    if let Some(cs) = &meta.checksum {
        table.add_row(Row::from(vec![Cell::new("Checksum"), Cell::new(cs)]));
    }
    if !meta.custom_fields.is_empty() {
        table.add_row(Row::from(vec![
            Cell::new("Custom fields"),
            Cell::new(format!("{:?}", meta.custom_fields)),
        ]));
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

fn convert_safetensor_to_ztensor(
    input: &str,
    output: &str,
    compress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
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
            if compress {
                Encoding::Zstd
            } else {
                Encoding::Raw
            },
            data,
            Some(DataEndianness::Little),
            ChecksumAlgorithm::None,
            Some(BTreeMap::new()),
        )?;
    }
    writer.finalize()?;
    Ok(())
}

fn convert_gguf_to_ztensor(
    input: &str,
    output: &str,
    compress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut container = gguf_rs::get_gguf_container(input)?;
    let model = container.decode()?;
    let mut writer = ZTensorWriter::create(output)?;

    let mut f = File::open(input)?;

    for tensor_info in model.tensors() {
        let ggml_dtype: gguf_rs::GGMLType = tensor_info
            .kind
            .try_into()
            .map_err(|e| format!("Unsupported GGUF dtype: {}", e))?;

        if let Some(dtype) = gguf_dtype_to_ztensor(&ggml_dtype) {
            let shape: Vec<u64> = tensor_info
                .shape
                .iter()
                .map(|&d| d as u64)
                .filter(|&d| d > 0)
                .collect();
            let numel: u64 = shape.iter().product();
            let type_size = match dtype {
                ztensor::DType::Float64 | ztensor::DType::Int64 | ztensor::DType::Uint64 => 8,
                ztensor::DType::Float32 | ztensor::DType::Int32 | ztensor::DType::Uint32 => 4,
                ztensor::DType::BFloat16
                | ztensor::DType::Float16
                | ztensor::DType::Int16
                | ztensor::DType::Uint16 => 2,
                ztensor::DType::Int8 | ztensor::DType::Uint8 | ztensor::DType::Bool => 1,
                _ => {
                    return Err(format!(
                        "Unsupported ztensor dtype for size calculation: {:?}",
                        dtype
                    )
                    .into());
                }
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
                if compress {
                    Encoding::Zstd
                } else {
                    Encoding::Raw
                },
                data,
                Some(DataEndianness::Little), // GGUF is typically little endian, adjust if needed
                ChecksumAlgorithm::None,
                Some(BTreeMap::new()),
            )?;
        } else {
            eprintln!(
                "Warning: Skipping tensor '{}' with unsupported GGUF dtype: {:?}",
                tensor_info.name, ggml_dtype
            );
        }
    }
    writer.finalize()?;
    Ok(())
}

// Helper function to convert PickleValue keys to strings for tensor naming
fn key_to_string(key_val: &serde_pickle::HashableValue) -> String {
    use serde_pickle::HashableValue;
    match key_val {
        HashableValue::String(s) => s.clone(),
        HashableValue::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
        HashableValue::I64(i) => i.to_string(),
        HashableValue::F64(f) => f.to_string(),
        HashableValue::Bool(b) => b.to_string(),
        HashableValue::Tuple(items) => items
            .iter()
            .map(key_to_string)
            .collect::<Vec<_>>()
            .join("_"),
        // Add more specific cases if other key types are common and need specific formatting
        _ => format!("key_{:?}", key_val).replace(|c: char| !c.is_alphanumeric() && c != '_', "_"),
    }
}

// Helper to parse a PickleValue as a potential tensor
fn try_parse_pickle_tensor(
    value: &PickleValue,
) -> Result<Option<(Vec<u64>, ztensor::DType, Vec<u8>)>, String> {
    let mut data_bytes = Vec::new();
    let mut inferred_dtype: Option<ztensor::DType> = None;
    let mut shape = Vec::new();

    fn ensure_shape_depth(shape: &Vec<u64>, depth: usize) -> Result<(), String> {
        if shape.len() != depth {
            Err(format!(
                "Dimension mismatch: expected scalar at depth {}, but tensor shape is deeper. Shape: {:?}, Depth: {}",
                depth, shape, depth
            ))
        } else {
            Ok(())
        }
    }

    fn collect_elements(
        current_value: &PickleValue,
        current_depth: usize,
        shape: &mut Vec<u64>,
        data_bytes: &mut Vec<u8>,
        dtype: &mut Option<ztensor::DType>,
    ) -> Result<(), String> {
        match current_value {
            PickleValue::List(items) | PickleValue::Tuple(items) => {
                if current_depth == 0 && items.is_empty() {
                    // Top-level empty list
                    shape.push(0); // Represents an empty tensor
                    // Default to Float32 for empty tensor if no other type info
                    if dtype.is_none() {
                        *dtype = Some(ztensor::DType::Float32);
                    }
                    return Ok(());
                }
                // Disallow empty lists/tuples within a tensor structure for now, or define behavior.
                if items.is_empty() {
                    return Err("Empty list/tuple found within tensor structure.".to_string());
                }

                if shape.len() == current_depth {
                    shape.push(items.len() as u64);
                } else if shape[current_depth] != items.len() as u64 {
                    return Err(format!(
                        "Jagged tensor: dimension mismatch at depth {}. Expected {}, got {}. Shape: {:?}",
                        current_depth,
                        shape[current_depth],
                        items.len(),
                        shape
                    ));
                }

                for item in items {
                    collect_elements(item, current_depth + 1, shape, data_bytes, dtype)?;
                }
                Ok(())
            }
            PickleValue::F64(f) => {
                if dtype.is_none() {
                    *dtype = Some(ztensor::DType::Float64);
                }
                if *dtype != Some(ztensor::DType::Float64) {
                    return Err("Mixed data types: expected Float64".to_string());
                }
                ensure_shape_depth(shape, current_depth)?;
                data_bytes.extend_from_slice(&f.to_le_bytes());
                Ok(())
            }
            PickleValue::I64(i) => {
                // serde_pickle reads Python int as I64
                if dtype.is_none() {
                    *dtype = Some(ztensor::DType::Int64);
                } // Default to Int64
                if *dtype != Some(ztensor::DType::Int64) {
                    return Err("Mixed data types: expected Int64".to_string());
                }
                ensure_shape_depth(shape, current_depth)?;
                data_bytes.extend_from_slice(&i.to_le_bytes());
                Ok(())
            }
            PickleValue::Bool(b) => {
                if dtype.is_none() {
                    *dtype = Some(ztensor::DType::Bool);
                }
                if *dtype != Some(ztensor::DType::Bool) {
                    return Err("Mixed data types: expected Bool".to_string());
                }
                ensure_shape_depth(shape, current_depth)?;
                data_bytes.push(if *b { 1 } else { 0 });
                Ok(())
            }
            _ => Err(format!(
                "Unsupported data type in tensor: {:?}",
                current_value
            )),
        }
    }

    match collect_elements(value, 0, &mut shape, &mut data_bytes, &mut inferred_dtype) {
        Ok(()) => {
            if let Some(dt) = inferred_dtype {
                // If shape is empty AND data_bytes is not empty, it was a scalar. zTensor uses empty shape vec for scalars.
                // If shape is [0], it was an empty list [], also a valid tensor.
                Ok(Some((shape, dt, data_bytes)))
            } else {
                // This case should ideally not be reached if collect_elements succeeds and infers a dtype.
                // If value was an empty list, shape would be [0] and dtype set.
                Ok(None)
            }
        }
        Err(_ /*e*/) => {
            // eprintln!("Tensor parsing error for value {:?}: {}", value.variant_name(), e); // Debugging
            Ok(None) // Treat parsing errors as "not a tensor" for this path
        }
    }
}

// Recursively traverses PickleValue, identifies tensors, and flattens names.
fn pickle_value_to_ztensor_components(
    prefix: &str,
    value: &PickleValue,
    tensors: &mut Vec<(String, Vec<u64>, ztensor::DType, Vec<u8>)>,
) -> Result<(), String> {
    match value {
        PickleValue::Dict(items) => {
            for (key_val, sub_value) in items {
                let key_str = key_to_string(key_val);
                let new_prefix = if prefix.is_empty() {
                    key_str
                } else {
                    format!("{}.{}", prefix, key_str)
                };
                pickle_value_to_ztensor_components(&new_prefix, sub_value, tensors)?;
            }
        }
        _ => {
            // Attempt to parse any other value as a tensor (includes lists, tuples, scalars)
            if let Some((shape, dtype, data)) = try_parse_pickle_tensor(value)? {
                // Add if data is present, or if it\'s an explicitly empty tensor (shape [0])
                if !data.is_empty() || shape == vec![0] {
                    // Use current prefix as the name if it\'s a direct tensor,
                    // or if it\'s a scalar found not inside a dict.
                    let tensor_name = if prefix.is_empty() {
                        "tensor".to_string()
                    } else {
                        prefix.to_string()
                    };
                    tensors.push((tensor_name, shape, dtype, data));
                }
                // If it was parsed as a tensor, we don\'t recurse further into its structure.
            } else {
                // If not a dict and not a parseable tensor, it might be a list of other things.
                // The prompt says "discard other metadata".
                // If `value` was a List/Tuple and `try_parse_pickle_tensor` returned None,
                // it means it wasn\'t a uniform tensor. We could recurse into its elements
                // if they could be dicts leading to more tensors, using index-based names.
                if let PickleValue::List(sub_items) | PickleValue::Tuple(sub_items) = value {
                    for (idx, item) in sub_items.iter().enumerate() {
                        let new_prefix = if prefix.is_empty() {
                            format!("item_{}", idx)
                        } else {
                            format!("{}.{}", prefix, idx)
                        };
                        pickle_value_to_ztensor_components(&new_prefix, item, tensors)?;
                    }
                }
                // Otherwise (e.g. a standalone string, None not part of a dict), ignore.
            }
        }
    }
    Ok(())
}

fn convert_pickle_to_ztensor(
    input: &str,
    output: &str,
    compress: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let file =
        File::open(input).map_err(|e| format!("Failed to open input file '{}': {}", input, e))?;
    let reader = BufReader::new(file);

    let pkl_value: PickleValue = match serde_pickle::value_from_reader(reader, Default::default()) {
        Ok(val) => val,
        Err(e) => {
            return Err(format!(
                "Failed to deserialize Pickle file '{}': {}\n\
This often happens with PyTorch .pt/.bin files containing custom objects or large tensors.\n\
For best results, save your tensors in Python as a dict of numpy arrays or lists, e.g.:\n\
    import torch, pickle\n    tensors = {{k: v.cpu().numpy() for k, v in torch.load('pytorch_model.bin', map_location='cpu').items()}}\n    with open('model_simple.pkl', 'wb') as f: pickle.dump(tensors, f)\n\
Then use this tool on 'model_simple.pkl'.",
                input, e
            ).into());
        }
    };

    let mut writer = ZTensorWriter::create(output).map_err(|e| {
        format!(
            "Failed to create ZTensorWriter for output '{}': {}",
            output, e
        )
    })?;

    let mut found_tensors: Vec<(String, Vec<u64>, ztensor::DType, Vec<u8>)> = Vec::new();

    pickle_value_to_ztensor_components("", &pkl_value, &mut found_tensors)
        .map_err(|e| format!("Error processing pickle structure from '{}': {}", input, e))?;

    if found_tensors.is_empty() {
        return Err(format!(
            "No compatible (numpy/pytorch-like) tensors found in '{}'.",
            input
        )
        .into());
    }

    for (name, shape, dtype, data) in found_tensors {
        let cleaned_name = if name.is_empty() {
            "unnamed_tensor".to_string()
        } else {
            name
        };
        writer.add_tensor(
            &cleaned_name,
            shape,
            dtype,
            Layout::Dense, // Assuming dense layout
            if compress {
                Encoding::Zstd
            } else {
                Encoding::Raw
            },
            data,
            Some(DataEndianness::Little), // Common default; NumPy/PyTorch use native.
            ChecksumAlgorithm::None,
            Some(BTreeMap::new()), // No custom fields extracted
        )?;
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
        BFloat16 | Float16 | Int16 | Uint16 => 2,
        Int8 | Uint8 | Bool => 1,
        _ => return None,
    })
}

fn shape_numel(shape: &[u64]) -> u64 {
    shape.iter().product()
}

fn print_tensors_table(tensors: &[ztensor::TensorMetadata]) {
    let mut table = Table::new();
    table
        .load_preset(comfy_table::presets::UTF8_HORIZONTAL_ONLY)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec![
        "#",
        "Name",
        "Shape",
        "DType",
        "Encoding",
        "Layout",
        "Offset",
        "Size",
        "On-disk Size",
    ]);
    for (i, meta) in tensors.iter().enumerate() {
        let shape_str = if meta.shape.is_empty() {
            "(scalar)".to_string()
        } else {
            format!(
                "[{}]",
                meta.shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let dtype_str = format!("{:?}", meta.dtype);
        let encoding_str = format!("{:?}", meta.encoding);
        let layout_str = format!("{:?}", meta.layout);
        let offset_str = meta.offset.to_string();
        let numel = shape_numel(&meta.shape);
        let size = dtype_size(&meta.dtype).map(|s| s as u64 * numel);
        let size_str = size
            .map(|s| format_size(s, DECIMAL))
            .unwrap_or("?".to_string());
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

fn detect_format_from_extension(input: &str) -> Option<&'static str> {
    let input = input.to_ascii_lowercase();
    if input.ends_with(".safetensor") || input.ends_with(".safetensors") {
        Some("safetensor")
    } else if input.ends_with(".gguf") {
        Some("gguf")
    } else if input.ends_with(".pkl") || input.ends_with(".pickle") {
        Some("pickle")
    } else {
        None
    }
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Some(Commands::Convert {
            input,
            output,
            compress,
            format,
        }) => {
            let format = format.to_ascii_lowercase();
            let detected_format = if format == "auto" {
                detect_format_from_extension(input)
            } else {
                Some(format.as_str())
            };
            match detected_format {
                Some("safetensor") => {
                    match convert_safetensor_to_ztensor(input, output, *compress) {
                        Ok(()) => println!("Successfully converted {} to {}", input, output),
                        Err(e) => {
                            eprintln!("Conversion failed: {}", e);
                            process::exit(1);
                        }
                    }
                }
                Some("gguf") => match convert_gguf_to_ztensor(input, output, *compress) {
                    Ok(()) => println!(
                        "Successfully converted GGUF {} to zTensor {}",
                        input, output
                    ),
                    Err(e) => {
                        eprintln!("GGUF to zTensor conversion failed: {}", e);
                        process::exit(1);
                    }
                },
                Some("pickle") => match convert_pickle_to_ztensor(input, output, *compress) {
                    Ok(()) => println!(
                        "Successfully converted Pickle {} to zTensor {}",
                        input, output
                    ),
                    Err(e) => {
                        eprintln!("Pickle to zTensor conversion failed: {}", e);
                        process::exit(1);
                    }
                },
                _ => {
                    eprintln!(
                        "Could not auto-detect input format for '{}'. Please specify --format [safetensor|gguf|pickle]",
                        input
                    );
                    process::exit(1);
                }
            }
        }
        Some(Commands::Compress { input, output }) => match compress_ztensor(input, output) {
            Ok(()) => println!("Successfully compressed {} to {}", input, output),
            Err(e) => {
                eprintln!("Compression failed: {}", e);
                process::exit(1);
            }
        },
        Some(Commands::Decompress { input, output }) => match decompress_ztensor(input, output) {
            Ok(()) => println!("Successfully decompressed {} to {}", input, output),
            Err(e) => {
                eprintln!("Decompression failed: {}", e);
                process::exit(1);
            }
        },
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
