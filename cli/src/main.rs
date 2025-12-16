use clap::CommandFactory;
use clap::{Parser, Subcommand};
use comfy_table::{Cell, ContentArrangement, Row, Table, presets::UTF8_FULL};
use safetensors::tensor::SafeTensors;
use serde_pickle::Value as PickleValue;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::process;
use ztensor::{ChecksumAlgorithm, ZTensorReader};
use ztensor::{Encoding, ZTensorWriter};

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
        about = "Convert one or more tensor files (SafeTensor, GGUF, Pickle) to a single zTensor file.",
        long_about = "Convert one or more tensor files to a single zTensor file.\n\nSupported input formats: SafeTensor (.safetensor), GGUF (.gguf), Pickle (.pkl, .pickle).\n\nExamples:\n  ztensor convert model.safetensor model.zt\n  ztensor convert --format gguf model1.gguf model2.gguf model.zt --preserve-original false\n  ztensor convert --format pickle model1.pkl model2.pkl model.zt\n"
    )]
    Convert {
        /// Input files (.safetensor, .gguf, .pkl, etc)
        #[arg(help = "Paths to the input tensor files (SafeTensor, GGUF, Pickle)", required = true)]
        inputs: Vec<String>,
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
        /// Preserve original files after converting
        #[arg(long, default_value_t = true, help = "Preserve original files after converting (default: true)")]
        preserve_original: bool,
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
    /// Merge multiple zTensor files into a single file
    #[command(
        about = "Merge multiple zTensor files into a single file.",
        long_about = "Merge multiple zTensor files into a single file. Optionally delete the originals after merging for storage savings.\n\nExample:\n  ztensor merge --preserve-original false merged.zt file1.zt file2.zt file3.zt\n"
    )]
    Merge {
        /// Output .zt file (merged)
        #[arg(help = "Path to the output merged .zt file")]
        output: String,
        /// Input .zt files to merge
        #[arg(help = "Paths to the input .zt files to merge", required = true)]
        inputs: Vec<String>,
        /// Preserve original files after merging
        #[arg(long, default_value_t = true, help = "Preserve original files after merging (default: true)")]
        preserve_original: bool,
    },
}

fn print_tensor_metadata(name: &str, tensor: &ztensor::Tensor) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec!["Field", "Value"]);
    table.add_row(Row::from(vec![Cell::new("Name"), Cell::new(name)]));
    table.add_row(Row::from(vec![
        Cell::new("DType"),
        Cell::new(format!("{:?}", tensor.dtype)),
    ]));
    // Encoding is per-component now. We can show it for "data" or "values" if present.
    // For summary, maybe just show format.
    table.add_row(Row::from(vec![
        Cell::new("Format"),
        Cell::new(&tensor.format),
    ]));
    
    let shape_str = if tensor.shape.is_empty() {
        "(scalar)".to_string()
    } else {
        format!(
            "[{}]",
            tensor.shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    table.add_row(Row::from(vec![Cell::new("Shape"), Cell::new(shape_str)]));
    
    // Components info
    let mut components_str = String::new();
    for (key, comp) in &tensor.components {
        components_str.push_str(&format!("{}: [offset={}, len={}", key, comp.offset, comp.length));
        if let Some(enc) = &comp.encoding {
             components_str.push_str(&format!(", enc={:?}", enc));
        }
         if let Some(dig) = &comp.digest {
             components_str.push_str(&format!(", digest={}", dig));
        }
        components_str.push_str("]\n");
    }
    table.add_row(Row::from(vec![
        Cell::new("Components"),
        Cell::new(components_str.trim()),
    ]));
    
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
                    return Err("Mixed data types: expected Int64". to_string());
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


fn compress_ztensor(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = ZTensorReader::open(input)?;
    let mut writer = ZTensorWriter::create(output)?;
    let tensors = reader.list_tensors().clone(); // Iterate over metadata
    for (name, tensor) in tensors {
        if tensor.format != "dense" {
            eprintln!("Skipping non-dense tensor '{}' during compression (sparse compression not yet implemented in CLI)", name);
            continue;
        }
        let data = reader.read_tensor(&name)?;
        writer.add_tensor(
            &name,
            tensor.shape,
            tensor.dtype,
            Encoding::Zstd,
            data,
            ChecksumAlgorithm::None,
        )?;
    }
    writer.finalize()?;
    Ok(())
}

fn decompress_ztensor(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = ZTensorReader::open(input)?;
    let mut writer = ZTensorWriter::create(output)?;
    let tensors = reader.list_tensors().clone();
    for (name, tensor) in tensors {
        if tensor.format != "dense" {
            eprintln!("Skipping non-dense tensor '{}' during decompression (sparse support minimal)", name);
            continue;
        }
        let data = reader.read_tensor(&name)?;
        writer.add_tensor(
            &name,
            tensor.shape,
            tensor.dtype,
            Encoding::Raw,
            data,
            ChecksumAlgorithm::None,
        )?;
    }
    writer.finalize()?;
    Ok(())
}


fn print_tensors_table(tensors: &BTreeMap<String, ztensor::Tensor>) {
    let mut table = Table::new();
    table
        .load_preset(comfy_table::presets::UTF8_HORIZONTAL_ONLY)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table.set_header(vec![
        "#",
        "Name",
        "Shape",
        "DType",
        "Format",
        "Components", // Summarize count?
    ]);
    for (i, (name, tensor)) in tensors.iter().enumerate() {
        let shape_str = if tensor.shape.is_empty() {
            "(scalar)".to_string()
        } else {
            format!(
                "[{}]",
                tensor.shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let dtype_str = format!("{:?}", tensor.dtype);
        let format_str = &tensor.format;
        let components_count = tensor.components.len().to_string();
        
        table.add_row(Row::from(vec![
            Cell::new(i),
            Cell::new(name),
            Cell::new(shape_str),
            Cell::new(dtype_str),
            Cell::new(format_str),
            Cell::new(components_count),
        ]));
    }
    println!("{}", table);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputFormat {
    SafeTensor,
    GGUF,
    Pickle,
}

fn detect_format_from_extension(input: &str) -> Option<InputFormat> {
    let input = input.to_ascii_lowercase();
    if input.ends_with(".safetensor") || input.ends_with(".safetensors") {
        Some(InputFormat::SafeTensor)
    } else if input.ends_with(".gguf") {
        Some(InputFormat::GGUF)
    } else if input.ends_with(".pkl") || input.ends_with(".pickle") {
        Some(InputFormat::Pickle)
    } else {
        None
    }
}

fn detect_format_for_inputs(inputs: &[String], format: &str) -> Option<InputFormat> {
    if format == "auto" {
        let mut detected: Option<InputFormat> = None;
        for input in inputs {
            let this_format = detect_format_from_extension(input);
            if let Some(fmt) = this_format {
                if let Some(prev) = detected {
                    if prev != fmt {
                        return None; // Mixed formats
                    }
                } else {
                    detected = Some(fmt);
                }
            } else {
                return None;
            }
        }
        detected
    } else {
        match format {
            "safetensor" => Some(InputFormat::SafeTensor),
            "gguf" => Some(InputFormat::GGUF),
            "pickle" => Some(InputFormat::Pickle),
            _ => None,
        }
    }
}

fn merge_ztensor_files(
    inputs: &[String],
    output: &str,
    preserve_original: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use std::collections::HashSet;
    let mut writer = ZTensorWriter::create(output)?;
    let mut seen_names = HashSet::new();
    for input in inputs {
        let mut reader = ZTensorReader::open(input)?;
        let tensors = reader.list_tensors().clone();
        for (name, tensor) in tensors {
            if seen_names.contains(&name) {
                return Err(format!("Duplicate tensor name '{}' found in file '{}'. Aborting merge.", name, input).into());
            }
             if tensor.format != "dense" {
                 eprintln!("Skipping non-dense tensor '{}' during merge", name);
                 continue;
            }
            let data = reader.read_tensor(&name)?;
            writer.add_tensor(
                &name,
                tensor.shape,
                tensor.dtype,
                Encoding::Raw, // Default to raw if merged, or could preserve? Reading decompresses.
                data,
                ChecksumAlgorithm::None,
            )?;
            seen_names.insert(name);
        }
        if !preserve_original {
            // Remove the file after processing
            fs::remove_file(input)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn convert_safetensors_to_ztensor(
    inputs: &[String],
    output: &str,
    compress: bool,
    preserve_original: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use std::collections::HashSet;
    let mut writer = ZTensorWriter::create(output)?;
    let mut seen_names = HashSet::new();
    for input in inputs {
        let file = File::open(input)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;
        for (name, tensor) in st.tensors() {
            if seen_names.contains(&name) {
                return Err(format!("Duplicate tensor name '{}' found in file '{}'. Aborting merge.", name, input).into());
            }
            let dtype = safetensor_dtype_to_ztensor(&tensor.dtype())
                .ok_or_else(|| format!("Unsupported dtype: {:?}", tensor.dtype()))?;
            let shape: Vec<u64> = tensor.shape().iter().map(|&d| d as u64).collect();
            let data = tensor.data().to_vec();
            
            writer.add_tensor(
                &name,
                shape,
                dtype,
                if compress { Encoding::Zstd } else { Encoding::Raw },
                data,
                ChecksumAlgorithm::None,
            )?;
            seen_names.insert(name);
        }
        if !preserve_original {
             fs::remove_file(input)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn convert_ggufs_to_ztensor(
    inputs: &[String],
    output: &str,
    compress: bool,
    preserve_original: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use std::collections::HashSet;
    let mut writer = ZTensorWriter::create(output)?;
    let mut seen_names = HashSet::new();
    for input in inputs {
        let mut container = gguf_rs::get_gguf_container(input)?;
        let model = container.decode()?;
        let mut f = File::open(input)?;
        for tensor_info in model.tensors() {
            let ggml_dtype: gguf_rs::GGMLType = tensor_info
                .kind
                .try_into()
                .map_err(|e| format!("Unsupported GGUF dtype: {}", e))?;
            if let Some(dtype) = gguf_dtype_to_ztensor(&ggml_dtype) {
                if seen_names.contains(&tensor_info.name) {
                    return Err(format!("Duplicate tensor name '{}' found in file '{}'. Aborting merge.", tensor_info.name, input).into());
                }
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
                };
                let data_size = numel * type_size;
                let mut data = vec![0u8; data_size as usize];
                f.seek(SeekFrom::Start(tensor_info.offset))?;
                f.read_exact(&mut data)?;
                writer.add_tensor(
                    &tensor_info.name,
                    shape,
                    dtype,
                    if compress { Encoding::Zstd } else { Encoding::Raw },
                    data,
                    ChecksumAlgorithm::None,
                )?;
                seen_names.insert(tensor_info.name.clone());
            } else {
                eprintln!(
                    "Warning: Skipping tensor '{}' with unsupported GGUF dtype: {:?}",
                    tensor_info.name, ggml_dtype
                );
            }
        }
        if !preserve_original {
            fs::remove_file(input)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn convert_pickles_to_ztensor(
    inputs: &[String],
    output: &str,
    compress: bool,
    preserve_original: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use std::collections::HashSet;
    let mut writer = ZTensorWriter::create(output)?;
    let mut seen_names = HashSet::new();
    for input in inputs {
        let file = File::open(input).map_err(|e| format!("Failed to open input file '{}': {}", input, e))?;
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
            if seen_names.contains(&name) {
                return Err(format!("Duplicate tensor name '{}' found in file '{}'. Aborting merge.", name, input).into());
            }
            let cleaned_name = if name.is_empty() {
                "unnamed_tensor".to_string()
            } else {
                name
            };
            writer.add_tensor(
                &cleaned_name,
                shape,
                dtype,
                if compress { Encoding::Zstd } else { Encoding::Raw },
                data,
                ChecksumAlgorithm::None,
            )?;
            seen_names.insert(cleaned_name);
        }
        if !preserve_original {
            fs::remove_file(input)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Some(Commands::Convert {
            inputs,
            output,
            compress,
            format,
            preserve_original,
        }) => {
            if inputs.is_empty() {
                eprintln!("No input files provided for convert.");
                process::exit(1);
            }
            if Path::new(output).exists() {
                eprintln!("Output file '{}' already exists. Please remove it or choose a different name.", output);
                process::exit(1);
            }
            let format = format.to_ascii_lowercase();
            let detected_format = detect_format_for_inputs(inputs, &format);
            match detected_format {
                Some(InputFormat::SafeTensor) => {
                    match convert_safetensors_to_ztensor(inputs, output, *compress, *preserve_original) {
                        Ok(()) => println!("Successfully converted and merged {} safetensor files into {}", inputs.len(), output),
                        Err(e) => {
                            eprintln!("Conversion failed: {}", e);
                            process::exit(1);
                        }
                    }
                }
                Some(InputFormat::GGUF) => {
                    match convert_ggufs_to_ztensor(inputs, output, *compress, *preserve_original) {
                        Ok(()) => println!("Successfully converted and merged {} GGUF files into {}", inputs.len(), output),
                        Err(e) => {
                            eprintln!("Conversion failed: {}", e);
                            process::exit(1);
                        }
                    }
                }
                Some(InputFormat::Pickle) => {
                    match convert_pickles_to_ztensor(inputs, output, *compress, *preserve_original) {
                        Ok(()) => println!("Successfully converted and merged {} pickle files into {}", inputs.len(), output),
                        Err(e) => {
                            eprintln!("Conversion failed: {}", e);
                            process::exit(1);
                        }
                    }
                }
                _ => {
                    eprintln!(
                        "Could not auto-detect input format for the provided files or mixed/unsupported formats. Please specify --format [safetensor|gguf|pickle] and ensure all inputs are of the same type."
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
            let reader = ZTensorReader::open(&file).unwrap_or_else(|e| {
                eprintln!("Error opening file: {}", e);
                process::exit(1);
            });
            let tensors = reader.list_tensors();
            println!("File: {}", file);
            println!("Version: {}", reader.manifest.version);
            println!("Generator: {}", reader.manifest.generator);
            println!("Attributes: {:?}", reader.manifest.attributes);
            println!("Total Tensors: {}", tensors.len());
            println!();

            if tensors.len() < 20 {
                for (name, tensor) in tensors {
                   println!("--- Tensor: {} ---", name);
                   print_tensor_metadata(name, tensor);
                   println!();
                }
            } else {
                 print_tensors_table(tensors);
            }
        }
        Some(Commands::Merge { output, inputs, preserve_original }) => {
            if inputs.is_empty() {
                eprintln!("No input files provided for merge.");
                process::exit(1);
            }
            if Path::new(output).exists() {
                eprintln!("Output file '{}' already exists. Please remove it or choose a different name.", output);
                process::exit(1);
            }
            match merge_ztensor_files(inputs, output, *preserve_original) {
                Ok(()) => println!("Successfully merged {} files into {}", inputs.len(), output),
                Err(e) => {
                    eprintln!("Merge failed: {}", e);
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
