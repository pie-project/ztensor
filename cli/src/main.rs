//! zTensor CLI: inspect, convert, and compress tensor files

mod commands;
mod extractors;
mod pickle;
mod utils;

use anyhow::{Result, bail};
use clap::CommandFactory;
use clap::{Parser, Subcommand};
use std::path::Path;

use commands::{
    compress_ztensor, decompress_ztensor, merge_ztensor_files,
    print_tensor_metadata, print_tensors_table, run_conversion,
};
use extractors::{GgufExtractor, PickleExtractor, SafeTensorExtractor};
use utils::{detect_format_for_inputs, InputFormat};
use ztensor::ZTensorReader;

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

fn main() -> Result<()> {
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
                bail!("No input files provided for convert.");
            }
            if Path::new(output).exists() {
                bail!("Output file '{}' already exists. Please remove it or choose a different name.", output);
            }
            
            let format = format.to_ascii_lowercase();
            let detected_format = detect_format_for_inputs(inputs, &format);
            
            match detected_format {
                Some(InputFormat::SafeTensor) => {
                    run_conversion(SafeTensorExtractor, inputs, output, *compress, *preserve_original)?;
                    println!("Successfully converted {} safetensor file(s) into {}", inputs.len(), output);
                }
                Some(InputFormat::GGUF) => {
                    run_conversion(GgufExtractor, inputs, output, *compress, *preserve_original)?;
                    println!("Successfully converted {} GGUF file(s) into {}", inputs.len(), output);
                }
                Some(InputFormat::Pickle) => {
                    run_conversion(PickleExtractor, inputs, output, *compress, *preserve_original)?;
                    println!("Successfully converted {} pickle file(s) into {}", inputs.len(), output);
                }
                None => {
                    bail!(
                        "Could not auto-detect input format for the provided files or mixed/unsupported formats. \
                        Please specify --format [safetensor|gguf|pickle] and ensure all inputs are of the same type."
                    );
                }
            }
        }
        Some(Commands::Compress { input, output }) => {
            compress_ztensor(input, output)?;
            println!("Successfully compressed {} to {}", input, output);
        }
        Some(Commands::Decompress { input, output }) => {
            decompress_ztensor(input, output)?;
            println!("Successfully decompressed {} to {}", input, output);
        }
        Some(Commands::Info { file }) => {
            let reader = ZTensorReader::open(file)
                .map_err(|e| anyhow::anyhow!("Failed to open file '{}': {}", file, e))?;
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
                bail!("No input files provided for merge.");
            }
            if Path::new(output).exists() {
                bail!("Output file '{}' already exists. Please remove it or choose a different name.", output);
            }
            merge_ztensor_files(inputs, output, *preserve_original)?;
            println!("Successfully merged {} files into {}", inputs.len(), output);
        }
        None => {
            Cli::command().print_help()?;
            println!();
        }
    }
    
    Ok(())
}
