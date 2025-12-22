//! zTensor CLI: inspect, convert, and compress tensor files



mod commands;
mod extractors;
mod pickle;
mod utils;

use anyhow::{Result, bail};
use clap::CommandFactory;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::Path;

use commands::{
    compress_ztensor, decompress_ztensor, download_hf, merge_ztensor_files,
    migrate_ztensor, print_tensor_metadata, print_tensors_table, run_conversion,
};
use extractors::{GgufExtractor, PickleExtractor, SafeTensorExtractor};
use ztensor::ZTensorReader;

/// Input format selection with auto-detection
#[derive(ValueEnum, Clone, Debug, Default, PartialEq)]
pub enum FormatArg {
    /// Auto-detect from file extension
    #[default]
    Auto,
    /// SafeTensor format (.safetensor, .safetensors)
    Safetensor,
    /// GGUF format (.gguf)
    Gguf,
    /// Pickle format (.pkl, .pickle)
    Pickle,
}

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
    /// Convert tensor files (SafeTensor, GGUF, Pickle) to zTensor format
    #[command(
        about = "Convert one or more tensor files to a single zTensor file.",
        long_about = "Convert one or more tensor files to a single zTensor file.\n\n\
            Supported input formats: SafeTensor, GGUF, Pickle.\n\n\
            Examples:\n  \
            ztensor convert model.safetensor -o model.zt\n  \
            ztensor convert -f gguf model1.gguf model2.gguf -o model.zt\n  \
            ztensor convert -c --delete-original *.pkl -o model.zt\n"
    )]
    Convert {
        /// Input files (.safetensor, .gguf, .pkl, etc)
        #[arg(required = true)]
        inputs: Vec<String>,

        /// Output .zt file
        #[arg(short = 'o', long, required = true)]
        output: String,

        /// Compress tensor data with zstd
        #[arg(short = 'c', long)]
        compress: bool,

        /// Zstd compression level (1-22). Implies --compress.
        #[arg(short = 'l', long)]
        level: Option<i32>,

        /// Input format (auto-detects from extension by default)
        #[arg(short = 'f', long, value_enum, default_value_t = FormatArg::Auto)]
        format: FormatArg,

        /// Delete original files after successful conversion
        #[arg(long)]
        delete_original: bool,
    },

    /// Compress a zTensor file (raw → zstd)
    #[command(
        about = "Compress an uncompressed zTensor file using zstd encoding.",
        long_about = "Compress a zTensor file that uses raw encoding to zstd encoding.\n\n\
            Example:\n  ztensor compress model_raw.zt -o model_compressed.zt\n"
    )]
    Compress {
        /// Input .zt file (uncompressed)
        input: String,

        /// Output .zt file (compressed)
        #[arg(short = 'o', long, required = true)]
        output: String,

        /// Zstd compression level (1-22, default 3)
        #[arg(short = 'l', long)]
        level: Option<i32>,
    },

    /// Decompress a zTensor file (zstd → raw)
    #[command(
        about = "Decompress a compressed zTensor file to raw encoding.",
        long_about = "Decompress a zTensor file that uses zstd encoding to raw encoding.\n\n\
            Example:\n  ztensor decompress model_compressed.zt -o model_raw.zt\n"
    )]
    Decompress {
        /// Input .zt file (compressed)
        input: String,

        /// Output .zt file (uncompressed)
        #[arg(short = 'o', long, required = true)]
        output: String,
    },

    /// Migrate a v0.1.0 ztensor file to v1.1.0 format
    #[command(
        about = "Migrate a zTensor file from version 0.1.0 to 1.1.0.",
        long_about = "Migrate a legacy zTensor file (v0.1.0) to the current format (v1.1.0).\n\n\
            Example:\n  ztensor migrate old_model.zt -o new_model.zt\n"
    )]
    Migrate {
        /// Input .zt file (v0.1.0)
        input: String,

        /// Output .zt file (v1.1.0)
        #[arg(short = 'o', long, required = true)]
        output: String,

        /// Compress tensor data with zstd
        #[arg(short = 'c', long)]
        compress: bool,

        /// Zstd compression level (1-22). Implies --compress.
        #[arg(short = 'l', long)]
        level: Option<i32>,
    },

    /// Show metadata and stats for a zTensor file
    #[command(
        about = "Display metadata and statistics for a zTensor file.",
        long_about = "Display metadata and statistics for a zTensor file, including tensor names, \
            shapes, data types, encodings, and sizes.\n\n\
            Example:\n  ztensor info model.zt\n"
    )]
    Info {
        /// Path to the .zt file
        file: String,
    },

    /// Merge multiple zTensor files into one
    #[command(
        about = "Merge multiple zTensor files into a single file.",
        long_about = "Merge multiple zTensor files into a single file.\n\n\
            Examples:\n  \
            ztensor merge file1.zt file2.zt -o merged.zt\n  \
            ztensor merge --delete-original *.zt -o combined.zt\n"
    )]
    Merge {
        /// Input .zt files to merge
        #[arg(required = true)]
        inputs: Vec<String>,

        /// Output .zt file (merged)
        #[arg(short = 'o', long, required = true)]
        output: String,

        /// Delete original files after successful merge
        #[arg(long)]
        delete_original: bool,
    },

    /// Download safetensors from HuggingFace and convert to zTensor format
    #[command(
        about = "Download safetensors from a HuggingFace repository and convert to zTensor.",
        long_about = "Download all *.safetensors files from a HuggingFace repository, \
            convert each to .zt format immediately after download, and save to the output directory.\n\n\
            The original safetensors files are not saved to disk (only cached by hf-hub).\n\n\
            Examples:\n  \
            ztensor download-hf mlx-community/Meta-Llama-3-8B-Instruct-8bit\n  \
            ztensor download-hf openai-community/gpt2 -o ./models\n  \
            ztensor download-hf private/model --token hf_xxxxx\n"
    )]
    DownloadHf {
        /// HuggingFace repository ID (e.g., mlx-community/Meta-Llama-3-8B-Instruct-8bit)
        repo: String,

        /// Output directory for .zt files
        #[arg(short = 'o', long, default_value = ".")]
        output_dir: String,

        /// HuggingFace API token for private repositories
        #[arg(long)]
        token: Option<String>,

        /// Repository revision (branch, tag, or commit)
        #[arg(long, default_value = "main")]
        revision: String,

        /// Compress tensor data with zstd
        #[arg(short = 'c', long)]
        compress: bool,

        /// Zstd compression level (1-22). Implies --compress.
        #[arg(short = 'l', long)]
        level: Option<i32>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match &cli.command {
        Some(Commands::Convert {
            inputs,
            output,
            compress,
            level,
            format,
            delete_original,
        }) => {
            if inputs.is_empty() {
                bail!("No input files provided for convert.");
            }
            if Path::new(output).exists() {
                bail!("Output file '{}' already exists. Please remove it or choose a different name.", output);
            }
            
            let detected_format = match format {
                FormatArg::Auto => utils::detect_format_for_inputs_auto(inputs),
                FormatArg::Safetensor => Some(utils::InputFormat::SafeTensor),
                FormatArg::Gguf => Some(utils::InputFormat::GGUF),
                FormatArg::Pickle => Some(utils::InputFormat::Pickle),
            };
            
            // Invert delete_original to get preserve flag
            let preserve = !delete_original;
            
            match detected_format {
                Some(utils::InputFormat::SafeTensor) => {
                    run_conversion(SafeTensorExtractor, inputs, output, *compress, *level, preserve)?;
                    println!("Successfully converted {} safetensor file(s) into {}", inputs.len(), output);
                }
                Some(utils::InputFormat::GGUF) => {
                    run_conversion(GgufExtractor, inputs, output, *compress, *level, preserve)?;
                    println!("Successfully converted {} GGUF file(s) into {}", inputs.len(), output);
                }
                Some(utils::InputFormat::Pickle) => {
                    run_conversion(PickleExtractor, inputs, output, *compress, *level, preserve)?;
                    println!("Successfully converted {} pickle file(s) into {}", inputs.len(), output);
                }
                None => {
                    bail!(
                        "Could not auto-detect input format. Files may have mixed or unsupported extensions.\n\
                        Please specify --format [safetensor|gguf|pickle] and ensure all inputs are the same type."
                    );
                }
            }
        }
        Some(Commands::Compress { input, output, level }) => {
            if Path::new(output).exists() {
                bail!("Output file '{}' already exists. Please remove it or choose a different name.", output);
            }
            compress_ztensor(input, output, *level)?;
            println!("Successfully compressed {} to {}", input, output);
        }
        Some(Commands::Decompress { input, output }) => {
            if Path::new(output).exists() {
                bail!("Output file '{}' already exists. Please remove it or choose a different name.", output);
            }
            decompress_ztensor(input, output)?;
            println!("Successfully decompressed {} to {}", input, output);
        }
        Some(Commands::Migrate { input, output, compress, level }) => {
            if Path::new(output).exists() {
                bail!("Output file '{}' already exists. Please remove it or choose a different name.", output);
            }
            migrate_ztensor(input, output, *compress, *level)?;
            println!("Successfully migrated {} to {}", input, output);
        }
        Some(Commands::Info { file }) => {
            let reader = ZTensorReader::open(file)
                .map_err(|e| anyhow::anyhow!("Failed to open file '{}': {}", file, e))?;
            let objects = reader.list_objects();
            
            println!("File: {}", file);
            println!("Version: {}", reader.manifest.version);
            println!("Attributes: {:?}", reader.manifest.attributes);
            println!("Total Objects: {}", objects.len());
            println!();

            if objects.len() < 20 {
                for (name, obj) in objects {
                   println!("--- Object: {} ---", name);
                   print_tensor_metadata(name, obj);
                   println!();
                }
            } else {
                print_tensors_table(objects);
            }
        }
        Some(Commands::Merge { inputs, output, delete_original }) => {
            if inputs.is_empty() {
                bail!("No input files provided for merge.");
            }
            if Path::new(output).exists() {
                bail!("Output file '{}' already exists. Please remove it or choose a different name.", output);
            }
            let preserve = !delete_original;
            merge_ztensor_files(inputs, output, preserve)?;
            println!("Successfully merged {} files into {}", inputs.len(), output);
        }
        Some(Commands::DownloadHf {
            repo,
            output_dir,
            token,
            revision,
            compress,
            level,
        }) => {
            download_hf(repo, output_dir, token.as_deref(), revision, *compress, *level)?;
        }
        None => {
            Cli::command().print_help()?;
            println!();
        }
    }
    
    Ok(())
}
