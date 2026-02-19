//! zTensor CLI: inspect, convert, and compress tensor files

mod commands;
mod utils;

use anyhow::{bail, Result};
use clap::CommandFactory;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::Path;
use ztensor::TensorReader;

use commands::{
    compress_ztensor, decompress_ztensor, download_hf, merge_ztensor_files, migrate_ztensor,
    print_tensor_metadata, print_tensors_table, run_conversion,
};

/// Checksum algorithm selection
#[derive(ValueEnum, Clone, Debug, Default, PartialEq)]
pub enum ChecksumArg {
    /// No checksum
    #[default]
    None,
    /// CRC-32C checksum
    Crc32c,
    /// SHA-256 checksum
    Sha256,
}

impl ChecksumArg {
    pub fn to_checksum(&self) -> ztensor::Checksum {
        match self {
            Self::None => ztensor::Checksum::None,
            Self::Crc32c => ztensor::Checksum::Crc32c,
            Self::Sha256 => ztensor::Checksum::Sha256,
        }
    }
}

#[derive(Parser)]
#[command(
    name = "ztensor",
    version,
    about = "zTensor CLI: inspect, convert, and compress tensor files",
    long_about = "zTensor CLI is a tool for inspecting, converting, and compressing tensor files.\n\nSupported formats:\n  - SafeTensors (.safetensors)\n  - GGUF (.gguf)\n  - PyTorch/Pickle (.pt, .bin, .pth, .pkl)\n  - NumPy (.npz)\n  - ONNX (.onnx)\n  - HDF5 (.h5, .hdf5)\n  - zTensor (.zt)\n\nYou can convert between formats, compress/decompress zTensor files, and view metadata/stats.",
    author,
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert tensor files to zTensor format
    #[command(
        about = "Convert one or more tensor files to a single zTensor file.",
        long_about = "Convert one or more tensor files to a single zTensor file.\n\n\
            Supported input formats: SafeTensors, GGUF, PyTorch/Pickle, NumPy, ONNX, HDF5.\n\n\
            Examples:\n  \
            ztensor convert model.safetensors -o model.zt\n  \
            ztensor convert model.gguf -o model.zt -c\n  \
            ztensor convert --checksum crc32c *.npz -o model.zt\n  \
            ztensor convert -c --delete-original *.pt -o model.zt\n"
    )]
    Convert {
        /// Input files (.safetensors, .gguf, .pt, .pkl, .npz, .onnx, .h5, etc)
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

        /// Checksum algorithm for data integrity
        #[arg(long, value_enum, default_value_t = ChecksumArg::None)]
        checksum: ChecksumArg,

        /// Delete original files after successful conversion
        #[arg(long)]
        delete_original: bool,
    },

    /// Compress a zTensor file (raw -> zstd)
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

    /// Decompress a zTensor file (zstd -> raw)
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

    /// Migrate a v0.1.0 ztensor file to v1.2.0 format
    #[command(
        about = "Migrate a zTensor file from version 0.1.0 to 1.2.0.",
        long_about = "Migrate a legacy zTensor file (v0.1.0) to the current format (v1.2.0).\n\n\
            Example:\n  ztensor migrate old_model.zt -o new_model.zt\n"
    )]
    Migrate {
        /// Input .zt file (v0.1.0)
        input: String,

        /// Output .zt file (v1.2.0)
        #[arg(short = 'o', long, required = true)]
        output: String,

        /// Compress tensor data with zstd
        #[arg(short = 'c', long)]
        compress: bool,

        /// Zstd compression level (1-22). Implies --compress.
        #[arg(short = 'l', long)]
        level: Option<i32>,
    },

    /// Show metadata and stats for a tensor file
    #[command(
        about = "Display metadata and statistics for a tensor file.",
        long_about = "Display metadata and statistics for a tensor file, including tensor names, \
            shapes, data types, encodings, and sizes.\n\n\
            Supports all formats: .zt, .safetensors, .gguf, .pt, .npz, .onnx, .h5\n\n\
            Example:\n  ztensor info model.zt\n  ztensor info model.safetensors\n"
    )]
    Info {
        /// Path to the tensor file
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

    /// Append tensors from one .zt file into another
    #[command(
        about = "Append tensors from a source .zt file into an existing .zt file.",
        long_about = "Append all tensors from a source .zt file into a target .zt file.\n\
            The target file is modified in-place. Duplicate tensor names cause an error.\n\n\
            Example:\n  ztensor append model.zt --from extra_weights.zt\n"
    )]
    Append {
        /// Target .zt file to append into
        target: String,

        /// Source .zt file containing tensors to append
        #[arg(long = "from", required = true)]
        source: String,
    },

    /// Remove tensors by name from a .zt file
    #[command(
        about = "Remove named tensors from a .zt file.",
        long_about = "Remove one or more tensors by name from a .zt file, writing the result to a new file.\n\n\
            Example:\n  ztensor remove model.zt unused_layer old_bias -o trimmed.zt\n"
    )]
    Remove {
        /// Input .zt file
        input: String,

        /// Tensor names to remove
        #[arg(required = true)]
        names: Vec<String>,

        /// Output .zt file
        #[arg(short = 'o', long, required = true)]
        output: String,
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
            checksum,
            delete_original,
        }) => {
            if inputs.is_empty() {
                bail!("No input files provided for convert.");
            }
            if Path::new(output).exists() {
                bail!(
                    "Output file '{}' already exists. Please remove it or choose a different name.",
                    output
                );
            }

            let compression = commands::get_compression(*compress, *level);
            let preserve = !delete_original;

            run_conversion(
                inputs,
                output,
                compression,
                checksum.to_checksum(),
                preserve,
            )?;
            println!(
                "Successfully converted {} file(s) into {}",
                inputs.len(),
                output
            );
        }
        Some(Commands::Compress {
            input,
            output,
            level,
        }) => {
            if Path::new(output).exists() {
                bail!(
                    "Output file '{}' already exists. Please remove it or choose a different name.",
                    output
                );
            }
            compress_ztensor(input, output, *level)?;
            println!("Successfully compressed {} to {}", input, output);
        }
        Some(Commands::Decompress { input, output }) => {
            if Path::new(output).exists() {
                bail!(
                    "Output file '{}' already exists. Please remove it or choose a different name.",
                    output
                );
            }
            decompress_ztensor(input, output)?;
            println!("Successfully decompressed {} to {}", input, output);
        }
        Some(Commands::Migrate {
            input,
            output,
            compress,
            level,
        }) => {
            if Path::new(output).exists() {
                bail!(
                    "Output file '{}' already exists. Please remove it or choose a different name.",
                    output
                );
            }
            migrate_ztensor(input, output, *compress, *level)?;
            println!("Successfully migrated {} to {}", input, output);
        }
        Some(Commands::Info { file }) => {
            let reader = ztensor::open(file)
                .map_err(|e| anyhow::anyhow!("Failed to open file '{}': {}", file, e))?;
            let objects = reader.tensors();

            println!("File: {}", file);
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
        Some(Commands::Merge {
            inputs,
            output,
            delete_original,
        }) => {
            if inputs.is_empty() {
                bail!("No input files provided for merge.");
            }
            if Path::new(output).exists() {
                bail!(
                    "Output file '{}' already exists. Please remove it or choose a different name.",
                    output
                );
            }
            let preserve = !delete_original;
            merge_ztensor_files(inputs, output, preserve)?;
            println!("Successfully merged {} files into {}", inputs.len(), output);
        }
        Some(Commands::Append { target, source }) => {
            if !Path::new(target).exists() {
                bail!("Target file '{}' does not exist.", target);
            }
            if !Path::new(source).exists() {
                bail!("Source file '{}' does not exist.", source);
            }

            let src_reader = ztensor::Reader::open_any_path(source)
                .map_err(|e| anyhow::anyhow!("Failed to open source '{}': {}", source, e))?;
            let src_objects = src_reader.tensors().clone();

            // Check for duplicate names upfront before modifying the target file.
            // Writer::append truncates the manifest, so we must validate first.
            {
                let target_reader = ztensor::Reader::open(target)
                    .map_err(|e| anyhow::anyhow!("Failed to read target '{}': {}", target, e))?;
                let existing = target_reader.tensors();
                for (name, _) in &src_objects {
                    if existing.contains_key(name.as_str()) {
                        bail!(
                            "Duplicate tensor name '{}' already exists in target '{}'",
                            name,
                            target
                        );
                    }
                }
            }

            let mut writer = ztensor::Writer::append(target)
                .map_err(|e| anyhow::anyhow!("Failed to open target '{}': {}", target, e))?;

            let mut count = 0;
            for (name, obj) in &src_objects {
                let tensor = src_reader
                    .read_object(name)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;

                // Infer compression/checksum from first component
                let first_comp = obj
                    .components
                    .values()
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("No components for '{}'", name))?;

                let compression = match first_comp.encoding {
                    ztensor::Encoding::Zstd => ztensor::writer::Compression::Zstd(3),
                    ztensor::Encoding::Raw => ztensor::writer::Compression::Raw,
                };

                let checksum = match &first_comp.digest {
                    Some(d) if d.starts_with("sha256:") => ztensor::Checksum::Sha256,
                    Some(d) if d.starts_with("crc32c:") => ztensor::Checksum::Crc32c,
                    _ => ztensor::Checksum::None,
                };

                writer
                    .write_object(name, &tensor, compression, checksum)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                count += 1;
            }

            writer.finish().map_err(|e| anyhow::anyhow!("{}", e))?;
            println!(
                "Successfully appended {} tensor(s) from {} into {}",
                count, source, target
            );
        }
        Some(Commands::Remove {
            input,
            names,
            output,
        }) => {
            if Path::new(output).exists() {
                bail!(
                    "Output file '{}' already exists. Please remove it or choose a different name.",
                    output
                );
            }
            let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            ztensor::remove_tensors(input, output, &name_refs)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            println!(
                "Successfully removed {} tensor(s), wrote result to {}",
                names.len(),
                output
            );
        }
        Some(Commands::DownloadHf {
            repo,
            output_dir,
            token,
            revision,
            compress,
            level,
        }) => {
            download_hf(
                repo,
                output_dir,
                token.as_deref(),
                revision,
                *compress,
                *level,
            )?;
        }
        None => {
            Cli::command().print_help()?;
            println!();
        }
    }

    Ok(())
}
