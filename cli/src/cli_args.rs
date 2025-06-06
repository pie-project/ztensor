// Command-line argument parsing and command enum
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ztensor", version, about = "zTensor CLI: inspect and convert tensor files", author, propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Convert a tensor file (SafeTensor, GGUF, Pickle) to zTensor format
    Convert {
        /// Input file (.safetensor, .gguf, .pkl, etc)
        input: String,
        /// Output .zt file
        output: String,
        /// Compress tensor data with zstd
        #[arg(long)]
        compress: bool,
        /// Input format: auto, safetensor, gguf, pickle
        #[arg(long, value_name = "FORMAT", default_value = "auto")]
        format: String,
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
    /// Show metadata and stats for a zTensor file
    Info {
        /// Path to the .zt file
        file: String,
    },
}
