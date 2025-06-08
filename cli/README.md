# ztensor-cli

A command-line tool for inspecting, converting, and compressing tensor files in the [zTensor](../README.md) format.

## Installation

You can install `ztensor-cli` from cargo:

```sh
cargo install ztensor-cli
```

## Features
- Display metadata and stats for `.zt` (zTensor) files
- Convert one or more `.safetensor`, `.gguf`, and `.pkl`/`.pickle` files to a single `.zt` (zTensor) file
- Merge multiple `.zt` files into one
- Compress and decompress zTensor files (zstd encoding)

## Supported Formats
- SafeTensor (`.safetensor`, `.safetensors`)
- GGUF (`.gguf`)
- Pickle (`.pkl`, `.pickle`)
- zTensor (`.zt`)

## Usage

### Show zTensor file info

```sh
ztensor info <file.zt>
```

### Convert one or more SafeTensor, GGUF, or Pickle files to zTensor

```sh
ztensor convert <input1> [<input2> ...] <output.zt>
# Auto-detects format from extension

ztensor convert --format safetensor model1.safetensor model2.safetensor model.zt
ztensor convert --format gguf model1.gguf model2.gguf model.zt --compress
ztensor convert --format pickle model1.pkl model2.pkl model.zt

# By default, input files are preserved. To delete them after conversion:
ztensor convert --format safetensor model1.safetensor model2.safetensor model.zt --preserve-original false
```

### Merge multiple zTensor files

```sh
ztensor merge merged.zt file1.zt file2.zt file3.zt
# To delete the originals after merging:
ztensor merge --preserve-original false merged.zt file1.zt file2.zt
```

### Compress or Decompress zTensor files

```sh
ztensor compress <input_raw.zt> <output_compressed.zt>
ztensor decompress <input_compressed.zt> <output_raw.zt>
```

## Example `ztensor info` output

```sh  
zTensor file: test.zt
Number of tensors: 146
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 #     Name                                 Shape            DType      Encoding   Layout   Offset       Size        On-disk Size
══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 0     model.layers.14.self_attn.k_proj.w   [512, 2048]      BFloat16   Zstd       Dense    64           2.10 MB     1.64 MB
       eight
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 1     model.layers.11.self_attn.q_proj.w   [2048, 2048]     BFloat16   Zstd       Dense    1644800      8.39 MB     6.58 MB
       eight
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 2     model.layers.8.self_attn.k_proj.we   [512, 2048]      BFloat16   Zstd       Dense    8220224      2.10 MB     1.64 MB
       ight
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 3     model.norm.weight                    [2048]           BFloat16   Zstd       Dense    9863488      4.10 kB     2.19 kB
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 4     model.layers.12.self_attn.o_proj.w   [2048, 2048]     BFloat16   Zstd       Dense    9865728      8.39 MB     6.56 MB
       eight
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 5     model.layers.12.self_attn.q_proj.w   [2048, 2048]     BFloat16   Zstd       Dense    16422272     8.39 MB     6.58 MB
       eight
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 6     model.layers.0.post_attention_laye   [2048]           BFloat16   Zstd       Dense    23006720     4.10 kB     1.93 kB
       rnorm.weight
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

## Help

Run `ztensor --help` or any subcommand with `--help` for detailed usage and options.

## License
MIT
