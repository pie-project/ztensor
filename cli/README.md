# ztensor-cli

A command-line tool for inspecting, converting, and compressing tensor files in the [zTensor](https://github.com/pie-project/ztensor) format.

## Installation

```sh
cargo install ztensor-cli
```

## Features

- Display metadata and stats for `.zt` (zTensor) files
- Convert `.safetensor`, `.gguf`, and `.pkl` files to zTensor
- Merge multiple `.zt` files into one
- Compress and decompress zTensor files (zstd encoding)

## Supported Formats

| Extension | Format |
|-----------|--------|
| `.safetensor`, `.safetensors` | SafeTensor |
| `.gguf` | GGUF |
| `.pkl`, `.pickle` | Pickle |
| `.zt` | zTensor |

## Usage

### Inspect a zTensor file

```sh
ztensor info model.zt
```

### Convert files to zTensor

```sh
# Auto-detect format from extension
ztensor convert model.safetensors -o model.zt

# Multiple inputs
ztensor convert shard1.safetensors shard2.safetensors -o model.zt

# Explicit format with compression
ztensor convert -f gguf -c model.gguf -o model.zt

# Delete originals after successful conversion
ztensor convert --delete-original *.pkl -o model.zt
```

**Options:**
- `-o, --output` — Output .zt file (required)
- `-f, --format` — Input format: `auto`, `safetensor`, `gguf`, `pickle` (default: `auto`)
- `-c, --compress` — Compress with zstd
- `--delete-original` — Delete input files after conversion

### Merge zTensor files

```sh
ztensor merge file1.zt file2.zt file3.zt -o merged.zt

# Delete originals after merging
ztensor merge --delete-original *.zt -o combined.zt
```

### Compress / Decompress

```sh
# Compress (raw → zstd)
ztensor compress input.zt -o compressed.zt

# Decompress (zstd → raw)
ztensor decompress compressed.zt -o raw.zt
```

## Example Output

```
$ ztensor info model.zt
File: model.zt
Version: 1
Generator: ztensor-cli
Total Tensors: 146

┌───┬──────────────────────────────────────┬──────────────┬──────────┬────────┬────────────┐
│ # │ Name                                 │ Shape        │ DType    │ Format │ Components │
├───┼──────────────────────────────────────┼──────────────┼──────────┼────────┼────────────┤
│ 0 │ model.layers.0.self_attn.k_proj      │ [512, 2048]  │ BFloat16 │ dense  │ 1          │
│ 1 │ model.layers.0.self_attn.q_proj      │ [2048, 2048] │ BFloat16 │ dense  │ 1          │
│ 2 │ model.norm.weight                    │ [2048]       │ BFloat16 │ dense  │ 1          │
└───┴──────────────────────────────────────┴──────────────┴──────────┴────────┴────────────┘
```

## Help

```sh
ztensor --help
ztensor convert --help
```

## License

MIT
