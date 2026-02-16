---
sidebar_position: 5
---

# CLI Reference

The `ztensor` CLI inspects, converts, and manipulates tensor files.

## Installation

```bash
cargo install ztensor-cli
```

## Commands

### `convert` - Convert formats

Converts one or more tensor files to a single `.zt` file, detecting the input format by extension.

**Supported input formats:** SafeTensors (`.safetensors`), GGUF (`.gguf`), PyTorch/Pickle (`.pt`, `.bin`, `.pth`, `.pkl`), NumPy (`.npz`), ONNX (`.onnx`), HDF5 (`.h5`, `.hdf5`)

```bash
# Basic conversion
ztensor convert model.safetensors -o model.zt

# With compression
ztensor convert model.gguf -o model.zt -c

# Specific compression level (1-22)
ztensor convert model.safetensors -o model.zt -l 10

# With checksum (none, crc32c, sha256)
ztensor convert model.npz -o model.zt --checksum crc32c

# Multiple input files
ztensor convert part1.safetensors part2.safetensors -o model.zt

# Delete originals after conversion
ztensor convert --delete-original *.safetensors -o model.zt
```

### `info` - Inspect metadata

Prints tensor names, shapes, dtypes, and file properties for any supported format.

```bash
ztensor info model.zt
ztensor info model.safetensors
ztensor info model.gguf
ztensor info weights.npz
```

### `compress` - Compress files

Compresses an existing raw zTensor file with zstd.

```bash
# Default level (3)
ztensor compress raw.zt -o compressed.zt

# Specific level (1-22)
ztensor compress raw.zt -o compressed.zt -l 19
```

### `decompress` - Decompress files

```bash
ztensor decompress compressed.zt -o raw.zt
```

### `merge` - Merge files

Combines multiple zTensor files into one.

```bash
ztensor merge part1.zt part2.zt -o merged.zt
```

### `migrate` - Migrate legacy files

Converts legacy v0.1.0 files to the current v1.2.0 format.

```bash
ztensor migrate old_model.zt -o new_model.zt
```

### `download-hf` - Download from HuggingFace

Downloads safetensors from HuggingFace Hub and converts to zTensor.

```bash
ztensor download-hf microsoft/resnet-18 -o ./models
ztensor download-hf openai-community/gpt2 -o ./models -c
ztensor download-hf openai-community/gpt2 -o ./models -c -l 10
ztensor download-hf private/model -o ./models --token hf_xxxxx
```
