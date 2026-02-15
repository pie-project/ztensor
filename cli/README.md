# ztensor-cli

A command-line tool for inspecting, converting, and compressing tensor files in the [zTensor](https://github.com/pie-project/ztensor) format.

## Installation

```sh
cargo install ztensor-cli
```

## Features

- Display metadata and stats for `.zt` (zTensor) files
- Convert from multiple formats to zTensor (auto-detected from extension)
- Download from HuggingFace and convert to zTensor in one step
- Merge multiple `.zt` files into one
- Compress and decompress zTensor files (zstd encoding)
- Migrate legacy v0.1.0 files to v1.2.0

## Supported Formats

| Extension | Format |
|-----------|--------|
| `.safetensors` | SafeTensors |
| `.gguf` | GGUF |
| `.pkl`, `.pickle`, `.pt`, `.pth`, `.bin` | Pickle |
| `.npz` | NumPy |
| `.onnx` | ONNX |
| `.h5`, `.hdf5` | HDF5 |
| `.zt` | zTensor |

## Usage

```sh
# Inspect metadata
ztensor info model.zt

# Convert (format auto-detected from extension)
ztensor convert model.safetensors -o model.zt

# Convert with compression
ztensor convert model.safetensors -o model.zt -c

# Merge files
ztensor merge part1.zt part2.zt -o merged.zt

# Download from HuggingFace
ztensor download-hf microsoft/resnet-18 -o ./models

# Compress / decompress
ztensor compress raw.zt -o compressed.zt
ztensor decompress compressed.zt -o raw.zt

# Migrate legacy files
ztensor migrate old.zt -o new.zt
```

## Documentation

See the [full CLI reference](https://pie-project.github.io/ztensor/cli) for all commands and options.

## License

MIT
