# ztensor-cli

A command-line tool for inspecting, converting, and compressing tensor files in the [zTensor](../README.md) format.

## Features
- Display metadata and stats for `.zt` (zTensor) files
- Convert `.safetensor`, `.gguf`, and `.pkl`/`.pickle` files to `.zt` (zTensor) files
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

### Convert SafeTensor, GGUF, or Pickle to zTensor

```sh
ztensor convert <input> <output.zt>
# Auto-detects format from extension

ztensor convert --format safetensor model.safetensor model.zt
ztensor convert --format gguf model.gguf model.zt --compress
ztensor convert --format pickle model.pkl model.zt
```

### Compress or Decompress zTensor files

```sh
ztensor compress <input_raw.zt> <output_compressed.zt>
ztensor decompress <input_compressed.zt> <output_raw.zt>
```

## Help

Run `ztensor --help` or any subcommand with `--help` for detailed usage and options.

## License
MIT
