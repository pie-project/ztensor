# ztensor-cli

A command-line tool for inspecting, converting, and compressing tensor files in the [zTensor](../README.md) format.

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

## Help

Run `ztensor --help` or any subcommand with `--help` for detailed usage and options.

## License
MIT
