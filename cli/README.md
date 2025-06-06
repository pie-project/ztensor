# ztensor-cli

A command-line tool for inspecting and converting tensor files in the [zTensor](../README.md) format.

## Features
- Display metadata and stats for `.zt` (zTensor) files
- Convert `.safetensor` files to `.zt` (zTensor) files

## Usage

### Show zTensor file info

```sh
ztensor <file.zt>
```

### Convert SafeTensor to zTensor

```sh
ztensor convert <input.safetensor> <output.zt>
```

## License
MIT
