# zTensor

A fast, memory-efficient tensor serialization format optimized for machine learning workloads.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Zero-copy reads** via memory mapping (mmap)
- **64-byte alignment** for SIMD/AVX-512 compatibility
- **Sparse tensor support** (CSR and COO formats)
- **Optional zstd compression**
- **Integrity verification** with CRC32C checksums
- **Cross-platform** — Little-endian, portable format
- **PyTorch & NumPy Interoperability**

## Installation

### Python

```bash
pip install ztensor
```

### Rust

```toml
[dependencies]
ztensor = "0.1"
```

## Quick Start: Python

### Basic Usage with NumPy

```python
import numpy as np
from ztensor import Writer, Reader

# Write tensors
with Writer("model.zt") as w:
    w.add_tensor("weights", np.random.randn(1024, 768).astype(np.float32))
    w.add_tensor("bias", np.zeros(768, dtype=np.float32))

# Read tensors (zero-copy where possible)
with Reader("model.zt") as r:
    # Returns a numpy-like view
    weights = r.read_tensor("weights")
    print(f"Weights shape: {weights.shape}, dtype: {weights.dtype}")
```

### PyTorch Integration

```python
import torch
from ztensor import Writer, Reader

# Write PyTorch tensors directly
t = torch.randn(10, 10)
with Writer("torch_model.zt") as w:
    w.add_tensor("embedding", t)

# Read back as PyTorch tensors
with Reader("torch_model.zt") as r:
    # 'to="torch"' returns a torch.Tensor sharing memory with the file (if mmap)
    embedding = r.read_tensor("embedding", to="torch")
    print(embedding.size())
```

### Sparse Tensors

Supports **CSR** (Compressed Sparse Row) and **COO** (Coordinate) formats.

```python
import scipy.sparse
from ztensor import Writer, Reader

csr = scipy.sparse.csr_matrix([[1, 0], [0, 2]], dtype=np.float32)

with Writer("sparse.zt") as w:
    # Add CSR tensor
    w.add_sparse_csr("my_csr", csr.data, csr.indices, csr.indptr, csr.shape)

with Reader("sparse.zt") as r:
    # Read back as scipy.sparse.csr_matrix
    matrix = r.read_tensor("my_csr", to="numpy")
```

### Compression

Use Zstandard (zstd) compression to reduce file size.

```python
with Writer("compressed.zt") as w:
    w.add_tensor("big_data", data, compress=True)
```

## Quick Start: Rust

### Basic Usage

```rust
use ztensor::{ZTensorWriter, ZTensorReader, DType, Encoding, ChecksumAlgorithm};

// Write
let mut writer = ZTensorWriter::create("model.zt")?;
writer.add_tensor("weights", vec![1024, 768], DType::Float32, 
                  Encoding::Raw, data_bytes, ChecksumAlgorithm::None)?;
writer.finalize()?;

// Read
let mut reader = ZTensorReader::open("model.zt")?;
// Read as specific type (automatically handles endianness)
let weights: Vec<f32> = reader.read_tensor_as("weights")?;
```

### Sparse Tensors

```rust
// Write CSR
writer.add_csr_tensor(
    "sparse_data",
    vec![100, 100],      // shape
    DType::Float32,
    values_bytes,        // standard LE bytes
    indices,             // Vec<u64>
    indptr,              // Vec<u64>
    Encoding::Raw,
    ChecksumAlgorithm::None
)?;

// Read CSR
let csr = reader.read_csr_tensor::<f32>("sparse_data")?;
println!("Values: {:?}", csr.values);
```

### Compression

```rust
// Write with compression
writer.add_tensor(
    "compressed_data",
    vec![512, 512],
    DType::Float32,
    Encoding::Zstd, // Use zstd encoding
    data_bytes,
    ChecksumAlgorithm::Crc32c // Optional checksum
)?;

// Read (auto-decompresses)
let data: Vec<f32> = reader.read_tensor_as("compressed_data")?;
```

## CLI

The `ztensor` CLI tool allows you to inspect and manipulate zTensor files.

### Inspect Metadata
Print tensor names, shapes, and properties.
```bash
ztensor info model.zt
```

### Convert Other Formats
Convert SafeTensors, GGUF, or Pickle files to zTensor.
```bash
# Auto-detect format
ztensor convert model.safetensors model.zt

# Explicit format with compression
ztensor convert --format gguf --compress llama.gguf llama.zt
```

### Compression Tools
```bash
# Compress an existing raw file
ztensor compress raw.zt compressed.zt

# Decompress a file
ztensor decompress compressed.zt raw.zt
```

### Merge Files
Combine multiple zTensor files into one.
```bash
ztensor merge -o merged.zt part1.zt part2.zt
```

## Supported Data Types

| Type | Description |
|------|-------------|
| `float32`, `float16`, `bfloat16`, `float64` | Floating point |
| `int8`, `int16`, `int32`, `int64` | Signed integers |
| `uint8`, `uint16`, `uint32`, `uint64` | Unsigned integers |
| `bool` | Boolean |

## File Format

See [SPEC.md](SPEC.md) for the complete specification.

## License

MIT
