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

## Quick Start

### Python

```python
import numpy as np
from ztensor import Writer, Reader

# Write tensors
with Writer("model.zt") as w:
    w.add_tensor("weights", np.random.randn(1024, 768).astype(np.float32))
    w.add_tensor("bias", np.zeros(768, dtype=np.float32))

# Write with compression
with Writer("model_compressed.zt") as w:
    w.add_tensor("weights", weights, compress=True)  # zstd compression

# Read tensors (auto-decompresses if needed)
with Reader("model.zt") as r:
    weights = r.read_tensor("weights")
    bias = r.read_tensor("bias")
```

### Rust

```rust
use ztensor::{ZTensorWriter, ZTensorReader, DType, Encoding, ChecksumAlgorithm};

// Write
let mut writer = ZTensorWriter::create("model.zt")?;
writer.add_tensor("weights", vec![1024, 768], DType::Float32, 
                  Encoding::Raw, data_bytes, ChecksumAlgorithm::None)?;
writer.finalize()?;

// Write with compression
writer.add_tensor("weights", vec![1024, 768], DType::Float32,
                  Encoding::Zstd, data_bytes, ChecksumAlgorithm::None)?;

// Read (auto-decompresses)
let mut reader = ZTensorReader::open("model.zt")?;
let data = reader.read_tensor("weights")?;
```

## CLI

Inspect zTensor files:

```bash
ztensor info model.zt
ztensor list model.zt
```

## File Format

See [SPEC_NEW.md](SPEC_NEW.md) for the complete specification.

```
+---------------------------+
| Magic: ZTEN1000 (8B)      |
+---------------------------+
| Component blobs (64B aligned) |
+---------------------------+
| CBOR Manifest             |
+---------------------------+
| Manifest size (8B)        |
+---------------------------+
```

## Supported Data Types

| Type | Description |
|------|-------------|
| `float32`, `float16`, `float64`, `bfloat16` | Floating point |
| `int8`, `int16`, `int32`, `int64` | Signed integers |
| `uint8`, `uint16`, `uint32`, `uint64` | Unsigned integers |
| `bool` | Boolean |

## License

MIT
