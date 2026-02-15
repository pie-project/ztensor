# ztensor
[![Crates.io](https://img.shields.io/crates/v/ztensor.svg)](https://crates.io/crates/ztensor)
[![Docs.rs](https://docs.rs/ztensor/badge.svg)](https://docs.rs/ztensor)
[![PyPI](https://img.shields.io/pypi/v/ztensor.svg)](https://pypi.org/project/ztensor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple, safe tensor serialization format with zero-copy reads, streaming writes, and built-in compression.

## Installation

```bash
pip install ztensor           # Python
cargo add ztensor              # Rust
cargo install ztensor-cli     # CLI
```

## Features

- Zero-copy memory-mapped reads
- Streaming, append-only writes
- Zstd compression and checksums
- Dense, sparse (CSR/COO), and quantized tensors
- Convert from SafeTensors, GGUF, Pickle, NumPy, ONNX, and HDF5

## Documentation

- [Website](https://pie-project.github.io/ztensor/) — guides, API reference, benchmarks
- [Spec](SPEC.md) — file format specification
- [docs.rs](https://docs.rs/ztensor) — Rust API docs

## License

MIT
