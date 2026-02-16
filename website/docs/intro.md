---
sidebar_position: 1
slug: /
---

import { CrossFormatChart } from '@site/src/components/BenchmarkCharts';

# zTensor

Unified, zero-copy, and safe I/O for deep learning formats.

## Quick start

```bash
pip install ztensor
```

```python
from ztensor.numpy import load_file, save_file

# Read any format through one API
tensors = load_file("model.safetensors")  # or .pt, .gguf, .npz, .onnx, .h5
save_file(tensors, "model.zt")
```

## Cross-format reading

zTensor reads `.safetensors`, `.pt`, `.gguf`, `.npz`, `.onnx`, `.h5`, and `.zt` files through a single API. Format detection is automatic. In zero-copy mode, it consistently achieves ~2 GB/s across all formats.

<CrossFormatChart />

| Format | zTensor | zTensor (zc off) | Reference impl. |
| :--- | :--- | :--- | :--- |
| .safetensors | **2.19 GB/s** | 1.46 GB/s | 1.33 GB/s ([`safetensors`](https://github.com/huggingface/safetensors)) |
| .pt | **2.04 GB/s** | 1.33 GB/s | 0.89 GB/s ([`torch`](https://github.com/pytorch/pytorch)) |
| .npz | **2.11 GB/s** | 1.41 GB/s | 1.04 GB/s ([`numpy`](https://github.com/numpy/numpy)) |
| .gguf | **2.11 GB/s** | 1.38 GB/s | 1.39 GB/s / 2.15 GB/s† ([`gguf`](https://github.com/ggml-org/ggml)) |
| .onnx | **2.07 GB/s** | 1.29 GB/s | 0.76 GB/s ([`onnx`](https://github.com/onnx/onnx)) |
| .h5 | **1.96 GB/s** | 1.30 GB/s | 1.35 GB/s ([`h5py`](https://github.com/h5py/h5py)) |

*Llama 3.2 1B shapes (~2.8 GB). †GGUF's native reader also supports mmap (2.15 GB/s). See [Benchmarks](./benchmarks) for full results.*

## The .zt format

Existing tensor formats each solve part of the problem, but none solve it cleanly:

- **Pickle-based formats** (`.pt`, `.bin`) execute arbitrary code on load.
- **SafeTensors** is safe but treats every tensor as a flat, dense array of a fixed dtype. New data types cannot be added without a spec change.
- **GGUF** handles quantization but bakes each scheme into the dtype enum, coupling the format to the llama.cpp ecosystem.
- **NumPy `.npz`** has no alignment guarantees (no mmap), no compression beyond zip, and no structured metadata.

`.zt` models each tensor as a composite object with typed components, so dense, sparse, and quantized data all fit without extending the format. No arbitrary code is executed. It also supports zero-copy mmap reads, zstd compression, integrity checksums, and streaming writes.

| Feature | .zt | .safetensors | .gguf | .pt (pickle) | .npz | .onnx | .h5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-copy read | ✓ | ✓ | ✓ | ~² | ~² | | |
| Safe (no code exec) | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ |
| Streaming / append | ✓ | | | | ~³ | | ✓ |
| Sparse tensors | ✓ | | | ✓ | | | |
| Per-tensor compression | ✓ | | | | ✗¹ | | ✓ |
| Extensible types | ✓ | | | N/A | | ✓ | ✓ |

¹ `.npz` uses archive-level zip/deflate, not per-tensor compression.
² Partial support (requires specific alignment or uncompressed data).
³ Zip append support (not standard API).

Read the full [specification](./spec).

## Get started

- **Python** (`pip install ztensor`): [Python API](python)
- **Rust** (`ztensor` crate): [Rust API](rust)
- **CLI** (`cargo install ztensor-cli`): [CLI Reference](cli)
