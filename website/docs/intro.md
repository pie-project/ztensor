---
sidebar_position: 1
slug: /
---

import { CrossFormatChart } from '@site/src/components/BenchmarkCharts';

# zTensor

One reader for every tensor format, and a format of its own.

## Cross-format reading

zTensor reads `.safetensors`, `.pt`, `.gguf`, `.npz`, `.onnx`, `.h5`, and `.zt` files through a single API. Format detection is automatic. Because it uses memory-mapped I/O, it is often faster than the reference implementations. For `.pt` files, it parses pickle using a restricted VM in Rust that only extracts tensor metadata, so no arbitrary code can execute.

<CrossFormatChart />

| Source format | zTensor | Reference impl. | Speedup |
| :--- | :--- | :--- | :--- |
| .safetensors | 2.27 GB/s | 1.48 GB/s ([`safetensors`](https://github.com/huggingface/safetensors)) | **+53%** |
| .pt | 2.26 GB/s | 1.44 GB/s ([`torch`](https://github.com/pytorch/pytorch)) | **+57%** |
| .npz | 2.35 GB/s | 1.15 GB/s ([`numpy`](https://github.com/numpy/numpy)) | **+104%** |
| .gguf | 2.29 GB/s | 2.34 GB/s ([`gguf`](https://github.com/ggml-org/ggml)) | -2% |
| .onnx | 2.28 GB/s | 0.79 GB/s ([`onnx`](https://github.com/onnx/onnx)) | **+189%** |
| .h5 | 2.41 GB/s | 1.48 GB/s ([`h5py`](https://github.com/h5py/h5py)) | **+63%** |

See [Benchmarks](./benchmarks) for full results.

## Why .zt?

Most tensor formats treat a tensor as a flat array of one dtype. Pickle-based formats (`.pt`, `.bin`) also execute arbitrary code on load. SafeTensors solved the security problem but still can't represent sparse matrices, quantized weight groups, or newer types like FP8 without a spec change. GGUF handles quantization but bakes each scheme into a dtype enum, coupling the format to a single ecosystem.

`.zt` models each tensor as a composite object with typed components, so dense, sparse, and quantized data all fit without extending the format. It also supports zero-copy mmap reads, zstd compression, integrity checksums, and streaming writes, without executing arbitrary code.

| Feature | .zt | .safetensors | .gguf | .pt (pickle) | .npz | .onnx | .h5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-copy read | Yes | Yes | Yes | No | No | No | No |
| Safe (no code execution) | Yes | Yes | Yes | No | Yes | Yes | Yes |
| Streaming / append | Yes | No | No | No | No | No | No |
| Sparse tensors | Yes | No | No | Yes | No | No | No |
| Compression (zstd) | Yes | No | No | No | No | No | No |
| Extensible type system | Yes | No | No | N/A | No | Yes | Yes |
| Parser complexity | Low | Low | Medium | High | Low | High | High |

Read the full [specification](./spec).

## Get started

- **Python** (`pip install ztensor`) — [Python API](python)
- **Rust** (`ztensor` crate) — [Rust API](rust)
- **CLI** (`cargo install ztensor-cli`) — [CLI Reference](cli)
