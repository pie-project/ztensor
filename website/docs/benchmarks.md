---
sidebar_position: 7
---

import {
  CrossFormatChart,
  DistributionComparisonChart, WriteDistributionChart,
  ZstdTradeoffChart, CompressionThroughputChart,
} from '@site/src/components/BenchmarkCharts';

# Benchmarks

*All benchmarks read into NumPy arrays. Linux, NVMe SSD, median of 7 runs after 2 warmup, cold reads via `posix_fadvise(POSIX_FADV_DONTNEED)`. Reproducible with `python benchmark/bench.py`.*

## The ztensor Reader

zTensor reads `.safetensors`, `.pt`, `.gguf`, `.npz`, `.onnx`, and `.h5` files through a single mmap-backed API. For most formats, this is faster than the format's own reader.

<CrossFormatChart />

| Source format | zTensor | Reference impl. | Speedup |
|---|---|---|---|
| **.zt** | **2.50 GB/s** | n/a | n/a |
| **.safetensors** | **2.27 GB/s** | 1.48 GB/s ([`safetensors`](https://github.com/huggingface/safetensors)) | **+53%** |
| **.pt** | **2.26 GB/s** | 1.44 GB/s ([`torch`](https://github.com/pytorch/pytorch)) | **+57%** |
| **.npz** | **2.35 GB/s** | 1.15 GB/s ([`numpy`](https://github.com/numpy/numpy)) | **+104%** |
| **.gguf** | 2.29 GB/s | **2.34 GB/s** ([`gguf`](https://github.com/ggml-org/ggml)) | -2% |
| **.onnx** | **2.28 GB/s** | 0.79 GB/s ([`onnx`](https://github.com/onnx/onnx)) | **+189%** |
| **.h5** | **2.41 GB/s** | 1.48 GB/s ([`h5py`](https://github.com/h5py/h5py)) | **+63%** |

*Llama 3.2 1B parameter shapes (~2.8 GB). ONNX measured at 1 GB (protobuf 2 GB limit).*

Why the gap? The safetensors and pickle readers copy data during deserialization. NumPy's `np.load()` decompresses and copies. ONNX deserializes protobuf into arrays. h5py copies data out of HDF5 datasets on read. zTensor bypasses all of this by memory-mapping the raw data regions directly, regardless of the container format. GGUF's own reader also uses mmap, so the two are nearly equivalent.

For `.pt` files, zTensor uses a minimal pickle VM implemented in Rust that only recognizes tensor reconstruction opcodes (`_rebuild_tensor_v2/v3/v4`) and extracts metadata (dtype, shape, storage offset) without executing arbitrary Python code. This is safe by design, unlike `torch.load()`, which invokes Python's `pickle.load()` and can execute arbitrary code embedded in the file.

---

## The .zt Format

The following benchmarks compare `.zt` against other formats. Each format is read and written through its own reference implementation (e.g., `safetensors.numpy.load_file` for SafeTensors, `numpy.load` for NPZ, `torch.load` for pickle).

### Read Throughput

Three workloads at 512MB: **Large** (few big matrices), **Mixed** (realistic model shapes: one large embedding, several medium weights, many small biases), **Small** (many ~10KB parameters).

<DistributionComparisonChart />

| Format | Large | Mixed | Small |
|---|---|---|---|
| **ztensor** | 2.14 GB/s | **2.31 GB/s** | **1.91 GB/s** |
| safetensors | 1.28 GB/s | 1.35 GB/s | 1.35 GB/s |
| pickle | 1.43 GB/s | 1.43 GB/s | 1.57 GB/s |
| npz | 1.12 GB/s | 1.12 GB/s | 0.22 GB/s |
| gguf | **2.34 GB/s** | 2.29 GB/s | 0.20 GB/s |
| onnx | 0.78 GB/s | 0.78 GB/s | 0.67 GB/s |
| hdf5 | 1.33 GB/s | 1.40 GB/s | 0.16 GB/s |

zTensor and GGUF trade the lead on large tensors, but GGUF drops to 0.20 GB/s on small tensors due to per-tensor mmap overhead. zTensor leads on mixed and small workloads. Pickle and safetensors are competitive on small tensors thanks to eager in-memory deserialization, while HDF5, GGUF, and NPZ pay heavy per-tensor I/O overhead on small workloads.

### Write Throughput

For large and mixed workloads, zTensor, GGUF, pickle, and HDF5 all write at near-memcpy speed (3.6-3.9 GB/s). SafeTensors is notably slower. With many small tensors, per-tensor overhead reduces throughput across all formats.

<WriteDistributionChart />

| Format | Large | Mixed | Small |
|---|---|---|---|
| **ztensor** | 3.60 GB/s | 3.65 GB/s | 1.43 GB/s |
| safetensors | 1.72 GB/s | 1.75 GB/s | 1.30 GB/s |
| pickle | 3.58 GB/s | 3.65 GB/s | **1.76 GB/s** |
| npz | 2.39 GB/s | 2.38 GB/s | 0.50 GB/s |
| **gguf** | **3.81 GB/s** | **3.90 GB/s** | 1.01 GB/s |
| onnx | 0.29 GB/s | 0.29 GB/s | 0.34 GB/s |
| hdf5 | 3.68 GB/s | 3.67 GB/s | 0.27 GB/s |

### Compression

`.zt` supports optional zstd compression per component. Compression effectiveness varies dramatically by workload: structured data like pruned or quantized weights can shrink by 50-75%.

<ZstdTradeoffChart />

| Workload | Description | Compressed size | Reduction |
|---|---|---|---|
| Dense fp32 | Random float32 weights | 92% | 8% |
| Quantized int8 | 4-bit values in int8 storage | 52% | **48%** |
| Pruned 80% | Float32 with 80% zero weights | 27% | **73%** |
| Ternary | {-1, 0, 1} quantized weights | 25% | **75%** |

*All ratios at zstd level 3, the recommended default.*

Read and write throughput, with and without zstd:

<CompressionThroughputChart />

| Workload | Read | Read zstd-3 | Write | Write zstd-3 |
|---|---|---|---|---|
| Dense fp32 | 2.31 GB/s | 0.45 GB/s | 3.65 GB/s | 0.72 GB/s |
| Quantized int8 | 2.31 GB/s | 0.73 GB/s | 3.65 GB/s | 0.24 GB/s |
| Pruned 80% | 2.31 GB/s | 0.59 GB/s | 3.65 GB/s | 0.39 GB/s |
| Ternary | 2.31 GB/s | 0.90 GB/s | 3.65 GB/s | 0.45 GB/s |

Compression trades throughput for disk savings. Compressed read speed is higher for more compressible data, since there are fewer bytes to load from disk. Random float data is nearly incompressible (92%), but real model weights have far more structure. Pruned models (common after magnitude pruning) and low-bit quantized weights compress dramatically because their byte patterns are highly redundant: long runs of zeros for sparse weights, or few unique byte values for quantized data.

