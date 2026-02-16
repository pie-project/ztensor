---
sidebar_position: 7
---

import {
  CrossFormatChart,
  DistributionComparisonChart, WriteDistributionChart,
  ZstdTradeoffChart, CompressionThroughputChart,
} from '@site/src/components/BenchmarkCharts';

# Benchmarks

*Linux, NVMe SSD, median of 5 runs after 2 warmup, cold reads via `posix_fadvise(POSIX_FADV_DONTNEED)`. Both zero-copy and copy modes are shown where applicable.*

## Cross-format reading

zTensor reads `.safetensors`, `.pt`, `.gguf`, `.npz`, `.onnx`, `.h5`, and `.zt` through a single mmap-backed API. The results below measure throughput when loading a Llama 3.2 1B-shaped model (~2.8 GB) from each format, compared against each format's native library.

<CrossFormatChart />

| Source format | zTensor | zTensor (zc off) | Reference impl. |
|---|---|---|---|
| **.zt** | **2.19 GB/s** | 1.37 GB/s | n/a |
| **.safetensors** | **2.19 GB/s** | 1.46 GB/s | 1.33 GB/s / 1.35 GB/s† ([`safetensors`](https://github.com/huggingface/safetensors)) |
| **.pt** | **2.04 GB/s** | 1.33 GB/s | 0.89 GB/s ([`torch`](https://github.com/pytorch/pytorch)) |
| **.npz** | **2.11 GB/s** | 1.41 GB/s | 1.04 GB/s ([`numpy`](https://github.com/numpy/numpy)) |
| **.gguf** | **2.11 GB/s** | 1.38 GB/s | 1.39 GB/s / 2.15 GB/s† ([`gguf`](https://github.com/ggml-org/ggml)) |
| **.onnx** | **2.07 GB/s** | 1.29 GB/s | 0.76 GB/s ([`onnx`](https://github.com/onnx/onnx)) |
| **.h5** | **1.96 GB/s** | 1.30 GB/s | 1.35 GB/s ([`h5py`](https://github.com/h5py/h5py)) |

*ONNX measured at 1 GB (protobuf 2 GB limit). †Native zero-copy where available (GGUF mmap views, SafeTensors `safe_open`).*

**Zero-copy vs. copy.** By default (`copy=False`), zTensor returns mmap-backed arrays with no memory copy. Setting `copy=True` reads into owned arrays. Some reference implementations also support zero-copy (GGUF mmap, SafeTensors `safe_open`); their numbers are shown with a dagger (†). Formats with serialization overhead (pickle for `.pt`, zip for `.npz`, protobuf for `.onnx`) are slower in both modes. For formats that also use mmap internally, copy-mode throughput converges because both implementations perform the same mmap-then-copy sequence.

**Safety.** For `.pt` files, zTensor uses a restricted pickle VM in Rust that only recognizes tensor reconstruction opcodes and extracts metadata without executing arbitrary code, unlike `torch.load()`, which invokes `pickle.load()`.

---

## Format comparison

The benchmarks below compare `.zt` against other formats, where each format uses its own reference implementation.

**Read throughput.** Three workloads at 512 MB: **Large** (few big matrices), **Mixed** (realistic model shapes), **Small** (many ~10 KB parameters).

<DistributionComparisonChart />

| Format | Large | Mixed | Small |
|---|---|---|---|
| ztensor | 2.08 GB/s | 2.02 GB/s | **1.76 GB/s** |
| ztensor (zc off) | 1.25 GB/s | 1.31 GB/s | 1.46 GB/s |
| safetensors | 1.23 GB/s | 1.32 GB/s | 1.35 GB/s |
| pickle | 1.25 GB/s | 1.36 GB/s | 1.40 GB/s |
| npz | 1.05 GB/s | 1.06 GB/s | 0.22 GB/s |
| gguf | **2.32 GB/s** | **2.31 GB/s** | 0.21 GB/s |
| gguf (zc off) | 1.40 GB/s | 1.40 GB/s | 0.20 GB/s |
| onnx | 0.73 GB/s | 0.75 GB/s | 0.65 GB/s |
| hdf5 | 1.28 GB/s | 1.33 GB/s | 0.16 GB/s |

With copy enabled, all mmap-based formats converge to similar throughput since the bottleneck is the memory copy itself. In zero-copy mode, ztensor maintains ~2 GB/s across all workloads. GGUF's native mmap is fast on large tensors (2.32 GB/s) but has high per-tensor overhead on small tensors (0.21 GB/s); ztensor avoids this overhead, sustaining 1.76 GB/s even with many small parameters.

**Write throughput.** For large and mixed workloads, ztensor, GGUF, pickle, and HDF5 all write at near-memcpy speed (3.6-3.9 GB/s). SafeTensors is notably slower (~1.7 GB/s). With many small tensors, per-tensor overhead reduces throughput across all formats.

<WriteDistributionChart />

| Format | Large | Mixed | Small |
|---|---|---|---|
| **ztensor** | 3.62 GB/s | 3.65 GB/s | 1.42 GB/s |
| safetensors | 1.72 GB/s | 1.77 GB/s | 1.48 GB/s |
| pickle | 3.62 GB/s | 3.68 GB/s | **2.00 GB/s** |
| npz | 2.40 GB/s | 2.40 GB/s | 0.51 GB/s |
| **gguf** | **3.85 GB/s** | **3.86 GB/s** | 1.06 GB/s |
| onnx | 0.28 GB/s | 0.29 GB/s | 0.32 GB/s |
| hdf5 | 3.67 GB/s | 3.69 GB/s | 0.27 GB/s |

**Compression.** `.zt` supports optional per-component zstd compression. Effectiveness varies by workload: random float32 weights are nearly incompressible (8% reduction), but structured data compresses dramatically. Pruned weights (73%) and ternary quantization (75%) compress well because their byte patterns are highly redundant.

<ZstdTradeoffChart />

| Workload | Description | Compressed size | Reduction |
|---|---|---|---|
| Dense fp32 | Random float32 weights | 92% | 8% |
| Quantized int8 | 4-bit values in int8 storage | 52% | **48%** |
| Pruned 80% | Float32 with 80% zero weights | 27% | **73%** |
| Ternary | {-1, 0, 1} quantized weights | 25% | **75%** |

*Zstd level 3, the recommended default.*

**Compression throughput.** Compression trades throughput for disk savings. More compressible data reads faster because less I/O is needed.

<CompressionThroughputChart />

| Workload | Read | Read zstd-3 | Write | Write zstd-3 |
|---|---|---|---|---|
| Dense fp32 | 1.31 GB/s | 0.45 GB/s | 3.65 GB/s | 0.72 GB/s |
| Quantized int8 | 1.31 GB/s | 0.73 GB/s | 3.65 GB/s | 0.24 GB/s |
| Pruned 80% | 1.31 GB/s | 0.59 GB/s | 3.65 GB/s | 0.39 GB/s |
| Ternary | 1.31 GB/s | 0.90 GB/s | 3.65 GB/s | 0.45 GB/s |

---

## Reproducing

All benchmarks can be reproduced using the scripts in `benchmark/`:

```bash
pip install ztensor safetensors torch numpy gguf onnx h5py

# Cross-format reading (Llama 3.2 1B shapes)
python benchmark/bench.py run --dist llama-1b --runs 5 --warmup 2

# Format comparison (512 MB, three workloads)
python benchmark/bench.py run --size 512 --dist large --runs 5 --warmup 2
python benchmark/bench.py run --size 512 --dist mixed --runs 5 --warmup 2
python benchmark/bench.py run --size 512 --dist small --runs 5 --warmup 2

# Full sweep (all sizes, distributions, scenarios)
python benchmark/bench.py sweep --runs 5 --warmup 2
```
