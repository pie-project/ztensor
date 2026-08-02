---
sidebar_position: 5
---

# Benchmarks

*Linux, i9-13900K, 61 GiB RAM, Samsung 980 PRO NVMe, ext4. Median of 5 runs
after 1 warmup; cold reads drop the page cache with
`posix_fadvise(POSIX_FADV_DONTNEED)`. Both zero-copy and copy modes are shown
where applicable. Reproduce with `benchmark/bench.py` — every number below is
its output.*

## Cross-format reading

zTensor reads `.safetensors`, `.pt`, `.gguf`, `.npz`, `.onnx`, `.h5` and `.zt`
through one mmap-backed API. The results below load a Llama 3.2 1B-shaped model
(~2.8 GB) from each format, against each format's own library.

![Cross-format read throughput](../static/charts/cross_format_read.svg)

| Source format | zTensor | zTensor (zc off) | Reference impl. |
|---|---|---|---|
| **.zt** | **2.27 GB/s** | 0.96 GB/s | n/a |
| **.safetensors** | **2.47 GB/s** | 1.00 GB/s | 1.57 GB/s / 1.59 GB/s† ([`safetensors`](https://github.com/huggingface/safetensors)) |
| **.pt** | **2.29 GB/s** | 0.83 GB/s | 1.60 GB/s ([`torch`](https://github.com/pytorch/pytorch)) |
| **.npz** | **2.33 GB/s** | 0.94 GB/s | 0.80 GB/s ([`numpy`](https://github.com/numpy/numpy)) |
| **.gguf** | 2.37 GB/s | 0.92 GB/s | 1.57 GB/s / **2.52 GB/s**† ([`gguf`](https://github.com/ggml-org/ggml)) |
| **.onnx** | **2.30 GB/s** | 0.82 GB/s | 0.81 GB/s ([`onnx`](https://github.com/onnx/onnx)) |
| **.h5** | **2.36 GB/s** | 0.95 GB/s | 1.47 GB/s ([`h5py`](https://github.com/h5py/h5py)) |

*ONNX measured at 1 GB (protobuf caps a message at 2 GB). †Native zero-copy
where available (GGUF mmap, SafeTensors `safe_open`).*

**Zero-copy vs. copy.** By default (`copy=False`) zTensor returns
mmap-backed arrays with no memory copy; `copy=True` reads into owned arrays.
The spread between the two columns is the memcpy, and it is most of the cost:
once the bytes have to be copied, every format converges on what memory
bandwidth allows. The formats with real serialization overhead — pickle for
`.pt`, zip for `.npz`, protobuf for `.onnx` — stay slower in both modes because
that work does not go away.

**Where zTensor does not win.** GGUF's own reader is faster on its own files
(2.52 vs 2.37 GB/s): it maps the file and hands back block pointers, which is
the same thing zTensor does with one more layer of indirection. Reading is not
where a container format earns its keep; the columns above are close because
they are all measuring the same mmap.

**Safety.** For `.pt`, zTensor runs a restricted pickle VM that recognizes only
tensor-reconstruction opcodes and extracts metadata without executing anything,
unlike `torch.load()`, which calls `pickle.load()`. It also refuses
non-contiguous tensors rather than reading a transposed tensor's storage as if
it were dense.

---

## Writing

Each format written by its own reference implementation, three workloads at
512 MB: **Large** (few big matrices), **Mixed** (realistic model shapes),
**Small** (many ~10 KB parameters).

![Write throughput by workload](../static/charts/write_throughput.svg)

| Format | Large | Mixed | Small |
|---|---|---|---|
| **ztensor** | 3.29 GB/s | 3.62 GB/s | 0.80 GB/s |
| safetensors | 5.18 GB/s | **6.27 GB/s** | 2.62 GB/s |
| pickle | 5.91 GB/s | 6.03 GB/s | **2.86 GB/s** |
| npz | 1.10 GB/s | 1.15 GB/s | 0.54 GB/s |
| gguf | 4.78 GB/s | 6.25 GB/s | 1.30 GB/s |
| onnx | 0.29 GB/s | 0.30 GB/s | 0.35 GB/s |
| hdf5 | **6.13 GB/s** | 5.96 GB/s | 0.28 GB/s |

zTensor is not the fastest writer, and the reason is that it is not writing the
same file. Canonical form places every tensor on a 64 KiB boundary, computes an
XXH3 digest for each one, and shares a blob between byte-identical tensors — so
a canonical write hashes all the bytes and pads between them. safetensors
writes a header and then concatenates. That is a real cost of a real guarantee,
and it is paid once per artifact rather than on every load.

## Alignment is a tradeoff {#alignment-is-a-tradeoff}

The padding is per *tensor*, not per byte — about 32 KiB on average — so what
it costs depends entirely on how large the tensors are:

| Workload | 4 KiB floor | 64 KiB canonical |
|---|---|---|
| Large (few big matrices) | 1.00× payload, 1.98 GB/s | 1.00× payload, 1.11 GB/s |
| Mixed (realistic shapes) | 1.00× payload, 2.05 GB/s | 1.00× payload, 1.23 GB/s |
| **Small (~10 KB tensors)** | **1.21× payload, 1.32 GB/s** | **6.41× payload, 0.39 GB/s** |

For a transformer checkpoint — weight matrices in the tens to hundreds of
megabytes — 64 KiB placement is free, and it buys per-tensor mapping and
eviction on every page size in use (4 KiB x86/ARM, 16 KiB Apple silicon, 64 KiB
ARM64 distributions). For a model made of many tiny tensors it is ruinous: 51k
tensors each rounded to a page turn 512 MB into 3.4 GB, and every read then
pays for the padding too.

**This is a known limitation of canonical form as specified.** Blanket
alignment is the wrong default for small-tensor models, and the fix is to align
*selectively* — only tensors large enough that a page of padding is noise
against them. Until the spec says so, use
`Writer::options().canonical(false).align(4096).create(path)`
(`zt convert --align 4096`, `ztensor.numpy.save_file(..., align=4096)`) for
checkpoints of that shape.

## What the alignment buys

- Each tensor can be memory-mapped, registered, and evicted independently.
- `madvise(MADV_DONTNEED)` on one tensor cannot drop a neighbour's pages, so a
  streaming loader can release weights it has finished with.
- The offsets satisfy O_DIRECT and GPUDirect Storage block alignment.

None of that is available at arbitrary alignment, which is why `caps().evict`
is true only for files that have it. Files written at the 4 KiB floor, and
every foreign format, report what they actually support instead.

## A note on the 2.0 rewrite

The figures on this page and in the README were first measured before the crate
rewrite. The rewrite changed the API, not the read path's shape, so the question
was whether the new indirection cost anything. It was measured
directly: the in-repo harness (`cargo run --release -p benchmark`) run
alternately against `df9c1c6` (the last commit before the rewrite) and the
2.0 tree, three rounds of nine runs each at 1 GiB.

| Operation | 1.3 | 2.0 | |
| --- | ---: | ---: | ---: |
| read `.zt` zero-copy (warm) | 6.34 GB/s | 6.11 GB/s | −3.6% |
| read `.safetensors` zero-copy (warm) | 6.96 GB/s | 6.84 GB/s | −1.7% |
| read `.zt` zero-copy (cold) | 2.62 GB/s | 2.65 GB/s | +1.1% |
| copy `.zt` into owned buffers | 10.08 GB/s | 9.48 GB/s | −6.0% |

Reads are within a few percent, which is where the catalog indirection and the
`Bytes` enum land. **Writes are not reported here because this harness cannot
measure them at that size**: at 1 GiB the run-to-run spread on every write row
exceeded 70%, since what is being timed is the kernel's writeback rather than
the writer. At 512 MiB, where the spread falls to about ±5%, the two trees are
indistinguishable (2.02–2.23 GB/s canonical write on both).

**If you re-run this, pin `ZTENSOR_BENCH_DIR` for every tree you compare.** It
defaults to `target/bench` beside the manifest, so two checkouts can easily
land on different filesystems — measuring one on tmpfs and one on NVMe
produced an apparent 40% write regression that was entirely the storage.

## Reproducing

```bash
cd benchmark
pip install -r requirements.txt          # torch, numpy, safetensors, h5py, gguf, onnx
python bench.py --dist llama-1b --scenario fastest --runs 5
python bench.py --dist small --size 512 --runs 5
```

The harness writes its files to `benchmark/bench_out`. Put that on real
storage: on tmpfs a "cold cache" read never reaches a device, and the number
would be fiction.
