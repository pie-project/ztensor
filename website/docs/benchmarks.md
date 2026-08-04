---
sidebar_position: 5
---

# Benchmarks

*Linux, i9-13900K, 61 GiB RAM, Samsung 980 PRO NVMe, ext4. Median of 5 runs
after 1 warmup; cold reads drop the page cache with
`posix_fadvise(POSIX_FADV_DONTNEED)`. Both zero-copy and copy modes are shown
where applicable. Reproduce with `benchmark/bench.py`; every number below is
its output.*

## Reading

zTensor reads `.safetensors`, `.pt`, `.gguf`, `.npz`, `.onnx`, `.h5` and `.zt`
through one mmap-backed API. The results below load a Llama 3.2 1B-shaped model
(~2.8 GB) from each format, against each format's own library.

![Cross-format read throughput](../static/charts/cross_format_read.svg)

*The table is on the [introduction](./intro.md#reading); the analysis is here.*

### Zero-copy and copy

By default (`copy=False`) zTensor returns
mmap-backed arrays with no memory copy; `copy=True` reads into owned arrays.
The spread between the two columns is the memcpy, and it is most of the cost:
once the bytes have to be copied, every format converges on what memory
bandwidth allows. Formats with real serialization overhead stay slower in
both modes, because that work does not go away: pickle for `.pt`, zip for
`.npz`, protobuf for `.onnx`.

### GGUF

GGUF's own reader beats zTensor on its own files, 2.52 against 2.37 GB/s. It
maps the file and hands back block pointers, the same thing zTensor does with
one more layer of indirection. The columns are close because they are all
measuring the same mmap.

### Safety

For `.pt`, zTensor runs a restricted pickle VM that recognizes only
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

*The table is on the [introduction](./intro.md#writing).*

zTensor is not the fastest writer, because it is not writing the same file.
Canonical form places every tensor on a 64 KiB boundary, computes an XXH3
digest for each one, and shares a blob between byte-identical tensors, so a
canonical write hashes all the bytes and pads between them. safetensors writes
a header and concatenates. The extra work buys the digests and the alignment,
and you pay for it once per artifact instead of on every load.

## Alignment tradeoff {#alignment-is-a-tradeoff}

The padding is per tensor rather than per byte, about 32 KiB on average, so
what it costs depends on how large the tensors are:

| Workload | 4 KiB floor | 64 KiB canonical |
|---|---|---|
| Large (few big matrices) | 1.00× payload, 1.98 GB/s | 1.00× payload, 1.11 GB/s |
| Mixed (realistic shapes) | 1.00× payload, 2.05 GB/s | 1.00× payload, 1.23 GB/s |
| **Small (~10 KB tensors)** | **1.21× payload, 1.32 GB/s** | **6.41× payload, 0.39 GB/s** |

For a transformer checkpoint, where weight matrices run to tens or hundreds
of megabytes, 64 KiB placement costs nothing measurable and gives per-tensor
mapping and eviction on every page size in use (4 KiB x86/ARM, 16 KiB Apple
silicon, 64 KiB ARM64 distributions). For a model made of many tiny tensors it
is very expensive: 51k tensors each rounded up to a page turn 512 MB into
3.4 GB, and every read then pays for the padding as well.

### Known limitation

Blanket alignment is the wrong default for small-tensor models. The fix is to
align
selectively, aligning only tensors large enough that a page of padding does
not matter. Until the spec covers that, use
`Writer::options().canonical(false).align(4096).create(path)`
(`zt convert --align 4096`, `ztensor.numpy.save_file(..., align=4096)`) for
checkpoints of that shape.

### What it buys

- Each tensor can be memory-mapped, registered, and evicted independently.
- `madvise(MADV_DONTNEED)` on one tensor cannot drop a neighbour's pages, so a
  streaming loader can release weights it has finished with.
- The offsets satisfy O_DIRECT and GPUDirect Storage block alignment.

None of that works at arbitrary alignment, so `caps().evict` is true only for
files that have 64 KiB placement. Files written at the 4 KiB floor, and every
foreign format, report what they support.

## The 2.0 rewrite

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

Reads are within a few percent, which is the cost of the catalog indirection
and the `Bytes` enum. **Writes are not reported here because this harness cannot
measure them at that size**: at 1 GiB the run-to-run spread on every write row
exceeded 70%, since what is being timed is the kernel's writeback rather than
the writer. At 512 MiB, where the spread falls to about ±5%, the two trees are
indistinguishable (2.02–2.23 GB/s canonical write on both).

### Benchmark directory

`ZTENSOR_BENCH_DIR` matters for every tree you compare. It
defaults to `target/bench` beside the manifest, so two checkouts can easily
land on different filesystems. Measuring one on tmpfs and one on NVMe
produced an apparent 40% write regression that turned out to be the storage.

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
