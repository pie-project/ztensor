---
sidebar_position: 5
---

# Benchmarks

All numbers below come from `cargo run --release -p benchmark`, which is in
the repository — re-run it and you should get the same shape of result on
comparable hardware. Nothing here is hand-picked: the harness prints the
table you see.

**What is measured.** A synthetic checkpoint whose layout mirrors a
transformer (a couple of large embedding matrices, then per-layer weights
and norms), written and read through the public API. Every read traverses
**every byte** — a benchmark that only resolves pointers, or samples one
byte per page, measures address arithmetic rather than I/O.

**Machine.** Intel Core i9-13900K, 61 GiB RAM, Samsung 980 PRO NVMe, Linux,
ext4. Median of 9 runs, 1 GiB of tensor data, 32 MiB layer weights
(typical of a transformer's projection matrices).

## Throughput

| Operation | Throughput | Median time |
| --- | --- | --- |
| Write `.zt` (canonical, 64 KiB placement) | 1.66 GB/s | 646 ms |
| Write `.zt` (4 KiB floor) | 1.58 GB/s | 680 ms |
| Write `.safetensors` | 1.35 GB/s | 798 ms |
| Read `.zt` zero-copy, full traversal (warm cache) | 6.62 GB/s | 162 ms |
| Read `.safetensors` zero-copy, full traversal (warm cache) | 6.72 GB/s | 160 ms |
| Copy `.zt` into owned buffers (warm cache, memcpy) | 9.61 GB/s | 112 ms |
| Read `.zt` zero-copy, full traversal (cold cache) | 2.38 GB/s | 452 ms |
| Verify every digest (XXH3) | 13.99 GB/s | 77 ms |
| Convert `.safetensors` → canonical `.zt` | 0.88 GB/s | 1214 ms |

Reading is where the honest answer is *"the same"*: `.zt` and
`.safetensors` are both memory-mapped byte ranges, so a warm traversal is
bounded by memory bandwidth and the two land within noise of each other
(6.6 vs 6.7 GB/s). Any claim that one format reads dramatically faster than
the other, at equal alignment and equal work, should be treated with
suspicion — there is no mechanism for it.

The cold-cache row is the one that reflects the device (2.38 GB/s here);
the harness omits it entirely if dropping the page cache had no measurable
effect, rather than publishing a number that came out of RAM.

The copy row is listed for completeness but is **not comparable** to the
traversal rows above it: it measures `memcpy`, while the traversals sum
every byte.

## Open latency

Time to learn what is in the file — the cost a loader pays before it can
plan anything:

| Format | 50 tensors | 1538 tensors |
| --- | --- | --- |
| `.zt` | 0.04 ms | 0.95 ms |
| `.safetensors` | 0.03 ms | 0.68 ms |

Both are a single metadata read plus a parse. `.zt` is marginally slower
because opening also **validates**: bounds, alignment, blob non-overlap,
size equations, and the manifest hash. That work is the point — a
safetensors reader that skipped its own header checks would be faster
still, and wrong.

## File size

The 64 KiB canonical placement costs padding — how much depends entirely
on how large the tensors are, since the cost is per tensor, not per byte:

| Model shape | `.zt` canonical (64 KiB) | `.zt` 4 KiB floor | `.safetensors` |
| --- | --- | --- | --- |
| 50 × 32 MiB tensors (transformer-like) | **+0.15%** | +0.01% | +0.00% |
| 1538 × 1 MiB tensors (worst case) | **+4.68%** | +0.27% | +0.01% |

Average padding is half the alignment, i.e. ~32 KiB per tensor. For real
checkpoints — whose weight matrices are tens to hundreds of MiB — that is
a rounding error. It only becomes visible on models made of many tiny
tensors, and for those a writer can drop to the 4 KiB floor
(`Writer::create_with_alignment`, `zt convert --align 4096`) and give up
per-tensor mapping on 16 KiB and 64 KiB page systems.

## What the alignment buys

The padding is not decoration. At 64 KiB every tensor starts on a page
boundary on **every** platform in use (4 KiB x86/ARM, 16 KiB Apple
Silicon, 64 KiB ARM64 distributions), which means:

- each tensor can be memory-mapped, registered, and evicted independently;
- `madvise(MADV_DONTNEED)` on one tensor cannot drop a neighbour's pages,
  so a streaming loader can release weights it has finished with;
- the offsets satisfy O_DIRECT and GPUDirect Storage block alignment.

None of that is available at arbitrary alignment, which is why the
capability ladder reports tier 3 only for files that have it. `.zt` files
written at the 4 KiB floor, and every foreign format, report what they
actually support instead.

## Reproducing

```bash
# defaults: 512 MiB of data, 1 MiB tensors, 5 runs
cargo run --release -p benchmark

# the configuration used above
cargo run --release -p benchmark -- --size-mb 1024 --tensor-mb 32 --runs 9

# put the files somewhere else (must be real storage, not tmpfs)
ZTENSOR_BENCH_DIR=/mnt/nvme/bench cargo run --release -p benchmark
```

The harness refuses to fake a cold-cache measurement: if the file lives on
tmpfs or the cache drop is otherwise a no-op, it prints a note and omits
the row.
