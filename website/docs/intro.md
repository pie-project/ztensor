---
sidebar_position: 1
slug: /
---

# zTensor

zTensor is two things:

1. **A container format** (`.zt`) for tensor data — aligned, verifiable,
   and designed so that the parts of it that must never change are
   separated from the parts that are expected to.
2. **A universal loader** — one API that reads `.safetensors`, `.gguf`,
   `.npz`, PyTorch `.pt`, HDF5, and ONNX by projecting each into the same
   object model, and writes exactly one format: canonical `.zt`.

```rust
use ztensor::{Source, Writer};

// Read anything.
let src = ztensor_compat::open("model.safetensors")?;
for t in src.tensors() {
    println!("{}: {:?} {}", t.name(), t.shape(), t.layout());
}

// Write one thing: a canonical, digest-carrying .zt file.
let mut w = Writer::create("model.zt")?;
w.ingest(&src)?;
w.finish()?;
```

## What a `.zt` file is

![The layout of a .zt file](../static/diagrams/file-layout.svg)

Read the last 40 bytes and you know where the manifest is; read the
manifest and you know where every tensor is.


## Why another format

The formats in use today each gave up something that matters for serving
large models:

- **Alignment.** safetensors packs tensors back to back, so a tensor
  rarely starts on a page boundary. You cannot map, register, or evict one
  tensor without touching its neighbours.
- **Extensibility.** GGUF encodes each quantization scheme into the
  container itself, so every new scheme is a format change.
- **Safety.** `.pt` is pickle: reading it is executing it.
- **Verifiability.** None of them carry per-tensor digests, so a corrupt
  byte surfaces as a wrong answer rather than an error.

### What `.zt` does instead

`.zt` fixes those without inventing a new problem: 64 KiB placement in
canonical files, a vocabulary of versioned profiles instead of built-in
quantization, no code execution anywhere, and an XXH3 digest per tensor
plus one over the manifest.

![In canonical .zt every page holds one tensor; packed back to back, two pages end up holding two tensors each](../static/diagrams/alignment.svg)

The page is the unit the operating system deals in, so it is also the unit
of ownership. A tensor whose pages are its own can be mapped, registered
with a driver, and dropped from the cache without touching anything else.

## The three layers

The format is organized so that the mortal parts can die without taking
the rest with them:

![The three layers: bytes in L0, a manifest record in L1, vocabulary identifiers in L2, and above them a framework that decides what the values mean](../static/diagrams/layers.svg)

A manifest entry says where a tensor *is* by naming a byte range in L0, and
what it *is* by naming a profile in L2 — so the layer that is allowed to
change never touches the layer that is not.

### A name is checked, not interpreted

What zTensor does with an L2 name is check it and hand it over. Deciding
that `zt.quant_group/1` with an `f4_e2m1` payload and `f8_e8m0` scales
dequantizes to bf16 is the framework's business, not the format's. A name
it does not know is refused rather than guessed at, because silently
reinterpreting one is how a format returns a wrong tensor instead of an
error.

See the [format specification](./spec.md) for the normative rules and
[profiles](https://github.com/pie-project/ztensor/tree/main/spec/profiles)
for the registry.

## One model, several files

A part may name a shard, so a manifest can address bytes in files other than
its own. That is all sharding is. A part that names none lives here, which is
why a single-file model never mentions sharding at all — the one file is the
degenerate case, not a special case.

The name is only a label. What the table records is **size and digest**, so
moving or renaming the files cannot break or silently change a model.

![A root manifest naming shards by size and digest; the shards are containers with no manifest of their own](../static/diagrams/sharding.svg)

### Overlays

The same mechanism is how an overlay works: a LoRA stores only its deltas
and points at the base model's blobs, without copying them.

## Performance

*Llama 3.2 1B shapes (~2.8 GB), Linux, i9-13900K, NVMe SSD, median of 5 runs.
[Benchmarks](./benchmarks.md) has the method, the charts and the analysis.*

### Reading

One mmap-backed API across every format, against each format's own library.

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

### Writing

Each format written by its own reference implementation, three workloads at
512 MB: **Large** (few big matrices), **Mixed** (realistic model shapes),
**Small** (many ~10 KB parameters).

| Format | Large | Mixed | Small |
|---|---|---|---|
| **ztensor** | 3.29 GB/s | 3.62 GB/s | 0.80 GB/s |
| safetensors | 5.18 GB/s | **6.27 GB/s** | 2.62 GB/s |
| pickle | 5.91 GB/s | 6.03 GB/s | **2.86 GB/s** |
| npz | 1.10 GB/s | 1.15 GB/s | 0.54 GB/s |
| gguf | 4.78 GB/s | 6.25 GB/s | 1.30 GB/s |
| onnx | 0.29 GB/s | 0.30 GB/s | 0.35 GB/s |
| hdf5 | **6.13 GB/s** | 5.96 GB/s | 0.28 GB/s |

zTensor is not the fastest writer, because it is not writing the same file: a
canonical write hashes every byte, pads to 64 KiB, and shares blobs between
identical tensors. That cost is paid once per artifact rather than on every
load.

## The capability ladder

Different files support different things, and the API says so rather than
pretending otherwise:

| Operation | What it gives you | Where it works |
| --- | --- | --- |
| `bytes()` | decoded bytes, saying whether they were borrowed or copied | every format |
| `map()` | a borrow, or an error | mapped raw parts |
| `locate()` | the exact range of one file, for your own I/O | raw parts |
| `verify()` | a digest check | `.zt` |
| `evict()` | drops these pages without touching a neighbour's | page-exclusive parts |

`caps()` reports which of those will work, per part, and `map()` returns an
error rather than silently falling back to a copy. To get what a foreign
checkpoint cannot offer, convert it — that is what `ingest` is for.

## Getting started

### Install

```bash
cargo add ztensor            # the format
cargo add ztensor-compat     # foreign-format projections
pip install ztensor          # Python bindings
```

### Command line

```bash
zt ls model.gguf                    # inspect anything
zt convert model.gguf model.zt      # canonical, verifiable output
zt verify model.zt --deep           # structure + digests + shards
zt diff a.safetensors b.zt          # compare across formats
```
