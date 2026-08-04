---
sidebar_position: 1
slug: /
---

# zTensor

`.zt` is a container format for tensor data. Tensors sit on page boundaries,
each one carries a digest, and a manifest describes them. The parts of the
format that are frozen are kept separate from the parts expected to grow.

The crate that reads `.zt` also reads `.safetensors`, `.gguf`, `.npz`, PyTorch
`.pt`, HDF5 and ONNX, projecting each into the same object model so that one
piece of code handles all of them. It writes a single format, canonical `.zt`.

```rust
use ztensor::{Source, Writer};

// Any supported format, detected from the file itself.
let src = ztensor_compat::open("model.safetensors")?;
for t in src.tensors() {
    println!("{}: {:?} {}", t.name(), t.shape(), t.layout());
}

// Out: a canonical .zt, aligned and digest-carrying.
let mut w = Writer::create("model.zt")?;
w.ingest(&src)?;
w.finish()?;
```

## File layout

![The layout of a .zt file](../static/diagrams/file-layout.svg)

Eight magic bytes, then the tensor blobs at aligned offsets, then the
manifest, then a 40-byte footer. The footer holds the manifest's offset and
length, so opening a file means seeking to the end, reading 40 bytes, and
reading the manifest. That gives the byte range of every tensor without
touching the weights, whether the file is 100 MB or 100 GB.

## Existing formats

The formats in common use were each built around a different priority, and
serving a large model off disk runs into what they set aside.

safetensors packs tensors back to back, so a tensor rarely starts on a page
boundary and cannot be mapped, registered or evicted without touching its
neighbours. GGUF encodes each quantization scheme into the container, which
makes every new scheme a format change. Reading a `.pt` file executes whatever
code the pickle stream contains. None of them carry per-tensor digests, so a
corrupt byte surfaces as a wrong answer.

### Comparison

| | `.zt` | `.safetensors` | `.gguf` | `.pt` (pickle) | `.npz` | `.onnx` | `.h5` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-copy read | ✓ | ✓ | ✓ | ~² | ~² | | |
| Page-aligned tensors | ✓ | | | | | | |
| Per-tensor digest | ✓ | | | | | | |
| Safe (no code execution) | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ |
| Streaming / append | ✓ | | | | ~³ | | ✓ |
| Sparse tensors | ✓ | | | ✓ | | | |
| Per-tensor compression | ✓ | | | | ✗¹ | | ✓ |
| Extensible types | ✓ | | | N/A | | ✓ | ✓ |
| Byte-reproducible | ✓ | | | | | | |

¹ `.npz` compresses the zip archive, not each tensor. ² Partial: needs the
right alignment, or uncompressed data. ³ Zip append, not through the standard
API.

Blank cells are not a criticism. safetensors was built as a safe replacement
for pickle and does that job well; alignment and digests were not among its
goals. `.zt` was designed later, with those properties as requirements from
the start. [Formats](./formats.md) has the per-format detail.

### Design choices

A canonical `.zt` file places every tensor on a 64 KiB boundary and records an
XXH3 digest for each tensor and for the manifest. Quantization schemes are
named by a registry of versioned profiles rather than built into the
container, so adding one is a registry entry and not a format revision. The
manifest is CBOR against a fixed schema, so reading a file decodes data and
calls nothing.

![In canonical .zt every page holds one tensor; packed back to back, two pages end up holding two tensors each](../static/diagrams/alignment.svg)

Alignment matters because an operating system maps, protects and evicts memory
a page at a time. Two tensors sharing a page cannot be handled separately:
mapping one maps the other, and releasing one either releases its neighbour or
does nothing. Placement costs padding, and
[Benchmarks](./benchmarks.md#alignment-is-a-tradeoff) measures that cost
against the tensor sizes where it starts to matter.

## The three layers

The format separates the parts that are frozen from the parts that are
expected to change.

![The three layers: bytes in L0, a manifest record in L1, vocabulary identifiers in L2, and above them a framework that decides what the values mean](../static/diagrams/layers.svg)

A manifest entry gives a tensor's location as a byte range in L0, and its
meaning as a profile name in L2. Adding a profile to L2 does not change L1 or
L0. What separates the three is how long each is meant to last:

| Layer | Contents | Contract |
| --- | --- | --- |
| L0: Container | Magic, 40-byte footer, aligned blob heap, byte order | Frozen |
| L1: Manifest | Deterministic-CBOR schema | Gated by the footer's version integer |
| L2: Vocabulary | Layouts, logical types, encodings, digests | Expected to change; registry-managed |

### Unregistered names

zTensor checks an L2 name against its registry and passes it through without
acting on it. Working out that `zt.quant_group/1` with an `f4_e2m1` payload
and `f8_e8m0` scales dequantizes to bf16 belongs to the framework.

An unregistered name is an error. Guessing would mean decoding under the wrong
scheme, and the result of that has the right shape and dtype, so nothing
downstream can tell it from the right answer.

See the [format specification](./spec.md) for the normative rules and
[profiles](https://github.com/pie-project/ztensor/tree/main/spec/profiles) for
the registry.

## Sharding

A part in a manifest may carry a `shard` field naming another file, and a part
without one lives in the file whose manifest it is in. Sharding is that field
plus a table at the top of the manifest saying what each name refers to, which
is why a model that fits in one file contains no trace of the mechanism.

The files themselves are not special. Every one of them is an ordinary `.zt`
container with the same magic, the same aligned blobs, and the same 40-byte
footer. A shard differs in one respect: it has no manifest of its own (spec
§7.2), so it makes no claim about what its bytes are or where they start. The
root's manifest makes those claims for all of them.

![Three .zt files with identical anatomy: model.zt, model-001.zt and model-002.zt. Only model.zt has a manifest; in the two shards that slot is empty.](../static/diagrams/sharding.svg)

A table row records a shard's size and digest, not a path. Finding the file is
left to the reader, which can go by name, by content address, or by scanning a
directory for something of that size and digest. Renaming or moving the files
therefore cannot break the model, and a different file under the same name is
caught at open.

### Overlays

An overlay is the same mechanism pointed at a model someone else published. A
LoRA registers the base model as a shard and stores only its own deltas,
addressing the base weights where they already are instead of copying them.

## Performance

*Llama 3.2 1B shapes (~2.8 GB), Linux, i9-13900K, NVMe SSD, median of 5 runs.
[Benchmarks](./benchmarks.md) has the method, the charts and the analysis.*

### Reading

One mmap-backed API across every format, compared against each format's own
library.

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
512 MB: Large (few big matrices), Mixed (realistic model shapes), Small (many
10 KB parameters).

| Format | Large | Mixed | Small |
|---|---|---|---|
| **ztensor** | 3.29 GB/s | 3.62 GB/s | 0.80 GB/s |
| safetensors | 5.18 GB/s | **6.27 GB/s** | 2.62 GB/s |
| pickle | 5.91 GB/s | 6.03 GB/s | **2.86 GB/s** |
| npz | 1.10 GB/s | 1.15 GB/s | 0.54 GB/s |
| gguf | 4.78 GB/s | 6.25 GB/s | 1.30 GB/s |
| onnx | 0.29 GB/s | 0.30 GB/s | 0.35 GB/s |
| hdf5 | **6.13 GB/s** | 5.96 GB/s | 0.28 GB/s |

zTensor is slower here because a canonical write does more: it hashes every
byte, pads to 64 KiB and shares blobs between identical tensors. Those costs
land once, when the artifact is written, and not on the loads afterwards.

## Two identities

Hashing the whole file tells you whether you have this exact artifact. The
answer changes when the same tensors are re-aligned or split across files.
Canonical form exists to pin that down, and it is single-file for the same
reason.

The content digest is computed from the manifest alone and leaves layout out
of the input. The same tensors give the same value whether they sit in one
file or fifty shards, packed at 4 KiB or 64 KiB, raw or compressed. Computing
it costs one manifest read, whatever the model weighs.

```bash
zt id model.zt          # sha256:4811d499...
zt verify model.zt --canonical
```

## Capabilities

A projected `.npz` cannot offer what a canonical `.zt` can, so the API reports
per part what will work:

| Operation | What it gives you | Where it works |
| --- | --- | --- |
| `bytes()` | decoded bytes, saying whether they were borrowed or copied | every format |
| `map()` | a borrow, or an error | mapped raw parts |
| `locate()` | the exact range of one file, for your own I/O | raw parts |
| `verify()` | a digest check | `.zt` |
| `evict()` | drops these pages without touching a neighbour's | page-exclusive parts |

`caps()` answers all five questions at once, before you commit to a plan.
`map()` returns an error where it cannot borrow, so a loader that meant to be
zero-copy finds out at that call. When a foreign checkpoint does not support
what you need, `ingest` converts it to a `.zt` that does.

## Testing

The parsers here read files from wherever a model came from, so the contract
is explicit: hostile input yields an error, never a panic, an unbounded
allocation, or a fabricated tensor. Three things hold them to it. A 76-file
conformance corpus is regenerated from code and diffed in CI. Fuzz targets
cover the container, the CBOR codec and all six foreign parsers. A test file
holds the reproducer behind every hardening fix.
[Formats](./formats.md#testing) has the detail, and
[platform support](./formats.md#platform-support) says which systems this is
run on.

## Install

### Crates

```bash
cargo add ztensor            # the format
cargo add ztensor-compat     # foreign-format projections
cargo install ztensor-cli    # the `zt` command
pip install ztensor          # Python bindings
```

### Command line

```bash
zt ls model.gguf                    # inspect anything
zt convert model.gguf model.zt      # canonical, verifiable output
zt verify model.zt --deep           # structure + digests + shards
zt diff a.safetensors b.zt          # compare across formats
```
