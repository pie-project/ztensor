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
let src = ztensor_compat::open_any("model.safetensors")?;
for (name, obj) in src.manifest().objects.iter() {
    println!("{name}: {:?} {}", obj.shape, obj.layout.as_str());
}

// Write one thing: a canonical, digest-carrying .zt file.
let mut w = Writer::create("model.zt")?;
w.ingest(src.as_ref())?;
w.finish()?;
```

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

`.zt` fixes those without inventing a new problem: 64 KiB placement in
canonical files, a vocabulary of versioned profiles instead of built-in
quantization, no code execution anywhere, and an XXH3 digest per tensor
plus one over the manifest.

## The three layers

The format is organized so that the mortal parts can die without taking
the rest with them:

| Layer | Contents | Contract |
| --- | --- | --- |
| **L0 — Container** | Magic, 40-byte footer, aligned blob heap, byte order | **Frozen.** Never changes. |
| **L1 — Manifest** | Deterministic-CBOR schema: objects, parts, blob references, shards | Gated by the footer's version integer; evolves rarely. |
| **L2 — Vocabulary** | Layout profiles, logical types, encodings, digest algorithms | **Deliberately mortal.** Namespaced, versioned, registry-managed. |

A reader needs L0 and L1 to be useful at all; it can refuse anything in L2
it does not recognize, and it must — silently reinterpreting unknown
vocabulary is how formats produce wrong tensors instead of errors.

See the [format specification](./spec.md) for the normative rules and
[profiles](https://github.com/pie-project/ztensor/tree/main/spec/profiles)
for the registry.

## The capability ladder

Different files support different things, and the API says so rather than
pretending otherwise:

| Tier | Guarantee | Where you get it |
| --- | --- | --- |
| 0 | Enumerate objects and metadata | every format |
| 1 | Decoded read (owned bytes) | every format |
| 2 | Zero-copy view | mapped sources, raw parts |
| 3 | Tier 2 + page-exclusive + verifiable | canonical `.zt` |

`caps()` reports the truth for each part, and `view()` returns an error
rather than silently falling back to a copy. To move a foreign checkpoint
up to tier 3, convert it — that is what `ingest` is for.

## Getting started

```bash
cargo add ztensor            # the format
cargo add ztensor-compat     # foreign-format projections
pip install ztensor          # Python bindings
```

```bash
zt ls model.gguf                    # inspect anything
zt convert model.gguf model.zt      # canonical, verifiable output
zt verify model.zt --deep           # structure + digests + shards
zt diff a.safetensors b.zt          # compare across formats
```
