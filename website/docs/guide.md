---
sidebar_position: 2
---

# Guide

## Reading

`open_any` sniffs the format and returns a [`Source`] — the trait every
projection implements.

```rust
use ztensor::Source;

let src = ztensor_compat::open_any("model.gguf")?;

for (name, obj) in &src.manifest().objects {
    println!("{name}: {:?} {}", obj.shape, obj.layout.as_str());
}

let bytes = src.read("blk.0.attn_q.weight", "data")?;   // tier 1: owned
let view  = src.view("blk.0.attn_q.weight", "data")?;   // tier 2: borrowed
```

`view` returns an error when zero-copy is impossible (an encoded part, a
compressed ZIP entry, a tensor in another shard). It never quietly hands
back a copy, because a caller that asked for zero-copy is usually asking
because a copy would be too expensive.

Check what a part supports before relying on it:

```rust
let caps = src.caps("blk.0.attn_q.weight", "data")?;
if caps.tier() >= 3 {
    // canonical .zt: page-exclusive and digest-verifiable
}
```

## Writing

Only `.zt` is written, and canonical form is the default: 64 KiB
placement, digests on every part, byte-identical parts sharing one blob.

```rust
use ztensor::{DType, Writer};

let mut w = Writer::create("model.zt")?;
// canonical form requires ascending name order
w.add_dense("a.weight", &[4096, 4096], DType::BF16, &weights)?;
w.add_dense("b.bias", &[4096], DType::F32, &bias)?;
w.finish()?;
```

Two canonical writes of the same tensors produce **byte-identical files**,
so a file's hash is a stable identity for the model.

For non-distribution files — smaller alignment, compression, sharding —
use `create_with_alignment`:

```rust
let mut w = Writer::create_with_alignment("scratch.zt", 4096)?;
```

## Converting

```rust
let src = ztensor_compat::open_any("model.pt")?;
let mut w = Writer::create("model.zt")?;
w.ingest(src.as_ref())?;   // reads tier 1, writes canonical
w.finish()?;
```

This is the upgrade path: the foreign file has no digests and arbitrary
alignment; the result is tier 3. Layouts survive the trip — a GGUF `q8_0`
tensor stays a `gguf.q8_0/1` object with its block bytes intact, not a
dequantized approximation.

## Verifying

```rust
let model = ztensor::Model::open("model.zt")?;
model.verify("a.weight", "data")?;   // digest + logical-type content rules
model.verify_shards()?;              // whole-file digests of every shard
```

The ladder is cheapest-first: structure and the manifest hash are checked
at open, per-part digests when you ask, whole-shard digests only in
`verify_shards`. Hashing 100 GB on every load would be a tax nobody pays,
so it is opt-in.

## Sharded models and overlays

A multi-file model is one root manifest plus data shards. Shard identity
is `(size, digest)` — **never a file name**, so renaming a model cannot
break or change it.

```rust
let model = ztensor::Model::open("model.zt")?;    // positional resolver
```

Because a blob reference names a shard, a file can reference another
model's blobs. That is how an overlay works: a LoRA stores only its
deltas and points at the base model's tensors.

```rust
let mut w = Writer::create_with_alignment("lora.zt", 4096)?;
let shard = w.add_shard(base_size, &base_digest)?;
w.link_object("base.weight", base.get("base.weight").unwrap(), shard)?;
w.add_dense("base.weight.lora_a", &[64], DType::F32, &delta)?;
w.finish()?;
```

## Streaming weights

For canonical files, each tensor owns its pages, so they can be released
individually:

```rust
let r = ztensor::Reader::open("model.zt")?;
r.prefetch("blk.0.attn_q.weight", "data")?;   // MADV_WILLNEED
// ... upload to the device ...
r.evict("blk.0.attn_q.weight", "data")?;      // MADV_DONTNEED, exact range
```

`evict` refuses when a part shares a page with another blob rather than
dropping a neighbour's cache — the check that makes 64 KiB placement worth
its padding.

## Python

```python
import ztensor, numpy as np

src = ztensor.open("model.safetensors")
info = src.info("layer.weight")            # shape, dtype, layout, nbytes
w = np.frombuffer(src.read("layer.weight"), dtype=np.float32)

ztensor.convert("model.safetensors", "model.zt")
ztensor.verify("model.zt", deep=True)
```

The bindings do not link against numpy: `read()` returns `bytes` and you
interpret them, which keeps the wheel small and the dependency graph
empty.
