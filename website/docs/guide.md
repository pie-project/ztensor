---
sidebar_position: 2
---

# Guide

## Reading

`open` sniffs the format and returns a [`Source`] — one type whether it is a
`.zt` file, a sharded model, a foreign checkpoint, or a snapshot spread over
several files.

```rust
let src = ztensor_compat::open("model.gguf")?;

for t in src.tensors() {
    println!("{}: {:?} {}", t.name(), t.shape(), t.layout());
}

let t = src.tensor("blk.0.attn_q.weight")?;
let bytes = t.bytes()?;     // the best available; says which it gave
let view = t.map()?;        // a borrow, or an error
let at = t.locate()?;       // (store, offset, len) — read it yourself
```

### Three ways to get bytes

Three methods because there are three intents. `bytes()` serves the caller who
just wants the data and would rather not write the branch; `map()` serves the
one whose plan is invalid without a borrow, and errors instead of quietly
copying; `locate()` serves the one doing its own I/O — io_uring, cuFile, a
staged host-to-device copy — and wants the address, not the bytes.

### Asking what a part supports

Ask what a part supports before relying on it:

```rust
let caps = src.tensor("blk.0.attn_q.weight")?.caps()?;
if caps.evict {
    // its pages are its own: releasing them cannot disturb a neighbour
}
```

Every `Caps` field is named after the method it gates and is computed by that
method's own precondition, so the report and the behaviour cannot drift apart.

### Metadata without mapping

When only the metadata is wanted — a planner deciding what to load — open
without mapping:

```rust
let src = ztensor::Source::index("model.zt")?;   // header reads, no mapping
let at = src.tensor("blk.0.attn_q.weight")?.locate()?;
```

## Writing

Only `.zt` is written, and canonical form is the default: 64 KiB placement,
digests on every part, byte-identical parts sharing one blob.

```rust
use ztensor::{DType, Writer};

let mut w = Writer::create("model.zt")?;
// canonical form requires ascending name order
w.add("a.weight", [4096u64, 4096], DType::BF16, &weights)?;
w.add("b.bias", [4096u64], DType::F32, &bias)?;
w.finish()?;
```

Two canonical writes of the same tensors produce **byte-identical files**, so a
file's hash is a stable identity for the model.

### Building anything else

`add` is the one-liner over `object`, which builds anything else — several
parts, a layout profile, attributes, or a part streamed a chunk at a time:

```rust
w.object("q")
    .shape([4096u64, 4096])
    .layout("zt.quant_group/1")
    .attr("group_size", 128u64)
    .part("data").dtype(DType::U8).bytes(&payload)
    .part("scales").dtype(DType::U8).logical("f8_e8m0").bytes(&scales)
    .add()?;
```

### Leaving canonical form

For non-distribution files — smaller alignment, compression, sharding — say so:

```rust
let mut w = Writer::options().canonical(false).align(4096).create("scratch.zt")?;
```

Alignment and canonical form are separate questions, and asking for one while
meaning the other is refused rather than obeyed.

### Publishing atomically

To publish where readers are watching, let the writer do the dance:

```rust
let mut w = Writer::publish("model.zt")?;   // writes beside it, renames at the end
```

Nothing appears at the path until `finish`, and a writer dropped without
finishing leaves nothing behind.

## Converting

```rust
let src = ztensor_compat::open("model.pt")?;
let mut w = Writer::create("model.zt")?;
w.ingest(&src)?;
w.finish()?;
```

This is the upgrade path: the foreign file has no digests and arbitrary
alignment; the result has both, and pages of its own. Layouts survive the trip
— a GGUF `q8_0` tensor stays a `gguf.q8_0/1` object with its block bytes
intact, not a dequantized approximation.

## Verifying

```rust
let src = ztensor::Source::open("model.zt")?;
src.tensor("a.weight")?.verify()?;   // digest + logical-type content rules
src.verify_shards()?;                // whole-file digests of every shard
```

`verify` returns `Verified::Digest` when it checked one and
`Verified::NoDigest` when there was none to check; a *mismatch* is
`Err(Reject { rule: Digest, .. })`, because that is a rejected file rather than
an answer.

### The ladder is cheapest-first

Structure and the manifest hash are checked at
open, per-part digests when you ask, whole-shard digests only in
`verify_shards`. Hashing 100 GB on every load would be a tax nobody pays, so it
is opt-in.

## Sharded models and overlays

A multi-file model is one root manifest plus data shards. The root names each
shard, but a name is only a label: identity is `(size, digest)`, so renaming a
model cannot break or change it.

```rust
let model = ztensor::Source::open("model.zt")?;   // positional resolver
```

### Finding the shards

A resolver turns a name into bytes, and can ignore the name entirely:

```rust
// Match on size and digest instead — the one convention that survives a rename.
let model = ztensor::Source::options()
    .resolver(ztensor::DirectoryResolver::scan("checkpoint/")?)
    .open("checkpoint/model.zt")?;
```

### Overlays

Because a part may name a shard, a file can reference another model's blobs.
That is how an overlay works: a LoRA stores only its deltas and points at the
base model's tensors.

```rust
let base = ztensor::Source::open("base.zt")?;
let object = base.manifest().unwrap().object("base.weight")?.clone();

let mut w = ztensor::Writer::options().canonical(false).align(4096).create("lora.zt")?;
w.add_shard("base", &ztensor::shard_identity("base.zt")?)?;
w.link("base.weight", &object, "base")?;
w.add("base.weight.lora_a", [64u64], DType::F32, &delta)?;
w.finish()?;
```

### A set nobody wrote together

A set of files that were *not* written for each other — a sharded safetensors
snapshot — is the other shape, and it is not the same claim:

```rust
let src = ztensor_compat::open_all(&paths)?;   // one name space, nothing verified
```

Nothing binds that set but the caller's list, so nothing pretends to verify it.
What it checks is that the names do not collide.

## Streaming weights

For canonical files, each tensor owns its pages, so they can be released
individually:

```rust
let src = ztensor::Source::open("model.zt")?;
let t = src.tensor("blk.0.attn_q.weight")?;
t.prefetch()?;   // MADV_WILLNEED
// ... upload to the device ...
t.evict()?;      // MADV_DONTNEED, exact range
```

`evict` refuses when a part shares a page with another blob rather than
dropping a neighbour's cache — the check that makes 64 KiB placement worth its
padding.

## Python

```python
import ztensor, torch

with ztensor.open("model.safetensors") as src:
    for t in src:
        print(t.name, t.shape, t.dtype, t.caps.map)

    t = src["layer.weight"]
    w = torch.from_dlpack(t)          # zero-copy; DLPack can say bfloat16
    t.location                        # (path, offset, nbytes) for your own I/O
    t["scales"]                       # parts are tensors too

ztensor.convert("model.safetensors", "model.zt")
ztensor.verify("model.zt", deep=True)
```

The bindings link against no framework: tensors export DLPack and the buffer
protocol, so numpy, torch and jax all read them without this package knowing
any of them exist. `ztensor.numpy` is a safetensors-shaped shim for migrating
existing code.
