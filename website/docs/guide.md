---
sidebar_position: 2
---

# Guide

## Reading

`open` detects the format and returns a [`Source`]. The same type comes back
for a `.zt` file, a sharded model, a foreign checkpoint, or a snapshot spread
over several files.

```rust
let src = ztensor_compat::open("model.gguf")?;

for t in src.tensors() {
    println!("{}: {:?} {}", t.name(), t.shape(), t.layout());
}

let t = src.tensor("blk.0.attn_q.weight")?;
let bytes = t.bytes()?;     // the best available; says which it gave
let view = t.map()?;        // a borrow, or an error
let at = t.locate()?;       // (store, offset, len) for your own read
```

### Three ways to get bytes

There are three methods because callers want three different things.
`bytes()` is for code that just wants the data and does not want to write the
branch. `map()` is for code that needs a borrow and should fail if it cannot
have one, so it errors rather than copying. `locate()` is for code doing its
own I/O with io_uring, cuFile or a staged host-to-device copy, which needs
the address rather than the bytes.

### Asking what a part supports

Ask what a part supports before relying on it:

```rust
let caps = src.tensor("blk.0.attn_q.weight")?.caps()?;
if caps.evict {
    // its pages are its own: releasing them cannot disturb a neighbour
}
```

Each `Caps` field is named after the method it gates, and is computed from
that method's own precondition. The report and the behaviour therefore stay
in sync.

### Metadata without mapping

When you only need the metadata, for example in a planner deciding what to
load, open the file without mapping it:

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

Two canonical writes of the same tensors produce byte-identical files, so a
file's hash is a stable identity for the model.

### Building anything else

`add` is a shorthand for `object`, which builds everything else: several
parts, a layout profile, attributes, or a part streamed a chunk at a time.

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

For files that are not for distribution, ask for what you want instead:
smaller alignment, compression, or sharding.

```rust
let mut w = Writer::options().canonical(false).align(4096).create("scratch.zt")?;
```

Alignment and canonical form are separate options. Asking for an alignment
while leaving canonical form on is an error, and the message says how to mean
what you probably meant.

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

This is the upgrade path. The foreign file has no digests and arbitrary
alignment; the result has digests and 64 KiB placement. Layouts are preserved:
a GGUF `q8_0` tensor stays a `gguf.q8_0/1` object with its block bytes intact,
so nothing is dequantized along the way.

## Verifying

```rust
let src = ztensor::Source::open("model.zt")?;
src.tensor("a.weight")?.verify()?;   // digest + logical-type content rules
src.verify_shards()?;                // whole-file digests of every shard
```

`verify` returns `Verified::Digest` when it checked a digest and
`Verified::NoDigest` when the part carries none. A mismatch comes back as
`Err(Reject { rule: Digest, .. })`, since the file has failed a rule.

### What is checked when

Structure and the manifest hash are checked when the file is opened. Per-part
digests are checked when you ask for them, and whole-shard digests only in
`verify_shards`. Hashing 100 GB on every load would be too slow to be
practical, so the expensive checks are opt-in.

## Sharded models and overlays

A multi-file model is one root manifest plus data shards. The root names each
shard, but the name is only a label. A shard's identity is its size and
digest, so renaming the files does not break the model or change it.

```rust
let model = ztensor::Source::open("model.zt")?;   // positional resolver
```

### Finding the shards

A resolver turns a shard name into a file. It can also ignore the name:

```rust
// Match on size and digest instead, which still works after a rename.
let model = ztensor::Source::options()
    .resolver(ztensor::DirectoryResolver::scan("checkpoint/")?)
    .open("checkpoint/model.zt")?;
```

### Overlays

Because a part may name a shard, a file can reference another model's blobs.
An overlay uses this: a LoRA stores only its deltas and points at the base
model's tensors.

```rust
let base = ztensor::Source::open("base.zt")?;
let object = base.manifest().unwrap().object("base.weight")?.clone();

let mut w = ztensor::Writer::options().canonical(false).align(4096).create("lora.zt")?;
w.add_shard("base", &ztensor::shard_identity("base.zt")?)?;
w.link("base.weight", &object, "base")?;
w.add("base.weight.lora_a", [64u64], DType::F32, &delta)?;
w.finish()?;
```

### Files that were not written together

A sharded safetensors snapshot is a different case. The files were not written
for each other, and opening them together makes a weaker claim:

```rust
let src = ztensor_compat::open_all(&paths)?;   // one name space, nothing verified
```

The only thing tying the set together is the caller's list, so there is
nothing to verify it against. What `open_all` does check is that no tensor
name appears in two files.

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

`evict` refuses when a part shares a page with another blob, so it will not
drop a neighbour's cache.

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
