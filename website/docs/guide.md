---
sidebar_position: 2
---

# Guide

## Where things live

The crate's modules are the spec's layers, so a section of the specification
and the code that implements it have the same address:

| Module | What is in it |
| --- | --- |
| `ztensor::format` | **L0 + L1**, frozen: the magic, the footer, the alignment floor, the manifest schema, its CBOR mapping, and the rules that decide conformance and canonical form. Nothing here opens a file. |
| `ztensor::vocab` | **L2**, open: layouts, logical types and encodings, which another crate can extend. |
| `ztensor::read` | Opening `.zt` and getting at bytes. |
| `ztensor::write` | Producing `.zt`. |
| `ztensor::provide` | For a crate projecting a *foreign* format into a `Source`. Reading a checkpoint needs nothing from it. |

The names a consumer actually uses are re-exported at the crate root, and only
those — `Source`, `Tensor`, `Part`, `Writer`, `DType` and about a dozen more.
The format constants, the shard resolvers, the free functions that take a path
instead of a source (`read::shard_identity`, `read::manifest_of`,
`read::canonical_violations`) and the projection machinery keep their module
paths, so that what you see first is what you are likely to want.

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
let data = t.data()?;       // the part that holds the bytes
let bytes = data.bytes()?;  // the best available; says which it gave
let view = data.map()?;     // a borrow, or an error
let at = data.locate()?;    // (store, offset, len) for your own read
```

### Bytes belong to a part

A dense tensor keeps all of its bytes in one part, `"data"`; a quantized one
spreads them over that part and its scales. So `data()` is a step rather than
a shorthand — which part is being addressed is exactly what a caller must not
lose track of, and `t.parts()` lists the rest.

`verify()` is the one byte-level operation that stays on the tensor, because
it covers *every* part: a quantized tensor whose scales had rotted must not
pass because its payload did.

### Three ways to get bytes

There are three methods because callers want three different things.
`bytes()` is for code that just wants the data and does not want to write the
branch; it hands back a `Cow`, borrowed when the file allowed it. `map()` is
for code that needs a borrow and should fail if it cannot have one, so it
errors rather than copying. `locate()` is for code doing its own I/O with
io_uring, cuFile or a staged host-to-device copy, which needs the address
rather than the bytes.

### Asking what a part supports

Ask what a part supports before relying on it:

```rust
let caps = src.tensor("blk.0.attn_q.weight")?.data()?.caps();
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
let src = ztensor::Source::options().map(false).open("model.zt")?;
let at = src.tensor("blk.0.attn_q.weight")?.data()?.locate()?;
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
canonical file's hash is a stable identity for that artifact. For an identity
that survives being re-aligned or split, see the content digest below.

### Building anything else

`add` is a shorthand for `object`, which builds everything else: several
parts, a layout profile, or attributes.

```rust
w.object("q", |o| {
    o.shape([4096u64, 4096])
        .layout("zt.quant_group/1")
        .attr("group_size", 128u64)
        .part("data", |p| p.dtype(DType::U8).bytes(&payload))
        .part("scales", |p| p.dtype(DType::U8).logical("f8_e8m0").bytes(&scales))
})?;
```

Each part is described inside its own scope. That is what makes `.dtype()`
before there is a part to apply it to a compile error rather than something
found out when the object is added. The cost is that a description borrows its
bytes, so a payload built inline — `.bytes(&encode(x))` — has to be bound to a
local first.

### Leaving canonical form

For files that are not for distribution, turn it off to insert in any order,
encode parts, or reference other files:

```rust
let mut w = Writer::options().canonical(false).create("scratch.zt")?;
```

Placement is not part of what you give up: a non-canonical writer still puts
blobs on 64 KiB boundaries, because that is what buys per-tensor page
exclusivity and a sharded model wants it as much as any other. Ask for
something else when the model is many small tensors, where a page of padding
each is most of the file:

```rust
let mut w = Writer::options().canonical(false).align(4096).create("tiny.zt")?;
```

Asking for an alignment while leaving canonical form on is an error, and the
message says how to mean what you probably meant.

### Adding to a finished file

`append` adds objects and shards to a `.zt` that already exists, without
rewriting the blobs in it:

```rust
let mut w = ztensor::Writer::append("model.zt")?;
w.add("extra.weight", [1024u64], DType::F32, &data)?;
w.finish()?;
```

Everything new goes past the old end of the file, so the cost is the size of
what you add. Adding a 4 KiB tensor to a 512 MiB file takes about 55 µs,
against 303 ms to rewrite the file. The alignment the file already uses is
carried forward, so a 64 KiB file stays one.

This is not atomic. From the first byte written until `finish` puts a footer
at the new end, the footer is not at EOF and no reader will open the file. The
original bytes are all still there, so a crashed append is undone by
truncating back to the old length. Use `publish` when the file is worth more
than the rewrite.

### Publishing atomically

To publish where readers are watching, let the writer do the dance:

```rust
let mut w = Writer::publish("model.zt")?;   // writes beside it, renames at the end
```

Nothing appears at the path until `finish`, and a writer dropped without
finishing leaves nothing behind.

### Is it canonical?

Canonical form is the recommended distribution format, and a file carries no
mark saying it is one. It does not need to: all six rules of the spec are
decidable from the bytes.

```bash
zt verify model.zt --canonical
```

```text
model.zt: ok. 3 part(s) digest-verified, 0 without digests
  canonical form: no
    rule 2: "t"/"data" is at offset 4096, which is not a multiple of 65536
```

A non-canonical file is still fully conforming; this answers a different
question from "is it valid".

### What model is this?

A file's hash identifies the *file*. Two files holding the same tensors have
different hashes if they were aligned or split differently. The content digest
identifies the *model*:

```bash
zt id model.zt
```

It is computed from the manifest alone, so it costs nothing on a 100 GB
checkpoint, and it is the same whether the model is one file or fifty.

```rust
let digest = ztensor::read::manifest_of("model.zt")?
    .unwrap()
    .content_digest(ztensor::DigestAlgorithm::Sha256)?;
```

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

A multi-file model is one root manifest plus its shards. The root names each
shard, but the name is only a label. A shard's identity is its size and
digest, so renaming the files does not break the model or change it.

```rust
let model = ztensor::Source::open("model.zt")?;   // positional resolver
```

### Writing the shards

A shard is an ordinary `.zt`. Write one, then ask it for its identity:

```rust
let mut w = Writer::create("model-00001.zt")?;
w.add("blk.0.attn_q.weight", [4096u64, 4096], DType::BF16, &weights)?;
w.finish()?;
```

The root records that identity and points its tensors at the shard:

```rust
let id = ztensor::read::shard_identity("model-00001.zt", DigestAlgorithm::Sha256)?;

let mut root = Writer::options().canonical(false).create("model.zt")?;
root.add_shard("00001", &id)?;
for (name, object) in &ztensor::read::manifest_of("model-00001.zt")?.unwrap().objects {
    root.link(name, object, "00001")?;
}
root.finish()?;
```

Linking through a shard that was never registered is an error, so the two
cannot drift apart.

The format also allows a shard with no manifest of its own, holding nothing
but bytes (spec §7.2). zTensor reads those, because other tools write them,
but it does not produce them and you should not want one: a shard without a
manifest cannot say which of its bytes are occupied, so nothing can prove a
tensor has its pages to itself, and `evict` is refused. It carries no
per-part digests either, so there is nothing to verify. A shard that is a
normal `.zt` keeps both.

### Finding the shards

A resolver turns a shard name into a file. It can also ignore the name:

```rust
// Match on size and digest instead, which still works after a rename.
let model = ztensor::Source::options()
    .resolver(ztensor::read::DirectoryResolver::scan("checkpoint/")?)
    .open("checkpoint/model.zt")?;
```

### Overlays

Because a part may name a shard, a file can reference another model's blobs.
An overlay uses this: a LoRA stores only its deltas and points at the base
model's tensors.

```rust
let base = ztensor::Source::open("base.zt")?;
let object = base.provenance().as_root().unwrap().object("base.weight")?.clone();

let mut w = ztensor::Writer::options().canonical(false).align(4096).create("lora.zt")?;
w.add_shard("base", &ztensor::read::shard_identity("base.zt", DigestAlgorithm::Xxh3)?)?;
w.link("base.weight", &object, "base")?;
w.add("base.weight.lora_a", [64u64], DType::F32, &delta)?;
w.finish()?;
```

`link` reads the byte range out of the shard's own manifest. A shard that
carries none — a data shard, spec §7.2 — has no manifest to read, so state the
range yourself:

```rust
w.object("t", |o| {
    o.shape([8192u64]).part("data", |p| {
        p.dtype(DType::U8)
            .digest(format!("xxh3:{hex}"))
            .external("00001", offset..offset + length)
    })
})?;
```

`digest` is only for parts like this one. Everything the writer writes it
hashes as it goes, and a digest supplied for those could only agree with the
computed one or be wrong.

### Signing a sharded model

A root records each shard's size and digest, so a root whose digests are
`sha256` commits to every shard byte. One signature over the root then covers
the whole model.

```rust
let id = ztensor::read::shard_identity("model-00001.zt", DigestAlgorithm::Sha256)?;
w.add_shard("00001", &id)?;
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

### Asking which of the three you have

The difference between a root, a data shard and a projection is what can be
verified, so it is one answer rather than two half-questions:

```rust
match src.provenance() {
    Provenance::Root(manifest) => { /* the file states its own structure */ }
    Provenance::DataShard      => { /* bytes only; it claims nothing */ }
    Provenance::Projection     => { /* whoever opened it made the claim */ }
}
```

`src.provenance().as_root()` is the shorthand for code that already knows it
opened a `.zt` root and only wants the manifest.

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
drop a neighbour's cache. It is present on every platform and refused where
there is no way to drop page cache, so `if caps.evict { t.evict()? }` compiles
everywhere.

### Writing a tensor that will not fit in memory

`stream` declares an object by part lengths and hands back a `Sink`, which is
a token rather than a borrow of the writer — that is what lets a producer
driven from outside (one copying weights off a device, say) hold both in one
structure.

```rust
let mut sink = w.stream("w", |o| {
    o.shape([n]).part("data", |p| p.dtype(DType::F16).length(nbytes))
})?;

for chunk in chunks { sink.write(&mut w, chunk)?; }
sink.close(&mut w)?;
```

When the bytes come from somewhere that already speaks `io::Write`, borrow the
pair together instead, and the compiler is what stops the writer being used for
anything else while the object is open:

```rust
std::io::copy(&mut reader, &mut sink.attach(&mut w))?;
```

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
