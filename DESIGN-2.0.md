# zTensor 2.0 — API redesign

The file format does not change. `spec/ztensor-v2-spec.md` (Draft 2), the
container, the CBOR manifest, digests, canonical form: all identical. Every
`.zt` file written by 1.x is read by 2.0 and vice versa. What changes is the
crate surface.

## Why

Three things were conflated.

**1. What a file says vs. what a consumer asks.** `Manifest` is the L1
structure literally written on disk — `blob: [shard, offset, length]`, where
`shard` indexes *that file's* shard table. `Source::manifest()` also used it as
the lookup index a consumer queries. Because one type served both, a set of N
self-describing files could not be a `Source` (merging manifests would rewrite
blob references against a shard table nobody wrote), so there were three reader
types — `Reader`, `Model`, `Composite` — with the same five methods and no
common trait. Splitting the two roles dissolves all three into one.

**2. Asking for a capability vs. receiving one.** `caps()` answered, `view()`
and `read()` served, and every caller wrote the `if caps.zero_copy { view }
else { read }` bridge by hand — including this library, three times over
(`verify`, the numpy layer, and every downstream planner).

**3. Vocabulary declared open, implemented closed.** The spec calls L2
"registry-managed"; the code had a `match` over built-ins with no way for a
downstream crate to register a layout, an encoding, or a logical type.

And one rung was missing. The ladder ran enumerate → copy → map → page, but the
thing a GPU loader actually wants is *the address*: tell me where the bytes are
and I will do the I/O myself (io_uring, cuFile/GDS, staged pinned H2D). That
was only reachable by reaching into `part.blob` and reconstructing which file
the shard index meant.

`tier()` is gone. It flattened four independent facts onto one ordinal that did
not match the operations it claimed to gate — `evict()` requires page
exclusivity and *not* a digest, yet tier 3 demanded both, so a part that could
be evicted reported tier 2. Nothing in the library ever branched on it.

## The surface

Five concepts: `Source`, `Tensor`, `Caps`, `Bytes`, `Vocabulary`.

### Reading

```rust
let src = Source::open("model.safetensors")?;   // any format
let src = Source::open("model.zt")?;            // follows the shard table if there is one
let src = Source::open_all(&paths)?;            // N files, one name space
let src = Source::index("model.zt")?;           // metadata + addresses, nothing mapped

for t in src.tensors() {
    println!("{} {:?} {}", t.name(), t.shape(), t.dtype());
}

let t = src.tensor("layer.0.mlp.w")?;
t.shape(); t.dtype(); t.logical(); t.layout(); t.attributes();
```

Three ways to get bytes, one per intent:

```rust
t.bytes()?     // -> Bytes<'_>   best available; says which it gave (Deref<[u8]>)
t.map()?       // -> &[u8]       borrowed or error — never a hidden copy
t.locate()?    // -> Location    { store, offset, len } — I will read it myself
```

Parts are addressed the same way; the tensor methods are sugar for `"data"`:

```rust
t.part("scales")?.map()?;
t.parts();                       // names
```

### Caps — one field per operation, computed by that operation's own predicate

```rust
pub struct Caps {
    pub map: bool,        // map() will succeed
    pub locate: bool,     // locate() gives the decoded bytes' exact range
    pub evict: bool,      // evict() will succeed (page-exclusive)
    pub verify: bool,     // verify() will check a digest rather than report none
    pub alignment: u64,   // largest power of two dividing the offset (a fact, not an operation)
}
```

The field and the method share a name and share the predicate function, so they
cannot drift. `caps.evict` implies `caps.map` implies `caps.locate`, but that is
a consequence, not an ordering anyone has to memorize.

```rust
pub enum Bytes<'a> { Mapped(&'a [u8]), Owned(Vec<u8>) }   // Deref<Target = [u8]>
pub enum Verified { Digest, NoDigest }                    // a mismatch is Err(Reject{Digest})
```

### The two layers

```
schema::{Manifest, Object, Part, BlobRef, Shard}   what a .zt file literally says
Catalog { name -> Entry }                          the resolved, process-local index
```

A projection of a foreign format produces a `Catalog`, never a `Manifest` — it
never had one. `Source::manifest()` returns `Some` only when the source is a
single `.zt` root, because only then is there a manifest that something actually
wrote.

A part's payload is one of three honest shapes:

```rust
pub enum Payload {
    At(Location),                                          // raw bytes, addressable
    Encoded { at: Location, encoding: String, decoded_len: u64 },
    Opaque  { store: StoreId, key: u64, decoded_len: u64 }, // only the projection can produce these
}
```

`At` is `locate`-able and `map`-able; `Encoded` and `Opaque` are neither, and
say so rather than degrading.

### Vocabulary as a value

```rust
let vocab = Vocabulary::standard()
    .with_layout(MyQuantGroup)     // impl vocab::Layout
    .with_encoding(MyCodec)        // impl vocab::Encoding
    .with_logical(Fp6E3M2);        // impl vocab::LogicalType

let src = Source::options().vocabulary(&vocab).open(path)?;
let w   = Writer::options().vocabulary(&vocab).create(out)?;
```

Unknown vocabulary is still refused, never guessed. The difference is that a
downstream crate can now make it known.

### Writing

One entry point per shape, and one underlying path:

```rust
let mut w = Writer::create("out.zt")?;                             // canonical
let mut w = Writer::options().canonical(false).align(4096).create(p)?;

w.add("layer.weight", [4096, 4096], DType::BF16, &bytes)?;         // sugar

w.object("q").shape([4096, 4096]).layout("zt.quant_group/1").attr("group", 32u64)
 .part("data").dtype(DType::U8).logical("f4_e2m1").bytes(&blocks)
 .part("scales").dtype(DType::U8).logical("f8_e8m0").bytes(&scales)
 .add()?;

let mut sink = w.object("big").shape([n]).part("data").dtype(DType::BF16).stream(nbytes)?;
sink.write(&mut w, chunk)?;    // the ticket writes into that writer (an FFI producer
sink.close(&mut w)?;           // holds both in one struct, so neither can borrow the other)

w.ingest(&src)?;               // any Source -> canonical .zt
w.publish()?;                  // temp file, fsync, rename. finish() stays for the plain case.
```

`add_dense`, `add_object`, `add_external_object`, `link_object` and
`stream_object` collapse into `object()`; `add()` is the one-liner over it.

### Python

The same five concepts, plus one export instead of a module per framework:

```python
with ztensor.open("model.safetensors") as src:   # or open([p1, p2, p3])
    for t in src:
        print(t.name, t.shape, t.dtype, t.caps.map)

    t = src["model.layers.0.mlp.w"]
    t.location                       # (path, offset, nbytes) for direct I/O
    torch.from_dlpack(t)             # zero-copy; dlpack carries bf16/fp8 that numpy cannot
    np.from_dlpack(t)
    memoryview(t)
    t["scales"]                      # parts are tensors too
```

`ztensor.numpy.load_file` / `save_file` stay, documented as what they are: a
safetensors-shaped shim for migration, not the API.

## Module map

| Module | Holds |
| --- | --- |
| `error` | `Error`, `Rule` (both `#[non_exhaustive]`) |
| `cbor` | deterministic CBOR codec (unchanged) |
| `schema` | `Manifest`, `Object`, `Part`, `BlobRef`, `Shard`, `DType`, constants, CBOR mapping |
| `vocab` | `Vocabulary`, `Layout`, `Encoding`, `LogicalType`, the standard set |
| `store` | `Store`, `StoreId` — one file, mapped or merely indexed |
| `catalog` | `Catalog`, `Entry`, `PartEntry`, `Payload`, `Location` |
| `source` | `Source`, `Tensor`, `Part`, `Caps`, `Bytes`, `Verified` |
| `validate` | the §3.6 / §8 rules over a `.zt` image |
| `writer` | `Writer`, `ObjectBuilder`, `Sink`, `DataShardWriter` |
| `csr` | `zt.sparse_csr/1` assembly (out of the core reader) |

`Layout` as an enum (`Dense` / `Other(String)`) is gone; a layout is a string id
in the schema and a trait in the vocabulary.

## Migration

1.x → 2.0 is a breaking crate change with no file-format component. The known
consumers are this repo's own crates and `pie`, which is migrated in the same
pass.
