# Migrating from 1.x to 2.0

**Your files are fine.** The container, the CBOR manifest, the digests and
canonical form are byte-for-byte what they were: a `.zt` written by any 1.x
release opens in 2.0, and 2.0 writes files 1.x can read. Nothing here is about
data. It is all about the crate surface.

If you want the reasoning behind the shapes below, see
[`DESIGN-2.0.md`](DESIGN-2.0.md). This file is the rename table.

The worked example is real: `pie`, the only downstream consumer, moved across
in one commit — its checkpoint reader lost 331 lines and gained 272.

---

## The one big idea

There used to be three reader types — `Reader` (one file), `Model` (a root
plus shards) and `Composite` (N files read together) — with the same five
methods and no common trait. There is now one:

```rust
let src = Source::open("model.zt")?;        // a file, and its shards if it has any
let src = Source::open_all(&paths)?;        // several files, one name space
let src = ztensor_compat::open("m.gguf")?;  // any format
```

They collapsed because the thing that separated them was a confusion: a
`Manifest` is what *one file says*, and a consumer needs an *index it can
query*. Those are now different types (`schema::Manifest` and `Catalog`), and
once they are, all three readers are the same type built three ways.

`Source` is a struct, not a trait. If you were generic over `Source`, take
`&ztensor::Source` instead.

---

## Reading

| 1.x | 2.0 |
| --- | --- |
| `Reader::open(p)`, `Model::open(p)` | `Source::open(p)` |
| `Model::open_with(p, &resolver)` | `Source::options().resolver(r).open(p)` |
| `Composite::new(sources)` | `Source::open_all(&paths)` or `Source::merge(v)` |
| `ztensor_compat::open_any(p)` | `ztensor_compat::open(p)` |
| `ztensor_compat::open_all(p)` → `Composite` | `ztensor_compat::open_all(p)` → `Source` |
| `src.manifest().objects` | `src.tensors()`, `src.tensor(name)?`, `src.names()` |
| `src.read(name, part)?` | `src.tensor(name)?.part(part)?.bytes()?` |
| `src.view(name, part)?` | `src.tensor(name)?.part(part)?.map()?` |
| — | `…locate()?` — the byte range, for your own I/O |
| `src.caps(name, part)?` | `src.tensor(name)?.part(part)?.caps()` |
| `r.verify(name, part)? -> bool` | `t.verify()? -> Verified` |
| `r.prefetch(name, part)` | `t.prefetch()` |
| `r.evict(name, part)` | `t.evict()` |
| `read_csr(&src, name)` | `csr::read(&tensor)` |
| `r.is_data_shard()` | `src.is_data_shard()` |
| `model.verify_shards()` | `src.verify_shards()` |

For the common case, the part is implied: `src.tensor(n)?.bytes()?` is the
`"data"` part.

`bytes()` is new and is what most call sites want. It returns the best the
source can do and says which it gave, so the `if caps.zero_copy { view } else
{ read }` bridge that every caller used to write is gone:

```rust
let bytes = t.bytes()?;          // Deref<Target = [u8]>
if bytes.is_mapped() { /* it was a borrow */ }
```

### Capabilities

`tier()` is gone. It flattened four independent facts onto one ordinal that
did not match the operations it claimed to gate — it demanded a digest before
admitting a part could be evicted, which eviction never needed. Ask for the
one you mean:

| 1.x | 2.0 |
| --- | --- |
| `caps.zero_copy` | `caps.map` |
| `caps.verifiable` | `caps.verify` |
| `caps.page_exclusive` | `caps.evict` |
| — | `caps.locate` |
| `caps.tier() >= 2` | `caps.map` |
| `caps.tier() == 3` | `caps.evict && caps.verify` |

Each field is now computed by the very predicate its method checks, so a
report and a behaviour cannot drift apart.

### New: opening without mapping

```rust
let src = Source::index("model.zt")?;   // header reads only
let at = src.tensor("w")?.locate()?;    // (store, offset, len)
```

What a planner wants: it answers where every tensor lives without mapping a
hundred gigabytes. `caps.locate` is true, `caps.map` is false, and `bytes()`
copies.

---

## The object model

The schema types moved to a module of their own, and `Part` at the crate root
is now the *handle*, not the manifest record.

| 1.x | 2.0 |
| --- | --- |
| `ztensor::Object` | `ztensor::schema::Object` |
| `ztensor::Part` (record) | `ztensor::schema::Part` |
| `ztensor::Part` | (now the handle returned by `tensor.part(..)`) |
| `part.ltype` | `part.logical` — the manifest key is still `"type"` |
| `ztensor::Layout` (enum) | a plain `String`; `obj.layout` is the id |
| `obj.layout.as_str()` | `obj.layout` / `tensor.layout()` |
| `logical_size(lt, dt, n)` | `vocab.size_of(lt, dt, n)` |
| `registered_dtype(lt)` | `vocab.dtype_of(lt)` |
| `DType::from_name(s)` | `s.parse::<DType>()` |
| `validate_bytes(buf)` | `validate_bytes(buf, &Vocabulary::standard())` |

`Catalog`, `Entry`, `Payload` and `Location` are new: the resolved index a
`Source` answers from.

---

## Writing

`add_dense`, `add_object`, `add_external_object`, `link_object` and
`stream_object` were five doors into one room. There is now one builder, and
`add` is the one-liner over it.

| 1.x | 2.0 |
| --- | --- |
| `Writer::create_with_alignment(p, a)` | `Writer::options().canonical(false).align(a).create(p)` |
| `w.add_dense(n, &shape, dt, data)` | `w.add(n, shape, dt, data)` |
| `w.add_object(n, &shape, layout, &parts, attrs)` | `w.object(n).shape(..).layout(..).part(..)…add()` |
| `w.add_external_object(..)` | `w.object(n)…part(p).external(part).add()` |
| `w.link_object(n, &obj, shard)` | `w.link(n, &obj, shard)` |
| `w.add_shard(size, digest)` | `w.add_shard(&Shard)`, or `shard_identity(path)?` |
| `w.stream_object(..)` → `ObjectWriter` | `w.object(n)…length(k).stream()` → `Sink` |
| `w.write_chunk(&mut obj, chunk)` | `sink.write(&mut w, chunk)` |
| `w.end_object(obj)` | `sink.close(&mut w)` |
| `w.ingest(src.as_ref())` | `w.ingest(&src)` |
| `DataShardWriter::finish() -> (u64, String)` | `-> Shard` |

Alignment and canonical form are separate questions now. Asking for an
alignment while meaning "let me insert out of order" is refused with an error
that says so, rather than quietly turning canonical form off:

```rust
Writer::options().align(4096).create(p)     // error: add .canonical(false)
Writer::options().canonical(false).align(4096).create(p)   // this
```

### New: publishing

Every producer was hand-writing the same three steps. The writer does them:

```rust
let mut w = Writer::publish("model.zt")?;   // writes beside it, renames on finish
```

Nothing exists at the path until `finish()`, and a writer dropped without
finishing removes its partial.

---

## Errors

`Error` and `Rule` are `#[non_exhaustive]`, so match with a wildcard arm.
Rather than destructuring, ask:

```rust
// 1.x
if let Err(Error::Reject { rule: Rule::Digest, .. }) = result { … }
// 2.0
if result.as_ref().err().and_then(Error::rule) == Some(Rule::Digest) { … }
```

`Error::reject` is now public, because a layout or encoding profile
registered from another crate has to be able to refuse a file exactly as a
built-in one does.

---

## Vocabulary: layouts, encodings and logical types

The spec calls L2 registry-managed; now the registry is a value you can
extend, and a downstream crate can add a profile that is validated exactly
like a built-in:

```rust
let vocab = Vocabulary::standard().with_layout(MyQuantGroup);
let src = Source::options().vocabulary(&vocab).open(path)?;
let w = Writer::options().vocabulary(&vocab).create(out)?;
```

A reader without the profile still opens the file and reads its bytes; it
just does not check the profile's rules. `layout_profile()` and
`encoding_profile()` are replaced by `vocab.layout()` / `vocab.encoding()`.

---

## Python

`ztensor.numpy.load_file` and `save_file` still work and still mean the same
thing — they are now documented as what they are, a safetensors-shaped shim
for migrating existing code. Everything else changed.

| 1.x | 2.0 |
| --- | --- |
| `src.keys()` | `src.names()`, or iterate: `for t in src` |
| `src.info(name)` | `t.shape`, `t.dtype`, `t.logical`, `t.layout`, `t.nbytes`, `t.parts` |
| `src.read(name)` | `src[name].tobytes()` |
| `src.view(name)` | `memoryview(src[name])`, or DLPack |
| `src.caps(name)["zero_copy"]` | `src[name].caps.map` |
| `src.caps(name)["tier"]` | ask the field you mean |
| `ztensor.verify(p) -> int` | `-> (digest_verified, without_digests)` |
| `Writer(path, align)` | `Writer(path, canonical=…, align=…, publish=…)` |
| `w.add(name, shape, dtype, data, compress)` | `w.add(name, data, shape=…, dtype=…, logical=…, encoding=…)` |

The source is a context manager and a mapping; tensors are handles:

```python
with ztensor.open("model.safetensors") as src:      # or open([shard1, shard2])
    for t in src:
        print(t.name, t.shape, t.dtype, t.caps.map)

    t = src["layer.weight"]
    arr = torch.from_dlpack(t)     # zero-copy, and DLPack can say bfloat16
    view = memoryview(t)
    t.location                     # (path, offset, nbytes) for your own I/O
    t["scales"]                    # parts are tensors too
```

Both zero-copy exports keep the mapping alive on their own, so an array or a
memoryview stays valid after the source it came from is closed.

There is no `ztensor.torch` and there will not be one: tensors export DLPack
and the buffer protocol, which is how numpy, torch and jax all read them
without this package knowing any of them exist.
