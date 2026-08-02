# Changelog

Notable changes to the `ztensor`, `ztensor-compat` and `ztensor-cli` crates
and the `ztensor` Python package, which are versioned together.

The **file format is versioned separately**: `.zt` is container version 2,
spec **Draft 3**.

## 2.0.0

A rewrite of the crate surface, and one change to the L1 manifest schema made
before the format's first release.

### Format (spec Draft 3)

Draft 3 changes how a part addresses bytes in another file. The footer version
integer is unchanged at `2`, because Draft 2 was never released: no `.zt` file
in the wild uses the old spelling.

- **Shards are named, not numbered.** The shard table is keyed by a name
  chosen by the producer rather than by an integer index. It was the only
  integer-keyed map in a manifest whose objects, parts and attributes are all
  keyed by name — and an index has to be renumbered when a shard is added or
  dropped, while a name does not. A name is a **label**: identity is still
  `(size, digest)`, still never a path.
- **A name is constrained** to `[A-Za-z0-9._-]`, no leading `.`, at most 64
  bytes. The conventional resolvers use a name as a single path component,
  so the format is what prevents a manifest from expressing `../../etc/passwd`
  — rather than every consumer having to sanitize it, and one of them
  eventually forgetting.
- **A blob reference is `[offset, length]`**, and the shard moved to its own
  optional `shard` field. Absent means the containing file. Every other
  optional part field already works this way, and it means a single-file
  manifest — the overwhelmingly common case — never mentions sharding at all.
  The spec claimed the single file was the degenerate case; now the bytes say
  so too.

`Writer::add_shard` takes a name and returns `()`; `Writer::link` takes a name;
`ShardResolver::resolve` receives `&str`; `BlobRef::shard` is
`Option<String>`. The positional convention resolves a shard named `n` to
`<stem>-<n>.zt`, so naming shards `00001-of-00003` reproduces the file names
checkpoints already ship with.

### Added

- **`DirectoryResolver`** scans a directory and matches shards by size and
  whole-file digest, ignoring names entirely — the one convention that keeps
  working after an arbitrary rename.

### Changed

- **One reader type.** `Reader`, `Model` and `Composite` — three types with
  the same five methods and no common trait — became `Source`, built three
  ways (`open`, `open_all`, `merge`). What separated them was a conflation:
  a manifest is what one file says, and a consumer needs an index it can
  query. Those are now `schema::Manifest` and `Catalog`.
- **`Source` is a struct, not a trait.** Foreign formats build a `Catalog`
  and hand back an ordinary `Source`; there is no per-format type in the
  public API.
- **Three ways to get bytes, one per intent.** `bytes()` gives the best the
  source can do and says whether it borrowed or copied; `map()` insists on a
  borrow; `locate()` gives the address so a caller can do its own I/O.
- **`Caps` fields are named after the operations they gate** — `map`,
  `locate`, `evict`, `verify` — and each is computed by that operation's own
  precondition, so the report cannot drift from the behaviour.
- **One writer entry point.** `add_dense`, `add_object`,
  `add_external_object`, `link_object` and `stream_object` became
  `Writer::object`, with `add` as the one-liner over it.
- **Alignment and canonical form are separate options.** Asking for an
  alignment no longer silently turns canonical form off; it is an error that
  says how to mean it.
- **`verify` returns `Verified`**, not `bool`: a checked digest and "there is
  no digest" are different answers, and a mismatch is still a rejection.
- Layouts are plain strings in the schema; `Layout` as an enum is gone.
  `Part::ltype` is now `Part::logical` — the manifest key is unchanged.
- `Error` and `Rule` are `#[non_exhaustive]`.
- Python: tensors are handles, sources are mappings and context managers.
  `ztensor.numpy` remains as a documented safetensors-shaped shim.

### Added

- **`Source::index`** — open without mapping. Answers where every tensor
  lives for the cost of a header read, which is what a planner wants.
- **`Part::locate`** — the exact byte range of the decoded bytes, for
  io_uring, cuFile, or a staged host-to-device copy.
- **`Writer::publish`** — writes beside the destination and renames into
  place on `finish`; a writer dropped without finishing leaves nothing.
- **`Vocabulary` is a value.** Layouts, encodings and logical types can be
  registered from another crate and are then validated exactly like the
  built-ins. `Error::reject` is public so those profiles can refuse a file.
- **DLPack and the buffer protocol** in Python, so numpy, torch and jax read
  tensors zero-copy without this package knowing they exist — and DLPack
  carries `bfloat16`, which the numpy dtype table cannot. The versioned
  protocol is supported, which is what lets these tensors be marked
  **read-only**: legacy DLPack cannot say it, and a framework that believes
  it may write into a read-only mapping will fault. `copy=True` hands over a
  buffer the consumer owns; `copy=False` on bytes that must be decoded is a
  `BufferError`, as is a request for any device but the host.
- **Sources are `Send + Sync`**, so a loader can read a checkpoint from
  several threads.
- `Source::merge`, `shard_identity`, `manifest_of`, `cbor::map`, and `From`
  impls for building attributes.

### Fixed

- Page exclusivity is now decided for shards too: a shard that is itself a
  container has a known occupancy, so a tensor in another file is as
  evictable as one at home. A manifest-less data shard still claims nothing.
- Python: an arbitrary part's dtype was reported as the tensor's; `bf16`
  tensors came back as flat `uint8` with the shape dropped; `load_file`
  with `dense_only=False` raised.
- Python: a zero-copy export now keeps the mapping alive itself, so an array
  or memoryview stays valid after the source it came from is closed.

### Removed

- `Caps::tier()`. It flattened four independent facts onto one ordinal that
  did not match the operations it claimed to gate — tier 3 demanded a digest
  before admitting a part could be evicted, which eviction never needed.
  Nothing in the library branched on it.
- The per-format types (`Safetensors`, `Gguf`, `Npz`, `Pt`, `Hdf5`, `Onnx`).
  Use `ztensor_compat::open`.
- `ztensor_compat::open_any` (now `open`), `layout_profile`,
  `encoding_profile`, `logical_size`, `registered_dtype`.

