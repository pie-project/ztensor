# Changelog

Notable changes to the `ztensor`, `ztensor-compat` and `ztensor-cli` crates
and the `ztensor` Python package, which are versioned together.

The **file format is versioned separately** and did not change in this
release. `.zt` is still container version 2, spec Draft 2.

## 2.0.0

A rewrite of the crate surface. No change to the format: every `.zt` file
written by a 1.x release reads unchanged, and the conformance corpus and the
canonical-determinism tests are the evidence.

See [MIGRATING.md](MIGRATING.md) for the rename table.

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

## 1.2.3 and earlier

Not catalogued here.
