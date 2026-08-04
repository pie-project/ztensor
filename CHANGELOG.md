# Changelog

Notable changes to the `ztensor`, `ztensor-compat` and `ztensor-cli` crates
and the `ztensor` Python package, which are versioned together.

The **file format is versioned separately**: `.zt` is container version 2,
spec **Draft 4**.

## 2.1.1

A second pass over the surface, after reading 2.1 back. The container is
untouched — no file changes meaning — and so is every operation. What changed
is which of them exist, since several turned out to be two ways of saying one
thing, and two others let a caller write something contradictory and compile.

**This breaks the API in a patch release**, which is not what a patch number
means. It is deliberate and it is a one-off: 2.1.0 has a single user, updated
in the same breath, and spending 3.0.0 on a surface that had been public for a
week would have priced the next real major bump out of the range where it
belongs. Anything depending on `ztensor = "2.1"` resolves into this and will
not compile; pin `=2.1.0` if that is a problem. From here on the number means
what it says.

### Removed

- **The nine `Tensor` methods that were `self.data()?.<the same name>()`** —
  `bytes`, `map`, `locate`, `caps`, `dtype`, `logical`, `nbytes`, `prefetch`,
  `evict`. They read as whole-tensor operations and were not: each addressed
  the `"data"` part, while `verify` on the same receiver covered *every* part.
  Nothing distinguished the two, so a reader who took `t.verify()` as evidence
  about `t.bytes()` was wrong in one direction and a reader who took
  `t.bytes()` for the whole tensor was wrong in the other.

  `verify` stays as it was; the rest move one call along. `t.data()?.map()`,
  and `t.parts()` for the ones that are not `"data"`.

- **`Bytes`** — it was `Cow<'a, [u8]>` with two of its methods rewritten and a
  name that collides with the `bytes` crate. `Part::bytes` now returns the
  `Cow`. `is_mapped()` is `matches!(b, Cow::Borrowed(_))`.

- **`Source::index`** — its body was `options().map(false).open(path)`, which
  is what `Options` is for.

- **`Source::from_parts_with`** — likewise: giving a projection a vocabulary
  is `Source::options().vocabulary(&v).from_parts(stores, catalog)`.

- **`Tensor::entry`** — no caller, and it leaked `provide::Entry` into the
  reader's surface.

- **`PositionalResolver` and `CasResolver`** — both were a closure with a
  struct around it, and `ShardResolver` is already implemented for closures.
  They are now `read::positional(root)` and `read::cas(dir)`, returning
  `impl ShardResolver`. `DirectoryResolver` stays a type: it holds a scan
  index and a digest cache.

### Changed

- **`PartBuilder::external` takes what the caller knows**: a shard name and a
  byte range, `external("00001", offset..end)`. It used to take a
  `format::Part`, so referencing a blob meant hand-assembling a manifest
  record — restating the dtype, nesting the shard name inside a `BlobRef`,
  and importing `format::Part` under an alias to keep it apart from
  `read::Part`. `format::Part` no longer appears in any writer signature, and
  the collision with it went with that.

- **`Writer::link` is unchanged** and is still the way to reference a shard
  that carries a manifest: it reads the range out of that manifest instead of
  asking for it.

- **Renames**, each towards a convention the crate or std already had:
  `Verified::checked` → `is_checked` (`Bytes::is_mapped` set the pattern),
  `Provenance::root` → `as_root` (`cbor::Value::as_map` set it),
  `Catalog::entry` → `get_key_value` (`entry` means the insertion API in
  every map in std, and `Entry` is a type in that module),
  `provide::Opaque` → `provide::Decode` with `read` → `decode` (it was named
  for what it is not; it decodes), `Store::with_opaque` → `with_decoder`.

- **`read::shard_identity` is no longer re-exported at the crate root.** Its
  two siblings, `manifest_of` and `canonical_violations`, never were, and
  which of the three is promoted should not depend on which is called most.

### Fixed

- **A part with two payloads is refused.** `p.bytes(&data).length(4)`
  compiled, silently dropped `data`, and wrote a file whose blob was never
  filled in. The builder methods return `Self` and so cannot refuse; the
  second one now records the contradiction and the object fails to build,
  naming both setters. The scoped-builder shape is what rules out catching
  this in the type system: the closure passed to `part` must return the type
  it was given, so no state can ride along in it.

- **A digest on a part the writer writes is refused.** It could only agree
  with the computed digest or be wrong. `PartBuilder::digest` is new and is
  for external parts, whose bytes this writer never sees.

- **An external part with an encoding is refused** rather than silently
  dropping the encoding, and points at `Writer::link`, which can read the
  decoded length out of the shard's manifest.

## 2.1.0

A rewrite of the crate's surface, and a format addition. The container is
unchanged: every `.zt` file keeps its meaning, and what moved is where the
names live and who does the checking.

The API breaks, which would ordinarily be a major bump. It is not one because
2.0.0 has no users to break — this is the shape the crate goes out in.

### Format (spec Draft 4)

- **A content digest** (§6.4). The whole-file hash of a canonical file
  identifies an *artifact*: those bytes, that layout. The content digest
  identifies the *model*, and is defined so that layout cannot reach it.
  Offsets, lengths, alignment, padding, blob sharing, encodings and the shard
  table are all absent from it, so the same tensors give the same answer
  whether they sit in one file or fifty, packed at 4 KiB or 64 KiB, raw or
  compressed.

  It is computed from the manifest alone, so a reader that has fetched a root
  and nothing else can compute it whatever the model weighs. It is defined
  only where every part carries a digest, and it is never stored in the file:
  a stored value is a claim that can be false.

  This is also why a canonical multi-file profile stays deferred. The reason
  to pin a shard-partition policy was to keep identity stable across splits,
  and identity is now stable without one.

### Changed

- **The modules are the spec's layers.** `format` is L0 + L1, frozen, and
  opens no file: the magic, the footer, the alignment floor, the manifest
  schema, its CBOR mapping, and the rules that decide conformance and
  canonical form. `vocab` is L2. `read` and `write` are what this
  implementation does with them, and `provide` is the face turned towards a
  crate projecting a foreign format. A spec section and the code that
  implements it now have the same address, and the dependency direction says
  which half a second implementation would have to agree with.

- **Twenty-two names at the crate root, from forty-seven.** What a consumer
  uses is re-exported and nothing else is: the ten format constants,
  `check_shard_name`, `BlobRef`, the four shard resolvers, and the projection
  types keep their module paths. `Store`, `StoreId` and `Location` stay at the
  root, because a consumer planning its own I/O has to know which file a
  tensor is in — that is the reader's business, not just a provider's.

- **Object descriptions are scoped.** `Writer::object` and `Writer::stream`
  take a closure, and each part is described inside another:

  ```rust
  w.object("q", |o| {
      o.shape([4096u64, 4096])
          .layout("zt.quant_group/1")
          .part("data", |p| p.dtype(DType::U8).bytes(&payload))
          .part("scales", |p| p.dtype(DType::U8).logical("f8_e8m0").bytes(&scales))
  })?;
  ```

  A part setter with no part to apply to was an error discovered when the
  object was added, and is now a thing that cannot be written. The builder is
  a plain description with no writer behind it, so the deferred-error field
  and the "applies to the most recently named part" bookkeeping are both gone.
  The cost is that a description borrows its bytes: a payload built inline has
  to be bound to a local first.

- **`Source::provenance`** replaces `manifest()` and `is_data_shard()`. Those
  were two partial views of one three-way fact — a `.zt` root, a data shard
  (§7.2), or a projection — which differ in *what can be verified*, so a
  consumer deciding how far to trust a checkpoint reads one answer.
  `provenance().root()` is the shorthand for code that already knows.

- **`Tensor::verify` verifies every part**, and `verify_all` is gone. A
  quantized tensor's bytes include its scales, so checking only `"data"`
  passed a tensor whose scales had rotted. One part at a time is
  `tensor.part(name)?.verify()`.

- **`zt.sparse_csr/1` assembly moved to `ztensor-compat`.** L2 is open and
  registry-managed by design, which a core module for exactly one layout
  contradicted. It is now the first layout profile living outside the core
  crate, which is where a profile added downstream would live.

### Added

- **`Writer::append`** adds objects and shards to a finished `.zt` without
  rewriting the blobs already in it, the way spec §2.5 describes: new blobs, a
  new manifest and a new footer go past the old end, and nothing already in
  the file is moved, overwritten or truncated. The cost is the size of what
  you add rather than the size of the file. Adding a 4 KiB tensor to a
  512 MiB file takes 55 µs against 303 ms to rewrite it.

  The alignment the file was written at is carried forward, worked out from
  the offsets it already has. A 64 KiB file stays a 64 KiB file, so the
  per-tensor page exclusivity that placement buys is not lost on the tensors
  added later. `.align()` overrides it.

  It is **not atomic**. From the first byte written until `finish` puts a
  footer at the new end, the footer is not at EOF and no reader will open the
  file. Every original byte is still there, so a crashed append is undone by
  truncating the file back to its old length. Use `Writer::publish` when the
  file is worth more than the rewrite.

  Canonical form forbids unreferenced blobs (§6.3 rule 1), which an append
  leaves behind, so this needs `.canonical(false)`.

- **`Manifest::content_digest`** and **`zt id`**, computing the digest above.

- **`read::canonical_violations`** and **`zt verify --canonical`** decide
  whether a file is in canonical form and name every rule it breaks. The spec
  calls canonical form the recommended distribution format, which is only
  worth saying if the receiver can tell; nothing is stored in a file to say
  it, and nothing needs to be, since all six rules of §6.3 are decidable from
  the bytes.

- **`DigestAlgorithm`**, with `sha256` alongside `xxh3`. §6.5 makes a
  cryptographic shard digest the thing that lets one signature over a root
  cover every shard byte, and that was previously impossible to produce
  through the API. Verification takes whatever algorithm a file used, so a
  build cannot write digests it is unable to check.

  `shard_identity` now takes the algorithm, rather than there being two
  functions for one operation. The choice is a real one and worth spelling.

- **`Sink::attach`** borrows a sink and its writer together as an
  `io::Write`, so a streamed part can be fed by `io::copy`, a `BufWriter` or
  an encoder. `Sink::write` stays: it takes the writer per call, which is what
  lets a producer driven from outside own both, and no borrow can express
  that. For the duration of an `attach`, the compiler rather than the sink's
  ticket is what stops the writer being used for anything else.

### Fixed

- **A `Sink` could drive a `Writer` that did not open it.** The check was
  whether *some* object was open on that writer, which is true of any writer
  mid-stream, so a sink handed the wrong one appended its bytes to whatever
  blob that writer had open. Two files quietly wrong, and the sink believing
  it had written a part it never wrote. Sinks now carry a ticket the writer
  checks on every call.

- **`DirectoryResolver` could not match a `sha256` shard table.** It hashed
  every candidate with one fixed algorithm and compared digest strings, so the
  very tables signing needs (§6.5) were the ones it could not resolve. It now
  indexes by size and computes the digest in the algorithm the shard asks for,
  which also means the scan reads no tensor bytes at all.

- **Leaving canonical form dropped the alignment to the 4 KiB floor.**
  Placement is not part of what canonical form is; 64 KiB is what buys
  per-tensor page exclusivity, and it matters just as much to a sharded
  model, which cannot be canonical at all (§6.3 rule 6). Every sharded root
  was silently losing eviction on any host with pages above 4 KiB unless its
  author knew to ask for the alignment back. A non-canonical writer now
  defaults to 64 KiB like any other; `.align()` still chooses.

- **`evict` did not exist off unix, but `Caps::evict` did.** The obvious
  `if caps.evict { part.evict()? }` therefore failed to *compile* on Windows,
  while the neighbouring `prefetch` was present everywhere and did nothing.
  `evict` is now present everywhere too and refused where there is no way to
  drop page cache, with `Caps::evict` reporting `false` there.

- **`Source::get` scanned linearly** through a `BTreeMap`, so addressing a
  thousand tensors by name cost a thousand scans. It needed the name as the
  catalog stores it and had no way to ask for it; `Catalog::entry` is that
  way. `Source::contains` is gone as `get(..).is_some()`.


### Removed

- **`DataShardWriter`.** It wrote the manifest-less shard of §7.2, and the
  name made it look like the way to produce a shard. It was the wrong way.
  §7.2 lets any `.zt` serve as a shard, and one with a manifest is strictly
  more useful: it states which of its bytes are occupied, so a consumer can
  prove a tensor has its pages to itself and evict it, and it carries
  per-part digests, so those bytes can be verified. A manifest-less shard
  gives up both, which is exactly what a streaming loader wants.

  Write shards with `Writer` and take their identity with `shard_identity`.
  Reading manifest-less shards is unchanged: other producers write them, and
  the format defines them.

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
  keyed by name. An index also has to be renumbered when a shard is added or
  dropped, while a name does not. A name is a **label**: identity is still
  `(size, digest)`, still never a path.
- **A name is constrained** to `[A-Za-z0-9._-]`, no leading `.`, at most 64
  bytes. The conventional resolvers use a name as a single path component,
  so the format is what prevents a manifest from expressing `../../etc/passwd`
  instead of leaving every consumer to sanitize it, with one of them
  eventually forgetting.
- **A blob reference is `[offset, length]`**, and the shard moved to its own
  optional `shard` field. Absent means the containing file. Every other
  optional part field already works this way, and it means a single-file
  manifest, which is the common case, never mentions sharding at all.
  The spec claimed the single file was the degenerate case; now the bytes say
  so too.

`Writer::add_shard` takes a name and returns `()`; `Writer::link` takes a name;
`ShardResolver::resolve` receives `&str`; `BlobRef::shard` is
`Option<String>`. The positional convention resolves a shard named `n` to
`<stem>-<n>.zt`, so naming shards `00001-of-00003` reproduces the file names
checkpoints already ship with.

### Changed

- **One reader type.** `Reader`, `Model` and `Composite` were three types
  with the same five methods and no common trait. They became `Source`, built
  three ways (`open`, `open_all`, `merge`). What separated them was a conflation:
  a manifest is what one file says, and a consumer needs an index it can
  query. Those are now `schema::Manifest` and `Catalog`.
- **`Source` is a struct, not a trait.** Foreign formats build a `Catalog`
  and hand back an ordinary `Source`; there is no per-format type in the
  public API.
- **Three ways to get bytes, one per intent.** `bytes()` gives the best the
  source can do and says whether it borrowed or copied; `map()` insists on a
  borrow; `locate()` gives the address so a caller can do its own I/O.
- **`Caps` fields are named after the operations they gate**: `map`,
  `locate`, `evict` and `verify`. Each is computed by that operation's own
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
  `Part::ltype` is now `Part::logical`; the manifest key is unchanged.
- `Error` and `Rule` are `#[non_exhaustive]`.
- Python: tensors are handles, sources are mappings and context managers.
  `ztensor.numpy` remains as a documented safetensors-shaped shim.

### Added

- **`DirectoryResolver`** scans a directory and matches shards by size and
  whole-file digest, ignoring names entirely. It is the one convention that
  keeps working after an arbitrary rename.
- **`Source::index`** opens without mapping. It answers where every tensor
  lives for the cost of a header read, which is what a planner wants.
- **`Part::locate`** gives the exact byte range of the decoded bytes, for
  io_uring, cuFile, or a staged host-to-device copy.
- **`Writer::publish`** writes beside the destination and renames into
  place on `finish`; a writer dropped without finishing leaves nothing.
- **`Vocabulary` is a value.** Layouts, encodings and logical types can be
  registered from another crate and are then validated exactly like the
  built-ins. `Error::reject` is public so those profiles can refuse a file.
- **DLPack and the buffer protocol** in Python, so numpy, torch and jax read
  tensors zero-copy without this package knowing they exist, and DLPack
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
  did not match the operations it claimed to gate: tier 3 demanded a digest
  before admitting a part could be evicted, which eviction never needed.
  Nothing in the library branched on it.
- The per-format types (`Safetensors`, `Gguf`, `Npz`, `Pt`, `Hdf5`, `Onnx`).
  Use `ztensor_compat::open`.
- `ztensor_compat::open_any` (now `open`), `layout_profile`,
  `encoding_profile`, `logical_size`, `registered_dtype`.

