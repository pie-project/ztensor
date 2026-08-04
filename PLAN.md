# zTensor v2 implementation plan

Fresh start on the `v2` orphan branch. No compatibility with the v1 codebase
(kept on `main` as reference material only; the pickle VM and gguf/npz
parsers are worth porting into the compat crate later).

## Confirmed decisions

- **Own mini-CBOR codec.** The spec restricts CBOR hard enough (no tags,
  8 value types, depth ≤ 32, deterministic profile mandatory) that a
  ~300-line codec is smaller than a dependency and enforces determinism
  and duplicate-key rejection structurally.
- **Sync + mmap only.** No async, no io_uring in core. If pie needs io_uring
  later it becomes another `Source` implementation, not a core change.
- **Read N formats, write one.** Foreign formats are read-only projections
  into the object model; the only writer is `.zt`.
- **Capability ladder** is the API spine:
  tier 0 enumerate/metadata → tier 1 decoded read (owned) →
  tier 2 zero-copy view → tier 3 page-exclusive verified mmap
  (canonical `.zt` only). Never degrade silently; `view()` errors rather
  than falling back to a copy.
- Spec: `spec/ztensor-v2-spec.md` (Draft 4). Alignment floor 4096, canonical
  placement 65536.

## Workspace layout

```text
ztensor/            core: .zt v2 format, object model, reader and writer
  src/format/       L0 container + L1 manifest, frozen; opens no files
  src/vocab.rs      L2 vocabulary, open and registry-managed
  src/read.rs       Source, Tensor, Part, resolvers
  src/write.rs      Writer, Sink
  src/provide/      the face turned towards foreign-format projections
  examples/minimal_reader.rs    executable-spec reference reader
ztensor-compat/     foreign-format projections + layout profiles  [M6]
ztensor-cli/        zt binary: ls / verify / convert / pack / diff [M7]
ztensor-py/         pyo3 bindings                                  [M8]
conformance/        golden corpus generator + runner               [M2]
spec/               Draft 4 + profile registry documents
```

Core dependencies: `memmap2`, `xxhash-rust` only.

## Milestones

- **M1: Core round-trip** (dense + raw, single file): mini-CBOR codec,
  models, writer (canonical placement default, dedup of identical parts),
  reader (mmap MAP_SHARED, §3.6 core validation), `minimal_reader` example.
  Exit: write→read round-trip; determinism test (same input twice →
  bit-identical files); tied-weight dedup test.
- **M2: Validation hardening + conformance corpus**: every §3.6 MUST as a
  rule-tagged error; handcrafted must-reject corpus (partial overlap,
  duplicate keys, truncation, bad UTF-8, nonzero odd nibble, …);
  cargo-fuzz on reader + manifest parser. Exit: corpus runner is a CI gate.
- **M3: Capability ladder API**: `Source` trait, `caps()`, page-exclusivity
  detection, madvise/eviction helpers. Exit: pie-consumable surface frozen.
- **M4: L2 framework**: `LayoutProfile`/`EncodingProfile` traits,
  `zt.sparse_csr/1` (assembly in `ztensor-compat`, the first profile to live
  outside core), `zt.zstd-seekable/1` (feature-gated), logical-type size
  functions incl. `f4_e2m1`.
- **M5: Sharding + overlays**: shard table, resolver trait (positional /
  CAS / custom), verification ladder, LoRA-overlay integration test.
- **M6: Compat crate**: safetensors → gguf (quant blocks as `gguf.*`
  profiles, tier 0/1) → npz → pt (pickle VM port; `allow_pickle` call-site
  gate, default off) → onnx/hdf5 backlog. Foreign golden files join the
  corpus.
- **M7: CLI**: ls (prints expected shard list), verify (ladder tier
  select), convert (→ canonical), diff (digest-based); `open_cached()`
  repack helper for pie.
- **M8: Python bindings.**
- **M9: pie integration**: weight streaming on tier 3 (canonical check →
  64 KiB mmap → per-tensor eviction), convert-cache fallback.

Ordering rationale: corpus (M2) lands before compat (M6) so the core's
rejection rules harden before lenient foreign parsers appear; ladder (M3)
lands before profiles (M4) so pie integration can start on dense-only.
