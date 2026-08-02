# zTensor

An aligned, verifiable container format for tensor data — and one loader
that reads every other tensor format into the same object model.

**Crates 1.3.0 · Format `.zt` version 2 · Spec Draft 2**

```rust
// Read anything: .zt, .safetensors, .gguf, .npz, .pt, .h5, .onnx
let src = ztensor_compat::open_any("model.safetensors")?;
let bytes = src.read("layer.weight", "data")?;

// Write one thing: a canonical, digest-carrying .zt file
let mut w = ztensor::Writer::create("model.zt")?;
w.ingest(src.as_ref())?;
w.finish()?;
```

```bash
zt ls model.gguf                    # inspect anything
zt convert model.gguf model.zt      # canonical, verifiable output
zt verify model.zt --deep           # structure + digests + shards
zt diff a.safetensors b.zt          # compare across formats
```

## What `.zt` gives you

- **64 KiB placement** in canonical files, so every tensor starts on a page
  boundary on every platform in use (4 KiB, 16 KiB, 64 KiB pages). Each
  tensor can be mapped, registered, and evicted independently — which is
  what makes weight streaming possible without disturbing neighbours.
- **Verifiability**: an XXH3 digest per tensor, plus one over the manifest
  itself. Corruption is an error, not a wrong answer.
- **Extensibility without version churn**: quantization schemes, sparse
  layouts, and compression are versioned *profiles* in a registry, not
  cases baked into the container. Unknown vocabulary is refused, never
  guessed at.
- **Reproducibility**: two canonical writes of the same tensors produce
  byte-identical files, so a file hash is a stable model identity.
- **No code execution** anywhere in the format.

## Performance

Measured by `cargo run --release -p benchmark` on an i9-13900K with a
Samsung 980 PRO NVMe: 1 GiB of tensor data, 32 MiB layer weights, median
of 9 runs. Full tables and methodology in
[the benchmarks page](website/docs/benchmarks.md).

| Operation | `.zt` | `.safetensors` |
| --- | --- | --- |
| Write | 1.66 GB/s | 1.35 GB/s |
| Read, full traversal (warm cache) | 6.62 GB/s | 6.72 GB/s |
| Read, full traversal (cold cache) | 2.38 GB/s | — |
| Verify every digest (XXH3) | 13.99 GB/s | not available |
| Open (enumerate 50 tensors) | 0.04 ms | 0.03 ms |

Reading is the same speed, and that is the honest answer: both are
memory-mapped byte ranges, so a warm traversal is bounded by memory
bandwidth. `.zt` opens marginally slower because opening also validates
bounds, alignment, blob non-overlap, size equations, and the manifest
hash.

The 64 KiB placement costs padding — about 32 KiB per tensor, so how much
depends entirely on tensor size:

| Model shape | `.zt` canonical | `.zt` 4 KiB floor | `.safetensors` |
| --- | --- | --- | --- |
| 50 × 32 MiB tensors (transformer-like) | +0.15% | +0.01% | +0.00% |
| 1538 × 1 MiB tensors (worst case) | +4.68% | +0.27% | +0.01% |

## Layout of this repository

| Path | What it is |
| --- | --- |
| `spec/` | The 646-line normative specification and 4 profile documents |
| `ztensor/` | Core crate: reader, writer, validation, sharding |
| `ztensor-compat/` | Foreign-format projections (feature-gated per format) |
| `ztensor-cli/` | The `zt` binary |
| `ztensor-py/` | Python bindings (abi3, Python ≥ 3.9) |
| `conformance/` | Golden corpus: 18 files a reader must accept, 52 it must reject |
| `fuzz/` | cargo-fuzz targets: container, CBOR codec, and all six parsers |
| `benchmark/` | The harness behind the numbers above |
| `website/docs/` | Documentation sources |

The core crate's entire dependency list is `memmap2`, `xxhash-rust`,
`unicode-normalization`, and `libc` — plus optional `zstd` for the
seekable-compression profile. The CBOR codec is 418 lines in-tree
(excluding its tests),
because the spec restricts CBOR hard enough (no tags, 8 value types,
deterministic encoding mandatory) that owning it is smaller than depending
on a general one — and it lets determinism be structural rather than
optional.

## Three layers, three lifetimes

The format separates what must never change from what is expected to:

| Layer | Contents | Contract |
| --- | --- | --- |
| **L0 — Container** | Magic, 40-byte footer, aligned blob heap, byte order | **Frozen** |
| **L1 — Manifest** | Deterministic-CBOR schema | Gated by the footer's version integer |
| **L2 — Vocabulary** | Layouts, logical types, encodings, digests | **Deliberately mortal**, registry-managed |

A minimal conforming reader is `ztensor/examples/minimal_reader.rs` — 68
lines, needing only a CBOR decoder and XXH3.

## The capability ladder

Formats differ in what they can guarantee, and the API says so instead of
pretending otherwise:

| Tier | Guarantee | Where |
| --- | --- | --- |
| 0 | Enumerate objects and metadata | every format |
| 1 | Decoded read (owned bytes) | every format |
| 2 | Zero-copy view | mapped sources, raw parts |
| 3 | Tier 2 + page-exclusive + verifiable | canonical `.zt` |

`view()` errors rather than silently copying; `caps()` reports the truth
per part. Converting a foreign checkpoint is how you move it to tier 3.

## Supported formats

| Format | Cargo feature | Notes |
| --- | --- | --- |
| `.zt` | — | Native, including sharded models and overlays |
| `.safetensors` | `safetensors` (default) | Exact-tiling validation defuses header aliasing |
| `.gguf` | `gguf` (default) | Quantized blocks kept verbatim as `gguf.<type>/1` |
| `.npz` / `.npy` | `npz` | Big-endian and Fortran-order arrays are refused, not reinterpreted |
| `.pt` / `.bin` | `pickle` | Restricted VM, no code execution; non-contiguous tensors refused |
| `.h5` / `.hdf5` | `hdf5` | Contiguous and chunked, deflate + shuffle filters |
| `.onnx` | `onnx` | Graph initializers; external data refused |

Reading is many-to-one by design: writing is `.zt` only. Tracking every
foreign format as a *producer* is unbounded work that no consumer needs.

`pickle` is opt-in because parsing pickle at all is a larger attack
surface than anything else here.

## Testing

79 tests, a 70-file conformance corpus that is the CI gate, and three
fuzz targets. The parsers consume untrusted files, so the contract they
are held to is explicit: hostile input yields an error — never a panic, an
unbounded allocation, or a fabricated tensor. `ztensor-compat/tests/hostile.rs`
pins that with the reproducers behind each hardening fix.

```bash
cargo test --all-features
cargo run -p conformance --bin gen        # regenerate golden files
cargo +nightly fuzz run fuzz_compat       # foreign-format parsers
```

## Status

Pre-1.0 in intent if not in number: the format is Draft 2 and may still
change. The crate line continues from the 1.2.x series, but this is a
complete rewrite — the API, the file format, and the guarantees are all
new, and `.zt` v2 files are not readable by 1.2.x.

## Documentation

- [Format specification](spec/ztensor-v2-spec.md)
- [Profile registry](spec/profiles/)
- [Guide](website/docs/guide.md)
- [Benchmarks](website/docs/benchmarks.md)

## License

MIT
