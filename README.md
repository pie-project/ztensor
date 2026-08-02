# zTensor

An aligned, verifiable container format for tensor data — and one loader
that reads every other tensor format into the same object model.

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

## Layout of this repository

| Path | What it is |
| --- | --- |
| `spec/` | The normative format specification and the profile registry |
| `ztensor/` | Core crate: reader, writer, validation, sharding |
| `ztensor-compat/` | Foreign-format projections (feature-gated per format) |
| `ztensor-cli/` | The `zt` binary |
| `ztensor-py/` | Python bindings (`pip install ztensor`) |
| `conformance/` | Golden corpus: files a conforming reader must accept and must reject |
| `fuzz/` | cargo-fuzz targets for the container, the CBOR codec, and every parser |
| `benchmark/` | The harness behind the published numbers |
| `website/docs/` | Documentation sources |

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

| Format | Feature | Notes |
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

## Status

Pre-1.0. The spec is Draft 2 and the format may still change; the crates
are versioned `2.0.0-alpha`.

## Documentation

- [Format specification](spec/ztensor-v2-spec.md)
- [Profile registry](spec/profiles/)
- [Guide](website/docs/guide.md)
- [Benchmarks](website/docs/benchmarks.md)

## License

MIT
