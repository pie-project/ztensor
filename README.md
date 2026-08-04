# zTensor

An aligned, verifiable container format for tensor data — and one loader that
reads every other tensor format into the same object model.

**[Documentation](https://pie-project.github.io/ztensor/)** ·
[Guide](https://pie-project.github.io/ztensor/guide.html) ·
[Specification](https://pie-project.github.io/ztensor/spec.html) ·
[Benchmarks](https://pie-project.github.io/ztensor/benchmarks.html)

**Crates 2.1.1 · Format `.zt` version 2 · Spec Draft 4**
*(the crate version and the format version are different numbers and do not
move together)*

```bash
zt ls model.gguf                    # inspect anything
zt convert model.gguf model.zt      # canonical, verifiable output
zt verify model.zt --deep           # structure + digests + shards
zt diff a.safetensors b.zt          # compare across formats
```

```rust
// In: .zt, .safetensors, .gguf, .npz, .pt, .h5, .onnx — detected from the file
let src = ztensor_compat::open("model.safetensors")?;
let bytes = src.tensor("layer.weight")?.data()?.map()?;   // a borrow, or an error

// Out: one format, aligned and digest-carrying
let mut w = ztensor::Writer::create("model.zt")?;
w.ingest(&src)?;
w.finish()?;
```

## The idea

A checkpoint is a few gigabytes of numbers that a machine has to get into
memory quickly and be sure it got right. `.zt` is built around those two
things.

- **Every tensor starts on a page boundary.** The operating system maps,
  protects and evicts memory a page at a time, so a tensor that shares a page
  with its neighbour cannot be handled on its own. Give each one its own pages
  and weights can be streamed in and dropped individually.
- **Every tensor carries a digest.** A flipped bit is an error you are told
  about, rather than an answer that is quietly wrong.
- **New quantization schemes are registry entries, not format revisions.** The
  container knows about *bytes and parts*; what the bytes mean is a versioned
  profile name it checks and passes through. An unknown name is refused rather
  than guessed at.
- **The same tensors always produce the same file.** So a file hash is a
  usable model identity — and for "same model, different packing", there is a
  content digest computed from metadata alone.
- **Nothing in the format executes code.**

## Compared to what is already out there

| | `.zt` | `.safetensors` | `.gguf` | `.pt` | `.npz` | `.onnx` | `.h5` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Zero-copy read | ✓ | ✓ | ✓ | ~ | ~ | | |
| Page-aligned tensors | ✓ | | | | | | |
| Per-tensor digest | ✓ | | | | | | |
| Safe (no code execution) | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ |
| Streaming / append | ✓ | | | | ~ | | ✓ |
| Sparse tensors | ✓ | | | ✓ | | | |
| Per-tensor compression | ✓ | | | | ~ | | ✓ |
| Extensible types | ✓ | | | N/A | | ✓ | ✓ |
| Byte-reproducible | ✓ | | | | | | |

A blank cell is not a criticism — `.safetensors` set out to be a safe, simple
replacement for pickle and is exactly that. The `.zt` column is what comes of
being designed once those columns already existed.
[Footnotes and per-format detail](https://pie-project.github.io/ztensor/formats.html).

## Reading is fast, and the same speed whatever went in

<p align="center">
  <img src="website/static/charts/cross_format_read.svg" alt="Cross-format read throughput" width="700">
</p>

| Format | zTensor | zTensor (zero-copy off) | The format's own library |
| :--- | :--- | :--- | :--- |
| .zt | **2.27 GB/s** | 0.96 GB/s | n/a |
| .safetensors | **2.47 GB/s** | 1.00 GB/s | 1.57 / 1.59 GB/s† |
| .pt | **2.29 GB/s** | 0.83 GB/s | 1.60 GB/s |
| .npz | **2.33 GB/s** | 0.94 GB/s | 0.80 GB/s |
| .gguf | 2.37 GB/s | 0.92 GB/s | 1.57 / **2.52 GB/s**† |
| .onnx | **2.30 GB/s** | 0.82 GB/s | 0.81 GB/s |
| .h5 | **2.36 GB/s** | 0.95 GB/s | 1.47 GB/s |

Every format lands at roughly the same number because after projection every
format is the same thing: a mapped byte range described by one object model.

*Llama 3.2 1B shapes (~2.8 GB), Linux, i9-13900K, NVMe SSD, median of 5 cold
runs. ONNX at 1 GB (protobuf's limit). †Native zero-copy where the library has
it.*

## Writing is slower, and that is the trade

<p align="center">
  <img src="website/static/charts/write_throughput.svg" alt="Write throughput by workload" width="700">
</p>

| Format | Large | Mixed | Small |
| :--- | :--- | :--- | :--- |
| **.zt** | 3.29 GB/s | 3.62 GB/s | 0.80 GB/s |
| .safetensors | 5.18 GB/s | **6.27 GB/s** | 2.62 GB/s |
| .pt (pickle) | 5.91 GB/s | 6.03 GB/s | **2.86 GB/s** |
| .npz | 1.10 GB/s | 1.15 GB/s | 0.54 GB/s |
| .gguf | 4.78 GB/s | 6.25 GB/s | 1.30 GB/s |
| .onnx | 0.29 GB/s | 0.30 GB/s | 0.35 GB/s |
| **.h5** | **6.13 GB/s** | 5.96 GB/s | 0.28 GB/s |

A canonical write hashes every byte, pads each tensor out to 64 KiB and shares
one blob between tensors that turn out identical. That is paid once, when the
artifact is written, instead of on every load of it afterwards.

`Small` is where the padding shows: 51k tiny tensors each rounded up to a page
make a file 6.4× its payload. The same workload at the 4 KiB floor is
1.32 GB/s into 1.21×. [Alignment is a
tradeoff](https://pie-project.github.io/ztensor/benchmarks.html#alignment-is-a-tradeoff)
has the curve.

## Install

```bash
cargo add ztensor            # the format
cargo add ztensor-compat     # foreign-format projections
cargo install ztensor-cli    # the `zt` command
pip install ztensor          # Python bindings
```

Tested on Linux and macOS; Windows is unverified rather than supported.
[Details](https://pie-project.github.io/ztensor/formats.html#platform-support).

## Repository

| Path | What it is |
| --- | --- |
| `spec/` | The normative specification, and the profile registry |
| `ztensor/` | Core crate: reader, writer, validation, sharding |
| `ztensor-compat/` | Foreign-format projections, feature-gated per format |
| `ztensor-cli/` | The `zt` binary |
| `ztensor-py/` | Python bindings (abi3, Python ≥ 3.9) |
| `conformance/` | Golden corpus: 19 files a reader must accept, 57 it must reject |
| `fuzz/` | Fuzz targets: container, CBOR codec, all six parsers |
| `benchmark/` | The harness behind the numbers above |
| `website/docs/` | Documentation sources |

```bash
cargo test --all-features
```

## Status

Pre-1.0 in intent if not in number: the format is Draft 4 and may still
change.

## License

MIT
