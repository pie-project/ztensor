# zTensor

An aligned, verifiable container format for tensor data, and a loader that
reads the other common tensor formats into the same object model.

[Documentation](https://pie-project.github.io/ztensor/) ·
[Guide](https://pie-project.github.io/ztensor/guide.html) ·
[Specification](https://pie-project.github.io/ztensor/spec.html) ·
[Benchmarks](https://pie-project.github.io/ztensor/benchmarks.html)

## Design

A checkpoint is a few gigabytes of numbers that a machine has to load quickly
and check for damage. `.zt` is built around those two requirements.

Every tensor starts on a page boundary. An operating system maps, protects and
evicts memory a page at a time, so a tensor sharing a page with its neighbour
cannot be handled on its own. Canonical files give each tensor its own pages,
so weights can be streamed in and released one at a time.

Every tensor carries an XXH3 digest, and so does the manifest. A flipped bit
produces an error at load time.

Quantization schemes are registry entries, not format revisions. The container
deals in bytes and parts. What the bytes mean is a versioned profile name that
zTensor validates and passes through. An unregistered name is rejected.

The same tensors always produce the same file, so a file hash works as a model
identity. For comparing models that were packed differently there is a content
digest computed from the manifest alone.

Nothing in the format executes code.

## Format comparison

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

Blank cells are not a criticism. safetensors was built as a safe replacement
for pickle and does that job well; alignment and digests were not among its
goals. `.zt` was designed later, with those properties as requirements from
the start. The full table with footnotes is on the
[formats page](https://pie-project.github.io/ztensor/formats.html).

## Read throughput

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

The numbers cluster because after projection every format is a mapped byte
range described by one object model, so the read path is the same code in
every row.

*Llama 3.2 1B shapes (~2.8 GB). Linux, i9-13900K, NVMe SSD, median of 5 cold
runs. ONNX measured at 1 GB, which is protobuf's message limit. †Native
zero-copy where the library has it.*

## Write throughput

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

zTensor is slower here. A canonical write hashes every byte, pads each tensor
up to 64 KiB, and shares a single blob between tensors that turn out to be
identical. Those costs land once, when the artifact is written, and not on the
loads afterwards.

The `Small` column is where padding shows up: 51k tiny tensors each rounded up
to a page produce a file 6.4 times its payload. Writing the same workload at
the 4 KiB floor gives 1.32 GB/s into 1.21 times the payload. The
[benchmarks page](https://pie-project.github.io/ztensor/benchmarks.html#alignment-is-a-tradeoff)
has the curve.

## Install

```bash
cargo add ztensor            # the format
cargo add ztensor-compat     # foreign-format projections
cargo install ztensor-cli    # the `zt` command
pip install ztensor          # Python bindings
```

CI runs on Linux and macOS. Windows is unverified, not supported; the
[formats page](https://pie-project.github.io/ztensor/formats.html#platform-support)
explains what that means.

## Repository

| Path | What it is |
| --- | --- |
| `spec/` | The normative specification and the profile registry |
| `ztensor/` | Core crate: reader, writer, validation, sharding |
| `ztensor-compat/` | Foreign-format projections, feature-gated per format |
| `ztensor-cli/` | The `zt` binary |
| `ztensor-py/` | Python bindings (abi3, Python 3.9 and up) |
| `conformance/` | Golden corpus: 19 files a reader must accept, 57 it must reject |
| `fuzz/` | Fuzz targets for the container, the CBOR codec and all six parsers |
| `benchmark/` | The harness behind the numbers above |
| `website/docs/` | Documentation sources |

```bash
cargo test --all-features
```

## Versions

The crates share one number and are tagged `v*`. The Python package has its
own and is tagged `py-v*`, so a change confined to the bindings does not move
`ztensor` on crates.io. The file format is separate from both: `.zt` is
container version 2, spec Draft 4.

## Status

Pre-1.0 in intent if not in number. The format is at Draft 4 and may still
change.

## License

MIT
