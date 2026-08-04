---
sidebar_position: 3
---

# Formats

Seven formats can be read, one can be written. The side-by-side grid of what
each can do is on the [front page](./intro.md#comparison); this page has the
detail behind it.

## Reading

| Format | Cargo feature | What the projection does |
| --- | --- | --- |
| `.zt` | n/a | Native, including sharded models and overlays |
| `.safetensors` | `safetensors` (default) | Validates that the tensors tile the data region exactly, which defuses header aliasing |
| `.gguf` | `gguf` (default) | Keeps quantized blocks verbatim, named `gguf.<type>/1` |
| `.npz` / `.npy` | `npz` | Refuses big-endian and Fortran-order arrays instead of reinterpreting them |
| `.pt` / `.bin` | `pickle` | Restricted VM that extracts tensor metadata and executes nothing; non-contiguous tensors refused |
| `.h5` / `.hdf5` | `hdf5` | Contiguous and chunked datasets, deflate and shuffle filters |
| `.onnx` | `onnx` | Graph initializers; external data refused |

`pickle` is opt-in. Parsing pickle at all is a wider attack surface than
everything else here put together.

## Writing

Only `.zt`. Tracking every foreign format as a producer would be unbounded
work that no consumer asked for. The reason to read them is to get models out
of them, and the reason to write is to get a file with alignment and digests.

## Platform support

Developed and tested on Linux and macOS, the two platforms CI runs.

The crate has no unix-only dependency. Its `cfg(not(unix))` path is compiled
on every push against `wasm32-wasip1`, the cheapest non-unix target available,
and the behaviour it stands in for is tested on unix. Windows is not tested,
so treat it as unverified.

Two capabilities are unix-only because they have no equivalent elsewhere:
`prefetch` (`madvise(WILLNEED)`) and `evict` (`madvise(DONTNEED)`). Reading,
writing, mapping, addressing and verification are all portable.

## Testing

The parsers here read files from wherever a model came from, so the contract
is explicit: hostile input yields an error, never a panic, an unbounded
allocation, or a fabricated tensor.

A conformance corpus of 76 files, 19 a reader must accept and 57 it must
reject, is regenerated from code and diffed in CI. A change to the reader that
changes what it accepts fails the build, rather than surfacing in whichever
reader tries the file next.

Fuzz targets cover the container, the CBOR codec and all six foreign parsers.

`ztensor-compat/tests/hostile.rs` holds the reproducer behind every hardening
fix, so a fix cannot silently come undone.

```bash
cargo test --all-features
cargo run -p conformance --bin gen        # regenerate the golden files
cargo +nightly fuzz run fuzz_compat       # the foreign-format parsers
```

## Dependencies

`memmap2`, `xxhash-rust`, `unicode-normalization` and `libc`, plus `zstd` when
the seekable-compression profile is enabled.

The CBOR codec is 418 lines in-tree rather than a dependency. The spec
restricts CBOR hard enough (no tags, eight value types, deterministic encoding
mandatory) that owning it comes out smaller than a general-purpose one, and it
lets determinism be structural instead of a flag someone can forget.

A minimal conforming reader is
[`ztensor/examples/minimal_reader.rs`](https://github.com/pie-project/ztensor/blob/main/ztensor/examples/minimal_reader.rs),
at 68 lines, needing only a CBOR decoder and XXH3.
