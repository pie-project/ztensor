---
sidebar_position: 6
---

# File Format Specification

**Version:** 1.2.0 &middot; **Extension:** `.zt`

## Motivation

Existing tensor formats each solve part of the problem, but none solve it cleanly:

- **Pickle-based formats** (PyTorch `.pt`, `.bin`) execute arbitrary code during loading. This is a fundamental security flaw: a model file can run anything on the reader's machine. No amount of sandboxing makes deserializing code safe by default.
- **SafeTensors** solved the security problem, but treats every tensor as a flat, dense array of a fixed dtype. There is no way to represent sparse matrices, quantized weight groups, or newer numeric types like FP8 without extending the format itself. Adding a new dtype requires a spec change and a library update on both sides.
- **GGUF** handles quantization well but bakes quantized types into the dtype enum. Each new quantization scheme requires a new dtype constant, and the format is tightly coupled to the llama.cpp ecosystem.
- **NumPy `.npz`** is simple and widely supported, but has no alignment guarantees (no mmap), no compression options beyond zip, and no room for structured metadata.

The common thread is that most formats equate "tensor" with "flat array of one dtype." The moment you need a tensor that is *structurally* more complex (sparse indices alongside values, or quantized weights alongside scales), the format either can't express it or forces you to flatten everything into separate, unrelated arrays with naming conventions gluing them together.

## Design Goals

zTensor starts from a different premise: a tensor is a composite object, not a flat buffer. The format is built around three principles:

- **Simple.** The file is a flat sequence of data blobs followed by a single metadata index. No nested containers, no back-patching, no code execution. A minimal reader is a few dozen lines.
- **Performant.** Data blobs are 64-byte aligned for direct memory-mapping and SIMD access. Metadata is separated from bulk data, so a reader can enumerate every object's name, shape, and type without touching any data bytes.
- **Extensible.** New object layouts, data types, and metadata fields can be added without breaking existing readers. The format uses optional fields and an open type system, not a fixed enum, so it evolves without version bumps.

## Key Concepts

A `.zt` file contains named **objects**. An object is the format's abstraction for a tensor: instead of treating a tensor as a flat array of one dtype, an object is a *composite* with a `shape`, a `format` that describes its layout, and one or more **components** that hold the actual bytes.

A component is a single contiguous blob on disk. It has a **storage type** (`dtype`) that determines byte width, and an optional **logical type** (`type`) that says what those bytes *mean*. For a standard float32 object, `dtype` is `"f32"` and `type` is absent, so no interpretation is needed. For an FP8 object, `dtype` is `"u8"` (because FP8 is stored as raw bytes) while `type` is `"f8_e4m3fn"` (telling the reader how to decode them). This separation keeps the storage layer stable (readers always know how many bytes to read) while the logical layer can grow to cover new numeric formats without any changes to the container.

This design makes a dense object simple (one component named `"data"`), but also supports sparse matrices (separate `values`, `indices`, `indptr` components) and quantized weights (separate `packed_weight`, `scales`, `zeros` components) without special-casing in the container format. The object knows its own structure.

The rest of this document is organized from abstract to concrete: the manifest schema (what you write), the type system (how bytes are interpreted), the object formats (what layouts exist), and finally the binary layout (how it's arranged on disk).

## 1. Manifest Schema

The manifest is a CBOR-encoded map (RFC 7049) stored near the end of the file. Its location and how to find it are described in [Binary Layout](#4-binary-layout).

### Root

```json
{
  "version": "1.2.0",
  "attributes": {
    "framework": "PyTorch",
    "license": "Apache-2.0"
  },
  "objects": {
    "layer1.weight": { "..." : "..." },
    "layer1.bias": { "..." : "..." }
  }
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `version` | string | Yes | Spec version (e.g., `"1.2.0"`). |
| `attributes` | map | No | Arbitrary key-value metadata for the whole file. |
| `objects` | map | Yes | Named object definitions. Keys are object names (e.g., `"layer1.weight"`). |

### Object

Each entry in `objects` describes one logical object.

| Field | Type | Description |
| --- | --- | --- |
| `shape` | `[uint64]` | Dimensions (e.g., `[1024, 768]`). |
| `format` | string | Layout: `dense`, `sparse_csr`, `sparse_coo`, `quantized_group`, etc. See [Object Formats](#3-object-formats). |
| `attributes` | map | Optional per-object metadata. |
| `components` | map | One entry per data blob. Keys are role names (e.g., `"data"`, `"scales"`). |

### Component

A component points to a single contiguous blob on disk and describes how to interpret its bytes.

```json
{
  "dtype": "u8",
  "type": "f8_e4m3fn",
  "offset": 1024,
  "length": 4096,
  "encoding": "raw",
  "digest": "sha256:8f4a..."
}
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `dtype` | string | *required* | Storage type, one of the 13 fixed primitives. Determines byte width. |
| `type` | string | `null` | Logical type. When absent, the logical type equals `dtype`. |
| `offset` | uint64 | *required* | Absolute byte offset in the file. Must be a multiple of 64. |
| `length` | uint64 | *required* | Bytes on disk. For `"zstd"` encoding, this is the **compressed** size. |
| `uncompressed_length` | uint64 | `null` | Original size before compression. Required when `encoding` is `"zstd"`. |
| `encoding` | string | `"raw"` | `"raw"` or `"zstd"`. |
| `digest` | string | `null` | Checksum of the stored (possibly compressed) bytes. Format: `"algorithm:hex"` (e.g., `"sha256:8f4a..."`). |

## 2. Type System

### Storage types (`dtype`)

A closed set of 13 hardware-native primitives. Every component must use one of these as its `dtype`. The `dtype` alone determines the byte width of each element.

| Category | Types | Encoding |
| --- | --- | --- |
| Float | `f64`, `f32`, `f16`, `bf16` | IEEE 754 / BFloat16 |
| Signed integer | `i64`, `i32`, `i16`, `i8` | Two's complement |
| Unsigned integer | `u64`, `u32`, `u16`, `u8` | Unsigned |
| Boolean | `bool` | 1 byte. `0x00` = false, `0x01` = true. |

### Logical types (`type`)

An open, extensible set that gives meaning to raw `dtype` bytes. When `type` is absent, the logical type is the same as `dtype`, and no extra interpretation is needed.

**Simple types:** one storage element per logical element (1:1):

| `type` | `dtype` | Notes |
| --- | --- | --- |
| `f8_e4m3fn` | `u8` | NVIDIA / OCP FP8 |
| `f8_e5m2` | `u8` | OCP FP8 |
| `f8_e4m3fnuz` | `u8` | AMD FP8 |
| `f8_e5m2fnuz` | `u8` | AMD FP8 |

**Compound types:** multiple storage elements per logical element:

| `type` | `dtype` | Ratio | Notes |
| --- | --- | --- | --- |
| `complex64` | `f32` | 2:1 | Interleaved `[real, imag]` pairs |
| `complex128` | `f64` | 2:1 | Interleaved `[real, imag]` pairs |

**Computing data size:**

- Simple: `product(shape) * byte_size(dtype)`
- Compound: `product(shape) * ratio * byte_size(dtype)`

Readers that encounter an unrecognized `type` MAY fall back to loading raw `dtype` elements.

## 3. Object Formats

Each object's `format` field selects how its components are interpreted. All index components (`indices`, `indptr`, `coords`) across sparse formats MUST use `dtype: "u64"`.

### `dense`

Standard contiguous array in row-major (C-contiguous) order.

| Component | Description |
| --- | --- |
| `data` | The data elements. |

Readers SHOULD memory-map `data` when `encoding` is `"raw"`.

### `sparse_csr`

Compressed Sparse Row.

| Component | Description |
| --- | --- |
| `values` | Non-zero elements. |
| `indices` | Column index for each value. |
| `indptr` | Row pointers (length = rows + 1). |

### `sparse_coo`

Coordinate list.

| Component | Description |
| --- | --- |
| `values` | Non-zero elements. |
| `coords` | Coordinate indices, stored as a flat array of length `ndim * nnz` in Structure-of-Arrays order: all row indices first, then all column indices, etc. |

### `quantized_group`

Block-wise quantization (e.g., GPTQ). Packed weights with separate scale and zero-point arrays.

| Component | Description |
| --- | --- |
| `packed_weight` | Quantized data (e.g., `i32` packing 8 x 4-bit values). |
| `scales` | Per-group scaling factors. |
| `zeros` | Per-group zero-points. |

Quantization parameters (`bits`, `group_size`, `packing`) are stored in the object's `attributes`.

**Example:** 4-bit GPTQ, shape `[4096, 4096]`:

```json
{
  "shape": [4096, 4096],
  "format": "quantized_group",
  "attributes": {
    "bits": 4,
    "group_size": 128,
    "packing": "8_per_i32"
  },
  "components": {
    "packed_weight": { "dtype": "i32", "offset": 1024,    "length": 8388608 },
    "scales":        { "dtype": "f16", "offset": 8389632, "length": 262144 },
    "zeros":         { "dtype": "f16", "offset": 8651776, "length": 262144 }
  }
}
```

## 4. Binary Layout

The on-disk format is append-only: data blobs are written sequentially, then the manifest is appended at the end. This makes writing simple (no seeking back to patch headers) and keeps all metadata in one place.

### File structure

```text
+---------------------------------------+ <-- Offset 0
| Magic Header (8 bytes)                |     "ZTEN1000"
+---------------------------------------+
|                                       |
| Component Blob A                      | <-- offset % 64 == 0
|                                       |
+---------------------------------------+
| Zero Padding (0-63 bytes)             |
+---------------------------------------+
| Component Blob B                      | <-- offset % 64 == 0
+---------------------------------------+
| ...                                   |
+---------------------------------------+
| CBOR Manifest (variable length)       |
+---------------------------------------+
| Manifest Size (8 bytes, uint64 LE)    |
+---------------------------------------+
| Magic Footer (8 bytes)                |     "ZTEN1000"
+---------------------------------------+ <-- EOF
```

### Byte order

All multi-byte values in a `.zt` file are **Little-Endian**: structural integers (manifest size), component data, and all `dtype` elements. Writers on big-endian hosts MUST byte-swap before writing.

The only exception is CBOR's own internal length prefixes, which are Big-Endian per RFC 7049. This is handled transparently by any CBOR library.

### Alignment and padding

- Every component blob starts at an offset divisible by **64**. This enables direct memory-mapping and SIMD access without copying.
- Gaps between blobs are filled with `0x00` bytes.
- The magic footer repeats the header (`ZTEN1000`) so readers can detect truncated files.

## 5. Reading a File

### Algorithm

1. Seek to `EOF - 16`. Read 16 bytes.
2. Verify the last 8 bytes are `ZTEN1000`. If not, the file is corrupt or not a `.zt` file.
3. Decode the first 8 bytes as `uint64 LE` to get `manifest_size`.
4. If `manifest_size > 1 GB`, abort (prevents denial-of-service via oversized manifests).
5. Seek to `EOF - 16 - manifest_size`. Read `manifest_size` bytes.
6. Decode the buffer as CBOR to get the manifest.
7. To load a component:
   - Seek to `component.offset`.
   - Read `component.length` bytes.
   - If `encoding` is `"zstd"`, decompress (using `uncompressed_length` to pre-allocate).
   - Interpret the resulting bytes as `component.dtype` elements.

### Security rules

- **No code execution.** Parsers MUST NOT evaluate or execute any data. No pickle, no eval.
- **Bounds checking.** `offset + length` MUST NOT exceed the file size.
- **Decompression limits.** When `uncompressed_length` is present, reject values exceeding a reasonable maximum before decompressing.
- **Padding bytes.** Writers MUST set all padding to `0x00`. Readers MAY ignore padding content.

## Appendix: Version Policy

Minor version increments (e.g., 1.1 to 1.2) only add optional fields or new logical types. Readers MUST ignore unknown fields. Major version increments (e.g., 1.x to 2.x) may change the container structure or manifest schema.
