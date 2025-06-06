# zTensor File Format

**Version 0.1.0**

zTensor is a binary format for storing large multi-dimensional arrays (tensors), designed for efficient, safe, and flexible access. It supports raw and compressed (zstd) encodings, quantized and sparse layouts, and is extensible.

For dense tensors stored with encoding: "raw" and matching endianness, zTensor enables zero-copy access to tensor data.

## File Layout

```
+-------------------------------+
| Magic Number (8 bytes)        |
+-------------------------------+
| Tensor Blob 0 (aligned)       |
+-------------------------------+
| Padding (if needed)           |
+-------------------------------+
| Tensor Blob 1 (aligned)       |
+-------------------------------+
| ...                           |
+-------------------------------+
| CBOR Metadata Array           |
+-------------------------------+
| CBOR Array Size (8 bytes)     |
+-------------------------------+
```

- **Magic Number:** ASCII "ZTEN0001" at offset 0.
- **Tensor Blobs:** Each tensor's data, starting at a 64-byte aligned offset. No per-blob headers. Padding (undefined value, usually zero) is inserted as needed.
- **CBOR Metadata Array:** At the end of the file, a CBOR-encoded array of metadata maps (one per tensor).
- **CBOR Array Size:** Last 8 bytes, little-endian uint64, gives the size of the CBOR array.

## Tensor Metadata (CBOR Map)
Each tensor's metadata is a CBOR map with these required fields:
- `name` (string): Tensor name.
- `offset` (uint64): Absolute file offset to tensor data (multiple of 64).
- `size` (uint64): On-disk size in bytes (compressed if applicable).
- `dtype` (string): Data type (see below).
- `shape` (array): Array of dimensions (empty for scalar).
- `encoding` (string): Data encoding (see below).
- `layout` (string): "dense" (default) or "sparse". For sparse, see below.

Optional fields:
- `data_endianness` (string): "little" or "big" (default: little, for raw multi-byte types).
- `checksum` (string): e.g., "crc32c:0x1234ABCD" or "sha256:...".
- Custom fields are allowed; unknown keys are ignored by readers.

## Supported Data Types (`dtype`)
- `float64`, `float32`, `float16`, `bfloat16`
- `int64`, `int32`, `int16`, `int8`
- `uint64`, `uint32`, `uint16`, `uint8`
- `bool`

## Supported Encodings (`encoding`)
- `raw`: Direct binary dump of tensor elements (with `data_endianness` if multi-byte).
- `zstd`: Zstandard-compressed data. `size` is compressed size.

## Layouts
- `dense` (default): Standard contiguous tensor data.
- `sparse`: Data is stored in a sparse format. The metadata map must include a `sparse_format` field (e.g., "csr", "coo") and any additional fields required to describe the sparse structure (such as index arrays, indptr, etc.).

## Index Reading
To read the index:
1. Read the last 8 bytes for the CBOR array size.
2. Seek backwards by that amount to read the CBOR metadata array.
3. 
As a result, the start offset of the metadata is: (file size) - (size of the metadata) - (8 byte).

## Zero-Tensor Files
A valid zTensor file may contain zero tensors:
- 8 bytes magic, 1 byte empty CBOR array (`0x80`), 8 bytes size (`0x01...00`).
- Total: 17 bytes.

## Extensibility
- New `dtype`, `encoding`, and `layout` values may be added in future versions.
- Custom metadata fields are allowed; unknown fields are ignored.

## Notes
- All offsets are absolute and account for the 8-byte magic number.
- All tensor data blobs are 64-byte aligned.
- No per-blob headers; all metadata is in the CBOR array at the end of the file.
- For `encoding: "raw"` and multi-byte `dtype`, data is little-endian unless `data_endianness` is specified.
- For sparse tensors, the metadata must fully describe the sparse structure.
