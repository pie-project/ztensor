# zTensor File Format Specification

**Version 0.1.0**

## 1. Overview

zTensor is a binary file format for storing large multi-dimensional arrays, designed for safety, efficient read access, and flexibility.
It supports compressed encodings for tensors (using `zstd`), quantized formats and sparse tensors.

## 2. File Structure

A zTensor file has the following layout:

```
+--------------------------------------+
| [Magic Number]                       | <--- 8 bytes, e.g., "ZTEN0001"
+--------------------------------------+
| [Tensor Blob 0 Data]                 | <--- Starts at an aligned offset
+--------------------------------------+
| [Padding for Tensor Blob 0 (Optional)] | <--- 0 or more bytes to align next blob
+--------------------------------------+
| [Tensor Blob 1 Data]                 | <--- Starts at an aligned offset
+--------------------------------------+
| [Padding for Tensor Blob 1 (Optional)] |
+--------------------------------------+
| ...                                  |
+--------------------------------------+
| [Tensor Blob N-1 Data]               |
+--------------------------------------+
| [Padding for Tensor Blob N-1 (Optional)]|
+--------------------------------------+
| [CBOR Blob (Metadata Array)]         | <--- CBOR-encoded array of tensor metadata
+--------------------------------------+
| [Total CBOR Blob Size]               | <--- uint64_t, little-endian
+--------------------------------------+
```

## 3. Components

### 3.1. Magic Number

* **Purpose:** Identifies the file as a zTensor file and indicates the base format version.
* **Placement:** The first 8 bytes of the file (offset 0).
* **Value:** The ASCII string `"ZTEN0001"`.
* All subsequent `offset` values in the metadata must account for the 8-byte size of the magic number.

### 3.2. Tensor Data Section

This section contains the binary data for all tensors.

* **Tensor Blobs (`tensor_blob`):**
    * Each `tensor_blob` is a contiguous block of binary data representing a single tensor.
    * The specific binary representation of the data within the blob (e.g., raw, compressed) is defined by the `encoding` field in its corresponding metadata entry.
    * No per-blob metadata or headers are stored within the `tensor_blob` itself.

* **Alignment & Padding:**
    * **Requirement:** Each `tensor_blob` **must** start at an offset that is a multiple of the defined alignment value of **64 bytes**.
    * **Padding:** If the preceding component (magic number or previous tensor blob + its own padding) does not naturally end at a 64-byte aligned boundary for the next tensor, padding bytes (value undefined, typically zero) must be inserted. The `offset` field in the metadata will always point to the 64-byte aligned start of the tensor data.

### 3.3. Index

The index is located at the very end of the file and provides metadata for all tensors. It consists of two parts:

* **3.3.1. CBOR Blob (Metadata Array):**
    * **Format:** A single CBOR-encoded array. Each element in the array is a CBOR map object, where each map represents the metadata for one tensor in the file.
    * The order of metadata objects in this array typically corresponds to the order of `tensor_blob`s in the file, though this is not strictly required as each metadata object contains an absolute offset.

* **3.3.2. Total CBOR Blob Size:**
    * **Purpose:** Specifies the total size in bytes of the immediately preceding `CBOR Blob (Metadata Array)`.
    * **Format:** A single `uint64_t` (8-byte unsigned integer).
    * **Endianness:** Little Endian.
    * **Placement:** The last 8 bytes of the zTensor file.

To read the index, a parser first reads the `Total CBOR Blob Size` (last 8 bytes), then seeks backwards by that amount from the end of the `Total CBOR Blob Size` field to read the `CBOR Blob` itself.

## 4. Tensor Metadata Object Structure

Each element in the `CBOR Blob` array is a CBOR map containing key-value pairs describing a tensor. All keys are CBOR text strings.

**Required Fields:**

* **`name`** (string): A UTF-8 string representing the name of the tensor.
* **`offset`** (uint): A `uint64_t` representing the absolute byte offset from the beginning of the file to the start of this tensor's `tensor_blob`. This offset must be a multiple of 64.
* **`size`** (uint): A `uint64_t` representing the on-disk size in bytes of this tensor's `tensor_blob` (e.g., if compressed, this is the compressed size).
* **`dtype`** (string): A string identifying the data type of the elements in the tensor. See Section 5.1.
* **`shape`** (array): A CBOR array of `uint64_t` integers representing the dimensions of the tensor. An empty array `[]` denotes a scalar.
* **`encoding`** (string): A string identifying the encoding used for the `tensor_blob` data. See Section 5.2.

**Optional Fields:**

* **`data_endianness`** (string): Specifies the endianness of the tensor data itself if `encoding` is `"raw"` and `dtype` is a multi-byte type. Valid values are `"little"` or `"big"`. If this field is absent under these conditions, the tensor data is assumed to be **Little Endian**.
* **`checksum`** (string): A string representing a checksum of the `tensor_blob` data, for integrity verification.
    * **Format Example:** `"crc32c:0x1234ABCD"` or `"sha256:..."`. The specific algorithm and representation of the checksum value should be agreed upon by producers and consumers.

**Custom Fields:**
* Users can include any other custom key-value pairs in this CBOR map for additional metadata. Parsers must ignore unknown fields to maintain forward compatibility.

## 5. Field Definitions

### 5.1. `dtype` (Data Type)

The `dtype` string specifies the element type of the tensor. Supported values (case-sensitive strings) for this version include:
* `"float64"`: IEEE 754 double-precision floating-point.
* `"float32"`: IEEE 754 single-precision floating-point.
* `"float16"`: IEEE 754 half-precision floating-point.
* `"bfloat16"`: BFloat16 floating-point.
* `"int64"`: Signed 64-bit integer.
* `"int32"`: Signed 32-bit integer.
* `"int16"`: Signed 16-bit integer.
* `"int8"`: Signed 8-bit integer.
* `"uint64"`: Unsigned 64-bit integer.
* `"uint32"`: Unsigned 32-bit integer.
* `"uint16"`: Unsigned 16-bit integer.
* `"uint8"`: Unsigned 8-bit integer.
* `"bool"`: Boolean (typically represented as 1 byte per value, where non-zero is true).

### 5.2. `encoding`

The `encoding` string specifies how the tensor data in the `tensor_blob` is stored.
* `"raw"`: Data is stored as a direct binary dump of the tensor elements in their native `dtype` format and specified `data_endianness`.
* `"zstd"`: Data is compressed using Zstandard. The `size` field refers to the compressed size.

## 6. Endianness

* **Index Fields:** The `Total CBOR Blob Size` field is **Little Endian**. CBOR itself handles endianness for its encoded numbers internally.
* **Tensor Data (`tensor_blob`):** For `encoding: "raw"` and multi-byte `dtype`s (e.g., `float32`, `int64`):
    * The data is **Little Endian** by default.
    * The optional `data_endianness` metadata field can explicitly specify `"little"` or `"big"`. If this field is absent, Little Endian is assumed.

## 7. Zero-Tensor Files

A zTensor file containing zero tensors is valid.
* The file starts with the 8-byte Magic Number (`"ZTEN0001"`).
* The `CBOR Blob (Metadata Array)` will be an empty CBOR array (represented as `0x80` in CBOR binary, which is 1 byte).
* The `Total CBOR Blob Size` will be `1` (stored as a little-endian `uint64_t`: `0x0100000000000000`).
* Thus, a minimal valid zTensor file with zero tensors will be 8 (magic) + 1 (empty CBOR array) + 8 (size field) = 17 bytes.

## 8. Extensibility

* New `dtype` and `encoding` string values may be defined in future versions of this specification. Consumers should be prepared to handle unknown string values gracefully (e.g., by skipping the tensor or reporting an unsupported feature).
* Custom key-value pairs can be added to the tensor metadata objects. Consumers must ignore keys they do not understand.
* Future versions of the zTensor specification might define new well-known optional fields or modify the Magic Number.

---
This version of the spec is more assertive about the design choices based on the consolidated recommendations.