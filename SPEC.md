# zTensor File Format Specification

**Version:** 1.0.0

**Status:** Draft Standard

**File Extension:** `.zt` (recommended)

## 1\. Design Philosophy

zTensor v1.0 separates **Physical Storage** (Components) from **Logical Tensors**.

  * **Physical Layer:** A flat list of named, aligned, binary blobs (Components).
  * **Logical Layer:** A metadata Manifest that describes how to assemble these components into high-level structures (Dense Tensors, Sparse Matrices, Graphs).

This decoupling ensures that the file format does not need to change when new AI architectures (e.g., Ragged Tensors, Quantized LoRA adapters) are invented.

## 2\. Global Constants

  * **Endianness:** **Little-Endian (LE)** only. All integers and multi-byte data types must be written in LE.
  * **Alignment:** **64 bytes**. All binary components must start at an offset divisible by 64.
  * **Padding:** **Zero (`0x00`)**. All padding bytes used for alignment must be zero to prevent data leakage.
  * **Memory Order:** **Row-major (C-contiguous)**. Dense tensors are stored in row-major order.

## 3\. File Structure

The file is written as an append-only stream. The index (Manifest) is located at the end to support streaming writes.

```text
+---------------------------------------+
| Magic Number (8 bytes)                |
+---------------------------------------+ <--- Offset 8
|                                       |
| Component Blob A (Aligned 64B)        |
|                                       |
+---------------------------------------+
| Zero Padding (0-63 bytes)             |
+---------------------------------------+
|                                       |
| Component Blob B (Aligned 64B)        |
|                                       |
+---------------------------------------+
| ...                                   |
+---------------------------------------+
| CBOR Manifest (Metadata)              |
+---------------------------------------+
| Manifest Size (8 bytes, uint64 LE)    |
+---------------------------------------+ <--- EOF
```

### 3.1 Header & Footer

| Field | Size | Type | Value / Description |
| :--- | :--- | :--- | :--- |
| **Magic Number** | 8 bytes | ASCII | `ZTEN1000` (Version 1.0) |
| **Manifest Size** | 8 bytes | uint64 | Size of the CBOR payload in bytes. |

-----

## 4\. The Manifest (Metadata)

The metadata is a **CBOR-encoded Map** located at the end of the file. Unlike v0.1, this is a **Manifest Object**, not a list.

### 4.1 Root Structure

```json
{
  "version": "1.0",
  "generator": "zTensor-Rust v0.1.0",
  "attributes": {
    "license": "Apache-2.0",
    "description": "ResNet-50 Checkpoint",
    "framework": "PyTorch",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "tensors": {
    "layer1.weight": { ... },
    "layer1.bias": { ... }
  }
}
```

#### Recommended Attribute Keys

| Key | Description |
| :--- | :--- |
| `license` | License identifier (e.g., `Apache-2.0`, `MIT`) |
| `description` | Human-readable description of the file contents |
| `framework` | Source framework (e.g., `PyTorch`, `TensorFlow`, `JAX`) |
| `model_name` | Name of the model (e.g., `ResNet-50`, `Llama-3-8B`) |
| `created_at` | ISO 8601 timestamp of file creation |

### 4.2 Tensor Object Schema

Each value in the `tensors` map represents a **Logical Tensor**.

| Field | Type | Description |
| :--- | :--- | :--- |
| `dtype` | string | Data type (`float32`, `float16`, `float64`, `bfloat16`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bool`) |
| `shape` | array | Dimensions, e.g., `[1024, 768]`. |
| `format` | string | The schema used to assemble components (e.g., `dense`, `sparse_csr`, `sparse_coo`). |
| `components` | map | Key-Value pairs mapping **Component Roles** to **Physical Locations**. |

### 4.3 Component Object Schema

Describes a physical byte range in the file.

| Field | Type | Description |
| :--- | :--- | :--- |
| `offset` | uint64 | Absolute file offset (Must be multiple of 64). |
| `length` | uint64 | Size of the data in bytes (compressed size if encoded). |
| `encoding` | string | Optional. `raw` (default) or `zstd`. |
| `digest` | string | Optional. Checksum in format `algorithm:value`. Supported algorithms: `crc32c` (e.g., `crc32c:0x1A2B3C4D`), `sha256` (e.g., `sha256:8f4a...`). |

-----

## 5\. Logical Layouts (Schemas)

The power of v1.0 lies in the `format` field. Parsers switch behavior based on this string.

### 5.1 Format: `dense`

Standard contiguous array in **row-major (C-contiguous)** order.

  * **Required Components:** `data`
  * **Description:** The raw binary dump of the tensor values.

**Example Manifest Entry:**

```json
"embedding_table": {
  "dtype": "float32",
  "shape": [5000, 256],
  "format": "dense",
  "components": {
    "data": { "offset": 64, "length": 5120000 }
  }
}
```

### 5.2 Format: `sparse_csr` (Compressed Sparse Row)

Common for sparse matrices in scientific computing.

  * **Required Components:**
    1.  `values`: Non-zero elements (dtype as specified in tensor).
    2.  `indices`: Column indices for values (`uint64`, 8 bytes per index).
    3.  `indptr`: Row pointers (`uint64`, 8 bytes per pointer, length = rows + 1).

**Example Manifest Entry:**

```json
"adjacency_matrix": {
  "dtype": "float32",
  "shape": [1000, 1000],
  "format": "sparse_csr",
  "components": {
    "values":  { "offset": 64, "length": 400 },
    "indices": { "offset": 512, "length": 400 },
    "indptr":  { "offset": 960, "length": 8008 }
  }
}
```

### 5.3 Format: `sparse_coo` (Coordinate List)

Common for PyTorch sparse tensors.

  * **Required Components:**
    1.  `values`: Non-zero elements (dtype as specified in tensor).
    2.  `coords`: Coordinate matrix (`uint64`, 8 bytes per index).

**Coordinate Storage Order:**
The `coords` component stores a flattened matrix of shape $(ndim \times nnz)$ in **row-major order**:
  - First `nnz` values are indices for dimension 0
  - Next `nnz` values are indices for dimension 1
  - And so on for each dimension

**Example Manifest Entry:**

```json
"sparse_embeddings": {
  "dtype": "float32",
  "shape": [10000, 512],
  "format": "sparse_coo",
  "components": {
    "values": { "offset": 64, "length": 4000 },
    "coords": { "offset": 4096, "length": 16000 }
  }
}
```

In this example, `coords` contains 2000 `uint64` values (2 dimensions × 1000 non-zeros).

-----

## 6\. Parsing Algorithm

To read a zTensor v1.0 file:

1.  **Read Footer:** Seek to `FILE_SIZE - 8`. Read 8 bytes as `uint64` (Little Endian) to get `manifest_size`.
2.  **Validate Size:** Verify `manifest_size < FILE_SIZE - 16` to prevent overflow.
3.  **Read Manifest:** Seek to `FILE_SIZE - 8 - manifest_size`. Read `manifest_size` bytes.
4.  **Decode:** Parse bytes as CBOR.
5.  **Check Version:** Compare `version` field. See §6.1 for version negotiation rules.
6.  **Select Tensor:** Look up the desired tensor by name in the `tensors` map.
7.  **Check Format:**
      * If `dense`: Locate the `data` component.
      * If `sparse_...`: Locate the specific components required by that format.
8.  **Retrieve Data:**
      * Seek to `component.offset`.
      * Read `component.length` bytes.
      * If `encoding == "zstd"`, decompress.
      * If `encoding == "raw"` or absent, cast bytes to `dtype` (using Little Endian).

### 6.1 Version Negotiation

Versioning follows **Semantic Versioning** principles:

| File Version | Reader Behavior |
| :--- | :--- |
| Same major & minor (e.g., `1.0` ↔ `1.0`) | ✅ Proceed normally |
| Same major, higher minor (e.g., file `1.1`, reader `1.0`) | ⚠️ Warn, proceed (may miss new features) |
| Same major, lower minor (e.g., file `1.0`, reader `1.1`) | ✅ Proceed normally |
| Different major (e.g., file `2.0`, reader `1.x`) | ❌ Fail with version error |

-----

## 7\. Implementation Guidelines

### 7.1 Security

  * Readers **MUST** verify that `offset + length <= FILE_SIZE`.
  * Readers **MUST** verify that `manifest_size < FILE_SIZE - 16` (magic header + footer).
  * Readers **SHOULD** warn if `offset` is not 64-byte aligned (performance warning).
  * Readers **SHOULD** validate checksums when `digest` is present.

### 7.2 Memory Mapping

For `dense` tensors with `encoding: "raw"`, readers should prefer **memory mapping (mmap)**. The 64-byte alignment guarantees that the pointer returned by mmap is valid for SIMD operations (AVX-512) without copying.

### 7.3 Extensions

New formats (e.g., `quantized_4bit`) can be added simply by defining which components they require (e.g., `data`, `scale`, `zero_point`), without changing the file version.

**Example: Quantized 4-bit Format**

```json
"quantized_layer": {
  "dtype": "uint8",
  "shape": [4096, 4096],
  "format": "quantized_4bit",
  "components": {
    "data": { "offset": 64, "length": 8388608 },
    "scale": { "offset": 8388672, "length": 16384 },
    "zero_point": { "offset": 8405056, "length": 16384 }
  }
}
```