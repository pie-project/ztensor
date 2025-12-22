# zTensor Specification

**Version:** 1.1.0  
**Extension:** `.zt`

## Part I: The Container (Physical Layer)

The zTensor Container is an append-only, binary-safe envelope. It handles storage, alignment, and compression but is agnostic to the logical meaning of the data.

### 1. File Structure

The file is a stream of aligned binary blobs followed by a metadata index.

```text
+---------------------------------------+ <--- Offset 0
| Magic Header (8 bytes)                | ASCII: "ZTEN1000"
+---------------------------------------+
|                                       |
| Component Blob A (Aligned 64B)        | <--- Offset % 64 == 0
|                                       |
+---------------------------------------+
| Zero Padding (0-63 bytes)             | <--- Must be 0x00
+---------------------------------------+
| ... (Additional Blobs)                |
+---------------------------------------+
| CBOR Manifest (Variable Size)         | <--- Standard CBOR (RFC 7049)
+---------------------------------------+
| Manifest Size (8 bytes, uint64 LE)    |
+---------------------------------------+
| Magic Footer (8 bytes)                | ASCII: "ZTEN1000"
+---------------------------------------+ <--- EOF
```

### 2. Global Constants & Endianness

* **Structural Integers:** Little-Endian (LE)
  * Applies to: `Manifest Size` (at EOF) and binary blob contents (unless specified otherwise by `dtype`).

* **Manifest Encoding:** CBOR (RFC 7049)
  * Applies to: The internal structure of the metadata map.
  * *Note:* CBOR uses Network Byte Order (Big-Endian) for its internal length prefixes.

* **Alignment:** 64 bytes
  * All binary components must start at an offset divisible by 64.

* **Padding:** Zero (`0x00`)
  * **Tail Protection:** The footer repeats the Magic Number to prevent crashes on truncated files.

## Part II: The Manifest (Metadata Layer)

The Manifest is a **CBOR-encoded Map** located at `EOF - 16 - manifest_size`.

### 1. Root Object

```json
{
  "version": "1.1.0",
  "attributes": {
    "framework": "PyTorch",
    "license": "Apache-2.0"
  },
  "objects": {
    "layer1.weight": { ... },
    "layer1.bias": { ... }
  }
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `version` | string | **Yes** | The zTensor spec version (e.g., "1.2.0"). |
| `attributes` | map | No | Global metadata (key-value pairs). |
| `objects` | map | **Yes** | The map of logical tensor definitions. |

### 2. The Logical Object (Tensor)

Defines the high-level geometry.

| Field | Type | Description |
| --- | --- | --- |
| `shape` | array | Logical dimensions (e.g., `[1024, 768]`). |
| `format` | string | Layout schema (`dense`, `sparse_csr`, `quantized_group`, etc.). |
| `components` | map | Mapping of **Roles** to **Component Definitions**. |

### 3. The Component Definition

Defines the physical storage, typing, and encoding of a specific blob.

```json
{
  "dtype": "f8_e4m3",
  "offset": 1024,
  "length": 4096,
  "encoding": "raw",
  "digest": "sha256:8f4a..."
}
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `dtype` | string | **Req** | The primitive type of the data (see §2.4). |
| `offset` | uint64 | **Req** | Absolute file offset (Must be multiple of 64). |
| `length` | uint64 | **Req** | Size of the stored data in bytes. |
| `encoding` | string | `"raw"` | `"raw"` (default) or `"zstd"`. |
| `digest` | string | `null` | Optional checksum of the **content only** (ignoring padding). |

### 4. Supported Data Types (`dtype`)

| Category | Types | Storage Notes |
| --- | --- | --- |
| **Floats** | `f64`, `f32`, `f16`, `bf16` | IEEE 754 Little-Endian. |
| **Float8** | `f8_e4m3`, `f8_e5m2` | OCP/NVIDIA Standard. Little-Endian. |
| **Complex** | `complex64`, `complex128` | Stored as contiguous pairs `[real, imag]`. |
| **Signed Int** | `i64`, `i32`, `i16`, `i8` | Two's complement Little-Endian. |
| **Unsigned** | `u64`, `u32`, `u16`, `u8` | Little-Endian. |
| **Boolean** | `bool` | **1 Byte per Bool.** `0x00` = False, `0x01` = True. (Allows direct mmap). |


## Part III: Standard Layouts (Logical Layer)

### 1. Format: `dense`

Standard contiguous array in **Row-Major (C-contiguous)** order.

* **Required Components:** `data`
* **Memory Mapping:** Readers **SHOULD** mmap `data` if `encoding` is `raw`.

### 2. Format: `sparse_csr`

Compressed Sparse Row. Allows mixed types (e.g., `f32` values with `u16` indices).

* **Required Components:**
  1. `values`: Non-zero elements.
  2. `indices`: Column indices (Integer).
  3. `indptr`: Row pointers (Integer, size = rows + 1).

### 3. Format: `sparse_coo`

Coordinate List.

* **Required Components:**
  1. `values`: Non-zero elements.
  2. `coords`: Coordinate indices.

* **Coordinate Ordering:** `coords` is a flattened array of shape `(ndim * nnz)` stored in **Structure-of-Arrays (SoA)** order.
  * *Storage Order:* `[row_indices... , col_indices...]`

### 4. Format: `quantized_group` (e.g., GPTQ)

Block-wise quantization where weights are packed, and scales/zeros are kept in higher precision.

* **Required Components:**
  1. `packed_weight`: The quantized data (e.g., `i32` or `u8` acting as a container).
  2. `scales`: Scaling factors.
  3. `zeros`: Zero-points.

**Example (4-bit GPTQ):** Logical shape `[4096, 4096]`, but physically stored as packed `i32`.

```json
"model.layers.0.self_attn.q_proj": {
  "shape": [4096, 4096],
  "format": "quantized_group",
  "attributes": {
    "bits": 4,
    "group_size": 128,
    "packing": "8_per_i32"
  },
  "components": {
    "packed_weight": { "dtype": "i32", "offset": 1024, "length": 8388608 }, 
    "scales":        { "dtype": "f16", "offset": 8389632, "length": 262144 },
    "zeros":         { "dtype": "f16", "offset": 8651776, "length": 262144 }
  }
}
```


## Part IV: Parsing & Safety

### 1. Parsing Algorithm

1. **Read Footer:** Seek to `EOF - 16`. Read 16 bytes.
2. **Verify Magic:** Ensure the last 8 bytes are `ZTEN1000`. If not, the file is corrupt.
3. **Read Size:** Decode the first 8 bytes as `uint64` (**Little-Endian**) to get `manifest_size`.
4. **Safety Check:** If `manifest_size > 1,073,741,824` (1GB), the parser **MUST** abort to prevent DoS/OOM attacks.
5. **Read Manifest:** Seek to `EOF - 16 - manifest_size`. Read `manifest_size` bytes.
6. **Decode CBOR:** Pass the buffer to a standard CBOR decoder. The decoder handles the internal structure (Big-Endian lengths) per RFC 7049.
7. **Load Component:**
   * Seek to `component.offset`.
   * Read `component.length`.
   * If `component.encoding == "zstd"`, decompress buffer.
   * Cast result to `component.dtype` (interpreting binary data as **Little-Endian**).

### 2. Security Guidelines

* **No Execution:** Parsers **MUST NOT** execute any data (no pickle/eval).
* **Bounds Check:** `offset + length` MUST NOT exceed `FILE_SIZE`.
* **Padding Integrity:** Parsers generally ignore padding bytes, but writers **MUST** set them to `0x00` to ensure identical file hashes for identical content.
