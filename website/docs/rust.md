---
sidebar_position: 4
---

# Rust API

```toml
[dependencies]
ztensor = "1.2"
```

For the full type-level documentation, see [docs.rs/ztensor](https://docs.rs/ztensor).

## Reading

### From a file

```rust
use ztensor::Reader;

let reader = Reader::open("model.zt")?;
let weights: Vec<f32> = reader.read_as("weights")?;
```

### Memory-mapped (zero-copy)

```rust
let reader = Reader::open_mmap("model.zt")?;

// Zero-copy typed slice into the file
let weights: &[f32] = reader.view_as("weights")?;
```

### Any format

The `open()` function auto-detects format by extension and returns a `Box<dyn TensorReader + Send>`:

```rust
use ztensor::open;

let reader = open("model.safetensors")?;  // SafeTensors
let reader = open("model.pt")?;           // PyTorch pickle
let reader = open("model.gguf")?;         // GGUF
let reader = open("weights.npz")?;        // NumPy NPZ
let reader = open("model.onnx")?;         // ONNX
let reader = open("model.h5")?;           // HDF5

// All readers implement the same trait
for (name, obj) in reader.tensors() {
    println!("{}: {:?} {:?}", name, obj.shape, obj.data_dtype()?);
}
```

Each format requires its corresponding feature flag; see [Feature flags](#feature-flags) below.

## Writing

### Dense tensors

```rust
use ztensor::Writer;

let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

let mut writer = Writer::create("model.zt")?;
writer.add("weights", &[2, 3], &data)?;
writer.finish()?;
```

### With compression and checksum

Use the builder returned by `add_with`:

```rust
use ztensor::{Writer, Compression, Checksum};

let mut writer = Writer::create("model.zt")?;

writer.add_with("weights", &[2, 3], &data)
    .compress(Compression::Zstd(3))
    .checksum(Checksum::Crc32c)
    .write()?;

writer.finish()?;
```

Compressed data is decompressed and checksums are verified automatically on read.

### Sparse tensors

#### CSR (Compressed Sparse Row)

```rust
use ztensor::{Writer, DType, Compression, Checksum};

let values: Vec<f32> = vec![1.0, 2.0, 3.0];
let indices: Vec<u64> = vec![0, 2, 1];      // column indices
let indptr: Vec<u64> = vec![0, 1, 3];       // row pointers

let mut writer = Writer::create("sparse.zt")?;
writer.add_csr(
    "sparse", vec![2, 3], DType::F32,
    &values, &indices, &indptr,
    Compression::Raw, Checksum::None,
)?;
writer.finish()?;

// Reading
let reader = Reader::open("sparse.zt")?;
let csr = reader.read_csr::<f32>("sparse")?;
println!("{:?}", csr.values);  // [1.0, 2.0, 3.0]
```

#### COO (Coordinate List)

```rust
let values: Vec<f32> = vec![1.0, 2.0, 3.0];
let coords: Vec<u64> = vec![0, 0, 0, 2, 1, 1];  // (row, col) pairs flattened

writer.add_coo(
    "coords", vec![2, 3], DType::F32,
    &values, &coords,
    Compression::Raw, Checksum::None,
)?;
```

## Appending to existing files

```rust
use ztensor::Writer;

// Open an existing .zt file for appending
let mut writer = Writer::append("model.zt")?;

// Add new tensors (existing ones are preserved)
let extra: Vec<f32> = vec![1.0, 2.0, 3.0];
writer.add("extra_weights", &[3], &extra)?;

writer.finish()?;
```

Append reads the existing manifest, truncates only the manifest/footer, writes new tensor data after the existing data, and rewrites the combined manifest. Existing tensor data and offsets are untouched.

## Removing tensors

```rust
// Remove tensors by name, writing the result to a new file
ztensor::remove_tensors("model.zt", "trimmed.zt", &["unused_layer", "old_bias"])?;
```

Returns an error if any name is not found. Preserves compression settings, checksums, and per-object attributes.

## Replacing tensor data in-place

```rust
// Overwrite a tensor's data without rewriting the entire file
let new_weights: Vec<f32> = vec![0.0; 2 * 3];
ztensor::replace_tensor("model.zt", "weights", bytemuck::cast_slice(&new_weights))?;
```

The replacement data must have the exact same byte size as the original. Only raw (uncompressed) dense tensors can be replaced. Checksums are recomputed automatically.

This is much faster than a full rewrite for large files — only the tensor's data region and the manifest are updated.

## Feature flags

Each external format reader is behind a feature flag:

| Feature | Formats | Extensions |
|---|---|---|
| `safetensors` | SafeTensors | `.safetensors` |
| `pickle` | PyTorch / Pickle | `.pt`, `.bin`, `.pth`, `.pkl` |
| `gguf` | GGUF | `.gguf` |
| `npz` | NumPy | `.npz` |
| `onnx` | ONNX | `.onnx` |
| `hdf5` | HDF5 | `.h5`, `.hdf5` |
| `all-formats` | All of the above | |

```toml
[dependencies]
# Individual features
ztensor = { version = "1.2", features = ["safetensors", "pickle"] }

# Or all at once
ztensor = { version = "1.2", features = ["all-formats"] }
```
