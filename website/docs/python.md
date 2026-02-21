---
sidebar_position: 3
---

# Python API

```bash
pip install ztensor
```

## Migrating from safetensors

ztensor provides drop-in replacements for `safetensors.torch` and `safetensors.numpy`:

```python
# Before
from safetensors.torch import save_file, load_file

# After — one-line change
from ztensor.torch import save_file, load_file
```

Both `ztensor.torch` and `ztensor.numpy` share the same interface:

```python
from ztensor.torch import save_file, load_file   # PyTorch tensors
from ztensor.numpy import save_file, load_file   # NumPy arrays
```

ztensor additionally supports:
- `compression` parameter on save functions
- `copy=False` for zero-copy mmap loading (the default)
- Reading other tensor formats through the same `load_file` call

### `save_file(tensors, filename, metadata=None, compression=False)`

Saves a dictionary of tensors to a `.zt` file.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tensors` | `dict[str, Tensor]` | required | `torch.Tensor` or `np.ndarray` values (must be contiguous) |
| `filename` | `str` or `PathLike` | required | Output file path |
| `metadata` | `dict[str, str]` | `None` | Optional metadata (for safetensors API compatibility) |
| `compression` | `bool` or `int` | `False` | `False` = raw, `True` = zstd level 3, `int` = specific level (1-22) |

```python
import torch
from ztensor.torch import save_file

tensors = {"weight": torch.randn(1024, 768), "bias": torch.zeros(768)}

save_file(tensors, "model.zt")                  # no compression
save_file(tensors, "model.zt", compression=True) # zstd level 3
save_file(tensors, "model.zt", compression=10)   # zstd level 10
```

Compressed files are decompressed automatically on load.

### `load_file(filename, ...)`

Loads a tensor file, detecting the format by extension.

**Supported formats:** `.zt`, `.safetensors`, `.pt` / `.pth` / `.bin`, `.gguf`, `.npz`, `.onnx`, `.h5` / `.hdf5`

**PyTorch:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` or `PathLike` | required | File to load |
| `device` | `str` or `int` | `"cpu"` | Target device (`"cpu"`, `"cuda:0"`, etc.) |
| `copy` | `bool` | `False` | `False` = zero-copy mmap, `True` = independent copies |

**NumPy:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filename` | `str` or `PathLike` | required | File to load |
| `copy` | `bool` | `False` | `False` = zero-copy mmap, `True` = independent copies |

```python
from ztensor.torch import load_file

loaded = load_file("model.zt")                    # zero-copy (default)
loaded = load_file("model.safetensors")           # any format
loaded = load_file("model.zt", device="cuda:0")   # load to GPU
loaded = load_file("model.zt", copy=True)         # independent copies
```

### The `copy` parameter

| Mode | Speed | Memory | Behavior |
|---|---|---|---|
| `copy=False` (default) | Zero-copy from mmap | Shared pages | Copy-on-write: reads are free, writes trigger per-page copies |
| `copy=True` | Standard | Full copy in RAM | Independent tensors, fully decoupled from the file |

The default `copy=False` uses `MAP_PRIVATE` (copy-on-write) memory mapping. Tensors are writable: modifications affect your process only, never the file on disk.

**When to use `copy=True`:**
- When you need tensors that outlive the file handle
- When you plan to heavily mutate tensor data in-place

### Bytes API

Serialize to and from raw bytes without touching disk.

```python
# PyTorch
from ztensor.torch import save, load
data = save({"x": torch.zeros(10)})
loaded = load(data)

# NumPy
from ztensor.numpy import save, load
data = save({"x": np.zeros(10, dtype=np.float32)})
loaded = load(data)
```

## Tensor

The `ztensor.Tensor` class represents a tensor object with its component arrays, shape, dtype, and format metadata. It is the primary return type of `reader.read()`.

### Construction

```python
import numpy as np
import ztensor

# Dense tensor — shape and dtype inferred from the "data" component
t = ztensor.Tensor({"data": np.ones((3, 4), dtype=np.float32)})

# Sparse CSR — explicit shape required
t = ztensor.Tensor(
    {"values": values, "indices": indices, "indptr": indptr},
    shape=[4096, 4096], dtype="float32", format="sparse_csr"
)
```

### Properties

| Property | Type | Description |
|---|---|---|
| `shape` | `list[int]` | Tensor dimensions |
| `dtype` | `str` | Storage dtype (e.g., `"float32"`) |
| `type` | `str \| None` | Logical type (e.g., `"f8_e4m3fn"`), `None` when same as dtype |
| `format` | `str` | Layout format (`"dense"`, `"sparse_csr"`, etc.) |
| `components` | `dict[str, np.ndarray]` | Component name to array mapping |
| `attributes` | `dict \| None` | Per-object attributes |

### Conversion (dense only)

```python
arr = t.numpy()              # -> np.ndarray
tensor = t.torch(device="cpu")  # -> torch.Tensor
```

## Reader

The `Reader` class provides direct, per-tensor access to files.

```python
from ztensor import Reader

reader = Reader("model.zt")
```

**Supported formats:** `.zt`, `.safetensors`, `.pt` / `.pth` / `.bin`, `.gguf`, `.npz`, `.onnx`, `.h5` / `.hdf5`

### Dict-like access

```python
tensor = reader["layer1.weight"]       # ztensor.Tensor, zero-copy
exists = "bias" in reader               # membership test
count = len(reader)                     # number of tensors
names = list(reader)                    # iterate tensor names
```

### `keys()`

Returns a list of all tensor names.

```python
names = reader.keys()
```

### `metadata(name)`

Returns metadata. Accepts a single name (returns `TensorMetadata`) or a list (returns `list[TensorMetadata]`).

```python
meta = reader.metadata("weights")
print(meta.shape, meta.dtype)  # [1024, 768] float32

metas = reader.metadata(["weight", "bias"])
```

### `read(name, *, copy=False)`

Reads tensor(s) as `ztensor.Tensor` objects. Supports all formats (dense, sparse, quantized).

```python
t = reader.read("weights")                    # -> ztensor.Tensor
d = reader.read(["weight", "bias"])            # -> dict[str, ztensor.Tensor]
t = reader.read("weights", copy=True)          # independent copy
```

### `read_numpy(name, *, copy=False)`

Reads dense tensor(s) as NumPy arrays.

```python
arr = reader.read_numpy("weights")                  # -> np.ndarray
d = reader.read_numpy(["weight", "bias"])            # -> dict[str, np.ndarray]
arr = reader.read_numpy("weights", copy=True)        # independent copy
```

### `read_torch(name, *, copy=False, device="cpu")`

Reads dense tensor(s) as torch.Tensors.

```python
t = reader.read_torch("weights")                         # -> torch.Tensor
d = reader.read_torch(["weight", "bias"])                 # -> dict[str, torch.Tensor]
t = reader.read_torch("weights", device="cuda:0")        # load to GPU
t = reader.read_torch("weights", copy=True)               # independent copy
```

### `read_into(name, dst)`

Copies tensor data directly into a pre-allocated destination array or tensor. This avoids intermediate allocations, which is useful for filling contiguous GPU arenas (e.g. on MPS or CUDA) without per-tensor allocation overhead.

The destination can be a `np.ndarray` or `torch.Tensor` on any device. Its shape and dtype must match the stored tensor — validation is delegated to `np.copyto` / `torch.Tensor.copy_()`.

**Single tensor:**

```python
dst = torch.empty(1024, 768, dtype=torch.float32, device="mps")
reader.read_into("layer.0.weight", dst)
```

**Batch (dict):**

Pass a `dict[str, dst]` to read multiple tensors. Reads are sorted by file offset internally for sequential I/O.

```python
reader.read_into({
    "layer.0.weight": weight_view,
    "layer.0.bias": bias_view,
})
```

**Arena pattern — contiguous GPU buffer with views:**

```python
reader = ztensor.Reader("model.zt")
for layer_idx in range(num_layers):
    arena = torch.empty(layer_bytes, dtype=torch.uint8, device="mps")
    offset = 0
    views = {}
    for name in layer_tensor_names(layer_idx):
        meta = reader.metadata(name)
        nbytes = meta.components["data"]["length"]
        view = arena[offset:offset+nbytes].view(torch_dtype).reshape(meta.shape)
        views[name] = view
        offset += nbytes
    reader.read_into(views)
```

### `format`

The detected file format: `"zt"`, `"safetensors"`, `"pickle"`, `"gguf"`, `"npz"`, `"onnx"`, `"hdf5"`, or `"unknown"`.

### Context manager

```python
with Reader("model.zt") as r:
    weights = r["weights"]
```

## Writer

Creates `.zt` files with per-tensor control over compression and checksums.

```python
from ztensor import Writer
```

### `write(name, tensor, *, compress=None, checksum="none")`

Writes a `ztensor.Tensor` object. Supports any format (dense, sparse, quantized).

```python
t = ztensor.Tensor({"data": np.ones((3, 4), dtype=np.float32)})
w.write("weights", t)
```

### `write_numpy(name, data, *, compress=None, checksum="none")`

Writes a dense tensor from a NumPy array.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Tensor name |
| `data` | `np.ndarray` | required | Contiguous array |
| `compress` | `bool` or `int` | `None` | `None`/`False` = raw, `True` = zstd level 3, `int` = specific level |
| `checksum` | `str` | `"none"` | `"none"`, `"crc32c"`, or `"sha256"` |

```python
w.write_numpy("weights", np_array)
w.write_numpy("compressed", np_array, compress=True)
w.write_numpy("verified", np_array, checksum="crc32c")
```

### `write_torch(name, data, *, compress=None, checksum="none")`

Writes a dense tensor from a torch.Tensor. Handles CPU transfer, contiguity, and bfloat16 automatically.

```python
w.write_torch("weights", torch_tensor)
w.write_torch("compressed", torch_tensor, compress=True)
```

### `add(name, data, compress=None, checksum="none")`

Alias for `write_numpy`. Maintained for backward compatibility.

### `finish()`

Writes the manifest and footer. Returns the total file size. Called automatically when using the context manager.

```python
with Writer("output.zt") as w:
    w.write_numpy("weights", weights)
    w.write_numpy("bias", bias)
# finish() called automatically
```

### Appending to existing files

`Writer.append(path)` opens an existing `.zt` file for appending new tensors. Existing tensors are preserved; new tensors are written after the existing data.

```python
w = Writer.append("model.zt")
w.write_numpy("extra_layer", new_weights)
w.finish()
```

Duplicate tensor names raise an error.

## Model save/load

These functions are specific to `ztensor.torch`.

### `save_model(model, filename, metadata=None, force_contiguous=True)`

Saves all parameters from a `torch.nn.Module`. Shared tensors are deduplicated automatically. If `force_contiguous=True` (default), non-contiguous parameters are copied to contiguous storage before writing.

```python
from ztensor.torch import save_model

model = torch.nn.Linear(10, 5)
save_model(model, "model.zt")
```

### `load_model(model, filename, strict=True, device="cpu", copy=False)`

Loads weights into an existing model. Returns `(missing_keys, unexpected_keys)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `torch.nn.Module` | required | Target model |
| `filename` | `str` or `PathLike` | required | File to load |
| `strict` | `bool` | `True` | Fail on missing/unexpected keys |
| `device` | `str` or `int` | `"cpu"` | Target device |
| `copy` | `bool` | `False` | `False` = zero-copy mmap, `True` = independent copies |

```python
from ztensor.torch import load_model

model = torch.nn.Linear(10, 5)
missing, unexpected = load_model(model, "model.zt")
```

## Removing tensors

### `ztensor.remove_tensors(input, output, names)`

Removes tensors by name from a `.zt` file, writing the result to a new file. Preserves compression settings, checksums, and per-object attributes.

| Parameter | Type | Description |
|---|---|---|
| `input` | `str` | Source `.zt` file |
| `output` | `str` | Output `.zt` file |
| `names` | `list[str]` | Tensor names to remove |

```python
import ztensor

ztensor.remove_tensors("model.zt", "trimmed.zt", ["unused_layer", "old_bias"])
```

Returns an error if any name is not found in the input file.

## Replacing tensor data in-place

### `ztensor.replace_tensor(path, name, data)`

Replaces the data of a dense tensor in-place within an existing `.zt` file. The replacement array must have the same byte size as the original. Only raw (uncompressed) tensors can be replaced. Checksums are recomputed automatically.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Path to the `.zt` file (modified in-place) |
| `name` | `str` | Name of the tensor to replace |
| `data` | `np.ndarray` | Replacement array (must be contiguous, same byte size) |

```python
import numpy as np
import ztensor

# Replace weights in-place (much faster than rewriting the whole file)
new_weights = np.zeros((1024, 768), dtype=np.float32)
ztensor.replace_tensor("model.zt", "weights", new_weights)
```

## Reference

### TensorMetadata

Returned by `reader.metadata(name)`.

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Tensor name |
| `shape` | `list[int]` | Tensor dimensions |
| `dtype` | `str` | Storage data type (e.g., `"float32"`) |
| `type` | `str \| None` | Logical type (e.g., `"f8_e4m3fn"`), `None` when same as `dtype` |
| `format` | `str` | Layout format (e.g., `"dense"`, `"sparse_csr"`) |

### `ztensor.open(path)`

Convenience alias for `Reader(path)`.

### `ztensor.ZTensorError`

Custom exception for ztensor errors. Subclass of `Exception`.

### Supported dtypes

| PyTorch | NumPy | zTensor |
|---|---|---|
| `torch.float64` | `np.float64` | `float64` |
| `torch.float32` | `np.float32` | `float32` |
| `torch.float16` | `np.float16` | `float16` |
| `torch.bfloat16` | `bfloat16` (ml_dtypes) | `bfloat16` |
| `torch.int64` | `np.int64` | `int64` |
| `torch.int32` | `np.int32` | `int32` |
| `torch.int16` | `np.int16` | `int16` |
| `torch.int8` | `np.int8` | `int8` |
| `torch.uint8` | `np.uint8` | `uint8` |
| `torch.bool` | `np.bool_` | `bool` |
| - | `np.uint64` | `uint64` |
| - | `np.uint32` | `uint32` |
| - | `np.uint16` | `uint16` |

:::note
`bfloat16` support in NumPy requires the [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes) package:
```bash
pip install ztensor[bfloat16]
```
:::
