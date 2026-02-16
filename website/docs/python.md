---
sidebar_position: 3
---

# Python API

```bash
pip install ztensor
```

## Quick start

```python
import numpy as np
from ztensor.numpy import save_file, load_file

# Save
save_file({"weight": np.random.randn(1024, 768).astype(np.float32)}, "model.zt")

# Load (any format)
tensors = load_file("model.zt")           # .zt
tensors = load_file("model.safetensors")  # .safetensors, .pt, .gguf, .npz, .onnx, .h5
```

The `ztensor.torch` and `ztensor.numpy` modules share the same interface:

```python
from ztensor.torch import save_file, load_file   # PyTorch tensors
from ztensor.numpy import save_file, load_file   # NumPy arrays
```

## Saving

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

## Loading

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

## Reader

The `Reader` class provides direct, per-tensor access to files. For most use cases, `load_file` above is simpler.

```python
from ztensor import Reader

reader = Reader("model.zt")
```

**Supported formats:** `.zt`, `.safetensors`, `.pt` / `.pth` / `.bin`, `.gguf`, `.npz`, `.onnx`, `.h5` / `.hdf5`

### Dict-like access

```python
weights = reader["layer1.weight"]       # numpy array, zero-copy
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

Returns a [`TensorMetadata`](#tensormetadata) object.

```python
meta = reader.metadata("weights")
print(meta.shape, meta.dtype)  # [1024, 768] float32
```

### `read_tensor(name)` / `read_tensors(names)`

Reads individual or multiple tensors as NumPy arrays.

```python
arr = reader.read_tensor("weights")
result = reader.read_tensors(["weight", "bias"])
```

### `load_torch(*, device="cpu", copy=False)` / `load_numpy(*, copy=False)`

Loads all tensors at once.

```python
tensors = reader.load_torch(device="cuda:0")
arrays = reader.load_numpy()
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

### `add(name, data, compress=None, checksum="none")`

Adds a dense tensor from a NumPy array.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Tensor name |
| `data` | `np.ndarray` | required | Contiguous array |
| `compress` | `bool` or `int` | `None` | `None`/`False` = raw, `True` = zstd level 3, `int` = specific level |
| `checksum` | `str` | `"none"` | `"none"`, `"crc32c"`, or `"sha256"` |

```python
w.add("weights", np_array)
w.add("compressed", np_array, compress=True)
w.add("verified", np_array, checksum="crc32c")
```

### `finish()`

Writes the manifest and footer. Returns the total file size. Called automatically when using the context manager.

```python
with Writer("output.zt") as w:
    w.add("weights", weights)
    w.add("bias", bias)
# finish() called automatically
```

## Bytes API

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

### Migrating from safetensors

The API is a drop-in replacement:

```python
# Before
from safetensors.torch import save_file, load_file

# After
from ztensor.torch import save_file, load_file
```

zTensor additionally supports:
- `compression` parameter on save functions
- `copy=False` for zero-copy mmap loading (the default)
- Reading other tensor formats through the same `load_file` call
