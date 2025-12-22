# ztensor Python API

A Python package providing high-performance bindings to the `ztensor` Rust core library for reading and writing `.zt` tensor files.

## Installation

```bash
pip install ztensor
```

**Optional dependencies:**
- `torch` - PyTorch support for tensor operations
- `scipy` - Reading sparse tensors as scipy sparse matrices
- `ml_dtypes` - NumPy bfloat16 support

## Quick Start

```python
import numpy as np
from ztensor import Reader, Writer

# Write tensors
with Writer("model.zt") as writer:
    writer.add_tensor("weights", np.random.randn(512, 1024).astype(np.float32))
    writer.add_tensor("bias", np.zeros(1024, dtype=np.float32))

# Read tensors
with Reader("model.zt") as reader:
    weights = reader["weights"]  # Returns numpy array
    bias = reader.read_tensor("bias")
```

---

## Core API

### Writer

Context manager for writing ztensor files.

```python
from ztensor import Writer
```

#### `Writer(file_path: str)`

Creates a new ztensor file writer.

```python
with Writer("output.zt") as writer:
    # Add tensors here
    pass
```

#### `writer.add_tensor(name, tensor, compress=False)`

Adds a NumPy array or PyTorch tensor to the file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tensor name (must be unique) |
| `tensor` | `np.ndarray` or `torch.Tensor` | Tensor data |
| `compress` | `bool \| int` | Compression: `False`/`0` (off), `True` (level 3), or `int` (level 1-22). |

```python
import numpy as np

with Writer("tensors.zt") as writer:
    # Basic tensor
    writer.add_tensor("embedding", np.zeros((512, 768), dtype=np.float32))
    
    # With compression (level 3)
    writer.add_tensor("large_tensor", np.random.randn(10000, 10000).astype(np.float32), compress=True)
    
    # With generic level (e.g. 1 for speed, 19 for size)
    writer.add_tensor("max_compressed", data, compress=19)
    
    # PyTorch tensor
    import torch
    writer.add_tensor("torch_tensor", torch.randn(100, 100))
```

**Supported dtypes:** `float64`, `float32`, `float16`, `bfloat16`, `int64`, `int32`, `int16`, `int8`, `uint64`, `uint32`, `uint16`, `uint8`, `bool`

#### `writer.add_sparse_csr(name, values, indices, indptr, shape)`

Adds a sparse CSR (Compressed Sparse Row) tensor.

```python
from scipy.sparse import csr_matrix

sparse = csr_matrix([[1, 0, 2], [0, 0, 3], [4, 0, 0]])
with Writer("sparse.zt") as writer:
    writer.add_sparse_csr("my_csr", sparse.data, sparse.indices, sparse.indptr, sparse.shape)
```

#### `writer.add_sparse_coo(name, values, indices, shape)`

Adds a sparse COO (Coordinate) tensor.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tensor name |
| `values` | array | Non-zero values |
| `indices` | array | Coordinates matrix of shape `(ndim, nnz)` |
| `shape` | tuple | Tensor shape |

```python
# Manual COO creation
values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
indices = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint64)  # (ndim, nnz)
shape = (3, 3)

with Writer("coo.zt") as writer:
    writer.add_sparse_coo("diagonal", values, indices, shape)
```

#### `writer.finalize()`

Finalizes the file (called automatically when using context manager).

---

### Reader

Context manager for reading ztensor files.

```python
from ztensor import Reader
```

#### `Reader(file_path: str)`

Opens a ztensor file for reading.

```python
with Reader("model.zt") as reader:
    print(f"Contains {len(reader)} tensors")
    print(f"Tensor names: {reader.tensor_names}")
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tensor_names` | `list[str]` | All tensor names in the file |
| `tensors` | `list[TensorMetadata]` | All tensor metadata objects |

#### `reader[key]`

Dict-like access to tensors.

```python
with Reader("model.zt") as reader:
    # By name (returns tensor data)
    weights = reader["weights"]
    
    # By index (returns TensorMetadata)
    first_meta = reader[0]
```

#### `reader.read_tensor(name, to='numpy')`

Reads a tensor by name.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tensor name |
| `to` | `str` | Output format: `'numpy'` or `'torch'` |

```python
with Reader("model.zt") as reader:
    # As NumPy array
    arr = reader.read_tensor("weights", to='numpy')
    
    # As PyTorch tensor
    tensor = reader.read_tensor("weights", to='torch')
```

**Sparse tensor behavior:**
- `to='numpy'` → Returns `scipy.sparse.csr_matrix` or `scipy.sparse.coo_matrix`
- `to='torch'` → Returns `torch.sparse_csr_tensor` or `torch.sparse_coo_tensor`

#### `reader.metadata(name)`

Gets metadata for a specific tensor.

```python
with Reader("model.zt") as reader:
    meta = reader.metadata("weights")
    print(f"Name: {meta.name}")
    print(f"Shape: {meta.shape}")
    print(f"Dtype: {meta.dtype_str}")
```

#### Iteration

```python
with Reader("model.zt") as reader:
    for metadata in reader:
        print(f"{metadata.name}: {metadata.shape}")
```

#### Membership testing

```python
with Reader("model.zt") as reader:
    if "weights" in reader:
        print("Found weights!")
```

---

### TensorMetadata

Metadata about a tensor stored in a ztensor file.

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Tensor name |
| `shape` | `tuple[int, ...]` | Tensor dimensions |
| `dtype_str` | `str` | ztensor dtype string (e.g., `'float32'`) |
| `dtype` | `np.dtype` | NumPy dtype |
| `layout` | `str` | `'dense'`, `'sparse_csr'`, or `'sparse_coo'` |
| `encoding` | `str \| None` | `'raw'`, `'zstd'`, or `None` |
| `offset` | `int` | Byte offset in file |
| `size` | `int` | Size in bytes (compressed if applicable) |
| `endianness` | `str \| None` | `'little'` or `'big'` |
| `checksum` | `str \| None` | Checksum string if present |

```python
with Reader("model.zt") as reader:
    meta = reader.metadata("weights")
    print(meta)  # <TensorMetadata name='weights' shape=(512, 768) dtype='float32'>
```

---

## PyTorch API (`ztensor.torch`)

A safetensors-compatible API for PyTorch users.

```python
from ztensor.torch import save_file, load_file
```

### `save_file(tensors, filename, metadata=None)`

Saves a dictionary of tensors to a file.

```python
import torch
from ztensor.torch import save_file

tensors = {
    "embedding": torch.zeros(512, 1024),
    "attention": torch.randn(256, 256)
}
save_file(tensors, "model.zt")
```

### `load_file(filename, device='cpu')`

Loads all tensors from a file.

```python
from ztensor.torch import load_file

# Load to CPU
tensors = load_file("model.zt")

# Load to GPU
tensors = load_file("model.zt", device="cuda:0")
```

### `save(tensors, metadata=None) -> bytes`

Serializes tensors to bytes.

```python
from ztensor.torch import save

tensors = {"weights": torch.randn(100, 100)}
data = save(tensors)
```

### `load(data) -> dict`

Deserializes tensors from bytes.

```python
from ztensor.torch import load

tensors = load(data)
```

### `save_model(model, filename, metadata=None, force_contiguous=True)`

Saves a PyTorch model, handling shared tensors automatically.

```python
import torch
from ztensor.torch import save_model

model = torch.nn.Linear(1024, 512)
save_model(model, "linear.zt")
```

### `load_model(model, filename, strict=True, device='cpu')`

Loads weights into a PyTorch model.

```python
from ztensor.torch import load_model

model = torch.nn.Linear(1024, 512)
missing, unexpected = load_model(model, "linear.zt", device="cuda:0")
```

---

## Error Handling

```python
from ztensor import ZTensorError

try:
    with Reader("nonexistent.zt") as reader:
        pass
except ZTensorError as e:
    print(f"Error: {e}")
```

---

## Complete Examples

### Basic Read/Write

```python
import numpy as np
from ztensor import Reader, Writer

# Create and write tensors
data = {
    "weights": np.random.randn(512, 1024).astype(np.float32),
    "bias": np.zeros(1024, dtype=np.float32),
}

with Writer("model.zt") as writer:
    for name, tensor in data.items():
        writer.add_tensor(name, tensor)

# Read and verify
with Reader("model.zt") as reader:
    for meta in reader:
        tensor = reader[meta.name]
        assert np.allclose(tensor, data[meta.name])
        print(f"✓ {meta.name}: {meta.shape}")
```

### Compressed Tensors

```python
import numpy as np
from ztensor import Writer, Reader

# Large tensor with compression
large_tensor = np.random.randn(5000, 5000).astype(np.float32)

with Writer("compressed.zt") as writer:
    writer.add_tensor("data", large_tensor, compress=True)

with Reader("compressed.zt") as reader:
    meta = reader.metadata("data")
    print(f"Encoding: {meta.encoding}")  # 'zstd'
    loaded = reader["data"]
```

### Sparse Tensors

```python
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from ztensor import Writer, Reader

# Create sparse matrices
csr = csr_matrix([[1, 0, 2], [0, 0, 3], [4, 0, 0]], dtype=np.float32)
coo = coo_matrix([[1, 0], [0, 2]], dtype=np.float64)

with Writer("sparse.zt") as writer:
    writer.add_sparse_csr("csr_data", csr.data, csr.indices, csr.indptr, csr.shape)

with Reader("sparse.zt") as reader:
    loaded_csr = reader.read_tensor("csr_data", to='numpy')
    print(type(loaded_csr))  # <class 'scipy.sparse._csr.csr_matrix'>
```

### PyTorch Integration

```python
import torch
from ztensor.torch import save_file, load_file, save_model, load_model

# Save/load tensors
tensors = {"embed": torch.randn(1000, 768)}
save_file(tensors, "tensors.zt")
loaded = load_file("tensors.zt", device="cpu")

# Save/load models
model = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
save_model(model, "transformer.zt")

new_model = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8)
load_model(new_model, "transformer.zt")
```

### Half-Precision Types

```python
import numpy as np
from ztensor import Writer, Reader

# float16
fp16 = np.array([1.0, 2.0, 3.0], dtype=np.float16)

# bfloat16 (requires ml_dtypes)
from ml_dtypes import bfloat16
bf16 = np.array([1.0, 2.0, 3.0], dtype=bfloat16)

with Writer("half.zt") as writer:
    writer.add_tensor("fp16", fp16)
    writer.add_tensor("bf16", bf16)

with Reader("half.zt") as reader:
    print(reader.metadata("bf16").dtype_str)  # 'bfloat16'
```

---

## API Reference Summary

| Class/Function | Description |
|----------------|-------------|
| `Writer` | Write tensors to `.zt` files |
| `Reader` | Read tensors from `.zt` files |
| `TensorMetadata` | Metadata container for tensors |
| `ZTensorError` | Exception for ztensor operations |
| `ztensor.torch.save_file` | Save dict of PyTorch tensors |
| `ztensor.torch.load_file` | Load PyTorch tensors from file |
| `ztensor.torch.save` | Serialize PyTorch tensors to bytes |
| `ztensor.torch.load` | Deserialize PyTorch tensors from bytes |
| `ztensor.torch.save_model` | Save PyTorch model state dict |
| `ztensor.torch.load_model` | Load PyTorch model state dict |
