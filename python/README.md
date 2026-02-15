# ztensor

[![PyPI](https://img.shields.io/pypi/v/ztensor.svg)](https://pypi.org/project/ztensor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python bindings for the [zTensor](https://github.com/pie-project/ztensor) tensor serialization library.

## Installation

```bash
pip install ztensor
```

**Optional dependencies:**
- `torch` — PyTorch support
- `scipy` — Reading sparse tensors as scipy sparse matrices
- `ml_dtypes` — NumPy bfloat16 support (`pip install ztensor[bfloat16]`)

## Quick Start

```python
import numpy as np
from ztensor import Writer, Reader

# Write tensors
with Writer("model.zt") as w:
    w.add("weights", np.random.randn(1024, 768).astype(np.float32))
    w.add("bias", np.zeros(768, dtype=np.float32))

# Read tensors (zero-copy mmap)
with Reader("model.zt") as r:
    weights = r["weights"]  # numpy array
    print(f"Shape: {weights.shape}, dtype: {weights.dtype}")
```

## PyTorch

```python
from ztensor.torch import save_file, load_file

save_file({"embed": torch.randn(1000, 768)}, "model.zt")
tensors = load_file("model.zt", device="cuda:0")
```

## Supported Data Types

`float64`, `float32`, `float16`, `bfloat16`, `int64`, `int32`, `int16`, `int8`, `uint64`, `uint32`, `uint16`, `uint8`, `bool`

## Documentation

See the [full API reference](https://pie-project.github.io/ztensor/python) for details on Writer, Reader, sparse tensors, compression, and more.

## License

MIT
