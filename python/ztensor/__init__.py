"""
zTensor - High-performance tensor serialization format.

This module provides Python bindings for the zTensor library via native
PyO3 bindings. It supports reading and writing zTensor (.zt) files,
as well as reading SafeTensors, PyTorch, GGUF, NumPy, ONNX, and HDF5 files.

Example:
    >>> import ztensor
    >>> reader = ztensor.Reader("model.zt")
    >>> tensor = reader["weights"]  # Returns NumPy array
    >>> reader.keys()
    ['weights', 'bias']
"""

from ._ztensor import (
    Reader,
    Writer,
    Tensor,
    TensorMetadata,
    open,
    remove_tensors,
    replace_tensor,
)

import numpy as np

# --- Optional ml_dtypes for bfloat16 in NumPy ---
try:
    from ml_dtypes import bfloat16 as np_bfloat16

    ML_DTYPES_AVAILABLE = True
except ImportError:
    np_bfloat16 = None
    ML_DTYPES_AVAILABLE = False

# --- Optional PyTorch Import ---
try:
    import importlib

    _torch = importlib.import_module("torch")
    TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    TORCH_AVAILABLE = False


class ZTensorError(Exception):
    """Custom exception for ztensor-related errors."""

    pass


# --- Type Mappings (used by torch.py and numpy.py) ---
DTYPE_NP_TO_ZT = {
    np.dtype("float64"): "float64",
    np.dtype("float32"): "float32",
    np.dtype("float16"): "float16",
    np.dtype("int64"): "int64",
    np.dtype("int32"): "int32",
    np.dtype("int16"): "int16",
    np.dtype("int8"): "int8",
    np.dtype("uint64"): "uint64",
    np.dtype("uint32"): "uint32",
    np.dtype("uint16"): "uint16",
    np.dtype("uint8"): "uint8",
    np.dtype("bool"): "bool",
}
if ML_DTYPES_AVAILABLE:
    DTYPE_NP_TO_ZT[np.dtype(np_bfloat16)] = "bfloat16"
DTYPE_ZT_TO_NP = {v: k for k, v in DTYPE_NP_TO_ZT.items()}

if TORCH_AVAILABLE:
    DTYPE_TORCH_TO_ZT = {
        _torch.float64: "float64",
        _torch.float32: "float32",
        _torch.float16: "float16",
        _torch.bfloat16: "bfloat16",
        _torch.int64: "int64",
        _torch.int32: "int32",
        _torch.int16: "int16",
        _torch.int8: "int8",
        _torch.uint8: "uint8",
        _torch.bool: "bool",
    }
    DTYPE_ZT_TO_TORCH = {v: k for k, v in DTYPE_TORCH_TO_ZT.items()}


__all__ = [
    "Reader",
    "Writer",
    "Tensor",
    "TensorMetadata",
    "ZTensorError",
    "open",
    "remove_tensors",
    "replace_tensor",
]
