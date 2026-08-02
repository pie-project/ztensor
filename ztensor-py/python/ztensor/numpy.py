"""NumPy convenience layer.

    import ztensor.numpy as ztnp

    tensors = ztnp.load_file("model.zt")              # zero-copy by default
    tensors = ztnp.load_file("model.zt", copy=True)   # owned arrays
    ztnp.save_file(tensors, "out.zt")

Zero-copy arrays are views onto the file's memory map: they stay valid as
long as the array (or anything derived from it) is alive, because each one
holds a reference back to the mapping. They are read-only — writing to a
file you did not open for writing is not something a loader should let you
do by accident.
"""

from __future__ import annotations

import numpy as np

from . import _ztensor

# Storage types, and the logical types that reinterpret them. Logical types
# numpy has no dtype for (fp8, fp4, ...) come back as raw bytes; nothing is
# silently reinterpreted.
_TO_NUMPY = {
    "f64": np.float64,
    "f32": np.float32,
    "f16": np.float16,
    "bf16": None,  # no numpy dtype; served as raw u16
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "u64": np.uint64,
    "u32": np.uint32,
    "u16": np.uint16,
    "u8": np.uint8,
}

_FROM_NUMPY = {
    np.dtype(np.float64): "f64",
    np.dtype(np.float32): "f32",
    np.dtype(np.float16): "f16",
    np.dtype(np.int64): "i64",
    np.dtype(np.int32): "i32",
    np.dtype(np.int16): "i16",
    np.dtype(np.int8): "i8",
    np.dtype(np.uint64): "u64",
    np.dtype(np.uint32): "u32",
    np.dtype(np.uint16): "u16",
    np.dtype(np.uint8): "u8",
    np.dtype(np.bool_): "u8",
}


def _numpy_dtype(info):
    """The numpy dtype for a part, or None when numpy cannot express it."""
    logical = info.get("type")
    if logical == "bool":
        return np.bool_
    if logical is not None:
        return None  # fp8/fp4/complex-packed: caller gets raw bytes
    return _TO_NUMPY.get(info["dtype"])


def load_file(path, copy: bool = False, dense_only: bool = True) -> dict:
    """Loads every tensor of a file of any supported format.

    With ``copy=False`` (the default) arrays are zero-copy views onto the
    mapping where the source allows it, and copies where it does not — the
    per-tensor answer is in ``Source.caps(name)["zero_copy"]``.
    """
    src = _ztensor.open(str(path))
    out = {}
    for name in src.keys():
        info = src.info(name)
        if dense_only and info["layout"] != "dense":
            continue
        dtype = _numpy_dtype(info)
        raw = dtype is None

        if copy or not src.caps(name)["zero_copy"]:
            buf = src.read(name)
        else:
            buf = src.view(name)

        arr = np.frombuffer(buf, dtype=np.uint8 if raw else dtype)
        out[name] = arr if raw else arr.reshape(tuple(info["shape"]))
    return out


def save_file(
    tensors: dict, path, align: int | None = None, compression: bool = False
) -> int:
    """Writes a dict of numpy arrays to a canonical ``.zt`` file.

    Canonical form requires ascending tensor names, so the dict is sorted
    on the way out; the resulting file is byte-identical for identical
    input regardless of insertion order.
    """
    if compression and align is None:
        align = 4096  # canonical form is raw; compressed files are not canonical
    w = _ztensor.Writer(str(path), align)
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        dtype = _FROM_NUMPY.get(arr.dtype)
        if dtype is None:
            raise ValueError(f"{name}: numpy dtype {arr.dtype} has no zTensor mapping")
        # `.view(uint8)` reinterprets without copying, which is what lets
        # the writer take the array's own bytes.
        raw = arr.reshape(-1).view(np.uint8)
        w.add(name, list(arr.shape), dtype, raw, compression)
    return w.finish()
