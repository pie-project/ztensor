"""A safetensors-shaped convenience layer, for migrating existing code.

    import ztensor.numpy as ztnp

    tensors = ztnp.load_file("model.zt")              # zero-copy where possible
    ztnp.save_file(tensors, "out.zt")

This is a **shim**, not the API. It exists so code written against
``safetensors.numpy`` keeps working, and it inherits that shape's limits: a
dict of arrays cannot express parts, capabilities, or where the bytes are. New
code should use :mod:`ztensor` directly, where a tensor is a handle you can ask
those questions of.

Zero-copy arrays are views onto the file's mapping and stay valid as long as
the array (or anything derived from it) is alive, because each one holds a
reference back through DLPack. They are read-only: writing to a file you did
not open for writing is not something a loader should let you do by accident.
"""

from __future__ import annotations

import numpy as np

from . import _ztensor

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
}

# numpy's bool_ is one byte per element, which is the storage the `bool`
# logical type pins. The logical type is what keeps 0/1 a promise.
_BOOL = np.dtype(np.bool_)

_ELEMENT_BYTES = {
    "f64": 8, "f32": 4, "f16": 2, "bf16": 2,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1,
    "u64": 8, "u32": 4, "u16": 2, "u8": 1,
}


def _as_array(tensor, copy: bool):
    """One tensor as a numpy array, or as its raw bytes when numpy has no
    dtype for it.

    Types numpy cannot express (bf16 without ``ml_dtypes``, fp8, fp4) come back
    as ``uint8`` with the element boundary kept as a trailing axis, so the shape
    is never silently dropped, because a flat byte array that used to be a
    matrix is the kind of thing that is noticed three bugs later.
    """
    if not copy:
        try:
            return np.from_dlpack(tensor)
        except (TypeError, ValueError, BufferError, RuntimeError):
            pass  # a dtype numpy has no name for, or bytes that had to be decoded

    raw = np.frombuffer(tensor.tobytes(), dtype=np.uint8)
    width = _ELEMENT_BYTES.get(tensor.dtype)
    shape = tuple(tensor.shape)
    if tensor.logical is None and width is not None:
        named = {
            "f64": np.float64, "f32": np.float32, "f16": np.float16,
            "i64": np.int64, "i32": np.int32, "i16": np.int16, "i8": np.int8,
            "u64": np.uint64, "u32": np.uint32, "u16": np.uint16, "u8": np.uint8,
        }.get(tensor.dtype)
        if named is not None:
            return raw.view(named).reshape(shape)
        # bf16 and friends: keep the shape, expose the element bytes.
        return raw.reshape(shape + (width,))
    return raw


def load_file(path, copy: bool = False, dense_only: bool = True) -> dict:
    """Loads the tensors of a file of any supported format into a dict.

    With ``copy=False`` (the default) arrays are zero-copy views where the
    source allows it, and copies where it does not. The per-tensor answer is
    ``src[name].caps.map``.

    A dict of arrays cannot express a tensor whose parts are separate arrays,
    so a non-dense layout has no place to go. With ``dense_only=True`` (the
    default) meeting one raises :class:`ValueError` naming the tensors, since
    returning a dict that is quietly missing them is the worse answer. With
    ``dense_only=False`` each comes back as the raw bytes of its first part.
    """
    with _ztensor.open(str(path)) as src:
        out = {}
        skipped = []
        for tensor in src.values():
            if tensor.layout != "dense":
                if dense_only:
                    skipped.append(f"{tensor.name} ({tensor.layout})")
                    continue
                first = tensor[tensor.parts[0]]
                out[tensor.name] = np.frombuffer(first.tobytes(), dtype=np.uint8)
                continue
            out[tensor.name] = _as_array(tensor, copy)
        if skipped:
            raise ValueError(
                f"{path}: a dict of arrays cannot hold "
                + ", ".join(skipped)
                + ". Pass dense_only=False for their raw bytes, or use "
                "ztensor.open() where a tensor keeps its parts."
            )
        return out


def save_file(
    tensors: dict, path, align: int | None = None, compression: bool = False
) -> int:
    """Writes a dict of numpy arrays to a canonical ``.zt`` file.

    Canonical form requires ascending tensor names, so the dict is sorted on
    the way out; the resulting file is byte-identical for identical input
    regardless of insertion order.
    """
    canonical = align is None and not compression
    if compression and align is None:
        align = 4096  # canonical form is raw; a compressed file is not canonical
    with _ztensor.Writer(str(path), canonical=canonical, align=align) as w:
        for name in sorted(tensors):
            arr = np.ascontiguousarray(tensors[name])
            logical = None
            if arr.dtype == _BOOL:
                dtype, logical = "u8", "bool"
            else:
                dtype = _FROM_NUMPY.get(arr.dtype)
            if dtype is None:
                raise ValueError(
                    f"{name}: numpy dtype {arr.dtype} has no zTensor mapping"
                )
            # `.view(uint8)` reinterprets without copying, which is what lets
            # the writer take the array's own bytes.
            raw = arr.reshape(-1).view(np.uint8)
            w.add(
                name,
                raw,
                shape=list(arr.shape),
                dtype=dtype,
                logical=logical,
                encoding="zt.zstd-seekable/1" if compression else None,
            )
        return w.finish()
