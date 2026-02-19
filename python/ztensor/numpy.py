"""
NumPy convenience functions for ztensor.

This module provides a safetensors-compatible API for saving and loading
NumPy arrays in the ztensor format.

Example:
    >>> from ztensor.numpy import save_file, load_file
    >>> import numpy as np
    >>> tensors = {"embedding": np.zeros((512, 1024))}
    >>> save_file(tensors, "model.zt")
    >>> loaded = load_file("model.zt")
"""

import os
import tempfile
from typing import Dict, Optional, Union

import numpy as np

from . import Reader, Writer, ZTensorError


def save_file(
    tensors: Dict[str, np.ndarray],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    compression: Union[bool, int] = False,
) -> None:
    """
    Saves a dictionary of tensors into `filename` in ztensor format.

    Args:
        tensors: The incoming tensors. Tensors need to be contiguous and dense.
        filename: The filename we're saving into.
        metadata: Optional text only metadata you might want to save in your
            header. For instance it can be useful to specify more about the
            underlying tensors. This is purely informative and does not
            affect tensor loading.
            NOTE: ztensor does not currently support custom metadata; this
            parameter is accepted for API compatibility with safetensors
            but will be ignored.
        compression: Compression settings. False/0 (raw), True (level 3), or int > 0 (level).
            Default: False.

    Returns:
        None

    Example:
        >>> from ztensor.numpy import save_file
        >>> import numpy as np
        >>> tensors = {"embedding": np.zeros((512, 1024)), "attention": np.zeros((256, 256))}
        >>> save_file(tensors, "model.zt")
    """
    _validate_tensors(tensors)

    from ._ztensor import save_file as _native_save_file

    _native_save_file(tensors, str(filename), compression=compression)


def save(
    tensors: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, str]] = None,
    compression: Union[bool, int] = False,
) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in ztensor format.

    Args:
        tensors: The incoming tensors. Tensors need to be contiguous and dense.
        metadata: Optional text only metadata you might want to save in your
            header. This is purely informative and does not affect tensor loading.
            NOTE: ztensor does not currently support custom metadata; this
            parameter is accepted for API compatibility with safetensors
            but will be ignored.
        compression: Compression settings. False/0 (raw), True (level 3), or int > 0 (level).
            Default: False.

    Returns:
        The raw bytes representing the format.

    Example:
        >>> from ztensor.numpy import save
        >>> import numpy as np
        >>> tensors = {"embedding": np.zeros((512, 1024)), "attention": np.zeros((256, 256))}
        >>> byte_data = save(tensors)
    """
    _validate_tensors(tensors)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zt") as tmp:
        tmp_path = tmp.name

    try:
        save_file(tensors, tmp_path, metadata=metadata, compression=compression)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_file(
    filename: Union[str, os.PathLike],
    copy: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Loads a tensor file into numpy format.

    Supports multiple formats (auto-detected by extension):
        - .zt: zTensor format
        - .safetensors: HuggingFace SafeTensors format
        - .pt, .bin, .pth: PyTorch format

    Args:
        filename: The name of the file which contains the tensors.
        copy: If False (default), returns arrays backed by memory-mapped data
            with copy-on-write semantics (zero-copy reads, much faster).
            Arrays are writable — writes trigger per-page copies at the OS level.
            If True, returns independent arrays by copying data.

    Returns:
        Dictionary that contains name as key, value as `np.ndarray`.

    Example:
        >>> from ztensor.numpy import load_file
        >>> loaded = load_file("model.zt")
        >>> loaded = load_file("model.safetensors")
        >>> loaded = load_file("model.pt")
        >>> # For full independent copies:
        >>> loaded = load_file("model.zt", copy=True)
    """
    reader = Reader(str(filename))
    return reader.read_numpy(reader.keys(), copy=copy)


def load(data: bytes) -> Dict[str, np.ndarray]:
    """
    Loads a ztensor file into numpy format from pure bytes.

    Args:
        data: The content of a ztensor file.

    Returns:
        Dictionary that contains name as key, value as `np.ndarray`.

    Example:
        >>> from ztensor.numpy import load
        >>> file_path = "./my_folder/bert.zt"
        >>> with open(file_path, "rb") as f:
        ...     data = f.read()
        >>> loaded = load(data)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zt") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        return load_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --- Helper Functions ---


def _validate_tensors(tensors: Dict[str, np.ndarray]) -> None:
    """Validates that all tensors are valid for saving."""
    if not isinstance(tensors, dict):
        raise ValueError(
            f"Expected a dict of [str, np.ndarray] but received {type(tensors)}"
        )

    for k, v in tensors.items():
        if not isinstance(v, np.ndarray):
            raise ValueError(
                f"Key `{k}` is invalid, expected np.ndarray but received {type(v)}"
            )


__all__ = [
    "save_file",
    "save",
    "load_file",
    "load",
]
