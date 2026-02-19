"""
PyTorch convenience functions for ztensor.

This module provides a safetensors-compatible API for saving and loading
PyTorch tensors in the ztensor format.

Example:
    >>> from ztensor.torch import save_file, load_file
    >>> import torch
    >>> tensors = {"embedding": torch.zeros((512, 1024))}
    >>> save_file(tensors, "model.zt")
    >>> loaded = load_file("model.zt")
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required to use ztensor.torch. "
        "Please install it with: pip install torch"
    )

from . import Reader, Writer, ZTensorError, DTYPE_ZT_TO_TORCH

# --- Torch dtype → numpy dtype for writer compatibility ---
_TORCH_TO_NP_DTYPE = {
    torch.float64: np.float64,
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

# bfloat16 handled separately via ml_dtypes or raw bytes


def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    compression: Union[bool, int] = False,
) -> None:
    """
    Saves a dictionary of tensors into `filename` in ztensor format.

    Args:
        tensors: The incoming tensors. Tensors need to be contiguous and dense.
        filename: The filename we're saving into.
        metadata: Optional text only metadata (accepted for API compatibility,
            currently ignored).
        compression: Compression settings. False/0 (raw), True (level 3), or int > 0 (level).
            Default: False.

    Example:
        >>> from ztensor.torch import save_file
        >>> import torch
        >>> tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
        >>> save_file(tensors, "model.zt")
    """
    _validate_tensors(tensors)

    # Convert all tensors to numpy, then use native batch save
    np_tensors = {}
    for name, tensor in tensors.items():
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        np_tensors[name] = _torch_to_numpy(tensor)

    from ._ztensor import save_file as _native_save_file

    _native_save_file(np_tensors, str(filename), compression=compression)


def save(
    tensors: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]] = None,
    compression: Union[bool, int] = False,
) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in ztensor format.

    Args:
        tensors: The incoming tensors. Tensors need to be contiguous and dense.
        metadata: Optional text only metadata (accepted for API compatibility,
            currently ignored).
        compression: Compression settings. False/0 (raw), True (level 3), or int > 0 (level).
            Default: False.

    Returns:
        The raw bytes representing the format.

    Example:
        >>> from ztensor.torch import save
        >>> import torch
        >>> tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
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
    device: Union[str, int] = "cpu",
    copy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Loads a tensor file into torch format.

    Supports multiple formats (auto-detected by extension):
        - .zt: zTensor format
        - .safetensors: HuggingFace SafeTensors format
        - .pt, .bin, .pth: PyTorch format

    Args:
        filename: The name of the file which contains the tensors.
        device: The device where the tensors need to be located after load.
            Available options are all regular torch device locations.
        copy: If False (default), returns tensors backed by memory-mapped data
            with copy-on-write semantics (zero-copy reads, much faster).
            Tensors are writable — writes trigger per-page copies at the OS level.
            If True, returns independent tensors by copying data.

    Returns:
        Dictionary that contains name as key, value as `torch.Tensor`.

    Example:
        >>> from ztensor.torch import load_file
        >>> loaded = load_file("model.zt")
        >>> loaded = load_file("model.safetensors")
        >>> loaded = load_file("model.pt")
        >>> # For full independent copies:
        >>> loaded = load_file("model.zt", copy=True)
    """
    if isinstance(device, int):
        device = f"cuda:{device}"

    reader = Reader(str(filename))
    return reader.read_torch(reader.keys(), device=device, copy=copy)


def load(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Loads a ztensor file into torch format from pure bytes.

    Args:
        data: The content of a ztensor file.

    Returns:
        Dictionary that contains name as key, value as `torch.Tensor` on cpu.

    Example:
        >>> from ztensor.torch import load
        >>> file_path = "./my_folder/bert.zt"
        >>> with open(file_path, "rb") as f:
        ...     data = f.read()
        >>> loaded = load(data)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zt") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        reader = Reader(tmp_path)
        return reader.read_torch(reader.keys(), device="cpu")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def save_model(
    model: torch.nn.Module,
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    force_contiguous: bool = True,
) -> None:
    """
    Saves a given torch model to specified filename.

    This method handles tensor sharing issues by detecting shared tensors
    and only saving each unique tensor once, recording the mapping in
    metadata so it can be restored on load.

    Args:
        model: The model to save on disk.
        filename: The filename location to save the file.
        metadata: Extra information to save along with the file.
        force_contiguous: Forcing the state_dict to be saved as contiguous
            tensors. Default: True.

    Example:
        >>> from ztensor.torch import save_model
        >>> import torch
        >>> model = torch.nn.Linear(10, 5)
        >>> save_model(model, "model.zt")
    """
    state_dict = model.state_dict()

    to_removes = _remove_duplicate_names(state_dict)

    if metadata is None:
        metadata = {}

    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del state_dict[to_remove]

    if force_contiguous:
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    try:
        save_file(state_dict, filename, metadata=metadata)
    except ValueError as e:
        msg = str(e)
        msg += " Or use save_model(..., force_contiguous=True), read the docs for potential caveats."
        raise ValueError(msg)


def load_model(
    model: torch.nn.Module,
    filename: Union[str, os.PathLike],
    strict: bool = True,
    device: Union[str, int] = "cpu",
    copy: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Loads a given filename onto a torch model.

    This method handles tensor sharing issues which are not allowed in
    ztensor format.

    Args:
        model: The model to load onto.
        filename: The filename location to load the file from.
        strict: Whether to fail if you're missing keys or having unexpected ones.
        device: The device where the tensors need to be located after load.
        copy: If False (default), returns tensors backed by memory-mapped data
            with copy-on-write semantics (zero-copy reads, much faster).
            If True, returns independent tensors by copying data.

    Returns:
        (missing, unexpected): A tuple of two lists.

    Example:
        >>> from ztensor.torch import load_model
        >>> import torch
        >>> model = torch.nn.Linear(10, 5)
        >>> missing, unexpected = load_model(model, "model.zt")
    """
    state_dict = load_file(filename, device=device, copy=copy)
    model_state_dict = model.state_dict()

    to_removes = _remove_duplicate_names(
        model_state_dict, preferred_names=list(state_dict.keys())
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = set(missing)

    for to_remove_group in to_removes.values():
        for to_remove in to_remove_group:
            if to_remove not in missing:
                unexpected.append(to_remove)
            else:
                missing.remove(to_remove)

    if strict and (missing or unexpected):
        missing_keys = ", ".join([f'"{k}"' for k in sorted(missing)])
        unexpected_keys = ", ".join([f'"{k}"' for k in sorted(unexpected)])
        error = f"Error(s) in loading state_dict for {model.__class__.__name__}:"
        if missing:
            error += f"\n    Missing key(s) in state_dict: {missing_keys}"
        if unexpected:
            error += f"\n    Unexpected key(s) in state_dict: {unexpected_keys}"
        raise RuntimeError(error)

    return list(missing), unexpected


# --- Helper Functions ---


def _torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy array, handling bfloat16."""
    if tensor.dtype == torch.bfloat16:
        # bfloat16: convert via uint16 view of the raw bytes
        try:
            from ml_dtypes import bfloat16 as np_bfloat16

            return tensor.view(torch.int16).numpy().view(np_bfloat16)
        except ImportError:
            # Fallback: store as raw uint16 bytes
            return tensor.view(torch.int16).numpy().view(np.uint16)
    else:
        np_dtype = _TORCH_TO_NP_DTYPE.get(tensor.dtype)
        if np_dtype is None:
            raise ZTensorError(f"Unsupported torch dtype: {tensor.dtype}")
        return tensor.numpy()


def _validate_tensors(tensors: Dict[str, torch.Tensor]) -> None:
    """Validates that all tensors are valid for saving."""
    if not isinstance(tensors, dict):
        raise ValueError(
            f"Expected a dict of [str, torch.Tensor] but received {type(tensors)}"
        )

    sparse_tensors = []
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(
                f"Key `{k}` is invalid, expected torch.Tensor but received {type(v)}"
            )
        if v.layout != torch.strided:
            sparse_tensors.append(k)

    if sparse_tensors:
        raise ValueError(
            f"You are trying to save sparse tensors: `{sparse_tensors}` which this library does not support."
            " You can make it a dense tensor before saving with `.to_dense()` but be aware this might"
            " make a much larger file than needed."
        )

    shared_pointers = _find_shared_tensors(tensors)
    failing = []
    for names in shared_pointers:
        if len(names) > 1:
            failing.append(names)

    if failing:
        failing_info = ", ".join([str(sorted(names)) for names in failing])
        raise ValueError(
            f"Some tensors share memory, this will lead to duplicate data being saved: {failing_info}."
            " Use `save_model` if you want to handle shared tensors automatically."
        )


def _storage_ptr(tensor: torch.Tensor) -> int:
    """Get the storage pointer of a tensor."""
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            return 0


def _storage_size(tensor: torch.Tensor) -> int:
    """Get the storage size of a tensor in bytes."""
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        try:
            return tensor.storage().size() * tensor.element_size()
        except NotImplementedError:
            return tensor.nelement() * tensor.element_size()


def _find_shared_tensors(state_dict: Dict[str, torch.Tensor]) -> List[set]:
    """Find tensors that share storage."""
    from collections import defaultdict

    tensors = defaultdict(set)
    for k, v in state_dict.items():
        if (
            v.device != torch.device("meta")
            and _storage_ptr(v) != 0
            and _storage_size(v) != 0
        ):
            tensors[(v.device, _storage_ptr(v), _storage_size(v))].add(k)

    return list(sorted(tensors.values()))


def _is_complete(tensor: torch.Tensor) -> bool:
    """Check if a tensor covers its entire storage."""
    return tensor.data_ptr() == _storage_ptr(
        tensor
    ) and tensor.nelement() * tensor.element_size() == _storage_size(tensor)


def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: Optional[List[str]] = None,
    discard_names: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Find duplicate tensor names that share storage and determine which to keep.

    Returns a dict mapping kept_name -> [names_to_remove].
    """
    from collections import defaultdict

    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)

    for shared in shareds:
        if len(shared) <= 1:
            continue

        complete_names = set(
            [name for name in shared if _is_complete(state_dict[name])]
        )

        if not complete_names:
            raise RuntimeError(
                "Error while trying to find names to remove to save state dict, but found no suitable name to keep"
                f" for saving amongst: {shared}. None is covering the entire storage. Refusing to save/load the model"
                " since you could be storing much more memory than needed."
            )

        keep_name = sorted(list(complete_names))[0]

        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]

        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)

    return to_remove


__all__ = [
    "save_file",
    "save",
    "load_file",
    "load",
    "save_model",
    "load_model",
]
