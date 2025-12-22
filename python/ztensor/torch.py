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

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required to use ztensor.torch. "
        "Please install it with: pip install torch"
    )

from . import Reader, Writer, ZTensorError


def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
    compression: Union[bool, int] = False,
) -> None:
    """
    Saves a dictionary of tensors into `filename` in ztensor format.

    There is no mechanism in place to prevent the caller from modifying
    the data while a file save occurs. Please be wary when calling
    `save_file` and modifying tensors referenced in the `tensors` dict
    concurrently; it may lead to corrupted files.

    Args:
        tensors: The incoming tensors. Tensors need to be contiguous and dense.
        filename: The filename we're saving into.
        metadata: Optional text only metadata you might want to save in your
            header. For instance it can be useful to specify more about the
            underlying tensors. This is purely informative and does not
            affect tensor loading.
            NOTE: ztensor does not currently support custom metadata; this
            parameter is accepted for API compatibility with safetensors
            binary format.
        compression: Compression settings. False/0 (raw), True (level 3), or int > 0 (level).
            Default: False.

    Returns:
        None

    Example:
        >>> from ztensor.torch import save_file
        >>> import torch
        >>> tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
        >>> save_file(tensors, "model.zt")
    """
    _validate_tensors(tensors)
    
    with Writer(str(filename)) as writer:
        for name, tensor in tensors.items():
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            writer.add_tensor(name, tensor, compress=compression)


def save(
    tensors: Dict[str, torch.Tensor],
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
        >>> from ztensor.torch import save
        >>> import torch
        >>> tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
        >>> byte_data = save(tensors)
    """
    _validate_tensors(tensors)
    
    # Create a temporary file, write to it, read the bytes, then clean up
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
) -> Dict[str, torch.Tensor]:
    """
    Loads a ztensor file into torch format.

    Args:
        filename: The name of the file which contains the tensors.
        device: The device where the tensors need to be located after load.
            Available options are all regular torch device locations.

    Returns:
        Dictionary that contains name as key, value as `torch.Tensor`.

    Example:
        >>> from ztensor.torch import load_file
        >>> file_path = "./my_folder/bert.zt"
        >>> loaded = load_file(file_path)
    """
    # Normalize device
    if isinstance(device, int):
        device = f"cuda:{device}"
    target_device = torch.device(device)
    
    # Create Reader instance directly without context manager to support zero-copy
    reader = Reader(str(filename))
    try:
        names = reader.tensor_names
        if not names:
            return {}
        # Use batch API for efficiency
        tensors = reader.read_tensors(names, to='torch')
        result = {}
        for name, tensor in zip(names, tensors):
            if target_device.type != "cpu":
                # This performs a copy to the device
                tensor = tensor.to(target_device)
            result[name] = tensor
        return result
    except Exception:
        lib.ztensor_reader_free(reader._ptr)
        reader._ptr = None
        raise


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
    # Write bytes to a temporary file and then load
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zt") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    
    try:
        return load_file(tmp_path, device="cpu")
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
            Some metadata will be added for each dropped tensors.
            This information will not be enough to recover the entire
            shared structure but might help understanding things.
            NOTE: ztensor does not currently support custom metadata; this
            parameter is accepted for API compatibility with safetensors.
        force_contiguous: Forcing the state_dict to be saved as contiguous
            tensors. This has no effect on the correctness of the model,
            but it could potentially change performance if the layout of
            the tensor was chosen specifically for that reason.

    Example:
        >>> from ztensor.torch import save_model
        >>> import torch
        >>> model = torch.nn.Linear(10, 5)
        >>> save_model(model, "model.zt")
    """
    state_dict = model.state_dict()
    
    # Handle tensor sharing by removing duplicates
    to_removes = _remove_duplicate_names(state_dict)
    
    if metadata is None:
        metadata = {}
    
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                # Record which tensor this was mapped to
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
) -> Tuple[List[str], List[str]]:
    """
    Loads a given filename onto a torch model.

    This method handles tensor sharing issues which are not allowed in
    ztensor format.

    Args:
        model: The model to load onto.
        filename: The filename location to load the file from.
        strict: Whether to fail if you're missing keys or having unexpected ones.
            When false, the function simply returns missing and unexpected names.
        device: The device where the tensors need to be located after load.
            Available options are all regular torch device locations.

    Returns:
        (missing, unexpected): A tuple of two lists.
            `missing` are names in the model which were not modified during loading.
            `unexpected` are names that are on the file, but weren't used during
            the load.

    Example:
        >>> from ztensor.torch import load_model
        >>> import torch
        >>> model = torch.nn.Linear(10, 5)
        >>> missing, unexpected = load_model(model, "model.zt")
    """
    state_dict = load_file(filename, device=device)
    model_state_dict = model.state_dict()
    
    # Handle tensor sharing
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
    
    # Check for shared tensors
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
        # Fallback for older torch versions
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            # Fallback for meta storage
            return 0


def _storage_size(tensor: torch.Tensor) -> int:
    """Get the storage size of a tensor in bytes."""
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        # Fallback for older torch versions
        try:
            return tensor.storage().size() * tensor.element_size()
        except NotImplementedError:
            # Fallback for meta storage
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
            # Need to add device as key because of multiple GPU
            tensors[(v.device, _storage_ptr(v), _storage_size(v))].add(k)
    
    return list(sorted(tensors.values()))


def _is_complete(tensor: torch.Tensor) -> bool:
    """Check if a tensor covers its entire storage."""
    return (
        tensor.data_ptr() == _storage_ptr(tensor) and
        tensor.nelement() * tensor.element_size() == _storage_size(tensor)
    )


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
        
        # Mechanism to preferentially select keys to keep
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
