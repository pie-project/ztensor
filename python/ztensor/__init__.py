import numpy as np
from .ztensor import ffi, lib

# --- Optional PyTorch Import ---
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- Optional ml_dtypes for bfloat16 in NumPy ---
try:
    from ml_dtypes import bfloat16 as np_bfloat16

    ML_DTYPES_AVAILABLE = True
except ImportError:
    np_bfloat16 = None
    ML_DTYPES_AVAILABLE = False


# --- Pythonic Wrapper ---
class ZTensorError(Exception):
    """Custom exception for ztensor-related errors."""
    pass


# A custom ndarray subclass to safely manage the lifetime of the CFFI pointer.
class _ZTensorView(np.ndarray):
    def __new__(cls, buffer, dtype, shape, view_ptr):
        obj = np.frombuffer(buffer, dtype=dtype).reshape(shape).view(cls)
        # Attach the object that owns the memory to an attribute.
        obj._owner = view_ptr
        return obj

    def __array_finalize__(self, obj):
        # This ensures that views and slices of our array also hold the reference.
        if obj is None: return
        self._owner = getattr(obj, '_owner', None)


def _get_last_error():
    """Retrieves the last error message from the Rust library."""
    err_msg_ptr = lib.ztensor_last_error_message()
    if err_msg_ptr != ffi.NULL:
        return ffi.string(err_msg_ptr).decode('utf-8')
    return "Unknown FFI error"


def _check_ptr(ptr, func_name=""):
    """Checks if a pointer from the FFI is null and raises an error if it is."""
    if ptr == ffi.NULL:
        raise ZTensorError(f"Error in {func_name}: {_get_last_error()}")
    return ptr


def _check_status(status, func_name=""):
    """Checks the integer status code from an FFI call and raises on failure."""
    if status != 0:
        raise ZTensorError(f"Error in {func_name}: {_get_last_error()}")


# --- Type Mappings ---
# NumPy Mappings
DTYPE_NP_TO_ZT = {
    np.dtype('float64'): 'float64', np.dtype('float32'): 'float32', np.dtype('float16'): 'float16',
    np.dtype('int64'): 'int64', np.dtype('int32'): 'int32',
    np.dtype('int16'): 'int16', np.dtype('int8'): 'int8',
    np.dtype('uint64'): 'uint64', np.dtype('uint32'): 'uint32',
    np.dtype('uint16'): 'uint16', np.dtype('uint8'): 'uint8',
    np.dtype('bool'): 'bool',
}
if ML_DTYPES_AVAILABLE:
    DTYPE_NP_TO_ZT[np.dtype(np_bfloat16)] = 'bfloat16'
DTYPE_ZT_TO_NP = {v: k for k, v in DTYPE_NP_TO_ZT.items()}

# PyTorch Mappings (if available)
if TORCH_AVAILABLE:
    DTYPE_TORCH_TO_ZT = {
        torch.float64: 'float64', torch.float32: 'float32', torch.float16: 'float16',
        torch.bfloat16: 'bfloat16',
        torch.int64: 'int64', torch.int32: 'int32',
        torch.int16: 'int16', torch.int8: 'int8',
        torch.uint8: 'uint8', torch.bool: 'bool',
    }
    DTYPE_ZT_TO_TORCH = {v: k for k, v in DTYPE_TORCH_TO_ZT.items()}


class TensorMetadata:
    """A Pythonic wrapper around the CTensorMetadata pointer."""

    def __init__(self, meta_ptr):
        self._ptr = ffi.gc(meta_ptr, lib.ztensor_metadata_free)
        _check_ptr(self._ptr, "TensorMetadata constructor")
        self._name = None
        self._dtype_str = None
        self._shape = None

    @property
    def name(self):
        if self._name is None:
            name_ptr = lib.ztensor_metadata_get_name(self._ptr)
            _check_ptr(name_ptr, "get_name")
            self._name = ffi.string(name_ptr).decode('utf-8')
            lib.ztensor_free_string(name_ptr)
        return self._name

    @property
    def dtype_str(self):
        if self._dtype_str is None:
            dtype_ptr = lib.ztensor_metadata_get_dtype_str(self._ptr)
            _check_ptr(dtype_ptr, "get_dtype_str")
            self._dtype_str = ffi.string(dtype_ptr).decode('utf-8')
            lib.ztensor_free_string(dtype_ptr)
        return self._dtype_str

    @property
    def dtype(self):
        """Returns the numpy dtype for this tensor."""
        dtype_str = self.dtype_str
        dt = DTYPE_ZT_TO_NP.get(dtype_str)
        if dt is None:
            if dtype_str == 'bfloat16':
                raise ZTensorError(
                    "Cannot read 'bfloat16' tensor as NumPy array because the 'ml_dtypes' "
                    "package is not installed. Please install it to proceed."
                )
            raise ZTensorError(f"Unsupported or unknown dtype string '{dtype_str}' found in tensor metadata.")
        return dt

    @property
    def shape(self):
        if self._shape is None:
            shape_len = lib.ztensor_metadata_get_shape_len(self._ptr)
            if shape_len > 0:
                shape_data_ptr = lib.ztensor_metadata_get_shape_data(self._ptr)
                _check_ptr(shape_data_ptr, "get_shape_data")
                self._shape = tuple(shape_data_ptr[i] for i in range(shape_len))
                lib.ztensor_free_u64_array(shape_data_ptr, shape_len)
            else:
                self._shape = tuple()
        return self._shape


class Reader:
    """A Pythonic context manager for reading zTensor files."""

    def __init__(self, file_path):
        path_bytes = file_path.encode('utf-8')
        ptr = lib.ztensor_reader_open(path_bytes)
        _check_ptr(ptr, f"Reader open: {file_path}")
        self._ptr = ffi.gc(ptr, lib.ztensor_reader_free)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ptr = None

    def get_metadata(self, name: str) -> TensorMetadata:
        """Retrieves metadata for a tensor by its name."""
        if self._ptr is None: raise ZTensorError("Reader is closed.")
        name_bytes = name.encode('utf-8')
        meta_ptr = lib.ztensor_reader_get_metadata_by_name(self._ptr, name_bytes)
        _check_ptr(meta_ptr, f"get_metadata: {name}")
        return TensorMetadata(meta_ptr)

    def read_tensor(self, name: str, to: str = 'numpy'):
        """
        Reads a tensor by name and returns it as a NumPy array or PyTorch tensor.
        This is a zero-copy operation for both formats (for CPU tensors).

        Args:
            name (str): The name of the tensor to read.
            to (str): The desired output format. Either 'numpy' (default) or 'torch'.

        Returns:
            np.ndarray or torch.Tensor: The tensor data.
        """
        if to not in ['numpy', 'torch']:
            raise ValueError(f"Unsupported format: '{to}'. Choose 'numpy' or 'torch'.")

        metadata = self.get_metadata(name)
        view_ptr = lib.ztensor_reader_read_tensor_view(self._ptr, metadata._ptr)
        _check_ptr(view_ptr, f"read_tensor: {name}")

        # Let CFFI manage the lifetime of the view pointer.
        view_ptr = ffi.gc(view_ptr, lib.ztensor_free_tensor_view)

        if to == 'numpy':
            # Use the custom _ZTensorView to safely manage the FFI pointer lifetime.
            return _ZTensorView(
                buffer=ffi.buffer(view_ptr.data, view_ptr.len),
                dtype=metadata.dtype,  # This property raises on unsupported dtypes
                shape=metadata.shape,
                view_ptr=view_ptr
            )

        elif to == 'torch':
            if not TORCH_AVAILABLE:
                raise ZTensorError("PyTorch is not installed. Cannot return a torch tensor.")

            # Get the corresponding torch dtype, raising if not supported.
            torch_dtype = DTYPE_ZT_TO_TORCH.get(metadata.dtype_str)
            if torch_dtype is None:
                raise ZTensorError(
                    f"Cannot read tensor '{name}' as a PyTorch tensor. "
                    f"The dtype '{metadata.dtype_str}' is not supported by PyTorch."
                )

            # Create a tensor directly from the buffer to avoid numpy conversion issues.
            buffer = ffi.buffer(view_ptr.data, view_ptr.len)
            torch_tensor = torch.frombuffer(buffer, dtype=torch_dtype).reshape(metadata.shape)

            # CRITICAL: Attach the memory owner to the tensor to manage its lifetime.
            # This ensures the Rust memory (held by view_ptr) is not freed while the
            # torch tensor is still in use.
            torch_tensor._owner = view_ptr

            return torch_tensor


class Writer:
    """A Pythonic context manager for writing zTensor files."""

    def __init__(self, file_path):
        path_bytes = file_path.encode('utf-8')
        ptr = lib.ztensor_writer_create(path_bytes)
        _check_ptr(ptr, f"Writer create: {file_path}")
        self._ptr = ptr
        self._finalized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ptr and not self._finalized:
            if exc_type is None:
                self.finalize()
            else:
                lib.ztensor_writer_free(self._ptr)
                self._ptr = None

    def add_tensor(self, name: str, tensor):
        """
        Adds a NumPy or PyTorch tensor to the file (zero-copy).
        Supports float16 and bfloat16 types.

        Args:
            name (str): The name of the tensor to add.
            tensor (np.ndarray or torch.Tensor): The tensor data to write.
        """
        if not self._ptr: raise ZTensorError("Writer is closed or finalized.")

        # --- Polymorphic tensor handling ---
        if isinstance(tensor, np.ndarray):
            tensor = np.ascontiguousarray(tensor)
            shape = tensor.shape
            dtype_str = DTYPE_NP_TO_ZT.get(tensor.dtype)
            data_ptr = ffi.cast("unsigned char*", tensor.ctypes.data)
            nbytes = tensor.nbytes

        elif TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                raise ZTensorError("Cannot write directly from a CUDA tensor. Copy to CPU first using .cpu().")
            tensor = tensor.contiguous()
            shape = tuple(tensor.shape)
            dtype_str = DTYPE_TORCH_TO_ZT.get(tensor.dtype)
            data_ptr = ffi.cast("unsigned char*", tensor.data_ptr())
            nbytes = tensor.numel() * tensor.element_size()

        else:
            supported = "np.ndarray" + (" or torch.Tensor" if TORCH_AVAILABLE else "")
            raise TypeError(f"Unsupported tensor type: {type(tensor)}. Must be {supported}.")

        if not dtype_str:
            msg = f"Unsupported dtype: {tensor.dtype}."
            if 'bfloat16' in str(tensor.dtype) and not ML_DTYPES_AVAILABLE:
                msg += " For NumPy bfloat16 support, please install the 'ml_dtypes' package."
            raise ZTensorError(msg)

        name_bytes = name.encode('utf-8')
        shape_array = np.array(shape, dtype=np.uint64)
        shape_ptr = ffi.cast("uint64_t*", shape_array.ctypes.data)
        dtype_bytes = dtype_str.encode('utf-8')

        status = lib.ztensor_writer_add_tensor(
            self._ptr, name_bytes, shape_ptr, len(shape),
            dtype_bytes, data_ptr, nbytes
        )
        _check_status(status, f"add_tensor: {name}")

    def finalize(self):
        """Finalizes the zTensor file, writing the metadata index."""
        if not self._ptr: raise ZTensorError("Writer is already closed or finalized.")
        status = lib.ztensor_writer_finalize(self._ptr)
        self._ptr = None
        self._finalized = True
        _check_status(status, "finalize")


__all__ = ["Reader", "Writer", "TensorMetadata", "ZTensorError"]
