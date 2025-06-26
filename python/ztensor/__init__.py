import numpy as np
from .ztensor import ffi, lib


# --- Pythonic Wrapper ---
class ZTensorError(Exception):
    """Custom exception for ztensor-related errors."""
    pass


# A custom ndarray subclass to safely manage the lifetime of the CFFI pointer.
class _ZTensorView(np.ndarray):
    def __new__(cls, buffer, dtype, shape, view_ptr):
        # Create an array from the buffer, reshape it, and cast it to our custom type.
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


# Type Mappings between NumPy and ztensor
DTYPE_NP_TO_ZT = {
    np.dtype('float64'): 'float64', np.dtype('float32'): 'float32',
    np.dtype('int64'): 'int64', np.dtype('int32'): 'int32',
    np.dtype('int16'): 'int16', np.dtype('int8'): 'int8',
    np.dtype('uint64'): 'uint64', np.dtype('uint32'): 'uint32',
    np.dtype('uint16'): 'uint16', np.dtype('uint8'): 'uint8',
    np.dtype('bool'): 'bool',
}
DTYPE_ZT_TO_NP = {v: k for k, v in DTYPE_NP_TO_ZT.items()}


class TensorMetadata:
    """A Pythonic wrapper around the CTensorMetadata pointer."""

    def __init__(self, meta_ptr):
        # The pointer is now automatically garbage collected by CFFI when this object dies.
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
            # ffi.string creates a copy, so we must free the Rust-allocated original.
            self._name = ffi.string(name_ptr).decode('utf-8')
            lib.ztensor_free_string(name_ptr)
        return self._name

    @property
    def dtype_str(self):
        if self._dtype_str is None:
            dtype_ptr = lib.ztensor_metadata_get_dtype_str(self._ptr)
            _check_ptr(dtype_ptr, "get_dtype_str")
            # ffi.string creates a copy, so we must free the Rust-allocated original.
            self._dtype_str = ffi.string(dtype_ptr).decode('utf-8')
            lib.ztensor_free_string(dtype_ptr)
        return self._dtype_str

    @property
    def dtype(self):
        """Returns the numpy dtype for this tensor."""
        return DTYPE_ZT_TO_NP.get(self.dtype_str)

    # RE-ENABLED: This property now works because the underlying FFI functions are available.
    @property
    def shape(self):
        if self._shape is None:
            shape_len = lib.ztensor_metadata_get_shape_len(self._ptr)
            if shape_len > 0:
                shape_data_ptr = lib.ztensor_metadata_get_shape_data(self._ptr)
                _check_ptr(shape_data_ptr, "get_shape_data")
                self._shape = tuple(shape_data_ptr[i] for i in range(shape_len))
                # Free the array that was allocated on the Rust side.
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
        # The pointer is automatically garbage collected by CFFI.
        self._ptr = ffi.gc(ptr, lib.ztensor_reader_free)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # CFFI's garbage collector handles freeing the reader pointer automatically.
        # No explicit free is needed here, simplifying the context manager.
        self._ptr = None

    def get_metadata(self, name: str) -> TensorMetadata:
        """Retrieves metadata for a tensor by its name."""
        if self._ptr is None: raise ZTensorError("Reader is closed.")
        name_bytes = name.encode('utf-8')
        meta_ptr = lib.ztensor_reader_get_metadata_by_name(self._ptr, name_bytes)
        _check_ptr(meta_ptr, f"get_metadata: {name}")
        return TensorMetadata(meta_ptr)

    def read_tensor(self, name: str) -> np.ndarray:
        """Reads a tensor by name and returns it as a NumPy array (zero-copy)."""
        metadata = self.get_metadata(name)
        view_ptr = lib.ztensor_reader_read_tensor_view(self._ptr, metadata._ptr)
        _check_ptr(view_ptr, f"read_tensor: {name}")

        # Let CFFI manage the lifetime of the view pointer.
        view_ptr = ffi.gc(view_ptr, lib.ztensor_free_tensor_view)

        # CORRECTED: Create array using the subclass, which handles reshaping and memory.
        array = _ZTensorView(
            buffer=ffi.buffer(view_ptr.data, view_ptr.len),
            dtype=metadata.dtype,
            shape=metadata.shape,
            view_ptr=view_ptr
        )

        return array


class Writer:
    """A Pythonic context manager for writing zTensor files."""

    def __init__(self, file_path):
        path_bytes = file_path.encode('utf-8')
        ptr = lib.ztensor_writer_create(path_bytes)
        _check_ptr(ptr, f"Writer create: {file_path}")
        # The pointer is consumed by finalize, so we don't use ffi.gc here.
        # The writer should be freed via finalize or ztensor_writer_free if finalize fails.
        self._ptr = ptr
        self._finalized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Automatically finalize on exit if not already done and no error occurred.
        if self._ptr and not self._finalized:
            if exc_type is None:
                self.finalize()
            else:
                # If an error occurred, don't finalize, just free the writer to prevent leaks.
                lib.ztensor_writer_free(self._ptr)
                self._ptr = None

    def add_tensor(self, name: str, tensor: np.ndarray):
        """Adds a NumPy array as a tensor to the file."""
        if not self._ptr: raise ZTensorError("Writer is closed or finalized.")

        name_bytes = name.encode('utf-8')
        tensor = np.ascontiguousarray(tensor)  # Ensure data is contiguous.

        shape_array = np.array(tensor.shape, dtype=np.uint64)
        shape_ptr = ffi.cast("uint64_t*", shape_array.ctypes.data)

        dtype_str = DTYPE_NP_TO_ZT.get(tensor.dtype)
        if not dtype_str:
            raise ZTensorError(f"Unsupported NumPy dtype: {tensor.dtype}")
        dtype_bytes = dtype_str.encode('utf-8')

        # CORRECTED: Cast to `unsigned char*` to match the CFFI definition and Rust FFI.
        data_ptr = ffi.cast("unsigned char*", tensor.ctypes.data)

        status = lib.ztensor_writer_add_tensor(
            self._ptr, name_bytes, shape_ptr, len(tensor.shape),
            dtype_bytes, data_ptr, tensor.nbytes
        )
        _check_status(status, f"add_tensor: {name}")

    def finalize(self):
        """Finalizes the zTensor file, writing the metadata index."""
        if not self._ptr: raise ZTensorError("Writer is already closed or finalized.")

        status = lib.ztensor_writer_finalize(self._ptr)
        self._ptr = None  # The writer pointer is consumed and invalidated by the Rust call.
        self._finalized = True
        _check_status(status, "finalize")


__all__ = ["Reader", "Writer", "TensorMetadata", "ZTensorError"]
