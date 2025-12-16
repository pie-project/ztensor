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
        # Cache for properties to avoid repeated FFI calls
        self._name = None
        self._dtype_str = None
        self._shape = None
        self._offset = None
        self._size = None
        self._layout = None
        self._encoding = None
        self._endianness = "not_checked"
        self._checksum = "not_checked"

    def __repr__(self):
        return f"<TensorMetadata name='{self.name}' shape={self.shape} dtype='{self.dtype_str}'>"

    @property
    def name(self):
        """The name of the tensor."""
        if self._name is None:
            name_ptr = lib.ztensor_metadata_get_name(self._ptr)
            _check_ptr(name_ptr, "get_name")
            self._name = ffi.string(name_ptr).decode('utf-8')
            lib.ztensor_free_string(name_ptr)
        return self._name

    @property
    def dtype_str(self):
        """The zTensor dtype string (e.g., 'float32')."""
        if self._dtype_str is None:
            dtype_ptr = lib.ztensor_metadata_get_dtype_str(self._ptr)
            _check_ptr(dtype_ptr, "get_dtype_str")
            self._dtype_str = ffi.string(dtype_ptr).decode('utf-8')
            lib.ztensor_free_string(dtype_ptr)
        return self._dtype_str

    @property
    def dtype(self):
        """The numpy dtype for this tensor."""
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
        """The shape of the tensor as a tuple."""
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

    @property
    def offset(self):
        """The on-disk offset of the tensor data in bytes."""
        if self._offset is None:
            self._offset = lib.ztensor_metadata_get_offset(self._ptr)
        return self._offset

    @property
    def size(self):
        """The on-disk size of the tensor data in bytes (can be compressed size)."""
        if self._size is None:
            self._size = lib.ztensor_metadata_get_size(self._ptr)
        return self._size

    @property
    def layout(self):
        """The tensor layout as a string (e.g., 'dense')."""
        if self._layout is None:
            layout_ptr = lib.ztensor_metadata_get_layout_str(self._ptr)
            _check_ptr(layout_ptr, "get_layout_str")
            self._layout = ffi.string(layout_ptr).decode('utf-8')
            lib.ztensor_free_string(layout_ptr)
        return self._layout

    @property
    def encoding(self):
        """The tensor encoding as a string (e.g., 'raw', 'zstd')."""
        if self._encoding is None:
            encoding_ptr = lib.ztensor_metadata_get_encoding_str(self._ptr)
            if encoding_ptr == ffi.NULL:
                self._encoding = None
            else:
                self._encoding = ffi.string(encoding_ptr).decode('utf-8')
                lib.ztensor_free_string(encoding_ptr)
        return self._encoding

    @property
    def endianness(self):
        """The data endianness ('little', 'big') if applicable, else None."""
        if self._endianness == "not_checked":
            endian_ptr = lib.ztensor_metadata_get_data_endianness_str(self._ptr)
            if endian_ptr == ffi.NULL:
                self._endianness = None
            else:
                self._endianness = ffi.string(endian_ptr).decode('utf-8')
                lib.ztensor_free_string(endian_ptr)
        return self._endianness

    @property
    def checksum(self):
        """The checksum string if present, else None."""
        if self._checksum == "not_checked":
            checksum_ptr = lib.ztensor_metadata_get_checksum_str(self._ptr)
            if checksum_ptr == ffi.NULL:
                self._checksum = None
            else:
                self._checksum = ffi.string(checksum_ptr).decode('utf-8')
                lib.ztensor_free_string(checksum_ptr)
        return self._checksum


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

    def __len__(self):
        """Returns the number of tensors in the file."""
        if self._ptr is None: raise ZTensorError("Reader is closed.")
        return lib.ztensor_reader_get_metadata_count(self._ptr)

    def __iter__(self):
        """Iterates over the metadata of all tensors in the file."""
        if self._ptr is None: raise ZTensorError("Reader is closed.")
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, key):
        """
        Retrieves metadata (int key) or reads tensor data (str key).
        
        Args:
            key: If int, returns TensorMetadata at that index.
                 If str, returns the tensor data (calls read_tensor).
        """
        if self._ptr is None: raise ZTensorError("Reader is closed.")
        
        if isinstance(key, int):
            if key >= len(self):
                raise IndexError("Tensor index out of range")
            meta_ptr = lib.ztensor_reader_get_metadata_by_index(self._ptr, key)
            _check_ptr(meta_ptr, f"get_metadata_by_index: {key}")
            return TensorMetadata(meta_ptr)
        elif isinstance(key, str):
            return self.read_tensor(key)
        else:
            raise TypeError(f"Invalid argument type for __getitem__: {type(key)}")

    def __contains__(self, name: str) -> bool:
        """Checks if a tensor with the given name exists in the file."""
        return name in self.tensor_names

    @property
    def tensors(self) -> list[TensorMetadata]:
        """Returns a list of all TensorMetadata objects in the file."""
        return list(self)

    @property
    def tensor_names(self) -> list[str]:
        """Returns a list of all tensor names in the file."""
        if self._ptr is None: raise ZTensorError("Reader is closed.")
        c_array_ptr = lib.ztensor_reader_get_all_tensor_names(self._ptr)
        _check_ptr(c_array_ptr, "get_all_tensor_names")
        c_array_ptr = ffi.gc(c_array_ptr, lib.ztensor_free_string_array)

        return [ffi.string(c_array_ptr.strings[i]).decode('utf-8') for i in range(c_array_ptr.len)]

    def metadata(self, name: str) -> TensorMetadata:
        """Retrieves metadata for a tensor by its name."""
        if self._ptr is None: raise ZTensorError("Reader is closed.")
        name_bytes = name.encode('utf-8')
        meta_ptr = lib.ztensor_reader_get_metadata_by_name(self._ptr, name_bytes)
        _check_ptr(meta_ptr, f"metadata: {name}")
        return TensorMetadata(meta_ptr)

    # Legacy alias for backward compatibility (optional, but good practice if not actively removing)
    def list_tensors(self): return self.tensors
    def get_tensor_names(self): return self.tensor_names
    def get_metadata(self, name): return self.metadata(name)

    def _read_component(self, tensor_name: str, component_name: str, dtype_func):
        """Reads a specific component as a numpy array."""
        t_name_bytes = tensor_name.encode('utf-8')
        c_name_bytes = component_name.encode('utf-8')
        
        view_ptr = lib.ztensor_reader_read_tensor_component(self._ptr, t_name_bytes, c_name_bytes)
        _check_ptr(view_ptr, f"read_component: {tensor_name}.{component_name}")
        view_ptr = ffi.gc(view_ptr, lib.ztensor_free_tensor_view)
        
        buffer = ffi.buffer(view_ptr.data, view_ptr.len)
        # Interpret bytes specific to the component type
        arr = np.frombuffer(buffer, dtype=dtype_func())
        return arr, view_ptr

    def read_tensor(self, name: str, to: str = 'numpy'):
        """
        Reads a tensor by name and returns it as a NumPy array or PyTorch tensor.
        Supports 'dense' (returns array/tensor) and 'sparse_csr'/'sparse_coo' (returns Scipy/Torch sparse).

        Args:
            name (str): The name of the tensor to read.
            to (str): The desired output format. 'numpy' (returns scipy.sparse for sparse) or 'torch'.

        Returns:
            Union[np.ndarray, torch.Tensor, scipy.sparse.spmatrix, torch.sparse_coo_tensor]
        """
        if self._ptr is None: raise ZTensorError("Reader is closed.")
        if to not in ['numpy', 'torch']:
             raise ValueError(f"Unsupported format: '{to}'.")

        metadata = self.metadata(name)
        layout = metadata.layout

        if layout == "dense":
            view_ptr = lib.ztensor_reader_read_tensor_view(self._ptr, metadata._ptr)
            _check_ptr(view_ptr, f"read_tensor: {name}")
            view_ptr = ffi.gc(view_ptr, lib.ztensor_free_tensor_view)

            if to == 'numpy':
                return _ZTensorView(
                    buffer=ffi.buffer(view_ptr.data, view_ptr.len),
                    dtype=metadata.dtype,
                    shape=metadata.shape,
                    view_ptr=view_ptr
                )
            elif to == 'torch':
                 if not TORCH_AVAILABLE: raise ZTensorError("PyTorch not installed.")
                 torch_dtype = DTYPE_ZT_TO_TORCH.get(metadata.dtype_str)
                 buffer = ffi.buffer(view_ptr.data, view_ptr.len)
                 torch_tensor = torch.frombuffer(buffer, dtype=torch_dtype).reshape(metadata.shape)
                 torch_tensor._owner = view_ptr
                 return torch_tensor

        elif layout == "sparse_csr":
            # Components: values (T), indices (u64), indptr (u64)
            vals, v_ref = self._read_component(name, "values", lambda: metadata.dtype)
            idxs, i_ref = self._read_component(name, "indices", lambda: np.uint64)
            ptrs, p_ref = self._read_component(name, "indptr", lambda: np.uint64)
            
            # Since FFI returns raw bytes swapped to native, we can trust np.frombuffer behavior.
            # Sparse indices in scipy/torch are usually int32 or int64.
            
            if to == 'numpy':
                # Return scipy.sparse.csr_matrix
                try:
                    from scipy.sparse import csr_matrix
                except ImportError:
                    raise ZTensorError("scipy is required for reading sparse tensors as numpy.")
                
                # Scipy needs appropriate index types (int32/int64).
                return csr_matrix((vals, idxs.astype(np.int32), ptrs.astype(np.int32)), shape=metadata.shape)
            
            elif to == 'torch':
                if not TORCH_AVAILABLE: raise ZTensorError("No Torch.")
                # Torch CSR: torch.sparse_csr_tensor(crow_indices, col_indices, values, size=None)
                # crowd_indices = indptr.
                t_vals = torch.from_numpy(vals).clone() # Copied from buffer?
                # We need to copy because ZTensorView might expire? 
                # Actually _read_component returns numpy wrapper on buffer.
                # If we construct torch tensor from it, we need to keep reference.
                # But sparse tensors in torch usually own their indices/values.
                # Let's clone to be safe and simple.
                t_indptr = torch.from_numpy(ptrs.astype(np.int64)).clone() 
                t_indices = torch.from_numpy(idxs.astype(np.int64)).clone()
                
                return torch.sparse_csr_tensor(t_indptr, t_indices, t_vals, size=metadata.shape)

        elif layout == "sparse_coo":
            # Components: values (T), coords (u64, shape=(ndim*nnz))
            vals, v_ref = self._read_component(name, "values", lambda: metadata.dtype)
            coords, c_ref = self._read_component(name, "coords", lambda: np.uint64)
            
            nnz = vals.shape[0]
            ndim = len(metadata.shape)
            # coords is flat [dim0... dim1...].
            # Reshape to (ndim, nnz)
            coords = coords.reshape((ndim, nnz))
            
            if to == 'numpy':
                 # Scipy coo_matrix: (data, (row, col))
                 if ndim != 2: raise ZTensorError("Scipy COO only supports 2D.")
                 from scipy.sparse import coo_matrix
                 return coo_matrix((vals, (coords[0], coords[1])), shape=metadata.shape)
            
            elif to == 'torch':
                 if not TORCH_AVAILABLE: raise ZTensorError("No Torch.")
                 t_vals = torch.from_numpy(vals).clone()
                 t_indices = torch.from_numpy(coords.astype(np.int64)).clone()
                 return torch.sparse_coo_tensor(t_indices, t_vals, size=metadata.shape)

        else:
            raise ZTensorError(f"Unsupported layout: {layout}")


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
                # If an error occurred, just free the handle without finalizing
                lib.ztensor_writer_free(self._ptr)
                self._ptr = None

    def add_tensor(self, name: str, tensor, compress: bool = False):
        """
        Adds a NumPy or PyTorch tensor to the file (zero-copy).
        Supports float16 and bfloat16 types.

        Args:
            name (str): The name of the tensor to add.
            tensor (np.ndarray or torch.Tensor): The tensor data to write.
            compress (bool): If True, compress the tensor data using zstd. Default: False.
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
            dtype_bytes, data_ptr, nbytes, 1 if compress else 0
        )
        _check_status(status, f"add_tensor: {name}")

    def add_sparse_csr(self, name: str, values, indices, indptr, shape):
        """
        Adds a sparse CSR tensor.
        
        Args:
            name: Tensor name.
            values: Numpy/Torch array of values.
            indices: Numpy/Torch array of column indices.
            indptr: Numpy/Torch array of index pointers.
            shape: Tuple of (rows, cols).
            
        Example:
            >>> csr = scipy.sparse.csr_matrix([[1, 0], [0, 2]])
            >>> writer.add_sparse_csr("my_csr", csr.data, csr.indices, csr.indptr, csr.shape)
        """
        if not self._ptr: raise ZTensorError("Writer is closed.")

        # Helper to get ptr and len for generic array
        def get_buffer_info(arr):
            if isinstance(arr, np.ndarray):
                arr = np.ascontiguousarray(arr)
                ptr = ffi.cast("unsigned char*", arr.ctypes.data)
                length = arr.nbytes
                count = arr.size
                dtype_str = DTYPE_NP_TO_ZT.get(arr.dtype)
                return arr, ptr, length, count, dtype_str
            elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
                if arr.is_cuda: raise ZTensorError("CUDA tensors not supported. Move to CPU.")
                arr = arr.contiguous()
                ptr = ffi.cast("unsigned char*", arr.data_ptr())
                length = arr.numel() * arr.element_size()
                count = arr.numel()
                dtype_str = DTYPE_TORCH_TO_ZT.get(arr.dtype)
                return arr, ptr, length, count, dtype_str
            else:
                raise TypeError(f"Unsupported array type: {type(arr)}")

        # Process inputs
        v_arr, v_ptr, v_len, _, v_dtype = get_buffer_info(values)
        i_arr, i_ptr, i_len, i_cnt, _ = get_buffer_info(indices)
        p_arr, p_ptr, p_len, p_cnt, _ = get_buffer_info(indptr)

        def ensure_u64_ptr(arr):
             # If numpy, cast to uint64 if not already
            if isinstance(arr, np.ndarray):
                 if arr.dtype != np.uint64:
                     arr = arr.astype(np.uint64) # Safe copy
                 arr = np.ascontiguousarray(arr)
                 return ffi.cast("uint64_t*", arr.ctypes.data), arr
            elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
                 # Torch doesn't strictly have uint64 (it has int64). Reinterpret cast is risky if negative.
                 # Indices shouldn't be negative.
                 # cffi cast int64* to uint64* is okay bitwise.
                 if arr.dtype != torch.int64 and arr.dtype != torch.int32: # what if int32?
                     raise TypeError("Indices must be integer type.")
                 # If int32, we MUST convert to int64/uint64 because FFI reads 64-bit strides.
                 if arr.dtype == torch.int32:
                      arr = arr.to(torch.int64)
                 arr = arr.contiguous()
                 return ffi.cast("uint64_t*", arr.data_ptr()), arr
            else:
                 raise TypeError("Unsupported type")

        i_ptr_u64, _i_keep = ensure_u64_ptr(indices)
        p_ptr_u64, _p_keep = ensure_u64_ptr(indptr)
        
        # Prepare metadata
        name_bytes = name.encode('utf-8')
        shape_array = np.array(shape, dtype=np.uint64)
        shape_ptr = ffi.cast("uint64_t*", shape_array.ctypes.data)
        dtype_bytes = v_dtype.encode('utf-8')

        status = lib.ztensor_writer_add_sparse_csr(
            self._ptr,
            name_bytes,
            shape_ptr, len(shape),
            dtype_bytes,
            v_ptr, v_len,
            i_ptr_u64, indices.shape[0] if hasattr(indices, 'shape') else len(indices),
            p_ptr_u64, indptr.shape[0] if hasattr(indptr, 'shape') else len(indptr),
        )
        _check_status(status, f"add_sparse_csr: {name}")

    def add_sparse_coo(self, name: str, values, indices, shape):
        """
        Adds a sparse COO tensor.
        
        Args:
            name: Tensor name
            values: Values array
            indices: Matrix of coordinates (ndim x nnz) in C-order (row-major).
            shape: Tensor shape
        """
        if not self._ptr: raise ZTensorError("Writer is closed.")
        
        def get_buffer_info(arr):
             if isinstance(arr, np.ndarray):
                 arr = np.ascontiguousarray(arr)
                 return arr, ffi.cast("unsigned char*", arr.ctypes.data), arr.nbytes, DTYPE_NP_TO_ZT.get(arr.dtype)
             elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
                 if arr.is_cuda: raise ZTensorError("No CUDA")
                 arr = arr.contiguous()
                 return arr, ffi.cast("unsigned char*", arr.data_ptr()), arr.numel() * arr.element_size(), DTYPE_TORCH_TO_ZT.get(arr.dtype)
             raise TypeError("Unsupported")

        v_arr, v_ptr, v_len, v_dtype = get_buffer_info(values)
        
        # indices
        def ensure_u64_ptr(arr):
            if isinstance(arr, np.ndarray):
                 if arr.dtype != np.uint64: arr = arr.astype(np.uint64) # Safe copy
                 arr = np.ascontiguousarray(arr)
                 return ffi.cast("uint64_t*", arr.ctypes.data), arr, arr.size
            elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
                 if arr.dtype != torch.int64: arr = arr.to(torch.int64)
                 arr = arr.contiguous()
                 return ffi.cast("uint64_t*", arr.data_ptr()), arr, arr.numel()
            raise TypeError("Unsupported")

        i_ptr_u64, _i_keep, i_count = ensure_u64_ptr(indices)

        name_bytes = name.encode('utf-8')
        shape_array = np.array(shape, dtype=np.uint64)
        shape_ptr = ffi.cast("uint64_t*", shape_array.ctypes.data)
        dtype_bytes = v_dtype.encode('utf-8')

        status = lib.ztensor_writer_add_sparse_coo(
            self._ptr,
            name_bytes,
            shape_ptr, len(shape),
            dtype_bytes,
            v_ptr, v_len,
            i_ptr_u64, i_count
        )
        _check_status(status, f"add_sparse_coo: {name}")

    def finalize(self):
        """Finalizes the zTensor file, writing the metadata index."""
        if not self._ptr: raise ZTensorError("Writer is already closed or finalized.")
        status = lib.ztensor_writer_finalize(self._ptr)
        self._ptr = None  # The writer is consumed in Rust
        self._finalized = True
        _check_status(status, "finalize")


__all__ = ["Reader", "Writer", "TensorMetadata", "ZTensorError"]
