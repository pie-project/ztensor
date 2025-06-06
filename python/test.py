import ctypes
import os
import platform

# --- Load the shared library ---
lib_name = ""
if platform.system() == "Linux":
    lib_name = "libztensor.so"
elif platform.system() == "Darwin": # macOS
    lib_name = "libztensor.dylib"
elif platform.system() == "Windows":
    lib_name = "ztensor.dll"
else:
    raise OSError("Unsupported platform")

# Adjust path to where your compiled library is (e.g., target/release/ or target/debug/)
# This assumes the library is in a 'target/debug' or 'target/release' folder relative to this script's parent
script_dir = os.path.dirname(os.path.abspath(__file__))
# Try release first, then debug
lib_path_release = os.path.join(script_dir, "..", "target", "release", lib_name)
lib_path_debug = os.path.join(script_dir, "..", "target", "debug", lib_name)

libztensor = None
if os.path.exists(lib_path_release):
    libztensor = ctypes.CDLL(lib_path_release)
elif os.path.exists(lib_path_debug):
    libztensor = ctypes.CDLL(lib_path_debug)
else:
    raise OSError(f"Shared library not found at {lib_path_release} or {lib_path_debug}")


# --- Define C function signatures ---

# Error handling
libztensor.ztensor_last_error_message.restype = ctypes.c_char_p

# Reader
libztensor.ztensor_reader_open.argtypes = [ctypes.c_char_p]
libztensor.ztensor_reader_open.restype = ctypes.c_void_p # CZTensorReader*

libztensor.ztensor_reader_free.argtypes = [ctypes.c_void_p]
libztensor.ztensor_reader_free.restype = None

libztensor.ztensor_reader_get_metadata_count.argtypes = [ctypes.c_void_p]
libztensor.ztensor_reader_get_metadata_count.restype = ctypes.c_size_t

libztensor.ztensor_reader_get_metadata_by_index.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libztensor.ztensor_reader_get_metadata_by_index.restype = ctypes.c_void_p # CTensorMetadata*

libztensor.ztensor_reader_get_metadata_by_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libztensor.ztensor_reader_get_metadata_by_name.restype = ctypes.c_void_p # CTensorMetadata*

# Metadata
libztensor.ztensor_metadata_free.argtypes = [ctypes.c_void_p]
libztensor.ztensor_metadata_free.restype = None

libztensor.ztensor_metadata_get_name.argtypes = [ctypes.c_void_p]
libztensor.ztensor_metadata_get_name.restype = ctypes.c_char_p # Returns allocated string

libztensor.ztensor_metadata_get_dtype_str.argtypes = [ctypes.c_void_p]
libztensor.ztensor_metadata_get_dtype_str.restype = ctypes.c_char_p

libztensor.ztensor_metadata_get_encoding_str.argtypes = [ctypes.c_void_p]
libztensor.ztensor_metadata_get_encoding_str.restype = ctypes.c_char_p

libztensor.ztensor_metadata_get_offset.argtypes = [ctypes.c_void_p]
libztensor.ztensor_metadata_get_offset.restype = ctypes.c_uint64

libztensor.ztensor_metadata_get_size.argtypes = [ctypes.c_void_p]
libztensor.ztensor_metadata_get_size.restype = ctypes.c_uint64

libztensor.ztensor_metadata_get_shape_len.argtypes = [ctypes.c_void_p]
libztensor.ztensor_metadata_get_shape_len.restype = ctypes.c_size_t

libztensor.ztensor_metadata_get_shape_data.argtypes = [ctypes.c_void_p]
libztensor.ztensor_metadata_get_shape_data.restype = ctypes.POINTER(ctypes.c_uint64) # Returns allocated array

# Tensor Data
libztensor.ztensor_reader_read_raw_tensor_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
libztensor.ztensor_reader_read_raw_tensor_data.restype = ctypes.POINTER(ctypes.c_uint8) # Returns allocated array

# Freeing
libztensor.ztensor_free_string.argtypes = [ctypes.c_char_p] # Takes char* directly
libztensor.ztensor_free_string.restype = None

libztensor.ztensor_free_u64_array.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
libztensor.ztensor_free_u64_array.restype = None

libztensor.ztensor_free_u8_array.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
libztensor.ztensor_free_u8_array.restype = None


# --- Pythonic Wrapper (Optional but Recommended) ---
class ZTensorError(Exception):
    pass

def _check_ptr(ptr, func_name=""):
    if not ptr:
        err_msg_ptr = libztensor.ztensor_last_error_message()
        err_msg = "Unknown FFI error"
        if err_msg_ptr:
            err_msg = ctypes.string_at(err_msg_ptr).decode('utf-8')
        raise ZTensorError(f"Error in {func_name}: {err_msg}")
    return ptr

class TensorMetadata:
    def __init__(self, meta_ptr):
        self._ptr = _check_ptr(meta_ptr, "TensorMetadata constructor")
        self._name = None
        self._dtype = None
        self._encoding = None
        # Add other fields as properties

    def __del__(self):
        if self._ptr:
            libztensor.ztensor_metadata_free(self._ptr)
            self._ptr = None
    
    @property
    def name(self):
        if self._name is None and self._ptr:
            name_ptr = libztensor.ztensor_metadata_get_name(self._ptr)
            if name_ptr:
                self._name = ctypes.string_at(name_ptr).decode('utf-8')
                libztensor.ztensor_free_string(name_ptr) # Free the Rust-allocated string
        return self._name

    @property
    def dtype_str(self):
        if self._dtype is None and self._ptr:
            dtype_ptr = libztensor.ztensor_metadata_get_dtype_str(self._ptr)
            if dtype_ptr:
                self._dtype = ctypes.string_at(dtype_ptr).decode('utf-8')
                libztensor.ztensor_free_string(dtype_ptr)
        return self._dtype
        
    @property
    def encoding_str(self):
        if self._encoding is None and self._ptr: # Simple caching
            ptr = libztensor.ztensor_metadata_get_encoding_str(self._ptr)
            if ptr:
                self._encoding = ctypes.string_at(ptr).decode('utf-8')
                libztensor.ztensor_free_string(ptr)
        return self._encoding

    @property
    def offset(self):
        return libztensor.ztensor_metadata_get_offset(self._ptr) if self._ptr else 0

    @property
    def size(self):
        return libztensor.ztensor_metadata_get_size(self._ptr) if self._ptr else 0

    @property
    def shape(self):
        if not self._ptr: return []
        shape_len = libztensor.ztensor_metadata_get_shape_len(self._ptr)
        if shape_len == 0: return []
        shape_data_ptr = libztensor.ztensor_metadata_get_shape_data(self._ptr)
        if not shape_data_ptr: return []
        
        # Create a Python list from the C array
        shape_list = [shape_data_ptr[i] for i in range(shape_len)]
        libztensor.ztensor_free_u64_array(shape_data_ptr, shape_len)
        return shape_list


class ZTensorReader:
    def __init__(self, file_path):
        path_bytes = file_path.encode('utf-8')
        self._ptr = _check_ptr(libztensor.ztensor_reader_open(path_bytes), "ZTensorReader open")

    def __del__(self):
        if self._ptr:
            libztensor.ztensor_reader_free(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__() # Ensure cleanup

    def get_metadata_count(self):
        return libztensor.ztensor_reader_get_metadata_count(self._ptr) if self._ptr else 0

    def get_metadata_by_index(self, index):
        if not self._ptr: raise ZTensorError("Reader is not open or already closed.")
        meta_ptr = libztensor.ztensor_reader_get_metadata_by_index(self._ptr, index)
        return TensorMetadata(meta_ptr) # Wrapper will free the meta_ptr

    def get_metadata_by_name(self, name):
        if not self._ptr: raise ZTensorError("Reader is not open or already closed.")
        name_bytes = name.encode('utf-8')
        meta_ptr = libztensor.ztensor_reader_get_metadata_by_name(self._ptr, name_bytes)
        return TensorMetadata(meta_ptr)

    def read_raw_tensor_data(self, metadata_obj: TensorMetadata):
        if not self._ptr or not metadata_obj._ptr:
            raise ZTensorError("Reader or metadata object is invalid.")
        
        data_len = ctypes.c_size_t()
        data_ptr = libztensor.ztensor_reader_read_raw_tensor_data(
            self._ptr, metadata_obj._ptr, ctypes.byref(data_len)
        )
        _check_ptr(data_ptr, "read_raw_tensor_data") # Checks for null pointer from FFI
        
        # Copy data into a Python bytes object
        # The C array returned by FFI is POINTER(c_uint8)
        py_data = ctypes.string_at(data_ptr, data_len.value) # This copies
        
        # Free the Rust-allocated buffer
        libztensor.ztensor_free_u8_array(data_ptr, data_len.value)
        return py_data


# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy ztensor file first (you'd use your Rust writer or a known file)
    # For this example, assume "test.zt" exists and is valid.
    # You would need to generate one with your Rust code first.
    # e.g. by running a small Rust main that uses ZTensorWriter.
    
    # Create a dummy file for testing if it doesn't exist
    dummy_file_path = "dummy.zt"
    if not os.path.exists(dummy_file_path):
        print(f"'{dummy_file_path}' not found. Please create a valid zTensor file for testing the bindings.")
        print("You can adapt the Rust tests to write a file, e.g., 'test_write_read_single_tensor_raw_adapted'")

    if os.path.exists(dummy_file_path):
        try:
            with ZTensorReader(dummy_file_path) as reader:
                print(f"Opened '{dummy_file_path}'")
                count = reader.get_metadata_count()
                print(f"Number of tensors: {count}")

                if count > 0:
                    for i in range(count):
                        print(f"\n--- Tensor {i} (by index) ---")
                        meta_idx = reader.get_metadata_by_index(i)
                        print(f"  Name: {meta_idx.name}")
                        print(f"  DType: {meta_idx.dtype_str}")
                        print(f"  Encoding: {meta_idx.encoding_str}")
                        print(f"  Shape: {meta_idx.shape}")
                        print(f"  Offset: {meta_idx.offset}")
                        print(f"  Size (on-disk): {meta_idx.size}")
                        
                        tensor_name = meta_idx.name # Get name to test get_by_name
                        del meta_idx # Explicitly test destructor

                        if tensor_name:
                            print(f"\n--- Tensor '{tensor_name}' (by name) ---")
                            meta_name = reader.get_metadata_by_name(tensor_name)
                            print(f"  Shape from by-name metadata: {meta_name.shape}")
                            
                            # Read tensor data
                            print(f"  Reading data for tensor: {meta_name.name}...")
                            raw_data = reader.read_raw_tensor_data(meta_name)
                            print(f"  Raw data size: {len(raw_data)} bytes")
                            print(f"  First 10 bytes: {raw_data[:10]}")


        except ZTensorError as e:
            print(f"An error occurred: {e}")
        except OSError as e:
            print(f"OS Error (likely library load issue): {e}")
    else:
        print(f"Skipping example usage as '{dummy_file_path}' does not exist.")