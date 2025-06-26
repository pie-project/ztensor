use lazy_static::lazy_static;
use libc::{c_char, c_int, c_uchar, c_void, size_t};
use std::ffi::{CStr, CString};
use std::io::{BufReader, BufWriter, Cursor};
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use crate::ZTensorError;
use crate::models::{ChecksumAlgorithm, DType, DataEndianness, Encoding, Layout, TensorMetadata};
use crate::reader::ZTensorReader;
use crate::writer::ZTensorWriter;

// --- Error Handling ---
lazy_static! {
    static ref LAST_ERROR: Mutex<Option<CString>> = Mutex::new(None);
}

fn update_last_error(err: ZTensorError) {
    let msg = CString::new(err.to_string())
        .unwrap_or_else(|_| CString::new("FFI: Unknown error").unwrap());
    *LAST_ERROR.lock().unwrap() = Some(msg);
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_last_error_message() -> *const c_char {
    match LAST_ERROR.lock().unwrap().as_ref() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    }
}

// --- Opaque Structs and Handles ---

// Reader is now generic over R: Read + Seek.
// For the C API, we'll expose a version that works with files.
pub type CZTensorReader = ZTensorReader<BufReader<std::fs::File>>;

// Writer is also generic. We'll expose a file-based and in-memory version.
pub type CZTensorWriter = ZTensorWriter<BufWriter<std::fs::File>>;
pub type CInMemoryZTensorWriter = ZTensorWriter<Cursor<Vec<u8>>>;

// Represents an owned, C-compatible TensorMetadata.
pub type CTensorMetadata = TensorMetadata;

// Represents a non-owning, zero-copy view of tensor data.
#[repr(C)]
pub struct CTensorDataView {
    pub data: *const c_uchar,
    pub len: size_t,
    // Private field to hold the owned data, making this struct self-contained.
    // The C side only sees a pointer to the data, but this struct owns the Vec.
    _owner: *mut c_void,
}

// --- Reader Functions ---

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_open(path_str: *const c_char) -> *mut CZTensorReader {
    if path_str.is_null() {
        update_last_error(ZTensorError::Other("Null path provided".into()));
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(path_str) };
    let path_res = path.to_str().map(Path::new);

    let path = match path_res {
        Ok(p) => p,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 path".into()));
            return ptr::null_mut();
        }
    };

    match ZTensorReader::open(path) {
        Ok(reader) => Box::into_raw(Box::new(reader)),
        Err(e) => {
            update_last_error(e);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_free(reader_ptr: *mut CZTensorReader) {
    if !reader_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(reader_ptr);
        };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_count(reader_ptr: *const CZTensorReader) -> size_t {
    let reader = unsafe {
        reader_ptr
            .as_ref()
            .expect("Null pointer passed to ztensor_reader_get_metadata_count")
    };
    reader.list_tensors().len()
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_by_name(
    reader_ptr: *const CZTensorReader,
    name_str: *const c_char,
) -> *mut CTensorMetadata {
    let reader = unsafe {
        reader_ptr
            .as_ref()
            .expect("Null pointer passed to ztensor_reader_get_metadata_by_name")
    };
    if name_str.is_null() {
        update_last_error(ZTensorError::Other("Null name pointer provided".into()));
        return ptr::null_mut();
    }
    let name = unsafe { CStr::from_ptr(name_str).to_str().unwrap() };

    match reader.get_tensor_metadata(name) {
        Some(metadata) => Box::into_raw(Box::new(metadata.clone())),
        None => {
            update_last_error(ZTensorError::TensorNotFound(name.to_string()));
            ptr::null_mut()
        }
    }
}

// --- Zero-Copy Tensor Data Reading ---

/// Reads tensor data into a view without copying. The view must be freed.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_read_tensor_view(
    reader_ptr: *mut CZTensorReader,
    metadata_ptr: *const CTensorMetadata,
) -> *mut CTensorDataView {
    let reader = unsafe {
        reader_ptr
            .as_mut()
            .expect("Null reader pointer to read_tensor_view")
    };
    let metadata = unsafe {
        metadata_ptr
            .as_ref()
            .expect("Null metadata pointer to read_tensor_view")
    };

    match reader.read_raw_tensor_data(metadata) {
        Ok(data_vec) => {
            let view = Box::new(CTensorDataView {
                data: data_vec.as_ptr(),
                len: data_vec.len(),
                _owner: Box::into_raw(Box::new(data_vec)) as *mut c_void,
            });
            Box::into_raw(view)
        }
        Err(e) => {
            update_last_error(e);
            ptr::null_mut()
        }
    }
}

/// Frees the tensor data view.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_tensor_view(view_ptr: *mut CTensorDataView) {
    if !view_ptr.is_null() {
        unsafe {
            let view = Box::from_raw(view_ptr);
            // This will drop the Vec<u8> that `_owner` points to
            let _ = Box::from_raw(view._owner as *mut Vec<u8>);
        }
    }
}

// --- Metadata Accessors ---
// (These are largely the same but ensured to be safe)

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_free(metadata_ptr: *mut CTensorMetadata) {
    if !metadata_ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(metadata_ptr);
        };
    }
}

fn to_cstring(s: String) -> *mut c_char {
    CString::new(s).map_or(ptr::null_mut(), |cs| cs.into_raw())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_name(metadata_ptr: *const CTensorMetadata) -> *mut c_char {
    let metadata = unsafe { metadata_ptr.as_ref().expect("Null metadata pointer") };
    to_cstring(metadata.name.clone())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_dtype_str(
    metadata_ptr: *const CTensorMetadata,
) -> *mut c_char {
    let metadata = unsafe { metadata_ptr.as_ref().expect("Null metadata pointer") };
    to_cstring(metadata.dtype.to_string_key())
}

// ... other metadata accessors like get_offset, get_size, etc.

/// Returns the number of dimensions in the tensor's shape.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_shape_len(metadata_ptr: *const CTensorMetadata) -> size_t {
    let metadata = unsafe { metadata_ptr.as_ref().expect("Null metadata pointer") };
    metadata.shape.len()
}

/// Returns a pointer to the shape data (an array of u64).
/// The caller owns this memory and must free it with `ztensor_free_u64_array`.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_shape_data(
    metadata_ptr: *const CTensorMetadata,
) -> *mut u64 {
    let metadata = unsafe { metadata_ptr.as_ref().expect("Null metadata pointer") };
    // Clone the shape into a new Vec, get its raw pointer, and forget it
    // so Rust doesn't deallocate it. The C side is now responsible.
    let mut shape_vec = metadata.shape.clone();
    let ptr = shape_vec.as_mut_ptr();
    std::mem::forget(shape_vec);
    ptr
}

/// Frees the shape array allocated by `ztensor_metadata_get_shape_data`.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_u64_array(ptr: *mut u64, len: size_t) {
    if !ptr.is_null() {
        unsafe {
            // Reconstitute the Vec from the raw parts and let it drop, freeing the memory.
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

// --- Writer Functions ---

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_writer_create(path_str: *const c_char) -> *mut CZTensorWriter {
    if path_str.is_null() {
        update_last_error(ZTensorError::Other("Null path provided".into()));
        return ptr::null_mut();
    }
    let path = unsafe { CStr::from_ptr(path_str) };
    let path = match path.to_str().map(Path::new) {
        Ok(p) => p,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 path".into()));
            return ptr::null_mut();
        }
    };

    match ZTensorWriter::create(path) {
        Ok(writer) => Box::into_raw(Box::new(writer)),
        Err(e) => {
            update_last_error(e);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_writer_free(writer_ptr: *mut CZTensorWriter) {
    if !writer_ptr.is_null() {
        let _ = unsafe { Box::from_raw(writer_ptr) };
        // Dropping the writer will close the file.
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_writer_add_tensor(
    writer_ptr: *mut CZTensorWriter,
    name_str: *const c_char,
    shape_ptr: *const u64,
    shape_len: size_t,
    dtype_str: *const c_char,
    data_ptr: *const c_uchar,
    data_len: size_t,
) -> c_int {
    let writer = unsafe { writer_ptr.as_mut().expect("Null writer pointer") };
    let name = unsafe { CStr::from_ptr(name_str).to_str().unwrap() };
    let shape = unsafe { slice::from_raw_parts(shape_ptr, shape_len) };
    let dtype_str = unsafe { CStr::from_ptr(dtype_str).to_str().unwrap() };
    let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };

    let dtype = match dtype_str {
        "float32" => DType::Float32,
        "uint8" => DType::Uint8,
        // ... add other dtypes
        _ => {
            update_last_error(ZTensorError::UnsupportedDType(dtype_str.to_string()));
            return -1;
        }
    };

    let res = writer.add_tensor(
        name,
        shape.to_vec(),
        dtype,
        Layout::Dense,
        Encoding::Raw,
        data.to_vec(),
        Some(DataEndianness::Little), // Defaulting to little, could be a parameter
        ChecksumAlgorithm::None,
        None,
    );

    match res {
        Ok(_) => 0,
        Err(e) => {
            update_last_error(e);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_writer_finalize(writer_ptr: *mut CZTensorWriter) -> c_int {
    if writer_ptr.is_null() {
        return -1;
    }
    // `from_raw` takes ownership and will drop the writer when it goes out of scope.
    // The writer's `drop` implementation should handle finalization.
    let writer = unsafe { Box::from_raw(writer_ptr) };
    match writer.finalize() {
        Ok(_) => 0,
        Err(e) => {
            update_last_error(e);
            -1
        }
    }
}

// --- Memory Freeing for C ---

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        };
    }
}
