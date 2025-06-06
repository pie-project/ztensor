use lazy_static::lazy_static;
use libc::{c_char, c_int, size_t};
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use crate::ZTensorError;
use crate::models::{DType, DataEndianness, Encoding, TensorMetadata}; // Assuming DType, Encoding, DataEndianness have to_string_key or similar
use crate::reader::ZTensorReader;

// --- Error Handling ---
lazy_static! {
    static ref LAST_ERROR_MESSAGE: Mutex<Option<CString>> = Mutex::new(None);
}

fn update_last_error(err: ZTensorError) {
    let msg = CString::new(err.to_string())
        .unwrap_or_else(|_| CString::new("Unknown error converting error message").unwrap());
    *LAST_ERROR_MESSAGE.lock().unwrap() = Some(msg);
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_last_error_message() -> *const c_char {
    match LAST_ERROR_MESSAGE.lock().unwrap().as_ref() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    }
}

// Helper to convert Rust Result into C-style error code
// 0 for success, -1 for error (and sets LAST_ERROR_MESSAGE)
fn result_to_status_code<T>(res: Result<T, ZTensorError>, default_val: T) -> (T, c_int) {
    match res {
        Ok(val) => (val, 0),
        Err(e) => {
            update_last_error(e);
            (default_val, -1)
        }
    }
}
fn status_code_from_result(res: Result<(), ZTensorError>) -> c_int {
    match res {
        Ok(()) => 0,
        Err(e) => {
            update_last_error(e);
            -1
        }
    }
}

// --- Opaque Structs and Handles ---
pub type CZTensorReader = ZTensorReader<std::io::BufReader<std::fs::File>>;
// CTensorMetadata will be an opaque pointer to a Rust-allocated TensorMetadata
pub type CTensorMetadata = TensorMetadata;

// --- Reader Functions ---
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_open(path_str: *const c_char) -> *mut CZTensorReader {
    if path_str.is_null() {
        update_last_error(ZTensorError::Other("Path string is null".to_string()));
        return ptr::null_mut();
    }
    let path_cstr = unsafe { CStr::from_ptr(path_str) };
    let path = match path_cstr.to_str() {
        Ok(s) => Path::new(s),
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 path string".to_string()));
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
            Box::from_raw(reader_ptr);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_count(reader_ptr: *const CZTensorReader) -> size_t {
    if reader_ptr.is_null() {
        update_last_error(ZTensorError::Other("Reader pointer is null".to_string()));
        return 0;
    }
    let reader = unsafe { &*reader_ptr };
    reader.list_tensors().len()
}

// Note: This returns a Boxed TensorMetadata. The C side will need to free it using ztensor_metadata_free.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_by_index(
    reader_ptr: *const CZTensorReader,
    index: size_t,
) -> *mut CTensorMetadata {
    if reader_ptr.is_null() {
        update_last_error(ZTensorError::Other("Reader pointer is null".to_string()));
        return ptr::null_mut();
    }
    let reader = unsafe { &*reader_ptr };
    match reader.list_tensors().get(index) {
        Some(metadata) => Box::into_raw(Box::new(metadata.clone())), // Clone and box
        None => {
            update_last_error(ZTensorError::Other(format!(
                "Metadata index {} out of bounds",
                index
            )));
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_by_name(
    reader_ptr: *const CZTensorReader,
    name_str: *const c_char,
) -> *mut CTensorMetadata {
    if reader_ptr.is_null() || name_str.is_null() {
        update_last_error(ZTensorError::Other(
            "Reader or name pointer is null".to_string(),
        ));
        return ptr::null_mut();
    }
    let reader = unsafe { &*reader_ptr };
    let name_cstr = unsafe { CStr::from_ptr(name_str) };
    let name = match name_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 name string".to_string()));
            return ptr::null_mut();
        }
    };

    match reader.get_tensor_metadata(name) {
        Some(metadata) => Box::into_raw(Box::new(metadata.clone())), // Clone and box
        None => {
            update_last_error(ZTensorError::TensorNotFound(name.to_string()));
            ptr::null_mut()
        }
    }
}

// --- Metadata Accessor Functions ---
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_free(metadata_ptr: *mut CTensorMetadata) {
    if !metadata_ptr.is_null() {
        unsafe {
            Box::from_raw(metadata_ptr);
        }
    }
}

// Helper for returning owned CStrings
fn string_to_c_char_ptr(s: String) -> *mut c_char {
    CString::new(s).map_or_else(|_| ptr::null_mut(), |cs| cs.into_raw())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_name(metadata_ptr: *const CTensorMetadata) -> *mut c_char {
    if metadata_ptr.is_null() {
        return ptr::null_mut();
    }
    let metadata = unsafe { &*metadata_ptr };
    string_to_c_char_ptr(metadata.name.clone())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_dtype_str(
    metadata_ptr: *const CTensorMetadata,
) -> *mut c_char {
    if metadata_ptr.is_null() {
        return ptr::null_mut();
    }
    let metadata = unsafe { &*metadata_ptr };
    string_to_c_char_ptr(metadata.dtype.to_string_key()) // Assumes DType has to_string_key()
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_encoding_str(
    metadata_ptr: *const CTensorMetadata,
) -> *mut c_char {
    if metadata_ptr.is_null() {
        return ptr::null_mut();
    }
    let metadata = unsafe { &*metadata_ptr };
    let encoding_str = match &metadata.encoding {
        // Encoding enum needs a similar to_string_key()
        Encoding::Raw => "raw".to_string(),
        Encoding::Zstd => "zstd".to_string(),
    };
    string_to_c_char_ptr(encoding_str)
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_offset(metadata_ptr: *const CTensorMetadata) -> u64 {
    if metadata_ptr.is_null() {
        return 0;
    }
    let metadata = unsafe { &*metadata_ptr };
    metadata.offset
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_size(metadata_ptr: *const CTensorMetadata) -> u64 {
    if metadata_ptr.is_null() {
        return 0;
    }
    let metadata = unsafe { &*metadata_ptr };
    metadata.size
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_shape_len(metadata_ptr: *const CTensorMetadata) -> size_t {
    if metadata_ptr.is_null() {
        return 0;
    }
    let metadata = unsafe { &*metadata_ptr };
    metadata.shape.len()
}

// Returns a copy of the shape data, must be freed by ztensor_free_u64_array
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_shape_data(
    metadata_ptr: *const CTensorMetadata,
) -> *mut u64 {
    if metadata_ptr.is_null() {
        return ptr::null_mut();
    }
    let metadata = unsafe { &*metadata_ptr };
    let mut shape_copy = metadata.shape.to_vec();
    let ptr = shape_copy.as_mut_ptr();
    std::mem::forget(shape_copy); // Prevent Rust from dropping the data
    ptr
}

// --- Tensor Data Reading ---
// Returns a copy of the tensor data, must be freed by ztensor_free_u8_array
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_read_raw_tensor_data(
    reader_ptr: *mut CZTensorReader,
    metadata_ptr: *const CTensorMetadata,
    out_data_len: *mut size_t,
) -> *mut u8 {
    if reader_ptr.is_null() || metadata_ptr.is_null() || out_data_len.is_null() {
        update_last_error(ZTensorError::Other(
            "Null pointer argument to read_raw_tensor_data".to_string(),
        ));
        if !out_data_len.is_null() {
            unsafe {
                *out_data_len = 0;
            }
        }
        return ptr::null_mut();
    }
    let reader = unsafe { &mut *reader_ptr };
    let metadata = unsafe { &*metadata_ptr };

    match reader.read_raw_tensor_data(metadata) {
        Ok(mut data_vec) => {
            unsafe {
                *out_data_len = data_vec.len();
            }
            let ptr = data_vec.as_mut_ptr();
            std::mem::forget(data_vec); // Transfer ownership to C
            ptr
        }
        Err(e) => {
            update_last_error(e);
            unsafe {
                *out_data_len = 0;
            }
            ptr::null_mut()
        }
    }
}

// --- Memory Freeing Functions for C side ---
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_string(s_ptr: *mut c_char) {
    if !s_ptr.is_null() {
        unsafe {
            CString::from_raw(s_ptr);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_u64_array(arr_ptr: *mut u64, len: size_t) {
    if !arr_ptr.is_null() {
        unsafe {
            Vec::from_raw_parts(arr_ptr, len, len);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_u8_array(arr_ptr: *mut u8, len: size_t) {
    if !arr_ptr.is_null() {
        unsafe {
            Vec::from_raw_parts(arr_ptr, len, len);
        }
    }
}
