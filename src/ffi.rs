use lazy_static::lazy_static;
use libc::{c_char, c_int, c_uchar, c_void, size_t};
use std::ffi::{CStr, CString};
use std::io::{BufWriter, Cursor};
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use memmap2::Mmap;

use crate::ZTensorError;
use crate::models::{ChecksumAlgorithm, DType, Encoding, Tensor};
use crate::reader::ZTensorReader;
use crate::writer::ZTensorWriter;

// --- C-Compatible Structs & Handles ---

pub type CZTensorReader = ZTensorReader<Cursor<Mmap>>;
pub type CZTensorWriter = ZTensorWriter<BufWriter<std::fs::File>>;
pub type CInMemoryZTensorWriter = ZTensorWriter<Cursor<Vec<u8>>>;
pub type CTensorMetadata = (String, Tensor);

#[repr(C)]
pub struct CTensorDataView {
    pub data: *const c_uchar,
    pub len: size_t,
    _owner: *mut c_void,
}

#[repr(C)]
pub struct CStringArray {
    pub strings: *mut *mut c_char,
    pub len: size_t,
}

/// A C-compatible batch result containing multiple tensor data views.
#[repr(C)]
pub struct CTensorDataViewArray {
    pub views: *mut CTensorDataView,
    pub len: size_t,
}

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

// --- Internal Helpers ---

macro_rules! ztensor_handle {
    ($ptr:expr) => {
        if $ptr.is_null() {
            update_last_error(ZTensorError::Other("Null pointer passed as handle".into()));
            return ptr::null_mut();
        } else {
            unsafe { &*$ptr }
        }
    };
    (mut $ptr:expr) => {
        if $ptr.is_null() {
            update_last_error(ZTensorError::Other("Null pointer passed as handle".into()));
            return ptr::null_mut();
        } else {
            unsafe { &mut *$ptr }
        }
    };
    ($ptr:expr, $err_ret:expr) => {
        if $ptr.is_null() {
            update_last_error(ZTensorError::Other("Null pointer passed as handle".into()));
            return $err_ret;
        } else {
            unsafe { &*$ptr }
        }
    };
    (mut $ptr:expr, $err_ret:expr) => {
        if $ptr.is_null() {
            update_last_error(ZTensorError::Other("Null pointer passed as handle".into()));
            return $err_ret;
        } else {
            unsafe { &mut *$ptr }
        }
    };
}

fn to_cstring(s: String) -> *mut c_char {
    CString::new(s).map_or(ptr::null_mut(), |cs| cs.into_raw())
}

// --- Reader API ---

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_open(path_str: *const c_char) -> *mut CZTensorReader {
    if path_str.is_null() {
        update_last_error(ZTensorError::Other("Null path provided".into()));
        return ptr::null_mut();
    }
    let path = match unsafe { CStr::from_ptr(path_str).to_str() } {
        Ok(s) => Path::new(s),
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 path".into()));
            return ptr::null_mut();
        }
    };

    match ZTensorReader::open_mmap(path) {
        Ok(reader) => Box::into_raw(Box::new(reader)),
        Err(e) => {
            update_last_error(e);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_count(reader_ptr: *const CZTensorReader) -> size_t {
    let reader = ztensor_handle!(reader_ptr, 0);
    reader.list_tensors().len()
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_by_name(
    reader_ptr: *const CZTensorReader,
    name_str: *const c_char,
) -> *mut CTensorMetadata {
    let reader = ztensor_handle!(reader_ptr);
    if name_str.is_null() {
        update_last_error(ZTensorError::Other("Null name pointer provided".into()));
        return ptr::null_mut();
    }
    let name = match unsafe { CStr::from_ptr(name_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 name".into()));
            return ptr::null_mut();
        }
    };

    match reader.get_tensor(name) {
        Some(tensor) => Box::into_raw(Box::new((name.to_string(), tensor.clone()))),
        None => {
            update_last_error(ZTensorError::TensorNotFound(name.to_string()));
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_by_index(
    reader_ptr: *const CZTensorReader,
    index: size_t,
) -> *mut CTensorMetadata {
    let reader = ztensor_handle!(reader_ptr);
    let tensors = reader.list_tensors();
    match tensors.iter().nth(index) {
        Some((name, tensor)) => Box::into_raw(Box::new((name.clone(), tensor.clone()))),
        None => {
            update_last_error(ZTensorError::Other(format!(
                "Index {} is out of bounds for tensor list of length {}",
                index,
                tensors.len()
            )));
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_all_tensor_names(
    reader_ptr: *const CZTensorReader,
) -> *mut CStringArray {
    let reader = ztensor_handle!(reader_ptr);
    let names: Vec<CString> = reader
        .list_tensors()
        .keys()
        .map(|name| CString::new(name.as_str()).unwrap())
        .collect();

    let mut c_names: Vec<*mut c_char> = names.into_iter().map(|s| s.into_raw()).collect();
    let string_array = Box::new(CStringArray {
        strings: c_names.as_mut_ptr(),
        len: c_names.len(),
    });

    std::mem::forget(c_names);
    Box::into_raw(string_array)
}

/// Reads a single tensor by name.
/// 
/// # Arguments
/// * `reader_ptr` - Reader handle
/// * `name_str` - Tensor name
/// * `verify_checksum` - Whether to verify checksum (0 = false, non-zero = true)
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_read_tensor(
    reader_ptr: *mut CZTensorReader,
    name_str: *const c_char,
    verify_checksum: c_int,
) -> *mut CTensorDataView {
    let reader = ztensor_handle!(mut reader_ptr);
    
    let name = match unsafe { CStr::from_ptr(name_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 name".into()));
            return ptr::null_mut();
        }
    };

    match reader.read_tensor(name, verify_checksum != 0) {
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

/// Reads multiple tensors in batch.
/// 
/// # Arguments
/// * `reader_ptr` - Reader handle
/// * `names` - Array of tensor name C strings
/// * `names_len` - Number of names
/// * `verify_checksum` - Whether to verify checksum (0 = false, non-zero = true)
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_read_tensors(
    reader_ptr: *mut CZTensorReader,
    names: *const *const c_char,
    names_len: size_t,
    verify_checksum: c_int,
) -> *mut CTensorDataViewArray {
    let reader = ztensor_handle!(mut reader_ptr);
    
    // Parse names
    let name_strs: Vec<&str> = match (0..names_len)
        .map(|i| {
            let name_ptr = unsafe { *names.add(i) };
            if name_ptr.is_null() {
                return Err(ZTensorError::Other("Null name in batch read".into()));
            }
            unsafe { CStr::from_ptr(name_ptr).to_str() }
                .map_err(|_| ZTensorError::Other("Invalid UTF-8 name in batch".into()))
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(names) => names,
        Err(e) => {
            update_last_error(e);
            return ptr::null_mut();
        }
    };

    match reader.read_tensors(&name_strs, verify_checksum != 0) {
        Ok(data_vecs) => {
            let mut views: Vec<CTensorDataView> = data_vecs
                .into_iter()
                .map(|data_vec| {
                    CTensorDataView {
                        data: data_vec.as_ptr(),
                        len: data_vec.len(),
                        _owner: Box::into_raw(Box::new(data_vec)) as *mut c_void,
                    }
                })
                .collect();
            
            let result = Box::new(CTensorDataViewArray {
                views: views.as_mut_ptr(),
                len: views.len(),
            });
            std::mem::forget(views);
            Box::into_raw(result)
        }
        Err(e) => {
            update_last_error(e);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_read_tensor_component(
    reader_ptr: *mut CZTensorReader,
    name_str: *const c_char,
    component_name_str: *const c_char,
) -> *mut CTensorDataView {
    let reader = ztensor_handle!(mut reader_ptr);
    
    let name = match unsafe { CStr::from_ptr(name_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 name".into()));
            return ptr::null_mut();
        }
    };
    
    let component_name = match unsafe { CStr::from_ptr(component_name_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 component name".into()));
            return ptr::null_mut();
        }
    };

    let tensor_opt = reader.get_tensor(name);
    let tensor = match tensor_opt {
        Some(t) => t.clone(),
        None => {
            update_last_error(ZTensorError::TensorNotFound(name.to_string()));
            return ptr::null_mut();
        }
    };
    
    let component = match tensor.components.get(component_name) {
        Some(c) => c.clone(),
        None => {
            update_last_error(ZTensorError::Other(format!("Component '{}' not found in tensor '{}'", component_name, name)));
            return ptr::null_mut();
        }
    };

    match reader.read_component(&component) {
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

// --- Writer API ---

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_writer_create(path_str: *const c_char) -> *mut CZTensorWriter {
    if path_str.is_null() {
        update_last_error(ZTensorError::Other("Null path provided".into()));
        return ptr::null_mut();
    }
    let path = match unsafe { CStr::from_ptr(path_str).to_str() } {
        Ok(s) => Path::new(s),
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
pub extern "C" fn ztensor_writer_add_tensor(
    writer_ptr: *mut CZTensorWriter,
    name_str: *const c_char,
    shape_ptr: *const u64,
    shape_len: size_t,
    dtype_str: *const c_char,
    data_ptr: *const c_uchar,
    data_len: size_t,
    compress: c_int,
) -> c_int {
    let writer = ztensor_handle!(mut writer_ptr, -1);
    let name = match unsafe { CStr::from_ptr(name_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 name".into()));
            return -1;
        }
    };
    let shape = unsafe { slice::from_raw_parts(shape_ptr, shape_len) };
    let dtype_str = match unsafe { CStr::from_ptr(dtype_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 dtype string".into()));
            return -1;
        }
    };
    let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };

    let dtype = match dtype_str {
        "float64" => DType::Float64,
        "float32" => DType::Float32,
        "float16" => DType::Float16,
        "bfloat16" => DType::BFloat16,
        "int64" => DType::Int64,
        "int32" => DType::Int32,
        "int16" => DType::Int16,
        "int8" => DType::Int8,
        "uint64" => DType::Uint64,
        "uint32" => DType::Uint32,
        "uint16" => DType::Uint16,
        "uint8" => DType::Uint8,
        "bool" => DType::Bool,
        _ => {
            update_last_error(ZTensorError::UnsupportedDType(dtype_str.to_string()));
            return -1;
        }
    };

    let encoding = if compress != 0 { Encoding::Zstd } else { Encoding::Raw };

    let res = writer.add_tensor(
        name,
        shape.to_vec(),
        dtype,
        encoding,
        data.to_vec(),
        ChecksumAlgorithm::None,
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
pub extern "C" fn ztensor_writer_add_sparse_csr(
    writer_ptr: *mut CZTensorWriter,
    name_str: *const c_char,
    shape_ptr: *const u64,
    shape_len: size_t,
    dtype_str: *const c_char,
    values_ptr: *const c_uchar,
    values_len: size_t,
    indices_ptr: *const u64,
    indices_len: size_t,
    indptr_ptr: *const u64,
    indptr_len: size_t,
) -> c_int {
    let writer = ztensor_handle!(mut writer_ptr, -1);
    let name = match unsafe { CStr::from_ptr(name_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 name".into()));
            return -1;
        }
    };
    let shape = unsafe { slice::from_raw_parts(shape_ptr, shape_len) };
    let dtype_str = match unsafe { CStr::from_ptr(dtype_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 dtype string".into()));
            return -1;
        }
    };
    let values_bytes = unsafe { slice::from_raw_parts(values_ptr, values_len) }.to_vec();
    let indices = unsafe { slice::from_raw_parts(indices_ptr, indices_len) }.to_vec();
    let indptr = unsafe { slice::from_raw_parts(indptr_ptr, indptr_len) }.to_vec();

    let dtype = match dtype_str {
        "float64" => DType::Float64,
        "float32" => DType::Float32,
        "float16" => DType::Float16,
        "bfloat16" => DType::BFloat16,
        "int64" => DType::Int64,
        "int32" => DType::Int32,
        "int16" => DType::Int16,
        "int8" => DType::Int8,
        "uint64" => DType::Uint64,
        "uint32" => DType::Uint32,
        "uint16" => DType::Uint16,
        "uint8" => DType::Uint8,
        "bool" => DType::Bool,
        _ => {
            update_last_error(ZTensorError::UnsupportedDType(dtype_str.to_string()));
            return -1;
        }
    };

    let res = writer.add_csr_tensor(
        name,
        shape.to_vec(),
        dtype,
        values_bytes,
        indices,
        indptr,
        Encoding::Raw,
        ChecksumAlgorithm::None,
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
pub extern "C" fn ztensor_writer_add_sparse_coo(
    writer_ptr: *mut CZTensorWriter,
    name_str: *const c_char,
    shape_ptr: *const u64,
    shape_len: size_t,
    dtype_str: *const c_char,
    values_ptr: *const c_uchar,
    values_len: size_t,
    indices_ptr: *const u64,
    indices_len: size_t,
) -> c_int {
    let writer = ztensor_handle!(mut writer_ptr, -1);
    let name = match unsafe { CStr::from_ptr(name_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 name".into()));
            return -1;
        }
    };
    let shape = unsafe { slice::from_raw_parts(shape_ptr, shape_len) };
    let dtype_str = match unsafe { CStr::from_ptr(dtype_str).to_str() } {
        Ok(s) => s,
        Err(_) => {
            update_last_error(ZTensorError::Other("Invalid UTF-8 dtype string".into()));
            return -1;
        }
    };
    let values_bytes = unsafe { slice::from_raw_parts(values_ptr, values_len) }.to_vec();
    let indices = unsafe { slice::from_raw_parts(indices_ptr, indices_len) }.to_vec();

    let dtype = match dtype_str {
        "float64" => DType::Float64,
        "float32" => DType::Float32,
        "float16" => DType::Float16,
        "bfloat16" => DType::BFloat16,
        "int64" => DType::Int64,
        "int32" => DType::Int32,
        "int16" => DType::Int16,
        "int8" => DType::Int8,
        "uint64" => DType::Uint64,
        "uint32" => DType::Uint32,
        "uint16" => DType::Uint16,
        "uint8" => DType::Uint8,
        "bool" => DType::Bool,
        _ => {
            update_last_error(ZTensorError::UnsupportedDType(dtype_str.to_string()));
            return -1;
        }
    };

    let res = writer.add_coo_tensor(
        name,
        shape.to_vec(),
        dtype,
        values_bytes,
        indices,
        Encoding::Raw,
        ChecksumAlgorithm::None,
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
    let writer = unsafe { Box::from_raw(writer_ptr) };
    match writer.finalize() {
        Ok(_) => 0,
        Err(e) => {
            update_last_error(e);
            -1
        }
    }
}

// --- Metadata API ---

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_name(metadata_ptr: *const CTensorMetadata) -> *mut c_char {
    let (name, _) = ztensor_handle!(metadata_ptr);
    to_cstring(name.clone())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_dtype_str(
    metadata_ptr: *const CTensorMetadata,
) -> *mut c_char {
    let (_, metadata) = ztensor_handle!(metadata_ptr);
    to_cstring(metadata.dtype.to_string_key())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_offset(metadata_ptr: *const CTensorMetadata) -> u64 {
    let (_, metadata) = ztensor_handle!(metadata_ptr, 0);
    metadata.components.get("data").map_or(0, |c| c.offset)
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_size(metadata_ptr: *const CTensorMetadata) -> u64 {
    let (_, metadata) = ztensor_handle!(metadata_ptr, 0);
    metadata.components.get("data").map_or(0, |c| c.length)
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_layout_str(
    metadata_ptr: *const CTensorMetadata,
) -> *mut c_char {
    let (_, metadata) = ztensor_handle!(metadata_ptr);
    to_cstring(metadata.format.clone())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_encoding_str(
    metadata_ptr: *const CTensorMetadata,
) -> *mut c_char {
    let (_, metadata) = ztensor_handle!(metadata_ptr);
    metadata.components.get("data")
        .map_or(ptr::null_mut(), |c| to_cstring(format!("{:?}", c.encoding).to_lowercase()))
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_data_endianness_str(
    _metadata_ptr: *const CTensorMetadata,
) -> *mut c_char {
    to_cstring("little".to_string())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_checksum_str(
    metadata_ptr: *const CTensorMetadata,
) -> *mut c_char {
    let (_, metadata) = ztensor_handle!(metadata_ptr);
    metadata.components.get("data")
        .and_then(|c| c.digest.as_ref())
        .map_or(ptr::null_mut(), |s| to_cstring(s.clone()))
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_shape_len(metadata_ptr: *const CTensorMetadata) -> size_t {
    let (_, metadata) = ztensor_handle!(metadata_ptr, 0);
    metadata.shape.len()
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_shape_data(
    metadata_ptr: *const CTensorMetadata,
) -> *mut u64 {
    let (_, metadata) = ztensor_handle!(metadata_ptr);
    let mut shape_vec = metadata.shape.clone();
    let ptr = shape_vec.as_mut_ptr();
    std::mem::forget(shape_vec);
    ptr
}

// --- Memory Management API ---

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_free(reader_ptr: *mut CZTensorReader) {
    if !reader_ptr.is_null() {
        let _ = unsafe { Box::from_raw(reader_ptr) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_writer_free(writer_ptr: *mut CZTensorWriter) {
    if !writer_ptr.is_null() {
        let _ = unsafe { Box::from_raw(writer_ptr) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_free(metadata_ptr: *mut CTensorMetadata) {
    if !metadata_ptr.is_null() {
        let _ = unsafe { Box::from_raw(metadata_ptr) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_tensor_view(view_ptr: *mut CTensorDataView) {
    if !view_ptr.is_null() {
        unsafe {
            let view = Box::from_raw(view_ptr);
            let _ = Box::from_raw(view._owner as *mut Vec<u8>);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_tensor_view_array(arr_ptr: *mut CTensorDataViewArray) {
    if arr_ptr.is_null() {
        return;
    }
    unsafe {
        let arr = Box::from_raw(arr_ptr);
        let views = Vec::from_raw_parts(arr.views, arr.len, arr.len);
        for view in views {
            let _ = Box::from_raw(view._owner as *mut Vec<u8>);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = unsafe { CString::from_raw(s) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_string_array(arr_ptr: *mut CStringArray) {
    if arr_ptr.is_null() {
        return;
    }
    unsafe {
        let arr = Box::from_raw(arr_ptr);
        let strings = Vec::from_raw_parts(arr.strings, arr.len, arr.len);
        for s_ptr in strings {
            let _ = CString::from_raw(s_ptr);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_u64_array(ptr: *mut u64, len: size_t) {
    if !ptr.is_null() {
        let _ = unsafe { Vec::from_raw_parts(ptr, len, len) };
    }
}
