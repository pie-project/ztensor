//! FFI bindings for Python/C interop.
//!
//! This module exposes a C-compatible API for the zTensor library.
//! Memory management follows these rules:
//! - Caller owns: Reader/Writer handles (must call free)
//! - Views: Data is owned by Rust Vec, freed via ztensor_free_tensor_view

use libc::{c_char, c_int, c_uchar, c_void, size_t};
use std::ffi::{CStr, CString};
use std::io::{BufWriter, Cursor};
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use memmap2::Mmap;

use crate::error::ZTensorError;
use crate::models::{ChecksumAlgorithm, DType, Encoding, Object};
use crate::reader::ZTensorReader;
use crate::writer::ZTensorWriter;

// --- Type Aliases ---

use crate::compat::{LegacyReader, is_legacy_file};

// --- Type Aliases ---

pub enum CZTensorReader {
    V1(ZTensorReader<Cursor<Mmap>>),
    Legacy(LegacyReader<Cursor<Mmap>>),
}
pub type CZTensorWriter = ZTensorWriter<BufWriter<std::fs::File>>;
pub type CObjectMetadata = (String, Object);

// --- C-Compatible Structs ---

/// View into tensor data without copying.
/// The data is owned by the _owner Vec and must be freed via ztensor_free_tensor_view.
/// If _owner is null, the data is a view into the memory-mapped file and the reader must outlive this view.
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

#[repr(C)]
pub struct CTensorDataViewArray {
    pub views: *mut CTensorDataView,
    pub len: size_t,
}

// --- Error Handling ---

static LAST_ERROR: Mutex<Option<CString>> = Mutex::new(None);

fn update_last_error(err: ZTensorError) {
    let msg = CString::new(err.to_string())
        .unwrap_or_else(|_| CString::new("FFI: Unknown error").unwrap());
    *LAST_ERROR.lock().unwrap() = Some(msg);
}

fn set_error(msg: &str) {
    let msg = CString::new(msg).unwrap_or_else(|_| CString::new("FFI error").unwrap());
    *LAST_ERROR.lock().unwrap() = Some(msg);
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_last_error_message() -> *const c_char {
    match LAST_ERROR.lock().unwrap().as_ref() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    }
}

// --- Helper Macros ---

macro_rules! ztensor_handle {
    ($ptr:expr) => {
        if $ptr.is_null() {
            set_error("Null pointer passed as handle");
            return ptr::null_mut();
        } else {
            unsafe { &*$ptr }
        }
    };
    (mut $ptr:expr) => {
        if $ptr.is_null() {
            set_error("Null pointer passed as handle");
            return ptr::null_mut();
        } else {
            unsafe { &mut *$ptr }
        }
    };
    ($ptr:expr, $err_ret:expr) => {
        if $ptr.is_null() {
            set_error("Null pointer passed as handle");
            return $err_ret;
        } else {
            unsafe { &*$ptr }
        }
    };
    (mut $ptr:expr, $err_ret:expr) => {
        if $ptr.is_null() {
            set_error("Null pointer passed as handle");
            return $err_ret;
        } else {
            unsafe { &mut *$ptr }
        }
    };
}

fn to_cstring(s: String) -> *mut c_char {
    CString::new(s).map_or(ptr::null_mut(), |cs| cs.into_raw())
}

fn parse_cstr<'a>(ptr: *const c_char) -> Result<&'a str, ()> {
    if ptr.is_null() {
        return Err(());
    }
    unsafe { CStr::from_ptr(ptr).to_str().map_err(|_| ()) }
}

fn parse_dtype(s: &str) -> Option<DType> {
    match s {
        "float64" | "f64" => Some(DType::F64),
        "float32" | "f32" => Some(DType::F32),
        "float16" | "f16" => Some(DType::F16),
        "bfloat16" | "bf16" => Some(DType::BF16),
        "int64" | "i64" => Some(DType::I64),
        "int32" | "i32" => Some(DType::I32),
        "int16" | "i16" => Some(DType::I16),
        "int8" | "i8" => Some(DType::I8),
        "uint64" | "u64" => Some(DType::U64),
        "uint32" | "u32" => Some(DType::U32),
        "uint16" | "u16" => Some(DType::U16),
        "uint8" | "u8" => Some(DType::U8),
        "bool" => Some(DType::Bool),
        _ => None,
    }
}

// --- Reader API ---

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_open(path_str: *const c_char) -> *mut CZTensorReader {
    let path = match parse_cstr(path_str) {
        Ok(s) => Path::new(s),
        Err(_) => {
            set_error("Invalid UTF-8 path");
            return ptr::null_mut();
        }
    };

    // Check version
    match is_legacy_file(path) {
        Ok(true) => {
             match LegacyReader::open_mmap(path) {
                Ok(reader) => Box::into_raw(Box::new(CZTensorReader::Legacy(reader))),
                Err(e) => {
                    update_last_error(e);
                    ptr::null_mut()
                }
            }
        },
        Ok(false) => {
            match ZTensorReader::open_mmap(path) {
                Ok(reader) => Box::into_raw(Box::new(CZTensorReader::V1(reader))),
                Err(e) => {
                    update_last_error(e);
                    ptr::null_mut()
                }
            }
        },
        Err(e) => {
            update_last_error(e);
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_count(reader_ptr: *const CZTensorReader) -> size_t {
    let reader = ztensor_handle!(reader_ptr, 0);
    match reader {
        CZTensorReader::V1(r) => r.list_objects().len(),
        CZTensorReader::Legacy(r) => r.list_objects().len(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_by_name(
    reader_ptr: *const CZTensorReader,
    name_str: *const c_char,
) -> *mut CObjectMetadata {
    let reader = ztensor_handle!(reader_ptr);
    let name = match parse_cstr(name_str) {
        Ok(s) => s,
        Err(_) => {
            set_error("Invalid UTF-8 name");
            return ptr::null_mut();
        }
    };

    let object = match reader {
        CZTensorReader::V1(r) => r.get_object(name),
        CZTensorReader::Legacy(r) => r.get_object(name),
    };

    match object {
        Some(obj) => Box::into_raw(Box::new((name.to_string(), obj.clone()))),
        None => {
            update_last_error(ZTensorError::ObjectNotFound(name.to_string()));
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_metadata_by_index(
    reader_ptr: *const CZTensorReader,
    index: size_t,
) -> *mut CObjectMetadata {
    let reader = ztensor_handle!(reader_ptr);
    let objects = match reader {
        CZTensorReader::V1(r) => r.list_objects(),
        CZTensorReader::Legacy(r) => r.list_objects(),
    };
    match objects.iter().nth(index) {
        Some((name, obj)) => Box::into_raw(Box::new((name.clone(), obj.clone()))),
        None => {
            set_error(&format!("Index {} out of bounds (len={})", index, objects.len()));
            ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_all_tensor_names(
    reader_ptr: *const CZTensorReader,
) -> *mut CStringArray {
    let reader = ztensor_handle!(reader_ptr);
    let names: Vec<CString> = match reader {
        CZTensorReader::V1(r) => r.list_objects().keys(),
        CZTensorReader::Legacy(r) => r.list_objects().keys(),
    }
        .filter_map(|name| CString::new(name.as_str()).ok())
        .collect();

    let mut c_names: Vec<*mut c_char> = names.into_iter().map(|s| s.into_raw()).collect();
    let arr = Box::new(CStringArray {
        strings: c_names.as_mut_ptr(),
        len: c_names.len(),
    });

    std::mem::forget(c_names);
    Box::into_raw(arr)
}

/// Reads a single object by name. Returns a view that must be freed.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_read_tensor(
    reader_ptr: *mut CZTensorReader,
    name_str: *const c_char,
    verify_checksum: c_int,
) -> *mut CTensorDataView {
    let reader = ztensor_handle!(mut reader_ptr);
    let name = match parse_cstr(name_str) {
        Ok(s) => s,
        Err(_) => {
            set_error("Invalid UTF-8 name");
            return ptr::null_mut();
        }
    };

    let res = match reader {
        CZTensorReader::V1(r) => r.read_object(name, verify_checksum != 0),
        CZTensorReader::Legacy(r) => r.read_object(name, verify_checksum != 0),
    };

    match res {
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

/// Reads multiple objects in batch.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_read_tensors(
    reader_ptr: *mut CZTensorReader,
    names: *const *const c_char,
    names_len: size_t,
    verify_checksum: c_int,
) -> *mut CTensorDataViewArray {
    let reader = ztensor_handle!(mut reader_ptr);

    let name_strs: Vec<&str> = match (0..names_len)
        .map(|i| {
            let name_ptr = unsafe { *names.add(i) };
            parse_cstr(name_ptr).map_err(|_| ZTensorError::Other("Invalid UTF-8".into()))
        })
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(names) => names,
        Err(e) => {
            update_last_error(e);
            return ptr::null_mut();
        }
    };

    let mut views = Vec::with_capacity(name_strs.len());
    let should_verify = verify_checksum != 0;

    for name in name_strs {
        // Try zero-copy first if verification is not requested
        if !should_verify {
             let res = match reader {
                 CZTensorReader::V1(r) => r.get_object_slice(name),
                 CZTensorReader::Legacy(r) => r.get_object_slice(name),
             };
             if let Ok(slice) = res {
                 views.push(CTensorDataView {
                     data: slice.as_ptr(),
                     len: slice.len(),
                     _owner: ptr::null_mut(),
                 });
                 continue;
             }
        }

        // Fallback to copy (standard read)
        let res = match reader {
            CZTensorReader::V1(r) => r.read_object(name, should_verify),
            CZTensorReader::Legacy(r) => r.read_object(name, should_verify),
        };

        match res {
            Ok(data_vec) => {
                 views.push(CTensorDataView {
                    data: data_vec.as_ptr(),
                    len: data_vec.len(),
                    _owner: Box::into_raw(Box::new(data_vec)) as *mut c_void,
                });
            },
            Err(e) => {
                update_last_error(e);
                // Cleanup already allocated views
                for view in views {
                    if !view._owner.is_null() {
                        unsafe { let _ = Box::from_raw(view._owner as *mut Vec<u8>); }
                    }
                }
                return ptr::null_mut();
            }
        }
    }

    let result = Box::new(CTensorDataViewArray {
        views: views.as_mut_ptr(),
        len: views.len(),
    });
    std::mem::forget(views);
    Box::into_raw(result)
}

/// Reads a specific component from an object.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_read_tensor_component(
    reader_ptr: *mut CZTensorReader,
    name_str: *const c_char,
    component_name_str: *const c_char,
) -> *mut CTensorDataView {
    let reader = ztensor_handle!(mut reader_ptr);

    let name = match parse_cstr(name_str) {
        Ok(s) => s,
        Err(_) => {
            set_error("Invalid UTF-8 name");
            return ptr::null_mut();
        }
    };

    let component_name = match parse_cstr(component_name_str) {
        Ok(s) => s,
        Err(_) => {
            set_error("Invalid UTF-8 component name");
            return ptr::null_mut();
        }
    };

    let obj_opt = match reader {
        CZTensorReader::V1(r) => r.get_object(name),
        CZTensorReader::Legacy(r) => r.get_object(name),
    };

    let obj = match obj_opt {
        Some(o) => o.clone(),
        None => {
            update_last_error(ZTensorError::ObjectNotFound(name.to_string()));
            return ptr::null_mut();
        }
    };

    let component = match obj.components.get(component_name) {
        Some(c) => c.clone(),
        None => {
            set_error(&format!("Component '{}' not found in '{}'", component_name, name));
            return ptr::null_mut();
        }
    };

    let res = match reader {
        CZTensorReader::V1(r) => r.read_component(&component),
        CZTensorReader::Legacy(r) => r.read_component(&component),
    };

    match res {
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

/// Reads a zero-copy slice of a tensor.
/// Returns a view that must be freed with ztensor_free_tensor_view.
/// The returned view has no owner (null), so the reader must outlive the view.
#[unsafe(no_mangle)]
pub extern "C" fn ztensor_reader_get_tensor_slice(
    reader_ptr: *const CZTensorReader,
    name_str: *const c_char,
) -> *mut CTensorDataView {
    let reader = ztensor_handle!(reader_ptr);
    let name = match parse_cstr(name_str) {
        Ok(s) => s,
        Err(_) => {
            set_error("Invalid UTF-8 name");
            return ptr::null_mut();
        }
    };

    let res = match reader {
        CZTensorReader::V1(r) => r.get_object_slice(name),
        CZTensorReader::Legacy(r) => r.get_object_slice(name),
    };

    match res {
        Ok(slice) => {
            let view = Box::new(CTensorDataView {
                data: slice.as_ptr(),
                len: slice.len(),
                _owner: ptr::null_mut(),
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
    let path = match parse_cstr(path_str) {
        Ok(s) => Path::new(s),
        Err(_) => {
            set_error("Invalid UTF-8 path");
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

    let name = match parse_cstr(name_str) {
        Ok(s) => s,
        Err(_) => {
            set_error("Invalid UTF-8 name");
            return -1;
        }
    };

    let shape = unsafe { slice::from_raw_parts(shape_ptr, shape_len) };

    let dtype_s = match parse_cstr(dtype_str) {
        Ok(s) => s,
        Err(_) => {
            set_error("Invalid UTF-8 dtype");
            return -1;
        }
    };

    let dtype = match parse_dtype(dtype_s) {
        Some(d) => d,
        None => {
            update_last_error(ZTensorError::UnsupportedDType(dtype_s.to_string()));
            return -1;
        }
    };

    let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
    let compression = if compress == 0 { crate::writer::Compression::Raw } else { crate::writer::Compression::Zstd(compress) };

    match writer.add_object_bytes(name, shape.to_vec(), dtype, compression, data, ChecksumAlgorithm::None) {
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

    let name = match parse_cstr(name_str) {
        Ok(s) => s,
        Err(_) => { set_error("Invalid name"); return -1; }
    };

    let shape = unsafe { slice::from_raw_parts(shape_ptr, shape_len) };
    let dtype_s = match parse_cstr(dtype_str) {
        Ok(s) => s,
        Err(_) => { set_error("Invalid dtype"); return -1; }
    };

    let dtype = match parse_dtype(dtype_s) {
        Some(d) => d,
        None => { update_last_error(ZTensorError::UnsupportedDType(dtype_s.to_string())); return -1; }
    };

    let values = unsafe { slice::from_raw_parts(values_ptr, values_len) };
    let indices = unsafe { slice::from_raw_parts(indices_ptr, indices_len) };
    let indptr = unsafe { slice::from_raw_parts(indptr_ptr, indptr_len) };

    match writer.add_csr_object_bytes(name, shape.to_vec(), dtype, values, indices, indptr, crate::writer::Compression::Raw, ChecksumAlgorithm::None) {
        Ok(_) => 0,
        Err(e) => { update_last_error(e); -1 }
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

    let name = match parse_cstr(name_str) {
        Ok(s) => s,
        Err(_) => { set_error("Invalid name"); return -1; }
    };

    let shape = unsafe { slice::from_raw_parts(shape_ptr, shape_len) };
    let dtype_s = match parse_cstr(dtype_str) {
        Ok(s) => s,
        Err(_) => { set_error("Invalid dtype"); return -1; }
    };

    let dtype = match parse_dtype(dtype_s) {
        Some(d) => d,
        None => { update_last_error(ZTensorError::UnsupportedDType(dtype_s.to_string())); return -1; }
    };

    let values = unsafe { slice::from_raw_parts(values_ptr, values_len) };
    let coords = unsafe { slice::from_raw_parts(indices_ptr, indices_len) };

    match writer.add_coo_object_bytes(name, shape.to_vec(), dtype, values, coords, crate::writer::Compression::Raw, ChecksumAlgorithm::None) {
        Ok(_) => 0,
        Err(e) => { update_last_error(e); -1 }
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
        Err(e) => { update_last_error(e); -1 }
    }
}

// --- Metadata API ---

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_name(metadata_ptr: *const CObjectMetadata) -> *mut c_char {
    let (name, _) = ztensor_handle!(metadata_ptr);
    to_cstring(name.clone())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_dtype_str(metadata_ptr: *const CObjectMetadata) -> *mut c_char {
    let (_, obj) = ztensor_handle!(metadata_ptr);
    // Get dtype from "data" (dense) or "values" (sparse) component
    let dtype = obj.components.get("data")
        .or_else(|| obj.components.get("values"))
        .map(|c| c.dtype);
    let dtype_str = match dtype {
        Some(DType::F64) => "float64",
        Some(DType::F32) => "float32",
        Some(DType::F16) => "float16",
        Some(DType::BF16) => "bfloat16",
        Some(DType::I64) => "int64",
        Some(DType::I32) => "int32",
        Some(DType::I16) => "int16",
        Some(DType::I8) => "int8",
        Some(DType::U64) => "uint64",
        Some(DType::U32) => "uint32",
        Some(DType::U16) => "uint16",
        Some(DType::U8) => "uint8",
        Some(DType::Bool) => "bool",
        None => "unknown",
    };
    to_cstring(dtype_str.to_string())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_offset(metadata_ptr: *const CObjectMetadata) -> u64 {
    let (_, obj) = ztensor_handle!(metadata_ptr, 0);
    obj.components.get("data").map_or(0, |c| c.offset)
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_size(metadata_ptr: *const CObjectMetadata) -> u64 {
    let (_, obj) = ztensor_handle!(metadata_ptr, 0);
    obj.components.get("data").map_or(0, |c| c.length)
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_layout_str(metadata_ptr: *const CObjectMetadata) -> *mut c_char {
    let (_, obj) = ztensor_handle!(metadata_ptr);
    to_cstring(obj.format.clone())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_encoding_str(metadata_ptr: *const CObjectMetadata) -> *mut c_char {
    let (_, obj) = ztensor_handle!(metadata_ptr);
    obj.components.get("data")
        .map_or(ptr::null_mut(), |c| to_cstring(format!("{:?}", c.encoding).to_lowercase()))
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_data_endianness_str(_metadata_ptr: *const CObjectMetadata) -> *mut c_char {
    to_cstring("little".to_string())
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_checksum_str(metadata_ptr: *const CObjectMetadata) -> *mut c_char {
    let (_, obj) = ztensor_handle!(metadata_ptr);
    obj.components.get("data")
        .and_then(|c| c.digest.as_ref())
        .map_or(ptr::null_mut(), |s| to_cstring(s.clone()))
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_shape_len(metadata_ptr: *const CObjectMetadata) -> size_t {
    let (_, obj) = ztensor_handle!(metadata_ptr, 0);
    obj.shape.len()
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_metadata_get_shape_data(metadata_ptr: *const CObjectMetadata) -> *mut u64 {
    let (_, obj) = ztensor_handle!(metadata_ptr);
    let mut shape_vec = obj.shape.clone();
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
pub extern "C" fn ztensor_metadata_free(metadata_ptr: *mut CObjectMetadata) {
    if !metadata_ptr.is_null() {
        let _ = unsafe { Box::from_raw(metadata_ptr) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ztensor_free_tensor_view(view_ptr: *mut CTensorDataView) {
    if !view_ptr.is_null() {
        unsafe {
            let view = Box::from_raw(view_ptr);
            if !view._owner.is_null() {
                let _ = Box::from_raw(view._owner as *mut Vec<u8>);
            }
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
            if !view._owner.is_null() {
                let _ = Box::from_raw(view._owner as *mut Vec<u8>);
            }
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
