//! PyO3 Python bindings for zTensor.
//!
//! Provides native Python classes that replace the CFFI-based bindings.
//! The `PyReader` wraps `Box<dyn TensorReader + Send>` for format-agnostic access,
//! and `PyWriter` wraps `Writer` for writing zTensor files.

use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyArrayMethods};

use crate::error::Error;
use crate::models::{Checksum, DType, Format};
use crate::reader::{TensorData, TensorReader};
use crate::writer::{Compression, Writer};

// =========================================================================
// DType ↔ NumPy dtype string conversion
// =========================================================================

fn dtype_to_numpy_str(dtype: DType) -> &'static str {
    match dtype {
        DType::F64 => "float64",
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::I64 => "int64",
        DType::I32 => "int32",
        DType::I16 => "int16",
        DType::I8 => "int8",
        DType::U64 => "uint64",
        DType::U32 => "uint32",
        DType::U16 => "uint16",
        DType::U8 => "uint8",
        DType::Bool => "bool",
    }
}

fn numpy_str_to_dtype(s: &str) -> Result<DType, PyErr> {
    match s {
        "float64" | "<f8" | "=f8" => Ok(DType::F64),
        "float32" | "<f4" | "=f4" => Ok(DType::F32),
        "float16" | "<f2" | "=f2" => Ok(DType::F16),
        "bfloat16" => Ok(DType::BF16),
        "int64" | "<i8" | "=i8" => Ok(DType::I64),
        "int32" | "<i4" | "=i4" => Ok(DType::I32),
        "int16" | "<i2" | "=i2" => Ok(DType::I16),
        "int8" | "|i1" => Ok(DType::I8),
        "uint64" | "<u8" | "=u8" => Ok(DType::U64),
        "uint32" | "<u4" | "=u4" => Ok(DType::U32),
        "uint16" | "<u2" | "=u2" => Ok(DType::U16),
        "uint8" | "|u1" => Ok(DType::U8),
        "bool" | "|b1" => Ok(DType::Bool),
        _ => Err(PyValueError::new_err(format!("Unsupported dtype: {}", s))),
    }
}

fn zt_err(e: Error) -> PyErr {
    match &e {
        Error::ObjectNotFound(_) => PyKeyError::new_err(e.to_string()),
        Error::Io(_) => PyIOError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

// =========================================================================
// PyTensorMeta — metadata for a single tensor
// =========================================================================

/// Metadata for a single tensor in a zTensor file.
#[pyclass(name = "TensorMetadata")]
#[derive(Clone)]
pub struct PyTensorMeta {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    dtype: String,
    /// Logical type (e.g., "f8_e4m3fn"). None when equals dtype.
    #[pyo3(get)]
    r#type: Option<String>,
    #[pyo3(get)]
    shape: Vec<u64>,
    #[pyo3(get)]
    format: String,
}

#[pymethods]
impl PyTensorMeta {
    fn __repr__(&self) -> String {
        if let Some(ref t) = self.r#type {
            format!(
                "<TensorMetadata name='{}' shape={:?} dtype='{}' type='{}'>",
                self.name, self.shape, self.dtype, t
            )
        } else {
            format!(
                "<TensorMetadata name='{}' shape={:?} dtype='{}'>",
                self.name, self.shape, self.dtype
            )
        }
    }
}

// =========================================================================
// PyReader — format-agnostic tensor reader
// =========================================================================

/// Reader for tensor files (zTensor, SafeTensors, PyTorch, GGUF, NPZ, ONNX, HDF5).
///
/// Automatically detects the file format based on file extension.
/// Supports dict-like access: `reader["tensor_name"]` returns a NumPy array.
#[pyclass(name = "Reader")]
pub struct PyReader {
    inner: Box<dyn TensorReader + Send>,
    file_format: String,
}

// Safety: PyReader holds Box<dyn TensorReader + Send>.
// PyO3 requires Sync for pyclass. We guarantee single-threaded access
// through the GIL — Python holds the GIL when calling any method.
unsafe impl Sync for PyReader {}

/// Convert raw tensor bytes to a NumPy array.
fn bytes_to_numpy<'py>(
    py: Python<'py>,
    data: TensorData<'_>,
    dtype: DType,
    shape: &[u64],
) -> PyResult<PyObject> {
    let bytes = data.as_slice();
    let np_dtype_str = dtype_to_numpy_str(dtype);

    // For bfloat16, we need ml_dtypes
    if dtype == DType::BF16 {
        let n_elements = bytes.len() / 2;
        let arr = unsafe { PyArray1::<u16>::new(py, [n_elements], false) };
        unsafe {
            debug_assert_eq!(bytes.len(), n_elements * std::mem::size_of::<u16>());
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), arr.data() as *mut u8, bytes.len());
        }
        let np = py.import("numpy")?;
        let result = arr.into_pyobject(py)?.into_any();

        match py.import("ml_dtypes") {
            Ok(_ml) => {
                let bf16_dtype = np.call_method1("dtype", ("bfloat16",))?;
                let viewed = result.call_method1("view", (bf16_dtype,))?;
                let shape_tuple: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
                let reshaped = viewed.call_method1("reshape", (shape_tuple,))?;
                Ok(reshaped.into_pyobject(py)?.into_any().unbind())
            }
            Err(_) => Err(PyRuntimeError::new_err(
                "bfloat16 tensors require the 'ml_dtypes' package",
            )),
        }
    } else {
        let np = py.import("numpy")?;
        let np_dtype = np.call_method1("dtype", (np_dtype_str,))?;
        let byte_array = unsafe { PyArray1::<u8>::new(py, [bytes.len()], false) };
        unsafe {
            // Safety: byte_array was created with size [bytes.len()] on the line above.
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), byte_array.data(), bytes.len());
        }
        let flat = byte_array
            .into_pyobject(py)?
            .into_any()
            .call_method1("view", (np_dtype,))?;

        if shape.is_empty() {
            Ok(flat.into_pyobject(py)?.into_any().unbind())
        } else {
            let shape_tuple: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
            let reshaped = flat.call_method1("reshape", (shape_tuple,))?;
            Ok(reshaped.into_pyobject(py)?.into_any().unbind())
        }
    }
}

/// Intermediate result for tensor data that doesn't borrow from the reader.
enum ReadResult {
    /// Zero-copy: raw pointer and length into the reader's backing store.
    ZeroCopy { ptr: *const u8, len: usize },
    /// Owned copy of the data.
    Owned(Vec<u8>),
}

/// Convert raw tensor bytes to a zero-copy NumPy array.
///
/// Creates a numpy array backed by `slice` without copying. The `owner` Python
/// object is set as the array's base, preventing garbage collection while the
/// array (or any view of it) is alive.
fn bytes_to_numpy_borrowed<'py>(
    py: Python<'py>,
    slice: &[u8],
    dtype: DType,
    shape: &[u64],
    owner: Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let view = ArrayView1::from(slice);

    // Safety: `owner` (the PyReader) keeps the backing memory (mmap or buffer)
    // alive. numpy tracks owner as the array's base, preventing GC.
    let np_array = unsafe { PyArray1::<u8>::borrow_from_array(&view, owner) };

    let np = py.import("numpy")?;
    let result = if dtype == DType::BF16 {
        let u16_dtype = np.call_method1("dtype", ("uint16",))?;
        let viewed_u16 = np_array
            .into_pyobject(py)?
            .into_any()
            .call_method1("view", (u16_dtype,))?;

        match py.import("ml_dtypes") {
            Ok(_ml) => {
                let bf16_dtype = np.call_method1("dtype", ("bfloat16",))?;
                let viewed = viewed_u16.call_method1("view", (bf16_dtype,))?;
                let shape_tuple: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
                viewed.call_method1("reshape", (shape_tuple,))?
            }
            Err(_) => {
                return Err(PyRuntimeError::new_err(
                    "bfloat16 tensors require the 'ml_dtypes' package",
                ))
            }
        }
    } else {
        let np_dtype_str = dtype_to_numpy_str(dtype);
        let np_dtype = np.call_method1("dtype", (np_dtype_str,))?;
        let flat = np_array
            .into_pyobject(py)?
            .into_any()
            .call_method1("view", (np_dtype,))?;

        if shape.is_empty() {
            flat
        } else {
            let shape_tuple: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
            flat.call_method1("reshape", (shape_tuple,))?
        }
    };

    Ok(result.unbind())
}

/// Returns a cached numpy dtype object, creating it if not yet cached.
fn cached_np_dtype<'py>(
    py: Python<'py>,
    cache: &mut Vec<(DType, PyObject)>,
    np: &Bound<'py, PyModule>,
    dtype: DType,
) -> PyResult<PyObject> {
    for (d, obj) in cache.iter() {
        if *d == dtype {
            return Ok(obj.clone_ref(py));
        }
    }
    let np_dtype_str = dtype_to_numpy_str(dtype);
    let dt = np.call_method1("dtype", (np_dtype_str,))?.unbind();
    let ret = dt.clone_ref(py);
    cache.push((dtype, dt));
    Ok(ret)
}

/// Builds zero-copy numpy arrays for all tensors using a single borrow.
///
/// Collects all tensor metadata and mmap pointers in one borrow, sorts by
/// file offset for sequential page fault access, and caches numpy dtype objects.
fn build_numpy_dict_zerocopy<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyReader>,
) -> PyResult<Vec<(String, PyObject)>> {
    let np = py.import("numpy")?;
    let mut dtype_cache: Vec<(DType, PyObject)> = Vec::new();
    let mut results = Vec::new();

    // Single borrow to collect all tensor metadata and data pointers
    let mut tensor_infos: Vec<(String, DType, Vec<u64>, ReadResult, u64)> = {
        let this = slf.borrow();
        let mut infos = Vec::new();
        for (name, obj) in this.inner.tensors() {
            let component = obj.components.get("data").ok_or_else(|| {
                PyRuntimeError::new_err(format!("Missing 'data' component for '{}'", name))
            })?;
            let data = this.inner.read_data(name).map_err(zt_err)?;
            let result = match &data {
                TensorData::Borrowed(slice) => ReadResult::ZeroCopy {
                    ptr: slice.as_ptr(),
                    len: slice.len(),
                },
                TensorData::Owned(vec) => ReadResult::Owned(vec.clone()),
            };
            infos.push((
                name.clone(),
                component.dtype,
                obj.shape.clone(),
                result,
                component.offset,
            ));
        }
        infos
    };

    // Sort by file offset for sequential page fault access
    tensor_infos.sort_by_key(|info| info.4);

    for (name, dtype, shape, data, _) in &tensor_infos {
        let arr = match data {
            ReadResult::ZeroCopy { ptr, len } => {
                let slice = unsafe { std::slice::from_raw_parts(*ptr, *len) };
                if *dtype == DType::BF16 {
                    bytes_to_numpy_borrowed(py, slice, *dtype, shape, slf.clone().into_any())?
                } else {
                    let view = ArrayView1::from(slice);
                    let np_array =
                        unsafe { PyArray1::<u8>::borrow_from_array(&view, slf.clone().into_any()) };
                    let np_dtype = cached_np_dtype(py, &mut dtype_cache, &np, *dtype)?;
                    let flat = np_array
                        .into_pyobject(py)?
                        .into_any()
                        .call_method1("view", (np_dtype.bind(py),))?;
                    if shape.is_empty() {
                        flat.unbind()
                    } else {
                        let shape_tuple: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
                        flat.call_method1("reshape", (shape_tuple,))?.unbind()
                    }
                }
            }
            ReadResult::Owned(vec) => {
                bytes_to_numpy(py, TensorData::Borrowed(vec.as_slice()), *dtype, shape)?
            }
        };
        results.push((name.clone(), arr));
    }

    Ok(results)
}

/// Returns the torch dtype attribute name for a given DType.
fn dtype_to_torch_attr(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F64 => Some("float64"),
        DType::F32 => Some("float32"),
        DType::F16 => Some("float16"),
        DType::BF16 => Some("bfloat16"),
        DType::I64 => Some("int64"),
        DType::I32 => Some("int32"),
        DType::I16 => Some("int16"),
        DType::I8 => Some("int8"),
        DType::U8 => Some("uint8"),
        DType::Bool => Some("bool"),
        // torch doesn't natively support these unsigned types in older versions
        DType::U16 | DType::U32 | DType::U64 => None,
    }
}

/// Creates a torch tensor directly from raw byte data.
///
/// Uses `torch.empty(shape, dtype=dtype)` + memcpy into `data_ptr()`,
/// the same approach used by the native safetensors library.
fn create_torch_tensor<'py>(
    py: Python<'py>,
    torch: &Bound<'py, PyModule>,
    dtype: DType,
    shape: &[u64],
    src: &[u8],
) -> PyResult<PyObject> {
    let torch_attr = dtype_to_torch_attr(dtype).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "Unsupported dtype for torch: {:?}. Use load_numpy() instead.",
            dtype
        ))
    })?;
    let torch_dtype = torch.getattr(torch_attr)?;

    let shape_list: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", &torch_dtype)?;
    let tensor = torch.call_method("empty", (&shape_list,), Some(&kwargs))?;

    // Get storage data_ptr and memcpy from mmap/buffer directly
    let storage = tensor.call_method0("untyped_storage")?;
    let data_ptr = storage.call_method0("data_ptr")?.extract::<usize>()?;
    let nbytes = storage.call_method0("nbytes")?.extract::<usize>()?;

    let copy_len = src.len().min(nbytes);
    if copy_len > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), data_ptr as *mut u8, copy_len);
        }
    }

    Ok(tensor.unbind())
}

/// Core implementation for reading a single tensor, supporting zero-copy.
fn read_tensor_impl<'py>(slf: &Bound<'py, PyReader>, name: &str) -> PyResult<PyObject> {
    let py = slf.py();

    // Borrow the reader, extract metadata and data location, then release borrow.
    let (result, dtype, shape) = {
        let this = slf.borrow();
        let obj = this
            .inner
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?
            .clone();

        let dtype = obj.components.get("data").map(|c| c.dtype).ok_or_else(|| {
            PyRuntimeError::new_err(format!("Missing 'data' component for '{}'", name))
        })?;
        let shape = obj.shape.clone();

        let data = this.inner.read_data(name).map_err(zt_err)?;
        let result = match &data {
            TensorData::Borrowed(slice) => ReadResult::ZeroCopy {
                ptr: slice.as_ptr(),
                len: slice.len(),
            },
            TensorData::Owned(vec) => ReadResult::Owned(vec.clone()),
        };

        (result, dtype, shape)
        // `this` (PyRef) dropped here, releasing the borrow
    };

    match result {
        ReadResult::ZeroCopy { ptr, len } => {
            // Safety: ptr/len came from TensorData::Borrowed which borrows from
            // self.inner (the TensorReader). The PyReader (slf) owns the reader
            // and is passed as the numpy base object, preventing GC. The backing
            // memory is either mmap-pinned or a Vec inside the reader — both
            // stable for the lifetime of the PyReader.
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            bytes_to_numpy_borrowed(py, slice, dtype, &shape, slf.clone().into_any())
        }
        ReadResult::Owned(data) => bytes_to_numpy(py, TensorData::Owned(data), dtype, &shape),
    }
}

#[pymethods]
impl PyReader {
    /// Open a tensor file (auto-detects format by extension).
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        let file_format = match ext {
            "zt" => "zt",
            "safetensors" => "safetensors",
            "pt" | "pth" | "bin" | "pkl" => "pickle",
            "gguf" => "gguf",
            "npz" => "npz",
            "onnx" => "onnx",
            "h5" | "hdf5" => "hdf5",
            _ => "unknown",
        }
        .to_string();
        // Open .zt files with MAP_PRIVATE directly to avoid re-opening later
        // for zero-copy (copy=False) loads. MAP_PRIVATE has identical read
        // performance to MAP_SHARED; it only adds COW on writes.
        let inner: Box<dyn TensorReader + Send> = if file_format == "zt" || file_format == "unknown"
        {
            Box::new(crate::Reader::open_mmap_any(path).map_err(zt_err)?)
        } else {
            crate::open(path).map_err(zt_err)?
        };
        Ok(Self { inner, file_format })
    }

    /// Returns a list of all tensor names.
    fn keys(&self) -> Vec<String> {
        self.inner.keys().iter().map(|s| s.to_string()).collect()
    }

    /// Returns metadata for a tensor by name.
    fn metadata(&self, name: &str) -> PyResult<PyTensorMeta> {
        let obj = self
            .inner
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;
        let component = obj.components.get("data");
        let dtype_str = component
            .map(|c| dtype_to_numpy_str(c.dtype))
            .unwrap_or("unknown");
        let format_str = match &obj.format {
            Format::Dense => "dense",
            Format::SparseCsr => "sparse_csr",
            Format::SparseCoo => "sparse_coo",
            Format::QuantizedGroup => "quantized_group",
            Format::Other(s) => s.as_str(),
        };
        let tensor_type = component.and_then(|c| c.r#type.clone());
        Ok(PyTensorMeta {
            name: name.to_string(),
            dtype: dtype_str.to_string(),
            r#type: tensor_type,
            shape: obj.shape.clone(),
            format: format_str.to_string(),
        })
    }

    /// Reads a single tensor as a NumPy array (zero-copy when possible).
    fn read_tensor<'py>(slf: &Bound<'py, PyReader>, name: &str) -> PyResult<PyObject> {
        read_tensor_impl(slf, name)
    }

    /// Reads multiple tensors as a dict of {name: numpy_array}.
    fn read_tensors<'py>(
        slf: &Bound<'py, PyReader>,
        names: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let dict = PyDict::new(py);
        for name in &names {
            let arr = read_tensor_impl(slf, name)?;
            dict.set_item(name, arr)?;
        }
        Ok(dict)
    }

    /// Loads all tensors as a dict of torch.Tensor.
    ///
    /// When `copy=False` (default), returns tensors backed by memory-mapped data with
    /// copy-on-write semantics (zero-copy reads, much faster for inference).
    /// The tensors are writable — writes trigger per-page copies at the OS level.
    /// When `copy=True`, creates independent tensors via torch.empty() + memcpy.
    #[pyo3(signature = (*, device="cpu", copy=false))]
    fn load_torch<'py>(
        slf: &Bound<'py, PyReader>,
        device: &str,
        copy: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let torch = py.import("torch")?;
        let dict = PyDict::new(py);

        if copy {
            // Eager copy: torch.empty() + memcpy — writable tensors.
            // Sort by file offset for sequential I/O on cold mmap.
            let tensor_infos: Vec<(String, DType, Vec<u64>, u64, ReadResult)> = {
                let this = slf.borrow();
                let mut infos = Vec::new();
                for (name, obj) in this.inner.tensors() {
                    let component = obj.components.get("data").ok_or_else(|| {
                        PyRuntimeError::new_err(format!("Missing 'data' component for '{}'", name))
                    })?;
                    let dtype = component.dtype;
                    let offset = component.offset;
                    let shape = obj.shape.clone();
                    let data = this.inner.read_data(name).map_err(zt_err)?;
                    let result = match &data {
                        TensorData::Borrowed(slice) => ReadResult::ZeroCopy {
                            ptr: slice.as_ptr(),
                            len: slice.len(),
                        },
                        TensorData::Owned(vec) => ReadResult::Owned(vec.clone()),
                    };
                    infos.push((name.clone(), dtype, shape, offset, result));
                }
                infos
            };

            // Sort by file offset for sequential access
            let mut sorted_infos = tensor_infos;
            sorted_infos.sort_by_key(|info| info.3);

            for (name, dtype, shape, _offset, data) in &sorted_infos {
                let src_slice = match data {
                    ReadResult::ZeroCopy { ptr, len } => unsafe {
                        std::slice::from_raw_parts(*ptr, *len)
                    },
                    ReadResult::Owned(vec) => vec.as_slice(),
                };
                let tensor = create_torch_tensor(py, &torch, *dtype, shape, src_slice)?;
                dict.set_item(name, tensor)?;
            }
        } else {
            // Zero-copy: numpy mmap view → torch.from_numpy — COW-writable tensors.
            // Optimized bulk path: single borrow, cached dtypes, offset-sorted.
            let from_numpy = torch.getattr("from_numpy")?;
            let arrays = build_numpy_dict_zerocopy(py, slf)?;
            for (name, arr) in arrays {
                let tensor = from_numpy.call1((arr,))?;
                dict.set_item(&name, tensor)?;
            }
        }

        // Device transfer if needed
        if device != "cpu" {
            let target = torch.call_method1("device", (device,))?;
            let items: Vec<(PyObject, PyObject)> =
                dict.iter().map(|(k, v)| (k.unbind(), v.unbind())).collect();
            for (key, tensor) in items {
                let moved = tensor.call_method1(py, "to", (&target,))?;
                dict.set_item(&key, moved)?;
            }
        }

        Ok(dict)
    }

    /// Loads all tensors as a dict of numpy arrays.
    ///
    /// When `copy=False` (default), returns arrays backed by memory-mapped data with
    /// copy-on-write semantics (zero-copy reads, much faster).
    /// When `copy=True`, returns independent arrays (data is copied).
    #[pyo3(signature = (*, copy=false))]
    fn load_numpy<'py>(slf: &Bound<'py, PyReader>, copy: bool) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let dict = PyDict::new(py);

        if copy {
            // Independent arrays — collect data then create copies.
            // Sort by file offset for sequential I/O on cold mmap.
            let tensor_infos: Vec<(String, DType, Vec<u64>, u64, ReadResult)> = {
                let this = slf.borrow();
                let mut infos = Vec::new();
                for (name, obj) in this.inner.tensors() {
                    let component = obj.components.get("data").ok_or_else(|| {
                        PyRuntimeError::new_err(format!("Missing 'data' component for '{}'", name))
                    })?;
                    let dtype = component.dtype;
                    let offset = component.offset;
                    let shape = obj.shape.clone();
                    let data = this.inner.read_data(name).map_err(zt_err)?;
                    let result = match &data {
                        TensorData::Borrowed(slice) => ReadResult::ZeroCopy {
                            ptr: slice.as_ptr(),
                            len: slice.len(),
                        },
                        TensorData::Owned(vec) => ReadResult::Owned(vec.clone()),
                    };
                    infos.push((name.clone(), dtype, shape, offset, result));
                }
                infos
            };

            // Sort by file offset for sequential access
            let mut sorted_infos = tensor_infos;
            sorted_infos.sort_by_key(|info| info.3);

            for (name, dtype, shape, _offset, data) in &sorted_infos {
                let src_slice = match data {
                    ReadResult::ZeroCopy { ptr, len } => unsafe {
                        std::slice::from_raw_parts(*ptr, *len)
                    },
                    ReadResult::Owned(vec) => vec.as_slice(),
                };
                let arr = bytes_to_numpy(py, TensorData::Borrowed(src_slice), *dtype, shape)?;
                dict.set_item(name, arr)?;
            }
        } else {
            // Zero-copy mmap-backed arrays with optimized bulk path:
            // single borrow, cached dtypes, offset-sorted for sequential I/O.
            let arrays = build_numpy_dict_zerocopy(py, slf)?;
            for (name, arr) in arrays {
                dict.set_item(&name, arr)?;
            }
        }

        Ok(dict)
    }

    /// Number of tensors in the file.
    fn __len__(&self) -> usize {
        self.inner.tensors().len()
    }

    /// Check if a tensor name exists.
    fn __contains__(&self, name: &str) -> bool {
        self.inner.get(name).is_some()
    }

    /// Dict-like access: reader["name"] returns a NumPy array (zero-copy when possible).
    fn __getitem__<'py>(slf: &Bound<'py, PyReader>, name: &str) -> PyResult<PyObject> {
        read_tensor_impl(slf, name)
    }

    /// The detected file format: "zt", "safetensors", "pickle", "gguf", "npz", "onnx", "hdf5", or "unknown".
    #[getter]
    fn format(&self) -> &str {
        &self.file_format
    }

    fn __repr__(&self) -> String {
        let count = self.inner.tensors().len();
        format!(
            "<ztensor.Reader [{}, {} tensor(s)]>",
            self.file_format, count
        )
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let names: Vec<String> = self.keys();
        PyList::new(py, names)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_val: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_tb: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> bool {
        false
    }
}

// =========================================================================
// PyWriter — zTensor file writer
// =========================================================================

/// Writer for creating zTensor files.
///
/// Usage:
///     writer = ztensor.Writer("output.zt")
///     writer.add("weights", numpy_array)
///     writer.finish()
///
/// Or as a context manager:
///     with ztensor.Writer("output.zt") as w:
///         w.add("weights", numpy_array)
#[pyclass(name = "Writer")]
pub struct PyWriter {
    inner: Option<Writer<std::io::BufWriter<std::fs::File>>>,
}

#[pymethods]
impl PyWriter {
    /// Create a new writer for the given file path.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = Writer::create(path).map_err(zt_err)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Add a dense tensor from a NumPy array.
    ///
    /// Args:
    ///     name: Tensor name.
    ///     data: NumPy array (must be contiguous).
    ///     compress: False/0 for raw, True for default zstd, int for specific level.
    ///     checksum: "none", "crc32c", or "sha256".
    #[pyo3(signature = (name, data, compress=None, checksum="none"))]
    fn add(
        &mut self,
        py: Python<'_>,
        name: &str,
        data: &Bound<'_, pyo3::types::PyAny>,
        compress: Option<&Bound<'_, pyo3::types::PyAny>>,
        checksum: &str,
    ) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Writer is already finished"))?;

        // Parse compression
        let compression = match compress {
            Some(obj) => parse_compression(obj)?,
            None => Compression::Raw,
        };

        // Parse checksum
        let checksum_alg = match checksum {
            "none" => Checksum::None,
            "crc32c" => Checksum::Crc32c,
            "sha256" => Checksum::Sha256,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown checksum: {}",
                    checksum
                )))
            }
        };

        // Extract bytes and metadata from the array
        let np = py.import("numpy")?;
        let arr = np.call_method1("ascontiguousarray", (data,))?;
        let dtype_str: String = arr.getattr("dtype")?.call_method0("__str__")?.extract()?;
        let dtype = numpy_str_to_dtype(&dtype_str)?;

        let shape: Vec<u64> = arr
            .getattr("shape")?
            .extract::<Vec<i64>>()?
            .iter()
            .map(|&s| s as u64)
            .collect();

        let nbytes: usize = arr.getattr("nbytes")?.extract()?;

        // Zero-copy: access numpy array's raw buffer directly via __array_interface__
        let iface = arr.getattr("__array_interface__")?;
        let data_tuple = iface.get_item("data")?;
        let ptr: usize = data_tuple.get_item(0)?.extract()?;
        // SAFETY: GIL is held for the entire method, array is guaranteed contiguous
        // by ascontiguousarray above, and the pointer is valid for this scope.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, nbytes) };

        writer
            .add_bytes(name, shape, dtype, compression, bytes, checksum_alg)
            .map_err(zt_err)
    }

    /// Finish the file (writes manifest and footer).
    fn finish(&mut self) -> PyResult<u64> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Writer is already finished"))?;
        writer.finish().map_err(zt_err)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_val: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_tb: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<bool> {
        if self.inner.is_some() {
            if exc_type.is_none() {
                self.finish()?;
            } else {
                self.inner.take();
            }
        }
        Ok(false)
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "<ztensor.Writer (open)>".to_string()
        } else {
            "<ztensor.Writer (finished)>".to_string()
        }
    }
}

fn parse_compression(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Compression> {
    // Try bool first (before int, since Python bool is a subclass of int)
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(if b {
            Compression::Zstd(3)
        } else {
            Compression::Raw
        });
    }
    if let Ok(level) = obj.extract::<i32>() {
        return Ok(if level <= 0 {
            Compression::Raw
        } else {
            Compression::Zstd(level)
        });
    }
    Err(PyTypeError::new_err("compress must be bool or int"))
}

// =========================================================================
// Top-level functions
// =========================================================================

/// Open a tensor file (auto-detects format by extension).
///
/// Supports .zt (zTensor), .safetensors, .pt/.bin/.pth (PyTorch).
#[pyfunction]
fn open(path: &str) -> PyResult<PyReader> {
    PyReader::new(path)
}

/// Save a dict of numpy arrays to a ztensor file in a single FFI call.
///
/// This is faster than using Writer + add() because it avoids per-tensor
/// FFI overhead and keeps all I/O in Rust.
#[pyfunction]
#[pyo3(signature = (tensors, filename, compression=None))]
fn save_file(
    py: Python<'_>,
    tensors: &Bound<'_, PyDict>,
    filename: &str,
    compression: Option<&Bound<'_, pyo3::types::PyAny>>,
) -> PyResult<()> {
    let compress = match compression {
        Some(obj) => parse_compression(obj)?,
        None => Compression::Raw,
    };

    let file = std::fs::File::create(filename).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let buf_writer = std::io::BufWriter::with_capacity(256 * 1024, file);
    let mut writer = Writer::new(buf_writer).map_err(zt_err)?;

    let np = py.import("numpy")?;

    for item in tensors.iter() {
        let (key, value) = item;
        let name: String = key.extract()?;
        let arr = np.call_method1("ascontiguousarray", (&value,))?;

        let dtype_str: String = arr.getattr("dtype")?.call_method0("__str__")?.extract()?;
        let dtype = numpy_str_to_dtype(&dtype_str)?;
        let shape: Vec<u64> = arr
            .getattr("shape")?
            .extract::<Vec<i64>>()?
            .iter()
            .map(|&s| s as u64)
            .collect();
        let nbytes: usize = arr.getattr("nbytes")?.extract()?;

        // Zero-copy: access numpy array's raw buffer directly
        let iface = arr.getattr("__array_interface__")?;
        let data_tuple = iface.get_item("data")?;
        let ptr: usize = data_tuple.get_item(0)?.extract()?;
        // SAFETY: GIL is held, array is contiguous from ascontiguousarray above.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, nbytes) };

        writer
            .add_bytes(&name, shape, dtype, compress, bytes, Checksum::None)
            .map_err(zt_err)?;
    }

    writer.finish().map_err(zt_err)?;
    Ok(())
}

/// The native zTensor Python module.
#[pymodule]
fn _ztensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReader>()?;
    m.add_class::<PyWriter>()?;
    m.add_class::<PyTensorMeta>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(save_file, m)?)?;
    Ok(())
}
