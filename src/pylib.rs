//! PyO3 Python bindings for zTensor.
//!
//! Provides native Python classes that replace the CFFI-based bindings.
//! The `PyReader` wraps `Box<dyn TensorReader + Send>` for format-agnostic access,
//! and `PyWriter` wraps `Writer` for writing zTensor files.

use std::collections::BTreeMap;

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

fn parse_checksum_str(s: &str) -> PyResult<Checksum> {
    match s {
        "none" => Ok(Checksum::None),
        "crc32c" => Ok(Checksum::Crc32c),
        "sha256" => Ok(Checksum::Sha256),
        _ => Err(PyValueError::new_err(format!("Unknown checksum: {}", s))),
    }
}

// =========================================================================
// CBOR ↔ Python value conversion
// =========================================================================

fn cbor_value_to_python(py: Python<'_>, val: &ciborium::Value) -> PyResult<PyObject> {
    match val {
        ciborium::Value::Text(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            Ok(n.into_pyobject(py)?.into_any().unbind())
        }
        ciborium::Value::Float(f) => Ok(f.into_pyobject(py)?.into_any().unbind()),
        ciborium::Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        ciborium::Value::Bytes(b) => Ok(b.as_slice().into_pyobject(py)?.into_any().unbind()),
        ciborium::Value::Null => Ok(py.None()),
        ciborium::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(cbor_value_to_python(py, item)?)?;
            }
            Ok(list.into_pyobject(py)?.into_any().unbind())
        }
        ciborium::Value::Map(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                let key = cbor_value_to_python(py, k)?;
                let val = cbor_value_to_python(py, v)?;
                dict.set_item(key, val)?;
            }
            Ok(dict.into_pyobject(py)?.into_any().unbind())
        }
        _ => Ok(py.None()),
    }
}

fn python_to_cbor_value(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<ciborium::Value> {
    // Check bool before int (bool is subclass of int in Python)
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(ciborium::Value::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ciborium::Value::Integer(i.into()));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(ciborium::Value::Float(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(ciborium::Value::Text(s));
    }
    if let Ok(b) = obj.extract::<Vec<u8>>() {
        if obj.is_instance_of::<pyo3::types::PyBytes>() {
            return Ok(ciborium::Value::Bytes(b));
        }
    }
    if obj.is_none() {
        return Ok(ciborium::Value::Null);
    }
    // List → CBOR Array
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(python_to_cbor_value(&item)?);
        }
        return Ok(ciborium::Value::Array(arr));
    }
    // Dict → CBOR Map
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            map.push((python_to_cbor_value(&k)?, python_to_cbor_value(&v)?));
        }
        return Ok(ciborium::Value::Map(map));
    }
    Err(PyTypeError::new_err(format!(
        "Cannot convert {} to CBOR value",
        obj.get_type().name()?
    )))
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
// PyTensor — native Python tensor type
// =========================================================================

/// A tensor object representing a zt Object.
///
/// Wraps one or more component arrays with metadata (shape, dtype, format).
/// Dense tensors have a single "data" component; sparse and quantized tensors
/// have multiple components (e.g., "values", "indices", "indptr").
///
/// Example:
///     t = ztensor.Tensor({"data": np_array})
///     t = ztensor.Tensor({"values": v, "indices": i, "indptr": p},
///                         shape=[4096, 4096], dtype="float32", format="sparse_csr")
#[pyclass(name = "Tensor")]
pub struct PyTensor {
    shape: Vec<i64>,
    dtype: String,
    type_: Option<String>,
    format_str: String,
    components: PyObject,
    attributes: Option<PyObject>,
}

#[pymethods]
impl PyTensor {
    /// Construct a Tensor from a dict of component numpy arrays.
    ///
    /// Args:
    ///     components: dict[str, np.ndarray] — component name to array mapping.
    ///     shape: Tensor shape. Inferred from components["data"] for dense tensors.
    ///     dtype: Storage dtype string. Inferred from primary component if not provided.
    ///     format: Layout format ("dense", "sparse_csr", etc.). Default: "dense".
    ///     type: Logical type (e.g., "f8_e4m3fn"). None when same as dtype.
    ///     attributes: Optional dict of per-object attributes.
    #[new]
    #[pyo3(signature = (components, *, shape=None, dtype=None, format="dense", r#type=None, attributes=None))]
    fn new(
        py: Python<'_>,
        components: &Bound<'_, PyDict>,
        shape: Option<Vec<i64>>,
        dtype: Option<&str>,
        format: &str,
        r#type: Option<&str>,
        attributes: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        // Determine primary component name
        let primary_comp_name: String = match format {
            "dense" => "data".to_string(),
            "sparse_csr" | "sparse_coo" => "values".to_string(),
            _ => {
                // Use first component
                components
                    .keys()
                    .get_item(0)
                    .ok()
                    .and_then(|k| k.extract::<String>().ok())
                    .unwrap_or_else(|| "data".to_string())
            }
        };

        // Infer shape from data component for dense
        let shape = match shape {
            Some(s) => s,
            None => {
                if format == "dense" {
                    let data_arr = components.get_item("data")?.ok_or_else(|| {
                        PyValueError::new_err(
                            "Dense tensor requires 'data' component or explicit shape",
                        )
                    })?;
                    data_arr.getattr("shape")?.extract::<Vec<i64>>()?
                } else {
                    return Err(PyValueError::new_err(
                        "Non-dense tensors require explicit shape",
                    ));
                }
            }
        };

        // Infer dtype from primary component
        let dtype_str = match dtype {
            Some(d) => d.to_string(),
            None => {
                let arr = components.get_item(&*primary_comp_name)?.ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Cannot infer dtype: missing '{}' component",
                        primary_comp_name
                    ))
                })?;
                arr.getattr("dtype")?.call_method0("__str__")?.extract()?
            }
        };

        Ok(Self {
            shape,
            dtype: dtype_str,
            type_: r#type.map(|s| s.to_string()),
            format_str: format.to_string(),
            components: components.into_pyobject(py)?.to_owned().into_any().unbind(),
            attributes: attributes
                .map(|a| {
                    a.into_pyobject(py)
                        .map(|o| o.to_owned().into_any().unbind())
                })
                .transpose()?,
        })
    }

    /// Tensor shape as a list of ints.
    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.shape.clone()
    }

    /// Storage dtype string (e.g., "float32").
    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }

    /// Logical type (e.g., "f8_e4m3fn"), or None when same as dtype.
    #[getter]
    fn r#type(&self) -> Option<&str> {
        self.type_.as_deref()
    }

    /// Layout format string ("dense", "sparse_csr", etc.).
    #[getter]
    fn format(&self) -> &str {
        &self.format_str
    }

    /// Component arrays as dict[str, np.ndarray].
    #[getter]
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        Ok(self.components.bind(py).downcast::<PyDict>()?.clone())
    }

    /// Per-object attributes dict, or None.
    #[getter]
    fn attributes<'py>(&self, py: Python<'py>) -> Option<PyObject> {
        self.attributes.as_ref().map(|a| a.clone_ref(py))
    }

    /// Convert to a NumPy array (dense tensors only).
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        if self.format_str != "dense" {
            return Err(PyRuntimeError::new_err(format!(
                "numpy() only supported for dense tensors, got '{}'",
                self.format_str
            )));
        }
        let components = self.components.bind(py).downcast::<PyDict>()?;
        let data = components
            .get_item("data")?
            .ok_or_else(|| PyRuntimeError::new_err("Dense tensor missing 'data' component"))?;
        Ok(data.unbind())
    }

    /// Convert to a torch.Tensor (dense tensors only).
    #[pyo3(signature = (*, device="cpu"))]
    fn torch<'py>(&self, py: Python<'py>, device: &str) -> PyResult<PyObject> {
        if self.format_str != "dense" {
            return Err(PyRuntimeError::new_err(format!(
                "torch() only supported for dense tensors, got '{}'",
                self.format_str
            )));
        }
        let torch = py.import("torch")?;
        let np_arr = self.numpy(py)?;
        let tensor = torch.call_method1("from_numpy", (np_arr,))?;
        if device != "cpu" {
            let moved = tensor.call_method1("to", (device,))?;
            Ok(moved.unbind())
        } else {
            Ok(tensor.unbind())
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<ztensor.Tensor shape={:?} dtype='{}' format='{}'>",
            self.shape, self.dtype, self.format_str
        )
    }
}

// =========================================================================
// NumPy array helpers
// =========================================================================

/// Intermediate result for tensor data that doesn't borrow from the reader.
enum ReadResult {
    /// Zero-copy: raw pointer and length into the reader's backing store.
    ZeroCopy { ptr: *const u8, len: usize },
    /// Owned copy of the data.
    Owned(Vec<u8>),
}

/// Convert raw tensor bytes to a NumPy array (always copies).
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

// =========================================================================
// Torch helpers
// =========================================================================

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
        DType::U16 | DType::U32 | DType::U64 => None,
    }
}

/// Creates a torch tensor directly from raw byte data.
fn create_torch_tensor<'py>(
    py: Python<'py>,
    torch: &Bound<'py, PyModule>,
    dtype: DType,
    shape: &[u64],
    src: &[u8],
) -> PyResult<PyObject> {
    let torch_attr = dtype_to_torch_attr(dtype).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "Unsupported dtype for torch: {:?}. Use read_numpy() instead.",
            dtype
        ))
    })?;
    let torch_dtype = torch.getattr(torch_attr)?;

    let shape_list: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", &torch_dtype)?;
    let tensor = torch.call_method("empty", (&shape_list,), Some(&kwargs))?;

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

/// Maps a torch dtype Python object to a zt DType.
fn torch_dtype_to_zt(
    torch: &Bound<'_, PyModule>,
    dtype: &Bound<'_, pyo3::types::PyAny>,
) -> PyResult<DType> {
    let pairs: &[(&str, DType)] = &[
        ("float64", DType::F64),
        ("float32", DType::F32),
        ("float16", DType::F16),
        ("bfloat16", DType::BF16),
        ("int64", DType::I64),
        ("int32", DType::I32),
        ("int16", DType::I16),
        ("int8", DType::I8),
        ("uint8", DType::U8),
        ("bool", DType::Bool),
    ];
    for (name, zt_dtype) in pairs {
        if dtype.is(&torch.getattr(*name)?) {
            return Ok(*zt_dtype);
        }
    }
    Err(PyValueError::new_err(format!(
        "Unsupported torch dtype: {}",
        dtype
    )))
}

// =========================================================================
// Reader internal helpers
// =========================================================================

/// Get the primary component dtype for a tensor object.
fn obj_primary_dtype(obj: &crate::models::Object) -> (&str, Option<&str>) {
    let comp = match &obj.format {
        Format::Dense => obj.components.get("data"),
        Format::SparseCsr | Format::SparseCoo => obj.components.get("values"),
        _ => obj.components.values().next(),
    };
    let dtype_str = comp
        .map(|c| dtype_to_numpy_str(c.dtype))
        .unwrap_or("unknown");
    let type_str = comp.and_then(|c| c.r#type.as_deref());
    (dtype_str, type_str)
}

/// Build TensorMetadata for a single tensor.
fn build_metadata(reader: &dyn TensorReader, name: &str) -> PyResult<PyTensorMeta> {
    let obj = reader
        .get(name)
        .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;
    let (dtype_str, type_str) = obj_primary_dtype(obj);
    Ok(PyTensorMeta {
        name: name.to_string(),
        dtype: dtype_str.to_string(),
        r#type: type_str.map(|s| s.to_string()),
        shape: obj.shape.clone(),
        format: obj.format.as_str().to_string(),
    })
}

/// Read a single tensor as a numpy array (dense only). Supports zero-copy.
fn read_numpy_single_impl<'py>(
    slf: &Bound<'py, PyReader>,
    name: &str,
    copy: bool,
) -> PyResult<PyObject> {
    let py = slf.py();

    let (result, dtype, shape) = {
        let this = slf.borrow();
        let obj = this
            .inner
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;

        if obj.format != Format::Dense {
            return Err(PyTypeError::new_err(format!(
                "read_numpy() requires dense tensor, '{}' has format '{}'",
                name,
                obj.format.as_str()
            )));
        }

        let component = obj.components.get("data").ok_or_else(|| {
            PyRuntimeError::new_err(format!("Missing 'data' component for '{}'", name))
        })?;
        let dtype = component.dtype;
        let shape = obj.shape.clone();

        let data = this.inner.read_data(name).map_err(zt_err)?;
        let result = if copy {
            ReadResult::Owned(data.into_owned())
        } else {
            match &data {
                TensorData::Borrowed(slice) => ReadResult::ZeroCopy {
                    ptr: slice.as_ptr(),
                    len: slice.len(),
                },
                TensorData::Owned(vec) => ReadResult::Owned(vec.clone()),
            }
        };

        (result, dtype, shape)
    };

    match result {
        ReadResult::ZeroCopy { ptr, len } => {
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            bytes_to_numpy_borrowed(py, slice, dtype, &shape, slf.clone().into_any())
        }
        ReadResult::Owned(data) => bytes_to_numpy(py, TensorData::Owned(data), dtype, &shape),
    }
}

/// Read multiple tensors as numpy arrays. Sorts by offset for sequential I/O.
fn read_numpy_many_impl<'py>(
    slf: &Bound<'py, PyReader>,
    names: &[String],
    copy: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let py = slf.py();
    let dict = PyDict::new(py);

    // Collect metadata and data pointers in a single borrow
    let mut tensor_infos: Vec<(String, DType, Vec<u64>, u64, ReadResult)> = {
        let this = slf.borrow();
        let mut infos = Vec::new();
        for name in names {
            let obj = this
                .inner
                .get(name)
                .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;

            if obj.format != Format::Dense {
                return Err(PyTypeError::new_err(format!(
                    "read_numpy() requires dense tensor, '{}' has format '{}'",
                    name,
                    obj.format.as_str()
                )));
            }

            let component = obj.components.get("data").ok_or_else(|| {
                PyRuntimeError::new_err(format!("Missing 'data' component for '{}'", name))
            })?;
            let dtype = component.dtype;
            let offset = component.offset;
            let shape = obj.shape.clone();
            let data = this.inner.read_data(name).map_err(zt_err)?;
            let result = if copy {
                ReadResult::Owned(data.into_owned())
            } else {
                match &data {
                    TensorData::Borrowed(slice) => ReadResult::ZeroCopy {
                        ptr: slice.as_ptr(),
                        len: slice.len(),
                    },
                    TensorData::Owned(vec) => ReadResult::Owned(vec.clone()),
                }
            };
            infos.push((name.clone(), dtype, shape, offset, result));
        }
        infos
    };

    // Sort by file offset for sequential page fault access
    tensor_infos.sort_by_key(|info| info.3);

    if copy {
        for (name, dtype, shape, _, data) in &tensor_infos {
            let src = match data {
                ReadResult::ZeroCopy { ptr, len } => unsafe {
                    std::slice::from_raw_parts(*ptr, *len)
                },
                ReadResult::Owned(vec) => vec.as_slice(),
            };
            let arr = bytes_to_numpy(py, TensorData::Borrowed(src), *dtype, shape)?;
            dict.set_item(name, arr)?;
        }
    } else {
        let np = py.import("numpy")?;
        let mut dtype_cache: Vec<(DType, PyObject)> = Vec::new();

        for (name, dtype, shape, _, data) in &tensor_infos {
            let arr = match data {
                ReadResult::ZeroCopy { ptr, len } => {
                    let slice = unsafe { std::slice::from_raw_parts(*ptr, *len) };
                    if *dtype == DType::BF16 {
                        bytes_to_numpy_borrowed(py, slice, *dtype, shape, slf.clone().into_any())?
                    } else {
                        let view = ArrayView1::from(slice);
                        let np_array = unsafe {
                            PyArray1::<u8>::borrow_from_array(&view, slf.clone().into_any())
                        };
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
            dict.set_item(name, arr)?;
        }
    }

    Ok(dict)
}

/// Read a single tensor as a PyTensor (all components, any format).
fn read_tensor_impl<'py>(slf: &Bound<'py, PyReader>, name: &str, copy: bool) -> PyResult<PyObject> {
    let py = slf.py();

    // Collect metadata and read all component data
    struct CompInfo {
        comp_name: String,
        dtype: DType,
        is_data_dense: bool,
        data: ReadResult,
    }

    let (shape, format_str, dtype_str, type_str, comp_infos, attributes) = {
        let this = slf.borrow();
        let obj = this
            .inner
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;

        let format_str = obj.format.as_str().to_string();
        let shape = obj.shape.clone();
        let (dtype_str, type_str) = obj_primary_dtype(obj);
        let dtype_str = dtype_str.to_string();
        let type_str = type_str.map(|s| s.to_string());
        let is_dense = obj.format == Format::Dense;

        let mut comp_infos = Vec::new();
        for (comp_name, comp) in &obj.components {
            let data = this
                .inner
                .read_component_data(name, comp_name)
                .map_err(zt_err)?;
            let result = if copy {
                ReadResult::Owned(data.into_owned())
            } else {
                match &data {
                    TensorData::Borrowed(slice) => ReadResult::ZeroCopy {
                        ptr: slice.as_ptr(),
                        len: slice.len(),
                    },
                    TensorData::Owned(vec) => ReadResult::Owned(vec.clone()),
                }
            };
            comp_infos.push(CompInfo {
                comp_name: comp_name.clone(),
                dtype: comp.dtype,
                is_data_dense: comp_name == "data" && is_dense,
                data: result,
            });
        }

        let attributes = obj.attributes.clone();

        (
            shape, format_str, dtype_str, type_str, comp_infos, attributes,
        )
    };

    // Build components dict
    let components = PyDict::new(py);
    for ci in &comp_infos {
        // For "data" component of dense tensors, reshape to tensor shape.
        // All other components stay flat 1D.
        let comp_shape: &[u64] = if ci.is_data_dense { &shape } else { &[] };

        let arr = match &ci.data {
            ReadResult::ZeroCopy { ptr, len } => {
                let slice = unsafe { std::slice::from_raw_parts(*ptr, *len) };
                bytes_to_numpy_borrowed(py, slice, ci.dtype, comp_shape, slf.clone().into_any())?
            }
            ReadResult::Owned(data) => bytes_to_numpy(
                py,
                TensorData::Borrowed(data.as_slice()),
                ci.dtype,
                comp_shape,
            )?,
        };
        components.set_item(&ci.comp_name, arr)?;
    }

    // Build attributes
    let py_attributes = match &attributes {
        Some(attrs) => {
            let dict = PyDict::new(py);
            for (k, v) in attrs {
                let py_val = cbor_value_to_python(py, v)?;
                dict.set_item(k, py_val)?;
            }
            Some(dict.unbind())
        }
        None => None,
    };

    let tensor = PyTensor {
        shape: shape.iter().map(|&s| s as i64).collect(),
        dtype: dtype_str,
        type_: type_str,
        format_str,
        components: components.unbind().into(),
        attributes: py_attributes.map(|d| -> Py<PyAny> { d.into() }),
    };

    Ok(tensor.into_pyobject(py)?.into_any().unbind())
}

// =========================================================================
// PyReader — format-agnostic tensor reader
// =========================================================================

/// Reader for tensor files (zTensor, SafeTensors, PyTorch, GGUF, NPZ, ONNX, HDF5).
///
/// Automatically detects the file format based on file extension.
/// Supports dict-like access: `reader["tensor_name"]` returns a `ztensor.Tensor`.
#[pyclass(name = "Reader")]
pub struct PyReader {
    inner: Box<dyn TensorReader + Send>,
    file_format: String,
}

// Safety: PyReader holds Box<dyn TensorReader + Send>.
// PyO3 requires Sync for pyclass. We guarantee single-threaded access
// through the GIL — Python holds the GIL when calling any method.
unsafe impl Sync for PyReader {}

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

    /// Returns metadata for a tensor by name (str → TensorMetadata, list → list[TensorMetadata]).
    fn metadata<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, pyo3::types::PyAny>,
    ) -> PyResult<PyObject> {
        if let Ok(s) = name.extract::<String>() {
            let meta = build_metadata(self.inner.as_ref(), &s)?;
            Ok(Py::new(py, meta)?.into_any())
        } else if let Ok(names) = name.extract::<Vec<String>>() {
            let list = PyList::empty(py);
            for n in &names {
                let meta = build_metadata(self.inner.as_ref(), n)?;
                list.append(Py::new(py, meta)?)?;
            }
            Ok(list.into_pyobject(py)?.into_any().unbind())
        } else {
            Err(PyTypeError::new_err("name must be str or list[str]"))
        }
    }

    /// Reads tensor(s) as ztensor.Tensor objects.
    ///
    /// Args:
    ///     name: str → single Tensor, list[str] → dict[str, Tensor]
    ///     copy: If False (default), zero-copy mmap arrays when possible.
    #[pyo3(signature = (name, *, copy=false))]
    fn read<'py>(
        slf: &Bound<'py, PyReader>,
        name: &Bound<'py, pyo3::types::PyAny>,
        copy: bool,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        if let Ok(s) = name.extract::<String>() {
            read_tensor_impl(slf, &s, copy)
        } else if let Ok(names) = name.extract::<Vec<String>>() {
            // Sort by minimum component offset for sequential mmap I/O
            let mut sorted_names: Vec<(String, u64)> = {
                let this = slf.borrow();
                names
                    .into_iter()
                    .map(|n| {
                        let min_offset = this
                            .inner
                            .get(&n)
                            .map(|obj| obj.components.values().map(|c| c.offset).min().unwrap_or(0))
                            .unwrap_or(0);
                        (n, min_offset)
                    })
                    .collect()
            };
            sorted_names.sort_by_key(|&(_, offset)| offset);

            let dict = PyDict::new(py);
            for (n, _) in &sorted_names {
                let tensor = read_tensor_impl(slf, n, copy)?;
                dict.set_item(n, tensor)?;
            }
            Ok(dict.into_pyobject(py)?.into_any().unbind())
        } else {
            Err(PyTypeError::new_err("name must be str or list[str]"))
        }
    }

    /// Reads dense tensor(s) as NumPy arrays.
    ///
    /// Args:
    ///     name: str → np.ndarray, list[str] → dict[str, np.ndarray]
    ///     copy: If False (default), zero-copy mmap arrays when possible.
    #[pyo3(signature = (name, *, copy=false))]
    fn read_numpy<'py>(
        slf: &Bound<'py, PyReader>,
        name: &Bound<'py, pyo3::types::PyAny>,
        copy: bool,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        if let Ok(s) = name.extract::<String>() {
            read_numpy_single_impl(slf, &s, copy)
        } else if let Ok(names) = name.extract::<Vec<String>>() {
            let dict = read_numpy_many_impl(slf, &names, copy)?;
            Ok(dict.into_pyobject(py)?.into_any().unbind())
        } else {
            Err(PyTypeError::new_err("name must be str or list[str]"))
        }
    }

    /// Reads dense tensor(s) as torch.Tensor.
    ///
    /// Args:
    ///     name: str → torch.Tensor, list[str] → dict[str, torch.Tensor]
    ///     copy: If False (default), zero-copy via numpy mmap + torch.from_numpy.
    ///     device: Target device ("cpu", "cuda:0", etc.).
    #[pyo3(signature = (name, *, copy=false, device="cpu"))]
    fn read_torch<'py>(
        slf: &Bound<'py, PyReader>,
        name: &Bound<'py, pyo3::types::PyAny>,
        copy: bool,
        device: &str,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        let torch = py.import("torch")?;

        if let Ok(s) = name.extract::<String>() {
            let tensor = read_torch_single_impl(slf, &s, copy, device, &torch)?;
            Ok(tensor)
        } else if let Ok(names) = name.extract::<Vec<String>>() {
            let dict = read_torch_many_impl(slf, &names, copy, device, &torch)?;
            Ok(dict.into_pyobject(py)?.into_any().unbind())
        } else {
            Err(PyTypeError::new_err("name must be str or list[str]"))
        }
    }

    /// Number of tensors in the file.
    fn __len__(&self) -> usize {
        self.inner.tensors().len()
    }

    /// Check if a tensor name exists.
    fn __contains__(&self, name: &str) -> bool {
        self.inner.get(name).is_some()
    }

    /// Dict-like access: reader["name"] returns a ztensor.Tensor (zero-copy).
    fn __getitem__<'py>(slf: &Bound<'py, PyReader>, name: &str) -> PyResult<PyObject> {
        read_tensor_impl(slf, name, false)
    }

    /// The detected file format.
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

/// Read a single dense tensor as a torch.Tensor.
fn read_torch_single_impl<'py>(
    slf: &Bound<'py, PyReader>,
    name: &str,
    copy: bool,
    device: &str,
    torch: &Bound<'py, PyModule>,
) -> PyResult<PyObject> {
    let py = slf.py();

    if copy {
        // Eager copy: torch.empty() + memcpy
        let (dtype, shape, data) = {
            let this = slf.borrow();
            let obj = this
                .inner
                .get(name)
                .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;
            if obj.format != Format::Dense {
                return Err(PyTypeError::new_err(format!(
                    "read_torch() requires dense tensor, '{}' has format '{}'",
                    name,
                    obj.format.as_str()
                )));
            }
            let comp = obj.components.get("data").ok_or_else(|| {
                PyRuntimeError::new_err(format!("Missing 'data' component for '{}'", name))
            })?;
            let dtype = comp.dtype;
            let shape = obj.shape.clone();
            let data = this.inner.read_data(name).map_err(zt_err)?;
            (dtype, shape, data.into_owned())
        };

        let tensor = create_torch_tensor(py, torch, dtype, &shape, &data)?;
        if device != "cpu" {
            let moved = tensor.call_method1(py, "to", (device,))?;
            Ok(moved)
        } else {
            Ok(tensor)
        }
    } else {
        // Zero-copy: numpy mmap → torch.from_numpy
        let np_arr = read_numpy_single_impl(slf, name, false)?;
        let from_numpy = torch.getattr("from_numpy")?;
        let tensor = from_numpy.call1((np_arr,))?;
        if device != "cpu" {
            let moved = tensor.call_method1("to", (device,))?;
            Ok(moved.unbind())
        } else {
            Ok(tensor.unbind())
        }
    }
}

/// Read multiple dense tensors as torch.Tensors. Sorts by offset for sequential I/O.
fn read_torch_many_impl<'py>(
    slf: &Bound<'py, PyReader>,
    names: &[String],
    copy: bool,
    device: &str,
    torch: &Bound<'py, PyModule>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = slf.py();
    let dict = PyDict::new(py);

    if copy {
        // Eager copy: collect data, sort by offset, create torch tensors
        let mut tensor_infos: Vec<(String, DType, Vec<u64>, u64, Vec<u8>)> = {
            let this = slf.borrow();
            let mut infos = Vec::new();
            for name in names {
                let obj = this
                    .inner
                    .get(name)
                    .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;
                if obj.format != Format::Dense {
                    return Err(PyTypeError::new_err(format!(
                        "read_torch() requires dense tensor, '{}' has format '{}'",
                        name,
                        obj.format.as_str()
                    )));
                }
                let comp = obj.components.get("data").ok_or_else(|| {
                    PyRuntimeError::new_err(format!("Missing 'data' component for '{}'", name))
                })?;
                let dtype = comp.dtype;
                let offset = comp.offset;
                let shape = obj.shape.clone();
                let data = this.inner.read_data(name).map_err(zt_err)?.into_owned();
                infos.push((name.clone(), dtype, shape, offset, data));
            }
            infos
        };

        tensor_infos.sort_by_key(|info| info.3);

        for (name, dtype, shape, _, data) in &tensor_infos {
            let tensor = create_torch_tensor(py, torch, *dtype, shape, data)?;
            dict.set_item(name, tensor)?;
        }
    } else {
        // Zero-copy: numpy mmap → torch.from_numpy
        let np_dict = read_numpy_many_impl(slf, names, false)?;
        let from_numpy = torch.getattr("from_numpy")?;
        for item in np_dict.iter() {
            let (key, arr) = item;
            let tensor = from_numpy.call1((arr,))?;
            dict.set_item(key, tensor)?;
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

// =========================================================================
// PyWriter — zTensor file writer
// =========================================================================

/// Writer for creating zTensor files.
///
/// Usage:
///     writer = ztensor.Writer("output.zt")
///     writer.write_numpy("weights", numpy_array)
///     writer.finish()
///
/// Or as a context manager:
///     with ztensor.Writer("output.zt") as w:
///         w.write_numpy("weights", numpy_array)
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

    /// Open an existing .zt file for appending new tensors.
    #[staticmethod]
    fn append(path: &str) -> PyResult<Self> {
        let inner = Writer::append(path).map_err(zt_err)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Write a ztensor.Tensor to the file.
    ///
    /// Args:
    ///     name: Tensor name.
    ///     data: A ztensor.Tensor object.
    ///     compress: False/0 for raw, True for default zstd, int for specific level.
    ///     checksum: "none", "crc32c", or "sha256".
    #[pyo3(signature = (name, data, *, compress=None, checksum="none"))]
    fn write(
        &mut self,
        py: Python<'_>,
        name: &str,
        data: &PyTensor,
        compress: Option<&Bound<'_, pyo3::types::PyAny>>,
        checksum: &str,
    ) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Writer is already finished"))?;

        let compression = match compress {
            Some(obj) => parse_compression(obj)?,
            None => Compression::Raw,
        };
        let checksum_alg = parse_checksum_str(checksum)?;

        let np = py.import("numpy")?;
        let format = Format::from_str(&data.format_str);
        let shape: Vec<u64> = data.shape.iter().map(|&s| s as u64).collect();

        // Extract component data
        let components_dict = data.components.bind(py).downcast::<PyDict>()?;
        let mut component_data: Vec<(String, DType, Option<String>, Vec<u8>)> = Vec::new();

        for item in components_dict.iter() {
            let (comp_name_obj, arr_obj) = item;
            let comp_name: String = comp_name_obj.extract()?;

            let arr = np.call_method1("ascontiguousarray", (&arr_obj,))?;
            let dtype_str: String = arr.getattr("dtype")?.call_method0("__str__")?.extract()?;
            let dtype = numpy_str_to_dtype(&dtype_str)?;
            let nbytes: usize = arr.getattr("nbytes")?.extract()?;

            let iface = arr.getattr("__array_interface__")?;
            let data_tuple = iface.get_item("data")?;
            let ptr: usize = data_tuple.get_item(0)?.extract()?;
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, nbytes) };

            // Set logical type on the primary component
            let logical_type =
                if (comp_name == "data" && data.format_str == "dense") || comp_name == "values" {
                    data.type_.clone()
                } else {
                    None
                };

            component_data.push((comp_name, dtype, logical_type, bytes.to_vec()));
        }

        // Build refs for add_object
        let comp_refs: Vec<(&str, DType, Option<&str>, &[u8])> = component_data
            .iter()
            .map(|(name, dtype, lt, bytes)| {
                (name.as_str(), *dtype, lt.as_deref(), bytes.as_slice())
            })
            .collect();

        // Extract attributes
        let attributes = match &data.attributes {
            Some(attrs_obj) => {
                let attrs_dict = attrs_obj.bind(py).downcast::<PyDict>()?;
                let mut map = BTreeMap::new();
                for (k, v) in attrs_dict.iter() {
                    let key: String = k.extract()?;
                    let val = python_to_cbor_value(&v)?;
                    map.insert(key, val);
                }
                Some(map)
            }
            None => None,
        };

        writer
            .add_object(
                name,
                shape,
                format,
                &comp_refs,
                attributes,
                compression,
                checksum_alg,
            )
            .map_err(zt_err)
    }

    /// Write a dense tensor from a NumPy array.
    ///
    /// Args:
    ///     name: Tensor name.
    ///     data: NumPy array (must be contiguous).
    ///     compress: False/0 for raw, True for default zstd, int for specific level.
    ///     checksum: "none", "crc32c", or "sha256".
    #[pyo3(signature = (name, data, *, compress=None, checksum="none"))]
    fn write_numpy(
        &mut self,
        py: Python<'_>,
        name: &str,
        data: &Bound<'_, pyo3::types::PyAny>,
        compress: Option<&Bound<'_, pyo3::types::PyAny>>,
        checksum: &str,
    ) -> PyResult<()> {
        self.write_numpy_impl(py, name, data, compress, checksum)
    }

    /// Write a dense tensor from a torch.Tensor.
    ///
    /// Args:
    ///     name: Tensor name.
    ///     data: torch.Tensor (will be moved to CPU and made contiguous if needed).
    ///     compress: False/0 for raw, True for default zstd, int for specific level.
    ///     checksum: "none", "crc32c", or "sha256".
    #[pyo3(signature = (name, data, *, compress=None, checksum="none"))]
    fn write_torch(
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

        let compression = match compress {
            Some(obj) => parse_compression(obj)?,
            None => Compression::Raw,
        };
        let checksum_alg = parse_checksum_str(checksum)?;

        let torch = py.import("torch")?;

        // Move to CPU and make contiguous
        let tensor = {
            let device_type: String = data.getattr("device")?.getattr("type")?.extract()?;
            let t = if device_type != "cpu" {
                data.call_method0("cpu")?
            } else {
                data.clone()
            };
            if !t.call_method0("is_contiguous")?.extract::<bool>()? {
                t.call_method0("contiguous")?
            } else {
                t
            }
        };

        // Get zt dtype from torch dtype
        let torch_dtype_obj = tensor.getattr("dtype")?;
        let dtype = torch_dtype_to_zt(&torch, &torch_dtype_obj)?;

        // Get shape
        let shape: Vec<u64> = tensor
            .getattr("shape")?
            .extract::<Vec<i64>>()?
            .iter()
            .map(|&s| s as u64)
            .collect();

        // Get raw bytes via data_ptr
        let numel = tensor.call_method0("numel")?.extract::<usize>()?;
        let element_size = tensor.call_method0("element_size")?.extract::<usize>()?;
        let nbytes = numel * element_size;
        let data_ptr = tensor.call_method0("data_ptr")?.extract::<usize>()?;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, nbytes) };

        writer
            .add_bytes(name, shape, dtype, compression, bytes, checksum_alg)
            .map_err(zt_err)
    }

    /// Add a dense tensor from a NumPy array (alias for write_numpy).
    #[pyo3(signature = (name, data, compress=None, checksum="none"))]
    fn add(
        &mut self,
        py: Python<'_>,
        name: &str,
        data: &Bound<'_, pyo3::types::PyAny>,
        compress: Option<&Bound<'_, pyo3::types::PyAny>>,
        checksum: &str,
    ) -> PyResult<()> {
        self.write_numpy_impl(py, name, data, compress, checksum)
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

impl PyWriter {
    /// Internal shared implementation for write_numpy and add.
    fn write_numpy_impl(
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

        let compression = match compress {
            Some(obj) => parse_compression(obj)?,
            None => Compression::Raw,
        };
        let checksum_alg = parse_checksum_str(checksum)?;

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

        let iface = arr.getattr("__array_interface__")?;
        let data_tuple = iface.get_item("data")?;
        let ptr: usize = data_tuple.get_item(0)?.extract()?;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, nbytes) };

        writer
            .add_bytes(name, shape, dtype, compression, bytes, checksum_alg)
            .map_err(zt_err)
    }
}

// =========================================================================
// Top-level functions
// =========================================================================

/// Open a tensor file (auto-detects format by extension).
#[pyfunction]
fn open(path: &str) -> PyResult<PyReader> {
    PyReader::new(path)
}

/// Save a dict of numpy arrays to a ztensor file in a single FFI call.
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

        let iface = arr.getattr("__array_interface__")?;
        let data_tuple = iface.get_item("data")?;
        let ptr: usize = data_tuple.get_item(0)?.extract()?;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, nbytes) };

        writer
            .add_bytes(&name, shape, dtype, compress, bytes, Checksum::None)
            .map_err(zt_err)?;
    }

    writer.finish().map_err(zt_err)?;
    Ok(())
}

/// Remove tensors by name from a .zt file, writing the result to a new file.
#[pyfunction]
fn remove_tensors(input: &str, output: &str, names: Vec<String>) -> PyResult<()> {
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    crate::remove_tensors(input, output, &name_refs).map_err(zt_err)
}

/// Replace the data of a dense tensor in-place within an existing .zt file.
#[pyfunction]
fn replace_tensor(
    py: Python<'_>,
    path: &str,
    name: &str,
    data: &Bound<'_, pyo3::types::PyAny>,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (data,))?;

    let nbytes: usize = arr.getattr("nbytes")?.extract()?;

    let iface = arr.getattr("__array_interface__")?;
    let data_tuple = iface.get_item("data")?;
    let ptr: usize = data_tuple.get_item(0)?.extract()?;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr as *const u8, nbytes) };

    crate::replace_tensor(path, name, bytes).map_err(zt_err)
}

/// The native zTensor Python module.
#[pymodule]
fn _ztensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReader>()?;
    m.add_class::<PyWriter>()?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyTensorMeta>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(save_file, m)?)?;
    m.add_function(wrap_pyfunction!(remove_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(replace_tensor, m)?)?;
    Ok(())
}
