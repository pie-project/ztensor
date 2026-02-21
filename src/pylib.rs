//! PyO3 Python bindings for zTensor.
//!
//! Provides low-level `_Reader` and `_Writer` primitives. Python-side
//! wrapper classes (`Reader`, `Writer`, `Tensor`, `TensorMetadata`) in
//! `python/ztensor/` handle orchestration (dtype conversion, batch reads,
//! offset sorting, framework integration).

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
// ReadResult — intermediate result for raw buffer reads
// =========================================================================

/// Intermediate result for tensor data that doesn't borrow from the reader.
enum ReadResult {
    /// Zero-copy: raw pointer and length into the reader's backing store.
    ZeroCopy { ptr: *const u8, len: usize },
    /// Owned copy of the data.
    Owned(Vec<u8>),
}

// =========================================================================
// Torch helpers
// =========================================================================

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
// _Reader — low-level format-agnostic tensor reader
// =========================================================================

/// Low-level reader for tensor files. Python `Reader` wraps this.
#[pyclass(name = "_Reader")]
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

    /// Number of tensors in the file.
    fn __len__(&self) -> usize {
        self.inner.tensors().len()
    }

    /// Check if a tensor name exists.
    fn __contains__(&self, name: &str) -> bool {
        self.inner.get(name).is_some()
    }

    /// The detected file format.
    #[getter]
    fn format(&self) -> &str {
        &self.file_format
    }

    /// Returns full metadata for a tensor as a dict.
    ///
    /// Returns:
    ///     dict with keys: shape, format, components, attributes.
    ///     components is a dict of {name: {dtype, type, offset, length, encoding}}.
    fn _get_metadata<'py>(&self, py: Python<'py>, name: &str) -> PyResult<PyObject> {
        let obj = self
            .inner
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;

        let result = PyDict::new(py);

        // shape
        let shape: Vec<i64> = obj.shape.iter().map(|&s| s as i64).collect();
        result.set_item("shape", shape)?;

        // format
        result.set_item("format", obj.format.as_str())?;

        // components
        let comps_dict = PyDict::new(py);
        for (comp_name, comp) in &obj.components {
            let comp_dict = PyDict::new(py);
            comp_dict.set_item("dtype", dtype_to_numpy_str(comp.dtype))?;
            comp_dict.set_item("type", comp.r#type.as_deref().map(|s| s.to_string()))?;
            comp_dict.set_item("offset", comp.offset)?;
            comp_dict.set_item("length", comp.length)?;
            comp_dict.set_item(
                "encoding",
                match comp.encoding {
                    crate::models::Encoding::Raw => "raw",
                    crate::models::Encoding::Zstd => "zstd",
                },
            )?;
            comps_dict.set_item(comp_name.as_str(), &comp_dict)?;
        }
        result.set_item("components", &comps_dict)?;

        // attributes
        match &obj.attributes {
            Some(attrs) => {
                let attrs_dict = PyDict::new(py);
                for (k, v) in attrs {
                    let py_val = cbor_value_to_python(py, v)?;
                    attrs_dict.set_item(k, py_val)?;
                }
                result.set_item("attributes", &attrs_dict)?;
            }
            None => {
                result.set_item("attributes", py.None())?;
            }
        }

        Ok(result.into_pyobject(py)?.into_any().unbind())
    }

    /// Returns raw bytes and dtype string for any component of a tensor.
    ///
    /// Returns:
    ///     (buffer, dtype): A tuple of (numpy uint8 array, dtype string).
    fn _read_component_raw<'py>(
        slf: &Bound<'py, PyReader>,
        name: &str,
        component: &str,
    ) -> PyResult<PyObject> {
        let py = slf.py();

        let (result, dtype) = {
            let this = slf.borrow();
            let obj = this
                .inner
                .get(name)
                .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;
            let comp = obj.components.get(component).ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "Component '{}' not found in tensor '{}'",
                    component, name
                ))
            })?;
            let dtype = comp.dtype;
            let data = this
                .inner
                .read_component_data(name, component)
                .map_err(zt_err)?;
            let result = match &data {
                TensorData::Borrowed(slice) => ReadResult::ZeroCopy {
                    ptr: slice.as_ptr(),
                    len: slice.len(),
                },
                TensorData::Owned(vec) => ReadResult::Owned(vec.clone()),
            };
            (result, dtype)
        };

        let buffer = match result {
            ReadResult::ZeroCopy { ptr, len } => {
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                let view = ArrayView1::from(slice);
                unsafe { PyArray1::<u8>::borrow_from_array(&view, slf.clone().into_any()) }
                    .into_pyobject(py)?
                    .into_any()
            }
            ReadResult::Owned(data) => {
                let arr = unsafe { PyArray1::<u8>::new(py, [data.len()], false) };
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), arr.data(), data.len());
                }
                arr.into_pyobject(py)?.into_any()
            }
        };

        let dtype_str = dtype_to_numpy_str(dtype);
        Ok((buffer, dtype_str).into_pyobject(py)?.into_any().unbind())
    }

    /// Returns raw bytes, dtype string, and shape for a dense tensor.
    ///
    /// Returns:
    ///     (buffer, dtype, shape): A tuple of (numpy uint8 array, dtype string, shape list).
    ///     The buffer is zero-copy (mmap-backed) when possible.
    fn _read_raw<'py>(slf: &Bound<'py, PyReader>, name: &str) -> PyResult<PyObject> {
        let py = slf.py();

        let (result, dtype, shape) = {
            let this = slf.borrow();
            let obj = this
                .inner
                .get(name)
                .ok_or_else(|| PyKeyError::new_err(format!("Tensor '{}' not found", name)))?;
            if obj.format != Format::Dense {
                return Err(PyTypeError::new_err(format!(
                    "_read_raw() requires dense tensor, '{}' has format '{}'",
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
            let result = match &data {
                TensorData::Borrowed(slice) => ReadResult::ZeroCopy {
                    ptr: slice.as_ptr(),
                    len: slice.len(),
                },
                TensorData::Owned(vec) => ReadResult::Owned(vec.clone()),
            };
            (result, dtype, shape)
        };

        let buffer = match result {
            ReadResult::ZeroCopy { ptr, len } => {
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                let view = ArrayView1::from(slice);
                unsafe { PyArray1::<u8>::borrow_from_array(&view, slf.clone().into_any()) }
                    .into_pyobject(py)?
                    .into_any()
            }
            ReadResult::Owned(data) => {
                let arr = unsafe { PyArray1::<u8>::new(py, [data.len()], false) };
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), arr.data(), data.len());
                }
                arr.into_pyobject(py)?.into_any()
            }
        };

        let dtype_str = dtype_to_numpy_str(dtype);
        Ok((buffer, dtype_str, shape)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }
}

// =========================================================================
// _Writer — low-level zTensor file writer
// =========================================================================

/// Low-level writer for creating zTensor files. Python `Writer` wraps this.
#[pyclass(name = "_Writer")]
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

    /// Write a tensor object (duck-typed: reads .shape, .format, .type,
    /// .components, .attributes from any Python object).
    #[pyo3(signature = (name, data, *, compress=None, checksum="none"))]
    fn _write_object(
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

        // Read format, shape, type from the Python object
        let format_str: String = data.getattr("format")?.extract()?;
        let format = Format::from_str(&format_str);
        let shape: Vec<u64> = data
            .getattr("shape")?
            .extract::<Vec<i64>>()?
            .iter()
            .map(|&s| s as u64)
            .collect();
        let type_: Option<String> = data.getattr("type")?.extract()?;

        // Extract component data from .components dict
        let components_obj = data.getattr("components")?;
        let components_dict = components_obj.downcast::<PyDict>()?;
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
                if (comp_name == "data" && format_str == "dense") || comp_name == "values" {
                    type_.clone()
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
        let attrs_obj = data.getattr("attributes")?;
        let attributes = if attrs_obj.is_none() {
            None
        } else {
            let attrs_dict = attrs_obj.downcast::<PyDict>()?;
            let mut map = BTreeMap::new();
            for (k, v) in attrs_dict.iter() {
                let key: String = k.extract()?;
                let val = python_to_cbor_value(&v)?;
                map.insert(key, val);
            }
            Some(map)
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
    #[pyo3(signature = (name, data, *, compress=None, checksum="none"))]
    fn _write_numpy(
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

    /// Write a dense tensor from a torch.Tensor.
    #[pyo3(signature = (name, data, *, compress=None, checksum="none"))]
    fn _write_torch(
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

    /// Discard the writer without finishing (used on exception in __exit__).
    fn _discard(&mut self) {
        self.inner.take();
    }

    /// Finish the file (writes manifest and footer).
    fn _finish(&mut self) -> PyResult<u64> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Writer is already finished"))?;
        writer.finish().map_err(zt_err)
    }
}

// =========================================================================
// Top-level functions
// =========================================================================

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
    m.add_function(wrap_pyfunction!(save_file, m)?)?;
    m.add_function(wrap_pyfunction!(remove_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(replace_tensor, m)?)?;
    Ok(())
}
