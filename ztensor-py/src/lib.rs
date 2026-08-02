//! Python bindings: the universal loader surface, pythonically.
//!
//! `read()` returns `bytes` (a copy). Interpret them with numpy on the
//! Python side: `np.frombuffer(src.read(name), dtype=...)` — the bindings
//! never link against numpy.

use std::os::raw::c_int;

use pyo3::exceptions::{PyBufferError, PyIOError, PyKeyError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};


fn err(e: ztensor::Error) -> PyErr {
    match e {
        ztensor::Error::Io(io) => PyIOError::new_err(io.to_string()),
        ztensor::Error::NotFound(what) => PyKeyError::new_err(what),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// A tensor source: any supported format projected into the zTensor
/// object model.
#[pyclass(unsendable)]
struct Source {
    inner: Box<dyn ztensor::Source>,
    path: String,
}

#[pymethods]
impl Source {
    /// Sorted tensor names.
    fn keys(&self) -> Vec<String> {
        self.inner.manifest().objects.keys().cloned().collect()
    }

    fn __len__(&self) -> usize {
        self.inner.manifest().objects.len()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.manifest().objects.contains_key(name)
    }

    fn __repr__(&self) -> String {
        format!(
            "Source({:?}, {} objects)",
            self.path,
            self.inner.manifest().objects.len()
        )
    }

    /// Metadata for one object: shape, layout, dtype, logical type, parts,
    /// and decoded byte size.
    fn info<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
        let obj = self
            .inner
            .manifest()
            .objects
            .get(name)
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
        let d = PyDict::new(py);
        d.set_item("shape", obj.shape.clone())?;
        d.set_item("layout", obj.layout.as_str())?;
        let parts: Vec<String> = obj.parts.keys().cloned().collect();
        if let Some(part) = obj.parts.values().next() {
            d.set_item("dtype", part.dtype.as_str())?;
            d.set_item("type", part.ltype.clone())?;
        }
        let nbytes: u64 = obj.parts.values().map(|p| p.decoded_size()).sum();
        d.set_item("nbytes", nbytes)?;
        d.set_item("parts", parts)?;
        Ok(d)
    }

    /// Decoded little-endian bytes of a part (tier 1).
    #[pyo3(signature = (name, part = "data"))]
    fn read<'py>(&self, py: Python<'py>, name: &str, part: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.read(name, part).map_err(err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Zero-copy window onto a part's bytes (tier 2).
    ///
    /// Raises `ValueError` when the part cannot be viewed without a copy —
    /// an encoded part, a compressed archive entry, a foreign shard —
    /// rather than silently returning one.
    #[pyo3(signature = (name, part = "data"))]
    fn view(slf: Bound<'_, Self>, name: &str, part: &str) -> PyResult<TensorView> {
        let (ptr, len) = {
            let this = slf.borrow();
            let bytes = this.inner.view(name, part).map_err(err)?;
            (bytes.as_ptr(), bytes.len())
        };
        Ok(TensorView {
            owner: slf.unbind(),
            ptr,
            len,
        })
    }

    /// Capability report for a part, including the ladder tier.
    #[pyo3(signature = (name, part = "data"))]
    fn caps<'py>(&self, py: Python<'py>, name: &str, part: &str) -> PyResult<Bound<'py, PyDict>> {
        let caps = self.inner.caps(name, part).map_err(err)?;
        let d = PyDict::new(py);
        d.set_item("zero_copy", caps.zero_copy)?;
        d.set_item("alignment", caps.alignment)?;
        d.set_item("verifiable", caps.verifiable)?;
        d.set_item("page_exclusive", caps.page_exclusive)?;
        d.set_item("tier", caps.tier())?;
        Ok(d)
    }

    /// File-level attributes as a dict (strings and numbers where
    /// representable; nested values come back as Python lists/dicts).
    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner
            .manifest()
            .attributes
            .as_ref()
            .map(|v| value_to_py(py, v))
            .transpose()
    }
}

fn value_to_py<'py>(py: Python<'py>, v: &ztensor::cbor::Value) -> PyResult<Bound<'py, PyAny>> {
    use ztensor::cbor::Value as V;
    Ok(match v {
        V::Uint(n) => n.into_pyobject(py)?.into_any(),
        V::Nint(n) => (-1i128 - *n as i128).into_pyobject(py)?.into_any(),
        V::Float(x) => x.into_pyobject(py)?.into_any(),
        V::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        V::Null => py.None().into_bound(py),
        V::Text(s) => s.into_pyobject(py)?.into_any(),
        V::Bytes(b) => PyBytes::new(py, b).into_any(),
        V::Array(items) => {
            let list = pyo3::types::PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            list.into_any()
        }
        V::Map(entries) => {
            let d = PyDict::new(py);
            for (k, val) in entries {
                d.set_item(value_to_py(py, k)?, value_to_py(py, val)?)?;
            }
            d.into_any()
        }
    })
}

/// A read-only window onto a part's bytes, borrowed from the source's
/// memory map. Exposes the buffer protocol, so `np.frombuffer(view, ...)`
/// wraps the mapping without copying.
///
/// The view holds a reference to its `Source`, so the mapping cannot be
/// unmapped while any buffer over it is alive.
#[pyclass(unsendable)]
struct TensorView {
    #[allow(dead_code)]
    owner: Py<Source>,
    ptr: *const u8,
    len: usize,
}

#[pymethods]
impl TensorView {
    fn __len__(&self) -> usize {
        self.len
    }

    fn __repr__(&self) -> String {
        format!("TensorView({} bytes, zero-copy)", self.len)
    }

    /// Copies the window out as `bytes`.
    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        // SAFETY: `owner` keeps the mapping alive for the life of `self`.
        PyBytes::new(py, unsafe { std::slice::from_raw_parts(self.ptr, self.len) })
    }

    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view is null"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("tensor views are read-only"));
        }
        // SAFETY: caller provides a valid Py_buffer to fill in; the memory
        // described is owned by the source's mapping, which `owner` pins.
        unsafe {
            (*view).buf = slf.ptr as *mut std::ffi::c_void;
            (*view).len = slf.len as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                c"B".as_ptr() as *mut _
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = std::ptr::null_mut();
            (*view).strides = std::ptr::null_mut();
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            // Keep the view object alive for as long as the buffer is.
            let obj: Py<Self> = slf.into();
            (*view).obj = obj.into_ptr();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

/// Writes canonical `.zt` files.
#[pyclass(unsendable)]
struct Writer {
    inner: Option<ztensor::Writer>,
}

#[pymethods]
impl Writer {
    /// Opens a writer. Canonical form by default; pass `align` (a power of
    /// two >= 4096) for non-canonical placement.
    #[new]
    #[pyo3(signature = (path, align = None))]
    fn new(path: &str, align: Option<u64>) -> PyResult<Self> {
        let inner = match align {
            None => ztensor::Writer::create(path),
            Some(a) => ztensor::Writer::create_with_alignment(path, a),
        }
        .map_err(err)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Adds a dense tensor. `data` is little-endian bytes of exactly
    /// `prod(shape) * itemsize(dtype)`.
    ///
    /// With `compress=True` the part is stored through the
    /// `zt.zstd-seekable/1` profile. Canonical form is raw by definition,
    /// so a compressing writer must have been opened with an `align`.
    #[pyo3(signature = (name, shape, dtype, data, compress = false))]
    fn add(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        dtype: &str,
        // Any contiguous buffer — a numpy array goes in directly, with no
        // `tobytes()` copy of the tensor on the way.
        data: pyo3::buffer::PyBuffer<u8>,
        compress: bool,
    ) -> PyResult<()> {
        if !data.is_c_contiguous() {
            return Err(PyValueError::new_err("data must be C-contiguous"));
        }
        // SAFETY: the buffer is read-only for the duration of this call and
        // the GIL is held, so the exporter cannot resize or free it.
        let data: &[u8] =
            unsafe { std::slice::from_raw_parts(data.buf_ptr() as *const u8, data.item_count()) };
        let dtype = ztensor::DType::from_name(dtype)
            .ok_or_else(|| PyValueError::new_err(format!("unknown dtype {dtype:?}")))?;
        let w = self.writer()?;
        if !compress {
            return w.add_dense(name, &shape, dtype, data).map_err(err);
        }
        w.add_object(
            name,
            &shape,
            "dense",
            &[(
                "data",
                ztensor::PartDef {
                    dtype,
                    ltype: None,
                    encoding: Some("zt.zstd-seekable/1"),
                    data,
                },
            )],
            None,
        )
        .map_err(err)
    }

    /// Writes the manifest and footer; returns the file size. The writer
    /// cannot be used afterwards.
    fn finish(&mut self) -> PyResult<u64> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("writer already finished"))?
            .finish()
            .map_err(err)
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, _args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<()> {
        if self.inner.is_some() {
            self.finish()?;
        }
        Ok(())
    }
}

impl Writer {
    fn writer(&mut self) -> PyResult<&mut ztensor::Writer> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("writer already finished"))
    }
}

/// Opens a tensor file of any supported format.
#[pyfunction]
fn open(path: &str) -> PyResult<Source> {
    let inner = ztensor_compat::open_any(path).map_err(err)?;
    Ok(Source {
        inner,
        path: path.to_string(),
    })
}

/// Sniffs the format of a file: "zt", "safetensors", "gguf", "npz", "pt",
/// "hdf5", or "onnx".
#[pyfunction]
fn detect(path: &str) -> PyResult<&'static str> {
    ztensor_compat::detect(path).map_err(err)
}

/// Converts any supported format to a canonical (or `align`ed) `.zt` file.
/// Returns the output size in bytes.
#[pyfunction]
#[pyo3(signature = (src, dst, align = None))]
fn convert(src: &str, dst: &str, align: Option<u64>) -> PyResult<u64> {
    let source = ztensor_compat::open_any(src).map_err(err)?;
    let mut writer = match align {
        None => ztensor::Writer::create(dst),
        Some(a) => ztensor::Writer::create_with_alignment(dst, a),
    }
    .map_err(err)?;
    writer.ingest(source.as_ref()).map_err(err)?;
    writer.finish().map_err(err)
}

/// Verifies a `.zt` file: structural validation plus every part digest;
/// `deep=True` additionally checks whole-shard digests. Returns the number
/// of digest-verified parts.
#[pyfunction]
#[pyo3(signature = (path, deep = false))]
fn verify(path: &str, deep: bool) -> PyResult<u64> {
    let model = ztensor::Model::open(path).map_err(err)?;
    let manifest = model.manifest().clone();
    let mut verified = 0u64;
    for (name, obj) in &manifest.objects {
        for part in obj.parts.keys() {
            if model.verify(name, part).map_err(err)? {
                verified += 1;
            }
        }
    }
    if deep {
        model.verify_shards().map_err(err)?;
    }
    Ok(verified)
}

#[pymodule]
fn _ztensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Source>()?;
    m.add_class::<TensorView>()?;
    m.add_class::<Writer>()?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(convert, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    Ok(())
}
