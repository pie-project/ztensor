//! Python bindings: the zTensor object model, pythonically.
//!
//! Five concepts, the same five the Rust crate has: a [`Source`] you open, the
//! [`Tensor`]s in it, what each one [`Caps`] can do, where its bytes are
//! ([`Location`]), and a [`Writer`] for putting them somewhere.
//!
//! Tensors export the buffer protocol and DLPack, so `np.from_dlpack(t)`,
//! `torch.from_dlpack(t)` and `memoryview(t)` all work without this crate
//! knowing anything about numpy or torch. That is why there is no per-framework
//! module: DLPack is the interchange, and it can say `bfloat16` — which the
//! numpy dtype table never could.

use std::os::raw::{c_char, c_int, c_void};

use pyo3::exceptions::{PyBufferError, PyIOError, PyKeyError, PyTypeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyList, PyTuple};

fn err(e: ztensor::Error) -> PyErr {
    match e {
        ztensor::Error::Io(io) => PyIOError::new_err(io.to_string()),
        ztensor::Error::NotFound(what) => PyKeyError::new_err(what),
        other => PyValueError::new_err(other.to_string()),
    }
}

// =======================================================================
// Source
// =======================================================================

/// A tensor file — or several read as one — of any supported format.
#[pyclass(unsendable)]
struct Source {
    inner: Option<ztensor::Source>,
    label: String,
}

impl Source {
    fn get(&self) -> PyResult<&ztensor::Source> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("source is closed"))
    }
}

#[pymethods]
impl Source {
    /// Tensor names, sorted.
    fn names(&self) -> PyResult<Vec<String>> {
        Ok(self.get()?.names().map(str::to_string).collect())
    }

    /// Alias for [`names`], for the dict-shaped habit.
    fn keys(&self) -> PyResult<Vec<String>> {
        self.names()
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.get()?.len())
    }

    fn __contains__(&self, name: &str) -> PyResult<bool> {
        Ok(self.get()?.contains(name))
    }

    fn __getitem__(slf: Bound<'_, Self>, name: &str) -> PyResult<Tensor> {
        {
            let this = slf.borrow();
            if !this.get()?.contains(name) {
                return Err(PyKeyError::new_err(name.to_string()));
            }
        }
        Ok(Tensor {
            source: slf.unbind(),
            name: name.to_string(),
            part: None,
        })
    }

    /// Iterating a source yields its tensors, in name order.
    fn __iter__(slf: Bound<'_, Self>) -> PyResult<TensorIter> {
        let names = slf.borrow().names()?;
        Ok(TensorIter {
            source: slf.unbind(),
            names: names.into_iter(),
        })
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            None => format!("<Source {:?} closed>", self.label),
            Some(src) => format!("<Source {:?}, {} tensors>", self.label, src.len()),
        }
    }

    /// File-level attributes, or `None`.
    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.get()?
            .attributes()
            .map(|v| value_to_py(py, v))
            .transpose()
    }

    /// The files this source reads from.
    fn files(&self) -> PyResult<Vec<String>> {
        Ok(self
            .get()?
            .stores()
            .iter()
            .map(|s| s.path().display().to_string())
            .collect())
    }

    /// True for a `.zt` data shard: a container with no manifest.
    #[getter]
    fn is_data_shard(&self) -> PyResult<bool> {
        Ok(self.get()?.is_data_shard())
    }

    /// Verifies every part of every tensor. Returns `(checked, undigested)`.
    #[pyo3(signature = (deep = false))]
    fn verify(&self, deep: bool) -> PyResult<(u64, u64)> {
        let src = self.get()?;
        let (mut checked, mut undigested) = (0u64, 0u64);
        for tensor in src.tensors() {
            for name in tensor.parts() {
                match tensor.part(name).map_err(err)?.verify().map_err(err)? {
                    ztensor::Verified::Digest => checked += 1,
                    ztensor::Verified::NoDigest => undigested += 1,
                }
            }
        }
        if deep {
            src.verify_shards().map_err(err)?;
        }
        Ok((checked, undigested))
    }

    fn close(&mut self) {
        self.inner = None;
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, _args: &Bound<'_, PyTuple>) -> bool {
        self.close();
        false
    }
}

#[pyclass(unsendable)]
struct TensorIter {
    source: Py<Source>,
    names: std::vec::IntoIter<String>,
}

#[pymethods]
impl TensorIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<Tensor> {
        let name = self.names.next()?;
        Some(Tensor {
            source: self.source.clone_ref(py),
            name,
            part: None,
        })
    }
}

// =======================================================================
// Tensor
// =======================================================================

/// One tensor, or one part of one. Holding it has read nothing.
///
/// A dense tensor has a single part named `"data"`, and the byte methods here
/// address it. A quantized one has more — `t["scales"]` is the same class
/// pointed at another part, which is why parts answer every question tensors
/// do.
#[pyclass(unsendable)]
struct Tensor {
    source: Py<Source>,
    name: String,
    part: Option<String>,
}

/// Runs `f` against the underlying part. The closure gets a borrow that lives
/// only as long as the call, which is what keeps the mapping honest.
fn with_part<R>(
    py: Python<'_>,
    tensor: &Tensor,
    f: impl FnOnce(ztensor::Part<'_>) -> PyResult<R>,
) -> PyResult<R> {
    let source = tensor.source.bind(py).borrow();
    let src = source.get()?;
    let handle = src.tensor(&tensor.name).map_err(err)?;
    let part = match &tensor.part {
        None => handle.data().map_err(err)?,
        Some(name) => handle.part(name).map_err(err)?,
    };
    f(part)
}

#[pymethods]
impl Tensor {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The part this handle addresses: `"data"` unless it was indexed.
    #[getter]
    fn part(&self) -> &str {
        self.part.as_deref().unwrap_or("data")
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Vec<u64>> {
        let source = self.source.bind(py).borrow();
        Ok(source.get()?.tensor(&self.name).map_err(err)?.shape().to_vec())
    }

    #[getter]
    fn layout(&self, py: Python<'_>) -> PyResult<String> {
        let source = self.source.bind(py).borrow();
        Ok(source
            .get()?
            .tensor(&self.name)
            .map_err(err)?
            .layout()
            .to_string())
    }

    /// Storage type: `"f32"`, `"bf16"`, `"u8"`, ...
    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<String> {
        with_part(py, self, |p| Ok(p.dtype().as_str().to_string()))
    }

    /// Logical type laid over the storage type — `"bool"`, `"f8_e4m3fn"`,
    /// `"f4_e2m1"` — or `None`.
    #[getter]
    fn logical(&self, py: Python<'_>) -> PyResult<Option<String>> {
        with_part(py, self, |p| Ok(p.logical().map(str::to_string)))
    }

    /// Decoded size in bytes.
    #[getter]
    fn nbytes(&self, py: Python<'_>) -> PyResult<u64> {
        with_part(py, self, |p| Ok(p.nbytes()))
    }

    /// Part names of this tensor, sorted.
    #[getter]
    fn parts(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let source = self.source.bind(py).borrow();
        Ok(source
            .get()?
            .tensor(&self.name)
            .map_err(err)?
            .parts()
            .map(str::to_string)
            .collect())
    }

    /// Another part of the same tensor.
    fn __getitem__(slf: Bound<'_, Self>, part: &str) -> PyResult<Tensor> {
        let py = slf.py();
        let this = slf.borrow();
        {
            let source = this.source.bind(py).borrow();
            source
                .get()?
                .tensor(&this.name)
                .map_err(err)?
                .part(part)
                .map_err(err)?;
        }
        Ok(Tensor {
            source: this.source.clone_ref(py),
            name: this.name.clone(),
            part: Some(part.to_string()),
        })
    }

    /// What can be done with these bytes.
    #[getter]
    fn caps(&self, py: Python<'_>) -> PyResult<Caps> {
        with_part(py, self, |p| {
            let caps = p.caps();
            Ok(Caps {
                map: caps.map,
                locate: caps.locate,
                evict: caps.evict,
                verify: caps.verify,
                alignment: caps.alignment,
            })
        })
    }

    /// Where the bytes are, for a caller doing its own I/O. Raises when the
    /// decoded bytes are not one range of one file.
    #[getter]
    fn location(&self, py: Python<'_>) -> PyResult<Location> {
        with_part(py, self, |p| {
            let at = p.locate().map_err(err)?;
            Ok(Location {
                path: p.store().path().display().to_string(),
                offset: at.offset,
                nbytes: at.len,
            })
        })
    }

    /// Decoded bytes, copied.
    fn tobytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let owned = with_part(py, self, |p| Ok(p.bytes().map_err(err)?.into_owned()))?;
        Ok(PyBytes::new(py, &owned))
    }

    /// True when the bytes can be handed over without a copy.
    fn is_mapped(&self, py: Python<'_>) -> PyResult<bool> {
        with_part(py, self, |p| Ok(p.caps().map))
    }

    /// Checks this part's digest and its logical type's content rules.
    /// Returns True when a digest was actually checked.
    fn verify(&self, py: Python<'_>) -> PyResult<bool> {
        with_part(py, self, |p| Ok(p.verify().map_err(err)?.checked()))
    }

    fn prefetch(&self, py: Python<'_>) -> PyResult<()> {
        with_part(py, self, |p| p.prefetch().map_err(err))
    }

    fn evict(&self, py: Python<'_>) -> PyResult<()> {
        #[cfg(unix)]
        {
            with_part(py, self, |p| p.evict().map_err(err))
        }
        #[cfg(not(unix))]
        {
            let _ = py;
            Err(PyValueError::new_err("eviction is a unix capability"))
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let shape = self.shape(py)?;
        Ok(format!(
            "<Tensor {:?} {}[{}] {}>",
            self.name,
            self.dtype(py)?,
            shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(","),
            self.part()
        ))
    }

    // ---- interchange ----

    /// The buffer protocol, so `memoryview(t)` and anything built on it work.
    ///
    /// Only a mapped part can be exported this way: a buffer is a window onto
    /// memory that outlives the call, and bytes that had to be decoded have no
    /// such window.
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
        let py = slf.py();
        let (ptr, len) = with_part(py, &slf, |p| {
            let bytes = p.map().map_err(|_| {
                PyBufferError::new_err(
                    "this tensor cannot be exported as a buffer: its bytes are not a \
                     mapped range (use .tobytes())",
                )
            })?;
            Ok((bytes.as_ptr(), bytes.len()))
        })?;
        // SAFETY: the caller provides a valid Py_buffer to fill in. The memory
        // belongs to the source's mapping, which the `obj` reference below
        // keeps alive for as long as the buffer is.
        unsafe {
            (*view).buf = ptr as *mut c_void;
            (*view).len = len as isize;
            (*view).readonly = 1;
            (*view).itemsize = 1;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                c"B".as_ptr() as *mut c_char
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = std::ptr::null_mut();
            (*view).strides = std::ptr::null_mut();
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            let obj: Py<Self> = slf.into();
            (*view).obj = obj.into_ptr();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}

    /// DLPack export: `np.from_dlpack(t)`, `torch.from_dlpack(t)`.
    ///
    /// Zero-copy, and typed — DLPack can say `bfloat16`, which is the dtype
    /// most of these files are actually in.
    #[pyo3(signature = (stream = None, **_kwargs))]
    fn __dlpack__<'py>(
        slf: Bound<'py, Self>,
        stream: Option<Bound<'py, PyAny>>,
        _kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let py = slf.py();
        if let Some(stream) = &stream {
            if !stream.is_none() {
                return Err(PyValueError::new_err(
                    "these tensors are host memory; there is no stream to order against",
                ));
            }
        }
        let this = slf.borrow();
        let (ptr, len, dtype, shape) = with_part(py, &this, |p| {
            let bytes = p.map().map_err(|_| {
                PyValueError::new_err(
                    "this tensor cannot be exported through DLPack: its bytes are not a \
                     mapped range (use .tobytes())",
                )
            })?;
            let dtype = dl_dtype(p.dtype(), p.logical())?;
            Ok((bytes.as_ptr(), bytes.len(), dtype, ()))
        })?;
        let _ = (len, shape);

        // The logical shape, in elements of the exported dtype.
        let dims = this.shape(py)?;
        let shape: Vec<i64> = if dims.is_empty() {
            vec![1]
        } else {
            dims.iter().map(|&d| d as i64).collect()
        };

        let managed = Box::new(Managed {
            tensor: DlManagedTensor {
                dl_tensor: DlTensor {
                    data: ptr as *mut c_void,
                    device: DlDevice {
                        device_type: 1, // kDLCPU
                        device_id: 0,
                    },
                    ndim: shape.len() as c_int,
                    dtype,
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: std::ptr::null_mut(),
                deleter: Some(dlpack_deleter),
            },
            shape,
            // Keeps the source (and therefore the mapping) alive for as long as
            // the consumer holds the tensor.
            owner: slf.clone().unbind(),
        });
        let raw = Box::into_raw(managed);
        // SAFETY: `raw` is a live, uniquely-owned allocation; the pointers we
        // write into it point into that same allocation.
        unsafe {
            (*raw).tensor.dl_tensor.shape = (*raw).shape.as_mut_ptr();
            (*raw).tensor.manager_ctx = raw as *mut c_void;
        }

        let capsule = unsafe {
            // SAFETY: the capsule takes the pointer with a destructor that
            // frees it exactly once, and only if the consumer did not rename
            // the capsule to claim ownership.
            let ptr = ffi::PyCapsule_New(
                raw as *mut c_void,
                DLTENSOR_NAME.as_ptr() as *const c_char,
                Some(capsule_destructor),
            );
            Bound::from_owned_ptr_or_err(py, ptr)?
        };
        capsule.downcast_into::<PyCapsule>().map_err(Into::into)
    }

    /// `(device_type, device_id)` — always CPU here.
    fn __dlpack_device__(&self) -> (c_int, c_int) {
        (1, 0)
    }
}

// ---- DLPack ABI -------------------------------------------------------

const DLTENSOR_NAME: &std::ffi::CStr = c"dltensor";
const DLTENSOR_USED: &std::ffi::CStr = c"used_dltensor";

#[repr(C)]
#[derive(Clone, Copy)]
struct DlDevice {
    device_type: c_int,
    device_id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DlDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DlTensor {
    data: *mut c_void,
    device: DlDevice,
    ndim: c_int,
    dtype: DlDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DlManagedTensor {
    dl_tensor: DlTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DlManagedTensor)>,
}

/// What the capsule owns: the ABI struct, the shape it points at, and a
/// reference that keeps the mapping alive.
struct Managed {
    tensor: DlManagedTensor,
    shape: Vec<i64>,
    #[allow(dead_code)]
    owner: Py<Tensor>,
}

/// Called by the consumer once it is done with the tensor.
unsafe extern "C" fn dlpack_deleter(tensor: *mut DlManagedTensor) {
    if tensor.is_null() {
        return;
    }
    // SAFETY: manager_ctx is the `Managed` allocation this tensor came from,
    // and the deleter runs exactly once.
    unsafe {
        let ctx = (*tensor).manager_ctx as *mut Managed;
        if !ctx.is_null() {
            drop(Box::from_raw(ctx));
        }
    }
}

/// Called by Python if the capsule is dropped *unconsumed*. A consumer that
/// takes ownership renames the capsule, and then this must not free anything.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: called by CPython with a valid capsule.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, DLTENSOR_NAME.as_ptr() as *const c_char) == 0 {
            return; // renamed to used_dltensor: the consumer owns it now
        }
        let ptr =
            ffi::PyCapsule_GetPointer(capsule, DLTENSOR_NAME.as_ptr() as *const c_char)
                as *mut DlManagedTensor;
        if ptr.is_null() {
            return;
        }
        if let Some(deleter) = (*ptr).deleter {
            deleter(ptr);
        }
    }
}

/// Storage + logical type → the DLPack type triple.
fn dl_dtype(dtype: ztensor::DType, logical: Option<&str>) -> PyResult<DlDataType> {
    use ztensor::DType::*;
    // kDLInt = 0, kDLUInt = 1, kDLFloat = 2, kDLBfloat = 4, kDLBool = 6
    let (code, bits) = match (logical, dtype) {
        (Some("bool"), _) => (6u8, 8u8),
        (Some(other), _) => {
            return Err(PyTypeError::new_err(format!(
                "logical type {other:?} has no DLPack encoding; use .tobytes() and \
                 interpret it yourself"
            )))
        }
        (None, F64) => (2, 64),
        (None, F32) => (2, 32),
        (None, F16) => (2, 16),
        (None, BF16) => (4, 16),
        (None, I64) => (0, 64),
        (None, I32) => (0, 32),
        (None, I16) => (0, 16),
        (None, I8) => (0, 8),
        (None, U64) => (1, 64),
        (None, U32) => (1, 32),
        (None, U16) => (1, 16),
        (None, U8) => (1, 8),
    };
    Ok(DlDataType {
        code,
        bits,
        lanes: 1,
    })
}

// =======================================================================
// small value types
// =======================================================================

/// What one part can do. Every field is named after the operation it gates.
#[pyclass(get_all, frozen)]
#[derive(Clone)]
struct Caps {
    /// A zero-copy export (buffer protocol, DLPack) will succeed.
    map: bool,
    /// `.location` will give the exact range of the decoded bytes.
    locate: bool,
    /// `.evict()` will succeed: no other blob shares an OS page with this one.
    evict: bool,
    /// `.verify()` will check a digest rather than report there is none.
    verify: bool,
    /// Largest power of two dividing the file offset.
    alignment: u64,
}

#[pymethods]
impl Caps {
    fn __repr__(&self) -> String {
        format!(
            "<Caps map={} locate={} evict={} verify={} alignment={}>",
            self.map, self.locate, self.evict, self.verify, self.alignment
        )
    }
}

/// Where a part's bytes are, for a caller that wants to read them itself.
#[pyclass(get_all, frozen)]
#[derive(Clone)]
struct Location {
    path: String,
    offset: u64,
    nbytes: u64,
}

#[pymethods]
impl Location {
    fn __repr__(&self) -> String {
        format!(
            "<Location {:?} +{} ({} bytes)>",
            self.path, self.offset, self.nbytes
        )
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
            let list = PyList::empty(py);
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

fn py_to_value(v: &Bound<'_, PyAny>) -> PyResult<ztensor::cbor::Value> {
    use ztensor::cbor::Value as V;
    if v.is_none() {
        return Ok(V::Null);
    }
    if let Ok(b) = v.extract::<bool>() {
        return Ok(V::Bool(b));
    }
    if let Ok(n) = v.extract::<u64>() {
        return Ok(V::Uint(n));
    }
    if let Ok(n) = v.extract::<i64>() {
        return Ok(V::from(n));
    }
    if let Ok(x) = v.extract::<f64>() {
        return Ok(V::Float(x));
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(V::Text(s));
    }
    if let Ok(items) = v.downcast::<PyList>() {
        return Ok(V::Array(
            items.iter().map(|i| py_to_value(&i)).collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(d) = v.downcast::<PyDict>() {
        let mut entries = Vec::new();
        for (k, val) in d.iter() {
            let key = k.extract::<String>().map_err(|_| {
                PyTypeError::new_err("attribute keys must be strings (spec §3.5)")
            })?;
            entries.push((V::Text(key), py_to_value(&val)?));
        }
        return Ok(V::Map(entries));
    }
    Err(PyTypeError::new_err(format!(
        "{} is not representable as a zTensor attribute",
        v.get_type().name()?
    )))
}

// =======================================================================
// Writer
// =======================================================================

/// Writes `.zt` files.
#[pyclass(unsendable)]
struct Writer {
    inner: Option<ztensor::Writer>,
}

impl Writer {
    fn get(&mut self) -> PyResult<&mut ztensor::Writer> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("writer is already finished"))
    }
}

#[pymethods]
impl Writer {
    /// Opens a writer.
    ///
    /// Canonical form by default: 64 KiB placement, ascending names, a digest
    /// on every tensor, a byte-identical file for identical input. Pass
    /// `canonical=False` to choose your own `align` or insert in any order,
    /// and `publish=True` to write beside the target and rename into place, so
    /// nothing ever reads a half-written file.
    #[new]
    #[pyo3(signature = (path, canonical = true, align = None, publish = false))]
    fn new(path: &str, canonical: bool, align: Option<u64>, publish: bool) -> PyResult<Self> {
        let mut options = ztensor::writer::Options::default().canonical(canonical);
        if let Some(align) = align {
            options = options.align(align);
        }
        let inner = if publish {
            options.publish(path)
        } else {
            options.create(path)
        }
        .map_err(err)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Adds a dense tensor from any contiguous buffer (a numpy array goes in
    /// directly, with no `tobytes()` copy on the way).
    #[pyo3(signature = (name, data, shape, dtype, logical = None, encoding = None))]
    fn add(
        &mut self,
        name: &str,
        data: pyo3::buffer::PyBuffer<u8>,
        shape: Vec<u64>,
        dtype: &str,
        logical: Option<&str>,
        encoding: Option<&str>,
    ) -> PyResult<()> {
        if !data.is_c_contiguous() {
            return Err(PyValueError::new_err("data must be C-contiguous"));
        }
        // SAFETY: the buffer is read-only for the duration of this call and the
        // GIL is held, so the exporter cannot resize or free it.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.buf_ptr() as *const u8, data.item_count()) };
        let dtype: ztensor::DType = dtype
            .parse()
            .map_err(|_| PyValueError::new_err(format!("unknown dtype {dtype:?}")))?;
        let writer = self.get()?;
        let mut builder = writer.object(name).shape(shape).part("data").dtype(dtype);
        if let Some(logical) = logical {
            builder = builder.logical(logical);
        }
        if let Some(encoding) = encoding {
            builder = builder.encoding(encoding);
        }
        builder.bytes(bytes).add().map_err(err)
    }

    /// Sets file-level attributes from a dict.
    fn set_attributes(&mut self, attributes: &Bound<'_, PyDict>) -> PyResult<()> {
        let value = py_to_value(attributes.as_any())?;
        self.get()?.set_attributes(value);
        Ok(())
    }

    /// Copies every tensor of a source into this file.
    fn ingest(&mut self, source: &Source) -> PyResult<()> {
        let src = source.get()?;
        self.get()?.ingest(src).map_err(err)
    }

    /// Writes the manifest and footer; returns the file size. The writer
    /// cannot be used afterwards.
    fn finish(&mut self) -> PyResult<u64> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("writer is already finished"))?
            .finish()
            .map_err(err)
    }

    /// Throws the file away.
    fn abandon(&mut self) {
        if let Some(writer) = self.inner.take() {
            writer.abandon();
        }
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*args))]
    fn __exit__(&mut self, args: &Bound<'_, PyTuple>) -> PyResult<bool> {
        let failed = args
            .get_item(0)
            .map(|e| !e.is_none())
            .unwrap_or(false);
        if failed {
            // Leaving a partial file behind because the body raised would be
            // the one thing publishing exists to prevent.
            self.abandon();
        } else if self.inner.is_some() {
            self.finish()?;
        }
        Ok(false)
    }
}

// =======================================================================
// module functions
// =======================================================================

fn open_paths(paths: &Bound<'_, PyAny>, map: bool) -> PyResult<(ztensor::Source, String)> {
    let options = ztensor_compat::options().map(map);
    if let Ok(path) = paths.extract::<String>() {
        let src = options.open(&path).map_err(err)?;
        return Ok((src, path));
    }
    let list: Vec<String> = paths.extract().map_err(|_| {
        PyTypeError::new_err("expected a path or a sequence of paths")
    })?;
    let label = match list.len() {
        0 => "()".to_string(),
        1 => list[0].clone(),
        n => format!("{} + {} more", list[0], n - 1),
    };
    let src = options.open_all(&list).map_err(err)?;
    Ok((src, label))
}

/// Opens a tensor file of any supported format — or several, read as one name
/// space.
#[pyfunction]
#[pyo3(name = "open")]
fn open_(paths: &Bound<'_, PyAny>) -> PyResult<Source> {
    let (inner, label) = open_paths(paths, true)?;
    Ok(Source {
        inner: Some(inner),
        label,
    })
}

/// Opens without mapping: names, shapes, and addresses, but no borrowed bytes.
/// What a planner wants, and it costs two reads rather than a mapping of the
/// whole checkpoint.
#[pyfunction]
fn index(paths: &Bound<'_, PyAny>) -> PyResult<Source> {
    let (inner, label) = open_paths(paths, false)?;
    Ok(Source {
        inner: Some(inner),
        label,
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
    let source = ztensor_compat::open(src).map_err(err)?;
    let mut writer = match align {
        None => ztensor::Writer::create(dst),
        Some(a) => ztensor::Writer::options()
            .canonical(false)
            .align(a)
            .create(dst),
    }
    .map_err(err)?;
    writer.ingest(&source).map_err(err)?;
    writer.finish().map_err(err)
}

/// Verifies a file: structural validation plus every part digest; `deep=True`
/// additionally checks whole-shard digests. Returns
/// `(digest_verified, without_digests)`.
#[pyfunction]
#[pyo3(signature = (path, deep = false))]
fn verify(path: &str, deep: bool) -> PyResult<(u64, u64)> {
    let source = ztensor_compat::open(path).map_err(err)?;
    let (mut checked, mut undigested) = (0u64, 0u64);
    for tensor in source.tensors() {
        for name in tensor.parts() {
            match tensor.part(name).map_err(err)?.verify().map_err(err)? {
                ztensor::Verified::Digest => checked += 1,
                ztensor::Verified::NoDigest => undigested += 1,
            }
        }
    }
    if deep {
        source.verify_shards().map_err(err)?;
    }
    Ok((checked, undigested))
}

/// The OS page size.
#[pyfunction]
fn page_size() -> u64 {
    ztensor::page_size()
}

#[pymodule]
fn _ztensor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Source>()?;
    m.add_class::<Tensor>()?;
    m.add_class::<Caps>()?;
    m.add_class::<Location>()?;
    m.add_class::<Writer>()?;
    m.add_function(wrap_pyfunction!(open_, m)?)?;
    m.add_function(wrap_pyfunction!(index, m)?)?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(convert, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(page_size, m)?)?;
    let _ = DLTENSOR_USED;
    Ok(())
}
