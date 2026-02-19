//! zTensor file reader.
//!
//! Supports both v1.2 (ZTEN1000) and legacy v0.1.0 (ZTEN0001) files
//! through a unified `Reader` type.

use half::{bf16, f16};
use memmap2::{Mmap, MmapMut, MmapOptions};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::Error;
use crate::models::{
    Component, CooTensor, CsrTensor, DType, Encoding, Format, Manifest, Object, ALIGNMENT, MAGIC,
    MAGIC_V01, MAX_MANIFEST_SIZE,
};
use crate::utils::swap_endianness_in_place;

/// Max object size (32GB).
const MAX_OBJECT_SIZE: u64 = 32 * 1024 * 1024 * 1024;

// =========================================================================
// TensorElement trait
// =========================================================================

/// Trait for types that can be safely read from tensor byte data.
///
/// Each implementing type declares its corresponding [`DType`] at compile time.
/// Implemented for `f32`, `f64`, `i8`–`i64`, `u8`–`u64`, `bool`,
/// [`half::f16`], and [`half::bf16`].
///
/// # Examples
///
/// ```
/// use ztensor::TensorElement;
/// use ztensor::DType;
///
/// assert_eq!(f32::DTYPE, DType::F32);
/// assert_eq!(f32::SIZE, 4);
/// assert_eq!(u8::DTYPE, DType::U8);
/// ```
pub trait TensorElement: Sized + Default + Clone + 'static {
    /// The `DType` that corresponds to this Rust type.
    const DTYPE: DType;
    /// Size of one element in bytes.
    const SIZE: usize = std::mem::size_of::<Self>();
}

macro_rules! impl_tensor_element {
    ($t:ty, $d:expr) => {
        impl TensorElement for $t {
            const DTYPE: DType = $d;
        }
    };
}

impl_tensor_element!(f64, DType::F64);
impl_tensor_element!(f32, DType::F32);
impl_tensor_element!(i64, DType::I64);
impl_tensor_element!(i32, DType::I32);
impl_tensor_element!(i16, DType::I16);
impl_tensor_element!(i8, DType::I8);
impl_tensor_element!(u64, DType::U64);
impl_tensor_element!(u32, DType::U32);
impl_tensor_element!(u16, DType::U16);
impl_tensor_element!(u8, DType::U8);
impl_tensor_element!(bool, DType::Bool);
impl_tensor_element!(f16, DType::F16);
impl_tensor_element!(bf16, DType::BF16);

// =========================================================================
// Shared byte-to-typed conversion helpers
// =========================================================================

/// Validates alignment/size and reinterprets a byte slice as a typed slice (zero-copy).
pub(crate) fn bytes_as_typed<T: TensorElement>(bytes: &[u8]) -> Result<&[T], Error> {
    if !bytes.len().is_multiple_of(T::SIZE) {
        return Err(Error::InvalidFileStructure(format!(
            "Byte length {} is not a multiple of type size {}",
            bytes.len(),
            T::SIZE
        )));
    }

    let ptr = bytes.as_ptr();
    if !(ptr as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return Err(Error::Other(format!(
            "Memory not aligned for type {}",
            std::any::type_name::<T>()
        )));
    }

    let len = bytes.len() / T::SIZE;
    Ok(unsafe { std::slice::from_raw_parts(ptr as *const T, len) })
}

/// Validates size and copies byte data into a typed `Vec<T>`.
pub(crate) fn bytes_to_typed_vec<T: TensorElement>(bytes: &[u8]) -> Result<Vec<T>, Error> {
    if !bytes.len().is_multiple_of(T::SIZE) {
        return Err(Error::InvalidFileStructure(format!(
            "Byte length {} is not a multiple of type size {}",
            bytes.len(),
            T::SIZE
        )));
    }

    let num_elements = bytes.len() / T::SIZE;
    let mut result = vec![T::default(); num_elements];

    unsafe {
        std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, bytes.len())
            .copy_from_slice(bytes);
    }

    Ok(result)
}

// =========================================================================
// TensorData — zero-copy abstraction for tensor byte data
// =========================================================================

/// Tensor data that may be borrowed (zero-copy) or owned.
///
/// `Borrowed` holds a reference into memory-mapped file data.
/// `Owned` holds a freshly allocated `Vec<u8>` (from stream reads or decompression).
///
/// # Examples
///
/// ```no_run
/// use ztensor::TensorReader;
///
/// let reader = ztensor::open("model.zt")?;
/// let data = reader.read_data("weights")?;
///
/// // Access raw bytes
/// let bytes: &[u8] = data.as_slice();
///
/// // Or convert to owned
/// let owned: Vec<u8> = data.into_owned();
/// # Ok::<(), ztensor::Error>(())
/// ```
pub enum TensorData<'a> {
    /// Zero-copy reference to memory-mapped data.
    Borrowed(&'a [u8]),
    /// Owned byte buffer (from stream read or decompression).
    Owned(Vec<u8>),
}

impl<'a> TensorData<'a> {
    /// Returns a byte slice regardless of variant.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            TensorData::Borrowed(s) => s,
            TensorData::Owned(v) => v,
        }
    }

    /// Converts to an owned `Vec<u8>`, copying if borrowed.
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            TensorData::Borrowed(s) => s.to_vec(),
            TensorData::Owned(v) => v,
        }
    }

    /// Reinterprets the byte data as a typed slice.
    ///
    /// Returns an error if the dtype doesn't match or alignment/size is wrong.
    pub fn as_typed<T: TensorElement>(&self, expected_dtype: DType) -> Result<&[T], Error> {
        if T::DTYPE != expected_dtype {
            return Err(Error::TypeMismatch {
                expected: expected_dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: "TensorData::as_typed".to_string(),
            });
        }
        bytes_as_typed(self.as_slice())
    }

    /// Converts byte data into a typed `Vec<T>`, always copying.
    pub fn into_typed<T: TensorElement>(self, expected_dtype: DType) -> Result<Vec<T>, Error> {
        if T::DTYPE != expected_dtype {
            return Err(Error::TypeMismatch {
                expected: expected_dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: "TensorData::into_typed".to_string(),
            });
        }
        bytes_to_typed_vec(self.as_slice())
    }
}

// =========================================================================
// ComponentData / Tensor — unified multi-component abstraction
// =========================================================================

/// Metadata and data for a single component of a tensor object.
///
/// Returned as part of [`Tensor`] from [`TensorReader::read_object`].
pub struct ComponentData<'a> {
    /// Storage data type (e.g., F32, U64).
    pub dtype: DType,
    /// Optional logical type (e.g., "f8_e4m3fn"). When absent, equals `dtype`.
    pub logical_type: Option<String>,
    /// The raw byte data, possibly zero-copy from mmap.
    pub data: TensorData<'a>,
}

/// A fully-read tensor object with all its components.
///
/// This is the Rust counterpart of Python's `PyTensor`. It provides unified
/// access to dense, sparse, quantized, and custom tensor formats through
/// [`TensorReader::read_object`].
///
/// # Examples
///
/// ```no_run
/// use ztensor::TensorReader;
///
/// let reader = ztensor::open("model.zt")?;
/// let tensor = reader.read_object("weights")?;
/// println!("shape: {:?}, format: {}", tensor.shape, tensor.format);
///
/// // For dense tensors, convert to typed vec:
/// let data: Vec<f32> = tensor.into_dense()?;
/// # Ok::<(), ztensor::Error>(())
/// ```
pub struct Tensor<'a> {
    /// Logical dimensions.
    pub shape: Vec<u64>,
    /// Layout format (dense, sparse_csr, sparse_coo, etc.).
    pub format: Format,
    /// Optional per-object attributes.
    pub attributes: Option<BTreeMap<String, ciborium::Value>>,
    /// All components, keyed by role name (e.g., "data", "values", "indices").
    pub components: BTreeMap<String, ComponentData<'a>>,
}

impl<'a> Tensor<'a> {
    /// Converts a dense tensor into a typed `Vec<T>`.
    ///
    /// Returns an error if the tensor is not dense or the dtype doesn't match `T`.
    pub fn into_dense<T: TensorElement>(self) -> Result<Vec<T>, Error> {
        if self.format != Format::Dense {
            return Err(Error::TypeMismatch {
                expected: Format::Dense.as_str().to_string(),
                found: self.format.as_str().to_string(),
                context: "Tensor::into_dense".to_string(),
            });
        }
        let mut components = self.components;
        let comp = components
            .remove("data")
            .ok_or_else(|| Error::InvalidFileStructure("Missing 'data' component".to_string()))?;
        comp.data.into_typed(comp.dtype)
    }

    /// Converts a CSR sparse tensor into a [`CsrTensor<T>`].
    ///
    /// Returns an error if the tensor is not sparse_csr or the dtype doesn't match `T`.
    pub fn into_csr<T: TensorElement>(self) -> Result<CsrTensor<T>, Error> {
        if self.format != Format::SparseCsr {
            return Err(Error::TypeMismatch {
                expected: Format::SparseCsr.as_str().to_string(),
                found: self.format.as_str().to_string(),
                context: "Tensor::into_csr".to_string(),
            });
        }
        let mut components = self.components;

        let values_comp = components
            .remove("values")
            .ok_or_else(|| Error::InvalidFileStructure("Missing 'values' component".to_string()))?;
        let indices_comp = components.remove("indices").ok_or_else(|| {
            Error::InvalidFileStructure("Missing 'indices' component".to_string())
        })?;
        let indptr_comp = components
            .remove("indptr")
            .ok_or_else(|| Error::InvalidFileStructure("Missing 'indptr' component".to_string()))?;

        let values: Vec<T> = values_comp.data.into_typed(values_comp.dtype)?;
        let indices: Vec<u64> = indices_comp.data.into_typed(indices_comp.dtype)?;
        let indptr: Vec<u64> = indptr_comp.data.into_typed(indptr_comp.dtype)?;

        Ok(CsrTensor {
            shape: self.shape,
            indptr,
            indices,
            values,
        })
    }

    /// Converts a COO sparse tensor into a [`CooTensor<T>`].
    ///
    /// Returns an error if the tensor is not sparse_coo or the dtype doesn't match `T`.
    pub fn into_coo<T: TensorElement>(self) -> Result<CooTensor<T>, Error> {
        if self.format != Format::SparseCoo {
            return Err(Error::TypeMismatch {
                expected: Format::SparseCoo.as_str().to_string(),
                found: self.format.as_str().to_string(),
                context: "Tensor::into_coo".to_string(),
            });
        }
        let mut components = self.components;

        let values_comp = components
            .remove("values")
            .ok_or_else(|| Error::InvalidFileStructure("Missing 'values' component".to_string()))?;
        let coords_comp = components
            .remove("coords")
            .ok_or_else(|| Error::InvalidFileStructure("Missing 'coords' component".to_string()))?;

        let values: Vec<T> = values_comp.data.into_typed(values_comp.dtype)?;
        let all_coords: Vec<u64> = coords_comp.data.into_typed(coords_comp.dtype)?;

        let nnz = values.len();
        let ndim = self.shape.len();

        if all_coords.len() != nnz * ndim {
            return Err(Error::DataConversionError(
                "COO coords size mismatch".to_string(),
            ));
        }

        let mut indices = Vec::with_capacity(nnz);
        for i in 0..nnz {
            let mut idx = Vec::with_capacity(ndim);
            for d in 0..ndim {
                idx.push(all_coords[d * nnz + i]);
            }
            indices.push(idx);
        }

        Ok(CooTensor {
            shape: self.shape,
            indices,
            values,
        })
    }

    /// Converts all borrowed data to owned, producing a `Tensor<'static>`.
    ///
    /// This is useful when you need the tensor to outlive the reader.
    pub fn into_owned(self) -> Tensor<'static> {
        let components = self
            .components
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    ComponentData {
                        dtype: v.dtype,
                        logical_type: v.logical_type,
                        data: TensorData::Owned(v.data.into_owned()),
                    },
                )
            })
            .collect();
        Tensor {
            shape: self.shape,
            format: self.format,
            attributes: self.attributes,
            components,
        }
    }
}

// =========================================================================
// TensorReader trait — unified read interface
// =========================================================================

/// Unified read-only interface for all tensor file formats.
///
/// All methods take `&self`. Implementations that need internal mutation
/// (e.g., `Reader` which seeks a stream) use interior mutability.
///
/// Returned by [`ztensor::open`](crate::open) for format-agnostic access.
///
/// # Examples
///
/// ```no_run
/// use ztensor::TensorReader;
///
/// let reader = ztensor::open("model.safetensors")?;
/// for name in reader.keys() {
///     let obj = reader.get(name).unwrap();
///     println!("{}: {:?}", name, obj.shape);
/// }
/// # Ok::<(), ztensor::Error>(())
/// ```
pub trait TensorReader {
    /// Returns the file manifest containing object metadata.
    fn manifest(&self) -> &Manifest;

    /// Reads the raw byte data for a named dense tensor.
    ///
    /// Returns `TensorData::Borrowed` for zero-copy paths (mmap),
    /// or `TensorData::Owned` for stream/decompression paths.
    fn read_data(&self, name: &str) -> Result<TensorData<'_>, Error>;

    /// Lists all objects in the file.
    fn tensors(&self) -> &BTreeMap<String, Object> {
        &self.manifest().objects
    }

    /// Gets metadata for an object by name.
    fn get(&self, name: &str) -> Option<&Object> {
        self.manifest().objects.get(name)
    }

    /// Returns the names of all tensors in the file.
    fn keys(&self) -> Vec<&str> {
        self.manifest().objects.keys().map(|s| s.as_str()).collect()
    }

    /// Reads the raw byte data for a named component of a tensor.
    ///
    /// For non-`.zt` formats (safetensors, pytorch, etc.), only the `"data"`
    /// component is supported. `.zt` files support arbitrary component names
    /// (e.g., `"values"`, `"indices"`, `"indptr"` for sparse tensors).
    fn read_component_data(&self, name: &str, component: &str) -> Result<TensorData<'_>, Error> {
        if component == "data" {
            self.read_data(name)
        } else {
            Err(Error::Other(format!(
                "Component '{}' reading only supported for .zt files",
                component
            )))
        }
    }

    /// Reads a complete tensor object with all its components.
    ///
    /// Returns a [`Tensor`] containing all component data, shape, format,
    /// and attributes. Works for any format (dense, sparse, quantized, custom).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::TensorReader;
    ///
    /// let reader = ztensor::open("model.zt")?;
    /// let tensor = reader.read_object("weights")?;
    /// println!("format: {}, components: {}", tensor.format, tensor.components.len());
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    fn read_object(&self, name: &str) -> Result<Tensor<'_>, Error> {
        let obj = self
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;
        let mut components = BTreeMap::new();
        for (comp_name, comp_meta) in &obj.components {
            let data = self.read_component_data(name, comp_name)?;
            components.insert(
                comp_name.clone(),
                ComponentData {
                    dtype: comp_meta.dtype,
                    logical_type: comp_meta.r#type.clone(),
                    data,
                },
            );
        }
        Ok(Tensor {
            shape: obj.shape.clone(),
            format: obj.format.clone(),
            attributes: obj.attributes.clone(),
            components,
        })
    }

    /// Reads tensor data as a typed vector.
    ///
    /// This method is not available on `dyn TensorReader` trait objects.
    /// Use `read_data()` + `TensorData::into_typed()` instead.
    fn read<T: TensorElement>(&self, name: &str) -> Result<Vec<T>, Error>
    where
        Self: Sized,
    {
        let data = self.read_data(name)?;
        let obj = self
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;
        let component = obj.components.get("data").ok_or_else(|| {
            Error::InvalidFileStructure(format!("Missing 'data' component for {}", name))
        })?;
        data.into_typed(component.dtype)
    }
}

// =========================================================================
// Legacy v0.1.0 manifest parsing
// =========================================================================

/// v0.1.0 tensor metadata (from CBOR array).
#[derive(Debug, Clone, Deserialize)]
struct LegacyTensorMeta {
    name: String,
    offset: u64,
    size: u64,
    dtype: String,
    shape: Vec<u64>,
    encoding: String,
    #[serde(default)]
    layout: String,
    #[serde(default)]
    checksum: Option<String>,
}

impl LegacyTensorMeta {
    fn to_dtype(&self) -> Result<DType, Error> {
        match self.dtype.as_str() {
            "float64" => Ok(DType::F64),
            "float32" => Ok(DType::F32),
            "float16" => Ok(DType::F16),
            "bfloat16" => Ok(DType::BF16),
            "int64" => Ok(DType::I64),
            "int32" => Ok(DType::I32),
            "int16" => Ok(DType::I16),
            "int8" => Ok(DType::I8),
            "uint64" => Ok(DType::U64),
            "uint32" => Ok(DType::U32),
            "uint16" => Ok(DType::U16),
            "uint8" => Ok(DType::U8),
            "bool" => Ok(DType::Bool),
            other => Err(Error::UnsupportedDType(other.to_string())),
        }
    }

    fn to_encoding(&self) -> Encoding {
        match self.encoding.as_str() {
            "zstd" => Encoding::Zstd,
            _ => Encoding::Raw,
        }
    }
}

// =========================================================================
// ReadContext (for error messages)
// =========================================================================

#[derive(Clone)]
struct ReadContext<'a> {
    object_name: &'a str,
    component_name: &'a str,
}

impl<'a> ReadContext<'a> {
    fn new(object_name: &'a str, component_name: &'a str) -> Self {
        Self {
            object_name,
            component_name,
        }
    }

    fn unknown() -> Self {
        Self {
            object_name: "unknown",
            component_name: "unknown",
        }
    }
}

// =========================================================================
// Reader
// =========================================================================

/// Unified reader for zTensor files (v1.2 and legacy v0.1.0).
///
/// Provides typed, raw, and zero-copy access to tensor data. For mmap
/// zero-copy reads, use [`open_mmap`](Reader::open_mmap). For stream reads,
/// use [`open`](Reader::open) or [`new`](Reader::new).
///
/// # Examples
///
/// ```no_run
/// use ztensor::Reader;
///
/// // Stream read
/// let reader = Reader::open("model.zt")?;
/// let weights: Vec<f32> = reader.read_as("weights")?;
///
/// // Memory-mapped zero-copy read
/// let reader = Reader::open_mmap("model.zt")?;
/// let view: &[f32] = reader.view_as("weights")?;
/// # Ok::<(), ztensor::Error>(())
/// ```
pub struct Reader<R: Read + Seek> {
    reader: RefCell<R>,
    pub manifest: Manifest,
}

impl Reader<BufReader<File>> {
    /// Opens a zTensor file from a path using buffered I/O.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::Reader;
    ///
    /// let reader = Reader::open("model.zt")?;
    /// let weights: Vec<f32> = reader.read_as("weights")?;
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path)?;
        Self::new(BufReader::new(file))
    }

    /// Opens any zTensor file (v0.1.0 or v1.2) from path.
    pub fn open_any_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path)?;
        Self::open_any(BufReader::new(file))
    }
}

// Macro to generate view(), view_as(), and TensorReader for mmap-backed readers.
// Both Mmap (MAP_SHARED, read-only) and MmapMut (MAP_PRIVATE, COW) deref to [u8].
macro_rules! impl_mmap_reader {
    ($mmap_type:ty) => {
        impl Reader<Cursor<$mmap_type>> {
            /// Gets a zero-copy byte slice of an object's data.
            ///
            /// Only available for memory-mapped readers with dense, raw-encoded objects.
            /// For compressed objects, use [`read_as`](Reader::read_as) instead.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use ztensor::Reader;
            ///
            /// let reader = Reader::open_mmap("model.zt")?;
            /// let bytes: &[u8] = reader.view("weights")?;
            /// # Ok::<(), ztensor::Error>(())
            /// ```
            pub fn view(&self, name: &str) -> Result<&[u8], Error> {
                let obj = self
                    .manifest
                    .objects
                    .get(name)
                    .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;

                if obj.format != Format::Dense {
                    return Err(Error::TypeMismatch {
                        expected: Format::Dense.as_str().to_string(),
                        found: obj.format.as_str().to_string(),
                        context: format!("object '{}'", name),
                    });
                }

                let component = obj.components.get("data").ok_or_else(|| {
                    Error::InvalidFileStructure(format!(
                        "Dense object '{}' missing 'data' component",
                        name
                    ))
                })?;

                if component.encoding != Encoding::Raw {
                    return Err(Error::Other(format!(
                        "Zero-copy not supported for compressed component in '{}'",
                        name
                    )));
                }

                if component.offset % ALIGNMENT != 0 {
                    return Err(Error::InvalidAlignment {
                        offset: component.offset,
                        alignment: ALIGNMENT,
                    });
                }

                // SAFETY: Mmap data is pinned in memory by the OS. The RefCell only gates
                // cursor position mutations. The mmap lives as long as `self`, so this
                // slice is valid for the lifetime of `&self`.
                let borrow = self.reader.borrow();
                let mmap: &$mmap_type = borrow.get_ref();
                let start = component.offset as usize;
                let end = start + component.length as usize;

                if end > mmap.len() {
                    return Err(Error::InvalidFileStructure(format!(
                        "Component '{}' out of bounds (end={} > file_len={})",
                        name,
                        end,
                        mmap.len()
                    )));
                }

                // SAFETY: The mmap is owned by self (inside RefCell<Cursor<_>>).
                // Its memory address is stable (OS-mapped). We extend the lifetime
                // from the temporary Ref borrow to &self, which is safe because
                // the mmap cannot be moved or dropped while &self is alive.
                let slice = &mmap[start..end];
                Ok(unsafe { std::slice::from_raw_parts(slice.as_ptr(), slice.len()) })
            }

            /// Gets a zero-copy byte slice of a specific component of an object.
            ///
            /// Generalizes [`view`](Reader::view) to any component, not just
            /// the `"data"` component of dense objects.
            pub fn view_component(&self, name: &str, component_name: &str) -> Result<&[u8], Error> {
                let obj = self
                    .manifest
                    .objects
                    .get(name)
                    .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;

                let component = obj.components.get(component_name).ok_or_else(|| {
                    Error::InvalidFileStructure(format!(
                        "Object '{}' missing '{}' component",
                        name, component_name
                    ))
                })?;

                if component.encoding != Encoding::Raw {
                    return Err(Error::Other(format!(
                        "Zero-copy not supported for compressed component '{}' in '{}'",
                        component_name, name
                    )));
                }

                if component.offset % ALIGNMENT != 0 {
                    return Err(Error::InvalidAlignment {
                        offset: component.offset,
                        alignment: ALIGNMENT,
                    });
                }

                let borrow = self.reader.borrow();
                let mmap: &$mmap_type = borrow.get_ref();
                let start = component.offset as usize;
                let end = start + component.length as usize;

                if end > mmap.len() {
                    return Err(Error::InvalidFileStructure(format!(
                        "Component '{}/{}' out of bounds (end={} > file_len={})",
                        name,
                        component_name,
                        end,
                        mmap.len()
                    )));
                }

                let slice = &mmap[start..end];
                Ok(unsafe { std::slice::from_raw_parts(slice.as_ptr(), slice.len()) })
            }

            /// Gets a typed zero-copy slice of an object's data.
            ///
            /// Returns an error if the stored dtype doesn't match `T`.
            ///
            /// # Examples
            ///
            /// ```no_run
            /// use ztensor::Reader;
            ///
            /// let reader = Reader::open_mmap("model.zt")?;
            /// let weights: &[f32] = reader.view_as("weights")?;
            /// # Ok::<(), ztensor::Error>(())
            /// ```
            pub fn view_as<T: TensorElement>(&self, name: &str) -> Result<&[T], Error> {
                let dtype = self
                    .manifest
                    .objects
                    .get(name)
                    .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?
                    .data_dtype()?;
                if T::DTYPE != dtype {
                    return Err(Error::TypeMismatch {
                        expected: dtype.as_str().to_string(),
                        found: std::any::type_name::<T>().to_string(),
                        context: format!("object '{}'", name),
                    });
                }
                bytes_as_typed(self.view(name)?)
            }
        }

        impl TensorReader for Reader<Cursor<$mmap_type>> {
            fn manifest(&self) -> &Manifest {
                &self.manifest
            }

            fn read_data(&self, name: &str) -> Result<TensorData<'_>, Error> {
                match self.view(name) {
                    Ok(slice) => Ok(TensorData::Borrowed(slice)),
                    Err(_) => {
                        let data = self.read(name, true)?;
                        Ok(TensorData::Owned(data))
                    }
                }
            }

            fn read_component_data(
                &self,
                name: &str,
                component_name: &str,
            ) -> Result<TensorData<'_>, Error> {
                // Look up object and component first — real errors propagate.
                let obj = self
                    .manifest
                    .objects
                    .get(name)
                    .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;
                let comp = obj.components.get(component_name).ok_or_else(|| {
                    Error::InvalidFileStructure(format!(
                        "Missing '{}' component for '{}'",
                        component_name, name
                    ))
                })?;

                if comp.encoding == Encoding::Raw {
                    // Try zero-copy for raw components
                    let slice = self.view_component(name, component_name)?;
                    Ok(TensorData::Borrowed(slice))
                } else {
                    // Compressed: must read + decompress
                    let data = Self::read_component_impl(&mut *self.reader.borrow_mut(), comp)?;
                    Ok(TensorData::Owned(data))
                }
            }
        }
    };
}

impl_mmap_reader!(MmapMut);
impl_mmap_reader!(Mmap);

impl Reader<Cursor<MmapMut>> {
    /// Opens a zTensor file using memory mapping with MAP_PRIVATE (copy-on-write).
    ///
    /// The returned reader supports zero-copy [`view`](Reader::view) and
    /// [`view_as`](Reader::view_as). Writes to the mapped memory trigger
    /// copy-on-write at the OS level and never modify the file.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::Reader;
    ///
    /// let reader = Reader::open_mmap("model.zt")?;
    /// let weights: &[f32] = reader.view_as("weights")?;
    /// println!("first element: {}", weights[0]);
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map_copy(&file)? };
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential).ok();
        Self::new(Cursor::new(mmap))
    }

    /// Opens any zTensor file (v0.1.0 or v1.2) with MAP_PRIVATE (COW).
    pub fn open_mmap_any(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map_copy(&file)? };
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential).ok();
        Self::open_any(Cursor::new(mmap))
    }
}

impl Reader<Cursor<Mmap>> {
    /// Opens a v1.2 zTensor file using read-only memory mapping (MAP_SHARED).
    ///
    /// Faster than MAP_PRIVATE for read-only access (no COW page table overhead).
    pub fn open_mmap_shared(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential).ok();
        Self::new(Cursor::new(mmap))
    }

    /// Opens any zTensor file (v0.1.0 or v1.2) with read-only MAP_SHARED.
    pub fn open_mmap_shared_any(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential).ok();
        Self::open_any(Cursor::new(mmap))
    }
}

impl<R: Read + Seek> Reader<R> {
    /// Creates a reader for a v1.2 zTensor file.
    pub fn new(mut reader: R) -> Result<Self, Error> {
        let mut header_magic = [0u8; 8];
        reader.read_exact(&mut header_magic)?;
        if header_magic != *MAGIC {
            return Err(Error::InvalidMagicNumber {
                found: header_magic.to_vec(),
            });
        }

        reader.seek(SeekFrom::End(-16))?;
        let mut size_buf = [0u8; 8];
        reader.read_exact(&mut size_buf)?;
        let manifest_size = u64::from_le_bytes(size_buf);

        let mut footer_magic = [0u8; 8];
        reader.read_exact(&mut footer_magic)?;
        if footer_magic != *MAGIC {
            return Err(Error::InvalidMagicNumber {
                found: footer_magic.to_vec(),
            });
        }

        if manifest_size > MAX_MANIFEST_SIZE {
            return Err(Error::ManifestTooLarge {
                size: manifest_size,
            });
        }

        reader.seek(SeekFrom::End(-16 - manifest_size as i64))?;
        let mut cbor_buf = vec![0u8; manifest_size as usize];
        reader.read_exact(&mut cbor_buf)?;

        let manifest: Manifest = ciborium::from_reader(std::io::Cursor::new(&cbor_buf))
            .map_err(Error::CborDeserialize)?;

        Ok(Self {
            reader: RefCell::new(reader),
            manifest,
        })
    }

    /// Creates a reader for a legacy v0.1.0 zTensor file.
    fn new_legacy(mut reader: R) -> Result<Self, Error> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if magic != *MAGIC_V01 {
            return Err(Error::InvalidMagicNumber {
                found: magic.to_vec(),
            });
        }

        reader.seek(SeekFrom::End(-8))?;
        let mut size_buf = [0u8; 8];
        reader.read_exact(&mut size_buf)?;
        let cbor_size = u64::from_le_bytes(size_buf);

        if cbor_size > MAX_MANIFEST_SIZE {
            return Err(Error::ManifestTooLarge { size: cbor_size });
        }

        reader.seek(SeekFrom::End(-8 - cbor_size as i64))?;
        let mut cbor_buf = vec![0u8; cbor_size as usize];
        reader.read_exact(&mut cbor_buf)?;

        let tensors: Vec<LegacyTensorMeta> = ciborium::from_reader(std::io::Cursor::new(&cbor_buf))
            .map_err(Error::CborDeserialize)?;

        let mut objects = BTreeMap::new();
        for t in tensors {
            if !t.layout.is_empty() && t.layout != "dense" {
                continue;
            }

            let dtype = t.to_dtype()?;
            let encoding = t.to_encoding();

            let mut obj = Object::dense(t.shape, dtype, t.offset, t.size);
            let component = obj.components.get_mut("data").unwrap();
            component.encoding = encoding;
            component.digest = t.checksum.clone();

            objects.insert(t.name, obj);
        }

        let manifest = Manifest {
            version: "0.1.0".to_string(),
            attributes: None,
            objects,
        };

        Ok(Self {
            reader: RefCell::new(reader),
            manifest,
        })
    }

    /// Auto-detects file version and opens accordingly.
    pub fn open_any(mut reader: R) -> Result<Self, Error> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        reader.seek(SeekFrom::Start(0))?;

        if magic == *MAGIC_V01 {
            Self::new_legacy(reader)
        } else {
            Self::new(reader)
        }
    }

    /// Lists all objects in the file.
    pub fn tensors(&self) -> &BTreeMap<String, Object> {
        &self.manifest.objects
    }

    /// Gets metadata for an object by name.
    pub fn get(&self, name: &str) -> Option<&Object> {
        self.manifest.objects.get(name)
    }

    // =========================================================================
    // COMPONENT READING (free functions taking &mut R to avoid borrow conflicts)
    // =========================================================================

    fn read_component_into_impl(
        reader: &mut R,
        component: &Component,
        dst: &mut [u8],
        ctx: &ReadContext,
        verify_checksum: bool,
    ) -> Result<(), Error> {
        if !component.offset.is_multiple_of(ALIGNMENT) {
            return Err(Error::InvalidAlignment {
                offset: component.offset,
                alignment: ALIGNMENT,
            });
        }

        reader.seek(SeekFrom::Start(component.offset))?;

        match component.encoding {
            Encoding::Zstd => {
                let mut compressed = vec![0u8; component.length as usize];
                reader.read_exact(&mut compressed)?;

                if verify_checksum {
                    if let Some(ref digest) = component.digest {
                        Self::verify_checksum(digest, &compressed, ctx)?;
                    }
                }

                zstd::stream::copy_decode(Cursor::new(compressed), &mut *dst)
                    .map_err(Error::ZstdDecompression)?;
            }
            Encoding::Raw => {
                if dst.len() as u64 != component.length {
                    return Err(Error::InvalidFileStructure(format!(
                        "Component length mismatch for {}/{}: expected {}, got {}",
                        ctx.object_name,
                        ctx.component_name,
                        component.length,
                        dst.len()
                    )));
                }
                reader.read_exact(dst)?;

                if verify_checksum {
                    if let Some(ref digest) = component.digest {
                        Self::verify_checksum(digest, dst, ctx)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn read_component_impl(reader: &mut R, component: &Component) -> Result<Vec<u8>, Error> {
        let ctx = ReadContext::unknown();

        match component.encoding {
            Encoding::Zstd => {
                reader.seek(SeekFrom::Start(component.offset))?;
                let mut compressed = vec![0u8; component.length as usize];
                reader.read_exact(&mut compressed)?;

                if let Some(ref digest) = component.digest {
                    Self::verify_checksum(digest, &compressed, &ctx)?;
                }

                let mut decompressed = if let Some(uc_len) = component.uncompressed_length {
                    if uc_len > MAX_OBJECT_SIZE {
                        return Err(Error::ObjectTooLarge {
                            size: uc_len,
                            limit: MAX_OBJECT_SIZE,
                        });
                    }
                    Vec::with_capacity(uc_len as usize)
                } else {
                    Vec::new()
                };
                zstd::stream::copy_decode(Cursor::new(compressed), &mut decompressed)
                    .map_err(Error::ZstdDecompression)?;

                if decompressed.len() as u64 > MAX_OBJECT_SIZE {
                    return Err(Error::ObjectTooLarge {
                        size: decompressed.len() as u64,
                        limit: MAX_OBJECT_SIZE,
                    });
                }

                if let Some(uc_len) = component.uncompressed_length {
                    if decompressed.len() as u64 != uc_len {
                        return Err(Error::InvalidFileStructure(format!(
                            "Decompressed size {} != declared uncompressed_length {}",
                            decompressed.len(),
                            uc_len
                        )));
                    }
                }
                Ok(decompressed)
            }
            Encoding::Raw => {
                let mut data = vec![0u8; component.length as usize];
                Self::read_component_into_impl(reader, component, &mut data, &ctx, true)?;
                Ok(data)
            }
        }
    }

    /// Reads raw component data.
    pub fn read_component(&self, component: &Component) -> Result<Vec<u8>, Error> {
        Self::read_component_impl(&mut *self.reader.borrow_mut(), component)
    }

    fn verify_checksum(digest: &str, data: &[u8], ctx: &ReadContext) -> Result<(), Error> {
        if digest.starts_with("crc32c:0x") || digest.starts_with("crc32c:0X") {
            let expected_hex = &digest[9..];
            let expected = u32::from_str_radix(expected_hex, 16).map_err(|_| {
                Error::ChecksumFormatError(format!("Invalid CRC32C hex: {}", expected_hex))
            })?;
            let calculated = crc32c::crc32c(data);
            if calculated != expected {
                return Err(Error::ChecksumMismatch {
                    object_name: ctx.object_name.to_string(),
                    component_name: ctx.component_name.to_string(),
                    expected: format!("0x{:08X}", expected),
                    calculated: format!("0x{:08X}", calculated),
                });
            }
        } else if let Some(expected_hex) = digest.strip_prefix("sha256:") {
            let calculated = crate::utils::sha256_hex(data);
            if calculated != expected_hex.to_lowercase() {
                return Err(Error::ChecksumMismatch {
                    object_name: ctx.object_name.to_string(),
                    component_name: ctx.component_name.to_string(),
                    expected: expected_hex.to_string(),
                    calculated,
                });
            }
        }
        Ok(())
    }

    // =========================================================================
    // DENSE OBJECT READING
    // =========================================================================

    /// Reads raw byte data of a dense object.
    pub fn read(&self, name: &str, verify_checksum: bool) -> Result<Vec<u8>, Error> {
        let obj = self
            .manifest
            .objects
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;

        if obj.format != Format::Dense {
            return Err(Error::TypeMismatch {
                expected: Format::Dense.as_str().to_string(),
                found: obj.format.as_str().to_string(),
                context: format!("object '{}'", name),
            });
        }

        let component = obj.components.get("data").ok_or_else(|| {
            Error::InvalidFileStructure(format!("Dense object '{}' missing 'data' component", name))
        })?;

        let num_elements = obj.num_elements()?;
        let expected_size = num_elements * component.dtype.byte_size() as u64;

        if expected_size > MAX_OBJECT_SIZE {
            return Err(Error::ObjectTooLarge {
                size: expected_size,
                limit: MAX_OBJECT_SIZE,
            });
        }

        let dtype = component.dtype;
        let component = component.clone();

        let mut data = vec![0u8; expected_size as usize];
        let ctx = ReadContext::new(name, "data");
        Self::read_component_into_impl(
            &mut *self.reader.borrow_mut(),
            &component,
            &mut data,
            &ctx,
            verify_checksum,
        )?;

        if cfg!(target_endian = "big") && dtype.is_multi_byte() {
            swap_endianness_in_place(&mut data, dtype.byte_size());
        }

        Ok(data)
    }

    /// Reads multiple objects in batch.
    pub fn read_many(&self, names: &[&str], verify_checksum: bool) -> Result<Vec<Vec<u8>>, Error> {
        let mut results = Vec::with_capacity(names.len());
        for name in names {
            results.push(self.read(name, verify_checksum)?);
        }
        Ok(results)
    }

    // =========================================================================
    // TYPED READING
    // =========================================================================

    /// Reads object data as a typed vector.
    ///
    /// Returns an error if the stored dtype doesn't match `T`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::Reader;
    ///
    /// let reader = Reader::open("model.zt")?;
    /// let weights: Vec<f32> = reader.read_as("weights")?;
    /// let ids: Vec<i64> = reader.read_as("token_ids")?;
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn read_as<T: TensorElement>(&self, name: &str) -> Result<Vec<T>, Error> {
        let obj = self
            .manifest
            .objects
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;

        let component = obj.components.get("data").ok_or_else(|| {
            Error::InvalidFileStructure(format!("Missing 'data' component for {}", name))
        })?;

        if T::DTYPE != component.dtype {
            return Err(Error::TypeMismatch {
                expected: component.dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }

        if obj.format != Format::Dense {
            return Err(Error::TypeMismatch {
                expected: Format::Dense.as_str().to_string(),
                found: obj.format.as_str().to_string(),
                context: name.to_string(),
            });
        }

        let component = component.clone();
        let num_elements = obj.num_elements()? as usize;

        let byte_len = num_elements * T::SIZE;

        if byte_len as u64 > MAX_OBJECT_SIZE {
            return Err(Error::ObjectTooLarge {
                size: byte_len as u64,
                limit: MAX_OBJECT_SIZE,
            });
        }

        let mut typed_data = vec![T::default(); num_elements];
        let output_slice =
            unsafe { std::slice::from_raw_parts_mut(typed_data.as_mut_ptr() as *mut u8, byte_len) };

        let ctx = ReadContext::new(name, "data");
        Self::read_component_into_impl(
            &mut *self.reader.borrow_mut(),
            &component,
            output_slice,
            &ctx,
            true,
        )?;

        Ok(typed_data)
    }

    // =========================================================================
    // SPARSE READING
    // =========================================================================

    fn read_component_as<T: TensorElement>(
        reader: &mut R,
        component: &Component,
        ctx: &ReadContext,
    ) -> Result<Vec<T>, Error> {
        match component.encoding {
            Encoding::Zstd => {
                let bytes = Self::read_component_impl(reader, component)?;
                if bytes.len() as u64 > MAX_OBJECT_SIZE {
                    return Err(Error::ObjectTooLarge {
                        size: bytes.len() as u64,
                        limit: MAX_OBJECT_SIZE,
                    });
                }
                if bytes.len() % T::SIZE != 0 {
                    return Err(Error::InvalidFileStructure(format!(
                        "Decompressed byte length {} is not a multiple of type size {}",
                        bytes.len(),
                        T::SIZE
                    )));
                }
                let num_elements = bytes.len() / T::SIZE;
                let mut values = vec![T::default(); num_elements];
                unsafe {
                    std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, bytes.len())
                        .copy_from_slice(&bytes);
                }
                Ok(values)
            }
            Encoding::Raw => {
                if component.length > MAX_OBJECT_SIZE {
                    return Err(Error::ObjectTooLarge {
                        size: component.length,
                        limit: MAX_OBJECT_SIZE,
                    });
                }
                if !(component.length as usize).is_multiple_of(T::SIZE) {
                    return Err(Error::InvalidFileStructure(format!(
                        "Component byte length {} is not a multiple of type size {}",
                        component.length,
                        T::SIZE
                    )));
                }
                let num_elements = component.length as usize / T::SIZE;
                let mut values = vec![T::default(); num_elements];
                let byte_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        values.as_mut_ptr() as *mut u8,
                        component.length as usize,
                    )
                };
                Self::read_component_into_impl(reader, component, byte_slice, ctx, true)?;
                Ok(values)
            }
        }
    }

    fn read_u64_component(
        reader: &mut R,
        component: &Component,
        ctx: &ReadContext,
    ) -> Result<Vec<u64>, Error> {
        let bytes = match component.encoding {
            Encoding::Zstd => Self::read_component_impl(reader, component)?,
            Encoding::Raw => {
                if component.length > MAX_OBJECT_SIZE {
                    return Err(Error::ObjectTooLarge {
                        size: component.length,
                        limit: MAX_OBJECT_SIZE,
                    });
                }
                let mut buf = vec![0u8; component.length as usize];
                Self::read_component_into_impl(reader, component, &mut buf, ctx, true)?;
                buf
            }
        };

        if bytes.len() % 8 != 0 {
            return Err(Error::InvalidFileStructure(
                "Index component length not aligned to 8 bytes".into(),
            ));
        }

        Ok(bytes
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    /// Reads a COO sparse object as a [`CooTensor`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::Reader;
    ///
    /// let reader = Reader::open("sparse.zt")?;
    /// let coo = reader.read_coo::<f32>("matrix")?;
    /// println!("shape: {:?}, nnz: {}", coo.shape, coo.values.len());
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn read_coo<T: TensorElement>(&self, name: &str) -> Result<CooTensor<T>, Error> {
        let obj = self
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;

        if obj.format != Format::SparseCoo {
            return Err(Error::TypeMismatch {
                expected: Format::SparseCoo.as_str().to_string(),
                found: obj.format.as_str().to_string(),
                context: format!("object '{}'", name),
            });
        }

        let shape = obj.shape.clone();
        let val_comp = obj
            .components
            .get("values")
            .ok_or(Error::InvalidFileStructure("Missing 'values'".to_string()))?
            .clone();
        let coords_comp = obj
            .components
            .get("coords")
            .ok_or(Error::InvalidFileStructure("Missing 'coords'".to_string()))?
            .clone();

        let val_ctx = ReadContext::new(name, "values");
        let mut values: Vec<T> =
            Self::read_component_as(&mut *self.reader.borrow_mut(), &val_comp, &val_ctx)?;

        if cfg!(target_endian = "big") && val_comp.dtype.is_multi_byte() {
            let byte_len = values.len() * T::SIZE;
            let val_slice =
                unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len) };
            swap_endianness_in_place(val_slice, val_comp.dtype.byte_size());
        }

        let coords_ctx = ReadContext::new(name, "coords");
        let all_coords =
            Self::read_u64_component(&mut *self.reader.borrow_mut(), &coords_comp, &coords_ctx)?;

        let nnz = values.len();
        let ndim = shape.len();

        if all_coords.len() != nnz * ndim {
            return Err(Error::DataConversionError(
                "COO coords size mismatch".to_string(),
            ));
        }

        let mut indices = Vec::with_capacity(nnz);
        for i in 0..nnz {
            let mut idx = Vec::with_capacity(ndim);
            for d in 0..ndim {
                idx.push(all_coords[d * nnz + i]);
            }
            indices.push(idx);
        }

        Ok(CooTensor {
            shape,
            indices,
            values,
        })
    }

    /// Reads a CSR sparse object as a [`CsrTensor`].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ztensor::Reader;
    ///
    /// let reader = Reader::open("sparse.zt")?;
    /// let csr = reader.read_csr::<f32>("matrix")?;
    /// println!("shape: {:?}, nnz: {}", csr.shape, csr.values.len());
    /// # Ok::<(), ztensor::Error>(())
    /// ```
    pub fn read_csr<T: TensorElement>(&self, name: &str) -> Result<CsrTensor<T>, Error> {
        let obj = self
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;

        if obj.format != Format::SparseCsr {
            return Err(Error::TypeMismatch {
                expected: Format::SparseCsr.as_str().to_string(),
                found: obj.format.as_str().to_string(),
                context: format!("object '{}'", name),
            });
        }

        let shape = obj.shape.clone();
        let val_comp = obj
            .components
            .get("values")
            .ok_or(Error::InvalidFileStructure("Missing 'values'".to_string()))?
            .clone();
        let idx_comp = obj
            .components
            .get("indices")
            .ok_or(Error::InvalidFileStructure("Missing 'indices'".to_string()))?
            .clone();
        let ptr_comp = obj
            .components
            .get("indptr")
            .ok_or(Error::InvalidFileStructure("Missing 'indptr'".to_string()))?
            .clone();

        let val_ctx = ReadContext::new(name, "values");
        let mut values: Vec<T> =
            Self::read_component_as(&mut *self.reader.borrow_mut(), &val_comp, &val_ctx)?;

        if cfg!(target_endian = "big") && val_comp.dtype.is_multi_byte() {
            let byte_len = values.len() * T::SIZE;
            let val_slice =
                unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, byte_len) };
            swap_endianness_in_place(val_slice, val_comp.dtype.byte_size());
        }

        let idx_ctx = ReadContext::new(name, "indices");
        let indices =
            Self::read_u64_component(&mut *self.reader.borrow_mut(), &idx_comp, &idx_ctx)?;

        let ptr_ctx = ReadContext::new(name, "indptr");
        let indptr = Self::read_u64_component(&mut *self.reader.borrow_mut(), &ptr_comp, &ptr_ctx)?;

        Ok(CsrTensor {
            shape,
            indptr,
            indices,
            values,
        })
    }
}

// =========================================================================
// TensorReader implementation for Reader
// =========================================================================

// TensorReader for Reader<Cursor<MmapMut>> and Reader<Cursor<Mmap>>
// are generated by impl_mmap_reader! macro above.

impl TensorReader for Reader<BufReader<File>> {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read_data(&self, name: &str) -> Result<TensorData<'_>, Error> {
        let data = self.read(name, true)?;
        Ok(TensorData::Owned(data))
    }

    fn read_component_data(
        &self,
        name: &str,
        component_name: &str,
    ) -> Result<TensorData<'_>, Error> {
        let obj = self
            .manifest
            .objects
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;
        let comp = obj.components.get(component_name).ok_or_else(|| {
            Error::InvalidFileStructure(format!(
                "Missing '{}' component for '{}'",
                component_name, name
            ))
        })?;
        let data = Self::read_component_impl(&mut *self.reader.borrow_mut(), comp)?;
        Ok(TensorData::Owned(data))
    }
}
