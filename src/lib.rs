//! # ztensor
//!
//! High-performance tensor serialization with support for compression,
//! checksums, sparse tensors, and multiple file formats.
//!
//! ## Writing
//!
//! ```no_run
//! use ztensor::{Writer, Compression, Checksum};
//!
//! let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let mut writer = Writer::create("model.zt")?;
//!
//! // Simple: dtype inferred, no compression
//! writer.add("weights", &[2, 3], &data)?;
//!
//! // Builder: with compression and checksum
//! writer.add_with("biases", &[6], &data)
//!     .compress(Compression::Zstd(3))
//!     .checksum(Checksum::Crc32c)
//!     .write()?;
//!
//! writer.finish()?;
//! # Ok::<(), ztensor::Error>(())
//! ```
//!
//! ## Reading
//!
//! ```no_run
//! use ztensor::Reader;
//!
//! let reader = Reader::open("model.zt")?;
//! let weights: Vec<f32> = reader.read_as("weights")?;
//! # Ok::<(), ztensor::Error>(())
//! ```
//!
//! ## Memory-mapped access
//!
//! ```no_run
//! use ztensor::Reader;
//!
//! let reader = Reader::open_mmap("model.zt")?;
//! let weights: &[f32] = reader.view_as("weights")?;
//! # Ok::<(), ztensor::Error>(())
//! ```
//!
//! ## Multi-format reading
//!
//! The [`open`] function auto-detects format by extension and returns a
//! [`TensorReader`] trait object:
//!
//! ```no_run
//! let reader = ztensor::open("model.safetensors")?;
//! for name in reader.keys() {
//!     println!("{}", name);
//! }
//! # Ok::<(), ztensor::Error>(())
//! ```

pub mod compat;
pub mod error;
#[cfg(feature = "cffi")]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub mod ffi;
pub mod models;
pub mod reader;
pub mod utils;
pub mod writer;

#[cfg(feature = "safetensors")]
pub mod safetensors_reader;

#[cfg(feature = "pickle")]
#[doc(hidden)]
pub mod pickle_vm;
#[cfg(feature = "pickle")]
pub mod pytorch_reader;

#[cfg(feature = "gguf")]
pub mod gguf_reader;

#[cfg(feature = "npz")]
pub mod npz_reader;

#[cfg(feature = "onnx")]
pub mod onnx_reader;

#[cfg(feature = "hdf5")]
pub mod hdf5_reader;

#[cfg(feature = "python")]
mod pylib;

pub use compat::{is_legacy_file, is_legacy_format, LegacyReader};
pub use error::Error;
pub use models::{Checksum, Component, DType, Encoding, Format, Manifest, Object};
pub use reader::{Reader, TensorData, TensorElement, TensorReader};
pub use writer::{Compression, Writer};

#[cfg(feature = "safetensors")]
pub use safetensors_reader::SafeTensorsReader;

#[cfg(feature = "pickle")]
pub use pytorch_reader::PyTorchReader;

#[cfg(feature = "gguf")]
pub use gguf_reader::GgufReader;

#[cfg(feature = "npz")]
pub use npz_reader::NpzReader;

#[cfg(feature = "onnx")]
pub use onnx_reader::OnnxReader;

#[cfg(feature = "hdf5")]
pub use hdf5_reader::Hdf5Reader;

/// Opens a tensor file, automatically detecting the format by file extension.
///
/// Supported formats:
/// - `.safetensors` — SafeTensors format (requires `safetensors` feature)
/// - `.pt`, `.bin`, `.pth`, `.pkl` — PyTorch pickle format (requires `pickle` feature)
/// - `.gguf` — GGUF format (requires `gguf` feature)
/// - `.npz` — NumPy format (requires `npz` feature)
/// - `.onnx` — ONNX format (requires `onnx` feature)
/// - `.h5`, `.hdf5` — HDF5 format (requires `hdf5` feature)
/// - All other extensions — zTensor format (`.zt` or any)
///
/// Returns a trait object for format-agnostic access to tensor data.
///
/// # Examples
///
/// ```no_run
/// use ztensor::TensorReader;
///
/// let reader = ztensor::open("model.safetensors")?;
/// for name in reader.keys() {
///     let data = reader.read_data(name)?;
///     println!("{}: {} bytes", name, data.as_slice().len());
/// }
/// # Ok::<(), ztensor::Error>(())
/// ```
pub fn open(path: impl AsRef<std::path::Path>) -> Result<Box<dyn TensorReader + Send>, Error> {
    let path = path.as_ref();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        #[cfg(feature = "safetensors")]
        "safetensors" => Ok(Box::new(SafeTensorsReader::open(path)?)),

        #[cfg(not(feature = "safetensors"))]
        "safetensors" => Err(Error::Other(
            "SafeTensors support requires the 'safetensors' feature".to_string(),
        )),

        #[cfg(feature = "pickle")]
        "pt" | "bin" | "pth" | "pkl" => Ok(Box::new(PyTorchReader::open(path)?)),

        #[cfg(not(feature = "pickle"))]
        "pt" | "bin" | "pth" | "pkl" => Err(Error::Other(
            "PyTorch pickle support requires the 'pickle' feature".to_string(),
        )),

        #[cfg(feature = "gguf")]
        "gguf" => Ok(Box::new(GgufReader::open(path)?)),

        #[cfg(not(feature = "gguf"))]
        "gguf" => Err(Error::Other(
            "GGUF support requires the 'gguf' feature".to_string(),
        )),

        #[cfg(feature = "npz")]
        "npz" => Ok(Box::new(NpzReader::open(path)?)),

        #[cfg(not(feature = "npz"))]
        "npz" => Err(Error::Other(
            "NPZ support requires the 'npz' feature".to_string(),
        )),

        #[cfg(feature = "onnx")]
        "onnx" => Ok(Box::new(OnnxReader::open(path)?)),

        #[cfg(not(feature = "onnx"))]
        "onnx" => Err(Error::Other(
            "ONNX support requires the 'onnx' feature".to_string(),
        )),

        #[cfg(feature = "hdf5")]
        "h5" | "hdf5" => Ok(Box::new(Hdf5Reader::open(path)?)),

        #[cfg(not(feature = "hdf5"))]
        "h5" | "hdf5" => Err(Error::Other(
            "HDF5 support requires the 'hdf5' feature".to_string(),
        )),

        _ => Ok(Box::new(Reader::open_mmap_shared_any(path)?)),
    }
}

/// Removes tensors by name from a `.zt` file, writing the result to a new file.
///
/// Reads all tensors from `input`, skips those whose names appear in `names`,
/// and writes the remaining tensors to `output`. Preserves compression settings.
///
/// Returns an error if any name in `names` is not found in the input file.
///
/// # Examples
///
/// ```no_run
/// ztensor::remove_tensors("model.zt", "model_trimmed.zt", &["unused_layer"])?;
/// # Ok::<(), ztensor::Error>(())
/// ```
pub fn remove_tensors(
    input: impl AsRef<std::path::Path>,
    output: impl AsRef<std::path::Path>,
    names: &[&str],
) -> Result<(), Error> {
    let reader = Reader::open(input)?;
    let objects = reader.manifest.objects.clone();

    // Validate all names exist
    for name in names {
        if !objects.contains_key(*name) {
            return Err(Error::ObjectNotFound(name.to_string()));
        }
    }

    let names_set: std::collections::HashSet<&str> = names.iter().copied().collect();
    let mut writer = Writer::create(output)?;

    // Copy global attributes
    if let Some(attrs) = &reader.manifest.attributes {
        writer.set_attributes(attrs.clone());
    }

    for (name, obj) in &objects {
        if names_set.contains(name.as_str()) {
            continue;
        }

        if obj.format != Format::Dense {
            return Err(Error::Other(format!(
                "Cannot copy tensor '{}' with unsupported format '{}' (only dense tensors supported)",
                name,
                obj.format.as_str()
            )));
        }

        let data_comp = obj.components.get("data").ok_or_else(|| {
            Error::InvalidFileStructure(format!("Missing 'data' component for '{}'", name))
        })?;

        let data = reader.read(name, true)?;

        let compression = match data_comp.encoding {
            Encoding::Zstd => writer::Compression::Zstd(3),
            Encoding::Raw => writer::Compression::Raw,
        };

        let checksum = match &data_comp.digest {
            Some(d) if d.starts_with("sha256:") => Checksum::Sha256,
            Some(d) if d.starts_with("crc32c:") => Checksum::Crc32c,
            _ => Checksum::None,
        };

        writer.add_bytes(
            name,
            obj.shape.clone(),
            data_comp.dtype,
            compression,
            &data,
            checksum,
        )?;

        // Preserve per-object attributes
        if obj.attributes.is_some() {
            if let Some(written_obj) = writer.manifest_mut().objects.get_mut(name) {
                written_obj.attributes = obj.attributes.clone();
            }
        }
    }

    writer.finish()?;
    Ok(())
}

/// Replaces the data of a dense tensor in-place within an existing `.zt` file.
///
/// The replacement data must have the exact same byte size as the original.
/// Only raw (uncompressed) dense tensors can be replaced in-place.
/// If the tensor has a checksum, it is recomputed automatically.
///
/// This is much faster than a full rewrite for large files because only
/// the tensor's data region and the manifest are updated — all other
/// tensor data remains untouched on disk.
///
/// # Errors
///
/// Returns an error if:
/// - The tensor is not found
/// - The tensor is not dense or not raw-encoded
/// - The replacement data has a different byte size
///
/// # Examples
///
/// ```no_run
/// let new_data = vec![0u8; 1024];
/// ztensor::replace_tensor("model.zt", "weights", &new_data)?;
/// # Ok::<(), ztensor::Error>(())
/// ```
pub fn replace_tensor(
    path: impl AsRef<std::path::Path>,
    name: &str,
    data: &[u8],
) -> Result<(), Error> {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    let path = path.as_ref();
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;

    // Read footer: last 16 bytes = [manifest_size: u64 LE] [MAGIC: 8B]
    file.seek(SeekFrom::End(-16))?;
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let manifest_size = u64::from_le_bytes(size_buf);

    let mut footer_magic = [0u8; 8];
    file.read_exact(&mut footer_magic)?;
    if footer_magic != *models::MAGIC {
        return Err(Error::InvalidMagicNumber {
            found: footer_magic.to_vec(),
        });
    }

    // Read manifest
    let file_size = file.seek(SeekFrom::End(0))?;
    let manifest_start = file_size - 16 - manifest_size;
    file.seek(SeekFrom::Start(manifest_start))?;
    let mut cbor_buf = vec![0u8; manifest_size as usize];
    file.read_exact(&mut cbor_buf)?;

    let mut manifest: Manifest =
        ciborium::from_reader(std::io::Cursor::new(&cbor_buf)).map_err(Error::CborDeserialize)?;

    // Validate tensor exists and is replaceable
    let obj = manifest
        .objects
        .get(name)
        .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;

    if obj.format != Format::Dense {
        return Err(Error::Other(format!(
            "Cannot replace tensor '{}': only dense tensors supported (found '{}')",
            name,
            obj.format.as_str()
        )));
    }

    let component = obj.components.get("data").ok_or_else(|| {
        Error::InvalidFileStructure(format!("Dense object '{}' missing 'data' component", name))
    })?;

    if component.encoding != Encoding::Raw {
        return Err(Error::Other(format!(
            "Cannot replace tensor '{}': only raw (uncompressed) tensors supported",
            name
        )));
    }

    if data.len() as u64 != component.length {
        return Err(Error::InconsistentDataSize {
            expected: component.length,
            found: data.len() as u64,
        });
    }

    // Overwrite data at the component offset
    file.seek(SeekFrom::Start(component.offset))?;
    file.write_all(data)?;

    // Recompute checksum if one exists
    let new_digest = if let Some(ref digest) = component.digest {
        if digest.starts_with("crc32c:") {
            Some(format!("crc32c:0x{:08X}", crc32c::crc32c(data)))
        } else if digest.starts_with("sha256:") {
            Some(format!("sha256:{}", utils::sha256_hex(data)))
        } else {
            component.digest.clone()
        }
    } else {
        None
    };

    // Update manifest with new digest
    let obj_mut = manifest.objects.get_mut(name).unwrap();
    let comp_mut = obj_mut.components.get_mut("data").unwrap();
    comp_mut.digest = new_digest;

    // Rewrite manifest + footer
    let mut cbor = Vec::new();
    ciborium::into_writer(&manifest, &mut cbor).map_err(Error::CborSerialize)?;

    file.seek(SeekFrom::Start(manifest_start))?;
    file.write_all(&cbor)?;
    let cbor_size = cbor.len() as u64;
    file.write_all(&cbor_size.to_le_bytes())?;
    file.write_all(models::MAGIC)?;

    // Truncate to new size (manifest size may differ due to changed digest)
    let new_file_size = manifest_start + cbor_size + 16;
    file.set_len(new_file_size)?;
    file.flush()?;

    Ok(())
}

#[cfg(test)]

mod tests {
    use super::*;
    use crate::models::MAGIC;
    use crate::writer::Compression;
    use half::{bf16, f16};
    use std::io::{Cursor, Read, Seek, SeekFrom};

    #[test]
    fn test_write_read_empty() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = Writer::new(&mut buffer)?;
        let total_size = writer.finish()?;

        // MAGIC(8) + CBOR + SIZE(8) + MAGIC(8)
        assert!(total_size > 24);

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;
        assert!(reader.tensors().is_empty());

        // Verify header magic
        buffer.seek(SeekFrom::Start(0))?;
        let mut magic = [0u8; 8];
        buffer.read_exact(&mut magic)?;
        assert_eq!(&magic, MAGIC);

        // Verify footer magic
        buffer.seek(SeekFrom::End(-8))?;
        buffer.read_exact(&mut magic)?;
        assert_eq!(&magic, MAGIC);

        Ok(())
    }

    #[test]
    fn test_dense_f32_roundtrip() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        // Use typed API directly
        writer.add("test", &[2, 2], &data)?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        assert_eq!(reader.tensors().len(), 1);
        let retrieved_bytes = reader.read("test", true)?;
        let retrieved_floats: Vec<f32> = retrieved_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();

        assert_eq!(retrieved_floats, data);

        Ok(())
    }

    #[test]
    fn test_typed_reading() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let f32_data: Vec<f32> = vec![1.0, 2.5, -3.0, 4.25];
        writer.add("f32_obj", &[4], &f32_data)?;

        let u16_data: Vec<u16> = vec![10, 20, 30000, 65535];
        writer.add("u16_obj", &[2, 2], &u16_data)?;

        writer.finish()?;
        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let r1: Vec<f32> = reader.read_as("f32_obj")?;
        assert_eq!(r1, f32_data);

        let r2: Vec<u16> = reader.read_as("u16_obj")?;
        assert_eq!(r2, u16_data);

        // Type mismatch test
        match reader.read_as::<i32>("f32_obj") {
            Err(Error::TypeMismatch { .. }) => {}
            _ => panic!("Expected TypeMismatch error"),
        }

        Ok(())
    }

    #[test]
    fn test_compression_roundtrip() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
        writer
            .add_with("compressed", &[1000], &data)
            .compress(Compression::Zstd(0))
            .checksum(Checksum::Crc32c)
            .write()?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let obj = reader.get("compressed").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert_eq!(comp.encoding, Encoding::Zstd);
        assert!(comp.digest.is_some());

        let retrieved: Vec<f32> = reader.read_as("compressed")?;
        assert_eq!(retrieved, data);

        Ok(())
    }

    #[test]
    fn test_crc32c_checksum() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let data: Vec<u8> = (0..=20).collect();
        writer
            .add_with("checksummed", &[data.len() as u64], &data)
            .checksum(Checksum::Crc32c)
            .write()?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let obj = reader.get("checksummed").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert!(comp.digest.as_ref().unwrap().starts_with("crc32c:0x"));
        let offset = comp.offset;

        let retrieved = reader.read("checksummed", true)?;
        assert_eq!(retrieved, data);

        // Corrupt data and verify checksum fails
        drop(reader);

        buffer.seek(SeekFrom::Start(0))?;
        let mut file_bytes = Vec::new();
        buffer.read_to_end(&mut file_bytes)?;

        if file_bytes.len() > offset as usize {
            file_bytes[offset as usize] = file_bytes[offset as usize].wrapping_add(1);
        }

        let mut corrupted = Cursor::new(file_bytes);
        let corrupted_reader = Reader::new(&mut corrupted)?;

        match corrupted_reader.read("checksummed", true) {
            Err(Error::ChecksumMismatch { .. }) => {}
            Ok(_) => panic!("Expected ChecksumMismatch"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }

        Ok(())
    }

    #[test]
    fn test_sha256_checksum() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let data: Vec<u8> = (0..=100).collect();
        writer
            .add_with("sha256_test", &[data.len() as u64], &data)
            .checksum(Checksum::Sha256)
            .write()?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let obj = reader.get("sha256_test").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert!(comp.digest.as_ref().unwrap().starts_with("sha256:"));

        let retrieved = reader.read("sha256_test", true)?;
        assert_eq!(retrieved, data);

        Ok(())
    }

    #[test]
    fn test_f16_bf16_roundtrip() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let f16_data: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.5), f16::from_f32(-3.0)];
        writer.add("f16_obj", &[3], &f16_data)?;

        let bf16_data: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.5),
            bf16::from_f32(-3.0),
        ];
        writer.add("bf16_obj", &[3], &bf16_data)?;

        writer.finish()?;
        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let r1: Vec<f16> = reader.read_as("f16_obj")?;
        for (a, b) in r1.iter().zip(f16_data.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }

        let r2: Vec<bf16> = reader.read_as("bf16_obj")?;
        for (a, b) in r2.iter().zip(bf16_data.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }

        Ok(())
    }

    #[test]
    fn test_sparse_csr() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let values: Vec<f32> = vec![1.0, 2.0];
        let indices: Vec<u64> = vec![1, 2];
        let indptr: Vec<u64> = vec![0, 1, 2];

        writer.add_csr(
            "sparse_csr",
            vec![2, 3],
            DType::F32,
            &values,
            &indices,
            &indptr,
            Compression::Raw,
            Checksum::None,
        )?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let csr = reader.read_csr::<f32>("sparse_csr")?;
        assert_eq!(csr.shape, vec![2, 3]);
        assert_eq!(csr.values, values);
        assert_eq!(csr.indices, indices);
        assert_eq!(csr.indptr, indptr);

        Ok(())
    }

    #[test]
    fn test_sparse_coo() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let values: Vec<i32> = vec![10, 20];
        // SoA format: [row_indices..., col_indices...]
        let coords: Vec<u64> = vec![0, 1, 0, 2];

        writer.add_coo(
            "sparse_coo",
            vec![2, 3],
            DType::I32,
            &values,
            &coords,
            Compression::Raw,
            Checksum::None,
        )?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let coo = reader.read_coo::<i32>("sparse_coo")?;
        assert_eq!(coo.shape, vec![2, 3]);
        assert_eq!(coo.values, values);
        assert_eq!(coo.indices.len(), 2);
        assert_eq!(coo.indices[0], vec![0, 0]);
        assert_eq!(coo.indices[1], vec![1, 2]);

        Ok(())
    }

    #[test]
    fn test_invalid_magic() {
        let invalid = b"BADMAGIC";
        let mut buffer = Cursor::new(invalid.to_vec());
        match Reader::new(&mut buffer) {
            Err(Error::InvalidMagicNumber { found }) => {
                assert_eq!(found, invalid.to_vec());
            }
            _ => panic!("Expected InvalidMagicNumber"),
        }
    }

    #[test]
    fn test_object_not_found() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = Writer::new(&mut buffer)?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        match reader.read("nonexistent", true) {
            Err(Error::ObjectNotFound(name)) => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected ObjectNotFound"),
        }
        Ok(())
    }

    #[test]
    fn test_all_dtypes() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        macro_rules! add_dtype {
            ($name:expr, $dtype:expr, $val:expr, $t:ty) => {
                let val = $val as $t;
                // No need to convert to bytes!
                writer.add($name, &[1], &[val])?;
            };
        }

        add_dtype!("t_f64", DType::F64, 1.5f64, f64);
        add_dtype!("t_f32", DType::F32, 2.5f32, f32);
        add_dtype!("t_i64", DType::I64, -100i64, i64);
        add_dtype!("t_i32", DType::I32, -200i32, i32);
        add_dtype!("t_i16", DType::I16, -300i16, i16);
        add_dtype!("t_i8", DType::I8, -50i8, i8);
        add_dtype!("t_u64", DType::U64, 100u64, u64);
        add_dtype!("t_u32", DType::U32, 200u32, u32);
        add_dtype!("t_u16", DType::U16, 300u16, u16);
        add_dtype!("t_u8", DType::U8, 50u8, u8);
        writer.add_bytes(
            "t_bool",
            vec![1],
            DType::Bool,
            Compression::Raw,
            &[1u8],
            Checksum::None,
        )?;

        // Manual bool test using bytes directly since bool is not Pod
        writer.add_bytes(
            "t_bool_typed",
            vec![1],
            DType::Bool,
            Compression::Raw,
            &[1u8],
            Checksum::None,
        )?;

        writer.finish()?;
        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        assert_eq!(reader.read_as::<f64>("t_f64")?[0], 1.5);
        assert_eq!(reader.read_as::<f32>("t_f32")?[0], 2.5);
        assert_eq!(reader.read_as::<i64>("t_i64")?[0], -100);
        assert_eq!(reader.read_as::<i32>("t_i32")?[0], -200);
        assert_eq!(reader.read_as::<i16>("t_i16")?[0], -300);
        assert_eq!(reader.read_as::<i8>("t_i8")?[0], -50);
        assert_eq!(reader.read_as::<u64>("t_u64")?[0], 100);
        assert_eq!(reader.read_as::<u32>("t_u32")?[0], 200);
        assert_eq!(reader.read_as::<u16>("t_u16")?[0], 300);
        assert_eq!(reader.read_as::<u8>("t_u8")?[0], 50);
        assert_eq!(reader.read_as::<bool>("t_bool_typed")?[0], true);

        Ok(())
    }

    #[test]
    fn test_mmap_reader() -> Result<(), Error> {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_v11.zt");

        {
            let file = std::fs::File::create(&path)?;
            let mut writer = Writer::new(std::io::BufWriter::new(file))?;
            let data: Vec<f32> = vec![1.0, 2.0, 3.0];
            writer.add("test", &[3], &data)?;
            writer.finish()?;
        }

        let reader = Reader::open_mmap(&path)?;
        let data: Vec<f32> = reader.read_as("test")?;
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn test_uncompressed_length_with_zstd() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        writer
            .add_with("compressed", &[256], &data)
            .compress(Compression::Zstd(3))
            .write()?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let obj = reader.get("compressed").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert_eq!(comp.encoding, Encoding::Zstd);
        assert_eq!(comp.uncompressed_length, Some(256 * 4)); // 256 f32 = 1024 bytes
        assert!(comp.length < 256 * 4); // compressed should be smaller

        let retrieved: Vec<f32> = reader.read_as("compressed")?;
        assert_eq!(retrieved, data);

        Ok(())
    }

    #[test]
    fn test_uncompressed_length_none_for_raw() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        writer.add("raw", &[3], &data)?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let obj = reader.get("raw").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert_eq!(comp.encoding, Encoding::Raw);
        assert_eq!(comp.uncompressed_length, None);

        Ok(())
    }

    #[test]
    fn test_type_field_none_by_default() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = Writer::new(&mut buffer)?;

        let data: Vec<u8> = vec![1, 2, 3, 4];
        writer.add("u8_data", &[4], &data)?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;

        let obj = reader.get("u8_data").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert_eq!(comp.r#type, None);
        assert_eq!(comp.dtype, DType::U8);

        Ok(())
    }

    #[test]
    fn test_type_field_cbor_roundtrip() -> Result<(), Error> {
        use crate::models::{Component, Format, Manifest, Object};

        // Build a manifest with a type field set
        let mut components = std::collections::BTreeMap::new();
        components.insert(
            "data".to_string(),
            Component {
                dtype: DType::U8,
                r#type: Some("f8_e4m3fn".to_string()),
                offset: 64,
                length: 1024,
                uncompressed_length: None,
                encoding: Encoding::Raw,
                digest: None,
            },
        );

        let mut objects = std::collections::BTreeMap::new();
        objects.insert(
            "weights".to_string(),
            Object {
                shape: vec![32, 32],
                format: Format::Dense,
                attributes: None,
                components,
            },
        );

        let manifest = Manifest {
            version: "1.2.0".to_string(),
            attributes: None,
            objects,
        };

        // Serialize to CBOR and back
        let mut cbor = Vec::new();
        ciborium::into_writer(&manifest, &mut cbor).unwrap();
        let decoded: Manifest = ciborium::from_reader(std::io::Cursor::new(&cbor)).unwrap();

        let obj = decoded.objects.get("weights").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert_eq!(comp.r#type, Some("f8_e4m3fn".to_string()));
        assert_eq!(comp.dtype, DType::U8);
        assert_eq!(comp.uncompressed_length, None);

        Ok(())
    }

    #[test]
    fn test_manifest_version_1_2() -> Result<(), Error> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = Writer::new(&mut buffer)?;
        writer.finish()?;

        buffer.seek(SeekFrom::Start(0))?;
        let reader = Reader::new(&mut buffer)?;
        assert_eq!(reader.manifest.version, "1.2.0");

        Ok(())
    }

    #[test]
    fn test_attributes_ciborium_value() -> Result<(), Error> {
        use crate::models::Manifest;

        let mut attrs = std::collections::BTreeMap::new();
        attrs.insert(
            "framework".to_string(),
            ciborium::Value::Text("PyTorch".to_string()),
        );
        attrs.insert("version".to_string(), ciborium::Value::Integer(2.into()));

        let manifest = Manifest {
            version: "1.2.0".to_string(),
            attributes: Some(attrs),
            objects: std::collections::BTreeMap::new(),
        };

        // Serialize to CBOR and back
        let mut cbor = Vec::new();
        ciborium::into_writer(&manifest, &mut cbor).unwrap();
        let decoded: Manifest = ciborium::from_reader(std::io::Cursor::new(&cbor)).unwrap();

        let attrs = decoded.attributes.unwrap();
        assert_eq!(
            attrs.get("framework"),
            Some(&ciborium::Value::Text("PyTorch".to_string()))
        );
        assert_eq!(
            attrs.get("version"),
            Some(&ciborium::Value::Integer(2.into()))
        );

        Ok(())
    }
}
