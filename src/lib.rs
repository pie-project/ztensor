//! zTensor - A high-performance tensor serialization format
//!
//! This library provides read/write support for the zTensor v1.1 file format,
//! designed for efficient storage and retrieval of tensor data with support
//! for compression, checksums, and multiple tensor layouts.
//!
//! For backwards compatibility with v0.1.0 files, use the `compat` module.

pub mod compat;
pub mod error;
pub mod ffi;
pub mod models;
pub mod reader;
pub mod utils;
pub mod writer;

pub use error::ZTensorError;
pub use models::{ChecksumAlgorithm, Component, DType, Encoding, Manifest, Object};
pub use reader::{Pod, ZTensorReader};
pub use writer::ZTensorWriter;
pub use compat::{LegacyReader, is_legacy_file, is_legacy_format};

#[cfg(test)]

mod tests {
    use super::*;
    use crate::models::MAGIC;
    use crate::writer::Compression;
    use half::{bf16, f16};
    use std::io::{Cursor, Read, Seek, SeekFrom};

    #[test]
    fn test_write_read_empty() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = ZTensorWriter::new(&mut buffer)?;
        let total_size = writer.finalize()?;

        // MAGIC(8) + CBOR + SIZE(8) + MAGIC(8)
        assert!(total_size > 24);

        buffer.seek(SeekFrom::Start(0))?;
        let reader = ZTensorReader::new(&mut buffer)?;
        assert!(reader.list_objects().is_empty());

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
    fn test_dense_f32_roundtrip() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        // Use typed API directly
        writer.add_object("test", vec![2, 2], DType::F32, Compression::Raw, &data, ChecksumAlgorithm::None)?;
        writer.finalize()?;

        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        assert_eq!(reader.list_objects().len(), 1);
        let retrieved_bytes = reader.read_object("test", true)?;
        let retrieved_floats: Vec<f32> = retrieved_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
            
        assert_eq!(retrieved_floats, data);

        Ok(())
    }

    #[test]
    fn test_typed_reading() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let f32_data: Vec<f32> = vec![1.0, 2.5, -3.0, 4.25];
        writer.add_object("f32_obj", vec![4], DType::F32, Compression::Raw, &f32_data, ChecksumAlgorithm::None)?;

        let u16_data: Vec<u16> = vec![10, 20, 30000, 65535];
        writer.add_object("u16_obj", vec![2, 2], DType::U16, Compression::Raw, &u16_data, ChecksumAlgorithm::None)?;

        writer.finalize()?;
        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let r1: Vec<f32> = reader.read_object_as("f32_obj")?;
        assert_eq!(r1, f32_data);

        let r2: Vec<u16> = reader.read_object_as("u16_obj")?;
        assert_eq!(r2, u16_data);

        // Type mismatch test
        match reader.read_object_as::<i32>("f32_obj") {
            Err(ZTensorError::TypeMismatch { .. }) => {}
            _ => panic!("Expected TypeMismatch error"),
        }

        Ok(())
    }

    #[test]
    fn test_compression_roundtrip() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
        writer.add_object("compressed", vec![1000], DType::F32, Compression::Zstd(0), &data, ChecksumAlgorithm::Crc32c)?;
        writer.finalize()?;

        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let obj = reader.get_object("compressed").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert_eq!(comp.encoding, Encoding::Zstd);
        assert!(comp.digest.is_some());

        let retrieved: Vec<f32> = reader.read_object_as("compressed")?;
        assert_eq!(retrieved, data);

        Ok(())
    }

    #[test]
    fn test_crc32c_checksum() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let data: Vec<u8> = (0..=20).collect();
        writer.add_object("checksummed", vec![data.len() as u64], DType::U8, Compression::Raw, &data, ChecksumAlgorithm::Crc32c)?;
        writer.finalize()?;

        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let obj = reader.get_object("checksummed").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert!(comp.digest.as_ref().unwrap().starts_with("crc32c:0x"));
        let offset = comp.offset;

        let retrieved = reader.read_object("checksummed", true)?;
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
        let mut corrupted_reader = ZTensorReader::new(&mut corrupted)?;

        match corrupted_reader.read_object("checksummed", true) {
            Err(ZTensorError::ChecksumMismatch { .. }) => {}
            Ok(_) => panic!("Expected ChecksumMismatch"),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }

        Ok(())
    }

    #[test]
    fn test_sha256_checksum() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let data: Vec<u8> = (0..=100).collect();
        writer.add_object("sha256_test", vec![data.len() as u64], DType::U8, Compression::Raw, &data, ChecksumAlgorithm::Sha256)?;
        writer.finalize()?;

        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let obj = reader.get_object("sha256_test").unwrap();
        let comp = obj.components.get("data").unwrap();
        assert!(comp.digest.as_ref().unwrap().starts_with("sha256:"));

        let retrieved = reader.read_object("sha256_test", true)?;
        assert_eq!(retrieved, data);

        Ok(())
    }

    #[test]
    fn test_f16_bf16_roundtrip() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let f16_data: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.5),
            f16::from_f32(-3.0),
        ];
        writer.add_object("f16_obj", vec![3], DType::F16, Compression::Raw, &f16_data, ChecksumAlgorithm::None)?;

        let bf16_data: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.5),
            bf16::from_f32(-3.0),
        ];
        writer.add_object("bf16_obj", vec![3], DType::BF16, Compression::Raw, &bf16_data, ChecksumAlgorithm::None)?;

        writer.finalize()?;
        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let r1: Vec<f16> = reader.read_object_as("f16_obj")?;
        for (a, b) in r1.iter().zip(f16_data.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }

        let r2: Vec<bf16> = reader.read_object_as("bf16_obj")?;
        for (a, b) in r2.iter().zip(bf16_data.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }

        Ok(())
    }

    #[test]
    fn test_sparse_csr() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let values: Vec<f32> = vec![1.0, 2.0];
        let indices: Vec<u64> = vec![1, 2];
        let indptr: Vec<u64> = vec![0, 1, 2];

        writer.add_csr_object("sparse_csr", vec![2, 3], DType::F32, &values, &indices, &indptr, Compression::Raw, ChecksumAlgorithm::None)?;
        writer.finalize()?;

        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let csr = reader.read_csr_object::<f32>("sparse_csr")?;
        assert_eq!(csr.shape, vec![2, 3]);
        assert_eq!(csr.values, values);
        assert_eq!(csr.indices, indices);
        assert_eq!(csr.indptr, indptr);

        Ok(())
    }

    #[test]
    fn test_sparse_coo() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let values: Vec<i32> = vec![10, 20];
        // SoA format: [row_indices..., col_indices...]
        let coords: Vec<u64> = vec![0, 1, 0, 2];

        writer.add_coo_object("sparse_coo", vec![2, 3], DType::I32, &values, &coords, Compression::Raw, ChecksumAlgorithm::None)?;
        writer.finalize()?;

        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let coo = reader.read_coo_object::<i32>("sparse_coo")?;
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
        match ZTensorReader::new(&mut buffer) {
            Err(ZTensorError::InvalidMagicNumber { found }) => {
                assert_eq!(found, invalid.to_vec());
            }
            _ => panic!("Expected InvalidMagicNumber"),
        }
    }

    #[test]
    fn test_object_not_found() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = ZTensorWriter::new(&mut buffer)?;
        writer.finalize()?;

        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        match reader.read_object("nonexistent", true) {
            Err(ZTensorError::ObjectNotFound(name)) => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected ObjectNotFound"),
        }
        Ok(())
    }

    #[test]
    fn test_all_dtypes() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        macro_rules! add_dtype {
            ($name:expr, $dtype:expr, $val:expr, $t:ty) => {
                let val = $val as $t;
                // No need to convert to bytes!
                writer.add_object($name, vec![1], $dtype, Compression::Raw, &[val], ChecksumAlgorithm::None)?;
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
        writer.add_object_bytes("t_bool", vec![1], DType::Bool, Compression::Raw, &[1u8], ChecksumAlgorithm::None)?; 
        
        // Manual bool test using bytes directly since bool is not Pod
        writer.add_object_bytes("t_bool_typed", vec![1], DType::Bool, Compression::Raw, &[1u8], ChecksumAlgorithm::None)?;

        writer.finalize()?;
        buffer.seek(SeekFrom::Start(0))?;
        let mut reader = ZTensorReader::new(&mut buffer)?;

        assert_eq!(reader.read_object_as::<f64>("t_f64")?[0], 1.5);
        assert_eq!(reader.read_object_as::<f32>("t_f32")?[0], 2.5);
        assert_eq!(reader.read_object_as::<i64>("t_i64")?[0], -100);
        assert_eq!(reader.read_object_as::<i32>("t_i32")?[0], -200);
        assert_eq!(reader.read_object_as::<i16>("t_i16")?[0], -300);
        assert_eq!(reader.read_object_as::<i8>("t_i8")?[0], -50);
        assert_eq!(reader.read_object_as::<u64>("t_u64")?[0], 100);
        assert_eq!(reader.read_object_as::<u32>("t_u32")?[0], 200);
        assert_eq!(reader.read_object_as::<u16>("t_u16")?[0], 300);
        assert_eq!(reader.read_object_as::<u8>("t_u8")?[0], 50);
        assert_eq!(reader.read_object_as::<bool>("t_bool_typed")?[0], true);

        Ok(())
    }

    #[test]
    fn test_mmap_reader() -> Result<(), ZTensorError> {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_v11.zt");

        {
            let file = std::fs::File::create(&path)?;
            let mut writer = ZTensorWriter::new(std::io::BufWriter::new(file))?;
            let data: Vec<f32> = vec![1.0, 2.0, 3.0];
            writer.add_object("test", vec![3], DType::F32, Compression::Raw, &data, ChecksumAlgorithm::None)?;
            writer.finalize()?;
        }

        let mut reader = ZTensorReader::open_mmap(&path)?;
        let data: Vec<f32> = reader.read_object_as("test")?;
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        std::fs::remove_file(path)?;
        Ok(())
    }
}
