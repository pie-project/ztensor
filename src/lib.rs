pub mod error;
pub mod ffi;
pub mod models;
pub mod reader;
pub mod utils;
pub mod writer;

pub use error::ZTensorError;
// Adjusted exports for v1.0
pub use models::{ChecksumAlgorithm, DType, Encoding, Manifest, Tensor};
pub use reader::{Pod, ZTensorReader};
pub use writer::ZTensorWriter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MAGIC_NUMBER;
    use half::{bf16, f16};
    use std::io::{Cursor, Read, Seek};

    #[test]
    fn test_write_read_empty() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = ZTensorWriter::new(&mut buffer)?;
        let total_size = writer.finalize()?;

        // MAGIC (8) + CBOR(empty, ~something) + CBOR_SIZE (8)
        assert!(total_size > (MAGIC_NUMBER.len() as u64 + 8));

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let reader = ZTensorReader::new(&mut buffer)?;
        assert!(reader.list_tensors().is_empty());

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut magic_buf = [0u8; 8];
        buffer.read_exact(&mut magic_buf).unwrap();
        assert_eq!(&magic_buf, MAGIC_NUMBER);

        Ok(())
    }

    #[test]
    fn test_write_read_single_tensor_raw() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let tensor_name = "test_tensor_raw";
        let tensor_data_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let tensor_data_bytes: Vec<u8> = tensor_data_f32
            .iter()
            .flat_map(|f| f.to_le_bytes().to_vec())
            .collect();

        let shape = vec![2, 2];
        let dtype_val = DType::Float32; 

        writer.add_tensor(
            tensor_name,
            shape.clone(),
            dtype_val.clone(),
            Encoding::Raw,
            tensor_data_bytes.clone(),
            ChecksumAlgorithm::None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        assert_eq!(reader.list_tensors().len(), 1);
        
        let retrieved_data = reader.read_tensor(tensor_name, true)?;
        assert_eq!(retrieved_data, tensor_data_bytes);
        Ok(())
    }

    #[test]
    fn test_checksum_crc32c() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let tensor_name = "checksum_tensor";
        let tensor_data_bytes: Vec<u8> = (0..=20).collect();

        writer.add_tensor(
            tensor_name,
            vec![tensor_data_bytes.len() as u64],
            DType::Uint8,
            Encoding::Raw,
            tensor_data_bytes.clone(),
            ChecksumAlgorithm::Crc32c,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let tensor = reader.get_tensor(tensor_name).unwrap().clone();
        let data_comp = tensor.components.get("data").unwrap();

        assert!(data_comp.digest.is_some());
        let checksum_str = data_comp.digest.as_ref().unwrap();
        assert!(checksum_str.starts_with("crc32c:0x"));
        
        let offset = data_comp.offset;

        let retrieved_data = reader.read_tensor(tensor_name, true)?;
        assert_eq!(retrieved_data, tensor_data_bytes);

        // Corrupt data
        drop(reader);
        
        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut file_bytes = Vec::new();
        buffer.read_to_end(&mut file_bytes).unwrap();

        let tensor_offset = offset as usize;
        if file_bytes.len() > tensor_offset {
            file_bytes[tensor_offset] = file_bytes[tensor_offset].wrapping_add(1);
        }

        let mut corrupted_buffer = Cursor::new(file_bytes);
        let mut corrupted_reader = ZTensorReader::new(&mut corrupted_buffer)?;

        match corrupted_reader.read_tensor(tensor_name, true) {
            Err(ZTensorError::ChecksumMismatch { tensor_name: tn, component_name: cn, .. }) => {
                assert_eq!(tn, "checksum_tensor");
                assert_eq!(cn, "data");
            }
            Ok(_) => panic!("Checksum mismatch was not detected for corrupted data."),
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
        Ok(())
    }

    #[test]
    fn test_typed_data_retrieval() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let f32_data: Vec<f32> = vec![1.0, 2.5, -3.0, 4.25];
        let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer.add_tensor(
            "f32_tensor",
            vec![4],
            DType::Float32,
            Encoding::Raw,
            f32_bytes,
            ChecksumAlgorithm::None,
        )?;

        let u16_data: Vec<u16> = vec![10, 20, 30000, 65535];
        let u16_bytes: Vec<u8> = u16_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer.add_tensor(
            "u16_tensor",
            vec![2, 2],
            DType::Uint16,
            Encoding::Raw,
            u16_bytes,
            ChecksumAlgorithm::None,
        )?;

        writer.finalize()?;
        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let retrieved_f32: Vec<f32> = reader.read_tensor_as("f32_tensor")?;
        assert_eq!(retrieved_f32, f32_data);

        let retrieved_u16: Vec<u16> = reader.read_tensor_as("u16_tensor")?;
        assert_eq!(retrieved_u16, u16_data);

        match reader.read_tensor_as::<i32>("f32_tensor") {
            Err(ZTensorError::TypeMismatch { .. }) => { /* Expected */ }
            _ => panic!("Type mismatch error not triggered."),
        }
        Ok(())
    }
    
    #[test]
    fn test_mmap_reader() -> Result<(), ZTensorError> {
        let dir = std::env::temp_dir();
        let path = dir.join("test_mmap_tensor.zt");
        
        {
            let file = std::fs::File::create(&path)?;
            let mut writer = ZTensorWriter::new(std::io::BufWriter::new(file))?;
            let data: Vec<f32> = vec![1.0, 2.0, 3.0];
            let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes().to_vec()).collect();
            writer.add_tensor(
                "test",
                vec![3],
                DType::Float32,
                Encoding::Raw,
                bytes,
                ChecksumAlgorithm::None,
            )?;
            writer.finalize()?;
        }

        let mut reader = ZTensorReader::open_mmap(&path)?;
        let data: Vec<f32> = reader.read_tensor_as("test")?;
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn test_write_read_sparse_csr() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let name = "sparse_csr";
        let shape = vec![2, 3];
        let dtype = DType::Float32;
        let values: Vec<f32> = vec![1.0, 2.0];
        let values_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let indices: Vec<u64> = vec![1, 2];
        let indptr: Vec<u64> = vec![0, 1, 2];
        
        writer.add_csr_tensor(
            name,
            shape.clone(),
            dtype,
            values_bytes.clone(),
            indices.clone(),
            indptr.clone(),
            Encoding::Raw,
            ChecksumAlgorithm::None
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;
        
        let csr = reader.read_csr_tensor::<f32>(name)?;
        
        assert_eq!(csr.shape, shape);
        assert_eq!(csr.values, values);
        assert_eq!(csr.indices, indices);
        assert_eq!(csr.indptr, indptr);
        
        Ok(())
    }

    #[test]
    fn test_write_read_sparse_coo() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;
        
        let name = "sparse_coo";
        let shape = vec![2, 3];
        let dtype = DType::Int32;
        let values: Vec<i32> = vec![10, 20];
        let values_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        
        // Coords: Matrix (ndim x nnz). 
        // Elements at (0, 0) and (1, 2).
        let indices = vec![0, 1, 0, 2];
        
        writer.add_coo_tensor(
            name,
            shape.clone(),
            dtype,
            values_bytes,
            indices.clone(),
            Encoding::Raw,
            ChecksumAlgorithm::None
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;
        
        let coo = reader.read_coo_tensor::<i32>(name)?;
        
        assert_eq!(coo.shape, shape);
        assert_eq!(coo.values, values);
        assert_eq!(coo.indices.len(), 2);
        assert_eq!(coo.indices[0], vec![0, 0]);
        assert_eq!(coo.indices[1], vec![1, 2]);

        Ok(())
    }

    // ==================== NEW TESTS FOR IMPROVEMENTS ====================

    #[test]
    fn test_float16_roundtrip() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        // Create test f16 data
        let f16_data: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.5),
            f16::from_f32(-3.0),
            f16::from_f32(0.0),
        ];
        let f16_bytes: Vec<u8> = f16_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        writer.add_tensor(
            "f16_tensor",
            vec![4],
            DType::Float16,
            Encoding::Raw,
            f16_bytes,
            ChecksumAlgorithm::None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let retrieved: Vec<f16> = reader.read_tensor_as("f16_tensor")?;
        assert_eq!(retrieved.len(), f16_data.len());
        for (a, b) in retrieved.iter().zip(f16_data.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }

        Ok(())
    }

    #[test]
    fn test_bfloat16_roundtrip() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        // Create test bf16 data
        let bf16_data: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.5),
            bf16::from_f32(-3.0),
            bf16::from_f32(100.0),
        ];
        let bf16_bytes: Vec<u8> = bf16_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        writer.add_tensor(
            "bf16_tensor",
            vec![4],
            DType::BFloat16,
            Encoding::Raw,
            bf16_bytes,
            ChecksumAlgorithm::None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let retrieved: Vec<bf16> = reader.read_tensor_as("bf16_tensor")?;
        assert_eq!(retrieved.len(), bf16_data.len());
        for (a, b) in retrieved.iter().zip(bf16_data.iter()) {
            assert_eq!(a.to_f32(), b.to_f32());
        }

        Ok(())
    }

    #[test]
    fn test_compression_roundtrip() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        // Create larger data to make compression meaningful
        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        writer.add_tensor(
            "compressed_tensor",
            vec![1000],
            DType::Float32,
            Encoding::Zstd,
            data_bytes.clone(),
            ChecksumAlgorithm::Crc32c,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        // Verify tensor metadata shows compression
        let tensor = reader.get_tensor("compressed_tensor").unwrap();
        let data_comp = tensor.components.get("data").unwrap();
        assert_eq!(data_comp.encoding, Some(Encoding::Zstd));
        assert!(data_comp.digest.is_some());

        // Verify round-trip data integrity
        let retrieved: Vec<f32> = reader.read_tensor_as("compressed_tensor")?;
        assert_eq!(retrieved, data);

        Ok(())
    }

    #[test]
    fn test_scalar_tensor() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        // Scalar tensor has empty shape
        let scalar: f64 = 3.14159;
        let scalar_bytes = scalar.to_le_bytes().to_vec();

        writer.add_tensor(
            "scalar",
            vec![], // Empty shape = scalar
            DType::Float64,
            Encoding::Raw,
            scalar_bytes,
            ChecksumAlgorithm::None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let retrieved: Vec<f64> = reader.read_tensor_as("scalar")?;
        assert_eq!(retrieved.len(), 1);
        assert!((retrieved[0] - scalar).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_bool_tensor() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let bool_data: Vec<bool> = vec![true, false, true, true, false];
        let bool_bytes: Vec<u8> = bool_data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();

        writer.add_tensor(
            "bool_tensor",
            vec![5],
            DType::Bool,
            Encoding::Raw,
            bool_bytes,
            ChecksumAlgorithm::None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let retrieved: Vec<bool> = reader.read_tensor_as("bool_tensor")?;
        assert_eq!(retrieved, bool_data);

        Ok(())
    }

    #[test]
    fn test_multiple_tensors() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        // Add multiple tensors of different types
        let t1: Vec<f32> = vec![1.0, 2.0];
        let t2: Vec<i64> = vec![-100, 200, -300];
        let t3: Vec<u8> = vec![0, 127, 255];

        writer.add_tensor(
            "tensor_f32",
            vec![2],
            DType::Float32,
            Encoding::Raw,
            t1.iter().flat_map(|f| f.to_le_bytes()).collect(),
            ChecksumAlgorithm::None,
        )?;
        writer.add_tensor(
            "tensor_i64",
            vec![3],
            DType::Int64,
            Encoding::Raw,
            t2.iter().flat_map(|f| f.to_le_bytes()).collect(),
            ChecksumAlgorithm::None,
        )?;
        writer.add_tensor(
            "tensor_u8",
            vec![3],
            DType::Uint8,
            Encoding::Raw,
            t3.clone(),
            ChecksumAlgorithm::None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        assert_eq!(reader.list_tensors().len(), 3);

        let r1: Vec<f32> = reader.read_tensor_as("tensor_f32")?;
        let r2: Vec<i64> = reader.read_tensor_as("tensor_i64")?;
        let r3: Vec<u8> = reader.read_tensor_as("tensor_u8")?;

        assert_eq!(r1, t1);
        assert_eq!(r2, t2);
        assert_eq!(r3, t3);

        Ok(())
    }

    #[test]
    fn test_u64_vec_to_bytes() {
        let input: Vec<u64> = vec![0x0102030405060708, 0x090A0B0C0D0E0F10];
        let bytes = crate::utils::u64_vec_to_bytes(input.clone());
        
        // Verify length
        assert_eq!(bytes.len(), 16);
        
        // Verify we can reconstruct the values (little-endian)
        let reconstructed: Vec<u64> = bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(reconstructed, input);
    }

    #[test]
    fn test_align_offset() {
        // Already aligned
        assert_eq!(crate::utils::align_offset(0), (0, 0));
        assert_eq!(crate::utils::align_offset(64), (64, 0));
        assert_eq!(crate::utils::align_offset(128), (128, 0));

        // Need padding
        assert_eq!(crate::utils::align_offset(1), (64, 63));
        assert_eq!(crate::utils::align_offset(8), (64, 56));
        assert_eq!(crate::utils::align_offset(63), (64, 1));
        assert_eq!(crate::utils::align_offset(65), (128, 63));
    }

    #[test]
    fn test_is_little_endian() {
        // Just verify it's a const fn that returns a boolean
        let result = crate::utils::is_little_endian();
        // On most development machines this will be true
        #[cfg(target_endian = "little")]
        assert!(result);
        #[cfg(target_endian = "big")]
        assert!(!result);
    }

    #[test]
    fn test_invalid_magic_number() {
        let invalid_data = b"BADMAGIC";
        let mut buffer = Cursor::new(invalid_data.to_vec());
        
        match ZTensorReader::new(&mut buffer) {
            Err(ZTensorError::InvalidMagicNumber { found }) => {
                assert_eq!(found, invalid_data.to_vec());
            }
            _ => panic!("Expected InvalidMagicNumber error"),
        }
    }

    #[test]
    fn test_tensor_not_found() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = ZTensorWriter::new(&mut buffer)?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        match reader.read_tensor("nonexistent", true) {
            Err(ZTensorError::TensorNotFound(name)) => {
                assert_eq!(name, "nonexistent");
            }
            _ => panic!("Expected TensorNotFound error"),
        }

        Ok(())
    }

    #[test]
    fn test_type_mismatch_sparse() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        // Create a dense tensor
        let data: Vec<f32> = vec![1.0, 2.0];
        writer.add_tensor(
            "dense",
            vec![2],
            DType::Float32,
            Encoding::Raw,
            data.iter().flat_map(|f| f.to_le_bytes()).collect(),
            ChecksumAlgorithm::None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        // Try to read dense as COO - should fail
        match reader.read_coo_tensor::<f32>("dense") {
            Err(ZTensorError::TypeMismatch { expected, found, .. }) => {
                assert_eq!(expected, "sparse_coo");
                assert_eq!(found, "dense");
            }
            _ => panic!("Expected TypeMismatch error"),
        }

        // Try to read dense as CSR - should fail
        match reader.read_csr_tensor::<f32>("dense") {
            Err(ZTensorError::TypeMismatch { expected, found, .. }) => {
                assert_eq!(expected, "sparse_csr");
                assert_eq!(found, "dense");
            }
            _ => panic!("Expected TypeMismatch error"),
        }

        Ok(())
    }

    #[test]
    fn test_compressed_sparse() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let name = "compressed_csr";
        let shape = vec![100, 100];
        let dtype = DType::Float32;
        let values: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let values_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let indices: Vec<u64> = (0..50).map(|i| i as u64 * 2).collect();
        let indptr: Vec<u64> = (0..=100).map(|i| if i < 50 { i } else { 50 }).collect();
        
        writer.add_csr_tensor(
            name,
            shape.clone(),
            dtype,
            values_bytes,
            indices.clone(),
            indptr.clone(),
            Encoding::Zstd, // Compressed
            ChecksumAlgorithm::Crc32c,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;
        
        let csr = reader.read_csr_tensor::<f32>(name)?;
        
        assert_eq!(csr.shape, shape);
        assert_eq!(csr.values, values);
        assert_eq!(csr.indices, indices);
        assert_eq!(csr.indptr, indptr);
        
        Ok(())
    }

    #[test]
    fn test_all_dtypes() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        // Helper to add a tensor with a single value
        macro_rules! add_dtype_tensor {
            ($name:expr, $dtype:expr, $value:expr, $type:ty) => {
                let bytes = ($value as $type).to_le_bytes().to_vec();
                writer.add_tensor(
                    $name,
                    vec![1],
                    $dtype,
                    Encoding::Raw,
                    bytes,
                    ChecksumAlgorithm::None,
                )?;
            };
        }

        add_dtype_tensor!("t_f64", DType::Float64, 1.5f64, f64);
        add_dtype_tensor!("t_f32", DType::Float32, 2.5f32, f32);
        add_dtype_tensor!("t_i64", DType::Int64, -100i64, i64);
        add_dtype_tensor!("t_i32", DType::Int32, -200i32, i32);
        add_dtype_tensor!("t_i16", DType::Int16, -300i16, i16);
        add_dtype_tensor!("t_i8", DType::Int8, -50i8, i8);
        add_dtype_tensor!("t_u64", DType::Uint64, 100u64, u64);
        add_dtype_tensor!("t_u32", DType::Uint32, 200u32, u32);
        add_dtype_tensor!("t_u16", DType::Uint16, 300u16, u16);
        add_dtype_tensor!("t_u8", DType::Uint8, 50u8, u8);

        // f16 and bf16
        let f16_bytes = f16::from_f32(1.5).to_le_bytes().to_vec();
        writer.add_tensor("t_f16", vec![1], DType::Float16, Encoding::Raw, f16_bytes, ChecksumAlgorithm::None)?;
        
        let bf16_bytes = bf16::from_f32(2.5).to_le_bytes().to_vec();
        writer.add_tensor("t_bf16", vec![1], DType::BFloat16, Encoding::Raw, bf16_bytes, ChecksumAlgorithm::None)?;

        // bool
        writer.add_tensor("t_bool", vec![1], DType::Bool, Encoding::Raw, vec![1u8], ChecksumAlgorithm::None)?;

        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        // Verify all dtypes can be read back
        assert_eq!(reader.read_tensor_as::<f64>("t_f64")?[0], 1.5);
        assert_eq!(reader.read_tensor_as::<f32>("t_f32")?[0], 2.5);
        assert_eq!(reader.read_tensor_as::<i64>("t_i64")?[0], -100);
        assert_eq!(reader.read_tensor_as::<i32>("t_i32")?[0], -200);
        assert_eq!(reader.read_tensor_as::<i16>("t_i16")?[0], -300);
        assert_eq!(reader.read_tensor_as::<i8>("t_i8")?[0], -50);
        assert_eq!(reader.read_tensor_as::<u64>("t_u64")?[0], 100);
        assert_eq!(reader.read_tensor_as::<u32>("t_u32")?[0], 200);
        assert_eq!(reader.read_tensor_as::<u16>("t_u16")?[0], 300);
        assert_eq!(reader.read_tensor_as::<u8>("t_u8")?[0], 50);
        assert_eq!(reader.read_tensor_as::<f16>("t_f16")?[0].to_f32(), 1.5);
        assert_eq!(reader.read_tensor_as::<bf16>("t_bf16")?[0].to_f32(), 2.5);
        assert_eq!(reader.read_tensor_as::<bool>("t_bool")?[0], true);

        Ok(())
    }

    #[test]
    fn test_large_tensor() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        // 1MB tensor
        let size = 256 * 1024;
        let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
        let data_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        writer.add_tensor(
            "large",
            vec![size as u64],
            DType::Float32,
            Encoding::Zstd,
            data_bytes,
            ChecksumAlgorithm::Crc32c,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let retrieved: Vec<f32> = reader.read_tensor_as("large")?;
        assert_eq!(retrieved.len(), data.len());
        
        // Verify a few values
        assert!((retrieved[0] - data[0]).abs() < 1e-6);
        assert!((retrieved[size/2] - data[size/2]).abs() < 1e-6);
        assert!((retrieved[size-1] - data[size-1]).abs() < 1e-6);

        Ok(())
    }
}
