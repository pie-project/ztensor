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
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::{Cursor, Read, Seek};

    #[test]
    fn test_write_read_empty() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = ZTensorWriter::new(&mut buffer)?;
        let total_size = writer.finalize()?;

        // MAGIC (8) + CBOR(empty, ~something) + CBOR_SIZE (8)
        // Empty Manifest default CBOR size might be small. 
        // Manifest::default() -> empty tensors map.
        // We don't assert exact size as CBOR map encoding varies slightly, but we check validity.
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
    fn test_write_read_single_tensor_raw_adapted() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let tensor_name = "test_tensor_raw_adapted";
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
        
        let retrieved_data = reader.read_tensor(tensor_name)?;
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

        let retrieved_data = reader.read_tensor(tensor_name)?;
        assert_eq!(retrieved_data, tensor_data_bytes);

        // Corrupt data
        // Drop reader to release borrow on buffer
        drop(reader);
        
        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut file_bytes = Vec::new();
        buffer.read_to_end(&mut file_bytes).unwrap();

        let tensor_offset = offset as usize;
        if file_bytes.len() > tensor_offset {
            file_bytes[tensor_offset] = file_bytes[tensor_offset].wrapping_add(1);
        }

        let mut corrupted_buffer = Cursor::new(file_bytes);
        let mut corrupted_reader_instance = ZTensorReader::new(&mut corrupted_buffer)?;

        match corrupted_reader_instance.read_tensor(tensor_name) {
            Err(ZTensorError::ChecksumMismatch { .. }) => { /* Expected */ }
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

        let retrieved_f32: Vec<f32> = reader.read_typed_tensor_data("f32_tensor")?;
        assert_eq!(retrieved_f32, f32_data);

        let retrieved_u16: Vec<u16> = reader.read_typed_tensor_data("u16_tensor")?;
        assert_eq!(retrieved_u16, u16_data);

        match reader.read_typed_tensor_data::<i32>("f32_tensor") {
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
        let data: Vec<f32> = reader.read_typed_tensor_data("test")?;
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
        // Row 0 has col 1 (val 1.0). Row 1 has col 2 (val 2.0).
        let indices: Vec<u64> = vec![1, 2];
        let indptr: Vec<u64> = vec![0, 1, 2];
        
        writer.add_sparse_csr_tensor(
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
        
        // Coords: Matrix (ndim x nnz). Row-major implied by implementation logic.
        // Elements at (0, 0) and (1, 2).
        // Dim 0 indices: [0, 1]
        // Dim 1 indices: [0, 2]
        // Flattened: [0, 1, 0, 2]
        let indices = vec![0, 1, 0, 2];
        
        writer.add_sparse_coo_tensor(
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
        // CooTensor stores as [ [d0, d1...], [d0, d1...] ] per element (nnz outer).
        assert_eq!(coo.indices.len(), 2);
        assert_eq!(coo.indices[0], vec![0, 0]);
        assert_eq!(coo.indices[1], vec![1, 2]);

        Ok(())
    }
}
