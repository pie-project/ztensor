pub mod error;
pub mod models;
pub mod reader;
pub mod utils;
pub mod writer;
pub mod ffi;

pub use error::ZTensorError;
pub use models::{ChecksumAlgorithm, DType, DataEndianness, Encoding, TensorMetadata};
pub use reader::{Pod, ZTensorReader};
pub use writer::ZTensorWriter;

#[cfg(test)]
mod tests {
    use super::*;
    // SeekFrom is used by buffer.seek(std::io::SeekFrom::Start(0)) etc.
    use crate::models::{ALIGNMENT, MAGIC_NUMBER}; // `models::ALIGNMENT` is fine for tests needing it directly.
    use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
    use std::io::{Cursor, Read, Seek, SeekFrom, Write}; // SeekFrom IS used.

    #[test]
    fn test_write_read_empty() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = ZTensorWriter::new(&mut buffer)?;
        let total_size = writer.finalize()?;

        assert_eq!(total_size, (MAGIC_NUMBER.len() + 1 + 8) as u64);

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap(); // SeekFrom used
        let reader = ZTensorReader::new(&mut buffer)?;
        assert!(reader.list_tensors().is_empty());

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap(); // SeekFrom used
        let mut magic_buf = [0u8; 8];
        buffer.read_exact(&mut magic_buf).unwrap();
        assert_eq!(&magic_buf, MAGIC_NUMBER);

        let cbor_byte = buffer.read_u8().unwrap();
        assert_eq!(cbor_byte, 0x80);

        let cbor_size = buffer.read_u64::<LittleEndian>().unwrap();
        assert_eq!(cbor_size, 1);

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
        let dtype_val = DType::Float32; // Use `dtype_val` to avoid conflict if `dtype` module is in scope.

        writer.add_tensor(
            tensor_name,
            shape.clone(),
            dtype_val.clone(), // Use cloned dtype_val
            Encoding::Raw,
            tensor_data_bytes.clone(),
            Some(DataEndianness::Little),
            ChecksumAlgorithm::None,
            None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        assert_eq!(reader.list_tensors().len(), 1);
        // FIX: Clone metadata to release borrow on `reader`
        let metadata = reader.get_tensor_metadata(tensor_name).unwrap().clone();
        let retrieved_data = reader.read_raw_tensor_data(&metadata)?; // Pass reference to owned metadata
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
            None,
            ChecksumAlgorithm::Crc32c,
            None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let metadata_owned = reader.get_tensor_metadata(tensor_name).unwrap().clone();

        assert!(metadata_owned.checksum.is_some());
        let checksum_str = metadata_owned.checksum.as_ref().unwrap();
        assert!(checksum_str.starts_with("crc32c:0x"));

        let retrieved_data = reader.read_raw_tensor_data(&metadata_owned)?;
        assert_eq!(retrieved_data, tensor_data_bytes);

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut file_bytes = Vec::new();
        buffer.read_to_end(&mut file_bytes).unwrap();

        let tensor_offset = metadata_owned.offset as usize;
        if file_bytes.len() > tensor_offset {
            file_bytes[tensor_offset] = file_bytes[tensor_offset].wrapping_add(1);
        }

        let mut corrupted_buffer = Cursor::new(file_bytes);
        let mut corrupted_reader_instance = ZTensorReader::new(&mut corrupted_buffer)?;
        let corrupted_metadata_cloned = corrupted_reader_instance
            .get_tensor_metadata(tensor_name)
            .unwrap()
            .clone();

        match corrupted_reader_instance.read_raw_tensor_data(&corrupted_metadata_cloned) {
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
            Some(DataEndianness::Little),
            ChecksumAlgorithm::None,
            None,
        )?;

        let u16_data: Vec<u16> = vec![10, 20, 30000, 65535];
        let u16_bytes: Vec<u8> = u16_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer.add_tensor(
            "u16_tensor",
            vec![2, 2],
            DType::Uint16,
            Encoding::Raw,
            u16_bytes,
            Some(DataEndianness::Little),
            ChecksumAlgorithm::None,
            None,
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
}
