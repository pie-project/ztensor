pub mod error;
pub mod models;
pub mod reader;
pub mod utils;
pub mod writer;

pub use error::ZTensorError;
pub use models::{DType, DataEndianness, Encoding, TensorMetadata};
pub use reader::ZTensorReader;
pub use writer::ZTensorWriter;

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, ReadBytesExt};
    use models::{ALIGNMENT, MAGIC_NUMBER};
    use std::io::{Cursor, Seek, Read};

    #[test]
    fn test_write_read_empty() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let writer = ZTensorWriter::new(&mut buffer)?;
        let total_size = writer.finalize()?;

        assert_eq!(total_size, (MAGIC_NUMBER.len() + 1 + 8) as u64); // magic + empty_cbor_array (1 byte: 0x80) + cbor_size (8 bytes for value 1)

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let reader = ZTensorReader::new(&mut buffer)?;
        assert!(reader.list_tensors().is_empty());

        // Verify minimal file structure directly
        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut magic_buf = [0u8; 8];
        buffer.read_exact(&mut magic_buf).unwrap();
        assert_eq!(&magic_buf, MAGIC_NUMBER);

        let cbor_byte = buffer.read_u8().unwrap();
        assert_eq!(cbor_byte, 0x80); // Empty CBOR array

        let cbor_size = buffer.read_u64::<LittleEndian>().unwrap();
        assert_eq!(cbor_size, 1);

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
        let dtype = DType::Float32;

        writer.add_tensor(
            tensor_name,
            shape.clone(),
            dtype.clone(),
            Encoding::Raw,
            tensor_data_bytes.clone(),
            Some(DataEndianness::Little), // Storing as Little Endian
            None,
            None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        assert_eq!(reader.list_tensors().len(), 1);
        let metadata = reader.get_tensor_metadata(tensor_name).unwrap().clone();

        assert_eq!(metadata.name, tensor_name);
        assert_eq!(metadata.shape, shape);
        assert_eq!(metadata.dtype, dtype);
        assert_eq!(metadata.encoding, Encoding::Raw);
        assert_eq!(metadata.size, tensor_data_bytes.len() as u64);
        assert_eq!(
            metadata.offset,
            utils::calculate_padding(MAGIC_NUMBER.len() as u64).0
        );
        assert_eq!(metadata.data_endianness, Some(DataEndianness::Little));

        let retrieved_data = reader.read_raw_tensor_data(&metadata)?;
        assert_eq!(retrieved_data, tensor_data_bytes);

        // Assuming native is little-endian for this test part
        let retrieved_f32: Vec<f32> = retrieved_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(retrieved_f32, tensor_data_f32);

        Ok(())
    }

    #[test]
    fn test_write_read_single_tensor_zstd() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let tensor_name = "test_tensor_zstd";
        let tensor_data_u16: Vec<u16> = (0..100).map(|x| x as u16).collect();
        let tensor_data_bytes: Vec<u8> = tensor_data_u16
            .iter()
            .flat_map(|f| f.to_le_bytes().to_vec())
            .collect();

        let shape = vec![10, 10];
        let dtype = DType::Uint16;

        writer.add_tensor(
            tensor_name,
            shape.clone(),
            dtype.clone(),
            Encoding::Zstd,
            tensor_data_bytes.clone(),
            None, // Endianness not relevant for Zstd field itself
            None,
            None,
        )?;
        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let metadata = reader.get_tensor_metadata(tensor_name).unwrap().clone();
        assert_eq!(metadata.name, tensor_name);
        assert_eq!(metadata.encoding, Encoding::Zstd);
        // Size will be compressed size, so it won't be tensor_data_bytes.len()
        assert!(metadata.size < tensor_data_bytes.len() as u64);

        let retrieved_data_bytes = reader.read_raw_tensor_data(&metadata)?;
        assert_eq!(
            retrieved_data_bytes, tensor_data_bytes,
            "Decompressed data mismatch"
        );

        Ok(())
    }

    #[test]
    fn test_alignment_and_multiple_tensors() -> Result<(), ZTensorError> {
        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        // Tensor 1 (small, forces padding for next)
        let t1_data: Vec<u8> = vec![1, 2, 3, 4, 5]; // 5 bytes
        writer.add_tensor(
            "t1",
            vec![5],
            DType::Uint8,
            Encoding::Raw,
            t1_data.clone(),
            None,
            None,
            None,
        )?;

        // Tensor 2
        let t2_data: Vec<f32> = vec![1.0, 2.0]; // 8 bytes
        let t2_data_bytes: Vec<u8> = t2_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        writer.add_tensor(
            "t2",
            vec![2],
            DType::Float32,
            Encoding::Raw,
            t2_data_bytes.clone(),
            Some(DataEndianness::Little),
            None,
            None,
        )?;

        writer.finalize()?;

        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;

        let m1 = reader.get_tensor_metadata("t1").unwrap().clone();
        let m2 = reader.get_tensor_metadata("t2").unwrap().clone();

        let expected_t1_offset = utils::calculate_padding(MAGIC_NUMBER.len() as u64).0;
        assert_eq!(m1.offset, expected_t1_offset);
        assert_eq!(m1.offset % ALIGNMENT, 0);
        assert_eq!(m1.size, t1_data.len() as u64);

        let expected_t2_offset = utils::calculate_padding(m1.offset + m1.size).0;
        assert_eq!(m2.offset, expected_t2_offset);
        assert_eq!(m2.offset % ALIGNMENT, 0);
        assert_eq!(m2.size, t2_data_bytes.len() as u64);

        let r1_data = reader.read_raw_tensor_data(&m1)?;
        assert_eq!(r1_data, t1_data);
        let r2_data = reader.read_raw_tensor_data(&m2)?;
        assert_eq!(r2_data, t2_data_bytes);

        Ok(())
    }

    #[test]
    fn test_endianness_swap() -> Result<(), ZTensorError> {
        // This test assumes the host is Little Endian.
        // If host is Big Endian, this test would need adjustment or to be cfg'd out.
        if cfg!(target_endian = "big") {
            //println!("Skipping endianness swap test on big-endian host or test logic needs to be reversed.");
            return Ok(());
        }

        let mut buffer = Cursor::new(Vec::new());
        let mut writer = ZTensorWriter::new(&mut buffer)?;

        let tensor_name = "test_endian_tensor";
        // Native data is [0x0102, 0x0304] (u16 LE)
        let tensor_data_u16_native: Vec<u16> = vec![0x0201, 0x0403];
        // Convert to bytes in native (LE) endianness: [0x01, 0x02, 0x03, 0x04]
        let tensor_data_bytes_native: Vec<u8> = tensor_data_u16_native
            .iter()
            .flat_map(|val| val.to_le_bytes())
            .collect();

        // Add tensor, explicitly asking to store it as Big Endian
        writer.add_tensor(
            tensor_name,
            vec![2],
            DType::Uint16,
            Encoding::Raw,
            tensor_data_bytes_native.clone(), // Input data is native (LE)
            Some(DataEndianness::Big),        // Store as BE
            None,
            None,
        )?;
        writer.finalize()?;

        // Reader should convert it back to native (LE)
        buffer.seek(std::io::SeekFrom::Start(0)).unwrap();
        let mut reader = ZTensorReader::new(&mut buffer)?;
        let metadata = reader.get_tensor_metadata(tensor_name).unwrap().clone();

        assert_eq!(metadata.data_endianness, Some(DataEndianness::Big));

        let retrieved_data_bytes_native = reader.read_raw_tensor_data(&metadata)?;

        // The retrieved data should match the original native byte order
        assert_eq!(retrieved_data_bytes_native, tensor_data_bytes_native);

        let retrieved_u16_native: Vec<u16> = retrieved_data_bytes_native
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(retrieved_u16_native, tensor_data_u16_native);

        Ok(())
    }
}
