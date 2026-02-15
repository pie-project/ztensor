//! Backwards compatibility with zTensor v0.1.0 format.
//!
//! The unified `Reader` now handles both v0.1.0 and v1.2 files.
//! This module provides a type alias for backwards compatibility and
//! utility functions for format detection.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::Error;
use crate::models::MAGIC_V01;
use crate::reader::Reader;

/// Type alias for backwards compatibility.
/// `LegacyReader` is now just `Reader`.
pub type LegacyReader<R> = Reader<R>;

/// Checks if a reader contains v0.1.0 format by reading magic.
pub fn is_legacy_format<R: Read + Seek>(reader: &mut R) -> Result<bool, Error> {
    let pos = reader.stream_position()?;
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    reader.seek(SeekFrom::Start(pos))?;
    Ok(magic == *MAGIC_V01)
}

/// Checks if a file at path is v0.1.0 format.
pub fn is_legacy_file(path: impl AsRef<Path>) -> Result<bool, Error> {
    let mut file = File::open(path)?;
    is_legacy_format(&mut file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_is_legacy_format() {
        let v01_data = b"ZTEN0001";
        let v11_data = b"ZTEN1000";

        let mut cursor = Cursor::new(v01_data.to_vec());
        assert!(is_legacy_format(&mut cursor).unwrap());

        let mut cursor = Cursor::new(v11_data.to_vec());
        assert!(!is_legacy_format(&mut cursor).unwrap());
    }

    #[test]
    fn test_legacy_mmap() -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // 1. Create temp file
        let mut file = NamedTempFile::new()?;
        let path = file.path().to_path_buf();

        // 2. Write Magic
        file.write_all(MAGIC_V01)?;

        // Padding to align to 64 bytes (ALIGNMENT)
        let padding = vec![0u8; 56];
        file.write_all(&padding)?;

        // 3. Write Data (f32: 1.0, 2.0, 3.0, 4.0)
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let offset = 64u64;
        let size = bytes.len() as u64;
        file.write_all(bytes)?;

        // 4. Create Manifest (CBOR)
        #[derive(serde::Serialize)]
        struct LegacyTensorMetaSer {
            name: String,
            offset: u64,
            size: u64,
            dtype: String,
            shape: Vec<u64>,
            encoding: String,
            layout: String,
        }

        let meta = LegacyTensorMetaSer {
            name: "tensor1".to_string(),
            offset,
            size,
            dtype: "float32".to_string(),
            shape: vec![4],
            encoding: "raw".to_string(),
            layout: "dense".to_owned(),
        };

        let manifest = vec![meta];
        let mut cbor_data = Vec::new();
        ciborium::into_writer(&manifest, &mut cbor_data)?;
        file.write_all(&cbor_data)?;

        // 5. Write Manifest Size
        file.write_all(&(cbor_data.len() as u64).to_le_bytes())?;

        // Flush
        file.flush()?;

        // 6. Test Reader - now uses unified Reader via open_mmap_any
        let reader = Reader::open_mmap_any(&path)?;

        // Test untyped
        let slice = reader.view("tensor1")?;
        assert_eq!(slice, bytes);

        // Test typed
        let typed_slice = reader.view_as::<f32>("tensor1")?;
        assert_eq!(typed_slice, &data);

        Ok(())
    }
}
