//! ONNX format reader.
//!
//! Provides read-only access to `.onnx` files through the unified `TensorReader` API.
//! Extracts model weight initializers from the ONNX protobuf structure using a
//! minimal hand-written protobuf parser (no external dependency).
//!
//! Tensors with `raw_data` use memory mapping for zero-copy access. Tensors with
//! typed data fields (float_data, int32_data, etc.) are converted to raw bytes.
//!
//! Requires the `onnx` feature.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::error::Error;
use crate::models::{DType, Manifest, Object};
use crate::reader::{TensorData, TensorElement, TensorReader};

// ---- ONNX data types ----

fn onnx_dtype(type_id: u64) -> Result<DType, Error> {
    match type_id {
        1 => Ok(DType::F32),
        2 => Ok(DType::U8),
        3 => Ok(DType::I8),
        4 => Ok(DType::U16),
        5 => Ok(DType::I16),
        6 => Ok(DType::I32),
        7 => Ok(DType::I64),
        9 => Ok(DType::Bool),
        10 => Ok(DType::F16),
        11 => Ok(DType::F64),
        12 => Ok(DType::U32),
        13 => Ok(DType::U64),
        16 => Ok(DType::BF16),
        _ => Err(Error::UnsupportedDType(format!(
            "Unsupported ONNX data type: {}",
            type_id
        ))),
    }
}

// ---- Minimal protobuf wire format parser ----

/// Wire types in the protobuf encoding.
const WIRE_VARINT: u32 = 0;
const WIRE_64BIT: u32 = 1;
const WIRE_LENGTH_DELIMITED: u32 = 2;
const WIRE_32BIT: u32 = 5;

/// Cursor for parsing protobuf wire format from a byte slice.
struct ProtobufCursor<'a> {
    data: &'a [u8],
    pos: usize,
    /// Absolute offset of `data[0]` within the mmap. Used to compute
    /// absolute offsets for zero-copy raw_data references.
    base_offset: usize,
}

impl<'a> ProtobufCursor<'a> {
    fn new(data: &'a [u8], base_offset: usize) -> Self {
        Self {
            data,
            pos: 0,
            base_offset,
        }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Read a varint (LEB128 encoding).
    fn read_varint(&mut self) -> Result<u64, Error> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            if self.pos >= self.data.len() {
                return Err(Error::UnexpectedEof);
            }
            let byte = self.data[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(Error::InvalidFileStructure(
                    "Varint too long".to_string(),
                ));
            }
        }
    }

    /// Read a protobuf tag, returning (field_number, wire_type).
    fn read_tag(&mut self) -> Result<(u32, u32), Error> {
        let v = self.read_varint()?;
        let field_number = (v >> 3) as u32;
        let wire_type = (v & 0x7) as u32;
        Ok((field_number, wire_type))
    }

    /// Read a length-delimited field and return the byte slice.
    fn read_bytes(&mut self) -> Result<&'a [u8], Error> {
        let len = self.read_varint()? as usize;
        let end = self.pos.checked_add(len).ok_or(Error::UnexpectedEof)?;
        if end > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// Read a length-delimited field and return (absolute_offset, length, slice).
    fn read_bytes_with_offset(&mut self) -> Result<(usize, usize, &'a [u8]), Error> {
        let len = self.read_varint()? as usize;
        let end = self.pos.checked_add(len).ok_or(Error::UnexpectedEof)?;
        if end > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        let abs_offset = self.base_offset.checked_add(self.pos).ok_or(Error::UnexpectedEof)?;
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok((abs_offset, len, slice))
    }

    /// Skip a field of the given wire type.
    fn skip_field(&mut self, wire_type: u32) -> Result<(), Error> {
        match wire_type {
            WIRE_VARINT => {
                self.read_varint()?;
            }
            WIRE_64BIT => {
                let end = self.pos.checked_add(8).ok_or(Error::UnexpectedEof)?;
                if end > self.data.len() {
                    return Err(Error::UnexpectedEof);
                }
                self.pos = end;
            }
            WIRE_LENGTH_DELIMITED => {
                self.read_bytes()?;
            }
            WIRE_32BIT => {
                let end = self.pos.checked_add(4).ok_or(Error::UnexpectedEof)?;
                if end > self.data.len() {
                    return Err(Error::UnexpectedEof);
                }
                self.pos = end;
            }
            _ => {
                return Err(Error::InvalidFileStructure(format!(
                    "Unknown protobuf wire type: {}",
                    wire_type
                )));
            }
        }
        Ok(())
    }
}

// ---- TensorProto parsing ----

/// Parsed tensor info from an ONNX TensorProto message.
struct OnnxTensorInfo {
    name: String,
    dims: Vec<u64>,
    data_type: u64,
    data: OnnxTensorData,
}

/// Where the tensor data lives.
enum OnnxTensorData {
    /// raw_data field: zero-copy from mmap (absolute offset + length).
    RawData { offset: usize, length: usize },
    /// Typed data assembled from repeated fields into raw bytes.
    TypedData(Vec<u8>),
}

/// Parse a TensorProto message.
fn parse_tensor_proto<'a>(
    data: &'a [u8],
    base_offset: usize,
) -> Result<OnnxTensorInfo, Error> {
    let mut cursor = ProtobufCursor::new(data, base_offset);

    let mut name = String::new();
    let mut dims: Vec<u64> = Vec::new();
    let mut data_type: u64 = 0;
    let mut raw_data_offset: Option<(usize, usize)> = None;
    let mut float_data: Vec<f32> = Vec::new();
    let mut int32_data: Vec<i32> = Vec::new();
    let mut int64_data: Vec<i64> = Vec::new();
    let mut double_data: Vec<f64> = Vec::new();
    let mut uint64_data: Vec<u64> = Vec::new();
    let mut data_location: u64 = 0;

    while !cursor.is_empty() {
        let (field, wire) = cursor.read_tag()?;

        match (field, wire) {
            // field 1: dims (packed repeated int64)
            (1, WIRE_LENGTH_DELIMITED) => {
                let bytes = cursor.read_bytes()?;
                let mut sub = ProtobufCursor::new(bytes, 0);
                while !sub.is_empty() {
                    dims.push(sub.read_varint()?);
                }
            }
            // field 1: dims (single varint, non-packed)
            (1, WIRE_VARINT) => {
                dims.push(cursor.read_varint()?);
            }
            // field 2: data_type
            (2, WIRE_VARINT) => {
                data_type = cursor.read_varint()?;
            }
            // field 4: float_data (packed repeated float)
            (4, WIRE_LENGTH_DELIMITED) => {
                let bytes = cursor.read_bytes()?;
                if bytes.len() % 4 != 0 {
                    return Err(Error::InvalidFileStructure(
                        "float_data field length not aligned to 4 bytes".into(),
                    ));
                }
                for chunk in bytes.chunks_exact(4) {
                    float_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            // field 5: int32_data (packed repeated int32)
            (5, WIRE_LENGTH_DELIMITED) => {
                let bytes = cursor.read_bytes()?;
                if bytes.len() % 4 != 0 {
                    return Err(Error::InvalidFileStructure(
                        "int32_data field length not aligned to 4 bytes".into(),
                    ));
                }
                for chunk in bytes.chunks_exact(4) {
                    int32_data.push(i32::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            // field 5: int32_data (single varint, non-packed)
            (5, WIRE_VARINT) => {
                int32_data.push(cursor.read_varint()? as i32);
            }
            // field 7: int64_data (packed repeated int64)
            (7, WIRE_LENGTH_DELIMITED) => {
                let bytes = cursor.read_bytes()?;
                if bytes.len() % 8 != 0 {
                    return Err(Error::InvalidFileStructure(
                        "int64_data field length not aligned to 8 bytes".into(),
                    ));
                }
                for chunk in bytes.chunks_exact(8) {
                    int64_data.push(i64::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            // field 7: int64_data (single varint, non-packed)
            (7, WIRE_VARINT) => {
                int64_data.push(cursor.read_varint()? as i64);
            }
            // field 8: name (string)
            (8, WIRE_LENGTH_DELIMITED) => {
                let bytes = cursor.read_bytes()?;
                name = String::from_utf8(bytes.to_vec()).map_err(|e| {
                    Error::InvalidFileStructure(format!("Invalid UTF-8 in tensor name: {}", e))
                })?;
            }
            // field 10: double_data (packed repeated double)
            (10, WIRE_LENGTH_DELIMITED) => {
                let bytes = cursor.read_bytes()?;
                if bytes.len() % 8 != 0 {
                    return Err(Error::InvalidFileStructure(
                        "double_data field length not aligned to 8 bytes".into(),
                    ));
                }
                for chunk in bytes.chunks_exact(8) {
                    double_data.push(f64::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            // field 11: uint64_data (packed repeated uint64)
            (11, WIRE_LENGTH_DELIMITED) => {
                let bytes = cursor.read_bytes()?;
                if bytes.len() % 8 != 0 {
                    return Err(Error::InvalidFileStructure(
                        "uint64_data field length not aligned to 8 bytes".into(),
                    ));
                }
                for chunk in bytes.chunks_exact(8) {
                    uint64_data.push(u64::from_le_bytes(chunk.try_into().unwrap()));
                }
            }
            // field 11: uint64_data (single varint, non-packed)
            (11, WIRE_VARINT) => {
                uint64_data.push(cursor.read_varint()?);
            }
            // field 9: raw_data (bytes)
            (9, WIRE_LENGTH_DELIMITED) => {
                let (offset, length, _) = cursor.read_bytes_with_offset()?;
                raw_data_offset = Some((offset, length));
            }
            // field 14: data_location (enum)
            (14, WIRE_VARINT) => {
                data_location = cursor.read_varint()?;
            }
            // Skip all other fields
            (_, _) => {
                cursor.skip_field(wire)?;
            }
        }
    }

    if data_location == 1 {
        return Err(Error::Other(format!(
            "External data not supported for tensor '{}'. \
             Use onnx.load_external_data_for_model() to internalize data first.",
            name
        )));
    }

    // Determine data source
    let tensor_data = if let Some((offset, length)) = raw_data_offset {
        OnnxTensorData::RawData { offset, length }
    } else if !float_data.is_empty() {
        let bytes: Vec<u8> = float_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        OnnxTensorData::TypedData(bytes)
    } else if !int32_data.is_empty() {
        let bytes: Vec<u8> = int32_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        OnnxTensorData::TypedData(bytes)
    } else if !int64_data.is_empty() {
        let bytes: Vec<u8> = int64_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        OnnxTensorData::TypedData(bytes)
    } else if !double_data.is_empty() {
        let bytes: Vec<u8> = double_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        OnnxTensorData::TypedData(bytes)
    } else if !uint64_data.is_empty() {
        let bytes: Vec<u8> = uint64_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        OnnxTensorData::TypedData(bytes)
    } else {
        // Empty tensor (zero elements)
        OnnxTensorData::TypedData(Vec::new())
    };

    Ok(OnnxTensorInfo {
        name,
        dims,
        data_type,
        data: tensor_data,
    })
}

// ---- OnnxReader ----

/// Location of tensor data within the reader.
enum TensorDataLocation {
    /// Zero-copy from mmap (raw_data field).
    MmapRange { offset: usize, length: usize },
    /// Owned bytes (from typed data fields).
    Owned(Vec<u8>),
}

/// Reader for ONNX (.onnx) files.
///
/// Extracts initializer tensors from the ONNX model graph. Tensors with
/// `raw_data` use memory mapping for zero-copy access.
pub struct OnnxReader {
    mmap: Mmap,
    pub manifest: Manifest,
    data_locations: BTreeMap<String, TensorDataLocation>,
}

impl OnnxReader {
    /// Opens an ONNX file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.is_empty() {
            return Err(Error::InvalidFileStructure(
                "Empty ONNX file".to_string(),
            ));
        }

        // Parse ModelProto to find the graph field
        let graph_data = Self::find_graph(&mmap)?;

        // Parse GraphProto to collect initializer TensorProtos
        let tensor_infos = Self::parse_initializers(graph_data.0, graph_data.1)?;

        // Build manifest and data locations
        let mut objects = BTreeMap::new();
        let mut data_locations = BTreeMap::new();

        for info in tensor_infos {
            let dtype = onnx_dtype(info.data_type)?;

            let data_len = match &info.data {
                OnnxTensorData::RawData { length, .. } => *length,
                OnnxTensorData::TypedData(v) => v.len(),
            };

            let obj = Object::dense(info.dims, dtype, 0, data_len as u64);

            let location = match info.data {
                OnnxTensorData::RawData { offset, length } => {
                    TensorDataLocation::MmapRange { offset, length }
                }
                OnnxTensorData::TypedData(v) => TensorDataLocation::Owned(v),
            };

            data_locations.insert(info.name.clone(), location);
            objects.insert(info.name, obj);
        }

        let manifest = Manifest {
            version: "onnx".to_string(),
            attributes: None,
            objects,
        };

        Ok(Self {
            mmap,
            manifest,
            data_locations,
        })
    }

    /// Find the GraphProto field (field 7) in the ModelProto.
    /// Returns (slice, base_offset) of the graph data within the mmap.
    fn find_graph(mmap: &Mmap) -> Result<(&[u8], usize), Error> {
        let mut cursor = ProtobufCursor::new(mmap, 0);

        while !cursor.is_empty() {
            let (field, wire) = cursor.read_tag()?;
            if field == 7 && wire == WIRE_LENGTH_DELIMITED {
                let (offset, _length, slice) = cursor.read_bytes_with_offset()?;
                return Ok((slice, offset));
            }
            cursor.skip_field(wire)?;
        }

        Err(Error::InvalidFileStructure(
            "No graph field found in ONNX ModelProto".to_string(),
        ))
    }

    /// Parse initializer tensors (field 5) from a GraphProto.
    fn parse_initializers(
        graph_data: &[u8],
        graph_base_offset: usize,
    ) -> Result<Vec<OnnxTensorInfo>, Error> {
        let mut cursor = ProtobufCursor::new(graph_data, graph_base_offset);
        let mut tensors = Vec::new();

        while !cursor.is_empty() {
            let (field, wire) = cursor.read_tag()?;
            if field == 5 && wire == WIRE_LENGTH_DELIMITED {
                let (offset, _length, slice) = cursor.read_bytes_with_offset()?;
                let info = parse_tensor_proto(slice, offset)?;
                if !info.name.is_empty() {
                    tensors.push(info);
                }
            } else {
                cursor.skip_field(wire)?;
            }
        }

        Ok(tensors)
    }

    /// Gets a zero-copy reference to a tensor's raw data.
    pub fn view(&self, name: &str) -> Result<&[u8], Error> {
        match self.data_locations.get(name) {
            Some(TensorDataLocation::MmapRange { offset, length }) => {
                Ok(&self.mmap[*offset..*offset + *length])
            }
            Some(TensorDataLocation::Owned(data)) => Ok(data.as_slice()),
            None => Err(Error::ObjectNotFound(name.to_string())),
        }
    }

    /// Gets a typed zero-copy reference to a tensor's data.
    pub fn view_as<T: TensorElement>(&self, name: &str) -> Result<&[T], Error> {
        let dtype = self.manifest.objects.get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?
            .data_dtype()?;
        if T::DTYPE != dtype {
            return Err(Error::TypeMismatch {
                expected: dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }
        crate::reader::bytes_as_typed(self.view(name)?)
    }

    /// Reads tensor data as a copy.
    pub fn read(&self, name: &str) -> Result<Vec<u8>, Error> {
        self.view(name).map(|s| s.to_vec())
    }

    /// Reads tensor data as a typed vector.
    pub fn read_as<T: TensorElement>(&self, name: &str) -> Result<Vec<T>, Error> {
        let dtype = self.manifest.objects.get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?
            .data_dtype()?;
        if T::DTYPE != dtype {
            return Err(Error::TypeMismatch {
                expected: dtype.as_str().to_string(),
                found: std::any::type_name::<T>().to_string(),
                context: format!("object '{}'", name),
            });
        }
        crate::reader::bytes_to_typed_vec(self.view(name)?)
    }
}

impl TensorReader for OnnxReader {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read_data(&self, name: &str) -> Result<TensorData<'_>, Error> {
        match self.data_locations.get(name) {
            Some(TensorDataLocation::MmapRange { offset, length }) => {
                Ok(TensorData::Borrowed(&self.mmap[*offset..*offset + *length]))
            }
            Some(TensorDataLocation::Owned(data)) => Ok(TensorData::Borrowed(data.as_slice())),
            None => Err(Error::ObjectNotFound(name.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_dtype() {
        assert_eq!(onnx_dtype(1).unwrap(), DType::F32);
        assert_eq!(onnx_dtype(7).unwrap(), DType::I64);
        assert_eq!(onnx_dtype(10).unwrap(), DType::F16);
        assert_eq!(onnx_dtype(11).unwrap(), DType::F64);
        assert_eq!(onnx_dtype(16).unwrap(), DType::BF16);
        assert!(onnx_dtype(0).is_err()); // UNDEFINED
        assert!(onnx_dtype(8).is_err()); // STRING
    }

    #[test]
    fn test_varint_parsing() {
        // Single byte: 150 = 0b10010110 → varint [0x96, 0x01] = 150
        let data = [0x96u8, 0x01];
        let mut cursor = ProtobufCursor::new(&data, 0);
        assert_eq!(cursor.read_varint().unwrap(), 150);

        // Single byte: 1
        let data = [0x01u8];
        let mut cursor = ProtobufCursor::new(&data, 0);
        assert_eq!(cursor.read_varint().unwrap(), 1);

        // Zero
        let data = [0x00u8];
        let mut cursor = ProtobufCursor::new(&data, 0);
        assert_eq!(cursor.read_varint().unwrap(), 0);
    }

    #[test]
    fn test_tag_parsing() {
        // field 1, wire type 0 (varint): tag = (1 << 3) | 0 = 8
        let data = [0x08u8];
        let mut cursor = ProtobufCursor::new(&data, 0);
        assert_eq!(cursor.read_tag().unwrap(), (1, 0));

        // field 2, wire type 2 (length-delimited): tag = (2 << 3) | 2 = 18
        let data = [0x12u8];
        let mut cursor = ProtobufCursor::new(&data, 0);
        assert_eq!(cursor.read_tag().unwrap(), (2, 2));

        // field 7, wire type 2: tag = (7 << 3) | 2 = 58
        let data = [0x3Au8];
        let mut cursor = ProtobufCursor::new(&data, 0);
        assert_eq!(cursor.read_tag().unwrap(), (7, 2));
    }
}
