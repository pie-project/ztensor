//! HDF5 format reader.
//!
//! Provides read-only access to `.h5` / `.hdf5` files through the unified
//! `TensorReader` API. Supports contiguous and chunked storage layouts
//! with optional deflate/shuffle filter pipelines.
//!
//! Scoped to superblock v0/v1, IEEE float/integer datatypes.
//! This covers Keras/h5py model weight files.
//!
//! Requires the `hdf5` feature.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use crate::error::Error;
use crate::models::{DType, Manifest, Object};
use crate::reader::{TensorData, TensorReader};

// ---- HDF5 magic and signatures ----

const HDF5_MAGIC: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const TREE_SIGNATURE: &[u8; 4] = b"TREE";
const SNOD_SIGNATURE: &[u8; 4] = b"SNOD";
const HEAP_SIGNATURE: &[u8; 4] = b"HEAP";

// ---- HDF5 object header message types ----

const MSG_DATASPACE: u16 = 0x0001;
const MSG_DATATYPE: u16 = 0x0003;
const MSG_DATA_LAYOUT: u16 = 0x0008;
const MSG_FILTER_PIPELINE: u16 = 0x000B;
const MSG_ATTRIBUTE: u16 = 0x000C;
const MSG_CONTINUATION: u16 = 0x0010;
const MSG_SYMBOL_TABLE: u16 = 0x0011;

// ---- HDF5 datatype classes ----

const DT_CLASS_FIXED_POINT: u8 = 0; // integer
const DT_CLASS_FLOATING_POINT: u8 = 1; // float

// ---- HDF5 filter IDs ----

const FILTER_DEFLATE: u16 = 1;
const FILTER_SHUFFLE: u16 = 2;

// ---- Limits ----

const MAX_RECURSION_DEPTH: usize = 64;
const UNDEF_ADDR: u64 = u64::MAX;

// ---- Variable-width field context ----

/// Context carrying superblock parameters needed throughout parsing.
#[derive(Clone, Copy)]
struct Hdf5Ctx {
    offset_size: usize,
    length_size: usize,
}

impl Hdf5Ctx {
    /// Reads an offset-sized unsigned integer from `data` at `pos`.
    fn read_offset(&self, data: &[u8], pos: usize) -> Result<u64, Error> {
        read_uint(data, pos, self.offset_size)
    }

    /// Reads a length-sized unsigned integer from `data` at `pos`.
    fn read_length(&self, data: &[u8], pos: usize) -> Result<u64, Error> {
        read_uint(data, pos, self.length_size)
    }
}

/// Reads a little-endian unsigned integer of `size` bytes (1..=8).
fn read_uint(data: &[u8], pos: usize, size: usize) -> Result<u64, Error> {
    let end = pos.checked_add(size).ok_or(Error::UnexpectedEof)?;
    if end > data.len() {
        return Err(Error::UnexpectedEof);
    }
    let mut val: u64 = 0;
    for i in 0..size {
        val |= (data[pos + i] as u64) << (i * 8);
    }
    Ok(val)
}

fn read_u8(data: &[u8], pos: usize) -> Result<u8, Error> {
    data.get(pos).copied().ok_or(Error::UnexpectedEof)
}

fn read_u16_le(data: &[u8], pos: usize) -> Result<u16, Error> {
    let end = pos.checked_add(2).ok_or(Error::UnexpectedEof)?;
    if end > data.len() {
        return Err(Error::UnexpectedEof);
    }
    Ok(u16::from_le_bytes([data[pos], data[pos + 1]]))
}

fn read_u32_le(data: &[u8], pos: usize) -> Result<u32, Error> {
    let end = pos.checked_add(4).ok_or(Error::UnexpectedEof)?;
    if end > data.len() {
        return Err(Error::UnexpectedEof);
    }
    Ok(u32::from_le_bytes([
        data[pos],
        data[pos + 1],
        data[pos + 2],
        data[pos + 3],
    ]))
}

fn read_u64_le(data: &[u8], pos: usize) -> Result<u64, Error> {
    let end = pos.checked_add(8).ok_or(Error::UnexpectedEof)?;
    if end > data.len() {
        return Err(Error::UnexpectedEof);
    }
    Ok(u64::from_le_bytes([
        data[pos],
        data[pos + 1],
        data[pos + 2],
        data[pos + 3],
        data[pos + 4],
        data[pos + 5],
        data[pos + 6],
        data[pos + 7],
    ]))
}

/// Reads a NUL-terminated string from `data` starting at `pos`.
fn read_cstring(data: &[u8], pos: usize) -> Result<String, Error> {
    let start = pos;
    let mut end = start;
    while end < data.len() && data[end] != 0 {
        end += 1;
    }
    if end >= data.len() {
        return Err(Error::UnexpectedEof);
    }
    String::from_utf8(data[start..end].to_vec())
        .map_err(|e| Error::InvalidFileStructure(format!("Invalid UTF-8 in HDF5 string: {}", e)))
}

// ---- Filter pipeline ----

/// A single filter in the HDF5 filter pipeline.
#[derive(Clone, Debug)]
struct Hdf5Filter {
    id: u16,
    _cd_values: Vec<u32>,
}

// ---- Data layout result ----

/// Result of parsing a data layout message.
enum DataLayoutResult {
    Contiguous {
        addr: u64,
        size: u64,
    },
    Chunked {
        btree_addr: u64,
        chunk_dims: Vec<u32>,
    },
    Unsupported,
}

// ---- Data location ----

/// Where a dataset's data lives after parsing.
enum Hdf5DataLocation {
    /// Contiguous: zero-copy range in the mmap.
    MmapRange { offset: usize, length: usize },
    /// Chunked/compressed: reassembled into owned buffer.
    Owned(Vec<u8>),
}

// ---- Parsed dataset info ----

/// Info extracted from an HDF5 dataset's object header.
struct DatasetInfo {
    dtype: DType,
    shape: Vec<u64>,
    layout: DataLayoutResult,
    filters: Vec<Hdf5Filter>,
}

// ---- Chunk location for B-tree type 1 ----

/// A single chunk's location in the file.
struct ChunkLocation {
    /// Linear byte offset within the reassembled dataset (for sorting).
    linear_offset: u64,
    /// Address in file where the (possibly compressed) chunk data starts.
    file_addr: u64,
    /// Size of the chunk data on disk (possibly compressed).
    chunk_size: u32,
    /// Filter mask: bit i set means filter i was NOT applied to this chunk.
    filter_mask: u32,
}

// ---- Hdf5Reader ----

/// Reader for HDF5 (.h5 / .hdf5) files.
///
/// Uses memory mapping for efficient access to tensor data.
/// Supports contiguous and chunked storage with deflate/shuffle filters.
pub struct Hdf5Reader {
    mmap: Mmap,
    pub manifest: Manifest,
    data_locations: BTreeMap<String, Hdf5DataLocation>,
}

impl Hdf5Reader {
    /// Opens an HDF5 file using memory mapping.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() < 16 {
            return Err(Error::InvalidFileStructure(
                "File too small to be a valid HDF5 file".to_string(),
            ));
        }

        let data = &mmap[..];

        // 1. Find and validate superblock
        let sb_offset = find_superblock(data)?;

        // 2. Parse superblock v0 or v1
        let (ctx, root_btree_addr, root_heap_addr) = parse_superblock(data, sb_offset)?;

        // 3. Traverse the root group to collect datasets
        let mut objects = BTreeMap::new();
        let mut data_locations = BTreeMap::new();

        traverse_group(
            data,
            &ctx,
            root_btree_addr,
            root_heap_addr,
            "",
            &mut objects,
            &mut data_locations,
            0,
        )?;

        let manifest = Manifest {
            version: "hdf5".to_string(),
            attributes: None,
            objects,
        };

        Ok(Self {
            mmap,
            manifest,
            data_locations,
        })
    }

    /// Gets a zero-copy reference to an object's raw data.
    pub fn view(&self, name: &str) -> Result<&[u8], Error> {
        match self.data_locations.get(name) {
            Some(Hdf5DataLocation::MmapRange { offset, length }) => {
                Ok(&self.mmap[*offset..*offset + *length])
            }
            Some(Hdf5DataLocation::Owned(data)) => Ok(data.as_slice()),
            None => Err(Error::ObjectNotFound(name.to_string())),
        }
    }

    /// Reads tensor data as a typed vector.
    pub fn read_as<T: crate::reader::TensorElement>(&self, name: &str) -> Result<Vec<T>, Error> {
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
        crate::reader::bytes_to_typed_vec(self.view(name)?)
    }
}

impl TensorReader for Hdf5Reader {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read_data(&self, name: &str) -> Result<TensorData<'_>, Error> {
        let slice = self.view(name)?;
        Ok(TensorData::Borrowed(slice))
    }
}

// ---- Superblock parsing ----

/// Finds the superblock by scanning at 0, 512, 1024, 2048, ...
fn find_superblock(data: &[u8]) -> Result<usize, Error> {
    // First try offset 0
    if data.len() >= 8 && &data[0..8] == HDF5_MAGIC {
        return Ok(0);
    }
    // Then try powers of 2 starting at 512
    let mut offset = 512;
    while offset < data.len() {
        if offset + 8 <= data.len() && &data[offset..offset + 8] == HDF5_MAGIC {
            return Ok(offset);
        }
        offset *= 2;
    }
    Err(Error::InvalidMagicNumber {
        found: data[..8.min(data.len())].to_vec(),
    })
}

/// Parses a v0 or v1 superblock. Returns (ctx, root_btree_addr, root_heap_addr).
fn parse_superblock(data: &[u8], sb_offset: usize) -> Result<(Hdf5Ctx, u64, u64), Error> {
    // Magic already validated
    let pos = sb_offset + 8;

    let sb_version = read_u8(data, pos)?;
    if sb_version > 1 {
        return Err(Error::InvalidFileStructure(format!(
            "Unsupported HDF5 superblock version: {} (only v0/v1 supported)",
            sb_version
        )));
    }

    let offset_size = read_u8(data, pos + 5)? as usize;
    let length_size = read_u8(data, pos + 6)? as usize;

    if offset_size < 1 || offset_size > 8 || length_size < 1 || length_size > 8 {
        return Err(Error::InvalidFileStructure(format!(
            "Invalid HDF5 offset/length sizes: O={}, L={}",
            offset_size, length_size
        )));
    }

    let var_start = if sb_version == 0 {
        pos + 8 + 2 + 2 + 4
    } else {
        pos + 8 + 2 + 2 + 2 + 2 + 4
    };

    let ctx = Hdf5Ctx {
        offset_size,
        length_size,
    };

    let root_entry_offset = var_start + 4 * offset_size;
    let _link_name_offset = ctx.read_offset(data, root_entry_offset)?;
    let _obj_header_addr = ctx.read_offset(data, root_entry_offset + offset_size)?;
    let cache_type = read_u32_le(data, root_entry_offset + 2 * offset_size)?;

    if cache_type != 1 {
        return Err(Error::InvalidFileStructure(format!(
            "Root symbol table entry cache type {} (expected 1 for group)",
            cache_type
        )));
    }

    let scratch_offset = root_entry_offset + 2 * offset_size + 8;
    let btree_addr = ctx.read_offset(data, scratch_offset)?;
    let heap_addr = ctx.read_offset(data, scratch_offset + offset_size)?;

    Ok((ctx, btree_addr, heap_addr))
}

// ---- Local heap ----

/// Reads the local heap data segment address.
fn read_local_heap_data(data: &[u8], ctx: &Hdf5Ctx, heap_addr: u64) -> Result<usize, Error> {
    let pos = heap_addr as usize;
    let end = pos.checked_add(4).ok_or(Error::UnexpectedEof)?;
    if end > data.len() {
        return Err(Error::UnexpectedEof);
    }

    if &data[pos..pos + 4] != HEAP_SIGNATURE {
        return Err(Error::InvalidFileStructure(format!(
            "Expected HEAP signature at offset {}",
            pos
        )));
    }

    let data_seg_addr_offset = pos
        .checked_add(4 + 1 + 3 + ctx.length_size + ctx.length_size)
        .ok_or(Error::UnexpectedEof)?;
    let data_seg_addr = ctx.read_offset(data, data_seg_addr_offset)?;

    Ok(data_seg_addr as usize)
}

// ---- B-tree v1 traversal (type 0: groups) ----

/// Traverses a v1 B-tree (type 0, group) to collect SNODs.
fn traverse_btree_group(
    data: &[u8],
    ctx: &Hdf5Ctx,
    btree_addr: u64,
    heap_data_addr: usize,
    prefix: &str,
    objects: &mut BTreeMap<String, Object>,
    data_locations: &mut BTreeMap<String, Hdf5DataLocation>,
    depth: usize,
) -> Result<(), Error> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(Error::InvalidFileStructure(
            "HDF5 B-tree recursion depth exceeded".to_string(),
        ));
    }

    let pos = btree_addr as usize;
    let end = pos.checked_add(4).ok_or(Error::UnexpectedEof)?;
    if end > data.len() {
        return Err(Error::UnexpectedEof);
    }

    if &data[pos..pos + 4] != TREE_SIGNATURE {
        return Err(Error::InvalidFileStructure(format!(
            "Expected TREE signature at offset {}",
            pos
        )));
    }

    let node_type = read_u8(data, pos + 4)?;
    if node_type != 0 {
        // Not a group B-tree; skip
        return Ok(());
    }

    let node_level = read_u8(data, pos + 5)?;
    let entries_used = read_u16_le(data, pos + 6)? as usize;
    let _left_sibling = ctx.read_offset(data, pos + 8)?;
    let _right_sibling = ctx.read_offset(data, pos + 8 + ctx.offset_size)?;

    let keys_start = pos + 8 + 2 * ctx.offset_size;

    if node_level > 0 {
        for i in 0..entries_used {
            let child_offset =
                keys_start + ctx.length_size + i * (ctx.length_size + ctx.offset_size);
            let child_addr = ctx.read_offset(data, child_offset)?;
            if child_addr != UNDEF_ADDR {
                traverse_btree_group(
                    data,
                    ctx,
                    child_addr,
                    heap_data_addr,
                    prefix,
                    objects,
                    data_locations,
                    depth + 1,
                )?;
            }
        }
    } else {
        for i in 0..entries_used {
            let child_offset =
                keys_start + ctx.length_size + i * (ctx.length_size + ctx.offset_size);
            let snod_addr = ctx.read_offset(data, child_offset)?;
            if snod_addr != UNDEF_ADDR {
                parse_snod(
                    data,
                    ctx,
                    snod_addr,
                    heap_data_addr,
                    prefix,
                    objects,
                    data_locations,
                    depth + 1,
                )?;
            }
        }
    }

    Ok(())
}

// ---- B-tree v1 traversal (type 1: raw data chunks) ----

/// Collects chunk locations from a type-1 B-tree for chunked datasets.
fn collect_chunk_locations(
    data: &[u8],
    ctx: &Hdf5Ctx,
    btree_addr: u64,
    ndims: usize,
    shape: &[u64],
    chunk_dims: &[u32],
    element_size: usize,
    depth: usize,
) -> Result<Vec<ChunkLocation>, Error> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(Error::InvalidFileStructure(
            "HDF5 chunk B-tree recursion depth exceeded".to_string(),
        ));
    }

    let pos = btree_addr as usize;
    if pos + 4 > data.len() {
        return Err(Error::UnexpectedEof);
    }

    if &data[pos..pos + 4] != TREE_SIGNATURE {
        return Err(Error::InvalidFileStructure(format!(
            "Expected TREE signature at offset {} for chunk B-tree",
            pos
        )));
    }

    let node_type = read_u8(data, pos + 4)?;
    if node_type != 1 {
        return Err(Error::InvalidFileStructure(format!(
            "Expected B-tree type 1 (raw data), got type {}",
            node_type
        )));
    }

    let node_level = read_u8(data, pos + 5)?;
    let entries_used = read_u16_le(data, pos + 6)? as usize;
    let _left_sibling = ctx.read_offset(data, pos + 8)?;
    let _right_sibling = ctx.read_offset(data, pos + 8 + ctx.offset_size)?;

    // Type-1 B-tree key: chunk_size(4) + filter_mask(4) + chunk_offsets(ndims * 8)
    // Note: ndims here includes the extra "element size" dimension in the B-tree key
    let key_size = 4 + 4 + (ndims + 1) * 8; // +1 for element size dimension
    let keys_start = pos + 8 + 2 * ctx.offset_size;

    let mut chunks = Vec::new();

    if node_level > 0 {
        // Internal node: recurse into children
        for i in 0..entries_used {
            let child_offset = keys_start + key_size + i * (key_size + ctx.offset_size);
            let child_addr = ctx.read_offset(data, child_offset)?;
            if child_addr != UNDEF_ADDR {
                let mut sub_chunks = collect_chunk_locations(
                    data,
                    ctx,
                    child_addr,
                    ndims,
                    shape,
                    chunk_dims,
                    element_size,
                    depth + 1,
                )?;
                chunks.append(&mut sub_chunks);
            }
        }
    } else {
        // Leaf node: extract chunk locations
        for i in 0..entries_used {
            let key_offset = keys_start + i * (key_size + ctx.offset_size);
            let chunk_size = read_u32_le(data, key_offset)?;
            let filter_mask = read_u32_le(data, key_offset + 4)?;

            // Read chunk offset coordinates (ndims dimensions)
            let mut offsets = Vec::with_capacity(ndims);
            for d in 0..ndims {
                offsets.push(read_u64_le(data, key_offset + 8 + d * 8)?);
            }

            // Compute linear byte offset from chunk coordinates using dataset shape
            let mut linear_offset: u64 = 0;
            let mut stride: u64 = element_size as u64;
            for d in (0..ndims).rev() {
                linear_offset += offsets[d] * stride;
                stride *= shape[d];
            }

            // Child pointer follows the key
            let child_offset = key_offset + key_size;
            let file_addr = ctx.read_offset(data, child_offset)?;

            if file_addr != UNDEF_ADDR {
                chunks.push(ChunkLocation {
                    linear_offset,
                    file_addr,
                    chunk_size,
                    filter_mask,
                });
            }
        }
    }

    Ok(chunks)
}

// ---- Filter application ----

/// Applies the reverse filter pipeline to decompress chunk data.
/// Filters whose corresponding bit is set in `filter_mask` are skipped
/// (per HDF5 spec, bit i set means filter i was NOT applied during write).
fn apply_filters(
    mut chunk_data: Vec<u8>,
    filters: &[Hdf5Filter],
    filter_mask: u32,
    element_size: usize,
) -> Result<Vec<u8>, Error> {
    // Filters are applied in reverse order during decompression
    for (i, filter) in filters.iter().enumerate().rev() {
        if filter_mask & (1 << i) != 0 {
            continue; // This filter was not applied to this chunk
        }
        match filter.id {
            FILTER_DEFLATE => {
                let mut decoder = flate2::read::ZlibDecoder::new(chunk_data.as_slice());
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed).map_err(|e| {
                    Error::Other(format!("HDF5 deflate decompression failed: {}", e))
                })?;
                chunk_data = decompressed;
            }
            FILTER_SHUFFLE => {
                chunk_data = unshuffle(&chunk_data, element_size);
            }
            _ => {
                return Err(Error::Other(format!(
                    "Unsupported HDF5 filter ID: {}",
                    filter.id
                )));
            }
        }
    }
    Ok(chunk_data)
}

/// Reverses the HDF5 shuffle filter (byte un-transposition).
fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }
    let num_elements = data.len() / element_size;
    let mut output = vec![0u8; data.len()];
    for i in 0..num_elements {
        for b in 0..element_size {
            output[i * element_size + b] = data[b * num_elements + i];
        }
    }
    output
}

/// Reads and reassembles a chunked dataset.
fn read_chunked_dataset(
    data: &[u8],
    ctx: &Hdf5Ctx,
    btree_addr: u64,
    shape: &[u64],
    chunk_dims: &[u32],
    dtype: DType,
    filters: &[Hdf5Filter],
) -> Result<Vec<u8>, Error> {
    let ndims = shape.len();
    let element_size = dtype.byte_size();

    // Total dataset size in bytes (checked to prevent overflow)
    let total_elements: u64 = shape.iter().product();
    let total_bytes = (total_elements as usize)
        .checked_mul(element_size)
        .ok_or_else(|| {
            Error::Other(format!(
                "Dataset size overflow: {} elements * {} bytes",
                total_elements, element_size
            ))
        })?;

    // Collect all chunk locations
    let mut chunks = collect_chunk_locations(
        data,
        ctx,
        btree_addr,
        ndims,
        shape,
        chunk_dims,
        element_size,
        0,
    )?;

    // Sort by linear offset for sequential assembly
    chunks.sort_by_key(|c| c.linear_offset);

    let mut output = vec![0u8; total_bytes];

    for chunk in &chunks {
        let addr = chunk.file_addr as usize;
        let size = chunk.chunk_size as usize;

        let chunk_end = addr.checked_add(size).ok_or_else(|| {
            Error::InvalidFileStructure(format!(
                "Chunk at offset {} with size {} overflows address space",
                addr, size
            ))
        })?;
        if chunk_end > data.len() {
            return Err(Error::InvalidFileStructure(format!(
                "Chunk at offset {} with size {} extends beyond file",
                addr, size
            )));
        }

        let raw = data[addr..addr + size].to_vec();

        let decompressed = if filters.is_empty() {
            raw
        } else {
            apply_filters(raw, filters, chunk.filter_mask, element_size)?
        };

        // Copy decompressed chunk into the correct position in the output.
        // For multi-dimensional datasets, we need to handle partial edge chunks
        // by computing per-chunk coordinates and copying row by row.
        let chunk_offset = chunk.linear_offset as usize;

        if ndims <= 1 {
            // 1D: simple linear copy
            let copy_len = decompressed
                .len()
                .min(total_bytes.saturating_sub(chunk_offset));
            output[chunk_offset..chunk_offset + copy_len]
                .copy_from_slice(&decompressed[..copy_len]);
        } else {
            // Multi-dimensional: copy chunk data considering partial edge chunks.
            // Compute chunk grid coordinates from linear_offset.
            let mut remaining = chunk_offset / element_size;
            let mut chunk_coords = vec![0u64; ndims];
            for d in (0..ndims).rev() {
                chunk_coords[d] = remaining as u64 % shape[d];
                remaining /= shape[d] as usize;
            }

            // For each element in the chunk, compute its position in the output
            copy_chunk_to_output(
                &decompressed,
                &mut output,
                shape,
                chunk_dims,
                &chunk_coords,
                element_size,
            );
        }
    }

    Ok(output)
}

/// Copies a decompressed chunk into the output buffer, handling edge chunks.
fn copy_chunk_to_output(
    chunk_data: &[u8],
    output: &mut [u8],
    shape: &[u64],
    chunk_dims: &[u32],
    chunk_start: &[u64],
    element_size: usize,
) {
    let ndims = shape.len();

    // Compute actual dimensions of this chunk (may be smaller at edges)
    let mut actual_dims = Vec::with_capacity(ndims);
    for d in 0..ndims {
        let remaining = shape[d] - chunk_start[d];
        actual_dims.push((chunk_dims[d] as u64).min(remaining) as usize);
    }

    // For the last two dimensions, copy contiguous rows
    // For higher dims, iterate over the outer dimensions

    // Compute dataset row stride (bytes per row in the last dimension)
    let row_len = actual_dims[ndims - 1] * element_size;

    // Total number of rows to copy (product of all dims except the last)
    let num_rows: usize = actual_dims[..ndims - 1].iter().product();

    let total_elements: u64 = shape.iter().product();
    let total_bytes = total_elements as usize * element_size;

    for row_idx in 0..num_rows {
        // Convert flat row index to per-dimension indices within the chunk
        let mut rem = row_idx;
        let mut src_offset = 0usize;
        let mut dst_linear = 0u64;

        // Compute strides for both chunk and dataset
        for d in (0..ndims - 1).rev() {
            let coord_in_chunk = rem % actual_dims[d];
            rem /= actual_dims[d];

            // Source offset within chunk data
            let mut chunk_stride = chunk_dims[ndims - 1] as usize * element_size;
            for dd in (d + 1..ndims - 1).rev() {
                chunk_stride *= chunk_dims[dd] as usize;
            }
            src_offset += coord_in_chunk * chunk_stride;

            // Destination linear index
            let abs_coord = chunk_start[d] + coord_in_chunk as u64;
            let mut ds_stride: u64 = 1;
            for dd in (d + 1..ndims).rev() {
                ds_stride *= shape[dd];
            }
            dst_linear += abs_coord * ds_stride;
        }

        // Add the last dimension's chunk_start offset
        dst_linear += chunk_start[ndims - 1];
        let dst_byte = dst_linear as usize * element_size;

        if dst_byte + row_len <= total_bytes && src_offset + row_len <= chunk_data.len() {
            output[dst_byte..dst_byte + row_len]
                .copy_from_slice(&chunk_data[src_offset..src_offset + row_len]);
        }
    }
}

// ---- Symbol table node (SNOD) ----

fn parse_snod(
    data: &[u8],
    ctx: &Hdf5Ctx,
    snod_addr: u64,
    heap_data_addr: usize,
    prefix: &str,
    objects: &mut BTreeMap<String, Object>,
    data_locations: &mut BTreeMap<String, Hdf5DataLocation>,
    depth: usize,
) -> Result<(), Error> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(Error::InvalidFileStructure(
            "HDF5 SNOD recursion depth exceeded".to_string(),
        ));
    }

    let pos = snod_addr as usize;
    let end = pos.checked_add(4).ok_or(Error::UnexpectedEof)?;
    if end > data.len() {
        return Err(Error::UnexpectedEof);
    }

    if &data[pos..pos + 4] != SNOD_SIGNATURE {
        return Err(Error::InvalidFileStructure(format!(
            "Expected SNOD signature at offset {}",
            pos
        )));
    }

    let _version = read_u8(data, pos + 4)?;
    let _reserved = read_u8(data, pos + 5)?;
    let num_symbols = read_u16_le(data, pos + 6)? as usize;

    let entry_size = 2 * ctx.offset_size + 4 + 4 + 16;
    let entries_start = pos + 8;

    for i in 0..num_symbols {
        let entry_pos = entries_start + i * entry_size;
        if entry_pos + entry_size > data.len() {
            break;
        }

        let link_name_offset = ctx.read_offset(data, entry_pos)?;
        let obj_header_addr = ctx.read_offset(data, entry_pos + ctx.offset_size)?;
        let cache_type = read_u32_le(data, entry_pos + 2 * ctx.offset_size)?;

        if obj_header_addr == UNDEF_ADDR || obj_header_addr == 0 {
            continue;
        }

        let name = read_cstring(data, heap_data_addr + link_name_offset as usize)?;
        if name.is_empty() {
            continue;
        }

        let full_name = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{}.{}", prefix, name)
        };

        if cache_type == 1 {
            // Group
            let scratch_pos = entry_pos + 2 * ctx.offset_size + 8;
            let child_btree = ctx.read_offset(data, scratch_pos)?;
            let child_heap = ctx.read_offset(data, scratch_pos + ctx.offset_size)?;

            let child_heap_data = read_local_heap_data(data, ctx, child_heap)?;
            traverse_btree_group(
                data,
                ctx,
                child_btree,
                child_heap_data,
                &full_name,
                objects,
                data_locations,
                depth + 1,
            )?;
        } else {
            // Dataset
            match parse_object_header(data, ctx, obj_header_addr as usize, &full_name, depth + 1)? {
                ObjectHeaderResult::Dataset(info) => {
                    let total_elems = info.shape.iter().product::<u64>();
                    let byte_size = (total_elems as usize)
                        .checked_mul(info.dtype.byte_size())
                        .ok_or_else(|| {
                            Error::Other(format!(
                                "Dataset '{}' size overflow: {} elements * {} bytes",
                                full_name,
                                total_elems,
                                info.dtype.byte_size()
                            ))
                        })?;
                    let location = match info.layout {
                        DataLayoutResult::Contiguous { addr, size } => {
                            if addr == UNDEF_ADDR {
                                continue;
                            }
                            let abs_offset = addr as usize;
                            let bsize = size as usize;
                            if abs_offset + bsize > data.len() {
                                return Err(Error::InvalidFileStructure(format!(
                                    "Dataset '{}' extends beyond file",
                                    full_name
                                )));
                            }
                            Hdf5DataLocation::MmapRange {
                                offset: abs_offset,
                                length: bsize,
                            }
                        }
                        DataLayoutResult::Chunked {
                            btree_addr,
                            chunk_dims,
                        } => {
                            let reassembled = read_chunked_dataset(
                                data,
                                ctx,
                                btree_addr,
                                &info.shape,
                                &chunk_dims,
                                info.dtype,
                                &info.filters,
                            )?;
                            Hdf5DataLocation::Owned(reassembled)
                        }
                        DataLayoutResult::Unsupported => continue,
                    };

                    let obj = Object::dense(
                        info.shape,
                        info.dtype,
                        0, // offset not meaningful for owned data
                        byte_size as u64,
                    );
                    data_locations.insert(full_name.clone(), location);
                    objects.insert(full_name, obj);
                }
                ObjectHeaderResult::Group(btree, heap) => {
                    let child_heap_data = read_local_heap_data(data, ctx, heap)?;
                    traverse_btree_group(
                        data,
                        ctx,
                        btree,
                        child_heap_data,
                        &full_name,
                        objects,
                        data_locations,
                        depth + 1,
                    )?;
                }
                ObjectHeaderResult::Skip => {}
            }
        }
    }

    Ok(())
}

// ---- Group traversal entry point ----

fn traverse_group(
    data: &[u8],
    ctx: &Hdf5Ctx,
    btree_addr: u64,
    heap_addr: u64,
    prefix: &str,
    objects: &mut BTreeMap<String, Object>,
    data_locations: &mut BTreeMap<String, Hdf5DataLocation>,
    depth: usize,
) -> Result<(), Error> {
    let heap_data_addr = read_local_heap_data(data, ctx, heap_addr)?;
    traverse_btree_group(
        data,
        ctx,
        btree_addr,
        heap_data_addr,
        prefix,
        objects,
        data_locations,
        depth,
    )
}

// ---- Object header parsing ----

enum ObjectHeaderResult {
    Dataset(DatasetInfo),
    Group(u64, u64),
    Skip,
}

/// Parses a v1 object header to extract dataset or group information.
fn parse_object_header(
    data: &[u8],
    ctx: &Hdf5Ctx,
    addr: usize,
    name: &str,
    depth: usize,
) -> Result<ObjectHeaderResult, Error> {
    if depth > MAX_RECURSION_DEPTH {
        return Err(Error::InvalidFileStructure(
            "HDF5 object header recursion depth exceeded".to_string(),
        ));
    }

    if addr.checked_add(12).ok_or(Error::UnexpectedEof)? > data.len() {
        return Err(Error::UnexpectedEof);
    }

    let version = read_u8(data, addr)?;
    if version != 1 {
        return Ok(ObjectHeaderResult::Skip);
    }

    let num_messages = read_u16_le(data, addr + 2)? as usize;
    let header_size = read_u32_le(data, addr + 8)? as usize;

    let msg_start = addr + 16;
    let msg_end = msg_start + header_size;

    let mut dtype: Option<DType> = None;
    let mut shape: Option<Vec<u64>> = None;
    let mut layout: Option<DataLayoutResult> = None;
    let mut filters: Vec<Hdf5Filter> = Vec::new();
    let mut group_info: Option<(u64, u64)> = None;

    parse_messages(
        data,
        ctx,
        msg_start,
        msg_end,
        num_messages,
        &mut dtype,
        &mut shape,
        &mut layout,
        &mut filters,
        &mut group_info,
        name,
        depth,
    )?;

    if let Some((btree, heap)) = group_info {
        return Ok(ObjectHeaderResult::Group(btree, heap));
    }

    if let (Some(dt), Some(sh), Some(lay)) = (dtype, shape, layout) {
        Ok(ObjectHeaderResult::Dataset(DatasetInfo {
            dtype: dt,
            shape: sh,
            layout: lay,
            filters,
        }))
    } else {
        Ok(ObjectHeaderResult::Skip)
    }
}

/// Parses header messages, following continuations.
#[allow(clippy::too_many_arguments)]
fn parse_messages(
    data: &[u8],
    ctx: &Hdf5Ctx,
    start: usize,
    end: usize,
    max_messages: usize,
    dtype: &mut Option<DType>,
    shape: &mut Option<Vec<u64>>,
    layout: &mut Option<DataLayoutResult>,
    filters: &mut Vec<Hdf5Filter>,
    group_info: &mut Option<(u64, u64)>,
    name: &str,
    depth: usize,
) -> Result<(), Error> {
    let mut pos = start;
    let mut messages_parsed = 0;

    while pos + 8 <= end && messages_parsed < max_messages {
        let msg_type = read_u16_le(data, pos)?;
        let msg_size = read_u16_le(data, pos + 2)? as usize;
        let _flags = read_u8(data, pos + 4)?;
        let msg_data_start = pos + 8;
        let msg_data_end = msg_data_start + msg_size;

        if msg_data_end > data.len() {
            break;
        }

        match msg_type {
            MSG_DATASPACE => {
                *shape = Some(parse_dataspace(data, msg_data_start)?);
            }
            MSG_DATATYPE => {
                *dtype = parse_datatype(data, msg_data_start).ok();
            }
            MSG_DATA_LAYOUT => {
                *layout = Some(parse_data_layout(data, ctx, msg_data_start)?);
            }
            MSG_FILTER_PIPELINE => {
                *filters = parse_filter_pipeline(data, msg_data_start)?;
            }
            MSG_CONTINUATION => {
                let cont_offset = ctx.read_offset(data, msg_data_start)?;
                let cont_length = ctx.read_offset(data, msg_data_start + ctx.offset_size)?;
                if cont_offset != UNDEF_ADDR && cont_length > 0 && depth < MAX_RECURSION_DEPTH {
                    parse_messages(
                        data,
                        ctx,
                        cont_offset as usize,
                        (cont_offset + cont_length) as usize,
                        max_messages - messages_parsed,
                        dtype,
                        shape,
                        layout,
                        filters,
                        group_info,
                        name,
                        depth + 1,
                    )?;
                }
            }
            MSG_SYMBOL_TABLE => {
                let btree = ctx.read_offset(data, msg_data_start)?;
                let heap = ctx.read_offset(data, msg_data_start + ctx.offset_size)?;
                *group_info = Some((btree, heap));
            }
            MSG_ATTRIBUTE | 0 => {}
            _ => {}
        }

        let next = msg_data_start + msg_size;
        pos = (next + 7) & !7;
        messages_parsed += 1;
    }

    Ok(())
}

// ---- Filter pipeline parsing ----

/// Parses a filter pipeline message (type 0x000B).
fn parse_filter_pipeline(data: &[u8], pos: usize) -> Result<Vec<Hdf5Filter>, Error> {
    let version = read_u8(data, pos)?;
    let nfilters = read_u8(data, pos + 1)? as usize;

    let mut filters = Vec::with_capacity(nfilters);

    // v1: version(1) + nfilters(1) + reserved(6)
    // v2: version(1) + nfilters(1)
    let mut fpos = if version == 1 { pos + 8 } else { pos + 2 };

    for _ in 0..nfilters {
        if fpos + 4 > data.len() {
            break;
        }

        let filter_id = read_u16_le(data, fpos)?;

        if version == 1 || filter_id >= 256 {
            // v1 layout: id(2) + name_len(2) + flags(2) + cd_nelmts(2) + name(padded) + cd_values
            let name_len = read_u16_le(data, fpos + 2)? as usize;
            let _flags = read_u16_le(data, fpos + 4)?;
            let cd_nelmts = read_u16_le(data, fpos + 6)? as usize;

            fpos += 8;

            // Name is padded to 8-byte multiple
            if name_len > 0 {
                let padded = (name_len + 7) & !7;
                fpos += padded;
            }

            // Client data values
            let mut cd_values = Vec::with_capacity(cd_nelmts);
            for _ in 0..cd_nelmts {
                if fpos + 4 > data.len() {
                    break;
                }
                cd_values.push(read_u32_le(data, fpos)?);
                fpos += 4;
            }
            // Pad cd_values to even count (v1 only)
            if version == 1 && cd_nelmts % 2 != 0 {
                fpos += 4;
            }

            filters.push(Hdf5Filter {
                id: filter_id,
                _cd_values: cd_values,
            });
        } else {
            // v2 with id < 256: id(2) + flags(2) + cd_nelmts(2) + cd_values
            let _flags = read_u16_le(data, fpos + 2)?;
            let cd_nelmts = read_u16_le(data, fpos + 4)? as usize;
            fpos += 6;

            let mut cd_values = Vec::with_capacity(cd_nelmts);
            for _ in 0..cd_nelmts {
                if fpos + 4 > data.len() {
                    break;
                }
                cd_values.push(read_u32_le(data, fpos)?);
                fpos += 4;
            }

            filters.push(Hdf5Filter {
                id: filter_id,
                _cd_values: cd_values,
            });
        }
    }

    Ok(filters)
}

// ---- Dataspace parsing ----

fn parse_dataspace(data: &[u8], pos: usize) -> Result<Vec<u64>, Error> {
    let version = read_u8(data, pos)?;
    let ndims = read_u8(data, pos + 1)? as usize;

    match version {
        1 => {
            let dims_start = pos + 8;
            let mut shape = Vec::with_capacity(ndims);
            for i in 0..ndims {
                shape.push(read_u64_le(data, dims_start + i * 8)?);
            }
            Ok(shape)
        }
        2 => {
            let dims_start = pos + 4;
            let mut shape = Vec::with_capacity(ndims);
            for i in 0..ndims {
                shape.push(read_u64_le(data, dims_start + i * 8)?);
            }
            Ok(shape)
        }
        _ => Err(Error::InvalidFileStructure(format!(
            "Unsupported HDF5 dataspace version: {}",
            version
        ))),
    }
}

// ---- Datatype parsing ----

fn parse_datatype(data: &[u8], pos: usize) -> Result<DType, Error> {
    if pos.checked_add(8).ok_or(Error::UnexpectedEof)? > data.len() {
        return Err(Error::UnexpectedEof);
    }

    let class_and_version = read_u8(data, pos)?;
    let dt_class = class_and_version & 0x0F;
    let class_bits_0 = read_u8(data, pos + 1)?;
    let dt_size = read_u32_le(data, pos + 4)? as usize;

    // Bit 0 of class_bits_0 is the byte order: 0 = little-endian, 1 = big-endian
    let is_big_endian = (class_bits_0 & 0x01) != 0;
    if is_big_endian && dt_size > 1 {
        return Err(Error::Other(
            "Big-endian HDF5 datatypes are not supported".to_string(),
        ));
    }

    match dt_class {
        DT_CLASS_FIXED_POINT => {
            let signed = (class_bits_0 & 0x08) != 0;
            match (dt_size, signed) {
                (8, true) => Ok(DType::I64),
                (4, true) => Ok(DType::I32),
                (2, true) => Ok(DType::I16),
                (1, true) => Ok(DType::I8),
                (8, false) => Ok(DType::U64),
                (4, false) => Ok(DType::U32),
                (2, false) => Ok(DType::U16),
                (1, false) => Ok(DType::U8),
                _ => Err(Error::UnsupportedDType(format!(
                    "HDF5 integer: size={}, signed={}",
                    dt_size, signed
                ))),
            }
        }
        DT_CLASS_FLOATING_POINT => match dt_size {
            8 => Ok(DType::F64),
            4 => Ok(DType::F32),
            2 => Ok(DType::F16),
            _ => Err(Error::UnsupportedDType(format!(
                "HDF5 float: size={}",
                dt_size
            ))),
        },
        _ => Err(Error::UnsupportedDType(format!(
            "HDF5 datatype class {} not supported (only integer and float)",
            dt_class
        ))),
    }
}

// ---- Data layout parsing ----

/// Parses a data layout message. Returns DataLayoutResult.
fn parse_data_layout(data: &[u8], ctx: &Hdf5Ctx, pos: usize) -> Result<DataLayoutResult, Error> {
    let version = read_u8(data, pos)?;

    match version {
        1 | 2 => {
            let ndims = read_u8(data, pos + 1)? as usize;
            let layout_class = read_u8(data, pos + 2)?;

            match layout_class {
                1 => {
                    // Contiguous
                    let addr_pos = pos + 8;
                    let addr = ctx.read_offset(data, addr_pos)?;
                    let dims_start = addr_pos + ctx.offset_size;
                    let mut total_size: u64 = 1;
                    for i in 0..ndims {
                        total_size *= read_u32_le(data, dims_start + i * 4)? as u64;
                    }
                    Ok(DataLayoutResult::Contiguous {
                        addr,
                        size: total_size,
                    })
                }
                2 => {
                    // Chunked v1/v2
                    let addr_pos = pos + 8;
                    let btree_addr = ctx.read_offset(data, addr_pos)?;
                    let dims_start = addr_pos + ctx.offset_size;
                    // ndims includes element size dimension for v1/v2
                    let mut chunk_dims = Vec::with_capacity(ndims.saturating_sub(1));
                    for i in 0..ndims.saturating_sub(1) {
                        chunk_dims.push(read_u32_le(data, dims_start + i * 4)?);
                    }
                    Ok(DataLayoutResult::Chunked {
                        btree_addr,
                        chunk_dims,
                    })
                }
                _ => Ok(DataLayoutResult::Unsupported),
            }
        }
        3 => {
            let layout_class = read_u8(data, pos + 1)?;

            match layout_class {
                0 => Ok(DataLayoutResult::Unsupported), // Compact
                1 => {
                    // Contiguous
                    let addr = ctx.read_offset(data, pos + 2)?;
                    let size = ctx.read_length(data, pos + 2 + ctx.offset_size)?;
                    Ok(DataLayoutResult::Contiguous { addr, size })
                }
                2 => {
                    // Chunked v3: dimensionality(1) + btree_addr(O) + chunk_dims[dim](4 each)
                    let dimensionality = read_u8(data, pos + 2)? as usize;
                    let btree_addr = ctx.read_offset(data, pos + 3)?;
                    let dims_start = pos + 3 + ctx.offset_size;
                    // dimensionality includes the extra element-size dimension
                    let ndims = dimensionality.saturating_sub(1);
                    let mut chunk_dims = Vec::with_capacity(ndims);
                    for i in 0..ndims {
                        chunk_dims.push(read_u32_le(data, dims_start + i * 4)?);
                    }
                    Ok(DataLayoutResult::Chunked {
                        btree_addr,
                        chunk_dims,
                    })
                }
                _ => Ok(DataLayoutResult::Unsupported),
            }
        }
        _ => Ok(DataLayoutResult::Unsupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_uint() {
        let data = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert_eq!(read_uint(&data, 0, 1).unwrap(), 0x01);
        assert_eq!(read_uint(&data, 0, 2).unwrap(), 0x0201);
        assert_eq!(read_uint(&data, 0, 4).unwrap(), 0x04030201);
        assert_eq!(read_uint(&data, 0, 8).unwrap(), 0x0807060504030201);
    }

    #[test]
    fn test_read_cstring() {
        let data = b"hello\0world\0";
        assert_eq!(read_cstring(data, 0).unwrap(), "hello");
        assert_eq!(read_cstring(data, 6).unwrap(), "world");
    }

    #[test]
    fn test_find_superblock_at_zero() {
        let mut data = vec![0u8; 1024];
        data[..8].copy_from_slice(HDF5_MAGIC);
        assert_eq!(find_superblock(&data).unwrap(), 0);
    }

    #[test]
    fn test_find_superblock_at_512() {
        let mut data = vec![0u8; 1024];
        data[512..520].copy_from_slice(HDF5_MAGIC);
        assert_eq!(find_superblock(&data).unwrap(), 512);
    }

    #[test]
    fn test_find_superblock_missing() {
        let data = vec![0u8; 1024];
        assert!(find_superblock(&data).is_err());
    }

    #[test]
    fn test_parse_datatype_f32() {
        let data = [
            0x11, 0x20, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x17, 0x08,
            0x00, 0x00, 0x7f, 0x00, 0x00, 0x00,
        ];
        assert_eq!(parse_datatype(&data, 0).unwrap(), DType::F32);
    }

    #[test]
    fn test_parse_datatype_i64_signed() {
        let data = [
            0x00, 0x08, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00,
        ];
        assert_eq!(parse_datatype(&data, 0).unwrap(), DType::I64);
    }

    #[test]
    fn test_parse_datatype_u32_unsigned() {
        let data = [
            0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00,
        ];
        assert_eq!(parse_datatype(&data, 0).unwrap(), DType::U32);
    }

    #[test]
    fn test_unshuffle() {
        // 3 elements of 4 bytes each
        // Shuffled: all byte-0s, all byte-1s, all byte-2s, all byte-3s
        let shuffled = vec![
            0x01, 0x05, 0x09, // byte 0 of elements 0,1,2
            0x02, 0x06, 0x0A, // byte 1
            0x03, 0x07, 0x0B, // byte 2
            0x04, 0x08, 0x0C, // byte 3
        ];
        let unshuffled = unshuffle(&shuffled, 4);
        assert_eq!(
            unshuffled,
            vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C]
        );
    }
}
