//! HDF5 format reader.
//!
//! Provides read-only access to `.h5` / `.hdf5` files through the unified
//! `TensorReader` API. Uses memory mapping for zero-copy access.
//!
//! Scoped to superblock v0/v1, contiguous storage layout, and IEEE
//! float/integer datatypes. This covers the common case of Keras/h5py
//! model weight files saved with default settings.
//!
//! Requires the `hdf5` feature.

use std::collections::BTreeMap;
use std::fs::File;
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
const MSG_ATTRIBUTE: u16 = 0x000C;
const MSG_CONTINUATION: u16 = 0x0010;
const MSG_SYMBOL_TABLE: u16 = 0x0011;

// ---- HDF5 datatype classes ----

const DT_CLASS_FIXED_POINT: u8 = 0; // integer
const DT_CLASS_FLOATING_POINT: u8 = 1; // float

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
    String::from_utf8(data[start..end].to_vec()).map_err(|e| {
        Error::InvalidFileStructure(format!("Invalid UTF-8 in HDF5 string: {}", e))
    })
}

// ---- Parsed dataset info ----

/// Info extracted from an HDF5 dataset's object header.
struct DatasetInfo {
    dtype: DType,
    shape: Vec<u64>,
    data_addr: u64,
    data_size: u64,
}

// ---- Hdf5Reader ----

/// Reader for HDF5 (.h5 / .hdf5) files.
///
/// Uses memory mapping for efficient zero-copy access to tensor data.
/// Supports superblock v0/v1 with contiguous storage and IEEE numeric types.
pub struct Hdf5Reader {
    mmap: Mmap,
    pub manifest: Manifest,
    /// Maps tensor name -> (byte_offset, byte_length) in the mmap.
    data_ranges: BTreeMap<String, (usize, usize)>,
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
        //    HDF5 superblock can be at offset 0, 512, 1024, 2048, ... (powers of 2 starting at 512)
        let sb_offset = find_superblock(data)?;

        // 2. Parse superblock v0 or v1
        let (ctx, root_btree_addr, root_heap_addr) = parse_superblock(data, sb_offset)?;

        // 3. Traverse the root group to collect datasets
        let mut objects = BTreeMap::new();
        let mut data_ranges = BTreeMap::new();

        traverse_group(
            data,
            &ctx,
            root_btree_addr,
            root_heap_addr,
            "",
            &mut objects,
            &mut data_ranges,
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
            data_ranges,
        })
    }

    /// Gets a zero-copy reference to an object's raw data.
    pub fn view(&self, name: &str) -> Result<&[u8], Error> {
        let (offset, length) = self
            .data_ranges
            .get(name)
            .ok_or_else(|| Error::ObjectNotFound(name.to_string()))?;
        Ok(&self.mmap[*offset..*offset + *length])
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

    // Superblock layout after magic (at pos = sb_offset + 8):
    //   +0: sb_version (1)
    //   +1: free_space_version (1)
    //   +2: root_group_sttab_version (1)
    //   +3: reserved (1)
    //   +4: shared_header_msg_version (1)
    //   +5: sizeof_offsets (1)
    //   +6: sizeof_lengths (1)
    //   +7: reserved (1)
    //   +8..9: group_leaf_node_K (2)
    //   +10..11: group_internal_node_K (2)
    //   +12..15: consistency_flags (4)                [v0]
    //     OR
    //   +12..13: indexed_storage_internal_node_K (2)  [v1]
    //   +14..15: reserved (2)                         [v1]
    //   +16..19: consistency_flags (4)                [v1]
    let offset_size = read_u8(data, pos + 5)? as usize;
    let length_size = read_u8(data, pos + 6)? as usize;

    if offset_size < 1 || offset_size > 8 || length_size < 1 || length_size > 8 {
        return Err(Error::InvalidFileStructure(format!(
            "Invalid HDF5 offset/length sizes: O={}, L={}",
            offset_size, length_size
        )));
    }

    let var_start = if sb_version == 0 {
        pos + 8 + 2 + 2 + 4 // 8 fixed bytes + leaf_K(2) + internal_K(2) + flags(4)
    } else {
        pos + 8 + 2 + 2 + 2 + 2 + 4 // + indexed_K(2) + reserved(2) + flags(4)
    };

    let ctx = Hdf5Ctx {
        offset_size,
        length_size,
    };

    // Variable-width fields: base_addr, <free_space_addr>, <end_of_file_addr>, <driver_info_addr>
    // base_address (O), free_space_addr (O), eof_addr (O), driver_info_addr (O)
    // Then: root group symbol table entry (symbol_table_entry size = O + O + 24 for cache)
    let root_entry_offset = var_start + 4 * offset_size;

    // Root group symbol table entry:
    //   link_name_offset (O) — always 0
    //   object_header_addr (O)
    //   cache_type (4 bytes)
    //   reserved (4 bytes)
    //   scratch-pad (16 bytes): btree_addr (O) + heap_addr (O)
    let _link_name_offset = ctx.read_offset(data, root_entry_offset)?;
    let _obj_header_addr = ctx.read_offset(data, root_entry_offset + offset_size)?;
    let cache_type = read_u32_le(data, root_entry_offset + 2 * offset_size)?;

    if cache_type != 1 {
        return Err(Error::InvalidFileStructure(format!(
            "Root symbol table entry cache type {} (expected 1 for group)",
            cache_type
        )));
    }

    // scratch-pad starts after cache_type(4) + reserved(4)
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

    // Validate signature
    if &data[pos..pos + 4] != HEAP_SIGNATURE {
        return Err(Error::InvalidFileStructure(format!(
            "Expected HEAP signature at offset {}", pos
        )));
    }

    // HEAP: sig(4) + version(1) + reserved(3) + data_seg_size(L) + free_list_head_offset(L) + data_seg_addr(O)
    let data_seg_addr_offset = pos.checked_add(4 + 1 + 3 + ctx.length_size + ctx.length_size)
        .ok_or(Error::UnexpectedEof)?;
    let data_seg_addr = ctx.read_offset(data, data_seg_addr_offset)?;

    Ok(data_seg_addr as usize)
}

// ---- B-tree v1 traversal ----

/// Traverses a v1 B-tree (type 0, group) to collect SNODs.
fn traverse_btree(
    data: &[u8],
    ctx: &Hdf5Ctx,
    btree_addr: u64,
    heap_data_addr: usize,
    prefix: &str,
    objects: &mut BTreeMap<String, Object>,
    data_ranges: &mut BTreeMap<String, (usize, usize)>,
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

    // Validate signature
    if &data[pos..pos + 4] != TREE_SIGNATURE {
        return Err(Error::InvalidFileStructure(format!(
            "Expected TREE signature at offset {}", pos
        )));
    }

    let node_type = read_u8(data, pos + 4)?;
    if node_type != 0 {
        // Type 0 = group nodes. Type 1 = raw data chunks (not needed for contiguous).
        return Ok(());
    }

    let node_level = read_u8(data, pos + 5)?;
    let entries_used = read_u16_le(data, pos + 6)? as usize;
    let _left_sibling = ctx.read_offset(data, pos + 8)?;
    let _right_sibling = ctx.read_offset(data, pos + 8 + ctx.offset_size)?;

    // Keys and child pointers
    // For group B-trees (type 0): key = size_of_lengths bytes (offset into local heap)
    // Layout: (entries_used+1) keys interleaved with entries_used child pointers
    // key[0], child[0], key[1], child[1], ..., key[entries_used]
    let keys_start = pos + 8 + 2 * ctx.offset_size;

    if node_level > 0 {
        // Internal node: children are B-tree nodes
        for i in 0..entries_used {
            let child_offset = keys_start + ctx.length_size + i * (ctx.length_size + ctx.offset_size);
            let child_addr = ctx.read_offset(data, child_offset)?;
            if child_addr != UNDEF_ADDR {
                traverse_btree(
                    data,
                    ctx,
                    child_addr,
                    heap_data_addr,
                    prefix,
                    objects,
                    data_ranges,
                    depth + 1,
                )?;
            }
        }
    } else {
        // Leaf node: children are SNOD addresses
        for i in 0..entries_used {
            let child_offset = keys_start + ctx.length_size + i * (ctx.length_size + ctx.offset_size);
            let snod_addr = ctx.read_offset(data, child_offset)?;
            if snod_addr != UNDEF_ADDR {
                parse_snod(
                    data,
                    ctx,
                    snod_addr,
                    heap_data_addr,
                    prefix,
                    objects,
                    data_ranges,
                    depth + 1,
                )?;
            }
        }
    }

    Ok(())
}

// ---- Symbol table node (SNOD) ----

fn parse_snod(
    data: &[u8],
    ctx: &Hdf5Ctx,
    snod_addr: u64,
    heap_data_addr: usize,
    prefix: &str,
    objects: &mut BTreeMap<String, Object>,
    data_ranges: &mut BTreeMap<String, (usize, usize)>,
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

    // Validate signature
    if &data[pos..pos + 4] != SNOD_SIGNATURE {
        return Err(Error::InvalidFileStructure(format!(
            "Expected SNOD signature at offset {}", pos
        )));
    }

    let _version = read_u8(data, pos + 4)?;
    let _reserved = read_u8(data, pos + 5)?;
    let num_symbols = read_u16_le(data, pos + 6)? as usize;

    // Symbol table entries start at offset 8
    // Each entry: link_name_offset(O) + obj_header_addr(O) + cache_type(4) + reserved(4) + scratch(16)
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

        // Read name from local heap
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
            // Group: scratch-pad has btree_addr and heap_addr
            let scratch_pos = entry_pos + 2 * ctx.offset_size + 8;
            let child_btree = ctx.read_offset(data, scratch_pos)?;
            let child_heap = ctx.read_offset(data, scratch_pos + ctx.offset_size)?;

            let child_heap_data = read_local_heap_data(data, ctx, child_heap)?;
            traverse_btree(
                data,
                ctx,
                child_btree,
                child_heap_data,
                &full_name,
                objects,
                data_ranges,
                depth + 1,
            )?;
        } else {
            // Dataset (or other): parse object header
            match parse_object_header(data, ctx, obj_header_addr as usize, &full_name, depth + 1)? {
                ObjectHeaderResult::Dataset(info) => {
                    let abs_offset = info.data_addr as usize;
                    let byte_size = info.data_size as usize;

                    if abs_offset + byte_size > data.len() {
                        return Err(Error::InvalidFileStructure(format!(
                            "Dataset '{}' extends beyond file: offset {} + size {} > file size {}",
                            full_name, abs_offset, byte_size, data.len()
                        )));
                    }

                    let obj = Object::dense(info.shape, info.dtype, abs_offset as u64, byte_size as u64);
                    data_ranges.insert(full_name.clone(), (abs_offset, byte_size));
                    objects.insert(full_name, obj);
                }
                ObjectHeaderResult::Group(btree, heap) => {
                    let child_heap_data = read_local_heap_data(data, ctx, heap)?;
                    traverse_btree(
                        data,
                        ctx,
                        btree,
                        child_heap_data,
                        &full_name,
                        objects,
                        data_ranges,
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
    data_ranges: &mut BTreeMap<String, (usize, usize)>,
    depth: usize,
) -> Result<(), Error> {
    let heap_data_addr = read_local_heap_data(data, ctx, heap_addr)?;
    traverse_btree(
        data,
        ctx,
        btree_addr,
        heap_data_addr,
        prefix,
        objects,
        data_ranges,
        depth,
    )
}

// ---- Object header parsing ----

enum ObjectHeaderResult {
    Dataset(DatasetInfo),
    Group(u64, u64), // btree_addr, heap_addr
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

    // v1 object header prefix: version(1) + reserved(1) + num_messages(2) + obj_ref_count(4) + obj_header_size(4)
    let version = read_u8(data, addr)?;
    if version != 1 {
        // v2 object headers not supported
        return Ok(ObjectHeaderResult::Skip);
    }

    let num_messages = read_u16_le(data, addr + 2)? as usize;
    let header_size = read_u32_le(data, addr + 8)? as usize;

    // Messages start after the 12-byte prefix, but we also need to handle padding
    // v1: prefix is 16 bytes (12 + 4 bytes padding to 8-byte boundary)
    let msg_start = addr + 16;
    let msg_end = msg_start + header_size;

    let mut dtype: Option<DType> = None;
    let mut shape: Option<Vec<u64>> = None;
    let mut data_addr: Option<u64> = None;
    let mut data_size: Option<u64> = None;
    let mut is_contiguous = false;
    let mut group_info: Option<(u64, u64)> = None;

    parse_messages(
        data, ctx, msg_start, msg_end, num_messages,
        &mut dtype, &mut shape, &mut data_addr, &mut data_size,
        &mut is_contiguous, &mut group_info, name, depth,
    )?;

    // If a symbol table message was found, this is a group
    if let Some((btree, heap)) = group_info {
        return Ok(ObjectHeaderResult::Group(btree, heap));
    }

    if let (Some(dt), Some(sh), Some(da), Some(ds)) = (dtype, shape, data_addr, data_size) {
        if !is_contiguous {
            return Ok(ObjectHeaderResult::Skip);
        }
        if da == UNDEF_ADDR {
            return Ok(ObjectHeaderResult::Skip);
        }
        Ok(ObjectHeaderResult::Dataset(DatasetInfo {
            dtype: dt,
            shape: sh,
            data_addr: da,
            data_size: ds,
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
    data_addr: &mut Option<u64>,
    data_size: &mut Option<u64>,
    is_contiguous: &mut bool,
    group_info: &mut Option<(u64, u64)>,
    name: &str,
    depth: usize,
) -> Result<(), Error> {
    let mut pos = start;
    let mut messages_parsed = 0;

    while pos + 8 <= end && messages_parsed < max_messages {
        // v1 message header: type(2) + size(2) + flags(1) + reserved(3)
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
                let layout = parse_data_layout(data, ctx, msg_data_start)?;
                if let Some((addr, size, contiguous)) = layout {
                    *data_addr = Some(addr);
                    *data_size = Some(size);
                    *is_contiguous = contiguous;
                }
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
                        data_addr,
                        data_size,
                        is_contiguous,
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
            MSG_ATTRIBUTE | 0 => {
                // Skip attributes and nil messages
            }
            _ => {
                // Skip unknown message types
            }
        }

        // Advance to next message (8-byte aligned)
        let next = msg_data_start + msg_size;
        pos = (next + 7) & !7;
        messages_parsed += 1;
    }

    Ok(())
}

// ---- Dataspace parsing ----

fn parse_dataspace(data: &[u8], pos: usize) -> Result<Vec<u64>, Error> {
    let version = read_u8(data, pos)?;
    let ndims = read_u8(data, pos + 1)? as usize;

    match version {
        1 => {
            // v1: version(1) + ndims(1) + flags(1) + reserved(5) + dims(ndims*8) + [max_dims]
            let dims_start = pos + 8;
            let mut shape = Vec::with_capacity(ndims);
            for i in 0..ndims {
                shape.push(read_u64_le(data, dims_start + i * 8)?);
            }
            Ok(shape)
        }
        2 => {
            // v2: version(1) + ndims(1) + flags(1) + type(1) + dims(ndims*8) + [max_dims]
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

    // Datatype message: class_and_version(1) + class_bit_fields(3) + size(4)
    let class_and_version = read_u8(data, pos)?;
    let dt_class = class_and_version & 0x0F;
    let _dt_version = (class_and_version >> 4) & 0x0F;
    let class_bits_0 = read_u8(data, pos + 1)?;
    let dt_size = read_u32_le(data, pos + 4)? as usize;

    match dt_class {
        DT_CLASS_FIXED_POINT => {
            // Integer: bit 3 of class_bits_0 = sign (0=unsigned, 1=signed)
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
        DT_CLASS_FLOATING_POINT => {
            match dt_size {
                8 => Ok(DType::F64),
                4 => Ok(DType::F32),
                2 => Ok(DType::F16),
                _ => Err(Error::UnsupportedDType(format!(
                    "HDF5 float: size={}",
                    dt_size
                ))),
            }
        }
        _ => Err(Error::UnsupportedDType(format!(
            "HDF5 datatype class {} not supported (only integer and float)",
            dt_class
        ))),
    }
}

// ---- Data layout parsing ----

/// Parses a data layout message. Returns (address, size, is_contiguous).
fn parse_data_layout(
    data: &[u8],
    ctx: &Hdf5Ctx,
    pos: usize,
) -> Result<Option<(u64, u64, bool)>, Error> {
    let version = read_u8(data, pos)?;

    match version {
        1 | 2 => {
            // v1/v2: version(1) + ndims(1) + layout_class(1) + reserved(5)
            let ndims = read_u8(data, pos + 1)? as usize;
            let layout_class = read_u8(data, pos + 2)?;

            if layout_class != 1 {
                // 0=compact, 1=contiguous, 2=chunked
                return Ok(Some((0, 0, false)));
            }

            // Contiguous: data_addr(O) + dims(ndims * 4)
            let addr_pos = pos + 8;
            let addr = ctx.read_offset(data, addr_pos)?;

            // Size: product of dimension sizes (each 4 bytes) — but last dim is element size
            let dims_start = addr_pos + ctx.offset_size;
            let mut total_size: u64 = 1;
            for i in 0..ndims {
                total_size *= read_u32_le(data, dims_start + i * 4)? as u64;
            }

            Ok(Some((addr, total_size, true)))
        }
        3 => {
            // v3: version(1) + layout_class(1)
            let layout_class = read_u8(data, pos + 1)?;

            match layout_class {
                0 => {
                    // Compact: size(2) + data(size)
                    // Not supported for zero-copy
                    Ok(Some((0, 0, false)))
                }
                1 => {
                    // Contiguous: address(O) + size(L)
                    let addr = ctx.read_offset(data, pos + 2)?;
                    let size = ctx.read_length(data, pos + 2 + ctx.offset_size)?;
                    Ok(Some((addr, size, true)))
                }
                2 => {
                    // Chunked: not supported
                    Ok(Some((0, 0, false)))
                }
                _ => Ok(None),
            }
        }
        _ => Ok(None),
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
        // class=1 (float), version=1, size=4
        let data = [0x11, 0x20, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
                     // properties (bit offset, bit precision, exponent location, etc.)
                     0x00, 0x00, 0x20, 0x00, 0x17, 0x08, 0x00, 0x00,
                     0x7f, 0x00, 0x00, 0x00];
        assert_eq!(parse_datatype(&data, 0).unwrap(), DType::F32);
    }

    #[test]
    fn test_parse_datatype_i64_signed() {
        // class=0 (integer), signed (bit 3 set in class_bits_0), size=8
        let data = [0x00, 0x08, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x40, 0x00];
        assert_eq!(parse_datatype(&data, 0).unwrap(), DType::I64);
    }

    #[test]
    fn test_parse_datatype_u32_unsigned() {
        // class=0 (integer), unsigned (bit 3 clear), size=4
        let data = [0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
                     0x00, 0x00, 0x20, 0x00];
        assert_eq!(parse_datatype(&data, 0).unwrap(), DType::U32);
    }
}
