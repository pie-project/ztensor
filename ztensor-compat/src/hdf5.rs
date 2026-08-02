//! HDF5 → zTensor object model projection.
//!
//! A self-contained parser for the subset that covers Keras/h5py weight
//! files: superblock v0/v1, v1 B-trees and object headers, contiguous and
//! chunked layouts, deflate + shuffle filters, IEEE little-endian
//! float/integer datatypes.
//!
//! Contiguous datasets are zero-copy views into the mmap. Chunked datasets
//! are reassembled (and defiltered) once at open into owned buffers; their
//! parts carry the `hdf5.chunked/1` encoding marker because the stored
//! blob range is not the decoded bytes — `view()` still works, serving the
//! reassembled buffer.
//!
//! Datasets with unsupported datatype classes (strings, compounds) are
//! skipped and listed in [`Hdf5::skipped`]; big-endian numeric data is
//! refused rather than reinterpreted.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use memmap2::Mmap;
use ztensor::{
    BlobRef, Caps, DType, Error, Layout, Manifest, Object, Part, Result, Source,
};

const MAGIC: &[u8; 8] = b"\x89HDF\r\n\x1a\n";
const UNDEF: u64 = u64::MAX;
const MAX_DEPTH: usize = 64;

const MSG_DATASPACE: u16 = 0x0001;
const MSG_DATATYPE: u16 = 0x0003;
const MSG_DATA_LAYOUT: u16 = 0x0008;
const MSG_FILTER_PIPELINE: u16 = 0x000B;
const MSG_CONTINUATION: u16 = 0x0010;
const MSG_SYMBOL_TABLE: u16 = 0x0011;

const FILTER_DEFLATE: u16 = 1;
const FILTER_SHUFFLE: u16 = 2;

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("hdf5: {}", detail.into()))
}

// ---- primitive readers ------------------------------------------------

fn uint(data: &[u8], pos: usize, size: usize) -> Result<u64> {
    let end = pos
        .checked_add(size)
        .filter(|&e| e <= data.len())
        .ok_or_else(|| bad("unexpected end of file"))?;
    let mut v = 0u64;
    for (i, &b) in data[pos..end].iter().enumerate() {
        v |= (b as u64) << (i * 8);
    }
    Ok(v)
}

fn u8_at(data: &[u8], pos: usize) -> Result<u8> {
    data.get(pos)
        .copied()
        .ok_or_else(|| bad("unexpected end of file"))
}

fn u16_at(data: &[u8], pos: usize) -> Result<u16> {
    uint(data, pos, 2).map(|v| v as u16)
}

fn u32_at(data: &[u8], pos: usize) -> Result<u32> {
    uint(data, pos, 4).map(|v| v as u32)
}

fn u64_at(data: &[u8], pos: usize) -> Result<u64> {
    uint(data, pos, 8)
}

fn cstring(data: &[u8], pos: usize) -> Result<String> {
    let end = data[pos..]
        .iter()
        .position(|&b| b == 0)
        .map(|i| pos + i)
        .ok_or_else(|| bad("unterminated string"))?;
    String::from_utf8(data[pos..end].to_vec()).map_err(|_| bad("invalid UTF-8 in name"))
}

/// Superblock parameters used throughout.
#[derive(Clone, Copy)]
struct Ctx {
    o: usize, // offset size
    l: usize, // length size
}

impl Ctx {
    fn offset(&self, data: &[u8], pos: usize) -> Result<u64> {
        uint(data, pos, self.o)
    }
    fn length(&self, data: &[u8], pos: usize) -> Result<u64> {
        uint(data, pos, self.l)
    }
}

// ---- parsed structures ------------------------------------------------

#[derive(Clone)]
struct Filter {
    id: u16,
}

enum DataLayout {
    Contiguous { addr: u64, size: u64 },
    Chunked { btree_addr: u64, chunk_dims: Vec<u32> },
    Unsupported,
}

struct DatasetInfo {
    dtype: DType,
    shape: Vec<u64>,
    layout: DataLayout,
    filters: Vec<Filter>,
}

enum Loc {
    Range { offset: u64, length: u64 },
    Owned(Vec<u8>),
}

pub struct Hdf5 {
    mmap: Mmap,
    manifest: Manifest,
    locations: BTreeMap<String, Loc>,
    skipped: Vec<String>,
}

impl std::fmt::Debug for Hdf5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hdf5")
            .field("len", &self.mmap.len())
            .field("objects", &self.manifest.objects.len())
            .field("skipped", &self.skipped.len())
            .finish()
    }
}

impl Hdf5 {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: read-only shared map of untrusted bytes.
        let mmap = unsafe { Mmap::map(&file)? };
        let mut walker = Walker {
            data: &mmap,
            objects: BTreeMap::new(),
            locations: BTreeMap::new(),
            skipped: Vec::new(),
        };
        let sb = find_superblock(&mmap)?;
        let (ctx, btree, heap) = parse_superblock(&mmap, sb)?;
        walker.group(&ctx, btree, heap, "", 0)?;
        let Walker {
            objects,
            locations,
            skipped,
            ..
        } = walker;
        Ok(Self {
            mmap,
            manifest: Manifest {
                attributes: None,
                shards: BTreeMap::new(),
                objects,
            },
            locations,
            skipped,
        })
    }

    /// Datasets present in the file but not projectable (string/compound
    /// datatypes, unsupported layouts). Listed, never silently absent.
    pub fn skipped(&self) -> &[String] {
        &self.skipped
    }

    fn location(&self, name: &str, part: &str) -> Result<&Loc> {
        if part != "data" {
            return Err(Error::NotFound(format!("part {name:?}/{part:?}")));
        }
        self.locations
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("object {name:?}")))
    }
}

impl Source for Hdf5 {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read(&self, object: &str, part: &str) -> Result<Vec<u8>> {
        Source::view(self, object, part).map(<[u8]>::to_vec)
    }

    fn view(&self, object: &str, part: &str) -> Result<&[u8]> {
        match self.location(object, part)? {
            Loc::Range { offset, length } => {
                Ok(&self.mmap[*offset as usize..(*offset + *length) as usize])
            }
            Loc::Owned(bytes) => Ok(bytes),
        }
    }

    fn caps(&self, object: &str, part: &str) -> Result<Caps> {
        let alignment = match self.location(object, part)? {
            Loc::Range { offset, .. } if *offset > 0 => {
                1u64 << offset.trailing_zeros().min(63)
            }
            _ => 1,
        };
        Ok(Caps {
            zero_copy: true,
            alignment,
            verifiable: false,
            page_exclusive: false,
        })
    }
}

// ---- superblock -------------------------------------------------------

fn find_superblock(data: &[u8]) -> Result<usize> {
    if data.len() >= 8 && &data[..8] == MAGIC {
        return Ok(0);
    }
    let mut off = 512;
    while off + 8 <= data.len() {
        if &data[off..off + 8] == MAGIC {
            return Ok(off);
        }
        off *= 2;
    }
    Err(bad("no superblock signature"))
}

fn parse_superblock(data: &[u8], sb: usize) -> Result<(Ctx, u64, u64)> {
    let pos = sb + 8;
    let version = u8_at(data, pos)?;
    if version > 1 {
        return Err(Error::Unsupported(format!(
            "hdf5: superblock v{version} (only v0/v1)"
        )));
    }
    let o = u8_at(data, pos + 5)? as usize;
    let l = u8_at(data, pos + 6)? as usize;
    if !(1..=8).contains(&o) || !(1..=8).contains(&l) {
        return Err(bad("invalid offset/length sizes"));
    }
    let ctx = Ctx { o, l };
    let var_start = if version == 0 {
        pos + 8 + 2 + 2 + 4
    } else {
        pos + 8 + 2 + 2 + 2 + 2 + 4
    };
    let root_entry = var_start + 4 * o;
    let cache_type = u32_at(data, root_entry + 2 * o)?;
    if cache_type != 1 {
        return Err(bad("root symbol table entry is not a cached group"));
    }
    let scratch = root_entry + 2 * o + 8;
    Ok((ctx, ctx.offset(data, scratch)?, ctx.offset(data, scratch + o)?))
}

fn heap_data_addr(data: &[u8], ctx: &Ctx, heap_addr: u64) -> Result<usize> {
    let pos = heap_addr as usize;
    if data.len() < pos + 4 || &data[pos..pos + 4] != b"HEAP" {
        return Err(bad("missing HEAP signature"));
    }
    let addr = ctx.offset(data, pos + 4 + 1 + 3 + ctx.l + ctx.l)?;
    Ok(addr as usize)
}

// ---- group traversal --------------------------------------------------

struct Walker<'a> {
    data: &'a [u8],
    objects: BTreeMap<String, Object>,
    locations: BTreeMap<String, Loc>,
    skipped: Vec<String>,
}

impl Walker<'_> {
    fn group(&mut self, ctx: &Ctx, btree: u64, heap: u64, prefix: &str, depth: usize) -> Result<()> {
        let heap_data = heap_data_addr(self.data, ctx, heap)?;
        self.btree_group(ctx, btree, heap_data, prefix, depth)
    }

    fn btree_group(
        &mut self,
        ctx: &Ctx,
        btree: u64,
        heap_data: usize,
        prefix: &str,
        depth: usize,
    ) -> Result<()> {
        if depth > MAX_DEPTH {
            return Err(bad("B-tree recursion too deep"));
        }
        let data = self.data;
        let pos = btree as usize;
        if data.len() < pos + 4 || &data[pos..pos + 4] != b"TREE" {
            return Err(bad("missing TREE signature"));
        }
        if u8_at(data, pos + 4)? != 0 {
            return Ok(()); // not a group B-tree
        }
        let level = u8_at(data, pos + 5)?;
        let entries = u16_at(data, pos + 6)? as usize;
        let keys_start = pos + 8 + 2 * ctx.o;
        for i in 0..entries {
            let child = ctx.offset(data, keys_start + ctx.l + i * (ctx.l + ctx.o))?;
            if child == UNDEF {
                continue;
            }
            if level > 0 {
                self.btree_group(ctx, child, heap_data, prefix, depth + 1)?;
            } else {
                self.snod(ctx, child, heap_data, prefix, depth + 1)?;
            }
        }
        Ok(())
    }

    fn snod(
        &mut self,
        ctx: &Ctx,
        snod: u64,
        heap_data: usize,
        prefix: &str,
        depth: usize,
    ) -> Result<()> {
        if depth > MAX_DEPTH {
            return Err(bad("SNOD recursion too deep"));
        }
        let data = self.data;
        let pos = snod as usize;
        if data.len() < pos + 4 || &data[pos..pos + 4] != b"SNOD" {
            return Err(bad("missing SNOD signature"));
        }
        let count = u16_at(data, pos + 6)? as usize;
        let entry_size = 2 * ctx.o + 4 + 4 + 16;
        for i in 0..count {
            let e = pos + 8 + i * entry_size;
            if e + entry_size > data.len() {
                break;
            }
            let link_name = ctx.offset(data, e)?;
            let header_addr = ctx.offset(data, e + ctx.o)?;
            let cache_type = u32_at(data, e + 2 * ctx.o)?;
            if header_addr == UNDEF || header_addr == 0 {
                continue;
            }
            let name = cstring(data, heap_data + link_name as usize)?;
            if name.is_empty() {
                continue;
            }
            let full = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };

            if cache_type == 1 {
                let scratch = e + 2 * ctx.o + 8;
                let btree = ctx.offset(data, scratch)?;
                let heap = ctx.offset(data, scratch + ctx.o)?;
                self.group(ctx, btree, heap, &full, depth + 1)?;
            } else {
                match parse_object_header(data, ctx, header_addr as usize, depth + 1)? {
                    Header::Group(btree, heap) => self.group(ctx, btree, heap, &full, depth + 1)?,
                    Header::Dataset(info) => self.dataset(ctx, &full, info)?,
                    Header::Skip => self.skipped.push(full),
                }
            }
        }
        Ok(())
    }

    fn dataset(&mut self, ctx: &Ctx, name: &str, info: DatasetInfo) -> Result<()> {
        let elems = info
            .shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| bad(format!("dataset {name:?} shape overflows")))?;
        let expected = elems
            .checked_mul(info.dtype.width())
            .ok_or_else(|| bad(format!("dataset {name:?} size overflows")))?;

        let (loc, blob, encoding) = match info.layout {
            DataLayout::Contiguous { addr, size } => {
                if addr == UNDEF {
                    self.skipped.push(name.to_string());
                    return Ok(());
                }
                if size != expected {
                    return Err(bad(format!(
                        "dataset {name:?} stores {size} bytes but shape implies {expected}"
                    )));
                }
                if addr + size > self.data.len() as u64 {
                    return Err(bad(format!("dataset {name:?} extends past file")));
                }
                (
                    Loc::Range {
                        offset: addr,
                        length: size,
                    },
                    BlobRef {
                        shard: 0,
                        offset: addr,
                        length: size,
                    },
                    None,
                )
            }
            DataLayout::Chunked {
                btree_addr,
                chunk_dims,
            } => {
                let bytes = read_chunked(
                    self.data,
                    ctx,
                    btree_addr,
                    &info.shape,
                    &chunk_dims,
                    info.dtype,
                    &info.filters,
                )?;
                if bytes.len() as u64 != expected {
                    return Err(bad(format!("dataset {name:?} reassembly size mismatch")));
                }
                (
                    Loc::Owned(bytes),
                    BlobRef {
                        shard: 0,
                        offset: 0,
                        length: 0,
                    },
                    Some("hdf5.chunked/1".to_string()),
                )
            }
            DataLayout::Unsupported => {
                self.skipped.push(name.to_string());
                return Ok(());
            }
        };

        let part = Part {
            dtype: info.dtype,
            ltype: None,
            blob,
            decoded_length: encoding.as_ref().map(|_| expected),
            encoding,
            digest: None,
        };
        let mut parts = BTreeMap::new();
        parts.insert("data".to_string(), part);
        self.locations.insert(name.to_string(), loc);
        self.objects.insert(
            name.to_string(),
            Object {
                shape: info.shape,
                layout: Layout::Dense,
                attributes: None,
                parts,
            },
        );
        Ok(())
    }
}

// ---- object header ----------------------------------------------------

enum Header {
    Dataset(DatasetInfo),
    Group(u64, u64),
    Skip,
}

fn parse_object_header(data: &[u8], ctx: &Ctx, addr: usize, depth: usize) -> Result<Header> {
    if depth > MAX_DEPTH {
        return Err(bad("object header recursion too deep"));
    }
    if u8_at(data, addr)? != 1 {
        return Ok(Header::Skip);
    }
    let num_messages = u16_at(data, addr + 2)? as usize;
    let header_size = u32_at(data, addr + 8)? as usize;

    let mut st = MessageState::default();
    parse_messages(
        data,
        ctx,
        addr + 16,
        addr + 16 + header_size,
        num_messages,
        &mut st,
        depth,
    )?;

    if let Some((btree, heap)) = st.group {
        return Ok(Header::Group(btree, heap));
    }
    match (st.dtype, st.shape, st.layout) {
        (Some(dtype), Some(shape), Some(layout)) => Ok(Header::Dataset(DatasetInfo {
            dtype,
            shape,
            layout,
            filters: st.filters,
        })),
        _ => Ok(Header::Skip),
    }
}

#[derive(Default)]
struct MessageState {
    dtype: Option<DType>,
    shape: Option<Vec<u64>>,
    layout: Option<DataLayout>,
    filters: Vec<Filter>,
    group: Option<(u64, u64)>,
}

fn parse_messages(
    data: &[u8],
    ctx: &Ctx,
    start: usize,
    end: usize,
    max_messages: usize,
    st: &mut MessageState,
    depth: usize,
) -> Result<()> {
    let mut pos = start;
    let mut parsed = 0;
    while pos + 8 <= end && parsed < max_messages {
        let msg_type = u16_at(data, pos)?;
        let msg_size = u16_at(data, pos + 2)? as usize;
        let body = pos + 8;
        if body + msg_size > data.len() {
            break;
        }
        match msg_type {
            MSG_DATASPACE => st.shape = Some(parse_dataspace(data, body)?),
            MSG_DATATYPE => st.dtype = parse_datatype(data, body)?,
            MSG_DATA_LAYOUT => st.layout = Some(parse_layout(data, ctx, body)?),
            MSG_FILTER_PIPELINE => st.filters = parse_filters(data, body)?,
            MSG_SYMBOL_TABLE => {
                st.group = Some((
                    ctx.offset(data, body)?,
                    ctx.offset(data, body + ctx.o)?,
                ));
            }
            MSG_CONTINUATION => {
                let cont = ctx.offset(data, body)?;
                let len = ctx.offset(data, body + ctx.o)?;
                if cont != UNDEF && len > 0 && depth < MAX_DEPTH {
                    parse_messages(
                        data,
                        ctx,
                        cont as usize,
                        (cont + len) as usize,
                        max_messages - parsed,
                        st,
                        depth + 1,
                    )?;
                }
            }
            _ => {}
        }
        pos = (body + msg_size + 7) & !7;
        parsed += 1;
    }
    Ok(())
}

fn parse_dataspace(data: &[u8], pos: usize) -> Result<Vec<u64>> {
    let version = u8_at(data, pos)?;
    let ndims = u8_at(data, pos + 1)? as usize;
    let dims_start = match version {
        1 => pos + 8,
        2 => pos + 4,
        v => return Err(Error::Unsupported(format!("hdf5: dataspace v{v}"))),
    };
    (0..ndims).map(|i| u64_at(data, dims_start + i * 8)).collect()
}

/// Returns `Ok(None)` for datatype classes without a projection (strings,
/// compounds — the dataset is skipped); errors for numeric-but-unreadable
/// (big-endian) data.
fn parse_datatype(data: &[u8], pos: usize) -> Result<Option<DType>> {
    let class = u8_at(data, pos)? & 0x0f;
    let bits0 = u8_at(data, pos + 1)?;
    let size = u32_at(data, pos + 4)? as usize;
    if class > 1 {
        return Ok(None);
    }
    if bits0 & 0x01 != 0 && size > 1 {
        return Err(Error::Unsupported(
            "hdf5: big-endian data is refused, not byte-swapped silently".into(),
        ));
    }
    let dtype = match (class, size, bits0 & 0x08 != 0) {
        (0, 8, true) => DType::I64,
        (0, 4, true) => DType::I32,
        (0, 2, true) => DType::I16,
        (0, 1, true) => DType::I8,
        (0, 8, false) => DType::U64,
        (0, 4, false) => DType::U32,
        (0, 2, false) => DType::U16,
        (0, 1, false) => DType::U8,
        (1, 8, _) => DType::F64,
        (1, 4, _) => DType::F32,
        (1, 2, _) => DType::F16,
        _ => return Ok(None),
    };
    Ok(Some(dtype))
}

fn parse_layout(data: &[u8], ctx: &Ctx, pos: usize) -> Result<DataLayout> {
    let version = u8_at(data, pos)?;
    Ok(match version {
        1 | 2 => {
            let ndims = u8_at(data, pos + 1)? as usize;
            let class = u8_at(data, pos + 2)?;
            let addr_pos = pos + 8;
            match class {
                1 => {
                    let addr = ctx.offset(data, addr_pos)?;
                    let mut size = 1u64;
                    for i in 0..ndims {
                        size = size
                            .checked_mul(u32_at(data, addr_pos + ctx.o + i * 4)? as u64)
                            .ok_or_else(|| bad("layout size overflow"))?;
                    }
                    DataLayout::Contiguous { addr, size }
                }
                2 => {
                    let btree_addr = ctx.offset(data, addr_pos)?;
                    let chunk_dims = (0..ndims.saturating_sub(1))
                        .map(|i| u32_at(data, addr_pos + ctx.o + i * 4))
                        .collect::<Result<_>>()?;
                    DataLayout::Chunked {
                        btree_addr,
                        chunk_dims,
                    }
                }
                _ => DataLayout::Unsupported,
            }
        }
        3 => {
            let class = u8_at(data, pos + 1)?;
            match class {
                1 => DataLayout::Contiguous {
                    addr: ctx.offset(data, pos + 2)?,
                    size: ctx.length(data, pos + 2 + ctx.o)?,
                },
                2 => {
                    let dim = u8_at(data, pos + 2)? as usize;
                    let btree_addr = ctx.offset(data, pos + 3)?;
                    let chunk_dims = (0..dim.saturating_sub(1))
                        .map(|i| u32_at(data, pos + 3 + ctx.o + i * 4))
                        .collect::<Result<_>>()?;
                    DataLayout::Chunked {
                        btree_addr,
                        chunk_dims,
                    }
                }
                _ => DataLayout::Unsupported,
            }
        }
        _ => DataLayout::Unsupported,
    })
}

fn parse_filters(data: &[u8], pos: usize) -> Result<Vec<Filter>> {
    let version = u8_at(data, pos)?;
    let n = u8_at(data, pos + 1)? as usize;
    let mut filters = Vec::with_capacity(n);
    let mut fpos = if version == 1 { pos + 8 } else { pos + 2 };
    for _ in 0..n {
        if fpos + 8 > data.len() {
            break;
        }
        let id = u16_at(data, fpos)?;
        if version == 1 || id >= 256 {
            let name_len = u16_at(data, fpos + 2)? as usize;
            let cd_n = u16_at(data, fpos + 6)? as usize;
            fpos += 8 + ((name_len + 7) & !7) + cd_n * 4;
            if version == 1 && !cd_n.is_multiple_of(2) {
                fpos += 4;
            }
        } else {
            let cd_n = u16_at(data, fpos + 4)? as usize;
            fpos += 6 + cd_n * 4;
        }
        filters.push(Filter { id });
    }
    Ok(filters)
}

// ---- chunked reassembly -----------------------------------------------

struct Chunk {
    linear_offset: u64,
    file_addr: u64,
    size: u32,
    filter_mask: u32,
}

#[allow(clippy::too_many_arguments)]
fn collect_chunks(
    data: &[u8],
    ctx: &Ctx,
    btree: u64,
    ndims: usize,
    shape: &[u64],
    element_size: usize,
    depth: usize,
    out: &mut Vec<Chunk>,
) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(bad("chunk B-tree recursion too deep"));
    }
    let pos = btree as usize;
    if data.len() < pos + 4 || &data[pos..pos + 4] != b"TREE" {
        return Err(bad("missing chunk TREE signature"));
    }
    if u8_at(data, pos + 4)? != 1 {
        return Err(bad("chunk B-tree has wrong node type"));
    }
    let level = u8_at(data, pos + 5)?;
    let entries = u16_at(data, pos + 6)? as usize;
    let key_size = 4 + 4 + (ndims + 1) * 8;
    let keys_start = pos + 8 + 2 * ctx.o;
    for i in 0..entries {
        if level > 0 {
            let child = ctx.offset(data, keys_start + key_size + i * (key_size + ctx.o))?;
            if child != UNDEF {
                collect_chunks(data, ctx, child, ndims, shape, element_size, depth + 1, out)?;
            }
        } else {
            let k = keys_start + i * (key_size + ctx.o);
            let size = u32_at(data, k)?;
            let filter_mask = u32_at(data, k + 4)?;
            let mut linear = 0u64;
            let mut stride = element_size as u64;
            for d in (0..ndims).rev() {
                linear += u64_at(data, k + 8 + d * 8)? * stride;
                stride = stride
                    .checked_mul(shape[d])
                    .ok_or_else(|| bad("chunk offset overflow"))?;
            }
            let file_addr = ctx.offset(data, k + key_size)?;
            if file_addr != UNDEF {
                out.push(Chunk {
                    linear_offset: linear,
                    file_addr,
                    size,
                    filter_mask,
                });
            }
        }
    }
    Ok(())
}

fn apply_filters(
    mut bytes: Vec<u8>,
    filters: &[Filter],
    mask: u32,
    element_size: usize,
) -> Result<Vec<u8>> {
    for (i, filter) in filters.iter().enumerate().rev() {
        if mask & (1 << i) != 0 {
            continue;
        }
        bytes = match filter.id {
            FILTER_DEFLATE => {
                let mut out = Vec::new();
                flate2::read::ZlibDecoder::new(bytes.as_slice())
                    .read_to_end(&mut out)
                    .map_err(|e| bad(format!("deflate: {e}")))?;
                out
            }
            FILTER_SHUFFLE => unshuffle(&bytes, element_size),
            other => {
                return Err(Error::Unsupported(format!("hdf5: filter id {other}")));
            }
        };
    }
    Ok(bytes)
}

fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }
    let n = data.len() / element_size;
    let mut out = vec![0u8; data.len()];
    for i in 0..n {
        for b in 0..element_size {
            out[i * element_size + b] = data[b * n + i];
        }
    }
    out
}

fn read_chunked(
    data: &[u8],
    ctx: &Ctx,
    btree: u64,
    shape: &[u64],
    chunk_dims: &[u32],
    dtype: DType,
    filters: &[Filter],
) -> Result<Vec<u8>> {
    let ndims = shape.len();
    let esize = dtype.width() as usize;
    let total = shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .and_then(|n| n.checked_mul(esize as u64))
        .ok_or_else(|| bad("dataset size overflow"))? as usize;

    let mut chunks = Vec::new();
    collect_chunks(data, ctx, btree, ndims, shape, esize, 0, &mut chunks)?;
    chunks.sort_by_key(|c| c.linear_offset);

    let mut out = vec![0u8; total];
    for chunk in &chunks {
        let addr = chunk.file_addr as usize;
        let end = addr
            .checked_add(chunk.size as usize)
            .filter(|&e| e <= data.len())
            .ok_or_else(|| bad("chunk extends past file"))?;
        let raw = data[addr..end].to_vec();
        let bytes = if filters.is_empty() {
            raw
        } else {
            apply_filters(raw, filters, chunk.filter_mask, esize)?
        };

        let offset = chunk.linear_offset as usize;
        if ndims <= 1 {
            let n = bytes.len().min(total.saturating_sub(offset));
            out[offset..offset + n].copy_from_slice(&bytes[..n]);
        } else {
            let mut rem = offset / esize;
            let mut start = vec![0u64; ndims];
            for d in (0..ndims).rev() {
                start[d] = rem as u64 % shape[d];
                rem /= shape[d] as usize;
            }
            copy_chunk(&bytes, &mut out, shape, chunk_dims, &start, esize);
        }
    }
    Ok(out)
}

fn copy_chunk(
    chunk: &[u8],
    out: &mut [u8],
    shape: &[u64],
    chunk_dims: &[u32],
    start: &[u64],
    esize: usize,
) {
    let ndims = shape.len();
    let actual: Vec<usize> = (0..ndims)
        .map(|d| (chunk_dims[d] as u64).min(shape[d] - start[d]) as usize)
        .collect();
    let row_len = actual[ndims - 1] * esize;
    let rows: usize = actual[..ndims - 1].iter().product();

    for row in 0..rows {
        let mut rem = row;
        let mut src = 0usize;
        let mut dst = 0u64;
        for d in (0..ndims - 1).rev() {
            let c = rem % actual[d];
            rem /= actual[d];
            let mut cstride = chunk_dims[ndims - 1] as usize * esize;
            for &cd in &chunk_dims[d + 1..ndims - 1] {
                cstride *= cd as usize;
            }
            src += c * cstride;
            let mut dstride = 1u64;
            for &s in &shape[d + 1..ndims] {
                dstride *= s;
            }
            dst += (start[d] + c as u64) * dstride;
        }
        dst += start[ndims - 1];
        let dst_byte = dst as usize * esize;
        if dst_byte + row_len <= out.len() && src + row_len <= chunk.len() {
            out[dst_byte..dst_byte + row_len].copy_from_slice(&chunk[src..src + row_len]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datatype_parsing() {
        // f32 (class 1, LE, size 4)
        let f32_msg = [0x11, 0x20, 0, 0, 4, 0, 0, 0];
        assert_eq!(parse_datatype(&f32_msg, 0).unwrap(), Some(DType::F32));
        // signed i64
        let i64_msg = [0x00, 0x08, 0, 0, 8, 0, 0, 0];
        assert_eq!(parse_datatype(&i64_msg, 0).unwrap(), Some(DType::I64));
        // unsigned u32
        let u32_msg = [0x00, 0x00, 0, 0, 4, 0, 0, 0];
        assert_eq!(parse_datatype(&u32_msg, 0).unwrap(), Some(DType::U32));
        // string class -> no projection, skip
        let str_msg = [0x03, 0x00, 0, 0, 8, 0, 0, 0];
        assert_eq!(parse_datatype(&str_msg, 0).unwrap(), None);
        // big-endian f32 -> refuse
        let be_msg = [0x11, 0x21, 0, 0, 4, 0, 0, 0];
        assert!(parse_datatype(&be_msg, 0).is_err());
    }

    #[test]
    fn unshuffle_roundtrip() {
        let shuffled = [1u8, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12];
        assert_eq!(
            unshuffle(&shuffled, 4),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
    }
}
