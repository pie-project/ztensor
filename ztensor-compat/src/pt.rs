//! PyTorch `.pt` / `.bin` → zTensor object model projection.
//!
//! A torch checkpoint is a ZIP: `<root>/data.pkl` (a pickle stream
//! describing the state dict) plus `<root>/data/<key>` raw storages.
//!
//! The pickle stream is evaluated by a restricted VM that executes **no
//! code**: it recognizes exactly the reconstruction patterns torch uses
//! (`persistent_load` storage tuples, `torch._utils._rebuild_tensor_v2`,
//! `collections.OrderedDict`) and materializes everything else as opaque.
//! Anything that would require reinterpreting bytes to "make work" — an
//! unknown storage dtype, a non-contiguous stride — is a loud refusal.
//!
//! This module is feature-gated (`pickle`): parsing pickle at all is a
//! larger attack surface than any other format here, and enabling it is an
//! explicit choice.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use memmap2::Mmap;
use ztensor::{
    BlobRef, Caps, DType, Error, Layout, Manifest, Object, Part, Result, Source,
};

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("pt: {}", detail.into()))
}

// =======================================================================
// Restricted pickle VM
// =======================================================================

const MAX_PICKLE_BYTES: usize = 256 << 20;
const MAX_STACK: usize = 10_000_000;
const MAX_MEMO: usize = 10_000_000;
const MAX_OPCODES: usize = 50_000_000;
const MAX_ITEMS: usize = 1_000_000;
const MAX_DEPTH: usize = 128;

#[derive(Debug, Clone)]
struct TensorRef {
    storage_key: String,
    /// Byte offset within the storage.
    byte_offset: u64,
    shape: Vec<u64>,
    dtype: DType,
    ltype: Option<&'static str>,
}

#[derive(Debug, Clone)]
enum Val {
    None,
    #[allow(dead_code)]
    Bool(bool),
    Int(i64),
    #[allow(dead_code)]
    Float(f64),
    #[allow(dead_code)]
    Bytes(Vec<u8>),
    Str(String),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global { module: String, name: String },
    Storage { key: String, dtype: DType, ltype: Option<&'static str> },
    Tensor(Box<TensorRef>),
    Mark,
    Opaque,
}

fn storage_dtype(name: &str) -> Option<(DType, Option<&'static str>)> {
    Some(match name {
        "DoubleStorage" => (DType::F64, None),
        "FloatStorage" => (DType::F32, None),
        "HalfStorage" => (DType::F16, None),
        "BFloat16Storage" => (DType::BF16, None),
        "LongStorage" => (DType::I64, None),
        "IntStorage" => (DType::I32, None),
        "ShortStorage" => (DType::I16, None),
        "CharStorage" => (DType::I8, None),
        "ByteStorage" => (DType::U8, None),
        "BoolStorage" => (DType::U8, Some("bool")),
        "Float8_e4m3fnStorage" => (DType::U8, Some("f8_e4m3fn")),
        "Float8_e5m2Storage" => (DType::U8, Some("f8_e5m2")),
        _ => return None,
    })
}

struct Vm<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Val>,
    memo: BTreeMap<u32, Val>,
    /// A refusal encountered mid-stream (e.g., a non-contiguous tensor):
    /// recorded here so the failure is loud, never a silently dropped
    /// tensor.
    refusal: Option<Error>,
}

impl<'a> Vm<'a> {
    fn byte(&mut self) -> Result<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| bad("pickle stream truncated"))?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| bad("pickle stream truncated"))?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn sized(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > MAX_PICKLE_BYTES {
            return Err(bad("pickle object exceeds size limit"));
        }
        self.take(n)
    }

    fn line(&mut self) -> Result<&'a [u8]> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(bad("pickle stream truncated"));
        }
        let line = &self.data[start..self.pos];
        self.pos += 1;
        Ok(line)
    }

    fn pop(&mut self) -> Val {
        self.stack.pop().unwrap_or(Val::None)
    }

    fn pop_to_mark(&mut self) -> Vec<Val> {
        let mark = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Val::Mark));
        match mark {
            Some(i) => {
                let items = self.stack.split_off(i + 1);
                self.stack.pop();
                items
            }
            None => Vec::new(),
        }
    }

    fn push_str(&mut self, bytes: &[u8]) {
        self.stack
            .push(Val::Str(String::from_utf8_lossy(bytes).into_owned()));
    }

    fn execute(&mut self) -> Result<()> {
        let mut ops = 0usize;
        while self.pos < self.data.len() {
            ops += 1;
            if ops > MAX_OPCODES {
                return Err(bad("pickle opcode limit exceeded"));
            }
            let op = self.byte()?;
            match op {
                0x80 => {
                    self.byte()?; // PROTO
                }
                0x95 => {
                    let n = self.u64()?; // FRAME
                    if self.pos as u64 + n > self.data.len() as u64 {
                        return Err(bad("FRAME length exceeds stream"));
                    }
                }
                0x2e => break,                                    // STOP
                0x28 => self.stack.push(Val::Mark),               // MARK
                0x29 => self.stack.push(Val::Tuple(Vec::new())),  // EMPTY_TUPLE
                0x5d => self.stack.push(Val::List(Vec::new())),   // EMPTY_LIST
                0x7d => self.stack.push(Val::Dict(Vec::new())),   // EMPTY_DICT
                0x4e => self.stack.push(Val::None),               // NONE
                0x88 => self.stack.push(Val::Bool(true)),         // NEWTRUE
                0x89 => self.stack.push(Val::Bool(false)),        // NEWFALSE
                0x4a => {
                    let v = self.u32()? as i32;
                    self.stack.push(Val::Int(v as i64)); // BININT
                }
                0x4b => {
                    let v = self.byte()?;
                    self.stack.push(Val::Int(v as i64)); // BININT1
                }
                0x4d => {
                    let v = self.u16()?;
                    self.stack.push(Val::Int(v as i64)); // BININT2
                }
                0x8a => {
                    // LONG1
                    let n = self.byte()? as usize;
                    let bytes = self.take(n)?;
                    self.stack.push(Val::Int(long_from_le(bytes)));
                }
                0x47 => {
                    // BINFLOAT (big-endian f64)
                    let v = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.stack.push(Val::Float(v));
                }
                0x58 => {
                    let n = self.u32()? as usize;
                    let b = self.sized(n)?;
                    self.push_str(b); // BINUNICODE
                }
                0x8c => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?;
                    self.push_str(b); // SHORT_BINUNICODE
                }
                0x8d => {
                    let n = self.u64()? as usize;
                    let b = self.sized(n)?;
                    self.push_str(b); // BINUNICODE8
                }
                0x55 => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?;
                    self.push_str(b); // SHORT_BINSTRING
                }
                0x43 => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Val::Bytes(b)); // SHORT_BINBYTES
                }
                0x44 => {
                    let n = self.u32()? as usize;
                    let b = self.sized(n)?.to_vec();
                    self.stack.push(Val::Bytes(b)); // BINBYTES
                }
                0x8e | 0x96 => {
                    let n = self.u64()? as usize;
                    let b = self.sized(n)?.to_vec();
                    self.stack.push(Val::Bytes(b)); // BINBYTES8 / BYTEARRAY8
                }
                0x63 => {
                    // GLOBAL
                    let module = String::from_utf8_lossy(self.line()?).into_owned();
                    let name = String::from_utf8_lossy(self.line()?).into_owned();
                    self.stack.push(Val::Global { module, name });
                }
                0x93 => {
                    // STACK_GLOBAL
                    let name = self.pop();
                    let module = self.pop();
                    match (module, name) {
                        (Val::Str(module), Val::Str(name)) => {
                            self.stack.push(Val::Global { module, name })
                        }
                        _ => self.stack.push(Val::Opaque),
                    }
                }
                0x85 => {
                    let a = self.pop();
                    self.stack.push(Val::Tuple(vec![a])); // TUPLE1
                }
                0x86 => {
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(Val::Tuple(vec![a, b])); // TUPLE2
                }
                0x87 => {
                    let c = self.pop();
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(Val::Tuple(vec![a, b, c])); // TUPLE3
                }
                0x74 => {
                    let items = self.pop_to_mark();
                    self.stack.push(Val::Tuple(items)); // TUPLE
                }
                0x6c => {
                    let items = self.pop_to_mark();
                    self.stack.push(Val::List(items)); // LIST
                }
                0x64 => {
                    let items = self.pop_to_mark();
                    self.stack.push(Val::Dict(pairs(items))); // DICT
                }
                0x52 | 0x81 => {
                    // REDUCE / NEWOBJ
                    let args = self.pop();
                    let callable = self.pop();
                    let v = self.reduce(callable, args)?;
                    self.stack.push(v);
                }
                0x92 => {
                    // NEWOBJ_EX
                    let _kwargs = self.pop();
                    let args = self.pop();
                    let callable = self.pop();
                    let v = self.reduce(callable, args)?;
                    self.stack.push(v);
                }
                0x51 => {
                    // BINPERSID
                    let pid = self.pop();
                    let v = self.persistent_load(pid);
                    self.stack.push(v);
                }
                0x62 => {
                    // BUILD
                    let state = self.pop();
                    let obj = self.pop();
                    self.stack.push(build(obj, state));
                }
                0x73 => {
                    // SETITEM
                    let value = self.pop();
                    let key = self.pop();
                    if let Some(Val::Dict(entries)) = self.stack.last_mut() {
                        if entries.len() >= MAX_ITEMS {
                            return Err(bad("dict size limit exceeded"));
                        }
                        entries.push((key, value));
                    }
                }
                0x75 => {
                    // SETITEMS
                    let items = self.pop_to_mark();
                    if let Some(Val::Dict(entries)) = self.stack.last_mut() {
                        if entries.len() + items.len() / 2 > MAX_ITEMS {
                            return Err(bad("dict size limit exceeded"));
                        }
                        entries.extend(pairs(items));
                    }
                }
                0x61 => {
                    // APPEND
                    let value = self.pop();
                    if let Some(Val::List(list)) = self.stack.last_mut() {
                        if list.len() >= MAX_ITEMS {
                            return Err(bad("list size limit exceeded"));
                        }
                        list.push(value);
                    }
                }
                0x65 => {
                    // APPENDS
                    let items = self.pop_to_mark();
                    if let Some(Val::List(list)) = self.stack.last_mut() {
                        if list.len() + items.len() > MAX_ITEMS {
                            return Err(bad("list size limit exceeded"));
                        }
                        list.extend(items);
                    }
                }
                0x71 => {
                    let idx = self.byte()? as u32;
                    self.memoize(idx); // BINPUT
                }
                0x72 => {
                    let idx = self.u32()?;
                    self.memoize(idx); // LONG_BINPUT
                }
                0x94 => {
                    let idx = self.memo.len() as u32;
                    self.memoize(idx); // MEMOIZE
                }
                0x68 => {
                    let idx = self.byte()? as u32;
                    self.memo_get(idx); // BINGET
                }
                0x6a => {
                    let idx = self.u32()?;
                    self.memo_get(idx); // LONG_BINGET
                }
                0x30 => {
                    self.pop(); // POP
                }
                0x31 => {
                    self.pop_to_mark(); // POP_MARK
                }
                0x49 => {
                    // INT (text)
                    let line = String::from_utf8_lossy(self.line()?).into_owned();
                    let s = line.trim();
                    let v = match s {
                        "00" => Val::Bool(false),
                        "01" => Val::Bool(true),
                        _ => Val::Int(s.parse().unwrap_or(0)),
                    };
                    self.stack.push(v);
                }
                0x91 | 0x90 => {
                    // FROZENSET / ADDITEMS: irrelevant to state dicts
                    let items = self.pop_to_mark();
                    if op == 0x91 {
                        self.stack.push(Val::Tuple(items));
                    }
                }
                0x97 => self.stack.push(Val::Opaque), // NEXT_BUFFER
                0x98 => {}                            // READONLY_BUFFER
                other => {
                    return Err(bad(format!(
                        "unsupported pickle opcode 0x{other:02x} at {}",
                        self.pos - 1
                    )))
                }
            }
            if self.stack.len() > MAX_STACK || self.memo.len() > MAX_MEMO {
                return Err(bad("pickle stack/memo limit exceeded"));
            }
        }
        match self.refusal.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn memoize(&mut self, idx: u32) {
        if let Some(v) = self.stack.last() {
            // Containers are memoized as opaque markers: state dicts never
            // need container back-references, and cloning bounded-size
            // scalars keeps the memo cheap.
            let stored = match v {
                Val::Dict(_) | Val::List(_) => Val::Opaque,
                other => other.clone(),
            };
            self.memo.insert(idx, stored);
        }
    }

    fn memo_get(&mut self, idx: u32) {
        let v = self.memo.get(&idx).cloned().unwrap_or(Val::None);
        self.stack.push(v);
    }

    fn reduce(&mut self, callable: Val, args: Val) -> Result<Val> {
        let Val::Global { module, name } = &callable else {
            return Ok(Val::Opaque);
        };
        if module == "collections" && name == "OrderedDict" {
            return Ok(Val::Dict(Vec::new()));
        }
        if module == "torch._utils" && name.starts_with("_rebuild_tensor") {
            if let Val::Tuple(items) = &args {
                return self.rebuild_tensor(items);
            }
        }
        Ok(Val::Opaque)
    }

    /// `_rebuild_tensor_v2(storage, storage_offset, size, stride, ...)`.
    /// Non-contiguous strides are refused: the storage bytes of such a
    /// tensor are not its row-major bytes, and reading them as dense would
    /// silently return wrong data.
    fn rebuild_tensor(&mut self, args: &[Val]) -> Result<Val> {
        if args.len() < 4 {
            return Ok(Val::Opaque);
        }
        let Val::Storage { key, dtype, ltype } = &args[0] else {
            return Ok(Val::Opaque);
        };
        let offset_elems = match &args[1] {
            Val::Int(v) if *v >= 0 => *v as u64,
            _ => return Ok(Val::Opaque),
        };
        let (Some(shape), Some(stride)) = (dims(&args[2]), dims(&args[3])) else {
            return Ok(Val::Opaque);
        };

        // Contiguity: stride[i] == product(shape[i+1..]), dims of size ≤ 1
        // exempt (their stride is arbitrary).
        let mut expected = 1u64;
        for (i, &dim) in shape.iter().enumerate().rev() {
            if dim > 1 && stride.get(i) != Some(&expected) {
                self.refusal = Some(Error::Unsupported(format!(
                    "pt: tensor on storage {key:?} is not contiguous \
                     (shape {shape:?}, stride {stride:?}); refusing to read it as dense"
                )));
                return Ok(Val::Opaque);
            }
            expected = expected.saturating_mul(dim.max(1));
        }

        Ok(Val::Tensor(Box::new(TensorRef {
            storage_key: key.clone(),
            byte_offset: offset_elems * dtype.width(),
            shape,
            dtype: *dtype,
            ltype: *ltype,
        })))
    }

    /// torch persistent id: `("storage", <StorageType>, key, location, numel)`.
    fn persistent_load(&mut self, pid: Val) -> Val {
        let Val::Tuple(items) = &pid else {
            return Val::Opaque;
        };
        if items.len() < 5 {
            return Val::Opaque;
        }
        let (Val::Str(tag), Val::Global { name, .. }, Val::Str(key), _, Val::Int(numel)) =
            (&items[0], &items[1], &items[2], &items[3], &items[4])
        else {
            return Val::Opaque;
        };
        if tag != "storage" || *numel < 0 {
            return Val::Opaque;
        }
        match storage_dtype(name) {
            Some((dtype, ltype)) => Val::Storage {
                key: key.clone(),
                dtype,
                ltype,
            },
            None => {
                self.refusal = Some(Error::Unsupported(format!(
                    "pt: storage type {name:?} has no registered projection"
                )));
                Val::Opaque
            }
        }
    }
}

fn dims(v: &Val) -> Option<Vec<u64>> {
    match v {
        Val::Tuple(items) | Val::List(items) => items
            .iter()
            .map(|d| match d {
                Val::Int(v) if *v >= 0 => Some(*v as u64),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn build(obj: Val, state: Val) -> Val {
    match (obj, state) {
        (Val::Dict(mut entries), Val::Dict(more)) => {
            entries.extend(more);
            Val::Dict(entries)
        }
        (_, state) => state,
    }
}

fn pairs(items: Vec<Val>) -> Vec<(Val, Val)> {
    let mut out = Vec::with_capacity(items.len() / 2);
    let mut it = items.into_iter();
    while let (Some(k), Some(v)) = (it.next(), it.next()) {
        out.push((k, v));
    }
    out
}

fn long_from_le(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    let n = bytes.len().min(8);
    let mut v = 0i64;
    for (i, &b) in bytes[..n].iter().enumerate() {
        v |= (b as i64) << (i * 8);
    }
    if bytes[n - 1] & 0x80 != 0 && n < 8 {
        v |= !0i64 << (n * 8);
    }
    v
}

/// Walks the unpickled tree and collects named tensors. Duplicate names
/// are rejected (a genuine Python dict cannot produce them).
fn collect_tensors(
    prefix: &str,
    v: &Val,
    out: &mut BTreeMap<String, TensorRef>,
    depth: usize,
) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(bad("structure nesting too deep"));
    }
    match v {
        Val::Tensor(t) => {
            if out.insert(prefix.to_string(), (**t).clone()).is_some() {
                return Err(bad(format!("duplicate tensor name {prefix:?}")));
            }
        }
        Val::Dict(entries) => {
            for (k, val) in entries {
                let name = match k {
                    Val::Str(s) if prefix.is_empty() => s.clone(),
                    Val::Str(s) => format!("{prefix}.{s}"),
                    _ => prefix.to_string(),
                };
                collect_tensors(&name, val, out, depth + 1)?;
            }
        }
        Val::List(items) | Val::Tuple(items) => {
            for (i, item) in items.iter().enumerate() {
                let name = if prefix.is_empty() {
                    i.to_string()
                } else {
                    format!("{prefix}.{i}")
                };
                collect_tensors(&name, item, out, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

// =======================================================================
// Container
// =======================================================================

enum StorageLoc {
    /// Stored entry: absolute range in the file.
    Stored { offset: u64, length: u64 },
    /// Compressed entry: decompressed lazily, cached.
    Compressed { zip_index: usize, length: u64 },
}

pub struct Pt {
    mmap: Mmap,
    archive: RefCell<zip::ZipArchive<File>>,
    cache: RefCell<BTreeMap<String, Vec<u8>>>,
    storages: BTreeMap<String, StorageLoc>,
    manifest: Manifest,
    /// tensor name → (storage key, byte offset within storage, byte len).
    tensors: BTreeMap<String, (String, u64, u64)>,
}

impl std::fmt::Debug for Pt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pt")
            .field("len", &self.mmap.len())
            .field("objects", &self.manifest.objects.len())
            .finish()
    }
}

impl Pt {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // SAFETY: read-only shared map of untrusted bytes.
        let mmap = unsafe { Mmap::map(&file)? };
        let mut archive = zip::ZipArchive::new(File::open(path)?)
            .map_err(|e| bad(format!("not a ZIP archive: {e}")))?;

        // Locate the pickle.
        let pickle_name = (0..archive.len())
            .filter_map(|i| archive.by_index_raw(i).ok().map(|e| e.name().to_string()))
            .find(|n| n.ends_with("data.pkl"))
            .ok_or_else(|| bad("no data.pkl entry"))?;
        let prefix = format!(
            "{}data/",
            pickle_name.strip_suffix("data.pkl").unwrap_or_default()
        );

        let mut pickle = Vec::new();
        archive
            .by_name(&pickle_name)
            .map_err(|e| bad(format!("{pickle_name}: {e}")))?
            .read_to_end(&mut pickle)?;

        // Run the VM and collect tensors.
        let mut vm = Vm {
            data: &pickle,
            pos: 0,
            stack: Vec::new(),
            memo: BTreeMap::new(),
            refusal: None,
        };
        vm.execute()?;
        let mut tensors = BTreeMap::new();
        for v in &vm.stack {
            collect_tensors("", v, &mut tensors, 0)?;
        }
        if tensors.is_empty() {
            return Err(bad("no tensors found"));
        }

        // Locate each referenced storage entry.
        let mut storages = BTreeMap::new();
        for t in tensors.values() {
            let key = &t.storage_key;
            if storages.contains_key(key) {
                continue;
            }
            let entry_name = format!("{prefix}{key}");
            let zip_index = archive
                .index_for_name(&entry_name)
                .ok_or_else(|| bad(format!("storage entry {entry_name:?} missing")))?;
            let entry = archive
                .by_index_raw(zip_index)
                .map_err(|e| bad(format!("storage entry {entry_name:?}: {e}")))?;
            let loc = if entry.compression() == zip::CompressionMethod::Stored {
                let start = entry.data_start();
                let size = entry.size();
                if start + size > mmap.len() as u64 {
                    return Err(bad(format!("storage {key:?} extends past file")));
                }
                StorageLoc::Stored {
                    offset: start,
                    length: size,
                }
            } else {
                StorageLoc::Compressed {
                    zip_index,
                    length: entry.size(),
                }
            };
            drop(entry);
            storages.insert(key.clone(), loc);
        }

        // Build the projection, bounds-checking every tensor against its
        // storage.
        let mut objects = BTreeMap::new();
        let mut locations = BTreeMap::new();
        for (name, t) in &tensors {
            let elems = t
                .shape
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| bad(format!("tensor {name:?} shape overflows")))?;
            let byte_len = ztensor::logical_size(t.ltype, t.dtype, elems)
                .ok_or_else(|| bad("size not computable"))?;
            let storage = &storages[&t.storage_key];
            let storage_len = match storage {
                StorageLoc::Stored { length, .. } | StorageLoc::Compressed { length, .. } => {
                    *length
                }
            };
            let end = t
                .byte_offset
                .checked_add(byte_len)
                .filter(|&e| e <= storage_len)
                .ok_or_else(|| {
                    bad(format!("tensor {name:?} extends past its storage"))
                })?;
            let _ = end;

            let (blob, encoding) = match storage {
                StorageLoc::Stored { offset, .. } => (
                    BlobRef {
                        shard: 0,
                        offset: offset + t.byte_offset,
                        length: byte_len,
                    },
                    None,
                ),
                StorageLoc::Compressed { .. } => (
                    BlobRef {
                        shard: 0,
                        offset: 0,
                        length: 0, // no stable stored range for a slice of a compressed storage
                    },
                    Some("pt.deflate/1".to_string()),
                ),
            };
            let part = Part {
                dtype: t.dtype,
                ltype: t.ltype.map(str::to_string),
                blob,
                decoded_length: encoding.as_ref().map(|_| byte_len),
                encoding,
                digest: None,
            };
            let mut parts = BTreeMap::new();
            parts.insert("data".to_string(), part);
            objects.insert(
                name.clone(),
                Object {
                    shape: t.shape.clone(),
                    layout: Layout::Dense,
                    attributes: None,
                    parts,
                },
            );
            locations.insert(name.clone(), (t.storage_key.clone(), t.byte_offset, byte_len));
        }

        Ok(Self {
            mmap,
            archive: RefCell::new(archive),
            cache: RefCell::new(BTreeMap::new()),
            storages,
            manifest: Manifest {
                attributes: None,
                shards: BTreeMap::new(),
                objects,
            },
            tensors: locations,
        })
    }

    fn location(&self, name: &str, part: &str) -> Result<&(String, u64, u64)> {
        if part != "data" {
            return Err(Error::NotFound(format!("part {name:?}/{part:?}")));
        }
        self.tensors
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("object {name:?}")))
    }

    fn ensure_cached(&self, key: &str, zip_index: usize, length: u64) -> Result<()> {
        if self.cache.borrow().contains_key(key) {
            return Ok(());
        }
        let mut archive = self.archive.borrow_mut();
        let mut entry = archive
            .by_index(zip_index)
            .map_err(|e| bad(format!("storage {key:?}: {e}")))?;
        let mut bytes = Vec::with_capacity(length as usize);
        entry
            .read_to_end(&mut bytes)
            .map_err(|e| bad(format!("storage {key:?}: {e}")))?;
        if bytes.len() as u64 != length {
            return Err(bad(format!("storage {key:?} decompressed size mismatch")));
        }
        self.cache.borrow_mut().insert(key.to_string(), bytes);
        Ok(())
    }
}

impl Source for Pt {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read(&self, object: &str, part: &str) -> Result<Vec<u8>> {
        let (key, off, len) = self.location(object, part)?;
        match &self.storages[key] {
            StorageLoc::Stored { offset, .. } => {
                let start = (offset + off) as usize;
                Ok(self.mmap[start..start + *len as usize].to_vec())
            }
            StorageLoc::Compressed { zip_index, length } => {
                self.ensure_cached(key, *zip_index, *length)?;
                let cache = self.cache.borrow();
                let storage = &cache[key];
                Ok(storage[*off as usize..(*off + *len) as usize].to_vec())
            }
        }
    }

    fn view(&self, object: &str, part: &str) -> Result<&[u8]> {
        let (key, off, len) = self.location(object, part)?;
        match &self.storages[key] {
            StorageLoc::Stored { offset, .. } => {
                let start = (offset + off) as usize;
                Ok(&self.mmap[start..start + *len as usize])
            }
            StorageLoc::Compressed { .. } => Err(Error::Unsupported(
                "tensor in a compressed storage has no zero-copy view".into(),
            )),
        }
    }

    fn caps(&self, object: &str, part: &str) -> Result<Caps> {
        let (key, off, _) = self.location(object, part)?;
        let (zero_copy, alignment) = match &self.storages[key] {
            StorageLoc::Stored { offset, .. } => {
                let abs = offset + off;
                (
                    true,
                    if abs == 0 {
                        1
                    } else {
                        1u64 << abs.trailing_zeros().min(63)
                    },
                )
            }
            StorageLoc::Compressed { .. } => (false, 1),
        };
        Ok(Caps {
            zero_copy,
            alignment,
            verifiable: false,
            page_exclusive: false,
        })
    }
}
