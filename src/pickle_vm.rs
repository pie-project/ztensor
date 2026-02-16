//! Minimal pickle virtual machine for PyTorch .pt files.
//!
//! Implements just enough of the pickle protocol to parse PyTorch state dicts.
//! PyTorch saves models as ZIP files containing:
//! - `archive/data.pkl`: Pickle protocol 2+ file describing the model structure
//! - `archive/data/N`: Raw storage files containing tensor data
//!
//! The pickle VM recognizes `torch._utils._rebuild_tensor_v2` and related
//! reconstruction patterns to extract tensor metadata without executing
//! arbitrary Python code.
//!
//! Requires the `pickle` feature.

use std::collections::BTreeMap;
use std::io::Read;

use crate::error::Error;
use crate::models::DType;

/// Maximum size for a single pickle bytes/string object (256 MiB).
const MAX_PICKLE_BYTES: usize = 256 * 1024 * 1024;

/// Maximum recursion depth when extracting tensors from pickle values.
const MAX_EXTRACT_DEPTH: usize = 128;

/// Maximum number of items on the pickle stack.
const MAX_STACK_SIZE: usize = 10_000_000;

/// Maximum number of entries in the pickle memo table.
const MAX_MEMO_SIZE: usize = 10_000_000;

/// Maximum number of opcodes to execute before aborting.
const MAX_OPCODES: usize = 50_000_000;

/// Maximum number of items in a single list/tuple/dict.
const MAX_CONTAINER_ITEMS: usize = 1_000_000;

/// Information about a tensor found in a PyTorch pickle.
#[derive(Debug, Clone)]
pub struct PtTensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub storage_key: String,
    pub storage_offset: usize,
    pub numel: usize,
}

/// A value on the pickle stack.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Bytes(Vec<u8>),
    String(String),
    Tuple(Vec<PickleValue>),
    List(Vec<PickleValue>),
    Dict(Vec<(PickleValue, PickleValue)>),
    Global { module: String, name: String },
    /// Represents a rebuilt tensor: (storage_key, storage_offset, shape, stride, dtype)
    TensorRef(Box<TensorRef>),
    /// Represents a storage: (storage_key, dtype, numel)
    StorageRef { key: String, dtype: DType, numel: usize },
    /// A MARK sentinel.
    Mark,
    /// Reduced object we don't understand (passthrough).
    Opaque,
}

#[derive(Debug, Clone)]
struct TensorRef {
    storage_key: String,
    storage_offset: usize,
    shape: Vec<u64>,
    dtype: DType,
}

/// Counts the total number of nodes in a PickleValue tree (iteratively to avoid stack overflow).
/// Returns early once `limit` is exceeded.
fn value_node_count(val: &PickleValue, limit: usize) -> usize {
    let mut count = 0usize;
    let mut work = vec![val];
    while let Some(v) = work.pop() {
        count += 1;
        if count > limit {
            return count;
        }
        match v {
            PickleValue::List(items) | PickleValue::Tuple(items) => {
                work.extend(items.iter());
            }
            PickleValue::Dict(pairs) => {
                for (k, v) in pairs {
                    work.push(k);
                    work.push(v);
                }
            }
            _ => {}
        }
    }
    count
}

/// Parses a PyTorch pickle and extracts tensor information.
pub fn parse_pytorch_pickle<R: Read>(reader: &mut R) -> Result<Vec<PtTensorInfo>, Error> {
    let mut data = Vec::new();
    reader.read_to_end(&mut data)?;

    let mut vm = PickleVM {
        data: &data,
        pos: 0,
        stack: Vec::new(),
        memo: BTreeMap::new(),
    };

    vm.execute()?;

    // The top of stack should be an OrderedDict or dict of tensor name -> TensorRef
    extract_tensors_from_stack(&vm.stack)
}

struct PickleVM<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<PickleValue>,
    memo: BTreeMap<u32, PickleValue>,
}

impl<'a> PickleVM<'a> {
    fn read_u8(&mut self) -> Result<u8, Error> {
        if self.pos >= self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u16_le(&mut self) -> Result<u16, Error> {
        if self.pos + 2 > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_u32_le(&mut self) -> Result<u32, Error> {
        if self.pos + 4 > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_i32_le(&mut self) -> Result<i32, Error> {
        Ok(self.read_u32_le()? as i32)
    }

    fn read_u64_le(&mut self) -> Result<u64, Error> {
        if self.pos + 8 > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::UnexpectedEof)?;
        if end > self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_line(&mut self) -> Result<&'a [u8], Error> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(Error::UnexpectedEof);
        }
        let line = &self.data[start..self.pos];
        self.pos += 1; // skip '\n'
        Ok(line)
    }

    fn find_mark(&self) -> Option<usize> {
        for i in (0..self.stack.len()).rev() {
            if matches!(self.stack[i], PickleValue::Mark) {
                return Some(i);
            }
        }
        None
    }

    fn pop_to_mark(&mut self) -> Vec<PickleValue> {
        if let Some(mark_idx) = self.find_mark() {
            let items: Vec<PickleValue> = self.stack.drain(mark_idx + 1..).collect();
            self.stack.pop(); // remove Mark
            items
        } else {
            Vec::new()
        }
    }

    fn execute(&mut self) -> Result<(), Error> {
        let mut opcode_count: usize = 0;
        loop {
            if self.pos >= self.data.len() {
                break;
            }

            opcode_count += 1;
            if opcode_count > MAX_OPCODES {
                return Err(Error::Other(
                    "Pickle opcode limit exceeded".into(),
                ));
            }

            let opcode = self.read_u8()?;

            match opcode {
                // PROTO
                0x80 => {
                    let _version = self.read_u8()?;
                }
                // FRAME
                0x95 => {
                    let frame_len = self.read_u64_le()? as usize;
                    let end = self.pos.checked_add(frame_len).ok_or_else(|| {
                        Error::InvalidFileStructure(
                            "Pickle FRAME length exceeds data bounds".into(),
                        )
                    })?;
                    if end > self.data.len() {
                        return Err(Error::InvalidFileStructure(
                            "Pickle FRAME length exceeds data bounds".into(),
                        ));
                    }
                }
                // STOP
                0x2e => break,
                // MARK
                0x28 => self.stack.push(PickleValue::Mark),
                // EMPTY_TUPLE
                0x29 => self.stack.push(PickleValue::Tuple(Vec::new())),
                // EMPTY_LIST
                0x5d => self.stack.push(PickleValue::List(Vec::new())),
                // EMPTY_DICT
                0x7d => self.stack.push(PickleValue::Dict(Vec::new())),
                // NONE
                0x4e => self.stack.push(PickleValue::None),
                // NEWTRUE
                0x88 => self.stack.push(PickleValue::Bool(true)),
                // NEWFALSE
                0x89 => self.stack.push(PickleValue::Bool(false)),
                // BININT
                0x4a => {
                    let v = self.read_i32_le()?;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                // BININT1
                0x4b => {
                    let v = self.read_u8()?;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                // BININT2
                0x4d => {
                    let v = self.read_u16_le()?;
                    self.stack.push(PickleValue::Int(v as i64));
                }
                // LONG1
                0x8a => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let val = long_from_bytes(bytes);
                    self.stack.push(PickleValue::Int(val));
                }
                // BINFLOAT
                0x47 => {
                    let bytes = self.read_bytes(8)?;
                    let v = f64::from_be_bytes(bytes.try_into().unwrap());
                    self.stack.push(PickleValue::Float(v));
                }
                // BINUNICODE (4-byte length)
                0x58 => {
                    let n = self.read_u32_le()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = String::from_utf8_lossy(bytes).to_string();
                    self.stack.push(PickleValue::String(s));
                }
                // SHORT_BINUNICODE (1-byte length)
                0x8c => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = String::from_utf8_lossy(bytes).to_string();
                    self.stack.push(PickleValue::String(s));
                }
                // BINUNICODE8 (8-byte length)
                0x8d => {
                    let n = self.read_u64_le()? as usize;
                    if n > MAX_PICKLE_BYTES {
                        return Err(Error::Other(format!(
                            "Pickle BINUNICODE8 size {} exceeds limit {}", n, MAX_PICKLE_BYTES
                        )));
                    }
                    let bytes = self.read_bytes(n)?;
                    let s = String::from_utf8_lossy(bytes).to_string();
                    self.stack.push(PickleValue::String(s));
                }
                // SHORT_BINBYTES
                0x43 => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?.to_vec();
                    self.stack.push(PickleValue::Bytes(bytes));
                }
                // BINBYTES
                0x44 => {
                    let n = self.read_u32_le()? as usize;
                    if n > MAX_PICKLE_BYTES {
                        return Err(Error::Other(format!(
                            "Pickle BINBYTES size {} exceeds limit {}", n, MAX_PICKLE_BYTES
                        )));
                    }
                    let bytes = self.read_bytes(n)?.to_vec();
                    self.stack.push(PickleValue::Bytes(bytes));
                }
                // BINBYTES8
                0x8e => {
                    let n = self.read_u64_le()? as usize;
                    if n > MAX_PICKLE_BYTES {
                        return Err(Error::Other(format!(
                            "Pickle BINBYTES8 size {} exceeds limit {}", n, MAX_PICKLE_BYTES
                        )));
                    }
                    let bytes = self.read_bytes(n)?.to_vec();
                    self.stack.push(PickleValue::Bytes(bytes));
                }
                // SHORT_BINSTRING
                0x55 => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = String::from_utf8_lossy(bytes).to_string();
                    self.stack.push(PickleValue::String(s));
                }
                // GLOBAL
                0x63 => {
                    let module_line = self.read_line()?;
                    let name_line = self.read_line()?;
                    let module = String::from_utf8_lossy(module_line).to_string();
                    let name = String::from_utf8_lossy(name_line).to_string();
                    self.stack.push(PickleValue::Global { module, name });
                }
                // STACK_GLOBAL
                0x93 => {
                    let name = self.stack.pop().unwrap_or(PickleValue::None);
                    let module = self.stack.pop().unwrap_or(PickleValue::None);
                    if let (PickleValue::String(m), PickleValue::String(n)) = (module, name) {
                        self.stack.push(PickleValue::Global {
                            module: m,
                            name: n,
                        });
                    } else {
                        self.stack.push(PickleValue::Opaque);
                    }
                }
                // TUPLE
                0x85 => {
                    // TUPLE1
                    if let Some(a) = self.stack.pop() {
                        self.stack.push(PickleValue::Tuple(vec![a]));
                    }
                }
                // TUPLE2
                0x86 => {
                    let b = self.stack.pop().unwrap_or(PickleValue::None);
                    let a = self.stack.pop().unwrap_or(PickleValue::None);
                    self.stack.push(PickleValue::Tuple(vec![a, b]));
                }
                // TUPLE3
                0x87 => {
                    let c = self.stack.pop().unwrap_or(PickleValue::None);
                    let b = self.stack.pop().unwrap_or(PickleValue::None);
                    let a = self.stack.pop().unwrap_or(PickleValue::None);
                    self.stack.push(PickleValue::Tuple(vec![a, b, c]));
                }
                // TUPLE (from MARK)
                0x74 => {
                    let items = self.pop_to_mark();
                    self.stack.push(PickleValue::Tuple(items));
                }
                // LIST (from MARK)
                0x6c => {
                    let items = self.pop_to_mark();
                    self.stack.push(PickleValue::List(items));
                }
                // DICT (from MARK)
                0x64 => {
                    let items = self.pop_to_mark();
                    let pairs = items_to_pairs(items);
                    self.stack.push(PickleValue::Dict(pairs));
                }
                // REDUCE
                0x52 => {
                    let args = self.stack.pop().unwrap_or(PickleValue::None);
                    let callable = self.stack.pop().unwrap_or(PickleValue::None);
                    let result = self.reduce(callable, args);
                    self.stack.push(result);
                }
                // BUILD
                0x62 => {
                    let state = self.stack.pop().unwrap_or(PickleValue::None);
                    let obj = self.stack.pop().unwrap_or(PickleValue::None);
                    let result = self.build(obj, state);
                    self.stack.push(result);
                }
                // NEWOBJ
                0x81 => {
                    let args = self.stack.pop().unwrap_or(PickleValue::None);
                    let cls = self.stack.pop().unwrap_or(PickleValue::None);
                    let result = self.reduce(cls, args);
                    self.stack.push(result);
                }
                // NEWOBJ_EX
                0x92 => {
                    let _kwargs = self.stack.pop();
                    let args = self.stack.pop().unwrap_or(PickleValue::None);
                    let cls = self.stack.pop().unwrap_or(PickleValue::None);
                    let result = self.reduce(cls, args);
                    self.stack.push(result);
                }
                // BINPERSID
                0x51 => {
                    let pid = self.stack.pop().unwrap_or(PickleValue::None);
                    let result = self.persistent_load(pid);
                    self.stack.push(result);
                }
                // SETITEM
                0x73 => {
                    let value = self.stack.pop().unwrap_or(PickleValue::None);
                    let key = self.stack.pop().unwrap_or(PickleValue::None);
                    if let Some(PickleValue::Dict(ref mut pairs)) = self.stack.last_mut() {
                        if pairs.len() >= MAX_CONTAINER_ITEMS {
                            return Err(Error::Other("Pickle dict size limit exceeded".into()));
                        }
                        pairs.push((key, value));
                    }
                }
                // SETITEMS
                0x75 => {
                    let items = self.pop_to_mark();
                    let pairs = items_to_pairs(items);
                    if let Some(PickleValue::Dict(ref mut existing)) = self.stack.last_mut() {
                        if existing.len() + pairs.len() > MAX_CONTAINER_ITEMS {
                            return Err(Error::Other("Pickle dict size limit exceeded".into()));
                        }
                        existing.extend(pairs);
                    }
                }
                // APPEND
                0x61 => {
                    let value = self.stack.pop().unwrap_or(PickleValue::None);
                    if let Some(PickleValue::List(ref mut list)) = self.stack.last_mut() {
                        if list.len() >= MAX_CONTAINER_ITEMS {
                            return Err(Error::Other("Pickle list size limit exceeded".into()));
                        }
                        list.push(value);
                    }
                }
                // APPENDS
                0x65 => {
                    let items = self.pop_to_mark();
                    if let Some(PickleValue::List(ref mut list)) = self.stack.last_mut() {
                        if list.len() + items.len() > MAX_CONTAINER_ITEMS {
                            return Err(Error::Other("Pickle list size limit exceeded".into()));
                        }
                        list.extend(items);
                    }
                }
                // BINPUT
                0x71 => {
                    let idx = self.read_u8()? as u32;
                    if let Some(val) = self.stack.last() {
                        self.memo.insert(idx, val.clone());
                    }
                }
                // LONG_BINPUT
                0x72 => {
                    let idx = self.read_u32_le()?;
                    if let Some(val) = self.stack.last() {
                        self.memo.insert(idx, val.clone());
                    }
                }
                // BINGET
                0x68 => {
                    let idx = self.read_u8()? as u32;
                    if let Some(val) = self.memo.get(&idx) {
                        if value_node_count(val, MAX_CONTAINER_ITEMS) > MAX_CONTAINER_ITEMS {
                            return Err(Error::Other(
                                "Pickle memo value too large to retrieve".into(),
                            ));
                        }
                        self.stack.push(val.clone());
                    } else {
                        self.stack.push(PickleValue::None);
                    }
                }
                // LONG_BINGET
                0x6a => {
                    let idx = self.read_u32_le()?;
                    if let Some(val) = self.memo.get(&idx) {
                        if value_node_count(val, MAX_CONTAINER_ITEMS) > MAX_CONTAINER_ITEMS {
                            return Err(Error::Other(
                                "Pickle memo value too large to retrieve".into(),
                            ));
                        }
                        self.stack.push(val.clone());
                    } else {
                        self.stack.push(PickleValue::None);
                    }
                }
                // MEMOIZE
                0x94 => {
                    let idx = self.memo.len() as u32;
                    if let Some(val) = self.stack.last() {
                        self.memo.insert(idx, val.clone());
                    }
                }
                // POP
                0x30 => {
                    self.stack.pop();
                }
                // DUP
                0x32 => {
                    if let Some(val) = self.stack.last() {
                        if value_node_count(val, MAX_CONTAINER_ITEMS) > MAX_CONTAINER_ITEMS {
                            return Err(Error::Other(
                                "Pickle value too large to duplicate".into(),
                            ));
                        }
                        self.stack.push(val.clone());
                    }
                }
                // POP_MARK
                0x31 => {
                    self.pop_to_mark();
                }
                // INT (text encoding)
                0x49 => {
                    let line = self.read_line()?;
                    let s = String::from_utf8_lossy(line);
                    let s = s.trim();
                    if s == "00" {
                        self.stack.push(PickleValue::Bool(false));
                    } else if s == "01" {
                        self.stack.push(PickleValue::Bool(true));
                    } else if let Ok(v) = s.parse::<i64>() {
                        self.stack.push(PickleValue::Int(v));
                    } else {
                        self.stack.push(PickleValue::Int(0));
                    }
                }
                // FROZENSET
                0x91 => {
                    let items = self.pop_to_mark();
                    self.stack.push(PickleValue::Tuple(items));
                }
                // ADDITEMS
                0x90 => {
                    let _items = self.pop_to_mark();
                    // For sets, we just ignore
                }
                // BYTEARRAY8 (Protocol 5)
                0x96 => {
                    let n = self.read_u64_le()? as usize;
                    if n > MAX_PICKLE_BYTES {
                        return Err(Error::Other(format!(
                            "Pickle BYTEARRAY8 size {} exceeds limit {}", n, MAX_PICKLE_BYTES
                        )));
                    }
                    let bytes = self.read_bytes(n)?.to_vec();
                    self.stack.push(PickleValue::Bytes(bytes));
                }
                // NEXT_BUFFER (Protocol 5) — out-of-band buffer, not used in .pt files
                0x97 => {
                    self.stack.push(PickleValue::Opaque);
                }
                // READONLY_BUFFER (Protocol 5) — no-op flag on top-of-stack
                0x98 => {}
                // Unknown opcode - skip
                other => {
                    return Err(Error::InvalidFileStructure(format!(
                        "Unknown pickle opcode: 0x{:02X} at position {}",
                        other,
                        self.pos - 1
                    )));
                }
            }

            if self.stack.len() > MAX_STACK_SIZE {
                return Err(Error::Other("Pickle stack size limit exceeded".into()));
            }
            if self.memo.len() > MAX_MEMO_SIZE {
                return Err(Error::Other("Pickle memo size limit exceeded".into()));
            }
        }

        Ok(())
    }

    fn reduce(&self, callable: PickleValue, args: PickleValue) -> PickleValue {
        match &callable {
            PickleValue::Global { module, name } => {
                // collections.OrderedDict() -> empty dict
                if module == "collections" && name == "OrderedDict" {
                    return PickleValue::Dict(Vec::new());
                }

                // torch._utils._rebuild_tensor_v2(storage, storage_offset, size, stride, ...)
                if module == "torch._utils"
                    && (name == "_rebuild_tensor_v2"
                        || name == "_rebuild_tensor_v3"
                        || name == "_rebuild_tensor_v4")
                {
                    if let PickleValue::Tuple(ref items) = args {
                        return self.rebuild_tensor(items);
                    }
                }

                PickleValue::Opaque
            }
            _ => PickleValue::Opaque,
        }
    }

    fn rebuild_tensor(&self, args: &[PickleValue]) -> PickleValue {
        // args: (storage, storage_offset, size, stride, requires_grad, ...)
        if args.len() < 3 {
            return PickleValue::Opaque;
        }

        let storage = &args[0];
        let storage_offset = match &args[1] {
            PickleValue::Int(v) => *v as usize,
            _ => 0,
        };
        let shape = match &args[2] {
            PickleValue::Tuple(dims) => {
                dims.iter()
                    .filter_map(|d| match d {
                        PickleValue::Int(v) => Some(*v as u64),
                        _ => None,
                    })
                    .collect::<Vec<u64>>()
            }
            _ => return PickleValue::Opaque,
        };

        match storage {
            PickleValue::StorageRef { key, dtype, .. } => {
                PickleValue::TensorRef(Box::new(TensorRef {
                    storage_key: key.clone(),
                    storage_offset: storage_offset * dtype.byte_size(),
                    shape,
                    dtype: *dtype,
                }))
            }
            _ => PickleValue::Opaque,
        }
    }

    fn build(&self, obj: PickleValue, state: PickleValue) -> PickleValue {
        // OrderedDict BUILD with a list of (key, value) pairs
        if let PickleValue::Dict(mut pairs) = obj {
            if let PickleValue::Dict(state_pairs) = state {
                pairs.extend(state_pairs);
                return PickleValue::Dict(pairs);
            }
        }
        // For other objects, just return the object
        state
    }

    fn persistent_load(&self, pid: PickleValue) -> PickleValue {
        // PyTorch persistent_load format: (typename, storage_key, device, numel)
        // or: ("storage", storage_type, storage_key, device, numel)
        if let PickleValue::Tuple(ref items) = pid {
            if items.len() >= 4 {
                // Check if first item is "storage"
                if let PickleValue::String(ref s) = items[0] {
                    if s == "storage" && items.len() >= 5 {
                        // ("storage", storage_type_global, key, device, numel)
                        let dtype = match &items[1] {
                            PickleValue::Global { name, .. } => pytorch_storage_to_dtype(name),
                            _ => None,
                        };
                        let key = match &items[2] {
                            PickleValue::String(s) => s.clone(),
                            _ => return PickleValue::Opaque,
                        };
                        let numel = match &items[4] {
                            PickleValue::Int(v) => *v as usize,
                            _ => 0,
                        };
                        if let Some(dtype) = dtype {
                            return PickleValue::StorageRef { key, dtype, numel };
                        }
                    }
                }

                // Old format: (storage_type_global, key, device, numel)
                let dtype = match &items[0] {
                    PickleValue::Global { name, .. } => pytorch_storage_to_dtype(name),
                    _ => None,
                };
                let key = match &items[1] {
                    PickleValue::String(s) => s.clone(),
                    _ => return PickleValue::Opaque,
                };
                let numel = match &items[3] {
                    PickleValue::Int(v) => *v as usize,
                    _ => 0,
                };
                if let Some(dtype) = dtype {
                    return PickleValue::StorageRef { key, dtype, numel };
                }
            }
        }
        PickleValue::Opaque
    }
}

fn pytorch_storage_to_dtype(name: &str) -> Option<DType> {
    match name {
        "DoubleStorage" => Some(DType::F64),
        "FloatStorage" => Some(DType::F32),
        "HalfStorage" => Some(DType::F16),
        "BFloat16Storage" => Some(DType::BF16),
        "LongStorage" => Some(DType::I64),
        "IntStorage" => Some(DType::I32),
        "ShortStorage" => Some(DType::I16),
        "CharStorage" => Some(DType::I8),
        "ByteStorage" => Some(DType::U8),
        "BoolStorage" => Some(DType::Bool),
        _ => None,
    }
}

fn long_from_bytes(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    // Only read up to 8 bytes (i64 is 64 bits)
    let usable = bytes.len().min(8);
    let mut val: i64 = 0;
    for (i, &b) in bytes[..usable].iter().enumerate() {
        val |= (b as i64) << (i * 8);
    }
    // Sign extend if negative (based on the most significant usable byte)
    let sign_byte = bytes[usable - 1];
    if sign_byte & 0x80 != 0 {
        let bits = usable * 8;
        if bits < 64 {
            val |= !0i64 << bits;
        }
    }
    val
}

fn items_to_pairs(items: Vec<PickleValue>) -> Vec<(PickleValue, PickleValue)> {
    let mut pairs = Vec::new();
    let mut iter = items.into_iter();
    while let Some(key) = iter.next() {
        if let Some(value) = iter.next() {
            pairs.push((key, value));
        }
    }
    pairs
}

fn extract_tensors_from_stack(stack: &[PickleValue]) -> Result<Vec<PtTensorInfo>, Error> {
    let mut tensors = Vec::new();

    for val in stack {
        extract_tensors_recursive("", val, &mut tensors, 0)?;
    }

    Ok(tensors)
}

fn extract_tensors_recursive(
    prefix: &str,
    val: &PickleValue,
    out: &mut Vec<PtTensorInfo>,
    depth: usize,
) -> Result<(), Error> {
    if depth > MAX_EXTRACT_DEPTH {
        return Err(Error::Other(format!(
            "Pickle structure exceeds maximum nesting depth of {}", MAX_EXTRACT_DEPTH
        )));
    }

    match val {
        PickleValue::Dict(pairs) => {
            for (key, value) in pairs {
                let name = match key {
                    PickleValue::String(s) => {
                        if prefix.is_empty() {
                            s.clone()
                        } else {
                            format!("{}.{}", prefix, s)
                        }
                    }
                    _ => prefix.to_string(),
                };
                match value {
                    PickleValue::TensorRef(tref) => {
                        out.push(PtTensorInfo {
                            name,
                            dtype: tref.dtype,
                            shape: tref.shape.clone(),
                            storage_key: tref.storage_key.clone(),
                            storage_offset: tref.storage_offset,
                            numel: tref.shape.iter().try_fold(1u64, |a, &b| a.checked_mul(b)).unwrap_or(0) as usize,
                        });
                    }
                    _ => extract_tensors_recursive(&name, value, out, depth + 1)?,
                }
            }
        }
        PickleValue::TensorRef(tref) => {
            out.push(PtTensorInfo {
                name: prefix.to_string(),
                dtype: tref.dtype,
                shape: tref.shape.clone(),
                storage_key: tref.storage_key.clone(),
                storage_offset: tref.storage_offset,
                numel: tref.shape.iter().try_fold(1u64, |a, &b| a.checked_mul(b)).unwrap_or(0) as usize,
            });
        }
        PickleValue::List(items) => {
            for (i, item) in items.iter().enumerate() {
                let name = if prefix.is_empty() {
                    format!("{}", i)
                } else {
                    format!("{}.{}", prefix, i)
                };
                extract_tensors_recursive(&name, item, out, depth + 1)?;
            }
        }
        PickleValue::Tuple(items) => {
            for (i, item) in items.iter().enumerate() {
                let name = if prefix.is_empty() {
                    format!("{}", i)
                } else {
                    format!("{}.{}", prefix, i)
                };
                extract_tensors_recursive(&name, item, out, depth + 1)?;
            }
        }
        _ => {}
    }

    Ok(())
}
