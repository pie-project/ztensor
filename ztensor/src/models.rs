//! Object model and manifest schema (spec L1/L2), plus container constants.

use std::collections::BTreeMap;

use crate::cbor::Value;
use crate::error::{Error, Result, Rule};

/// Magic bytes at offset 0 and at the end of the footer (spec §2.2).
pub const MAGIC: [u8; 8] = [0x89, b'Z', b'T', b'2', 0x0d, 0x0a, 0x1a, 0x0a];
/// Footer version integer defined by this implementation (spec §2.3).
pub const VERSION: u32 = 2;
/// Fixed footer size in bytes.
pub const FOOTER_LEN: u64 = 40;
/// Alignment floor: every blob offset is a multiple of this (spec §2.4).
pub const ALIGN_FLOOR: u64 = 4096;
/// Canonical placement alignment (spec §6.3).
pub const ALIGN_CANONICAL: u64 = 65536;
/// Manifest size cap (spec §3.1).
pub const MAX_MANIFEST_LEN: u64 = 1 << 30;
/// Maximum name length in bytes (spec §3.5).
pub const MAX_NAME_LEN: usize = 1024;
/// Maximum shape rank (spec §3.3).
pub const MAX_RANK: usize = 64;
/// Minimum container size: header magic plus footer (spec §2.1).
pub const MIN_FILE_LEN: u64 = MAGIC.len() as u64 + FOOTER_LEN;

/// Storage types: the closed set of 12 primitives (spec §4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

impl DType {
    pub fn width(self) -> u64 {
        match self {
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
            DType::I8 | DType::U8 => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::I16 => "i16",
            DType::I8 => "i8",
            DType::U64 => "u64",
            DType::U32 => "u32",
            DType::U16 => "u16",
            DType::U8 => "u8",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "f64" => DType::F64,
            "f32" => DType::F32,
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "i64" => DType::I64,
            "i32" => DType::I32,
            "i16" => DType::I16,
            "i8" => DType::I8,
            "u64" => DType::U64,
            "u32" => DType::U32,
            "u16" => DType::U16,
            "u8" => DType::U8,
            _ => return None,
        })
    }
}

/// Registered logical types (spec Appendix A): required storage type.
/// Returns `None` for unknown logical types (structural access only).
pub fn registered_dtype(ltype: &str) -> Option<DType> {
    Some(match ltype {
        "bool" | "f8_e4m3fn" | "f8_e5m2" | "f8_e4m3fnuz" | "f8_e5m2fnuz" | "f8_e8m0"
        | "f4_e2m1" => DType::U8,
        "complex64" => DType::F32,
        "complex128" => DType::F64,
        _ => return None,
    })
}

/// Size function of a registered logical type: decoded byte size for `n`
/// logical elements. `None` when the logical type is unknown.
pub fn logical_size(ltype: Option<&str>, dtype: DType, n: u64) -> Option<u64> {
    match ltype {
        None => n.checked_mul(dtype.width()),
        Some("bool" | "f8_e4m3fn" | "f8_e5m2" | "f8_e4m3fnuz" | "f8_e5m2fnuz" | "f8_e8m0") => {
            Some(n)
        }
        Some("f4_e2m1") => Some(n.div_ceil(2)),
        Some("complex64") => n.checked_mul(8),
        Some("complex128") => n.checked_mul(16),
        Some(_) => None,
    }
}

/// Object layout (spec §5). Only `dense` is core; everything else is a
/// namespaced profile identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layout {
    Dense,
    Other(String),
}

impl Layout {
    pub fn as_str(&self) -> &str {
        match self {
            Layout::Dense => "dense",
            Layout::Other(s) => s,
        }
    }

    pub fn from_name(s: &str) -> Self {
        match s {
            "dense" => Layout::Dense,
            other => Layout::Other(other.to_string()),
        }
    }
}

/// Blob reference: `[shard_index, offset, length]` (spec §3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobRef {
    pub shard: u64,
    pub offset: u64,
    pub length: u64,
}

/// A part: one blob plus its interpretation (spec §3.4).
#[derive(Debug, Clone, PartialEq)]
pub struct Part {
    pub dtype: DType,
    /// Logical type; `None` means the logical type equals `dtype`.
    pub ltype: Option<String>,
    pub blob: BlobRef,
    /// Encoding profile id; `None` means raw.
    pub encoding: Option<String>,
    /// Required iff `encoding` is present.
    pub decoded_length: Option<u64>,
    /// `"<algorithm>:<lowercase hex>"` over decoded bytes.
    pub digest: Option<String>,
}

impl Part {
    /// Decoded byte size: `length` for raw, `decoded_length` when encoded.
    pub fn decoded_size(&self) -> u64 {
        match self.encoding {
            None => self.blob.length,
            Some(_) => self.decoded_length.unwrap_or(0),
        }
    }
}

/// A named object (spec §3.3).
#[derive(Debug, Clone, PartialEq)]
pub struct Object {
    pub shape: Vec<u64>,
    pub layout: Layout,
    pub attributes: Option<Value>,
    pub parts: BTreeMap<String, Part>,
}

impl Object {
    /// Element count: product of dimensions; empty shape is a scalar (1).
    pub fn num_elements(&self) -> Result<u64> {
        self.shape.iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d)
                .ok_or_else(|| Error::reject(Rule::Shape, "shape product overflows u64"))
        })
    }
}

/// Shard identity: size and digest, never a name (spec §7.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Shard {
    pub size: u64,
    pub digest: String,
}

/// Root manifest (spec §3.2).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Manifest {
    pub attributes: Option<Value>,
    /// Keyed by shard index ≥ 1; index 0 (the containing file) never appears.
    pub shards: BTreeMap<u64, Shard>,
    pub objects: BTreeMap<String, Object>,
}

impl Manifest {
    /// Looks up an object by name.
    pub fn object(&self, name: &str) -> Result<&Object> {
        self.objects
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("object {name:?}")))
    }

    /// Looks up a part of an object.
    pub fn part(&self, name: &str, part: &str) -> Result<&Part> {
        self.object(name)?
            .parts
            .get(part)
            .ok_or_else(|| Error::NotFound(format!("part {name:?}/{part:?}")))
    }
}

/// Name rules (spec §3.5): non-empty UTF-8 (guaranteed by CBOR decode),
/// ≤ 1024 bytes, no NUL. NFC is a writer duty, not a reader check.
pub(crate) fn check_name(s: &str) -> Result<()> {
    if s.is_empty() || s.len() > MAX_NAME_LEN || s.contains('\0') {
        return Err(Error::reject(Rule::Name, format!("invalid name {s:?}")));
    }
    Ok(())
}

/// Attributes rules (spec §3.1/§3.5): the value MUST be a map whose
/// top-level keys are text obeying the name rules. Nested values are free
/// within the §3.1 type set (the codec already enforces that).
pub(crate) fn check_attributes(v: &Value) -> Result<()> {
    let entries = v
        .as_map()
        .ok_or_else(|| Error::reject(Rule::Schema, "'attributes' must be a map"))?;
    for (k, _) in entries {
        let key = k
            .as_text()
            .ok_or_else(|| Error::reject(Rule::Schema, "attribute keys must be text"))?;
        check_name(key)?;
    }
    Ok(())
}

/// Parses an `xxh3:` digest string to its expected value. Errors on other
/// algorithms (Unsupported) or malformed hex (Reject).
pub(crate) fn parse_xxh3(digest: &str) -> Result<u64> {
    let hex = digest.strip_prefix("xxh3:").ok_or_else(|| {
        Error::Unsupported(format!("digest algorithm in {digest:?} (only xxh3 supported)"))
    })?;
    u64::from_str_radix(hex, 16)
        .map_err(|_| Error::reject(Rule::Digest, format!("malformed digest {digest:?}")))
}

/// Content rules of registered logical types (spec Appendix A): applied
/// wherever bytes are decoded or verified; raw structural access is exempt.
pub(crate) fn check_logical_values(
    ltype: &str,
    bytes: &[u8],
    elems: Option<u64>,
) -> Result<()> {
    match ltype {
        "bool" => {
            if bytes.iter().any(|&b| b > 1) {
                return Err(Error::reject(
                    Rule::LayoutData,
                    "bool bytes must be 0x00 or 0x01",
                ));
            }
        }
        "f4_e2m1" => {
            // Packed two per byte, low nibble first: an odd element count
            // leaves the final high nibble unused, and it must be zero.
            if let Some(n) = elems {
                if n % 2 == 1 && bytes.last().is_some_and(|&b| b & 0xf0 != 0) {
                    return Err(Error::reject(
                        Rule::LayoutData,
                        "final odd f4 nibble must be zero",
                    ));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Digest format (spec §3.4): `"<algorithm>:<lowercase hex>"`.
pub(crate) fn check_digest(d: &str) -> Result<()> {
    let ok = d.split_once(':').is_some_and(|(algo, hex)| {
        !algo.is_empty()
            && !hex.is_empty()
            && algo.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
            && hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    });
    if !ok {
        return Err(Error::reject(
            Rule::Schema,
            format!("digest must be 'algo:lowercase-hex', got {d:?}"),
        ));
    }
    Ok(())
}

// =======================================================================
// Manifest <-> CBOR
// =======================================================================

// ---- schema decoding helpers ------------------------------------------

/// A required field that turned out to be absent.
fn missing<T>(what: &str, field: &str) -> Result<T> {
    Err(Error::reject(
        Rule::Schema,
        format!("{what} missing {field:?}"),
    ))
}

impl Value {
    fn text_or(&self, field: &str) -> Result<&str> {
        self.as_text()
            .ok_or_else(|| Error::reject(Rule::Schema, format!("{field:?} must be text")))
    }

    fn u64_or(&self, field: &str) -> Result<u64> {
        self.as_u64().ok_or_else(|| {
            Error::reject(Rule::Schema, format!("{field:?} must be an unsigned int"))
        })
    }

    fn map_or(&self, field: &str) -> Result<&[(Value, Value)]> {
        self.as_map()
            .ok_or_else(|| Error::reject(Rule::Schema, format!("{field:?} must be a map")))
    }

    fn array_or(&self, field: &str) -> Result<&[Value]> {
        self.as_array()
            .ok_or_else(|| Error::reject(Rule::Schema, format!("{field:?} must be an array")))
    }

    /// An array of unsigned integers (shapes, blob triples).
    fn uints_or(&self, field: &str) -> Result<Vec<u64>> {
        self.array_or(field)?
            .iter()
            .map(|v| v.u64_or(field))
            .collect()
    }
}


fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

impl Manifest {
    pub(crate) fn to_value(&self) -> Value {
        let mut root = Vec::new();
        if let Some(attrs) = &self.attributes {
            root.push((text("attributes"), attrs.clone()));
        }
        if !self.shards.is_empty() {
            let shards = self
                .shards
                .iter()
                .map(|(&idx, s)| {
                    (
                        Value::Uint(idx),
                        Value::Map(vec![
                            (text("size"), Value::Uint(s.size)),
                            (text("digest"), text(&s.digest)),
                        ]),
                    )
                })
                .collect();
            root.push((text("shards"), Value::Map(shards)));
        }
        let objects = self
            .objects
            .iter()
            .map(|(name, obj)| (text(name), obj.to_value()))
            .collect();
        root.push((text("objects"), Value::Map(objects)));
        Value::Map(root)
    }

    pub(crate) fn from_value(v: Value) -> Result<Manifest> {
        let entries = match v {
            Value::Map(m) => m,
            _ => return Err(Error::reject(Rule::Schema, "manifest root must be a map")),
        };
        let mut manifest = Manifest::default();
        let mut has_objects = false;
        for (k, val) in entries {
            let Some(key) = k.as_text() else {
                continue; // unknown (non-text) root key: ignore
            };
            match key {
                "attributes" => {
                    check_attributes(&val)?;
                    manifest.attributes = Some(val);
                }
                "shards" => manifest.shards = parse_shards(val)?,
                "objects" => {
                    has_objects = true;
                    manifest.objects = parse_objects(val)?;
                }
                _ => {} // unknown fields are ignored (spec §3.1)
            }
        }
        if !has_objects {
            return Err(Error::reject(Rule::Schema, "manifest missing 'objects'"));
        }
        Ok(manifest)
    }
}

fn parse_shards(v: Value) -> Result<BTreeMap<u64, Shard>> {
    let entries = v.map_or("shards")?;
    let mut shards = BTreeMap::new();
    for (k, val) in entries {
        let idx = k.u64_or("shard key")?;
        if idx == 0 {
            return Err(Error::reject(
                Rule::ShardIndex,
                "shard index 0 is the containing file and must not appear",
            ));
        }
        let m = val.map_or("shard entry")?;
        let mut size = None;
        let mut digest = None;
        for (fk, fv) in m {
            match fk.as_text() {
                Some("size") => size = fv.as_u64(),
                Some("digest") => digest = fv.as_text().map(str::to_string),
                _ => {}
            }
        }
        let Some(size) = size else {
            return missing("shard entry", "size");
        };
        let Some(digest) = digest else {
            return missing("shard entry", "digest");
        };
        check_digest(&digest)?;
        shards.insert(idx, Shard { size, digest });
    }
    Ok(shards)
}

fn parse_objects(v: Value) -> Result<BTreeMap<String, Object>> {
    let entries = v.map_or("objects")?;
    let mut objects = BTreeMap::new();
    for (k, val) in entries {
        let name = k.text_or("object name")?;
        check_name(name)?;
        objects.insert(name.to_string(), Object::from_value(val)?);
    }
    Ok(objects)
}

impl Object {
    fn to_value(&self) -> Value {
        let mut m = Vec::new();
        m.push((
            text("shape"),
            Value::Array(self.shape.iter().map(|&d| Value::Uint(d)).collect()),
        ));
        m.push((text("layout"), text(self.layout.as_str())));
        if let Some(attrs) = &self.attributes {
            m.push((text("attributes"), attrs.clone()));
        }
        let parts = self
            .parts
            .iter()
            .map(|(name, part)| (text(name), part.to_value()))
            .collect();
        m.push((text("parts"), Value::Map(parts)));
        Value::Map(m)
    }

    fn from_value(v: &Value) -> Result<Object> {
        let entries = v.map_or("object")?;
        let mut shape = None;
        let mut layout = None;
        let mut attributes = None;
        let mut parts = None;
        for (k, val) in entries {
            match k.as_text() {
                Some("shape") => shape = Some(val.uints_or("shape")?),
                Some("layout") => layout = Some(Layout::from_name(val.text_or("layout")?)),
                Some("attributes") => {
                    check_attributes(val)?;
                    attributes = Some(val.clone());
                }
                Some("parts") => parts = Some(parse_parts(val)?),
                _ => {}
            }
        }
        let (Some(shape), Some(layout), Some(parts)) = (shape, layout, parts) else {
            return missing("object", "shape/layout/parts");
        };
        if parts.is_empty() {
            return Err(Error::reject(Rule::Schema, "object has no parts"));
        }
        Ok(Object {
            shape,
            layout,
            attributes,
            parts,
        })
    }
}

fn parse_parts(v: &Value) -> Result<BTreeMap<String, Part>> {
    let entries = v.map_or("parts")?;
    let mut parts = BTreeMap::new();
    for (k, val) in entries {
        let name = k.text_or("part name")?;
        check_name(name)?;
        parts.insert(name.to_string(), Part::from_value(val)?);
    }
    Ok(parts)
}

impl Part {
    fn to_value(&self) -> Value {
        let mut m = Vec::new();
        m.push((text("dtype"), text(self.dtype.as_str())));
        if let Some(lt) = &self.ltype {
            m.push((text("type"), text(lt)));
        }
        m.push((
            text("blob"),
            Value::Array(vec![
                Value::Uint(self.blob.shard),
                Value::Uint(self.blob.offset),
                Value::Uint(self.blob.length),
            ]),
        ));
        if let Some(enc) = &self.encoding {
            m.push((text("encoding"), text(enc)));
        }
        if let Some(dl) = self.decoded_length {
            m.push((text("decoded_length"), Value::Uint(dl)));
        }
        if let Some(d) = &self.digest {
            m.push((text("digest"), text(d)));
        }
        Value::Map(m)
    }

    fn from_value(v: &Value) -> Result<Part> {
        let entries = v.map_or("part")?;
        let mut dtype = None;
        let mut ltype = None;
        let mut blob = None;
        let mut encoding = None;
        let mut decoded_length = None;
        let mut digest = None;
        for (k, val) in entries {
            match k.as_text() {
                Some("dtype") => {
                    let s = val.text_or("dtype")?;
                    dtype = Some(DType::from_name(s).ok_or_else(|| {
                        Error::reject(Rule::Schema, format!("unknown dtype {s:?}"))
                    })?);
                }
                Some("type") => ltype = Some(val.text_or("type")?.to_string()),
                Some("blob") => {
                    let nums = val.uints_or("blob")?;
                    let [shard, offset, length] = nums[..] else {
                        return Err(Error::reject(Rule::Schema, "'blob' must have 3 elements"));
                    };
                    blob = Some(BlobRef {
                        shard,
                        offset,
                        length,
                    });
                }
                Some("encoding") => encoding = Some(val.text_or("encoding")?.to_string()),
                Some("decoded_length") => {
                    decoded_length = Some(val.u64_or("decoded_length")?)
                }
                Some("digest") => {
                    let d = val.text_or("digest")?;
                    check_digest(d)?;
                    digest = Some(d.to_string());
                }
                _ => {}
            }
        }
        let (Some(dtype), Some(blob)) = (dtype, blob) else {
            return missing("part", "dtype/blob");
        };
        if encoding.is_some() != decoded_length.is_some() {
            return Err(Error::reject(
                Rule::Schema,
                "'decoded_length' is required iff 'encoding' is present",
            ));
        }
        Ok(Part {
            dtype,
            ltype,
            blob,
            encoding,
            decoded_length,
            digest,
        })
    }
}
