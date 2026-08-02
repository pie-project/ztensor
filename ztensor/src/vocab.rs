//! L2 vocabulary: layouts, encodings, and logical types — as a value.
//!
//! A profile is the code form of a registry mini-spec: implementable from its
//! text alone, and a reader that does not know one refuses to interpret rather
//! than guessing. The spec calls this layer registry-managed, so the registry
//! is a [`Vocabulary`] you can extend and hand to a reader or a writer:
//!
//! ```no_run
//! # use ztensor::{Source, Vocabulary};
//! # fn f(my_layout: impl ztensor::vocab::Layout + 'static) -> ztensor::Result<()> {
//! let vocab = Vocabulary::standard().with_layout(my_layout);
//! let src = Source::options().vocabulary(&vocab).open("model.zt")?;
//! # Ok(()) }
//! ```
//!
//! `dense` is a profile like any other; the container core has no layout
//! special cases at all.

use std::sync::{Arc, OnceLock};

use crate::error::{Error, Result, Rule};
use crate::schema::{DType, Object};

/// A layout profile: how an object's parts combine into a tensor.
///
/// `validate` runs at open time (and at write time) on metadata only — part
/// names, dtypes, and decoded sizes. Data-level rules (e.g. CSR index
/// monotonicity) run when the object is actually assembled.
pub trait Layout: Send + Sync {
    fn id(&self) -> &str;
    fn validate(&self, name: &str, obj: &Object, vocab: &Vocabulary) -> Result<()>;
}

/// An encoding profile: a byte-stream transform for one part.
pub trait Encoding: Send + Sync {
    fn id(&self) -> &str;
    fn encode(&self, decoded: &[u8]) -> Result<Vec<u8>>;
    /// Must produce exactly `decoded_length` bytes or reject.
    fn decode(&self, stored: &[u8], decoded_length: u64) -> Result<Vec<u8>>;
}

/// A logical type: an interpretation laid over a storage type (spec §4.2,
/// Appendix A).
///
/// Three facets, and a profile must answer all three: which storage type it
/// requires, how many bytes `n` elements occupy, and which byte patterns are
/// legal.
pub trait LogicalType: Send + Sync {
    fn id(&self) -> &str;
    /// The storage type this logical type pins, if it pins one.
    fn dtype(&self) -> Option<DType>;
    /// Decoded byte size of `elems` elements.
    fn size(&self, dtype: DType, elems: u64) -> Option<u64>;
    /// Content rules over decoded bytes. `elems` is `None` when the element
    /// count is not known (non-dense layouts).
    fn check(&self, _bytes: &[u8], _elems: Option<u64>) -> Result<()> {
        Ok(())
    }
}

/// The set of profiles a reader or writer knows.
///
/// Later registrations shadow earlier ones, so a caller can replace a standard
/// profile as well as add to it.
#[derive(Clone, Default)]
pub struct Vocabulary {
    layouts: Vec<Arc<dyn Layout>>,
    encodings: Vec<Arc<dyn Encoding>>,
    logicals: Vec<Arc<dyn LogicalType>>,
}

impl std::fmt::Debug for Vocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vocabulary")
            .field(
                "layouts",
                &self.layouts.iter().map(|p| p.id()).collect::<Vec<_>>(),
            )
            .field(
                "encodings",
                &self.encodings.iter().map(|p| p.id()).collect::<Vec<_>>(),
            )
            .field(
                "logicals",
                &self.logicals.iter().map(|p| p.id()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Vocabulary {
    /// Knows nothing. Every layout and logical type is structural, every
    /// encoding undecodable.
    pub fn empty() -> Self {
        Self::default()
    }

    /// The profiles this implementation ships: `dense`, `zt.sparse_csr/1`,
    /// `zt.zstd-seekable/1` (with the `zstd` feature), and the registered
    /// logical types of Appendix A.
    pub fn standard() -> Self {
        let mut v = Self::empty()
            .with_layout(Dense)
            .with_layout(SparseCsr)
            .with_logical(Bool)
            .with_logical(Fp4E2m1)
            .with_logical(Complex(64))
            .with_logical(Complex(128));
        for id in [
            "f8_e4m3fn",
            "f8_e5m2",
            "f8_e4m3fnuz",
            "f8_e5m2fnuz",
            "f8_e8m0",
        ] {
            v = v.with_logical(ByteWide(id));
        }
        #[cfg(feature = "zstd")]
        {
            v = v.with_encoding(zstd_seekable::ZstdSeekable);
        }
        v
    }

    /// The shared standard vocabulary, built once.
    pub(crate) fn shared() -> Arc<Vocabulary> {
        static STANDARD: OnceLock<Arc<Vocabulary>> = OnceLock::new();
        STANDARD
            .get_or_init(|| Arc::new(Vocabulary::standard()))
            .clone()
    }

    pub fn with_layout(mut self, profile: impl Layout + 'static) -> Self {
        self.layouts.push(Arc::new(profile));
        self
    }

    pub fn with_encoding(mut self, profile: impl Encoding + 'static) -> Self {
        self.encodings.push(Arc::new(profile));
        self
    }

    pub fn with_logical(mut self, profile: impl LogicalType + 'static) -> Self {
        self.logicals.push(Arc::new(profile));
        self
    }

    /// `None` means structural-only access: the object is readable as bytes,
    /// its layout rules unchecked.
    pub fn layout(&self, id: &str) -> Option<&dyn Layout> {
        self.layouts
            .iter()
            .rev()
            .find(|p| p.id() == id)
            .map(Arc::as_ref)
    }

    /// `None` means the stored bytes can be addressed but not decoded.
    pub fn encoding(&self, id: &str) -> Option<&dyn Encoding> {
        self.encodings
            .iter()
            .rev()
            .find(|p| p.id() == id)
            .map(Arc::as_ref)
    }

    pub fn logical(&self, id: &str) -> Option<&dyn LogicalType> {
        self.logicals
            .iter()
            .rev()
            .find(|p| p.id() == id)
            .map(Arc::as_ref)
    }

    /// Decoded byte size of `elems` elements of `dtype` seen through
    /// `logical`. `None` when the logical type is unregistered — the size is
    /// then simply unknown, and callers stay structural.
    pub fn size_of(&self, logical: Option<&str>, dtype: DType, elems: u64) -> Option<u64> {
        match logical {
            None => elems.checked_mul(dtype.width()),
            Some(id) => self.logical(id)?.size(dtype, elems),
        }
    }

    /// The storage type a logical type pins, if it is registered and pins one.
    pub fn dtype_of(&self, logical: &str) -> Option<DType> {
        self.logical(logical)?.dtype()
    }

    /// Content rules of a registered logical type; unknown types are exempt.
    pub fn check_values(
        &self,
        logical: &str,
        bytes: &[u8],
        elems: Option<u64>,
    ) -> Result<()> {
        match self.logical(logical) {
            Some(p) => p.check(bytes, elems),
            None => Ok(()),
        }
    }
}

// =======================================================================
// standard logical types (spec Appendix A)
// =======================================================================

/// A logical type stored one byte per element: the fp8 family.
struct ByteWide(&'static str);

impl LogicalType for ByteWide {
    fn id(&self) -> &str {
        self.0
    }
    fn dtype(&self) -> Option<DType> {
        Some(DType::U8)
    }
    fn size(&self, _dtype: DType, elems: u64) -> Option<u64> {
        Some(elems)
    }
}

struct Bool;

impl LogicalType for Bool {
    fn id(&self) -> &str {
        "bool"
    }
    fn dtype(&self) -> Option<DType> {
        Some(DType::U8)
    }
    fn size(&self, _dtype: DType, elems: u64) -> Option<u64> {
        Some(elems)
    }
    fn check(&self, bytes: &[u8], _elems: Option<u64>) -> Result<()> {
        if bytes.iter().any(|&b| b > 1) {
            return Err(Error::reject(
                Rule::LayoutData,
                "bool bytes must be 0x00 or 0x01",
            ));
        }
        Ok(())
    }
}

struct Fp4E2m1;

impl LogicalType for Fp4E2m1 {
    fn id(&self) -> &str {
        "f4_e2m1"
    }
    fn dtype(&self) -> Option<DType> {
        Some(DType::U8)
    }
    fn size(&self, _dtype: DType, elems: u64) -> Option<u64> {
        Some(elems.div_ceil(2))
    }
    /// Packed two per byte, low nibble first: an odd element count leaves the
    /// final high nibble unused, and it must be zero.
    fn check(&self, bytes: &[u8], elems: Option<u64>) -> Result<()> {
        if let Some(n) = elems {
            if n % 2 == 1 && bytes.last().is_some_and(|&b| b & 0xf0 != 0) {
                return Err(Error::reject(
                    Rule::LayoutData,
                    "final odd f4 nibble must be zero",
                ));
            }
        }
        Ok(())
    }
}

/// `complex64` / `complex128`: interleaved (re, im) pairs of the storage type.
struct Complex(u32);

impl LogicalType for Complex {
    fn id(&self) -> &str {
        match self.0 {
            64 => "complex64",
            _ => "complex128",
        }
    }
    fn dtype(&self) -> Option<DType> {
        Some(match self.0 {
            64 => DType::F32,
            _ => DType::F64,
        })
    }
    fn size(&self, _dtype: DType, elems: u64) -> Option<u64> {
        elems.checked_mul(if self.0 == 64 { 8 } else { 16 })
    }
}

// =======================================================================
// dense (spec §5.1)
// =======================================================================

struct Dense;

impl Layout for Dense {
    fn id(&self) -> &str {
        "dense"
    }

    fn validate(&self, name: &str, obj: &Object, vocab: &Vocabulary) -> Result<()> {
        if obj.parts.len() != 1 || !obj.parts.contains_key("data") {
            return Err(Error::reject(
                Rule::LayoutRule,
                format!("{name:?}: dense requires exactly one part named 'data'"),
            ));
        }
        let part = &obj.parts["data"];
        let elems = obj.num_elements()?;
        // Checkable only when the size function is known; unregistered logical
        // types stay structural (§4.2).
        if let Some(expected) = vocab.size_of(part.logical.as_deref(), part.dtype, elems) {
            if part.decoded_size() != expected {
                return Err(Error::reject(
                    Rule::DenseSize,
                    format!(
                        "{name:?}: decoded size {} != expected {expected}",
                        part.decoded_size()
                    ),
                ));
            }
        }
        Ok(())
    }
}

// =======================================================================
// zt.sparse_csr/1 (spec Appendix B)
// =======================================================================

struct SparseCsr;

/// `nnz` is derived from `indices` (whose size function is exact), then
/// `values` is validated against it — this stays well-defined even when
/// `values` uses a packed sub-byte type.
impl Layout for SparseCsr {
    fn id(&self) -> &str {
        "zt.sparse_csr/1"
    }

    fn validate(&self, name: &str, obj: &Object, vocab: &Vocabulary) -> Result<()> {
        let fail = |detail: String| Err(Error::reject(Rule::LayoutRule, detail));
        let [rows, _cols] = obj.shape[..] else {
            return fail(format!("{name:?}: sparse_csr requires rank-2 shape"));
        };
        if obj.parts.len() != 3 {
            return fail(format!("{name:?}: sparse_csr requires exactly 3 parts"));
        }
        let (Some(values), Some(indices), Some(indptr)) = (
            obj.parts.get("values"),
            obj.parts.get("indices"),
            obj.parts.get("indptr"),
        ) else {
            return fail(format!(
                "{name:?}: sparse_csr requires parts 'values', 'indices', 'indptr'"
            ));
        };

        if !matches!(indices.dtype, DType::U32 | DType::U64) || indices.logical.is_some() {
            return fail(format!("{name:?}: 'indices' must be plain u32 or u64"));
        }
        if indptr.dtype != indices.dtype || indptr.logical.is_some() {
            return fail(format!("{name:?}: 'indptr' must match the 'indices' dtype"));
        }
        let w = indices.dtype.width();

        let expected_indptr = rows
            .checked_add(1)
            .and_then(|n| n.checked_mul(w))
            .ok_or_else(|| Error::reject(Rule::Shape, format!("{name:?}: rows overflow")))?;
        if indptr.decoded_size() != expected_indptr {
            return fail(format!(
                "{name:?}: 'indptr' must hold rows+1 elements ({} bytes, got {})",
                expected_indptr,
                indptr.decoded_size()
            ));
        }

        if indices.decoded_size() % w != 0 {
            return fail(format!("{name:?}: 'indices' size not a multiple of {w}"));
        }
        let nnz = indices.decoded_size() / w;
        if let Some(expected) = vocab.size_of(values.logical.as_deref(), values.dtype, nnz) {
            if values.decoded_size() != expected {
                return fail(format!(
                    "{name:?}: 'values' decoded size {} != {expected} for nnz {nnz}",
                    values.decoded_size()
                ));
            }
        }
        Ok(())
    }
}

// =======================================================================
// zt.zstd-seekable/1 (spec Appendix C)
// =======================================================================

#[cfg(feature = "zstd")]
mod zstd_seekable {
    use std::io::Write;

    use super::Encoding;
    use crate::error::{Error, Result, Rule};

    /// Decoded bytes per frame. Spec: ≤ 16 MiB, all frames equal-sized
    /// except the last.
    const CHUNK: usize = 1 << 20;
    const MAX_FRAME: u64 = 16 << 20;
    const LEVEL: i32 = 3;
    const SKIPPABLE_MAGIC: u32 = 0x184D2A5E;
    const SEEKABLE_MAGIC: u32 = 0x8F92EAB1;

    pub struct ZstdSeekable;

    fn bad(detail: impl Into<String>) -> Error {
        Error::reject(Rule::Encoding, detail.into())
    }

    impl Encoding for ZstdSeekable {
        fn id(&self) -> &str {
            "zt.zstd-seekable/1"
        }

        fn encode(&self, decoded: &[u8]) -> Result<Vec<u8>> {
            let mut out = Vec::new();
            let mut entries: Vec<(u32, u32)> = Vec::new();
            for chunk in decoded.chunks(CHUNK) {
                let mut enc = zstd::stream::write::Encoder::new(Vec::new(), LEVEL)?;
                enc.include_checksum(true)?;
                enc.write_all(chunk)?;
                let frame = enc.finish()?;
                entries.push((frame.len() as u32, chunk.len() as u32));
                out.extend_from_slice(&frame);
            }
            // Seek table: a skippable frame, then the seekable footer.
            let content_len = entries.len() * 8 + 9;
            out.extend(SKIPPABLE_MAGIC.to_le_bytes());
            out.extend((content_len as u32).to_le_bytes());
            for (c, d) in &entries {
                out.extend(c.to_le_bytes());
                out.extend(d.to_le_bytes());
            }
            out.extend((entries.len() as u32).to_le_bytes());
            out.push(0u8); // descriptor: no per-frame checksum column
            out.extend(SEEKABLE_MAGIC.to_le_bytes());
            Ok(out)
        }

        fn decode(&self, stored: &[u8], decoded_length: u64) -> Result<Vec<u8>> {
            let n = stored.len();
            if n < 17 {
                return Err(bad("stream too short for a seek table"));
            }
            if stored[n - 4..] != SEEKABLE_MAGIC.to_le_bytes() {
                return Err(bad("missing seekable footer magic"));
            }
            let descriptor = stored[n - 5];
            if descriptor & 0x7f != 0 {
                return Err(bad("reserved descriptor bits set"));
            }
            let entry_size = if descriptor & 0x80 != 0 { 12 } else { 8 };
            let frames_n = u32::from_le_bytes(stored[n - 9..n - 5].try_into().unwrap()) as usize;
            let content_len = frames_n
                .checked_mul(entry_size)
                .and_then(|v| v.checked_add(9))
                .ok_or_else(|| bad("seek table size overflow"))?;
            let table_start = n
                .checked_sub(8 + content_len)
                .ok_or_else(|| bad("seek table larger than stream"))?;
            if stored[table_start..table_start + 4] != SKIPPABLE_MAGIC.to_le_bytes()
                || stored[table_start + 4..table_start + 8] != (content_len as u32).to_le_bytes()
            {
                return Err(bad("malformed seek table skippable frame"));
            }

            let mut expected: Vec<(u64, u64)> = Vec::with_capacity(frames_n);
            let mut pos = table_start + 8;
            let mut total_c = 0u64;
            let mut total_d = 0u64;
            for _ in 0..frames_n {
                let c = u32::from_le_bytes(stored[pos..pos + 4].try_into().unwrap()) as u64;
                let d = u32::from_le_bytes(stored[pos + 4..pos + 8].try_into().unwrap()) as u64;
                pos += entry_size;
                if d > MAX_FRAME {
                    return Err(bad("frame content exceeds 16 MiB"));
                }
                total_c += c;
                total_d += d;
                expected.push((c, d));
            }
            if total_d != decoded_length {
                return Err(bad(format!(
                    "seek table totals {total_d} bytes, declared decoded_length {decoded_length}"
                )));
            }
            if total_c != table_start as u64 {
                return Err(bad("frame sizes do not cover the stream"));
            }
            if let Some((_, first_d)) = expected.first() {
                for (_, d) in &expected[..expected.len() - 1] {
                    if d != first_d {
                        return Err(bad("non-final frames must be equal-sized"));
                    }
                }
            }

            let mut out = Vec::with_capacity(decoded_length as usize);
            let mut fpos = 0usize;
            for (c, d) in expected {
                let frame = &stored[fpos..fpos + c as usize];
                let decoded = zstd::bulk::decompress(frame, d as usize)
                    .map_err(|e| bad(format!("frame decompression failed: {e}")))?;
                if decoded.len() as u64 != d {
                    return Err(bad("frame decoded to a different size than declared"));
                }
                out.extend_from_slice(&decoded);
                fpos += c as usize;
            }
            Ok(out)
        }
    }
}
