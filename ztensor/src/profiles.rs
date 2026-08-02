//! L2 vocabulary: layout and encoding profiles.
//!
//! A profile is the code form of a registry mini-spec: it must be
//! implementable from its text alone, and a reader that does not know a
//! profile refuses to interpret — never guesses. `dense` is itself a
//! profile here; the container core has no layout special cases at all.

use crate::error::{Error, Result, Rule};
use crate::models::{logical_size, DType, Object};

/// A layout profile: how an object's parts combine into a tensor.
///
/// `validate` runs at open time (and at write time) on metadata only —
/// part names, dtypes, and decoded sizes. Data-level rules (e.g. CSR
/// index monotonicity) run when the object is actually assembled.
pub trait LayoutProfile: Send + Sync {
    fn id(&self) -> &'static str;
    fn validate(&self, name: &str, obj: &Object) -> Result<()>;
}

/// An encoding profile: a byte-stream transform for one part.
pub trait EncodingProfile: Send + Sync {
    fn id(&self) -> &'static str;
    fn encode(&self, decoded: &[u8]) -> Result<Vec<u8>>;
    /// Must produce exactly `decoded_length` bytes or reject.
    fn decode(&self, stored: &[u8], decoded_length: u64) -> Result<Vec<u8>>;
}

static LAYOUTS: &[&dyn LayoutProfile] = &[&Dense, &SparseCsr];

#[cfg(feature = "zstd")]
static ENCODINGS: &[&dyn EncodingProfile] = &[&zstd_seekable::ZstdSeekable];
#[cfg(not(feature = "zstd"))]
static ENCODINGS: &[&dyn EncodingProfile] = &[];

/// Looks up a built-in layout profile. `None` means structural-only access.
pub fn layout_profile(id: &str) -> Option<&'static dyn LayoutProfile> {
    LAYOUTS.iter().find(|p| p.id() == id).copied()
}

/// Looks up a built-in encoding profile. `None` means the stored bytes can
/// be accessed but not decoded.
pub fn encoding_profile(id: &str) -> Option<&'static dyn EncodingProfile> {
    ENCODINGS.iter().find(|p| p.id() == id).copied()
}

// =======================================================================
// dense (core, spec §5.1)
// =======================================================================

struct Dense;

impl LayoutProfile for Dense {
    fn id(&self) -> &'static str {
        "dense"
    }

    fn validate(&self, name: &str, obj: &Object) -> Result<()> {
        if obj.parts.len() != 1 || !obj.parts.contains_key("data") {
            return Err(Error::reject(
                Rule::LayoutRule,
                format!("{name:?}: dense requires exactly one part named 'data'"),
            ));
        }
        let part = &obj.parts["data"];
        let elems = obj.num_elements()?;
        // Checkable only when the size function is known; unknown logical
        // types stay structural (§4.2).
        if let Some(expected) = logical_size(part.ltype.as_deref(), part.dtype, elems) {
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
impl LayoutProfile for SparseCsr {
    fn id(&self) -> &'static str {
        "zt.sparse_csr/1"
    }

    fn validate(&self, name: &str, obj: &Object) -> Result<()> {
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

        if !matches!(indices.dtype, DType::U32 | DType::U64) || indices.ltype.is_some() {
            return fail(format!("{name:?}: 'indices' must be plain u32 or u64"));
        }
        if indptr.dtype != indices.dtype || indptr.ltype.is_some() {
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
        if let Some(expected) = logical_size(values.ltype.as_deref(), values.dtype, nnz) {
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

    use super::EncodingProfile;
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

    impl EncodingProfile for ZstdSeekable {
        fn id(&self) -> &'static str {
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
            let frames_n =
                u32::from_le_bytes(stored[n - 9..n - 5].try_into().unwrap()) as usize;
            let content_len = frames_n
                .checked_mul(entry_size)
                .and_then(|v| v.checked_add(9))
                .ok_or_else(|| bad("seek table size overflow"))?;
            let table_start = n
                .checked_sub(8 + content_len)
                .ok_or_else(|| bad("seek table larger than stream"))?;
            if stored[table_start..table_start + 4] != SKIPPABLE_MAGIC.to_le_bytes()
                || stored[table_start + 4..table_start + 8]
                    != (content_len as u32).to_le_bytes()
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
