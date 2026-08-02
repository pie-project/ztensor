//! The capability ladder: what a source can guarantee, queried per part.
//!
//! | Tier | Guarantee | Who provides it |
//! |------|-----------|-----------------|
//! | 0 | enumerate objects + metadata | every source |
//! | 1 | decoded read (owned bytes) | every source |
//! | 2 | zero-copy view | mapped sources, raw local parts |
//! | 3 | tier 2 + page-exclusive + verifiable | canonical `.zt` |
//!
//! Sources never degrade silently: `view()` on a part that cannot be
//! zero-copied is an error, not a hidden copy. Consumers that need a tier
//! ask [`Source::caps`] first.

use crate::error::Result;
use crate::models::Manifest;

/// Per-part capability report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Caps {
    /// A zero-copy [`Source::view`] will succeed.
    pub zero_copy: bool,
    /// Largest power of two dividing the part's file offset. The pointer
    /// alignment of a whole-file mapping is `min(alignment, page_size)`.
    pub alignment: u64,
    /// The part carries a digest, so a verified read is possible.
    pub verifiable: bool,
    /// No other blob shares an OS page with this part (at the current page
    /// size), so exact-range eviction cannot disturb a neighbor.
    pub page_exclusive: bool,
}

impl Caps {
    /// Highest ladder tier this part supports. Tier 0 (metadata) is always
    /// available and not represented.
    pub fn tier(&self) -> u8 {
        match (self.zero_copy, self.page_exclusive && self.verifiable) {
            (true, true) => 3,
            (true, false) => 2,
            _ => 1,
        }
    }

    /// Builds a report from a part's metadata plus the two properties only
    /// the source can decide.
    pub(crate) fn for_part(
        part: &crate::models::Part,
        zero_copy: bool,
        page_exclusive: bool,
    ) -> Self {
        Caps {
            zero_copy,
            alignment: 1u64 << part.blob.offset.trailing_zeros().min(63),
            verifiable: part.digest.is_some(),
            page_exclusive,
        }
    }
}

/// A tensor source: anything that projects into the zTensor object model.
///
/// The core [`Reader`](crate::Reader) is the identity projection; foreign
/// formats (safetensors, gguf, ...) implement this trait in the compat
/// crate by building a [`Manifest`] view of their own metadata.
pub trait Source {
    /// Tier 0: the object model view of this source.
    fn manifest(&self) -> &Manifest;

    /// Tier 1: decoded bytes, always by copy. Errors only on missing
    /// objects or vocabulary the implementation cannot decode — never by
    /// reinterpreting data.
    fn read(&self, object: &str, part: &str) -> Result<Vec<u8>>;

    /// Tier 2: zero-copy view of decoded bytes. Errors when zero-copy is
    /// impossible (encoded part, foreign shard, streaming source) — it
    /// never silently falls back to a copy.
    fn view(&self, object: &str, part: &str) -> Result<&[u8]>;

    /// Capability report for one part.
    fn caps(&self, object: &str, part: &str) -> Result<Caps>;
}

/// The OS page size (4096 on non-unix targets).
pub fn page_size() -> u64 {
    #[cfg(unix)]
    {
        // SAFETY: sysconf is always safe to call.
        let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if n > 0 {
            return n as u64;
        }
    }
    4096
}
