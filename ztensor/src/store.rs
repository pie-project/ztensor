//! One file, and what can be done with its bytes.
//!
//! A [`Store`] is opened in one of two ways. `map` establishes a read-only
//! shared mapping, which is what makes borrowed reads and page-exact eviction
//! possible. `index` opens the file without mapping it: enough to answer where
//! every tensor lives and to read a range on demand, which is all a planner
//! needs and costs no address space.
//!
//! Which one a caller got is not hidden — it is exactly the difference between
//! [`Caps::map`](crate::Caps::map) and [`Caps::locate`](crate::Caps::locate).

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};

/// Index of a [`Store`] within a [`Source`](crate::Source).
///
/// Process-local. Unrelated to a manifest's shard index, which is a claim one
/// file makes about others; this is where the bytes actually are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoreId(pub u32);

impl std::fmt::Display for StoreId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Bytes only the projection that opened the file can produce — a deflated zip
/// entry, a chunked HDF5 dataset. There is no address for these, so they are
/// readable and nothing else, and they say so.
pub trait Opaque {
    fn read(&self, key: u64, decoded_len: u64) -> Result<Vec<u8>>;
}

pub struct Store {
    path: PathBuf,
    file: File,
    len: u64,
    format: &'static str,
    map: Option<Mmap>,
    /// Every occupied byte range in this file, sorted and deduplicated —
    /// the basis for page exclusivity. Empty means *unknown*, and then
    /// exclusivity is never claimed rather than optimistically assumed.
    occupied: Vec<(u64, u64)>,
    opaque: Option<Box<dyn Opaque>>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("path", &self.path)
            .field("format", &self.format)
            .field("len", &self.len)
            .field("mapped", &self.map.is_some())
            .finish()
    }
}

impl Store {
    /// Opens and memory-maps a file.
    pub fn map(path: impl AsRef<Path>, format: &'static str) -> Result<Self> {
        let mut store = Self::index(path, format)?;
        // SAFETY: read-only shared map; the contents are treated as untrusted
        // bytes and never assumed stable beyond the validation snapshot.
        store.map = Some(unsafe { Mmap::map(&store.file)? });
        Ok(store)
    }

    /// Opens a file without mapping it. Ranges are read on demand.
    pub fn index(path: impl AsRef<Path>, format: &'static str) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        Ok(Self {
            path,
            file,
            len,
            format,
            map: None,
            occupied: Vec::new(),
            opaque: None,
        })
    }

    /// Declares every byte range this file is known to occupy. Without it a
    /// store never reports page exclusivity.
    pub fn with_occupied(mut self, mut ranges: Vec<(u64, u64)>) -> Self {
        ranges.sort_unstable();
        ranges.dedup();
        self.occupied = ranges;
        self
    }

    /// Attaches the reader for payloads that have no address.
    pub fn with_opaque(mut self, opaque: Box<dyn Opaque>) -> Self {
        self.opaque = Some(opaque);
        self
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The format label this file was detected as: `"zt"`, `"safetensors"`,
    /// `"gguf"`, `"npz"`, `"pt"`, `"hdf5"`, `"onnx"`.
    pub fn format(&self) -> &'static str {
        self.format
    }

    pub fn is_mapped(&self) -> bool {
        self.map.is_some()
    }

    /// The whole mapping, if there is one.
    pub fn bytes(&self) -> Option<&[u8]> {
        self.map.as_deref()
    }

    fn bounded(&self, offset: u64, len: u64) -> Result<(usize, usize)> {
        let end = offset
            .checked_add(len)
            .filter(|&e| e <= self.len)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "range {offset}+{len} is outside {} ({} bytes)",
                    self.path.display(),
                    self.len
                ))
            })?;
        // Sound on every platform: `end <= self.len`, and a range that large
        // could not have been mapped or read on a machine whose usize is
        // smaller.
        Ok((offset as usize, end as usize))
    }

    /// A borrowed slice of the mapping. `None` when the file is not mapped.
    pub fn slice(&self, offset: u64, len: u64) -> Result<Option<&[u8]>> {
        let (start, end) = self.bounded(offset, len)?;
        Ok(self.map.as_ref().map(|m| &m[start..end]))
    }

    /// Owned bytes of a range, from the mapping when there is one and from the
    /// file otherwise.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let (start, end) = self.bounded(offset, len)?;
        if let Some(map) = &self.map {
            return Ok(map[start..end].to_vec());
        }
        let mut buf = vec![0u8; end - start];
        read_exact_at(&self.file, &mut buf, offset)?;
        Ok(buf)
    }

    pub(crate) fn opaque(&self) -> Option<&dyn Opaque> {
        self.opaque.as_deref()
    }

    /// True iff the page-aligned envelope of `[offset, offset + len)`
    /// intersects no *other* occupied range, so exact-range eviction cannot
    /// disturb a neighbour. Zero-length ranges are vacuously exclusive; an
    /// unknown occupancy map is never exclusive.
    pub fn page_exclusive(&self, offset: u64, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        if self.occupied.is_empty() {
            return false;
        }
        let page = page_size();
        let (env_start, env_end) = page_envelope(offset, len, page);
        let Ok(i) = self.occupied.binary_search(&(offset, len)) else {
            return false; // not a range we know about
        };
        let prev_clear = i == 0 || {
            let (o, l) = self.occupied[i - 1];
            o + l <= env_start
        };
        let next_clear = i + 1 >= self.occupied.len() || self.occupied[i + 1].0 >= env_end;
        prev_clear && next_clear
    }

    /// Hints the OS to prefetch a range's pages. A no-op when unmapped.
    pub fn prefetch(&self, offset: u64, len: u64) -> Result<()> {
        let (start, end) = self.bounded(offset, len)?;
        #[cfg(unix)]
        if let Some(map) = &self.map {
            if end > start {
                map.advise_range(memmap2::Advice::WillNeed, start, end - start)?;
            }
        }
        let _ = (start, end);
        Ok(())
    }

    /// Drops the page cache for a range's exact page envelope.
    ///
    /// The caller is responsible for exclusivity; [`Source`](crate::Source)
    /// checks [`page_exclusive`](Self::page_exclusive) first, which is the
    /// same predicate `Caps::evict` reports.
    #[cfg(unix)]
    pub fn evict(&self, offset: u64, len: u64) -> Result<()> {
        let (_, _) = self.bounded(offset, len)?;
        let Some(map) = &self.map else {
            return Ok(());
        };
        if len == 0 {
            return Ok(());
        }
        let (start, end) = page_envelope(offset, len, page_size());
        let end = end.min(map.len() as u64);
        // SAFETY: the map is a read-only shared file mapping — DontNeed only
        // drops clean page-cache pages; later accesses re-fault from the file.
        // It cannot discard writes because none exist.
        unsafe {
            map.unchecked_advise_range(
                memmap2::UncheckedAdvice::DontNeed,
                start as usize,
                (end - start) as usize,
            )?;
        }
        Ok(())
    }
}

/// Page-aligned envelope of a byte range.
pub(crate) fn page_envelope(offset: u64, length: u64, page: u64) -> (u64, u64) {
    (
        offset & !(page - 1),
        (offset + length).div_ceil(page).saturating_mul(page),
    )
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

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut handle = file.try_clone()?;
        handle.seek(SeekFrom::Start(offset))?;
        handle.read_exact(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(ranges: &[(u64, u64)]) -> Store {
        let path = std::env::temp_dir().join("ztensor-store-exclusivity-probe");
        std::fs::write(&path, [0u8; 1]).unwrap();
        Store::index(&path, "zt")
            .unwrap()
            .with_occupied(ranges.to_vec())
    }

    /// The page-exclusivity predicate, which is the whole of `Caps::evict`.
    #[test]
    fn page_exclusivity() {
        // header, two blobs, manifest, footer — a typical 4 KiB-aligned file
        let s = store_with(&[(0, 8), (4096, 8), (8192, 100), (12288, 340), (12628, 40)]);
        let page = page_size();
        if page <= 4096 {
            assert!(s.page_exclusive(4096, 8));
            assert!(s.page_exclusive(8192, 100));
        }
        // 64 KiB-aligned blobs are exclusive on every page size in use
        let canonical = store_with(&[(0, 8), (65536, 8), (131072, 100), (196608, 380)]);
        assert!(canonical.page_exclusive(65536, 8));
        assert!(canonical.page_exclusive(131072, 100));
        // two ranges inside one page: neither is exclusive
        let packed = store_with(&[(4096, 100), (4200, 50)]);
        assert!(!packed.page_exclusive(4096, 100));
        assert!(!packed.page_exclusive(4200, 50));
        // zero-length is vacuous; an unlisted range is not exclusive
        assert!(s.page_exclusive(4096, 0));
        assert!(!s.page_exclusive(20480, 8));
        // and a store that never declared its occupancy claims nothing
        assert!(!store_with(&[]).page_exclusive(65536, 8));
    }
}
