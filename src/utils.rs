//! Utility functions for zTensor operations.

use crate::models::ALIGNMENT;

/// Aligns an offset up to the given alignment boundary.
/// Returns (aligned_offset, padding_bytes).
#[inline]
pub fn align_offset_to(current_offset: u64, alignment: u64) -> (u64, u64) {
    if alignment == 0 {
        return (current_offset, 0);
    }
    let remainder = current_offset % alignment;
    if remainder == 0 {
        (current_offset, 0)
    } else {
        let padding = alignment - remainder;
        (current_offset + padding, padding)
    }
}

/// Calculates the aligned offset and required padding for 64-byte alignment.
/// Returns (aligned_offset, padding_bytes).
#[inline]
pub fn align_offset(current_offset: u64) -> (u64, u64) {
    align_offset_to(current_offset, ALIGNMENT)
}

/// Returns true if the host system is little-endian.
#[inline]
pub const fn is_little_endian() -> bool {
    cfg!(target_endian = "little")
}

/// Swaps byte order of multi-byte elements in place.
pub fn swap_endianness_in_place(buffer: &mut [u8], element_size: usize) {
    if element_size <= 1 {
        return;
    }
    for chunk in buffer.chunks_exact_mut(element_size) {
        chunk.reverse();
    }
}

/// Computes SHA256 hash and returns hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(data);
    hex::encode(hash)
}

/// A writer that updates a checksum digest and counts bytes as it writes.
pub struct DigestWriter<W: std::io::Write> {
    inner: W,
    crc32: Option<crc32c::Crc32cHasher>,
    sha256: Option<sha2::Sha256>,
    pub bytes_written: u64,
}

impl<W: std::io::Write> DigestWriter<W> {
    pub fn new(inner: W, algorithm: crate::models::Checksum) -> Self {
        use crate::models::Checksum;
        let (crc32, sha256) = match algorithm {
            Checksum::None => (None, None),
            Checksum::Crc32c => (Some(crc32c::Crc32cHasher::default()), None),
            Checksum::Sha256 => (None, Some(sha2::Sha256::default())),
        };
        Self {
            inner,
            crc32,
            sha256,
            bytes_written: 0,
        }
    }

    pub fn finalize(self) -> Option<String> {
        use sha2::Digest;
        if let Some(hasher) = self.crc32 {
            use std::hash::Hasher;
            Some(format!("crc32c:0x{:08X}", hasher.finish() as u32))
        } else if let Some(hasher) = self.sha256 {
            let result = hasher.finalize();
            Some(format!("sha256:{}", hex::encode(result)))
        } else {
            None
        }
    }
}

impl<W: std::io::Write> std::io::Write for DigestWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes_written += n as u64;
        let slice = &buf[..n];

        if let Some(h) = &mut self.crc32 {
            use std::hash::Hasher;
            h.write(slice);
        }
        if let Some(h) = &mut self.sha256 {
            use sha2::Digest;
            h.update(slice);
        }

        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_offset_to() {
        assert_eq!(align_offset_to(0, 32), (0, 0));
        assert_eq!(align_offset_to(1, 32), (32, 31));
        assert_eq!(align_offset_to(31, 32), (32, 1));
        assert_eq!(align_offset_to(32, 32), (32, 0));
        assert_eq!(align_offset_to(33, 32), (64, 31));
        assert_eq!(align_offset_to(64, 32), (64, 0));
        assert_eq!(align_offset_to(0, 0), (0, 0));
    }

    #[test]
    fn test_align_offset_64() {
        assert_eq!(align_offset(0), (0, 0));
        assert_eq!(align_offset(1), (64, 63));
        assert_eq!(align_offset(64), (64, 0));
        assert_eq!(align_offset(65), (128, 63));
    }
}
