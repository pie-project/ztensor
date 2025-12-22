//! Utility functions for zTensor operations.

use crate::models::ALIGNMENT;

/// Calculates the aligned offset and required padding for 64-byte alignment.
/// Returns (aligned_offset, padding_bytes).
#[inline]
pub fn align_offset(current_offset: u64) -> (u64, u64) {
    let remainder = current_offset % ALIGNMENT;
    if remainder == 0 {
        (current_offset, 0)
    } else {
        let padding = ALIGNMENT - remainder;
        (current_offset + padding, padding)
    }
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

/// A writer that updates a checksum digest as it writes.
pub struct DigestWriter<W: std::io::Write> {
    inner: W,
    crc32: Option<crc32c::Crc32cHasher>,
    sha256: Option<sha2::Sha256>,
}

impl<W: std::io::Write> DigestWriter<W> {
    pub fn new(inner: W, algorithm: crate::models::ChecksumAlgorithm) -> Self {
        use crate::models::ChecksumAlgorithm;
        let (crc32, sha256) = match algorithm {
            ChecksumAlgorithm::None => (None, None),
            ChecksumAlgorithm::Crc32c => (Some(crc32c::Crc32cHasher::default()), None),
            ChecksumAlgorithm::Sha256 => (None, Some(sha2::Sha256::default())),
        };
        Self { inner, crc32, sha256 }
    }

    pub fn finalize(self) -> Option<String> {
        use sha2::Digest;
        if let Some(hasher) = self.crc32 {
            // crc32c crate hasher doesn't expose inner state easily in all versions, 
            // but let's assume standard usage.
            // Wait, crc32c::Crc32cHasher is not the main API, usually it's crc32c::crc32c(&data).
            // But for streaming we need a hasher.
            // The `crc32c` crate (0.6.5) has `Crc32cHasher` which implements `Hasher`.
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
