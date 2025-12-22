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

/// Converts a slice of u64 to bytes (zero-copy view).
#[inline]
pub fn u64_slice_to_bytes(v: &[u64]) -> &[u8] {
    bytemuck::cast_slice(v)
}

/// Converts a Vec<u64> to Vec<u8>.
#[inline]
pub fn u64_vec_to_bytes(v: Vec<u64>) -> Vec<u8> {
    bytemuck::cast_slice(&v).to_vec()
}

/// Computes SHA256 hash and returns hex string.
pub fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(data);
    hex::encode(hash)
}
