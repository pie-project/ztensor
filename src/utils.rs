use crate::models::ALIGNMENT;

pub fn calculate_padding(current_offset: u64) -> (u64, u64) {
    let remainder = current_offset % ALIGNMENT;
    if remainder == 0 {
        (current_offset, 0)
    } else {
        let padding = ALIGNMENT - remainder;
        (current_offset + padding, padding)
    }
}

#[cfg(target_endian = "little")]
pub const IS_NATIVE_LITTLE_ENDIAN: u8 = 1;
#[cfg(target_endian = "big")]
pub const IS_NATIVE_LITTLE_ENDIAN: u8 = 0;

/// Swaps byte order of a buffer of multi-byte elements in place.
/// Assumes all elements in the buffer are of the same type (element_size bytes).
pub fn swap_endianness_in_place(buffer: &mut [u8], element_size: usize) {
    if element_size <= 1 {
        return;
    }
    for chunk in buffer.chunks_mut(element_size) {
        if chunk.len() == element_size {
            // Ensure it's a full chunk
            chunk.reverse();
        }
    }
}
