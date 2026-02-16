use half::{bf16, f16};

pub fn make_f64_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 * 0.1).collect()
}
pub fn make_f32_data(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.1).collect()
}
pub fn make_f16_data(n: usize) -> Vec<f16> {
    (0..n).map(|i| f16::from_f32(i as f32 * 0.1)).collect()
}
pub fn make_bf16_data(n: usize) -> Vec<bf16> {
    (0..n).map(|i| bf16::from_f32(i as f32 * 0.1)).collect()
}
pub fn make_i64_data(n: usize) -> Vec<i64> {
    (0..n).map(|i| i as i64 - (n / 2) as i64).collect()
}
pub fn make_i32_data(n: usize) -> Vec<i32> {
    (0..n).map(|i| i as i32 - (n / 2) as i32).collect()
}
pub fn make_i16_data(n: usize) -> Vec<i16> {
    (0..n).map(|i| (i as i16).wrapping_mul(7)).collect()
}
pub fn make_i8_data(n: usize) -> Vec<i8> {
    (0..n).map(|i| (i % 128) as i8).collect()
}
pub fn make_u64_data(n: usize) -> Vec<u64> {
    (0..n).map(|i| i as u64 * 3).collect()
}
pub fn make_u32_data(n: usize) -> Vec<u32> {
    (0..n).map(|i| i as u32 * 5).collect()
}
pub fn make_u16_data(n: usize) -> Vec<u16> {
    (0..n).map(|i| i as u16 * 7).collect()
}
pub fn make_u8_data(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 256) as u8).collect()
}
pub fn make_bool_data(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 2) as u8).collect()
}
