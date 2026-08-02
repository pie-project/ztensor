//! Arbitrary bytes through the full §8/§3.6 validator: must never panic.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = ztensor::validate_bytes(data);
});
