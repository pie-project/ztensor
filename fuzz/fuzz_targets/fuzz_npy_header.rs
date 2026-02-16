#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = ztensor::npz_reader::parse_npy_header(data);
});
