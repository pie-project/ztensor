#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use ztensor::{Hdf5Reader, TensorReader};

fuzz_target!(|data: &[u8]| {
    let mut tmp = tempfile::Builder::new().suffix(".h5").tempfile().unwrap();
    tmp.write_all(data).unwrap();

    let reader = match Hdf5Reader::open(tmp.path()) {
        Ok(r) => r,
        Err(_) => return,
    };

    for name in reader.keys() {
        let _ = reader.read_data(name);
        let _ = TensorReader::read::<f32>(&reader, name);
    }
});
