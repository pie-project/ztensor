#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use ztensor::{PyTorchReader, TensorReader};

fuzz_target!(|data: &[u8]| {
    let reader = match PyTorchReader::from_reader(Cursor::new(data)) {
        Ok(r) => r,
        Err(_) => return,
    };

    for name in reader.keys() {
        let _ = reader.read_data(name);
        let _ = reader.read_as::<f32>(name);
    }
});
