#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use ztensor::Reader;

fuzz_target!(|data: &[u8]| {
    let reader = match Reader::new(Cursor::new(data)) {
        Ok(r) => r,
        Err(_) => return,
    };

    for name in reader.tensors().keys() {
        let _ = reader.read(name, true);
        let _ = reader.read_as::<f32>(name);
    }
});
