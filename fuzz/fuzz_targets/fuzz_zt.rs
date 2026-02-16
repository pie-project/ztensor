#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use ztensor::Reader;

fuzz_target!(|data: &[u8]| {
    let cursor = Cursor::new(data);

    let reader = match Reader::new(cursor) {
        Ok(r) => r,
        Err(_) => return,
    };

    let names: Vec<String> = reader.tensors().keys().cloned().collect();

    for name in &names {
        let _ = reader.read(name, true);
        let _ = reader.read_as::<f32>(name);
    }
});
