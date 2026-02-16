#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use ztensor::{GgufReader, TensorReader};

fuzz_target!(|data: &[u8]| {
    let mut tmp = tempfile::Builder::new()
        .suffix(".gguf")
        .tempfile()
        .unwrap();
    tmp.write_all(data).unwrap();

    let reader = match GgufReader::open(tmp.path()) {
        Ok(r) => r,
        Err(_) => return,
    };

    for name in reader.keys() {
        let _ = reader.read_data(name);
        let _ = reader.read_as::<f32>(name);
    }
});
