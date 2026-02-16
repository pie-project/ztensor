#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;

const EXTENSIONS: &[&str] = &[".zt", ".safetensors", ".pt", ".gguf", ".npz", ".onnx", ".h5"];

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // First byte selects the file extension; remaining bytes are the content.
    let ext_idx = data[0] as usize % EXTENSIONS.len();
    let content = &data[1..];

    let mut tmp = tempfile::Builder::new()
        .suffix(EXTENSIONS[ext_idx])
        .tempfile()
        .unwrap();
    tmp.write_all(content).unwrap();

    let reader = match ztensor::open(tmp.path()) {
        Ok(r) => r,
        Err(_) => return,
    };

    for name in reader.keys() {
        let _ = reader.read_data(name);
    }
});
