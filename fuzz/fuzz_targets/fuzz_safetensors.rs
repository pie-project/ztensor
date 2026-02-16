#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use ztensor::SafeTensorsReader;

fuzz_target!(|data: &[u8]| {
    let mut tmp = tempfile::Builder::new()
        .suffix(".safetensors")
        .tempfile()
        .unwrap();
    tmp.write_all(data).unwrap();

    let reader = match SafeTensorsReader::open(tmp.path()) {
        Ok(r) => r,
        Err(_) => return,
    };

    let names: Vec<String> = reader.manifest.objects.keys().cloned().collect();

    for name in &names {
        let _ = reader.read(name);
        let _ = reader.read_as::<f32>(name);
    }
});
