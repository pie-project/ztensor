#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use ztensor::PyTorchReader;

fuzz_target!(|data: &[u8]| {
    let cursor = Cursor::new(data.to_vec());

    let reader = match PyTorchReader::from_reader(cursor) {
        Ok(r) => r,
        Err(_) => return,
    };

    let names: Vec<String> = reader.manifest.objects.keys().cloned().collect();

    for name in &names {
        let _ = reader.read(name);
        let _ = reader.read_as::<f32>(name);
    }
});
