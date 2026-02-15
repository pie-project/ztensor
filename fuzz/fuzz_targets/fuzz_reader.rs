#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use ztensor::Reader;

fuzz_target!(|data: &[u8]| {
    // 1. Wrap the random fuzz bytes in a Cursor (simulating a file)
    let reader = Cursor::new(data);

    // 2. Try to open the zTensor file
    // We expect this to fail 99% of the time. We only care if it CRASHES.
    let reader = match Reader::new(reader) {
        Ok(r) => r,
        Err(_) => return, // Gracefully handle invalid files (this is correct behavior)
    };

    // 3. If we successfully opened it, try to interact with the manifest
    // This tests the data structures and allocation logic.
    let names: Vec<String> = reader.tensors().keys().cloned().collect();

    for name in &names {
        // Try to read the object.
        // We don't care about the data, just that it doesn't SEGFAULT or OOM.
        let _ = reader.read(name, false);

        // Also try reading as a specific type to test alignment checks
        let _ = reader.read_as::<f32>(name);
    }
});
