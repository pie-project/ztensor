#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Cursor;
use ztensor::ZTensorReader; // Assuming crate name is ztensor

fuzz_target!(|data: &[u8]| {
    // 1. Wrap the random fuzz bytes in a Cursor (simulating a file)
    let reader = Cursor::new(data);

    // 2. Try to open the zTensor file
    // We expect this to fail 99% of the time. We only care if it CRASHES.
    let mut ztensor = match ZTensorReader::new(reader) {
        Ok(z) => z,
        Err(_) => return, // Gracefully handle invalid files (this is correct behavior)
    };

    // 3. If we successfully opened it, try to interact with the manifest
    // This tests the data structures and allocation logic.
    let object_names: Vec<String> = ztensor.list_objects().keys().cloned().collect();

    for name in object_names {
        // Try to read the object. 
        // We don't care about the data, just that it doesn't SEGFAULT or OOM.
        let _ = ztensor.read_object(&name, false); 
        
        // Also try reading as a specific type to test alignment checks
        let _ = ztensor.read_object_as::<f32>(&name);
    }
});