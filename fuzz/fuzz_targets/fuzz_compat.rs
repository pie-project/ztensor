//! Every foreign-format parser, on arbitrary bytes: open must never panic.
//!
//! The first byte selects the projection so one corpus exercises all of
//! them; the rest is the file. Opening writes nothing, so a temp file per
//! iteration is the only way to reach the mmap-based readers.

#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    let Some((&selector, body)) = data.split_first() else {
        return;
    };
    let mut file = match tempfile::NamedTempFile::new() {
        Ok(f) => f,
        Err(_) => return,
    };
    if file.write_all(body).is_err() || file.flush().is_err() {
        return;
    }
    let path = file.path();

    match selector % 7 {
        0 => drop(ztensor_compat::Safetensors::open(path)),
        1 => drop(ztensor_compat::Gguf::open(path)),
        2 => drop(ztensor_compat::Npz::open(path)),
        3 => drop(ztensor_compat::Pt::open(path)),
        4 => drop(ztensor_compat::Hdf5::open(path)),
        5 => drop(ztensor_compat::Onnx::open(path)),
        _ => {
            // Detection plus a full read of everything it finds.
            if let Ok(src) = ztensor_compat::open_any(path) {
                use ztensor::Source;
                let names: Vec<(String, Vec<String>)> = src
                    .manifest()
                    .objects
                    .iter()
                    .map(|(n, o)| (n.clone(), o.parts.keys().cloned().collect()))
                    .collect();
                for (name, parts) in names {
                    for part in parts {
                        let _ = src.read(&name, &part);
                        let _ = src.view(&name, &part);
                        let _ = src.caps(&name, &part);
                    }
                }
            }
        }
    }
});
