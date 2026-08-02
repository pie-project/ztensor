//! Every foreign-format parser, on arbitrary bytes: opening must never panic.
//!
//! One corpus reaches all of them because detection is by magic, and the
//! projections are no longer separate types to call — so this drives the same
//! door every caller uses, then reads everything behind it. Opening writes
//! nothing, so a temp file per iteration is the only way in.

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

    // Half the iterations map the file and half only index it: the two take
    // different paths through every projection, and only one of them can hand
    // back a borrow.
    let opened = if selector % 2 == 0 {
        ztensor_compat::open(path)
    } else {
        ztensor_compat::index(path)
    };
    let Ok(src) = opened else {
        return;
    };
    let names: Vec<(String, Vec<String>)> = src
        .tensors()
        .map(|t| {
            (
                t.name().to_string(),
                t.parts().map(str::to_string).collect(),
            )
        })
        .collect();
    for (name, parts) in names {
        let Ok(tensor) = src.tensor(&name) else {
            continue;
        };
        for part in parts {
            let Ok(part) = tensor.part(&part) else {
                continue;
            };
            let _ = part.bytes();
            let _ = part.map();
            let _ = part.locate();
            let _ = part.caps();
            let _ = part.verify();
        }
    }
});
