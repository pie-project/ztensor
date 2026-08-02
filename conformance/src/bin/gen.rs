//! Exports the conformance corpus as golden files under `conformance/corpus/`.
//!
//! `valid/` holds files a conforming reader must accept (including data
//! shards and files that are only structurally readable); `reject/` holds
//! files it must reject — the expected rule for each lives in
//! `conformance::all_cases()`, which is the source of truth.

use std::fs;
use std::path::Path;

use conformance::{all_cases, Expect};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    let valid = root.join("valid");
    let reject = root.join("reject");
    fs::create_dir_all(&valid).unwrap();
    fs::create_dir_all(&reject).unwrap();

    let mut n_valid = 0;
    let mut n_reject = 0;
    for case in all_cases() {
        let (dir, n) = match case.expect {
            Expect::Reject(_) => (&reject, &mut n_reject),
            _ => (&valid, &mut n_valid),
        };
        fs::write(dir.join(format!("{}.zt", case.name)), &case.bytes).unwrap();
        *n += 1;
    }
    println!("wrote {n_valid} valid + {n_reject} reject cases to {}", root.display());
}
