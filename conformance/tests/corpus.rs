//! Conformance runner: executes every case in `all_cases()` and checks the
//! exported corpus files stay in sync. Also runs a deterministic mutation
//! smoke pass (a stable-toolchain stand-in for the cargo-fuzz targets).

use std::fs;
use std::path::{Path, PathBuf};

use conformance::{all_cases, Expect, Op};
use ztensor::{csr, validate_bytes, Error, Source, Vocabulary};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

#[test]
fn run_all_cases() {
    for case in all_cases() {
        run(&case);
    }
}

fn run(case: &conformance::Case) {
    let name = case.name;
    match case.op {
        Op::Open => {
            let result = validate_bytes(&case.bytes, &Vocabulary::standard());
            match (&case.expect, result) {
                (Expect::Valid, Ok(Some(_))) => {}
                (Expect::DataShard, Ok(None)) => {}
                (Expect::Reject(rule), Err(Error::Reject { rule: got, detail })) => {
                    assert_eq!(got, *rule, "{name}: wrong rule ({detail})");
                }
                (expect, got) => panic!("{name}: expected {expect:?}, got {got:?}"),
            }
        }
        Op::View(..) | Op::Verify(..) | Op::ReadCsr(..) => {
            let path = tmp(&format!("case-{name}.zt"));
            fs::write(&path, &case.bytes).unwrap();
            let source = Source::open(&path)
                .unwrap_or_else(|e| panic!("{name}: this case must open cleanly, got {e}"));
            let result = match case.op {
                Op::View(obj, part) => source
                    .tensor(obj)
                    .and_then(|t| t.part(part))
                    .and_then(|p| p.map())
                    .map(|_| ()),
                Op::Verify(obj, part) => source
                    .tensor(obj)
                    .and_then(|t| t.part(part))
                    .and_then(|p| p.verify())
                    .map(|_| ()),
                Op::ReadCsr(obj) => source.tensor(obj).and_then(|t| csr::read(&t)).map(|_| ()),
                Op::Open => unreachable!(),
            };
            match (&case.expect, result) {
                (Expect::Valid, Ok(())) => {}
                (Expect::Unsupported, Err(Error::Unsupported(_))) => {}
                (Expect::Reject(rule), Err(Error::Reject { rule: got, detail })) => {
                    assert_eq!(got, *rule, "{name}: wrong rule ({detail})");
                }
                (expect, got) => panic!("{name}: expected {expect:?}, got {got:?}"),
            }
        }
    }
}

/// The exported golden files under `corpus/` must match `all_cases()`
/// byte for byte. Regenerate with `cargo run -p conformance --bin gen`.
#[test]
fn corpus_files_in_sync() {
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
    assert!(
        corpus.is_dir(),
        "corpus/ missing; run `cargo run -p conformance --bin gen`"
    );
    let mut expected: std::collections::BTreeSet<PathBuf> = Default::default();
    for case in all_cases() {
        let dir = match case.expect {
            Expect::Reject(_) => "reject",
            _ => "valid",
        };
        let path = corpus.join(dir).join(format!("{}.zt", case.name));
        let on_disk = fs::read(&path)
            .unwrap_or_else(|_| panic!("{} missing; regenerate corpus", path.display()));
        assert_eq!(on_disk, case.bytes, "{} out of sync", case.name);
        expected.insert(path);
    }

    // Nothing else may live here. Without this check a stale golden file
    // from a renamed case, or from a fuzzer that mistook this for its own
    // corpus directory, ships as if it were normative.
    for dir in ["valid", "reject"] {
        for entry in fs::read_dir(corpus.join(dir)).unwrap() {
            let path = entry.unwrap().path();
            assert!(
                expected.contains(&path),
                "{} is not a corpus case; regenerate the corpus",
                path.display()
            );
        }
    }
}

/// Deterministic mutation smoke: no input may panic the validator.
#[test]
fn mutation_smoke() {
    let bases: Vec<Vec<u8>> = all_cases()
        .into_iter()
        .filter(|c| matches!(c.expect, Expect::Valid))
        .map(|c| c.bytes)
        .collect();

    let mut state = 0x9e3779b97f4a7c15u64; // fixed seed: reproducible
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for round in 0..20_000u32 {
        let base = &bases[(rng() as usize) % bases.len()];
        let mut bytes = base.clone();
        match rng() % 3 {
            0 => {
                // flip a byte
                let at = (rng() as usize) % bytes.len();
                bytes[at] ^= (rng() as u8) | 1;
            }
            1 => {
                // truncate
                bytes.truncate((rng() as usize) % bytes.len());
            }
            _ => {
                // splice a slice from another case
                let other = &bases[(rng() as usize) % bases.len()];
                let at = (rng() as usize) % bytes.len();
                let from = (rng() as usize) % other.len();
                let n = ((rng() as usize) % 64)
                    .min(bytes.len() - at)
                    .min(other.len() - from);
                bytes[at..at + n].copy_from_slice(&other[from..from + n]);
            }
        }
        // Must return, never panic. Result content is irrelevant here.
        let _ = validate_bytes(&bytes, &Vocabulary::standard());
        let _ = round;
    }
}
