//! Several self-describing files read as one.
//!
//! This is the shape every foreign format arrives in — a sharded safetensors
//! snapshot is N complete files with an index beside them — and it is not the
//! shape the spec's multi-file model has. The tests here are about that
//! difference: a composite gives one name space over N sources and remembers
//! which file each name came from, and it claims nothing about the set.

use std::path::PathBuf;

use ztensor::{Composite, CompositeSource, DType, Reader, Writer};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// Writes a one-object file and returns it as a composite member.
fn file(name: &str, object: &str, bytes: &[u8]) -> CompositeSource {
    let path = tmp(name);
    let mut w = Writer::create(&path).unwrap();
    w.add_dense(object, &[bytes.len() as u64], DType::U8, bytes)
        .unwrap();
    w.finish().unwrap();
    CompositeSource {
        label: path.display().to_string(),
        source: Box::new(Reader::open(&path).unwrap()),
    }
}

#[test]
fn objects_are_one_name_space_that_remembers_its_files() {
    let a = vec![1u8; 64];
    let b = vec![2u8; 32];
    let composite = Composite::new(vec![
        file("composite-second.zt", "layer.1.weight", &b),
        file("composite-first.zt", "layer.0.weight", &a),
    ])
    .unwrap();

    // Sorted across the whole composite, so the order does not depend on how
    // the caller listed the files -- here, deliberately out of order.
    let listed: Vec<(usize, String)> = composite
        .objects()
        .map(|(index, name, _)| (index, name.to_string()))
        .collect();
    assert_eq!(
        listed,
        vec![
            (1, "layer.0.weight".to_string()),
            (0, "layer.1.weight".to_string()),
        ]
    );

    assert_eq!(composite.source_of("layer.0.weight"), Some(1));
    assert_eq!(composite.source_of("layer.1.weight"), Some(0));
    assert_eq!(composite.source_of("absent"), None);

    assert_eq!(composite.read("layer.0.weight", "data").unwrap(), a);
    assert_eq!(composite.view("layer.1.weight", "data").unwrap(), b);
}

/// A part's blob offset stays relative to the file that holds it. Together
/// with the source index that is a complete address, and it is the pair a
/// consumer planning reads off disk needs.
#[test]
fn offsets_stay_relative_to_their_own_file() {
    let composite = Composite::new(vec![
        file("composite-off-a.zt", "w.a", &[7u8; 100]),
        file("composite-off-b.zt", "w.b", &[8u8; 100]),
    ])
    .unwrap();

    for (index, name, object) in composite.objects() {
        let part = &object.parts["data"];
        assert_eq!(
            part.blob.shard, 0,
            "{name} should address its own file, not a shard table"
        );
        assert!(part.blob.offset >= 64 * 1024, "{name} is not page-placed");
        // Both files were written the same way, so both payloads land at the
        // same offset -- which is only coherent because the offset belongs to
        // the file, not to the composite.
        assert_eq!(composite.source_of(name), Some(index));
    }
    let offsets: Vec<u64> = composite
        .objects()
        .map(|(_, _, object)| object.parts["data"].blob.offset)
        .collect();
    assert_eq!(offsets[0], offsets[1]);
}

/// A name in two files is a broken set, and there is no rule that would make
/// it whole: picking a winner would load half a model and say nothing.
#[test]
fn a_name_in_two_files_is_refused() {
    let err = Composite::new(vec![
        file("composite-dup-a.zt", "shared", &[1u8; 8]),
        file("composite-dup-b.zt", "shared", &[2u8; 8]),
    ])
    .unwrap_err();
    let message = format!("{err}");
    assert!(message.contains("shared"), "unhelpful message: {message}");
    assert!(
        message.contains("composite-dup-a") && message.contains("composite-dup-b"),
        "the message must name both files: {message}"
    );
}

/// Reading through a composite is reading through the file, with nothing
/// added and nothing lost: the capability report is the holder's own.
#[test]
fn capabilities_are_the_holding_file_s() {
    let path = tmp("composite-caps.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add_dense("w", &[128], DType::U8, &[3u8; 128]).unwrap();
    w.finish().unwrap();

    let alone = Reader::open(&path).unwrap();
    let direct = alone.caps("w", "data").unwrap();

    let composite = Composite::new(vec![CompositeSource {
        label: path.display().to_string(),
        source: Box::new(Reader::open(&path).unwrap()),
    }])
    .unwrap();
    assert_eq!(composite.caps("w", "data").unwrap(), direct);
    assert_eq!(direct.tier(), 3, "a canonical file should reach tier 3");
}

#[test]
fn an_absent_object_is_not_found() {
    let composite = Composite::new(vec![file("composite-missing.zt", "w", &[1u8; 8])]).unwrap();
    assert!(composite.get("w").is_some());
    assert!(composite.get("nope").is_none());
    assert!(composite.read("nope", "data").is_err());
}
