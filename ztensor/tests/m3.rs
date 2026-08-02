//! M3 exit criteria: capability ladder surface for pie — Source trait,
//! caps(), page-exclusivity, eviction round-trip.

use std::path::PathBuf;

use ztensor::{page_size, DType, Error, Reader, Source, Writer, ALIGN_CANONICAL};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn canonical_file(name: &str) -> PathBuf {
    let path = tmp(name);
    let mut w = Writer::create(&path).unwrap();
    w.add_dense("a.weight", &[64, 64], DType::F32, &vec![3u8; 16384])
        .unwrap();
    w.add_dense("b.weight", &[128], DType::BF16, &vec![4u8; 256])
        .unwrap();
    w.finish().unwrap();
    path
}

#[test]
fn canonical_reaches_tier3() {
    let r = Reader::open(canonical_file("caps.zt")).unwrap();
    for name in ["a.weight", "b.weight"] {
        let caps = r.caps(name, "data").unwrap();
        assert!(caps.zero_copy);
        assert!(caps.verifiable);
        assert!(caps.alignment >= ALIGN_CANONICAL, "{caps:?}");
        // On any page size up to 64 KiB, canonical placement is exclusive.
        if page_size() <= ALIGN_CANONICAL {
            assert!(caps.page_exclusive, "{caps:?}");
            assert_eq!(caps.tier(), 3);
        }
    }
}

#[test]
fn floor_alignment_still_tier3_on_small_pages() {
    let path = tmp("floor.zt");
    let mut w = Writer::create_with_alignment(&path, 4096).unwrap();
    w.add_dense("t", &[16], DType::F32, &[7u8; 64]).unwrap();
    w.finish().unwrap();
    let r = Reader::open(&path).unwrap();
    let caps = r.caps("t", "data").unwrap();
    assert!(caps.zero_copy && caps.verifiable);
    assert_eq!(caps.alignment, 4096);
    if page_size() == 4096 {
        assert_eq!(caps.tier(), 3);
    }
}

#[test]
fn source_trait_object() {
    let r = Reader::open(canonical_file("dyn.zt")).unwrap();
    let src: &dyn Source = &r;
    assert_eq!(src.manifest().objects.len(), 2);
    let bytes = src.read("b.weight", "data").unwrap();
    assert_eq!(bytes, vec![4u8; 256]);
    assert_eq!(src.view("b.weight", "data").unwrap(), &bytes[..]);
    assert!(src.caps("b.weight", "data").unwrap().tier() >= 2);
}

#[cfg(unix)]
#[test]
fn evict_and_reread() {
    let r = Reader::open(canonical_file("evict.zt")).unwrap();
    let before = r.read("a.weight", "data").unwrap();
    if r.caps("a.weight", "data").unwrap().page_exclusive {
        r.prefetch("a.weight", "data").unwrap();
        r.evict("a.weight", "data").unwrap();
        // Evicted pages re-fault from the file: content is unchanged.
        assert_eq!(r.read("a.weight", "data").unwrap(), before);
        assert!(r.verify("a.weight", "data").unwrap());
    }
}

#[test]
fn caps_not_found() {
    let r = Reader::open(canonical_file("nf.zt")).unwrap();
    assert!(matches!(r.caps("nope", "data"), Err(Error::NotFound(_))));
}
