//! Multi-file models (spec §7): the shard table, resolution, and the overlay
//! story — a LoRA root that references the base model's blobs directly.
//!
//! This is the shape a `Source` gets by *verification*: the root states each
//! shard's size and digest, so opening one checks that the files on disk are
//! the files the manifest meant. Contrast `merge.rs`, where nothing binds the
//! set and nothing pretends to.

use std::fs;
use std::path::PathBuf;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::{
    schema, shard_identity, BlobRef, DataShardWriter, DType, Error, Rule, Shard, ShardResolver,
    Source, Writer,
};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// Opens a root whose shards all live at one known path.
fn open_with_shard_at(root: &PathBuf, shard_path: PathBuf) -> ztensor::Result<Source> {
    Source::options()
        .resolver(move |_index: u64, _shard: &Shard| Ok(shard_path.clone()))
        .open(root)
}

/// Base model + LoRA overlay: the LoRA file stores only its deltas and
/// references the base's blobs through the shard table.
#[test]
fn lora_overlay() {
    let base_path = tmp("overlay-base.zt");
    let base_data: Vec<u8> = (0..1024u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut w = Writer::create(&base_path).unwrap();
    w.add("base.weight", [32u64, 32], DType::F32, &base_data)
        .unwrap();
    w.finish().unwrap();
    let base = shard_identity(&base_path).unwrap();

    let lora_path = tmp("overlay-lora.zt");
    let delta = vec![7u8; 256];
    let base_source = Source::open(&base_path).unwrap();
    let base_object = base_source
        .manifest()
        .unwrap()
        .object("base.weight")
        .unwrap()
        .clone();

    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&lora_path)
        .unwrap();
    let shard = w.add_shard(&base).unwrap();
    assert_eq!(shard, 1);
    w.link("base.weight", &base_object, shard).unwrap();
    w.add("base.weight.lora_a", [64u64], DType::F32, &delta)
        .unwrap();
    w.finish().unwrap();

    let model = open_with_shard_at(&lora_path, base_path.clone()).unwrap();

    // Cross-file borrow: the base tensor's bytes come from base.zt.
    assert_eq!(model.tensor("base.weight").unwrap().map().unwrap(), &base_data[..]);
    assert_eq!(
        &*model.tensor("base.weight.lora_a").unwrap().bytes().unwrap(),
        &delta[..]
    );

    // The address names the file it came from, which is the whole point of a
    // store id: two tensors of one model, two different files.
    let base_at = model.tensor("base.weight").unwrap().locate().unwrap();
    let lora_at = model.tensor("base.weight.lora_a").unwrap().locate().unwrap();
    assert_ne!(base_at.store, lora_at.store);
    assert_eq!(model.store(base_at.store).path(), base_path);
    assert_eq!(model.store(lora_at.store).path(), lora_path);

    // The base is itself a manifest-carrying container, so its occupancy is
    // known and page exclusivity is a fact rather than a guess — a tensor in
    // another file is as evictable as one at home.
    let caps = model.tensor("base.weight").unwrap().caps().unwrap();
    assert!(caps.map && caps.locate && caps.verify);
    if ztensor::page_size() <= ztensor::ALIGN_CANONICAL {
        assert!(caps.evict, "{caps:?}");
    }

    // The digest carried over by link verifies against the base's bytes.
    assert!(model.tensor("base.weight").unwrap().verify().unwrap().checked());
    model.verify_shards().unwrap();
}

#[test]
fn positional_shards() {
    // Data shard written by DataShardWriter, root by Writer.
    let shard_path = tmp("posmodel-00001.zt");
    let payload = vec![9u8; 8192];
    let mut ds = DataShardWriter::create_with_alignment(&shard_path, 4096).unwrap();
    let offset = ds.add_blob(&payload).unwrap();
    let identity = ds.finish().unwrap();

    // The data shard alone is a valid manifest-less file.
    assert!(Source::open(&shard_path).unwrap().is_data_shard());
    assert!(Source::open(&shard_path).unwrap().is_empty());

    let root_path = tmp("posmodel.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    let idx = w.add_shard(&identity).unwrap();
    let part = schema::Part {
        dtype: DType::U8,
        logical: None,
        blob: BlobRef {
            shard: idx,
            offset,
            length: payload.len() as u64,
        },
        encoding: None,
        decoded_length: None,
        digest: Some(format!("xxh3:{:016x}", xxh3_64(&payload))),
    };
    w.object("t")
        .shape([8192u64])
        .part("data")
        .dtype(DType::U8)
        .external(part)
        .add()
        .unwrap();
    w.finish().unwrap();

    // The positional convention: posmodel.zt -> posmodel-00001.zt
    let model = Source::open(&root_path).unwrap();
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &payload[..]);
    assert!(model.tensor("t").unwrap().verify().unwrap().checked());
    model.verify_shards().unwrap();

    // The contrast with `lora_overlay`: a data shard states no occupancy, so
    // nothing can prove this blob has its pages to itself, and eviction is
    // refused rather than assumed safe.
    let caps = model.tensor("t").unwrap().caps().unwrap();
    assert!(caps.map && caps.locate);
    assert!(!caps.evict, "a manifest-less shard cannot prove exclusivity");
}

#[test]
fn shard_size_mismatch_rejected() {
    let base_path = tmp("mismatch-base.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add("t", [4u64], DType::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let mut identity = shard_identity(&base_path).unwrap();
    let base_object = Source::open(&base_path)
        .unwrap()
        .manifest()
        .unwrap()
        .object("t")
        .unwrap()
        .clone();
    identity.size += 4096; // a claim the file on disk does not meet

    let root_path = tmp("mismatch-root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    let idx = w.add_shard(&identity).unwrap();
    w.link("t", &base_object, idx).unwrap();
    w.finish().unwrap();

    let err = open_with_shard_at(&root_path, base_path).unwrap_err();
    assert_eq!(err.rule(), Some(Rule::ShardIdentity), "{err}");
}

#[test]
fn shard_digest_mismatch_caught_by_deep_verify() {
    let base_path = tmp("digest-base.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add("t", [256u64], DType::U8, &[5u8; 256]).unwrap();
    w.finish().unwrap();
    let identity = shard_identity(&base_path).unwrap();
    let base_object = Source::open(&base_path)
        .unwrap()
        .manifest()
        .unwrap()
        .object("t")
        .unwrap()
        .clone();

    // Corrupt one data byte: size and footer stay valid.
    let mut bytes = fs::read(&base_path).unwrap();
    bytes[65536] ^= 0xff;
    let corrupted = tmp("digest-base-corrupt.zt");
    fs::write(&corrupted, &bytes).unwrap();

    let root_path = tmp("digest-root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    let idx = w.add_shard(&identity).unwrap();
    w.link("t", &base_object, idx).unwrap();
    w.finish().unwrap();

    // The cheap rungs pass: the size is right and the frame parses.
    let model = open_with_shard_at(&root_path, corrupted).unwrap();
    let err = model.verify_shards().unwrap_err();
    assert_eq!(err.rule(), Some(Rule::ShardIdentity), "{err}");
    // The per-part digest catches it independently.
    assert!(model.tensor("t").unwrap().verify().is_err());
}

#[test]
fn canonical_is_single_file() {
    let mut w = Writer::create(tmp("canon-shard.zt")).unwrap();
    let err = w
        .add_shard(&Shard {
            size: 4096,
            digest: "xxh3:0011223344556677".into(),
        })
        .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
}

#[test]
fn a_single_file_is_the_degenerate_case() {
    let path = tmp("single.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("t", [4u64], DType::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let model = Source::open(&path).unwrap(); // no shards: the resolver is never used
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &[1, 2, 3, 4]);
    model.verify_shards().unwrap(); // vacuous
}

#[test]
fn resolver_trait_objects() {
    // The CAS path shape (no file IO — just the mapping).
    let cas = ztensor::CasResolver {
        store: PathBuf::from("/store"),
    };
    let shard = Shard {
        size: 4096,
        digest: "xxh3:00ff00ff00ff00ff".into(),
    };
    let path = cas.resolve(7, &shard).unwrap();
    assert_eq!(path, PathBuf::from("/store/blobs/xxh3/00ff00ff00ff00ff"));
}
