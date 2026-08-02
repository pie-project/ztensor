//! M5 exit criteria: sharding mechanism and the overlay story — a LoRA
//! root referencing the base model's blobs directly.

use std::fs;
use std::path::PathBuf;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::{
    DType, Error, Model, Reader, Rule, ShardResolver, DataShardWriter, Writer,
};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

fn file_identity(path: &PathBuf) -> (u64, String) {
    let bytes = fs::read(path).unwrap();
    (bytes.len() as u64, format!("xxh3:{:016x}", xxh3_64(&bytes)))
}

/// Base model + LoRA overlay: the LoRA file stores only its deltas and
/// references the base's blobs through the shard table.
#[test]
fn lora_overlay() {
    let base_path = tmp("overlay-base.zt");
    let base_data: Vec<u8> = (0..1024u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut w = Writer::create(&base_path).unwrap();
    w.add_dense("base.weight", &[32, 32], DType::F32, &base_data)
        .unwrap();
    w.finish().unwrap();
    let (base_size, base_digest) = file_identity(&base_path);

    let lora_path = tmp("overlay-lora.zt");
    let delta = vec![7u8; 256];
    let base_reader = Reader::open(&base_path).unwrap();
    let mut w = Writer::create_with_alignment(&lora_path, 4096).unwrap();
    let shard = w.add_shard(base_size, &base_digest).unwrap();
    assert_eq!(shard, 1);
    w.link_object("base.weight", base_reader.get("base.weight").unwrap(), shard)
        .unwrap();
    w.add_dense("base.weight.lora_a", &[64], DType::F32, &delta)
        .unwrap();
    w.finish().unwrap();

    // Resolve the base by identity via a closure resolver.
    let resolver = |_idx: u64, _shard: &ztensor::Shard| Ok(base_path.clone());
    let model = Model::open_with(&lora_path, &resolver).unwrap();

    // Cross-file zero-copy: the base tensor's bytes come from base.zt.
    assert_eq!(model.view("base.weight", "data").unwrap(), &base_data[..]);
    assert_eq!(model.read("base.weight.lora_a", "data").unwrap(), delta);

    let caps = model.caps("base.weight", "data").unwrap();
    assert!(caps.zero_copy && caps.verifiable);
    assert!(!caps.page_exclusive, "exclusivity is unprovable across files");
    assert_eq!(caps.tier(), 2);

    // Digest carried over by link_object verifies against base bytes.
    assert!(model.verify("base.weight", "data").unwrap());
    model.verify_shards().unwrap();
}

#[test]
fn positional_shards() {
    // Data shard written by DataShardWriter, root by Writer.
    let shard_path = tmp("posmodel-00001.zt");
    let payload = vec![9u8; 8192];
    let mut ds = DataShardWriter::create_with_alignment(&shard_path, 4096).unwrap();
    let offset = ds.add_blob(&payload).unwrap();
    let (size, digest) = ds.finish().unwrap();

    // The data shard alone is a valid manifest-less file.
    assert!(Reader::open(&shard_path).unwrap().is_data_shard());

    let root_path = tmp("posmodel.zt");
    let mut w = Writer::create_with_alignment(&root_path, 4096).unwrap();
    let idx = w.add_shard(size, &digest).unwrap();
    let part = ztensor::Part {
        dtype: DType::U8,
        ltype: None,
        blob: ztensor::BlobRef {
            shard: idx,
            offset,
            length: payload.len() as u64,
        },
        encoding: None,
        decoded_length: None,
        digest: Some(format!("xxh3:{:016x}", xxh3_64(&payload))),
    };
    w.add_external_object("t", &[8192], "dense", &[("data", part)], None)
        .unwrap();
    w.finish().unwrap();

    // PositionalResolver: posmodel.zt -> posmodel-00001.zt
    let model = Model::open(&root_path).unwrap();
    assert_eq!(model.read("t", "data").unwrap(), payload);
    assert!(model.verify("t", "data").unwrap());
    model.verify_shards().unwrap();
}

#[test]
fn shard_size_mismatch_rejected() {
    let base_path = tmp("mismatch-base.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add_dense("t", &[4], DType::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let (size, digest) = file_identity(&base_path);

    let root_path = tmp("mismatch-root.zt");
    let mut w = Writer::create_with_alignment(&root_path, 4096).unwrap();
    let idx = w.add_shard(size + 4096, &digest).unwrap(); // wrong size
    w.link_object("t", Reader::open(&base_path).unwrap().get("t").unwrap(), idx)
        .unwrap();
    w.finish().unwrap();

    let resolver = |_: u64, _: &ztensor::Shard| Ok(base_path.clone());
    let err = Model::open_with(&root_path, &resolver).unwrap_err();
    assert!(
        matches!(err, Error::Reject { rule: Rule::ShardIdentity, .. }),
        "{err:?}"
    );
}

#[test]
fn shard_digest_mismatch_caught_by_deep_verify() {
    let base_path = tmp("digest-base.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add_dense("t", &[256], DType::U8, &[5u8; 256]).unwrap();
    w.finish().unwrap();
    let (size, digest) = file_identity(&base_path);

    // Corrupt one data byte: size and footer stay valid.
    let mut bytes = fs::read(&base_path).unwrap();
    bytes[65536] ^= 0xff;
    let corrupted = tmp("digest-base-corrupt.zt");
    fs::write(&corrupted, &bytes).unwrap();

    let root_path = tmp("digest-root.zt");
    let mut w = Writer::create_with_alignment(&root_path, 4096).unwrap();
    let idx = w.add_shard(size, &digest).unwrap();
    w.link_object("t", Reader::open(&base_path).unwrap().get("t").unwrap(), idx)
        .unwrap();
    w.finish().unwrap();

    let resolver = |_: u64, _: &ztensor::Shard| Ok(corrupted.clone());
    let model = Model::open_with(&root_path, &resolver).unwrap(); // cheap rungs pass
    let err = model.verify_shards().unwrap_err();
    assert!(
        matches!(err, Error::Reject { rule: Rule::ShardIdentity, .. }),
        "{err:?}"
    );
    // The per-part digest also catches it, independently.
    assert!(model.verify("t", "data").is_err());
}

#[test]
fn canonical_is_single_file() {
    let mut w = Writer::create(tmp("canon-shard.zt")).unwrap();
    let err = w.add_shard(4096, "xxh3:0011223344556677").unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)));
}

#[test]
fn single_file_model_is_the_degenerate_case() {
    let path = tmp("single.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add_dense("t", &[4], DType::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let model = Model::open(&path).unwrap(); // no shards: resolver never used
    assert_eq!(model.read("t", "data").unwrap(), vec![1, 2, 3, 4]);
    model.verify_shards().unwrap(); // vacuous
}

#[test]
fn resolver_trait_objects() {
    // CasResolver path shape (no file IO — just the mapping).
    let cas = ztensor::CasResolver {
        store: PathBuf::from("/store"),
    };
    let shard = ztensor::Shard {
        size: 4096,
        digest: "xxh3:00ff00ff00ff00ff".into(),
    };
    let path = cas.resolve(7, &shard).unwrap();
    assert_eq!(path, PathBuf::from("/store/blobs/xxh3/00ff00ff00ff00ff"));
}
