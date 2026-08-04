//! Multi-file models (spec §7): the shard table, resolution, and the overlay
//! story, where a LoRA root references the base model's blobs directly.
//!
//! This is the shape a `Source` gets by *verification*: the root states each
//! shard's size and digest, so opening one checks that the files on disk are
//! the files the manifest meant. Contrast `merge.rs`, where nothing binds the
//! set and nothing pretends to.

use std::fs;
use std::path::PathBuf;

use xxhash_rust::xxh3::xxh3_64;
use ztensor::{
    schema, shard_identity, BlobRef, DType, Error, Rule, Shard, ShardResolver, Source, Writer,
};

fn tmp(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}

/// Writes a data shard (spec §7.2): magic, one blob at `offset`, and a footer
/// whose manifest fields are all zero.
///
/// zTensor does not write these. It reads them, because other producers do, so
/// the tests build one the way such a producer would.
fn write_data_shard(path: &PathBuf, offset: u64, payload: &[u8]) {
    let end = offset as usize + payload.len();
    let mut bytes = vec![0u8; end];
    bytes[..8].copy_from_slice(&ztensor::MAGIC);
    bytes[offset as usize..end].copy_from_slice(payload);
    let mut footer = [0u8; 40];
    // offset, length and hash stay zero: that is what makes it a data shard.
    footer[24..28].copy_from_slice(&2u32.to_le_bytes());
    footer[32..40].copy_from_slice(&ztensor::MAGIC);
    bytes.extend_from_slice(&footer);
    fs::write(path, &bytes).unwrap();
}

/// Opens a root whose shards all live at one known path.
fn open_with_shard_at(root: &PathBuf, shard_path: PathBuf) -> ztensor::Result<Source> {
    Source::options()
        .resolver(move |_name: &str, _shard: &Shard| Ok(shard_path.clone()))
        .open(root)
}

/// Base model + LoRA overlay: the LoRA file stores only its deltas and
/// references the base's blobs through the shard table.
#[test]
fn lora_overlay() {
    let base_path = tmp("overlay-base.zt");
    let base_data: Vec<u8> = (0..1024u32)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
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
    w.add_shard("base", &base).unwrap();
    w.link("base.weight", &base_object, "base").unwrap();
    w.add("base.weight.lora_a", [64u64], DType::F32, &delta)
        .unwrap();
    w.finish().unwrap();

    let model = open_with_shard_at(&lora_path, base_path.clone()).unwrap();

    // Cross-file borrow: the base tensor's bytes come from base.zt.
    assert_eq!(
        model.tensor("base.weight").unwrap().map().unwrap(),
        &base_data[..]
    );
    assert_eq!(
        &*model.tensor("base.weight.lora_a").unwrap().bytes().unwrap(),
        &delta[..]
    );

    // The address names the file it came from, which is what a store id is
    // for: two tensors of one model, living in two different files.
    let base_at = model.tensor("base.weight").unwrap().locate().unwrap();
    let lora_at = model
        .tensor("base.weight.lora_a")
        .unwrap()
        .locate()
        .unwrap();
    assert_ne!(base_at.store, lora_at.store);
    assert_eq!(model.store(base_at.store).path(), base_path);
    assert_eq!(model.store(lora_at.store).path(), lora_path);

    // The base is itself a manifest-carrying container, so its occupancy is
    // known and page exclusivity is a fact rather than a guess, so a tensor in
    // another file is as evictable as one at home.
    let caps = model.tensor("base.weight").unwrap().caps().unwrap();
    assert!(caps.map && caps.locate && caps.verify);
    if ztensor::page_size() <= ztensor::ALIGN_CANONICAL {
        assert!(caps.evict, "{caps:?}");
    }

    // The digest carried over by link verifies against the base's bytes.
    assert!(model
        .tensor("base.weight")
        .unwrap()
        .verify()
        .unwrap()
        .checked());
    model.verify_shards().unwrap();
}

#[test]
fn positional_shards() {
    // A manifest-less shard, built here because zTensor no longer writes one:
    // this is the shape some *other* producer hands you, and reading it is
    // what has to keep working.
    let shard_path = tmp("posmodel-00001.zt");
    let payload = vec![9u8; 8192];
    let offset = 4096u64;
    write_data_shard(&shard_path, offset, &payload);
    let identity = shard_identity(&shard_path).unwrap();

    // The data shard alone is a valid manifest-less file.
    assert!(Source::open(&shard_path).unwrap().is_data_shard());
    assert!(Source::open(&shard_path).unwrap().is_empty());

    let root_path = tmp("posmodel.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    w.add_shard("00001", &identity).unwrap();
    let part = schema::Part {
        dtype: DType::U8,
        logical: None,
        blob: BlobRef {
            shard: Some("00001".into()),
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
    assert!(
        !caps.evict,
        "a manifest-less shard cannot prove exclusivity"
    );
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
    w.add_shard("base", &identity).unwrap();
    w.link("t", &base_object, "base").unwrap();
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
    w.add_shard("base", &identity).unwrap();
    w.link("t", &base_object, "base").unwrap();
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
        .add_shard(
            "base",
            &Shard {
                size: 4096,
                digest: "xxh3:0011223344556677".into(),
            },
        )
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

/// A name is a label, so the resolver is free to ignore it and match on
/// identity instead, which is the only thing that survives a rename.
#[test]
fn shards_found_by_identity_after_a_rename() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("byid");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let base_path = dir.join("original-name.zt");
    let mut w = Writer::create(&base_path).unwrap();
    w.add("t", [4u64], DType::U8, &[1, 2, 3, 4]).unwrap();
    w.finish().unwrap();
    let identity = shard_identity(&base_path).unwrap();
    let base_object = Source::open(&base_path)
        .unwrap()
        .manifest()
        .unwrap()
        .object("t")
        .unwrap()
        .clone();

    let root_path = dir.join("root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    w.add_shard("weights", &identity).unwrap();
    w.link("t", &base_object, "weights").unwrap();
    w.finish().unwrap();

    // Rename the shard: no convention that consults a name can find it now.
    let renamed = dir.join("something-else-entirely.zt");
    fs::rename(&base_path, &renamed).unwrap();
    assert!(Source::open(&root_path).is_err(), "positional must miss it");

    let model = Source::options()
        .resolver(ztensor::DirectoryResolver::scan(&dir).unwrap())
        .open(&root_path)
        .unwrap();
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &[1, 2, 3, 4]);
    assert_eq!(
        model
            .store(model.tensor("t").unwrap().locate().unwrap().store)
            .path(),
        renamed
    );
    model.verify_shards().unwrap();
}

/// The names a producer may choose are constrained so that a resolver can
/// spend one as a path component without sanitizing it first.
#[test]
fn a_shard_name_cannot_be_a_path() {
    let identity = Shard {
        size: 1 << 20,
        digest: "xxh3:0011223344556677".into(),
    };
    for name in [
        "../etc/passwd",
        "sub/dir",
        "",
        ".hidden",
        "a b",
        &"x".repeat(65),
    ] {
        let mut w = Writer::options()
            .canonical(false)
            .create(tmp("badname.zt"))
            .unwrap();
        // The writer reports reader rules as InvalidInput, carrying the
        // rule's own message. See `writer::invalid`.
        let err = w.add_shard(name, &identity).unwrap_err();
        assert!(
            matches!(err, Error::InvalidInput(_)),
            "{name:?} was accepted, or reported as {err:?}"
        );
        w.abandon();
    }
}

/// Registering the same name twice is fine if it means the same file, and an
/// error if it does not. Silently keeping one of two identities would make
/// the manifest a claim nobody checked.
#[test]
fn a_name_means_one_shard() {
    let identity = Shard {
        size: 1 << 20,
        digest: "xxh3:0011223344556677".into(),
    };
    let mut w = Writer::options()
        .canonical(false)
        .create(tmp("dupname.zt"))
        .unwrap();
    w.add_shard("base", &identity).unwrap();
    w.add_shard("base", &identity).unwrap();

    let other = Shard {
        size: 1 << 21,
        digest: "xxh3:8899aabbccddeeff".into(),
    };
    assert!(w.add_shard("base", &other).is_err());
    w.abandon();
}

#[test]
fn resolver_trait_objects() {
    // The CAS path shape (no file IO, just the mapping).
    let cas = ztensor::CasResolver {
        store: PathBuf::from("/store"),
    };
    let shard = Shard {
        size: 4096,
        digest: "xxh3:00ff00ff00ff00ff".into(),
    };
    let path = cas.resolve("anything", &shard).unwrap();
    assert_eq!(path, PathBuf::from("/store/blobs/xxh3/00ff00ff00ff00ff"));
}

/// A sha256 shard identity, produced and then checked.
///
/// §6.5 makes this the basis of signing: a root whose shard digests are
/// cryptographic commits to every shard byte, so one signature over the root
/// covers the model. That only holds if the digest can be *verified*, which is
/// why generating one without being able to check it would be worse than not
/// generating it at all.
///
/// The shard is an ordinary `.zt`, which is how shards are made: it keeps the
/// occupancy and the per-part digests that a manifest-less one throws away.
#[test]
fn a_sha256_shard_identity_round_trips() {
    use ztensor::DigestAlgorithm;

    let shard_path = tmp("sha-shard.zt");
    let payload = vec![3u8; 4096];
    let mut w = Writer::create(&shard_path).unwrap();
    w.add("borrowed", [4096u64], DType::U8, &payload).unwrap();
    w.finish().unwrap();
    let offset = Source::open(&shard_path)
        .unwrap()
        .tensor("borrowed")
        .unwrap()
        .locate()
        .unwrap()
        .offset;
    let from_writer = ztensor::shard_identity_with(&shard_path, DigestAlgorithm::Sha256).unwrap();

    assert!(
        from_writer.digest.starts_with("sha256:"),
        "{}",
        from_writer.digest
    );
    assert_eq!(from_writer.digest.len(), "sha256:".len() + 64);

    // Asking twice gives the same answer.
    let scanned = ztensor::shard_identity_with(&shard_path, DigestAlgorithm::Sha256).unwrap();
    assert_eq!(scanned, from_writer);
    // And the default is still xxh3, over the same bytes.
    let default = ztensor::shard_identity(&shard_path).unwrap();
    assert!(default.digest.starts_with("xxh3:"));
    assert_eq!(default.size, from_writer.size);

    // A root that records it, and a deep verify that checks it.
    let root_path = tmp("sha-root.zt");
    let mut w = Writer::options()
        .canonical(false)
        .align(4096)
        .create(&root_path)
        .unwrap();
    w.add_shard("data", &from_writer).unwrap();
    w.object("t")
        .shape([4096u64])
        .part("data")
        .dtype(DType::U8)
        .external(schema::Part {
            dtype: DType::U8,
            logical: None,
            blob: BlobRef {
                shard: Some("data".into()),
                offset,
                length: payload.len() as u64,
            },
            encoding: None,
            decoded_length: None,
            digest: None,
        })
        .add()
        .unwrap();
    w.finish().unwrap();

    let model = open_with_shard_at(&root_path, shard_path.clone()).unwrap();
    assert_eq!(&*model.tensor("t").unwrap().bytes().unwrap(), &payload[..]);
    model.verify_shards().unwrap();

    // A corrupted shard is caught by the sha256 path, not waved through.
    let mut bytes = fs::read(&shard_path).unwrap();
    bytes[4096] ^= 0xff;
    let corrupted = tmp("sha-shard-corrupt.zt");
    fs::write(&corrupted, &bytes).unwrap();
    let model = open_with_shard_at(&root_path, corrupted).unwrap();
    let err = model.verify_shards().unwrap_err();
    assert_eq!(err.rule(), Some(Rule::ShardIdentity), "{err}");
}

/// A part digest in any algorithm this build knows is checked, not waved
/// through.
///
/// The writer always writes `xxh3` part digests, which is what canonical form
/// requires (§6.3 rule 4) and what corruption detection wants. Verification is
/// the other half: a file from elsewhere may use `sha256`, and reading it must
/// check that digest rather than report "unsupported" or, worse, skip it.
#[test]
fn a_sha256_part_digest_is_verified() {
    use xxhash_rust::xxh3::xxh3_64;
    use ztensor::cbor::{self, Value};
    use ztensor::DigestAlgorithm;

    let data = vec![0xabu8; 256];
    let text = |s: &str| Value::Text(s.to_string());

    // Builds a file whose one part claims `digest`, over filler bytes that the
    // real payload matches.
    let build = |name: &str, digest: String| -> PathBuf {
        let manifest = Value::Map(vec![(
            text("objects"),
            Value::Map(vec![(
                text("t"),
                Value::Map(vec![
                    (text("shape"), Value::Array(vec![Value::Uint(256)])),
                    (text("layout"), text("dense")),
                    (
                        text("parts"),
                        Value::Map(vec![(
                            text("data"),
                            Value::Map(vec![
                                (text("dtype"), text("u8")),
                                (
                                    text("blob"),
                                    Value::Array(vec![Value::Uint(4096), Value::Uint(256)]),
                                ),
                                (text("digest"), text(&digest)),
                            ]),
                        )]),
                    ),
                ]),
            )]),
        )]);
        let encoded = cbor::encode(&manifest).unwrap();
        let mut bytes = vec![0u8; 8192];
        bytes[..8].copy_from_slice(&ztensor::MAGIC);
        bytes[4096..4096 + data.len()].copy_from_slice(&data);
        bytes.extend_from_slice(&encoded);
        let mut footer = [0u8; 40];
        footer[0..8].copy_from_slice(&8192u64.to_le_bytes());
        footer[8..16].copy_from_slice(&(encoded.len() as u64).to_le_bytes());
        footer[16..24].copy_from_slice(&xxh3_64(&encoded).to_le_bytes());
        footer[24..28].copy_from_slice(&2u32.to_le_bytes());
        footer[32..40].copy_from_slice(&ztensor::MAGIC);
        bytes.extend_from_slice(&footer);
        let path = tmp(name);
        fs::write(&path, &bytes).unwrap();
        path
    };

    let good = build("sha-part-ok.zt", DigestAlgorithm::Sha256.digest(&data));
    assert!(
        Source::open(&good)
            .unwrap()
            .tensor("t")
            .unwrap()
            .verify()
            .unwrap()
            .checked(),
        "a sha256 part digest must be checked, not skipped"
    );

    // The same file, claiming the digest of different bytes.
    let bad = build(
        "sha-part-bad.zt",
        DigestAlgorithm::Sha256.digest(&[0u8; 256]),
    );
    let err = Source::open(&bad)
        .unwrap()
        .tensor("t")
        .unwrap()
        .verify()
        .unwrap_err();
    assert_eq!(err.rule(), Some(Rule::Digest), "{err}");
}

/// The digests are the ones the rest of the world computes.
///
/// A digest that is merely self-consistent would pass every round-trip test in
/// this file and still be useless: the point of `sha256` here is that a tool
/// which has never seen this crate can check a signature over it.
#[test]
fn the_digests_match_the_published_vectors() {
    use ztensor::DigestAlgorithm;

    assert_eq!(
        DigestAlgorithm::Sha256.digest(b""),
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        DigestAlgorithm::Sha256.digest(b"abc"),
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    // A whole-file digest is computed a chunk at a time, so that path has to
    // produce a real sha256 too, not just a self-consistent one.
    let long = vec![0x5au8; 1 << 20];
    let path = tmp("sha-chunked.zt");
    let mut w = Writer::create(&path).unwrap();
    w.add("t", [long.len() as u64], DType::U8, &long).unwrap();
    w.finish().unwrap();

    let streamed = ztensor::shard_identity_with(&path, DigestAlgorithm::Sha256).unwrap();
    let whole_file = fs::read(&path).unwrap();
    assert_eq!(
        streamed.digest,
        DigestAlgorithm::Sha256.digest(&whole_file),
        "the chunked digest must equal the one-shot digest of the same bytes"
    );
    assert_eq!(streamed.size, whole_file.len() as u64);
}

/// `link_shard` is the four-step assembly done in one call.
///
/// Doing three of the four by hand still writes a file; the trouble surfaces
/// in whoever reads it. This checks the shortcut agrees with the long way,
/// byte for byte in what it produces.
#[test]
fn link_shard_matches_the_manual_assembly() {
    use ztensor::DigestAlgorithm;

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let shard = dir.join("ls-shard.zt");
    let mut w = Writer::create(&shard).unwrap();
    w.add("a.weight", [4u64], DType::F32, &[1u8; 16]).unwrap();
    w.add("b.weight", [8u64], DType::U8, &[2u8; 8]).unwrap();
    w.finish().unwrap();

    // The long way.
    let manual = dir.join("ls-manual.zt");
    let mut w = Writer::options().canonical(false).create(&manual).unwrap();
    let id = ztensor::shard_identity_with(&shard, DigestAlgorithm::Sha256).unwrap();
    w.add_shard("s", &id).unwrap();
    let m = ztensor::manifest_of(&shard).unwrap().unwrap();
    for (name, object) in &m.objects {
        w.link(name, object, "s").unwrap();
    }
    w.finish().unwrap();

    // The one call.
    let easy = dir.join("ls-easy.zt");
    let mut w = Writer::options().canonical(false).create(&easy).unwrap();
    let linked = w.link_shard("s", &shard, DigestAlgorithm::Sha256).unwrap();
    w.finish().unwrap();

    assert_eq!(linked, vec!["a.weight", "b.weight"]);
    assert_eq!(
        fs::read(&manual).unwrap(),
        fs::read(&easy).unwrap(),
        "the shortcut must produce exactly what the long way does"
    );

    // And it reads back.
    let sp = shard.clone();
    let src = Source::options()
        .resolver(move |_: &str, _: &Shard| Ok(sp.clone()))
        .open(&easy)
        .unwrap();
    assert_eq!(
        src.names().collect::<Vec<_>>(),
        vec!["a.weight", "b.weight"]
    );
    assert_eq!(
        &*src.tensor("a.weight").unwrap().bytes().unwrap(),
        &[1u8; 16]
    );
    src.verify_shards().unwrap();
}

/// A manifest-less shard names no tensors, so there is nothing to link and
/// saying so beats linking nothing and looking like it worked.
#[test]
fn link_shard_refuses_a_manifest_less_shard() {
    let path = tmp("ls-datashard.zt");
    write_data_shard(&path, 4096, &[1u8; 64]);

    let root = tmp("ls-datashard-root.zt");
    let mut w = Writer::options().canonical(false).create(&root).unwrap();
    let err = w
        .link_shard("s", &path, ztensor::DigestAlgorithm::Xxh3)
        .unwrap_err();
    assert!(format!("{err}").contains("no manifest"), "{err}");
    w.abandon();
}
