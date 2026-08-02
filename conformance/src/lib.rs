//! The conformance corpus, defined in code.
//!
//! `all_cases()` is the single source of truth: the test runner executes it
//! directly, and `src/bin/gen.rs` exports the same bytes as golden files
//! under `corpus/` for third-party implementations. A sync test keeps the
//! exported files honest.
//!
//! Every case states the operation it exercises and the exact expected
//! outcome — for rejections, the specific spec rule.

use xxhash_rust::xxh3::xxh3_64;
use ztensor::cbor::{self, Value};
use ztensor::{DType, Rule, Writer, MAGIC};

#[derive(Debug, Clone, Copy)]
pub enum Op {
    /// Validate the file image (spec §8 + §3.6).
    Open,
    /// Open, then take a zero-copy view of `(object, part)`.
    View(&'static str, &'static str),
    /// Open, then verify the digest of `(object, part)`.
    Verify(&'static str, &'static str),
    /// Open, then assemble a `zt.sparse_csr/1` object (data-level rules).
    ReadCsr(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Expect {
    Valid,
    DataShard,
    Reject(Rule),
    Unsupported,
}

pub struct Case {
    pub name: &'static str,
    pub bytes: Vec<u8>,
    pub op: Op,
    pub expect: Expect,
}

impl Case {
    fn open(name: &'static str, bytes: Vec<u8>, expect: Expect) -> Self {
        Case {
            name,
            bytes,
            op: Op::Open,
            expect,
        }
    }
}

// =======================================================================
// Builders
// =======================================================================

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

fn vmap(entries: Vec<(&str, Value)>) -> Value {
    Value::Map(entries.into_iter().map(|(k, v)| (text(k), v)).collect())
}

fn uints(ns: &[u64]) -> Value {
    Value::Array(ns.iter().map(|&n| Value::Uint(n)).collect())
}

fn part(dtype: &str, blob: [u64; 3], extra: Vec<(&str, Value)>) -> Value {
    let mut fields = vec![("dtype", text(dtype)), ("blob", uints(&blob))];
    fields.extend(extra);
    vmap(fields)
}

fn object(shape: &[u64], layout: &str, parts: Vec<(&str, Value)>) -> Value {
    vmap(vec![
        ("shape", uints(shape)),
        ("layout", text(layout)),
        ("parts", vmap(parts)),
    ])
}

fn dense(dtype: &str, shape: &[u64], offset: u64, length: u64) -> Value {
    object(shape, "dense", vec![("data", part(dtype, [0, offset, length], vec![]))])
}

fn manifest(objects: Vec<(&str, Value)>) -> Value {
    vmap(vec![("objects", vmap(objects))])
}

/// Assembles a full file image: magic, a data region `[8, data_end)` filled
/// with `0xab`, the manifest blob at the next 4 KiB boundary, and a correct
/// footer. `manifest_bytes` may be arbitrarily hostile CBOR.
pub fn assemble_raw(data_end: u64, manifest_bytes: &[u8]) -> Vec<u8> {
    let m_off = data_end.max(8).div_ceil(4096) * 4096;
    let mut bytes = vec![0xabu8; m_off as usize];
    bytes[..8].copy_from_slice(&MAGIC);
    for b in bytes.iter_mut().take(m_off as usize).skip(data_end as usize) {
        *b = 0; // padding after the data region is zero
    }
    bytes.extend_from_slice(manifest_bytes);
    let mut footer = [0u8; 40];
    footer[0..8].copy_from_slice(&m_off.to_le_bytes());
    footer[8..16].copy_from_slice(&(manifest_bytes.len() as u64).to_le_bytes());
    footer[16..24].copy_from_slice(&xxh3_64(manifest_bytes).to_le_bytes());
    footer[24..28].copy_from_slice(&2u32.to_le_bytes());
    footer[32..40].copy_from_slice(&MAGIC);
    bytes.extend_from_slice(&footer);
    bytes
}

pub fn assemble(data_end: u64, m: &Value) -> Vec<u8> {
    assemble_raw(data_end, &cbor::encode(m).unwrap())
}

/// `assemble`, then overwrite data-region bytes at the given offsets —
/// for cases whose *data* (not metadata) must be hostile.
pub fn assemble_with_data(data_end: u64, m: &Value, writes: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let mut bytes = assemble(data_end, m);
    for (offset, data) in writes {
        bytes[*offset as usize..*offset as usize + data.len()].copy_from_slice(data);
    }
    bytes
}

fn le_u64s(vals: &[u64]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A metadata-valid `zt.sparse_csr/1` file: shape [2, 3], nnz 3, u64
/// indices at 4096, indptr at 8192, f32 values at 12288 — with the given
/// index data planted.
fn csr_file(indices: &[u64], indptr: &[u64]) -> Vec<u8> {
    let m = manifest(vec![(
        "m",
        object(
            &[2, 3],
            "zt.sparse_csr/1",
            vec![
                ("indices", part("u64", [0, 4096, 24], vec![])),
                ("indptr", part("u64", [0, 8192, 24], vec![])),
                ("values", part("f32", [0, 12288, 12], vec![])),
            ],
        ),
    )]);
    assemble_with_data(
        12300,
        &m,
        &[(4096, le_u64s(indices)), (8192, le_u64s(indptr))],
    )
}

/// The digest of the `0xab` filler that `assemble` puts in blob positions.
fn filler_digest(len: usize) -> String {
    format!("xxh3:{:016x}", xxh3_64(&vec![0xabu8; len]))
}

/// A minimal valid file: one u8 tensor of 8 bytes at offset 4096.
fn minimal() -> Vec<u8> {
    assemble(4104, &manifest(vec![("t", dense("u8", &[8], 4096, 8))]))
}

fn patched(mut bytes: Vec<u8>, at: usize, with: &[u8]) -> Vec<u8> {
    bytes[at..at + with.len()].copy_from_slice(with);
    bytes
}

/// Writer-produced file, returned as bytes.
fn written(build: impl FnOnce(&mut Writer)) -> Vec<u8> {
    let path = std::env::temp_dir().join(format!(
        "zt-conformance-{}-{:x}",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut w = Writer::create(&path).unwrap();
    build(&mut w);
    w.finish().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    bytes
}

static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// =======================================================================
// The corpus
// =======================================================================

pub fn all_cases() -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    let minimal_len = minimal().len();

    // ---- valid --------------------------------------------------------
    cases.push(Case::open(
        "canonical-basic",
        written(|w| {
            w.add_dense("a.bias", &[4], DType::F32, &[1u8; 16]).unwrap();
            w.add_dense("a.weight", &[2, 4], DType::BF16, &[2u8; 16]).unwrap();
            w.add_dense("tied", &[2, 4], DType::BF16, &[2u8; 16]).unwrap();
        }),
        Expect::Valid,
    ));
    cases.push(Case::open("minimal", minimal(), Expect::Valid));
    cases.push(Case::open(
        "empty-objects",
        assemble(8, &manifest(vec![])),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "scalar",
        assemble(4100, &manifest(vec![("s", dense("f32", &[], 4096, 4))])),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "zero-size-tensor",
        assemble(4096, &manifest(vec![("z", dense("f32", &[0, 4], 4096, 0))])),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "aliased-identical-blobs",
        assemble(
            4104,
            &manifest(vec![
                ("a", dense("u8", &[8], 4096, 8)),
                ("b", dense("u8", &[8], 4096, 8)),
            ]),
        ),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "unknown-layout-structural",
        assemble(
            8296,
            &manifest(vec![(
                "q",
                object(
                    &[64],
                    "x.custom/1",
                    vec![
                        ("data", part("u8", [0, 4096, 32], vec![])),
                        ("scales", part("u8", [0, 8192, 104], vec![])),
                    ],
                ),
            )]),
        ),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "unknown-ltype-structural",
        assemble(
            4104,
            &manifest(vec![(
                "t",
                object(
                    &[999],
                    "dense",
                    vec![("data", part("u8", [0, 4096, 8], vec![("type", text("x.future/1"))]))],
                ),
            )]),
        ),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "f4-packed-odd-count",
        // f4_e2m1, 5 elements -> ceil(5/2) = 3 bytes
        assemble(
            4099,
            &manifest(vec![(
                "t",
                object(
                    &[5],
                    "dense",
                    vec![("data", part("u8", [0, 4096, 3], vec![("type", text("f4_e2m1"))]))],
                ),
            )]),
        ),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "shards-table",
        assemble(
            8,
            &Value::Map(vec![
                (
                    text("objects"),
                    vmap(vec![(
                        "t",
                        object(
                            &[8],
                            "dense",
                            vec![("data", part("u8", [1, 4096, 8], vec![]))],
                        ),
                    )]),
                ),
                (
                    text("shards"),
                    Value::Map(vec![(
                        Value::Uint(1),
                        vmap(vec![
                            ("size", Value::Uint(1 << 20)),
                            ("digest", text("xxh3:00112233445566aa")),
                        ]),
                    )]),
                ),
            ]),
        ),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "nonzero-reserved-ignored",
        patched(minimal(), minimal_len - 12, &[1, 0, 0, 0]),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "attributes-kitchen-sink",
        assemble(
            8,
            &Value::Map(vec![
                (
                    text("attributes"),
                    vmap(vec![
                        ("float", Value::Float(1.25)),
                        ("neg", Value::Nint(41)), // -42
                        ("nested", Value::Array(vec![Value::Null, Value::Bool(true)])),
                        ("raw", Value::Bytes(vec![0, 1, 2])),
                    ]),
                ),
                (text("objects"), vmap(vec![])),
            ]),
        ),
        Expect::Valid,
    ));
    cases.push(Case::open(
        "data-shard",
        {
            let mut b = MAGIC.to_vec();
            let mut footer = [0u8; 40];
            footer[24..28].copy_from_slice(&2u32.to_le_bytes());
            footer[32..40].copy_from_slice(&MAGIC);
            b.extend_from_slice(&footer);
            b
        },
        Expect::DataShard,
    ));

    // ---- tiered operations -------------------------------------------
    cases.push(Case {
        name: "verify-ok",
        bytes: assemble(
            4104,
            &manifest(vec![(
                "t",
                object(
                    &[8],
                    "dense",
                    vec![(
                        "data",
                        part("u8", [0, 4096, 8], vec![("digest", text(&filler_digest(8)))]),
                    )],
                ),
            )]),
        ),
        op: Op::Verify("t", "data"),
        expect: Expect::Valid,
    });
    cases.push(Case {
        name: "verify-digest-mismatch",
        bytes: assemble(
            4104,
            &manifest(vec![(
                "t",
                object(
                    &[8],
                    "dense",
                    vec![(
                        "data",
                        part(
                            "u8",
                            [0, 4096, 8],
                            vec![("digest", text("xxh3:0000000000000000"))],
                        ),
                    )],
                ),
            )]),
        ),
        op: Op::Verify("t", "data"),
        expect: Expect::Reject(Rule::Digest),
    });
    cases.push(Case {
        name: "view-unknown-encoding-refused",
        bytes: assemble(
            4101,
            &manifest(vec![(
                "t",
                object(
                    &[8],
                    "dense",
                    vec![(
                        "data",
                        part(
                            "u8",
                            [0, 4096, 5],
                            vec![
                                ("encoding", text("x.z/1")),
                                ("decoded_length", Value::Uint(8)),
                            ],
                        ),
                    )],
                ),
            )]),
        ),
        op: Op::View("t", "data"),
        expect: Expect::Unsupported,
    });
    cases.push(Case {
        name: "view-foreign-shard-refused",
        bytes: assemble(
            8,
            &Value::Map(vec![
                (
                    text("objects"),
                    vmap(vec![(
                        "t",
                        object(&[8], "dense", vec![("data", part("u8", [1, 4096, 8], vec![]))]),
                    )]),
                ),
                (
                    text("shards"),
                    Value::Map(vec![(
                        Value::Uint(1),
                        vmap(vec![
                            ("size", Value::Uint(1 << 20)),
                            ("digest", text("xxh3:00112233445566aa")),
                        ]),
                    )]),
                ),
            ]),
        ),
        op: Op::View("t", "data"),
        expect: Expect::Unsupported,
    });

    // ---- reject: container -------------------------------------------
    cases.push(Case::open(
        "file-too-small",
        vec![0u8; 10],
        Expect::Reject(Rule::FileTooSmall),
    ));
    cases.push(Case::open(
        "header-magic",
        patched(minimal(), 0, &[0x00]),
        Expect::Reject(Rule::HeaderMagic),
    ));
    cases.push(Case::open(
        "footer-magic",
        patched(minimal(), minimal_len - 1, &[0x00]),
        Expect::Reject(Rule::FooterMagic),
    ));
    cases.push(Case::open(
        "version-unsupported",
        patched(minimal(), minimal_len - 16, &3u32.to_le_bytes()),
        Expect::Reject(Rule::Version),
    ));
    cases.push(Case::open(
        "manifest-too-large",
        patched(minimal(), minimal_len - 32, &(2u64 << 30).to_le_bytes()),
        Expect::Reject(Rule::ManifestTooLarge),
    ));
    cases.push(Case::open(
        "manifest-out-of-bounds",
        patched(minimal(), minimal_len - 40, &(1u64 << 20).to_le_bytes()),
        Expect::Reject(Rule::ManifestBounds),
    ));
    cases.push(Case::open(
        "manifest-misaligned",
        patched(minimal(), minimal_len - 40, &4100u64.to_le_bytes()),
        Expect::Reject(Rule::BlobAlignment),
    ));
    cases.push(Case::open(
        "manifest-hash-mismatch",
        {
            let bytes = minimal();
            let m_off =
                u64::from_le_bytes(bytes[minimal_len - 40..minimal_len - 32].try_into().unwrap());
            patched(bytes, m_off as usize, &[0xff])
        },
        Expect::Reject(Rule::ManifestHash),
    ));
    cases.push(Case::open(
        "data-shard-nonzero-offset",
        {
            let mut b = MAGIC.to_vec();
            let mut footer = [0u8; 40];
            footer[0..8].copy_from_slice(&4096u64.to_le_bytes());
            footer[24..28].copy_from_slice(&2u32.to_le_bytes());
            footer[32..40].copy_from_slice(&MAGIC);
            b.extend_from_slice(&footer);
            b
        },
        Expect::Reject(Rule::ManifestBounds),
    ));

    // ---- reject: CBOR -------------------------------------------------
    cases.push(Case::open(
        "cbor-trailing-bytes",
        assemble_raw(8, &[0xa0, 0x00]),
        Expect::Reject(Rule::CborSyntax),
    ));
    cases.push(Case::open(
        "cbor-tag",
        assemble_raw(8, &[0xc0, 0x00]),
        Expect::Reject(Rule::CborSyntax),
    ));
    cases.push(Case::open(
        "cbor-indefinite",
        assemble_raw(8, &[0x9f, 0xff]),
        Expect::Reject(Rule::CborSyntax),
    ));
    cases.push(Case::open(
        "cbor-non-shortest",
        assemble_raw(8, &[0x18, 0x05]),
        Expect::Reject(Rule::CborDeterminism),
    ));
    cases.push(Case::open(
        "cbor-unsorted-keys",
        assemble_raw(8, &[0xa2, 0x61, b'b', 0x00, 0x61, b'a', 0x00]),
        Expect::Reject(Rule::CborDeterminism),
    ));
    cases.push(Case::open(
        "cbor-duplicate-key",
        assemble_raw(8, &[0xa2, 0x61, b'a', 0x00, 0x61, b'a', 0x00]),
        Expect::Reject(Rule::CborDuplicateKey),
    ));
    cases.push(Case::open(
        "cbor-depth",
        {
            let mut nested = vec![0x81u8; 40]; // 40 nested single-element arrays
            nested.push(0x00);
            assemble_raw(8, &nested)
        },
        Expect::Reject(Rule::CborDepth),
    ));

    // ---- reject: schema ----------------------------------------------
    cases.push(Case::open(
        "schema-missing-objects",
        assemble_raw(8, &[0xa0]),
        Expect::Reject(Rule::Schema),
    ));
    cases.push(Case::open(
        "schema-unknown-dtype",
        assemble(4104, &manifest(vec![("t", dense("f4", &[8], 4096, 8))])),
        Expect::Reject(Rule::Schema),
    ));
    cases.push(Case::open(
        "schema-blob-arity",
        assemble(
            4104,
            &manifest(vec![(
                "t",
                object(
                    &[8],
                    "dense",
                    vec![("data", vmap(vec![("dtype", text("u8")), ("blob", uints(&[0, 4096]))]))],
                ),
            )]),
        ),
        Expect::Reject(Rule::Schema),
    ));
    cases.push(Case::open(
        "schema-encoding-without-decoded-length",
        assemble(
            4104,
            &manifest(vec![(
                "t",
                object(
                    &[8],
                    "dense",
                    vec![("data", part("u8", [0, 4096, 8], vec![("encoding", text("x.z/1"))]))],
                ),
            )]),
        ),
        Expect::Reject(Rule::Schema),
    ));
    cases.push(Case::open(
        "schema-ltype-dtype-mismatch",
        assemble(
            4104,
            &manifest(vec![(
                "t",
                object(
                    &[2],
                    "dense",
                    vec![("data", part("f32", [0, 4096, 8], vec![("type", text("bool"))]))],
                ),
            )]),
        ),
        Expect::Reject(Rule::Schema),
    ));
    cases.push(Case::open(
        "schema-digest-uppercase-hex",
        assemble(
            4104,
            &manifest(vec![(
                "t",
                object(
                    &[8],
                    "dense",
                    vec![(
                        "data",
                        part("u8", [0, 4096, 8], vec![("digest", text("xxh3:00FF00FF00FF00FF"))]),
                    )],
                ),
            )]),
        ),
        Expect::Reject(Rule::Schema),
    ));
    cases.push(Case::open(
        "shard-table-key-zero",
        assemble(
            8,
            &Value::Map(vec![
                (text("objects"), vmap(vec![])),
                (
                    text("shards"),
                    Value::Map(vec![(
                        Value::Uint(0),
                        vmap(vec![
                            ("size", Value::Uint(1 << 20)),
                            ("digest", text("xxh3:00112233445566aa")),
                        ]),
                    )]),
                ),
            ]),
        ),
        Expect::Reject(Rule::ShardIndex),
    ));
    cases.push(Case::open(
        "shard-ref-not-in-table",
        assemble(
            8,
            &manifest(vec![(
                "t",
                object(&[8], "dense", vec![("data", part("u8", [7, 4096, 8], vec![]))]),
            )]),
        ),
        Expect::Reject(Rule::ShardIndex),
    ));

    // ---- reject: attributes ------------------------------------------
    cases.push(Case::open(
        "attributes-not-a-map",
        assemble_raw(
            8,
            &cbor::encode(&Value::Map(vec![
                (text("attributes"), Value::Uint(7)),
                (text("objects"), Value::Map(vec![])),
            ]))
            .unwrap(),
        ),
        Expect::Reject(Rule::Schema),
    ));
    cases.push(Case::open(
        "attribute-key-empty",
        assemble(
            8,
            &Value::Map(vec![
                (text("attributes"), Value::Map(vec![(text(""), Value::Null)])),
                (text("objects"), Value::Map(vec![])),
            ]),
        ),
        Expect::Reject(Rule::Name),
    ));
    cases.push(Case::open(
        "attribute-key-nul",
        assemble(
            8,
            &Value::Map(vec![
                (
                    text("attributes"),
                    Value::Map(vec![(text("a\u{0000}b"), Value::Null)]),
                ),
                (text("objects"), Value::Map(vec![])),
            ]),
        ),
        Expect::Reject(Rule::Name),
    ));
    cases.push(Case::open(
        "attribute-key-not-text",
        assemble(
            8,
            &Value::Map(vec![
                (
                    text("attributes"),
                    Value::Map(vec![(Value::Uint(1), Value::Null)]),
                ),
                (text("objects"), Value::Map(vec![])),
            ]),
        ),
        Expect::Reject(Rule::Schema),
    ));
    cases.push(Case::open(
        "object-attributes-not-a-map",
        assemble(
            4104,
            &manifest(vec![(
                "t",
                Value::Map(vec![
                    (text("shape"), uints(&[8])),
                    (text("layout"), text("dense")),
                    (text("attributes"), Value::Uint(1)),
                    (
                        text("parts"),
                        vmap(vec![("data", part("u8", [0, 4096, 8], vec![]))]),
                    ),
                ]),
            )]),
        ),
        Expect::Reject(Rule::Schema),
    ));

    // ---- reject: cross-shard overlap ---------------------------------
    cases.push(Case::open(
        "shard-blob-partial-overlap",
        assemble(
            8,
            &Value::Map(vec![
                (
                    text("objects"),
                    vmap(vec![
                        (
                            "a",
                            object(
                                &[2048],
                                "dense",
                                vec![("data", part("f32", [1, 4096, 8192], vec![]))],
                            ),
                        ),
                        (
                            "b",
                            object(
                                &[2],
                                "dense",
                                vec![("data", part("f32", [1, 8192, 8], vec![]))],
                            ),
                        ),
                    ]),
                ),
                (
                    text("shards"),
                    Value::Map(vec![(
                        Value::Uint(1),
                        vmap(vec![
                            ("size", Value::Uint(1 << 20)),
                            ("digest", text("xxh3:00112233445566aa")),
                        ]),
                    )]),
                ),
            ]),
        ),
        Expect::Reject(Rule::BlobOverlap),
    ));

    // ---- logical-type content rules (Appendix A) ---------------------
    cases.push(Case {
        name: "bool-invalid-byte",
        bytes: assemble(
            4104,
            &manifest(vec![(
                "t",
                object(
                    &[8],
                    "dense",
                    vec![("data", part("u8", [0, 4096, 8], vec![("type", text("bool"))]))],
                ),
            )]),
        ),
        op: Op::Verify("t", "data"),
        expect: Expect::Reject(Rule::LayoutData),
    });
    cases.push(Case {
        name: "f4-nonzero-odd-nibble",
        // 5 elements -> 3 bytes; the filler 0xab has a nonzero high nibble
        bytes: assemble(
            4099,
            &manifest(vec![(
                "t",
                object(
                    &[5],
                    "dense",
                    vec![(
                        "data",
                        part("u8", [0, 4096, 3], vec![("type", text("f4_e2m1"))]),
                    )],
                ),
            )]),
        ),
        op: Op::Verify("t", "data"),
        expect: Expect::Reject(Rule::LayoutData),
    });

    // ---- reject: names and shapes ------------------------------------
    cases.push(Case::open(
        "name-empty",
        assemble(4104, &manifest(vec![("", dense("u8", &[8], 4096, 8))])),
        Expect::Reject(Rule::Name),
    ));
    cases.push(Case::open(
        "name-nul",
        assemble(4104, &manifest(vec![("a\0b", dense("u8", &[8], 4096, 8))])),
        Expect::Reject(Rule::Name),
    ));
    cases.push(Case::open("name-too-long", {
        let long = "n".repeat(1025);
        let m = Value::Map(vec![(
            text("objects"),
            Value::Map(vec![(text(&long), dense("u8", &[8], 4096, 8))]),
        )]);
        assemble(4104, &m)
    }, Expect::Reject(Rule::Name)));
    cases.push(Case::open(
        "shape-rank-65",
        assemble(
            4097,
            &manifest(vec![("t", dense("u8", &[1u64; 65], 4096, 1))]),
        ),
        Expect::Reject(Rule::Shape),
    ));
    cases.push(Case::open(
        "shape-product-overflow",
        assemble(
            4104,
            &manifest(vec![("t", dense("u8", &[u64::MAX, 2], 4096, 8))]),
        ),
        Expect::Reject(Rule::Shape),
    ));

    // ---- reject: blobs ------------------------------------------------
    cases.push(Case::open(
        "blob-misaligned",
        assemble(4108, &manifest(vec![("t", dense("u8", &[8], 4100, 8))])),
        Expect::Reject(Rule::BlobAlignment),
    ));
    cases.push(Case::open(
        "blob-offset-zero",
        assemble(4104, &manifest(vec![("t", dense("u8", &[8], 0, 8))])),
        Expect::Reject(Rule::BlobAlignment),
    ));
    cases.push(Case::open(
        "blob-out-of-bounds",
        assemble(
            4104,
            &manifest(vec![("t", dense("u8", &[1 << 40], 4096, 1 << 40))]),
        ),
        Expect::Reject(Rule::BlobBounds),
    ));
    cases.push(Case::open(
        "blob-partial-overlap",
        assemble(
            12288,
            &manifest(vec![
                ("a", dense("f32", &[2048], 4096, 8192)),
                ("b", dense("f32", &[2], 8192, 8)),
            ]),
        ),
        Expect::Reject(Rule::BlobOverlap),
    ));

    // ---- zt.sparse_csr/1 ---------------------------------------------
    cases.push(Case::open(
        "csr-metadata-valid",
        csr_file(&[0, 2, 1], &[0, 2, 3]),
        Expect::Valid,
    ));
    cases.push(Case {
        name: "csr-data-valid",
        bytes: csr_file(&[0, 2, 1], &[0, 2, 3]),
        op: Op::ReadCsr("m"),
        expect: Expect::Valid,
    });
    cases.push(Case {
        name: "csr-indptr-not-zero-based",
        bytes: csr_file(&[0, 2, 1], &[1, 2, 3]),
        op: Op::ReadCsr("m"),
        expect: Expect::Reject(Rule::LayoutData),
    });
    cases.push(Case {
        name: "csr-index-out-of-cols",
        bytes: csr_file(&[0, 5, 1], &[0, 2, 3]),
        op: Op::ReadCsr("m"),
        expect: Expect::Reject(Rule::LayoutData),
    });
    cases.push(Case {
        name: "csr-row-not-increasing",
        bytes: csr_file(&[2, 0, 1], &[0, 2, 3]),
        op: Op::ReadCsr("m"),
        expect: Expect::Reject(Rule::LayoutData),
    });
    cases.push(Case::open(
        "csr-missing-part",
        assemble(
            8300,
            &manifest(vec![(
                "m",
                object(
                    &[2, 3],
                    "zt.sparse_csr/1",
                    vec![
                        ("indices", part("u64", [0, 4096, 24], vec![])),
                        ("values", part("f32", [0, 8192, 12], vec![])),
                    ],
                ),
            )]),
        ),
        Expect::Reject(Rule::LayoutRule),
    ));
    cases.push(Case::open(
        "csr-signed-indices",
        assemble(
            12300,
            &manifest(vec![(
                "m",
                object(
                    &[2, 3],
                    "zt.sparse_csr/1",
                    vec![
                        ("indices", part("i64", [0, 4096, 24], vec![])),
                        ("indptr", part("i64", [0, 8192, 24], vec![])),
                        ("values", part("f32", [0, 12288, 12], vec![])),
                    ],
                ),
            )]),
        ),
        Expect::Reject(Rule::LayoutRule),
    ));
    cases.push(Case::open(
        "csr-indptr-count",
        assemble(
            12300,
            &manifest(vec![(
                "m",
                object(
                    &[2, 3],
                    "zt.sparse_csr/1",
                    vec![
                        ("indices", part("u64", [0, 4096, 24], vec![])),
                        ("indptr", part("u64", [0, 8192, 16], vec![])),
                        ("values", part("f32", [0, 12288, 12], vec![])),
                    ],
                ),
            )]),
        ),
        Expect::Reject(Rule::LayoutRule),
    ));
    cases.push(Case::open(
        "csr-values-size",
        assemble(
            12300,
            &manifest(vec![(
                "m",
                object(
                    &[2, 3],
                    "zt.sparse_csr/1",
                    vec![
                        ("indices", part("u64", [0, 4096, 24], vec![])),
                        ("indptr", part("u64", [0, 8192, 24], vec![])),
                        ("values", part("f32", [0, 12288, 8], vec![])),
                    ],
                ),
            )]),
        ),
        Expect::Reject(Rule::LayoutRule),
    ));

    // ---- reject: layouts ---------------------------------------------
    cases.push(Case::open(
        "dense-extra-part",
        assemble(
            8296,
            &manifest(vec![(
                "t",
                object(
                    &[8],
                    "dense",
                    vec![
                        ("data", part("u8", [0, 4096, 8], vec![])),
                        ("extra", part("u8", [0, 8192, 8], vec![])),
                    ],
                ),
            )]),
        ),
        Expect::Reject(Rule::LayoutRule),
    ));
    cases.push(Case::open(
        "dense-size-mismatch",
        assemble(4104, &manifest(vec![("t", dense("f32", &[3], 4096, 8))])),
        Expect::Reject(Rule::DenseSize),
    ));
    cases.push(Case::open(
        "dense-f4-size-mismatch",
        assemble(
            4100,
            &manifest(vec![(
                "t",
                object(
                    &[5],
                    "dense",
                    vec![("data", part("u8", [0, 4096, 4], vec![("type", text("f4_e2m1"))]))],
                ),
            )]),
        ),
        Expect::Reject(Rule::DenseSize),
    ));

    cases
}
