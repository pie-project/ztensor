//! Reading tensors.
//!
//! A [`Source`] is a catalog over one or more [`Store`]s. One `.zt` file, a
//! `.zt` root plus its shards, a foreign checkpoint and a set of unrelated
//! files all come back as this same type. Only the way the catalog was built
//! differs.
//!
//! There are three ways to get at the bytes:
//!
//! - [`bytes`](Part::bytes) gives the best the source can do, and says
//!   whether it borrowed or copied.
//! - [`map`](Part::map) gives a borrow or an error, never a hidden copy.
//! - [`locate`](Part::locate) gives the address, so the caller can do its own
//!   I/O with io_uring, cuFile or a staged host-to-device copy.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::catalog::{Catalog, Entry, Location, PartEntry, Payload};
use crate::cbor::Value;
use crate::error::{Error, Result, Rule};
use crate::schema::{digest_matches, DType, DigestAlgorithm, Hasher, Manifest, Shard};
use crate::store::{Store, StoreId};
use crate::validate;
use crate::vocab::Vocabulary;

// =======================================================================
// capabilities
// =======================================================================

/// What can be done with one part's bytes.
///
/// Each field is named after the operation it gates, and is computed from
/// that operation's own precondition, so the report cannot disagree with the
/// behaviour. `evict` implies `map` implies `locate`, but that falls out of
/// the predicates rather than being a rule to remember.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Caps {
    /// [`Part::map`] will succeed: the bytes are raw and the file is mapped.
    pub map: bool,
    /// [`Part::locate`] will succeed: the decoded bytes are exactly one range
    /// of one file, so a caller can read them without this library.
    pub locate: bool,
    /// [`Part::evict`] will succeed: no other blob shares an OS page with
    /// this one, so dropping its pages cannot disturb a neighbour.
    pub evict: bool,
    /// [`Part::verify`] will check a digest rather than report that there is
    /// none to check.
    pub verify: bool,
    /// Largest power of two dividing the part's file offset. This describes
    /// the data rather than gating an operation: the pointer alignment of a
    /// mapping is `min(this, page)`.
    pub alignment: u64,
}

/// Bytes, and whether they were borrowed or copied.
///
/// Dereferences to `[u8]`, so most callers never look inside. A caller that
/// does care, such as a loader checking whether it just paid for a copy, can
/// ask [`is_mapped`](Bytes::is_mapped).
#[derive(Debug)]
pub enum Bytes<'a> {
    /// Borrowed from a file mapping.
    Mapped(&'a [u8]),
    /// Copied, decoded, or decompressed into memory.
    Owned(Vec<u8>),
}

impl Bytes<'_> {
    pub fn is_mapped(&self) -> bool {
        matches!(self, Bytes::Mapped(_))
    }

    pub fn into_owned(self) -> Vec<u8> {
        match self {
            Bytes::Mapped(s) => s.to_vec(),
            Bytes::Owned(v) => v,
        }
    }
}

impl std::ops::Deref for Bytes<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Mapped(s) => s,
            Bytes::Owned(v) => v,
        }
    }
}

impl AsRef<[u8]> for Bytes<'_> {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

/// The outcome of a successful [`Part::verify`].
///
/// A digest *mismatch* is not here: that is a rejected file, and it comes back
/// as `Err(Reject { rule: Rule::Digest, .. })`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verified {
    /// A digest was checked and matched.
    Digest,
    /// There is no digest; content rules (if any) passed.
    NoDigest,
}

impl Verified {
    /// True when a digest was actually checked.
    pub fn checked(self) -> bool {
        self == Verified::Digest
    }
}

// =======================================================================
// shard resolution
// =======================================================================

/// Resolves a shard name + identity to a file path. A name is a label the
/// producer chose, not a location: turning it into bytes is entirely the
/// transport's concern (spec §7.1, Appendix B).
///
/// A name is constrained by the format to `[A-Za-z0-9._-]`, no leading dot,
/// at most 64 bytes, so a resolver may use it as a single path component
/// without sanitizing it.
pub trait ShardResolver {
    fn resolve(&self, name: &str, shard: &Shard) -> Result<PathBuf>;
}

/// Closures are resolvers: `|name, shard| Ok(path)`.
impl<F: Fn(&str, &Shard) -> Result<PathBuf>> ShardResolver for F {
    fn resolve(&self, name: &str, shard: &Shard) -> Result<PathBuf> {
        self(name, shard)
    }
}

/// The positional convention (Appendix B): root `<dir>/<stem>.zt` maps a
/// shard named `n` to `<dir>/<stem>-<n>.zt`.
///
/// Naming shards `00001-of-00003` and so on therefore reproduces the file
/// names checkpoints already ship with.
pub struct PositionalResolver {
    dir: PathBuf,
    stem: String,
}

impl PositionalResolver {
    pub fn for_root(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self {
            dir: root.parent().unwrap_or(Path::new(".")).to_path_buf(),
            stem: root
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".to_string()),
        }
    }
}

impl ShardResolver for PositionalResolver {
    fn resolve(&self, name: &str, _shard: &Shard) -> Result<PathBuf> {
        Ok(self.dir.join(format!("{}-{name}.zt", self.stem)))
    }
}

/// The content-addressed convention (Appendix B): a shard with digest
/// `algo:hex` lives at `<store>/blobs/<algo>/<hex>`.
pub struct CasResolver {
    pub store: PathBuf,
}

impl ShardResolver for CasResolver {
    fn resolve(&self, _name: &str, shard: &Shard) -> Result<PathBuf> {
        let (algo, hex) = shard.digest.split_once(':').unwrap_or(("", ""));
        Ok(self.store.join("blobs").join(algo).join(hex))
    }
}

/// Finds shards by identity: scans a directory once and matches each file by
/// size and whole-file digest, ignoring what anything is called.
///
/// Use this when nobody agreed on the names: a directory someone handed you,
/// or files that were renamed on the way. Because it never looks at a name,
/// it still works after a rename.
pub struct DirectoryResolver {
    /// Digest -> (size, path). Keying on the digest alone means a lookup
    /// needs no allocation, and lets a file whose digest matches but whose
    /// size does not be reported as corrupt instead of missing.
    by_digest: BTreeMap<String, (u64, PathBuf)>,
}

impl DirectoryResolver {
    /// Indexes every `.zt` file directly inside `dir`.
    ///
    /// Digests are computed eagerly, so this costs one full read of every
    /// candidate file. Prefer a cheaper convention when one applies; this is
    /// the fallback for when no name can be trusted.
    ///
    /// Files that are not readable containers are skipped rather than
    /// reported: a directory is allowed to hold things that are not shards,
    /// and a shard that is genuinely missing surfaces at `resolve` with the
    /// identity that was being looked for.
    pub fn scan(dir: impl AsRef<Path>) -> Result<Self> {
        let mut by_digest = BTreeMap::new();
        for entry in std::fs::read_dir(dir.as_ref())? {
            let path = entry?.path();
            if path.extension().is_some_and(|e| e == "zt") {
                if let Ok(id) = shard_identity(&path) {
                    by_digest.insert(id.digest, (id.size, path));
                }
            }
        }
        Ok(Self { by_digest })
    }
}

impl ShardResolver for DirectoryResolver {
    fn resolve(&self, name: &str, shard: &Shard) -> Result<PathBuf> {
        match self.by_digest.get(&shard.digest) {
            Some((size, path)) if *size == shard.size => Ok(path.clone()),
            Some((size, path)) => Err(Error::reject(
                Rule::ShardIdentity,
                format!(
                    "shard {name:?}: {} has the expected digest but is {size} bytes, not {}",
                    path.display(),
                    shard.size
                ),
            )),
            None => Err(Error::NotFound(format!(
                "shard {name:?} ({} bytes, {}) is not in the scanned directory",
                shard.size, shard.digest
            ))),
        }
    }
}

// =======================================================================
// opening
// =======================================================================

/// How to open a source.
pub struct Options {
    vocab: Option<Arc<Vocabulary>>,
    resolver: Option<Box<dyn ShardResolver>>,
    map: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            vocab: None,
            resolver: None,
            map: true,
        }
    }
}

impl Options {
    /// The profiles the reader should know. Defaults to
    /// [`Vocabulary::standard`].
    pub fn vocabulary(mut self, vocab: &Vocabulary) -> Self {
        self.vocab = Some(Arc::new(vocab.clone()));
        self
    }

    /// How to find shard files. Defaults to [`PositionalResolver`].
    pub fn resolver(mut self, resolver: impl ShardResolver + 'static) -> Self {
        self.resolver = Some(Box::new(resolver));
        self
    }

    /// Map the files (the default). With `false`, files are opened but not
    /// mapped: metadata and addresses are available, borrowed reads are not.
    pub fn map(mut self, map: bool) -> Self {
        self.map = map;
        self
    }

    pub(crate) fn vocabulary_arc(&self) -> Arc<Vocabulary> {
        self.vocab.clone().unwrap_or_else(Vocabulary::shared)
    }

    fn open_store(&self, path: &Path, format: &'static str) -> Result<Store> {
        if self.map {
            Store::map(path, format)
        } else {
            Store::index(path, format)
        }
    }

    /// Opens a `.zt` file, following its shard table if it has one.
    pub fn open(self, path: impl AsRef<Path>) -> Result<Source> {
        let path = path.as_ref();
        let vocab = self.vocabulary_arc();
        let root = self.open_store(path, "zt")?;
        let parsed = validate::read(&root, &vocab)?;
        let root = root.with_occupied(parsed.occupied);

        let Some(manifest) = parsed.manifest else {
            // A data shard carries no manifest: it is a byte store some other
            // file addresses into, and there is nothing here to enumerate.
            return Ok(Source {
                stores: vec![root],
                catalog: Catalog::new(),
                manifest: None,
                data_shard: true,
                vocab,
            });
        };

        let mut stores = vec![root];
        let mut store_of: BTreeMap<&str, StoreId> = BTreeMap::new();

        let positional;
        let resolver: &dyn ShardResolver = match &self.resolver {
            Some(r) => r.as_ref(),
            None => {
                positional = PositionalResolver::for_root(path);
                &positional
            }
        };

        for (name, shard) in &manifest.shards {
            let shard_path = resolver.resolve(name, shard)?;
            let store = self.open_store(&shard_path, "zt")?;
            if store.len() != shard.size {
                return Err(Error::reject(
                    Rule::ShardIdentity,
                    format!(
                        "shard {name:?}: {} is {} bytes, the root expects {}",
                        shard_path.display(),
                        store.len(),
                        shard.size
                    ),
                ));
            }
            // The two cheap rungs of the ladder at open time: exact size, and
            // a container frame that parses. Digests are Source::verify_shards.
            let parsed = validate::read(&store, &vocab)
                .map_err(|e| Error::reject(Rule::ShardIdentity, format!("shard {name:?}: {e}")))?;
            store_of.insert(name.as_str(), StoreId(stores.len() as u32));
            stores.push(store.with_occupied(parsed.occupied));
        }

        let catalog = resolve_manifest(&manifest, &store_of)?;
        Ok(Source {
            stores,
            catalog,
            manifest: Some(manifest),
            data_shard: false,
            vocab,
        })
    }

    /// Opens several `.zt` files as one name space.
    pub fn open_all(self, paths: &[impl AsRef<Path>]) -> Result<Source> {
        let vocab = self.vocabulary_arc();
        let mut sources = Vec::with_capacity(paths.len());
        for path in paths {
            let opts = Options {
                vocab: Some(vocab.clone()),
                resolver: None,
                map: self.map,
            };
            sources.push(opts.open(path.as_ref())?);
        }
        Source::merge(sources)
    }
}

/// Turns a manifest's blob references into addresses.
fn resolve_manifest(manifest: &Manifest, store_of: &BTreeMap<&str, StoreId>) -> Result<Catalog> {
    let mut catalog = Catalog::new();
    catalog.set_attributes(manifest.attributes.clone());
    for (name, obj) in &manifest.objects {
        let mut parts = BTreeMap::new();
        for (pname, part) in &obj.parts {
            // No shard name is the containing file, which is always store 0.
            let store = match &part.blob.shard {
                None => StoreId(0),
                Some(s) => *store_of.get(s.as_str()).ok_or_else(|| {
                    Error::reject(
                        Rule::ShardRef,
                        format!("{name:?}/{pname:?}: shard {s:?} not resolved"),
                    )
                })?,
            };
            let at = Location {
                store,
                offset: part.blob.offset,
                len: part.blob.length,
            };
            let payload = match (&part.encoding, part.decoded_length) {
                (None, _) => Payload::At(at),
                (Some(encoding), Some(decoded_len)) => Payload::Encoded {
                    at,
                    encoding: encoding.clone(),
                    decoded_len,
                },
                (Some(_), None) => {
                    return Err(Error::reject(
                        Rule::Schema,
                        format!("{name:?}/{pname:?}: encoding without decoded_length"),
                    ))
                }
            };
            parts.insert(
                pname.clone(),
                PartEntry {
                    dtype: part.dtype,
                    logical: part.logical.clone(),
                    payload,
                    digest: part.digest.clone(),
                },
            );
        }
        catalog.insert(
            name.clone(),
            Entry {
                shape: obj.shape.clone(),
                layout: obj.layout.clone(),
                attributes: obj.attributes.clone(),
                parts,
            },
        );
    }
    Ok(catalog)
}

// =======================================================================
// Source
// =======================================================================

pub struct Source {
    stores: Vec<Store>,
    catalog: Catalog,
    /// The root manifest. `Some` only when this source is a single `.zt`
    /// root, since that is the only case where one was written.
    manifest: Option<Manifest>,
    data_shard: bool,
    vocab: Arc<Vocabulary>,
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source")
            .field("stores", &self.stores.len())
            .field("tensors", &self.catalog.len())
            .field("data_shard", &self.data_shard)
            .finish()
    }
}

impl Source {
    /// Opens a `.zt` file, including a sharded model. Shards are found with
    /// the positional convention.
    ///
    /// Foreign formats go through `ztensor_compat::open`, which detects the
    /// format and hands back one of these.
    pub fn open(path: impl AsRef<Path>) -> Result<Source> {
        Options::default().open(path)
    }

    /// Opens several `.zt` files as one name space.
    pub fn open_all(paths: &[impl AsRef<Path>]) -> Result<Source> {
        Options::default().open_all(paths)
    }

    /// Opens without mapping: metadata and addresses only.
    ///
    /// This is what a planner needs. It answers where every tensor lives and
    /// costs two reads instead of mapping the whole checkpoint.
    pub fn index(path: impl AsRef<Path>) -> Result<Source> {
        Options::default().map(false).open(path)
    }

    pub fn options() -> Options {
        Options::default()
    }

    /// Builds a source directly from a projection's stores and catalog. This
    /// is how the compat crate hands a foreign format back.
    pub fn from_parts(stores: Vec<Store>, catalog: Catalog) -> Result<Source> {
        Self::from_parts_with(stores, catalog, Vocabulary::shared())
    }

    pub fn from_parts_with(
        stores: Vec<Store>,
        catalog: Catalog,
        vocab: Arc<Vocabulary>,
    ) -> Result<Source> {
        for (name, entry) in catalog.iter() {
            for (pname, part) in &entry.parts {
                if part.payload.store().0 as usize >= stores.len() {
                    return Err(Error::InvalidInput(format!(
                        "{name:?}/{pname:?} addresses store {} of {}",
                        part.payload.store(),
                        stores.len()
                    )));
                }
            }
        }
        Ok(Source {
            stores,
            catalog,
            manifest: None,
            data_shard: false,
            vocab,
        })
    }

    /// Reads several sources as one name space.
    ///
    /// Every foreign snapshot arrives in this shape: N files that each
    /// describe themselves completely, with nothing binding them but the
    /// caller's list. There is no root, no digest and no size anyone
    /// promised, so nothing is verified across the set. What is checked is
    /// that the names do not collide, since a
    /// tensor in two files is a broken set and picking a winner would load
    /// half a model and say nothing.
    pub fn merge(sources: Vec<Source>) -> Result<Source> {
        let vocab = sources
            .first()
            .map(|s| s.vocab.clone())
            .unwrap_or_else(Vocabulary::shared);
        let mut stores: Vec<Store> = Vec::new();
        let mut merged = Catalog::new();
        let mut attributes: Option<Value> = None;

        for source in sources {
            let base = stores.len() as u32;
            let Source {
                stores: mut part_stores,
                mut catalog,
                ..
            } = source;
            catalog.rebase(|id| StoreId(id.0 + base));
            if attributes.is_none() {
                attributes = catalog.attributes().cloned();
            }
            stores.append(&mut part_stores);
            for (name, entry) in catalog.into_iter_sorted() {
                if let Some(previous) = merged.get(&name) {
                    let here = entry.store().map(|id| stores[id.0 as usize].path());
                    let there = previous.store().map(|id| stores[id.0 as usize].path());
                    return Err(Error::reject(
                        Rule::NameCollision,
                        format!(
                            "tensor {name:?} is in both {} and {}",
                            there.unwrap_or(Path::new("?")).display(),
                            here.unwrap_or(Path::new("?")).display()
                        ),
                    ));
                }
                merged.insert(name, entry);
            }
        }
        merged.set_attributes(attributes);
        Ok(Source {
            stores,
            catalog: merged,
            manifest: None,
            data_shard: false,
            vocab,
        })
    }

    pub fn len(&self) -> usize {
        self.catalog.len()
    }

    pub fn is_empty(&self) -> bool {
        self.catalog.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.catalog.contains(name)
    }

    /// Tensor names, sorted, across every file of this source.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.catalog.names()
    }

    pub fn tensors(&self) -> impl Iterator<Item = Tensor<'_>> {
        self.catalog.iter().map(|(name, entry)| Tensor {
            src: self,
            name,
            entry,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor<'_>> {
        self.get(name)
            .ok_or_else(|| Error::NotFound(format!("tensor {name:?}")))
    }

    pub fn get(&self, name: &str) -> Option<Tensor<'_>> {
        let (name, entry) = self.catalog.iter().find(|(n, _)| *n == name)?;
        Some(Tensor {
            src: self,
            name,
            entry,
        })
    }

    /// File-level attributes.
    pub fn attributes(&self) -> Option<&Value> {
        self.catalog.attributes()
    }

    /// The root manifest, when this source is a single `.zt` root. Foreign
    /// formats and merged sets do not have one, so they return `None`.
    pub fn manifest(&self) -> Option<&Manifest> {
        self.manifest.as_ref()
    }

    /// True if this is a `.zt` data shard: a container with no manifest, whose
    /// bytes some other file addresses into (spec §7.2).
    pub fn is_data_shard(&self) -> bool {
        self.data_shard
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Every file this source reads from, in the order that fixes their
    /// [`StoreId`]s: `stores()[k]` is `StoreId(k)`.
    pub fn stores(&self) -> &[Store] {
        &self.stores
    }

    /// The file an address names.
    ///
    /// # Panics
    ///
    /// If `id` did not come from this `Source`. Every [`Location`] this source
    /// hands out carries a valid one; a `StoreId` built by hand may not.
    pub fn store(&self, id: StoreId) -> &Store {
        &self.stores[id.0 as usize]
    }

    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Deep shard verification: the whole-file digest of every shard against
    /// the root's shard table. Only a `.zt` root has one.
    pub fn verify_shards(&self) -> Result<()> {
        let Some(manifest) = &self.manifest else {
            return Ok(());
        };
        // `open` pushes the root first, then one store per shard in the
        // manifest's own (name) order, so position `k + 1` is shard `k`.
        // Nothing enforces that, so check it here. Getting it wrong would
        // hash the wrong file and report a mismatch against a shard that is
        // fine.
        if self.stores.len() != manifest.shards.len() + 1 {
            return Err(Error::reject(
                Rule::ShardIdentity,
                format!(
                    "{} shards resolved for a table of {}",
                    self.stores.len().saturating_sub(1),
                    manifest.shards.len()
                ),
            ));
        }
        for (position, (name, shard)) in manifest.shards.iter().enumerate() {
            let store = &self.stores[position + 1];
            let mut hasher = Hasher::new(DigestAlgorithm::of_digest(&shard.digest)?);
            let mut at = 0u64;
            while at < store.len() {
                let n = (store.len() - at).min(1 << 20);
                hasher.update(&store.read(at, n)?);
                at += n;
            }
            if hasher.finish() != shard.digest {
                return Err(Error::reject(
                    Rule::ShardIdentity,
                    format!("shard {name:?}: digest mismatch"),
                ));
            }
        }
        Ok(())
    }
}

// =======================================================================
// handles
// =======================================================================

/// One named tensor of a [`Source`]. Holding one has read nothing.
#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    src: &'a Source,
    name: &'a str,
    entry: &'a Entry,
}

impl<'a> Tensor<'a> {
    pub fn name(&self) -> &'a str {
        self.name
    }

    pub fn shape(&self) -> &'a [u64] {
        &self.entry.shape
    }

    /// Layout profile id: `"dense"` for an ordinary tensor.
    pub fn layout(&self) -> &'a str {
        &self.entry.layout
    }

    pub fn attributes(&self) -> Option<&'a Value> {
        self.entry.attributes.as_ref()
    }

    pub fn entry(&self) -> &'a Entry {
        self.entry
    }

    pub fn num_elements(&self) -> Result<u64> {
        self.entry.num_elements()
    }

    /// Part names, sorted. A dense tensor has exactly one, `"data"`; a
    /// quantized one has its payload and its scales.
    pub fn parts(&self) -> impl Iterator<Item = &'a str> {
        self.entry.parts.keys().map(String::as_str)
    }

    pub fn part(&self, name: &str) -> Result<Part<'a>> {
        let (name, entry) = self
            .entry
            .parts
            .get_key_value(name)
            .ok_or_else(|| Error::NotFound(format!("part {:?}/{name:?}", self.name)))?;
        Ok(Part {
            src: self.src,
            tensor: self.name,
            name,
            entry,
        })
    }

    /// The `"data"` part, which is what every dense tensor has and what the
    /// sugar below addresses.
    pub fn data(&self) -> Result<Part<'a>> {
        self.part("data")
    }

    /// Storage type of the `"data"` part.
    pub fn dtype(&self) -> Result<DType> {
        Ok(self.data()?.dtype())
    }

    /// Logical type of the `"data"` part, if it has one.
    pub fn logical(&self) -> Result<Option<&'a str>> {
        Ok(self.data()?.logical())
    }

    /// Decoded byte size of the `"data"` part.
    pub fn nbytes(&self) -> Result<u64> {
        Ok(self.data()?.nbytes())
    }

    /// Bytes of the `"data"` part: the best the source can do, saying which.
    pub fn bytes(&self) -> Result<Bytes<'a>> {
        self.data()?.bytes()
    }

    /// Borrowed bytes of the `"data"` part, or an error.
    pub fn map(&self) -> Result<&'a [u8]> {
        self.data()?.map()
    }

    /// Address of the `"data"` part's bytes.
    pub fn locate(&self) -> Result<Location> {
        self.data()?.locate()
    }

    pub fn caps(&self) -> Result<Caps> {
        Ok(self.data()?.caps())
    }

    pub fn verify(&self) -> Result<Verified> {
        self.data()?.verify()
    }

    /// Verifies every part of this tensor. Returns whether a digest was
    /// checked for all of them.
    pub fn verify_all(&self) -> Result<Verified> {
        let mut all = Verified::Digest;
        for name in self.parts() {
            if self.part(name)?.verify()? == Verified::NoDigest {
                all = Verified::NoDigest;
            }
        }
        Ok(all)
    }

    pub fn prefetch(&self) -> Result<()> {
        self.data()?.prefetch()
    }

    #[cfg(unix)]
    pub fn evict(&self) -> Result<()> {
        self.data()?.evict()
    }
}

/// One part of a tensor: the unit that has bytes.
#[derive(Debug, Clone, Copy)]
pub struct Part<'a> {
    src: &'a Source,
    tensor: &'a str,
    name: &'a str,
    entry: &'a PartEntry,
}

impl<'a> Part<'a> {
    pub fn name(&self) -> &'a str {
        self.name
    }

    pub fn dtype(&self) -> DType {
        self.entry.dtype
    }

    pub fn logical(&self) -> Option<&'a str> {
        self.entry.logical.as_deref()
    }

    pub fn digest(&self) -> Option<&'a str> {
        self.entry.digest.as_deref()
    }

    /// Decoded byte size.
    pub fn nbytes(&self) -> u64 {
        self.entry.payload.decoded_len()
    }

    pub fn payload(&self) -> &'a Payload {
        &self.entry.payload
    }

    /// The file these bytes live in.
    pub fn store(&self) -> &'a Store {
        self.src.store(self.entry.payload.store())
    }

    // ---- the four predicates, each the precondition of one operation ----

    fn addressable(&self) -> Option<Location> {
        self.entry.payload.location()
    }

    fn mappable(&self) -> Option<Location> {
        let at = self.addressable()?;
        self.src.store(at.store).is_mapped().then_some(at)
    }

    fn evictable(&self) -> Option<Location> {
        let at = self.mappable()?;
        self.src
            .store(at.store)
            .page_exclusive(at.offset, at.len)
            .then_some(at)
    }

    /// What can be done with these bytes. Each field runs the same predicate
    /// the matching method runs, so this is not a summary kept in step by
    /// hand.
    pub fn caps(&self) -> Caps {
        Caps {
            map: self.mappable().is_some(),
            locate: self.addressable().is_some(),
            evict: self.evictable().is_some(),
            verify: self.entry.digest.is_some(),
            alignment: self.addressable().map(|at| at.alignment()).unwrap_or(1),
        }
    }

    // ---- the three ways to get bytes ----

    /// The address of the decoded bytes: exactly this range of this file.
    ///
    /// Errors when the bytes are not one contiguous raw range, as with an
    /// encoded part or an archive entry. There is no address in that case
    /// that would give the caller the tensor.
    pub fn locate(&self) -> Result<Location> {
        self.addressable().ok_or_else(|| {
            Error::Unsupported(format!(
                "{}: {} has no address; its bytes are {}",
                self.label(),
                self.name,
                self.shape_of_payload()
            ))
        })
    }

    /// Borrowed bytes. Errors rather than copying.
    pub fn map(&self) -> Result<&'a [u8]> {
        let Some(at) = self.mappable() else {
            let detail = if self.addressable().is_some() {
                "its file was opened without mapping".to_string()
            } else {
                format!("its bytes are {}", self.shape_of_payload())
            };
            return Err(Error::Unsupported(format!(
                "{}: no zero-copy view; {detail}",
                self.label()
            )));
        };
        Ok(self
            .src
            .store(at.store)
            .slice(at.offset, at.len)?
            .expect("mappable stores are mapped"))
    }

    /// Decoded bytes, the best way this source can produce them.
    pub fn bytes(&self) -> Result<Bytes<'a>> {
        match &self.entry.payload {
            Payload::At(at) => {
                let store = self.src.store(at.store);
                match store.slice(at.offset, at.len)? {
                    Some(slice) => Ok(Bytes::Mapped(slice)),
                    None => Ok(Bytes::Owned(store.read(at.offset, at.len)?)),
                }
            }
            Payload::Encoded {
                at,
                encoding,
                decoded_len,
            } => {
                let stored = self.src.store(at.store).read(at.offset, at.len)?;
                let profile = self.src.vocab.encoding(encoding).ok_or_else(|| {
                    Error::Unsupported(format!(
                        "{}: encoding profile {encoding:?} is not registered",
                        self.label()
                    ))
                })?;
                Ok(Bytes::Owned(profile.decode(&stored, *decoded_len)?))
            }
            Payload::Opaque {
                store,
                key,
                decoded_len,
            } => {
                let store = self.src.store(*store);
                let opaque = store.opaque().ok_or_else(|| {
                    Error::Unsupported(format!(
                        "{}: opaque payload with no reader attached",
                        self.label()
                    ))
                })?;
                Ok(Bytes::Owned(opaque.read(*key, *decoded_len)?))
            }
        }
    }

    // ---- verification and paging ----

    /// Checks this part's digest (if it has one) and the content rules of its
    /// logical type (if it is registered).
    ///
    /// A mismatch comes back as `Err(Reject { rule: Digest, .. })`, since the
    /// file has failed a rule. `Ok(NoDigest)` means there was nothing to
    /// check.
    pub fn verify(&self) -> Result<Verified> {
        let entry = self
            .src
            .catalog
            .get(self.tensor)
            .expect("part handles come from a catalog entry");
        let elems = if entry.layout == "dense" {
            Some(entry.num_elements()?)
        } else {
            None
        };
        if self.entry.digest.is_none() && self.entry.logical.is_none() {
            return Ok(Verified::NoDigest);
        }
        // Digests and content rules cover decoded bytes (§3.4).
        let bytes = self.bytes()?;
        if let Some(logical) = &self.entry.logical {
            self.src.vocab.check_values(logical, &bytes, elems)?;
        }
        match &self.entry.digest {
            None => Ok(Verified::NoDigest),
            Some(digest) => {
                if !digest_matches(digest, &bytes)? {
                    return Err(Error::reject(
                        Rule::Digest,
                        format!("digest mismatch for {}", self.label()),
                    ));
                }
                Ok(Verified::Digest)
            }
        }
    }

    /// Hints the OS to prefetch these pages.
    pub fn prefetch(&self) -> Result<()> {
        let Some(at) = self.mappable() else {
            return Ok(());
        };
        self.src.store(at.store).prefetch(at.offset, at.len)
    }

    /// Drops these pages from the page cache.
    ///
    /// Requires page exclusivity, so this never touches a page another blob
    /// occupies, which is what per-tensor eviction depends on.
    #[cfg(unix)]
    pub fn evict(&self) -> Result<()> {
        let at = self.evictable().ok_or_else(|| {
            Error::Unsupported(format!(
                "{}: not evictable; {}",
                self.label(),
                if self.mappable().is_some() {
                    "it shares an OS page with another blob"
                } else {
                    "its bytes are not a mapped range"
                }
            ))
        })?;
        self.src.store(at.store).evict(at.offset, at.len)
    }

    fn label(&self) -> String {
        format!("{:?}/{:?}", self.tensor, self.name)
    }

    fn shape_of_payload(&self) -> &'static str {
        match &self.entry.payload {
            Payload::At(_) => "a raw range",
            Payload::Encoded { .. } => "stored under an encoding profile",
            Payload::Opaque { .. } => "produced by the format's own reader",
        }
    }
}

/// The identity of a `.zt` container: its size and whole-file digest. This is
/// exactly what [`Writer::add_shard`](crate::Writer::add_shard) records.
///
/// The frame is checked first, so a file that is not a container is an error
/// rather than a digest of something else.
pub fn shard_identity(path: impl AsRef<Path>) -> Result<Shard> {
    shard_identity_with(path, DigestAlgorithm::Xxh3)
}

/// The identity of a `.zt` container, digested with `algo`.
///
/// Use [`DigestAlgorithm::Sha256`] for anything that will be signed or
/// distributed: the root manifest then commits to every shard byte, so one
/// signature over the root covers the whole model (§6.5). `Xxh3` is faster and
/// is what a local, unsigned set wants.
pub fn shard_identity_with(path: impl AsRef<Path>, algo: DigestAlgorithm) -> Result<Shard> {
    let store = Store::index(path.as_ref(), "zt")?;
    validate::read(&store, &Vocabulary::shared())?;
    let mut hasher = Hasher::new(algo);
    let mut at = 0u64;
    while at < store.len() {
        let n = (store.len() - at).min(1 << 20);
        hasher.update(&store.read(at, n)?);
        at += n;
    }
    Ok(Shard {
        size: store.len(),
        digest: hasher.finish(),
    })
}
