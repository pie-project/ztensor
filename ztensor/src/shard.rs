//! Multi-file models (spec §7): shard resolution and the [`Model`] type.
//!
//! The manifest stores shard **identity** (size + digest), never location;
//! a [`ShardResolver`] turns an identity into bytes. Verification is a
//! ladder, cheapest first: footer magic/version and exact file size at
//! open, whole-file digest on demand ([`Model::verify_shards`]).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result, Rule};
use crate::models::{parse_xxh3, Manifest, Object, Part, Shard};
use crate::reader::{check_container, decode_part, read_csr, verify, Csr, Reader};
use crate::source::{Caps, Source};

/// Resolves a shard index + identity to a file path. The format itself
/// contains no names or paths — resolution is entirely the transport's
/// concern (spec §7.1, Appendix D).
pub trait ShardResolver {
    fn resolve(&self, index: u64, shard: &Shard) -> Result<PathBuf>;
}

/// Closures are resolvers: `|index, shard| Ok(path)`.
impl<F: Fn(u64, &Shard) -> Result<PathBuf>> ShardResolver for F {
    fn resolve(&self, index: u64, shard: &Shard) -> Result<PathBuf> {
        self(index, shard)
    }
}

/// The positional convention (Appendix D): root `<dir>/<stem>.zt` maps
/// shard `k` to `<dir>/<stem>-<k as 5 digits>.zt`.
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
    fn resolve(&self, index: u64, _shard: &Shard) -> Result<PathBuf> {
        Ok(self.dir.join(format!("{}-{index:05}.zt", self.stem)))
    }
}

/// The content-addressed convention (Appendix D): a shard with digest
/// `algo:hex` lives at `<store>/blobs/<algo>/<hex>`.
pub struct CasResolver {
    pub store: PathBuf,
}

impl ShardResolver for CasResolver {
    fn resolve(&self, _index: u64, shard: &Shard) -> Result<PathBuf> {
        let (algo, hex) = shard.digest.split_once(':').unwrap_or(("", ""));
        Ok(self.store.join("blobs").join(algo).join(hex))
    }
}

/// A root file plus its resolved shards — the general opener: a
/// single-file model is simply the `N = 0` case.
pub struct Model {
    root: Reader,
    shards: BTreeMap<u64, Mmap>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("root", &self.root)
            .field("shards", &self.shards.len())
            .finish()
    }
}

impl Model {
    /// Opens a model using the positional resolver.
    pub fn open(root_path: impl AsRef<Path>) -> Result<Self> {
        let resolver = PositionalResolver::for_root(&root_path);
        Self::open_with(root_path, &resolver)
    }

    /// Opens a model with a custom resolver. Every shard is checked
    /// against the two cheap rungs of the verification ladder: exact file
    /// size, and footer magic + version. Digests are checked only by
    /// [`Model::verify_shards`].
    pub fn open_with(
        root_path: impl AsRef<Path>,
        resolver: &dyn ShardResolver,
    ) -> Result<Self> {
        let root = Reader::open(&root_path)?;
        let mut shards = BTreeMap::new();
        for (&index, shard) in &root.manifest().shards {
            let path = resolver.resolve(index, shard)?;
            let file = File::open(&path)?;
            let len = file.metadata()?.len();
            if len != shard.size {
                return Err(Error::reject(
                    Rule::ShardIdentity,
                    format!(
                        "shard {index}: file is {len} bytes, root expects {}",
                        shard.size
                    ),
                ));
            }
            // SAFETY: read-only shared map of untrusted bytes.
            let mmap = unsafe { Mmap::map(&file)? };
            check_container(&mmap).map_err(|e| {
                Error::reject(Rule::ShardIdentity, format!("shard {index}: {e}"))
            })?;
            shards.insert(index, mmap);
        }
        Ok(Self { root, shards })
    }

    /// The root file's reader.
    pub fn root(&self) -> &Reader {
        &self.root
    }

    pub fn manifest(&self) -> &Manifest {
        self.root.manifest()
    }

    /// Tier 0: iterate objects and their metadata.
    pub fn objects(&self) -> impl Iterator<Item = (&str, &Object)> {
        self.root.objects()
    }

    pub fn get(&self, name: &str) -> Option<&Object> {
        self.root.get(name)
    }

    /// Deep verification (ladder rung 3): whole-file digest of every shard
    /// against the root's shard table.
    pub fn verify_shards(&self) -> Result<()> {
        for (&index, shard) in &self.root.manifest().shards {
            let bytes: &[u8] = &self.shards[&index];
            if xxh3_64(bytes) != parse_xxh3(&shard.digest)? {
                return Err(Error::reject(
                    Rule::ShardIdentity,
                    format!("shard {index}: digest mismatch"),
                ));
            }
        }
        Ok(())
    }

    fn part(&self, name: &str, part: &str) -> Result<&Part> {
        self.root.manifest().part(name, part)
    }

    /// The stored (possibly encoded) bytes of a part, wherever it lives.
    /// Bounds were validated against each shard's declared size at open,
    /// and the actual file size was checked to match.
    fn stored(&self, p: &Part) -> &[u8] {
        if p.blob.shard == 0 {
            self.root.stored_slice(p)
        } else {
            let mmap = &self.shards[&p.blob.shard];
            let start = p.blob.offset as usize;
            &mmap[start..start + p.blob.length as usize]
        }
    }

    /// Tier 2: zero-copy view of decoded bytes, from the root or any shard.
    pub fn view(&self, name: &str, part: &str) -> Result<&[u8]> {
        let p = self.part(name, part)?;
        if let Some(enc) = &p.encoding {
            return Err(Error::Unsupported(format!(
                "encoded part (encoding {enc:?}) has no zero-copy view"
            )));
        }
        Ok(self.stored(p))
    }

    /// Tier 1: owned decoded bytes, from the root or any shard.
    pub fn read(&self, name: &str, part: &str) -> Result<Vec<u8>> {
        let p = self.part(name, part)?;
        decode_part(p, self.stored(p))
    }

    /// Capability report. Foreign-shard parts are never reported
    /// page-exclusive: the root only knows its own references into a
    /// shard, so exclusivity cannot be proven.
    pub fn caps(&self, name: &str, part: &str) -> Result<Caps> {
        let p = self.part(name, part)?;
        if p.blob.shard == 0 {
            return self.root.caps(name, part);
        }
        Ok(Caps::for_part(p, p.encoding.is_none(), false))
    }

    /// Verifies a part's digest and logical-type content rules.
    /// See [`ztensor::verify`](crate::verify).
    pub fn verify(&self, name: &str, part: &str) -> Result<bool> {
        verify(self, name, part)
    }

    /// Hints the OS to prefetch a part's pages, wherever the part lives.
    #[cfg(unix)]
    pub fn prefetch(&self, name: &str, part: &str) -> Result<()> {
        let p = self.part(name, part)?;
        if p.blob.length == 0 {
            return Ok(());
        }
        if p.blob.shard == 0 {
            return self.root.prefetch(name, part);
        }
        self.shards[&p.blob.shard].advise_range(
            memmap2::Advice::WillNeed,
            p.blob.offset as usize,
            p.blob.length as usize,
        )?;
        Ok(())
    }

    /// Reads and assembles a `zt.sparse_csr/1` object across shards.
    pub fn read_csr(&self, name: &str) -> Result<Csr> {
        read_csr(self, name)
    }

    /// Per-tensor eviction; root-local parts only (see [`Reader::evict`]).
    #[cfg(unix)]
    pub fn evict(&self, name: &str, part: &str) -> Result<()> {
        if self.part(name, part)?.blob.shard != 0 {
            return Err(Error::Unsupported(
                "eviction of foreign-shard parts is not supported".into(),
            ));
        }
        self.root.evict(name, part)
    }
}

impl Source for Model {
    fn manifest(&self) -> &Manifest {
        self.root.manifest()
    }

    fn read(&self, object: &str, part: &str) -> Result<Vec<u8>> {
        Model::read(self, object, part)
    }

    fn view(&self, object: &str, part: &str) -> Result<&[u8]> {
        Model::view(self, object, part)
    }

    fn caps(&self, object: &str, part: &str) -> Result<Caps> {
        Model::caps(self, object, part)
    }
}
