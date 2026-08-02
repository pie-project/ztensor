//! The resolved index: names to addresses.
//!
//! A [`Catalog`] is what a consumer queries, and it is deliberately not a
//! [`Manifest`](crate::schema::Manifest). A manifest is one file's own claim,
//! addressed through that file's shard table. A catalog is process-local: its
//! addresses are [`StoreId`]s, so it can span files that never heard of each
//! other — a sharded snapshot, a mixed set, a single foreign file — without
//! anyone having to claim an identity nobody wrote down.
//!
//! Every projection in the compat crate produces one of these. None of them
//! produces a manifest, because none of them ever had one.

use std::collections::BTreeMap;

use crate::cbor::Value;
use crate::error::{Error, Result, Rule};
use crate::schema::DType;
use crate::store::StoreId;

/// Where a part's decoded bytes are: a range of one store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Location {
    pub store: StoreId,
    pub offset: u64,
    pub len: u64,
}

impl Location {
    /// Largest power of two dividing the offset. The pointer alignment of a
    /// whole-file mapping is `min(alignment, page_size)`.
    pub fn alignment(&self) -> u64 {
        if self.offset == 0 {
            // Offset zero divides by everything; report the page size rather
            // than a meaningless 2^63.
            return crate::store::page_size();
        }
        1u64 << self.offset.trailing_zeros().min(63)
    }
}

/// How a part's bytes can be reached.
///
/// The three cases are the whole of what a source can honestly offer, and the
/// capability report is a direct reading of which one this is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    /// Raw decoded bytes, exactly at this range. Addressable and mappable.
    At(Location),
    /// Stored at this range under an encoding profile. The range is *not* the
    /// tensor, so it is not an address a consumer can read directly.
    Encoded {
        at: Location,
        encoding: String,
        decoded_len: u64,
    },
    /// Only the projection that opened the file can produce these bytes — a
    /// deflated archive entry, a chunked dataset. Readable, nothing more.
    Opaque {
        store: StoreId,
        key: u64,
        decoded_len: u64,
    },
}

impl Payload {
    /// The address of the decoded bytes, when there is one.
    pub fn location(&self) -> Option<Location> {
        match self {
            Payload::At(at) => Some(*at),
            _ => None,
        }
    }

    /// The store holding these bytes, however they are reached.
    pub fn store(&self) -> StoreId {
        match self {
            Payload::At(at) | Payload::Encoded { at, .. } => at.store,
            Payload::Opaque { store, .. } => *store,
        }
    }

    /// Decoded byte size.
    pub fn decoded_len(&self) -> u64 {
        match self {
            Payload::At(at) => at.len,
            Payload::Encoded { decoded_len, .. } | Payload::Opaque { decoded_len, .. } => {
                *decoded_len
            }
        }
    }
}

/// One part of a tensor: bytes plus their interpretation.
#[derive(Debug, Clone, PartialEq)]
pub struct PartEntry {
    pub dtype: DType,
    /// Logical type id; `None` means the logical type equals `dtype`.
    pub logical: Option<String>,
    pub payload: Payload,
    /// `"<algorithm>:<lowercase hex>"` over decoded bytes, when the format
    /// carries one. Most foreign formats do not.
    pub digest: Option<String>,
}

/// One named tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub shape: Vec<u64>,
    /// Layout profile id — `"dense"` for anything with a single `"data"` part.
    pub layout: String,
    pub attributes: Option<Value>,
    pub parts: BTreeMap<String, PartEntry>,
}

impl Entry {
    /// A single-part dense tensor, which is what most formats have.
    pub fn dense(shape: Vec<u64>, dtype: DType, logical: Option<String>, at: Location) -> Self {
        let mut parts = BTreeMap::new();
        parts.insert(
            "data".to_string(),
            PartEntry {
                dtype,
                logical,
                payload: Payload::At(at),
                digest: None,
            },
        );
        Entry {
            shape,
            layout: "dense".to_string(),
            attributes: None,
            parts,
        }
    }

    /// Element count: product of dimensions; empty shape is a scalar (1).
    pub fn num_elements(&self) -> Result<u64> {
        self.shape.iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d)
                .ok_or_else(|| Error::reject(Rule::Shape, "shape product overflows u64"))
        })
    }

    pub fn part(&self, name: &str) -> Result<&PartEntry> {
        self.parts
            .get(name)
            .ok_or_else(|| Error::NotFound(format!("part {name:?}")))
    }

    /// The store this tensor's bytes live in, taken from its first part.
    /// Every part of one tensor comes from one file in every format we read.
    pub(crate) fn store(&self) -> Option<StoreId> {
        self.parts.values().next().map(|p| p.payload.store())
    }
}

/// Names to entries, sorted, with the file-level attributes of whatever was
/// opened.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    entries: BTreeMap<String, Entry>,
    attributes: Option<Value>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a tensor, returning the entry it displaced (as `BTreeMap` does).
    /// Callers that must refuse collisions check [`get`](Self::get) first, so
    /// they can name both files in the error.
    pub fn insert(&mut self, name: impl Into<String>, entry: Entry) -> Option<Entry> {
        self.entries.insert(name.into(), entry)
    }

    pub fn set_attributes(&mut self, attributes: Option<Value>) {
        self.attributes = attributes;
    }

    pub fn attributes(&self) -> Option<&Value> {
        self.attributes.as_ref()
    }

    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Entry)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    pub(crate) fn into_iter_sorted(self) -> impl Iterator<Item = (String, Entry)> {
        self.entries.into_iter()
    }

    /// Rewrites every store id through `f`. Used when a catalog built against
    /// one file is folded into a source holding several.
    pub(crate) fn rebase(&mut self, f: impl Fn(StoreId) -> StoreId) {
        for entry in self.entries.values_mut() {
            for part in entry.parts.values_mut() {
                match &mut part.payload {
                    Payload::At(at) => at.store = f(at.store),
                    Payload::Encoded { at, .. } => at.store = f(at.store),
                    Payload::Opaque { store, .. } => *store = f(*store),
                }
            }
        }
    }
}
