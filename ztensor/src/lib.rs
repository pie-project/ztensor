//! zTensor v2 — an aligned, verifiable container format for tensor data.
//!
//! The format is specified in `spec/ztensor-v2-spec.md` (Draft 2), which
//! separates three layers:
//!
//! - **L0 container** (frozen): magic, 40-byte footer, aligned blob heap.
//! - **L1 manifest**: deterministic CBOR naming structure.
//! - **L2 vocabulary** (mortal, registry-managed): layouts, logical types,
//!   encodings.
//!
//! This crate reads and writes `.zt` with full spec validation. Foreign
//! formats (safetensors, gguf, ...) are projections that live in the separate
//! compat crate and arrive here as ordinary [`Source`]s.
//!
//! ```no_run
//! use ztensor::{DType, Source, Writer};
//!
//! let mut w = Writer::create("model.zt")?;
//! w.add("weights", [2u64, 2], DType::F32, &[0u8; 16])?;
//! w.finish()?;
//!
//! let src = Source::open("model.zt")?;
//! let t = src.tensor("weights")?;
//! let bytes = t.map()?;              // borrowed, or an error — never a hidden copy
//! # Ok::<(), ztensor::Error>(())
//! ```
//!
//! # Getting bytes
//!
//! Three methods, one per intent: [`bytes`](Part::bytes) gives the best the
//! source can do and says which it gave, [`map`](Part::map) insists on a
//! borrow, and [`locate`](Part::locate) gives the address so the caller can do
//! the I/O itself. What each will do is reported, per part, by [`Caps`] —
//! whose fields are named after those very methods.
//!
//! # Two layers of description
//!
//! [`schema::Manifest`] is what one `.zt` file literally says. [`Catalog`] is
//! the resolved index a consumer queries, whose addresses are [`StoreId`]s and
//! which can therefore span files that never heard of each other. Foreign
//! projections build catalogs; only a `.zt` root has a manifest.

pub mod catalog;
pub mod cbor;
pub mod csr;
mod error;
pub mod schema;
pub mod source;
pub mod store;
pub mod validate;
pub mod vocab;
pub mod writer;

pub use catalog::{Catalog, Entry, Location, PartEntry, Payload};
pub use error::{Error, Result, Rule};
pub use schema::{
    BlobRef, DType, Manifest, Object, Shard, ALIGN_CANONICAL, ALIGN_FLOOR, FOOTER_LEN, MAGIC,
    MAX_MANIFEST_LEN, MAX_NAME_LEN, MAX_RANK, MIN_FILE_LEN, VERSION,
};
pub use source::{
    shard_identity, Bytes, Caps, CasResolver, Part, PositionalResolver, ShardResolver, Source,
    Tensor, Verified,
};
pub use store::{page_size, Opaque, Store, StoreId};
pub use validate::{image as validate_bytes, manifest_of};
pub use vocab::Vocabulary;
pub use writer::{DataShardWriter, ObjectBuilder, Sink, Writer};
