//! zTensor v2: an aligned, verifiable container format for tensor data.
//!
//! zTensor carries what can be *proved* about a checkpoint's bytes: where they
//! are ([`Part::locate`]), whether they are intact ([`Part::verify`]), whether
//! they can be dropped without disturbing a neighbour ([`Part::evict`]), and
//! which model they are ([`Manifest::content_digest`]).
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
//! let bytes = t.map()?;              // borrowed, or an error; never a copy
//! # Ok::<(), ztensor::Error>(())
//! ```
//!
//! # The modules are the spec's layers
//!
//! The format is specified in `spec/ztensor-v2-spec.md` (Draft 4), which
//! separates three layers, and this crate is laid out to match so that a spec
//! section and the code that implements it have the same address:
//!
//! - [`format`](mod@format) — **L0 container and L1 manifest**, frozen. The magic, the
//!   40-byte footer, the alignment floor, the manifest schema and its CBOR
//!   mapping, and the rules that decide conformance and canonical form.
//!   Nothing here opens a file.
//! - [`vocab`](mod@vocab) — **L2**, open and registry-managed: layouts, logical types and
//!   encodings, which another crate can extend and which are then validated
//!   exactly like the built-ins.
//! - [`read`](mod@read) — opening `.zt` and getting at bytes.
//! - [`write`](mod@write) — producing `.zt`.
//! - [`provide`](mod@provide) — the face turned towards a crate that projects a *foreign*
//!   format into a [`Source`]. Reading a checkpoint needs nothing from it.
//!
//! The names a consumer actually uses are re-exported at the crate root, and
//! only those: the format constants, the builder types you never have to name,
//! the shard resolvers and the projection machinery keep their module paths.
//!
//! # Getting bytes
//!
//! Three methods, one per intent: [`bytes`](Part::bytes) gives the best the
//! source can do and says which it gave, [`map`](Part::map) insists on a
//! borrow, and [`locate`](Part::locate) gives the address so the caller can do
//! the I/O itself. [`Caps`] reports per part what each will do, and whose
//! fields are named after those very methods.
//!
//! # Two layers of description
//!
//! [`Manifest`] is what one `.zt` file literally says. [`Catalog`](provide::Catalog)
//! is the resolved index a consumer queries, whose addresses are [`StoreId`]s
//! and which can therefore span files that never heard of each other. Foreign
//! projections build catalogs; only a `.zt` root has a manifest, which is what
//! [`Source::provenance`] reports.

mod error;
pub mod format;
pub mod provide;
pub mod read;
pub mod vocab;
pub mod write;

pub use error::{Error, Result, Rule};
pub use format::{DType, DigestAlgorithm, Manifest, Object, Shard};
pub use provide::{Location, Store, StoreId};
pub use read::{shard_identity, Bytes, Caps, Part, Provenance, Source, Tensor, Verified};
pub use vocab::Vocabulary;
pub use write::{Sink, Writer};
