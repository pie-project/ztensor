//! zTensor v2 — an aligned, verifiable container format for tensor data.
//!
//! The format is specified in `spec/ztensor-v2-spec.md` (Draft 2). The
//! design separates three layers:
//!
//! - **L0 container** (frozen): magic, 40-byte footer, aligned blob heap.
//! - **L1 manifest**: deterministic CBOR naming structure.
//! - **L2 vocabulary** (mortal, registry-managed): layouts, logical types,
//!   encodings.
//!
//! This crate is the core implementation: reading and writing `.zt` files
//! with full spec validation. Foreign formats (safetensors, gguf, ...) are
//! projections that live in the separate compat crate.
//!
//! ```no_run
//! use ztensor::{Writer, Reader, DType};
//!
//! let mut w = Writer::create("model.zt")?;
//! w.add_dense("weights", &[2, 2], DType::F32, &[0u8; 16])?;
//! w.finish()?;
//!
//! let r = Reader::open("model.zt")?;
//! let bytes = r.view("weights", "data")?; // zero-copy
//! # Ok::<(), ztensor::Error>(())
//! ```

pub mod cbor;
mod error;
mod models;
mod profiles;
mod reader;
mod shard;
mod source;
mod writer;

pub use error::{Error, Result, Rule};
pub use models::{
    logical_size, registered_dtype, BlobRef, DType, Layout, Manifest, Object, Part, Shard,
    ALIGN_CANONICAL, ALIGN_FLOOR, FOOTER_LEN, MAGIC, MAX_MANIFEST_LEN, MAX_NAME_LEN, MAX_RANK,
    VERSION,
};
pub use profiles::{encoding_profile, layout_profile, EncodingProfile, LayoutProfile};
pub use reader::{read_csr, validate_bytes, Csr, Reader};
pub use shard::{
    CasResolver, Composite, CompositeSource, Model, PositionalResolver, ShardResolver,
};
pub use source::{page_size, Caps, Source};
pub use writer::{DataShardWriter, ObjectWriter, PartDef, StreamPart, Writer};
