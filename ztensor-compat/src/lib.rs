//! Foreign tensor formats, projected into the zTensor object model.
//!
//! Every reader here implements [`ztensor::Source`]: it builds a
//! [`ztensor::Manifest`] view of the foreign metadata and serves reads
//! through the capability ladder. Projections are read-only and honest —
//! `caps()` reports what the foreign bytes actually support (usually
//! tier 2 at best), and nothing is ever silently reinterpreted or
//! dequantized.
//!
//! Note that a projected manifest reports the foreign file's *actual*
//! offsets; `.zt` guarantees (the 4096 floor, page exclusivity, digests)
//! hold only for `.zt` files. To upgrade a foreign checkpoint to tier 3,
//! convert it: `ztensor::Writer::ingest` copies any [`ztensor::Source`]
//! into a canonical `.zt` file.

#[cfg(feature = "safetensors")]
mod safetensors;
#[cfg(feature = "safetensors")]
pub use safetensors::Safetensors;
