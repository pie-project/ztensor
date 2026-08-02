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

#[cfg(feature = "gguf")]
mod gguf;
#[cfg(feature = "gguf")]
pub use gguf::Gguf;

#[cfg(feature = "npz")]
mod npz;
#[cfg(feature = "npz")]
pub use npz::Npz;

#[cfg(feature = "pickle")]
mod pt;
#[cfg(feature = "pickle")]
pub use pt::Pt;

#[cfg(feature = "hdf5")]
mod hdf5;
#[cfg(feature = "hdf5")]
pub use hdf5::Hdf5;

#[cfg(feature = "onnx")]
mod onnx;
#[cfg(feature = "onnx")]
pub use onnx::Onnx;

mod detect;
pub use detect::{detect, open_any};
