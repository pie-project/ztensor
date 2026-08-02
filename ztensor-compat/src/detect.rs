//! Format detection: open any supported tensor file as a [`Source`].
//!
//! Detection is by magic bytes wherever the format has them; ONNX
//! (protobuf, magic-less) falls back to the file extension. Formats whose
//! feature is disabled are reported as such — never misdetected as
//! something else.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use ztensor::{Composite, CompositeSource, Error, Result, Source};

/// Sniffs the format of a tensor file. Returns a stable label:
/// `"zt"`, `"safetensors"`, `"gguf"`, `"npz"`, `"pt"`, `"hdf5"`, `"onnx"`.
pub fn detect(path: impl AsRef<Path>) -> Result<&'static str> {
    let path = path.as_ref();
    // A single read() may return fewer bytes than asked for; fill the
    // buffer so a short read cannot cause a mis-detection.
    let mut file = File::open(path)?;
    let mut head = [0u8; 9];
    let mut n = 0;
    while n < head.len() {
        match file.read(&mut head[n..])? {
            0 => break,
            got => n += got,
        }
    }
    let head = &head[..n];

    if head.len() >= 8 && head[..8] == ztensor::MAGIC {
        return Ok("zt");
    }
    if head.starts_with(b"GGUF") {
        return Ok("gguf");
    }
    if head.len() >= 8 && &head[..8] == b"\x89HDF\r\n\x1a\n" {
        return Ok("hdf5");
    }
    if head.starts_with(b"PK\x03\x04") {
        #[cfg(any(feature = "pickle", feature = "npz"))]
        {
            let is_pt = zip::ZipArchive::new(File::open(path)?)
                .ok()
                .map(|z| z.file_names().any(|n| n.ends_with("data.pkl")))
                .unwrap_or(false);
            return Ok(if is_pt { "pt" } else { "npz" });
        }
        #[cfg(not(any(feature = "pickle", feature = "npz")))]
        return Err(Error::Unsupported(
            "zip-container formats (.pt/.npz) are not compiled in".into(),
        ));
    }
    if head.len() >= 9 && head[8] == b'{' {
        let header_len = u64::from_le_bytes(head[..8].try_into().unwrap());
        if header_len > 0 && header_len < (100 << 20) {
            return Ok("safetensors");
        }
    }
    if path.extension().is_some_and(|e| e == "onnx") {
        return Ok("onnx");
    }
    Err(Error::Unsupported(format!(
        "cannot detect the format of {}",
        path.display()
    )))
}

/// Opens a tensor file of any supported format. `.zt` files (including
/// sharded models) open through [`ztensor::Model`].
pub fn open_any(path: impl AsRef<Path>) -> Result<Box<dyn Source>> {
    let path = path.as_ref();

    #[allow(dead_code)]
    fn unsupported(what: &str) -> Result<Box<dyn Source>> {
        Err(Error::Unsupported(format!(
            "{what} support is not compiled in (enable the ztensor-compat feature)"
        )))
    }

    match detect(path)? {
        "zt" => Ok(Box::new(ztensor::Model::open(path)?)),
        "gguf" => {
            #[cfg(feature = "gguf")]
            return Ok(Box::new(crate::Gguf::open(path)?));
            #[cfg(not(feature = "gguf"))]
            unsupported("gguf")
        }
        "hdf5" => {
            #[cfg(feature = "hdf5")]
            return Ok(Box::new(crate::Hdf5::open(path)?));
            #[cfg(not(feature = "hdf5"))]
            unsupported("hdf5")
        }
        "pt" => {
            #[cfg(feature = "pickle")]
            return Ok(Box::new(crate::Pt::open(path)?));
            #[cfg(not(feature = "pickle"))]
            unsupported("pickle (.pt)")
        }
        "npz" => {
            #[cfg(feature = "npz")]
            return Ok(Box::new(crate::Npz::open(path)?));
            #[cfg(not(feature = "npz"))]
            unsupported("npz")
        }
        "safetensors" => {
            #[cfg(feature = "safetensors")]
            return Ok(Box::new(crate::Safetensors::open(path)?));
            #[cfg(not(feature = "safetensors"))]
            unsupported("safetensors")
        }
        "onnx" => {
            #[cfg(feature = "onnx")]
            return Ok(Box::new(crate::Onnx::open(path)?));
            #[cfg(not(feature = "onnx"))]
            unsupported("onnx")
        }
        other => Err(Error::Unsupported(format!("unhandled format {other:?}"))),
    }
}

/// Opens several files as one [`Composite`].
///
/// What a sharded snapshot is: `model-00001-of-00003.safetensors` and its
/// siblings are each a whole file that describes itself, and the index beside
/// them is a naming convention outside the format. So this is a list of paths
/// and nothing more — the caller decides which files belong together, because
/// the files themselves never said.
///
/// Every path is opened with [`open_any`], so a set may mix formats. Nothing
/// here requires them to match: what makes the set a model is that the names
/// do not collide, which [`Composite::new`] checks.
pub fn open_all<P: AsRef<Path>>(paths: &[P]) -> Result<Composite> {
    let mut parts = Vec::with_capacity(paths.len());
    for path in paths {
        let path = path.as_ref();
        parts.push(CompositeSource {
            label: path.display().to_string(),
            source: open_any(path)?,
        });
    }
    Composite::new(parts)
}
