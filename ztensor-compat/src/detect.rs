//! Format detection: open any supported tensor file as a [`Source`].
//!
//! Detection is by magic bytes wherever the format has them; ONNX
//! (protobuf, magic-less) falls back to the file extension. Formats whose
//! feature is disabled are reported as such — never misdetected as
//! something else.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use ztensor::{Error, Result, Source};

/// Opens a tensor file of any supported format, sniffing the format from
/// its content. `.zt` files (including sharded models) open through
/// [`ztensor::Model`].
pub fn open_any(path: impl AsRef<Path>) -> Result<Box<dyn Source>> {
    let path = path.as_ref();
    let mut head = [0u8; 9];
    let n = File::open(path)?.read(&mut head)?;
    let head = &head[..n];

    #[allow(dead_code)]
    fn unsupported(what: &str) -> Result<Box<dyn Source>> {
        Err(Error::Unsupported(format!(
            "{what} support is not compiled in (enable the ztensor-compat feature)"
        )))
    }

    // .zt v2
    if head.len() >= 8 && head[..8] == ztensor::MAGIC {
        return Ok(Box::new(ztensor::Model::open(path)?));
    }
    // gguf
    if head.starts_with(b"GGUF") {
        #[cfg(feature = "gguf")]
        return Ok(Box::new(crate::Gguf::open(path)?));
        #[cfg(not(feature = "gguf"))]
        return unsupported("gguf");
    }
    // hdf5
    if head.len() >= 8 && &head[..8] == b"\x89HDF\r\n\x1a\n" {
        #[cfg(feature = "hdf5")]
        return Ok(Box::new(crate::Hdf5::open(path)?));
        #[cfg(not(feature = "hdf5"))]
        return unsupported("hdf5");
    }
    // ZIP container: torch .pt or numpy .npz
    if head.starts_with(b"PK\x03\x04") {
        #[cfg(any(feature = "pickle", feature = "npz"))]
        {
            let is_pt = zip::ZipArchive::new(File::open(path)?)
                .ok()
                .map(|z| z.file_names().any(|n| n.ends_with("data.pkl")))
                .unwrap_or(false);
            if is_pt {
                #[cfg(feature = "pickle")]
                return Ok(Box::new(crate::Pt::open(path)?));
                #[cfg(not(feature = "pickle"))]
                return unsupported("pickle (.pt)");
            }
            #[cfg(feature = "npz")]
            return Ok(Box::new(crate::Npz::open(path)?));
            #[cfg(not(feature = "npz"))]
            return unsupported("npz");
        }
        #[cfg(not(any(feature = "pickle", feature = "npz")))]
        return unsupported("zip-container formats (.pt/.npz)");
    }
    // safetensors: u64 LE header length followed by a JSON object
    if head.len() >= 9 && head[8] == b'{' {
        let header_len = u64::from_le_bytes(head[..8].try_into().unwrap());
        if header_len > 0 && header_len < (100 << 20) {
            #[cfg(feature = "safetensors")]
            return Ok(Box::new(crate::Safetensors::open(path)?));
            #[cfg(not(feature = "safetensors"))]
            return unsupported("safetensors");
        }
    }
    // onnx: no magic; go by extension
    if path.extension().is_some_and(|e| e == "onnx") {
        #[cfg(feature = "onnx")]
        return Ok(Box::new(crate::Onnx::open(path)?));
        #[cfg(not(feature = "onnx"))]
        return unsupported("onnx");
    }

    Err(Error::Unsupported(format!(
        "cannot detect the format of {}",
        path.display()
    )))
}
