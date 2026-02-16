#![cfg(feature = "hdf5")]

use std::io::Write;

use tempfile::NamedTempFile;

use ztensor::Hdf5Reader;

#[test]
fn hdf5_rejects_bad_magic() {
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(b"NOT_HDF5_MAGIC_BYTES").unwrap();
    tmp.flush().unwrap();
    assert!(Hdf5Reader::open(tmp.path()).is_err());
}

#[test]
fn hdf5_rejects_truncated_superblock() {
    let mut tmp = NamedTempFile::new().unwrap();
    // Real HDF5 signature but truncated
    tmp.write_all(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a])
        .unwrap();
    tmp.flush().unwrap();
    assert!(Hdf5Reader::open(tmp.path()).is_err());
}
