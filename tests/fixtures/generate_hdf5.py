#!/usr/bin/env python3
"""Generate HDF5 test fixtures for ztensor integration tests."""

import numpy as np
import h5py
import os

DIR = os.path.dirname(os.path.abspath(__file__))


def create_chunked_uncompressed():
    """Chunked layout, no compression."""
    path = os.path.join(DIR, "chunked_uncompressed.h5")
    data = np.arange(24, dtype=np.float32).reshape(6, 4)
    with h5py.File(path, "w") as f:
        f.create_dataset("weight", data=data, chunks=(3, 2))
    print(f"  {path}")


def create_chunked_gzip():
    """Chunked layout with gzip (deflate) compression."""
    path = os.path.join(DIR, "chunked_gzip.h5")
    data = np.arange(24, dtype=np.float32).reshape(6, 4)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "weight", data=data, chunks=(3, 2), compression="gzip", compression_opts=6
        )
    print(f"  {path}")


def create_chunked_shuffle_gzip():
    """Chunked layout with shuffle + gzip filters."""
    path = os.path.join(DIR, "chunked_shuffle_gzip.h5")
    data = np.arange(24, dtype=np.float32).reshape(6, 4)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "weight", data=data, chunks=(3, 2), compression="gzip", shuffle=True
        )
    print(f"  {path}")


def create_chunked_1d():
    """1D chunked dataset with gzip."""
    path = os.path.join(DIR, "chunked_1d.h5")
    data = np.arange(100, dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("bias", data=data, chunks=(32,), compression="gzip")
    print(f"  {path}")


def create_mixed_layouts():
    """File with both contiguous and chunked datasets."""
    path = os.path.join(DIR, "mixed_layouts.h5")
    with h5py.File(path, "w") as f:
        # Contiguous (no chunks, no compression)
        f.create_dataset("contiguous", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        # Chunked + gzip
        f.create_dataset(
            "chunked",
            data=np.arange(12, dtype=np.float32).reshape(3, 4),
            chunks=(2, 2),
            compression="gzip",
        )
    print(f"  {path}")


def create_contiguous_simple():
    """Simple contiguous dataset (regression test)."""
    path = os.path.join(DIR, "contiguous_simple.h5")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
    print(f"  {path}")


def create_keras_like():
    """Nested group structure like Keras model weights."""
    path = os.path.join(DIR, "keras_like.h5")
    with h5py.File(path, "w") as f:
        g1 = f.create_group("dense_1")
        g1.create_dataset(
            "kernel",
            data=np.ones((4, 3), dtype=np.float32),
            chunks=(4, 3),
            compression="gzip",
        )
        g1.create_dataset(
            "bias", data=np.zeros(3, dtype=np.float32), chunks=(3,), compression="gzip"
        )
        g2 = f.create_group("dense_2")
        g2.create_dataset(
            "kernel",
            data=np.ones((3, 2), dtype=np.float32) * 0.5,
            chunks=(3, 2),
            compression="gzip",
        )
        g2.create_dataset(
            "bias", data=np.zeros(2, dtype=np.float32), chunks=(2,), compression="gzip"
        )
    print(f"  {path}")


if __name__ == "__main__":
    print("Generating HDF5 test fixtures:")
    create_chunked_uncompressed()
    create_chunked_gzip()
    create_chunked_shuffle_gzip()
    create_chunked_1d()
    create_mixed_layouts()
    create_contiguous_simple()
    create_keras_like()
    print("Done.")
