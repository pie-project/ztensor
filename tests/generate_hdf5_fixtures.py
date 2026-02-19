"""Generate HDF5 test fixtures using h5py."""

import sys
import os
import numpy as np
import h5py


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)

    # contiguous_simple.h5 — single 1D f32 dataset
    with h5py.File(os.path.join(out_dir, "contiguous_simple.h5"), "w") as f:
        f.create_dataset("data", data=np.array([1, 2, 3, 4, 5], dtype=np.float32))

    # chunked_uncompressed.h5 — 2D f32, chunked, no compression
    with h5py.File(os.path.join(out_dir, "chunked_uncompressed.h5"), "w") as f:
        f.create_dataset(
            "weight",
            data=np.arange(24, dtype=np.float32).reshape(6, 4),
            chunks=(3, 2),
        )

    # chunked_gzip.h5 — 2D f32, chunked + gzip
    with h5py.File(os.path.join(out_dir, "chunked_gzip.h5"), "w") as f:
        f.create_dataset(
            "weight",
            data=np.arange(24, dtype=np.float32).reshape(6, 4),
            chunks=(3, 2),
            compression="gzip",
        )

    # chunked_shuffle_gzip.h5 — 2D f32, chunked + shuffle + gzip
    with h5py.File(os.path.join(out_dir, "chunked_shuffle_gzip.h5"), "w") as f:
        f.create_dataset(
            "weight",
            data=np.arange(24, dtype=np.float32).reshape(6, 4),
            chunks=(3, 2),
            shuffle=True,
            compression="gzip",
        )

    # chunked_1d.h5 — 1D f32, chunked
    with h5py.File(os.path.join(out_dir, "chunked_1d.h5"), "w") as f:
        f.create_dataset(
            "bias",
            data=np.arange(100, dtype=np.float32),
            chunks=(10,),
        )

    # mixed_layouts.h5 — contiguous + chunked+gzip in one file
    with h5py.File(os.path.join(out_dir, "mixed_layouts.h5"), "w") as f:
        f.create_dataset("contiguous", data=np.array([1, 2, 3], dtype=np.float32))
        f.create_dataset(
            "chunked",
            data=np.arange(12, dtype=np.float32).reshape(3, 4),
            chunks=(2, 2),
            compression="gzip",
        )

    # keras_like.h5 — nested groups mimicking Keras weight files
    with h5py.File(os.path.join(out_dir, "keras_like.h5"), "w") as f:
        g1 = f.create_group("dense_1")
        g1.create_dataset("kernel", data=np.ones((4, 3), dtype=np.float32))
        g1.create_dataset("bias", data=np.zeros(3, dtype=np.float32))
        g2 = f.create_group("dense_2")
        g2.create_dataset("kernel", data=np.full((3, 2), 0.5, dtype=np.float32))
        g2.create_dataset("bias", data=np.zeros(2, dtype=np.float32))


if __name__ == "__main__":
    main()
