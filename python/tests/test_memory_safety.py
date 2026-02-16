"""
Memory safety tests for the ztensor Python-Rust zero-copy boundary.

The zero-copy path works as follows:
  1. PyReader holds a TensorReader (backed by mmap)
  2. read_tensor_impl extracts a raw ptr/len from the mmap
  3. bytes_to_numpy_borrowed creates a numpy array via PyArray1::borrow_from_array,
     passing the PyReader as the numpy "base" object
  4. numpy's refcount on the base prevents GC of the PyReader (and its mmap)

These tests verify:
  - Base chain integrity: zero-copy arrays reference the Reader, copies don't
  - Derived views maintain the chain (slices, reshapes)
  - COW semantics: writes to zero-copy arrays don't corrupt the file
  - File deletion while arrays are alive (Linux holds the inode)
  - Mixing single-tensor reads with bulk loads
  - RSS-based leak detection for mmap memory (tracemalloc can't see mmap)
"""

import gc
import os
import resource
import tempfile

import numpy as np
import pytest

from ztensor import Reader, Writer

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _get_rss_kb():
    """Get current RSS in KB via getrusage (works on Linux/macOS)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _walk_base_chain(arr):
    """Walk numpy's base chain, yielding each base object."""
    obj = arr
    while hasattr(obj, "base") and obj.base is not None:
        obj = obj.base
        yield obj


@pytest.fixture
def zt_file():
    """Creates a .zt file with known tensor data for testing."""
    fd, path = tempfile.mkstemp(suffix=".zt")
    os.close(fd)

    tensors = {
        "small": np.arange(100, dtype=np.float32),
        "matrix": np.random.randn(64, 128).astype(np.float32),
        "vector": np.linspace(-1.0, 1.0, 256, dtype=np.float32),
    }

    with Writer(path) as w:
        for name, arr in tensors.items():
            w.add(name, arr)

    yield path, tensors

    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def large_zt_file():
    """Creates a larger .zt file (~4MB) to exercise page fault behavior."""
    fd, path = tempfile.mkstemp(suffix=".zt")
    os.close(fd)

    tensors = {
        "big": np.random.randn(512, 512).astype(np.float32),  # 1MB
        "bigger": np.random.randn(256, 1024).astype(np.float32),  # 1MB
        "biggest": np.random.randn(1024, 512).astype(np.float32),  # 2MB
    }

    with Writer(path) as w:
        for name, arr in tensors.items():
            w.add(name, arr)

    yield path, tensors

    if os.path.exists(path):
        os.remove(path)


# ============================================================================
# BASE CHAIN INTEGRITY
# ============================================================================


class TestBaseChain:
    """Verify the numpy base chain is correctly set up."""

    def test_zerocopy_base_chain_contains_reader(self, zt_file):
        """Zero-copy array's base chain must terminate at the Reader."""
        path, _ = zt_file

        reader = Reader(path)
        arr = reader["small"]

        # Walk the chain: reshaped → viewed → raw u8 (base=Reader)
        found_reader = any(isinstance(b, Reader) for b in _walk_base_chain(arr))
        assert found_reader, (
            f"Reader not found in base chain. "
            f"Chain types: {[type(b).__name__ for b in _walk_base_chain(arr)]}"
        )

    def test_copy_true_base_chain_excludes_reader(self, zt_file):
        """copy=True arrays must NOT have the Reader in their base chain."""
        path, _ = zt_file

        reader = Reader(path)
        tensors = reader.load_numpy(copy=True)

        for name, arr in tensors.items():
            found_reader = any(isinstance(b, Reader) for b in _walk_base_chain(arr))
            assert (
                not found_reader
            ), f"copy=True array '{name}' has Reader in base chain"

    def test_load_numpy_zerocopy_base_chain(self, zt_file):
        """load_numpy(copy=False) arrays must all have the Reader in their base chain."""
        path, _ = zt_file

        reader = Reader(path)
        tensors = reader.load_numpy(copy=False)

        for name, arr in tensors.items():
            found_reader = any(isinstance(b, Reader) for b in _walk_base_chain(arr))
            assert (
                found_reader
            ), f"copy=False array '{name}' missing Reader in base chain"

    def test_view_preserves_base_chain(self, zt_file):
        """Slicing a zero-copy array should preserve the base chain to the Reader."""
        path, expected = zt_file

        reader = Reader(path)
        arr = reader["matrix"]
        view = arr[10:20, ::2]  # strided slice
        view2 = view[:, :32]  # nested slice

        del reader
        del arr
        gc.collect()

        # The doubly-derived view should still be valid
        np.testing.assert_array_equal(view2, expected["matrix"][10:20, ::2][:, :32])

    def test_transpose_preserves_base_chain(self, zt_file):
        """Transposing a zero-copy array should preserve the base chain."""
        path, expected = zt_file

        reader = Reader(path)
        arr = reader["matrix"]
        transposed = arr.T

        del reader
        del arr
        gc.collect()

        np.testing.assert_array_equal(transposed, expected["matrix"].T)


# ============================================================================
# LIFETIME (use-after-free regression tests)
# ============================================================================


class TestLifetime:
    """
    Verify data survives after Reader reference is dropped.

    Note: these pass because the base chain keeps the Reader alive.
    If the base chain were broken, these would segfault — which is the
    regression we're guarding against.
    """

    def test_single_array_survives(self, zt_file):
        """Single zero-copy array survives del reader."""
        path, expected = zt_file

        reader = Reader(path)
        arr = reader["small"]
        del reader
        gc.collect()

        np.testing.assert_array_equal(arr, expected["small"])

    def test_bulk_load_survives(self, zt_file):
        """All arrays from load_numpy(copy=False) survive del reader."""
        path, expected = zt_file

        reader = Reader(path)
        tensors = reader.load_numpy(copy=False)
        del reader
        gc.collect()

        for name, exp in expected.items():
            np.testing.assert_array_equal(tensors[name], exp)

    def test_partial_deletion_order(self, zt_file):
        """Delete some arrays before the Reader, keep others."""
        path, expected = zt_file

        reader = Reader(path)
        arr_small = reader["small"]
        arr_matrix = reader["matrix"]
        arr_vector = reader["vector"]

        # Delete one array, then reader, then check remaining
        del arr_matrix
        del reader
        gc.collect()

        np.testing.assert_array_equal(arr_small, expected["small"])
        np.testing.assert_array_equal(arr_vector, expected["vector"])

    def test_mixed_single_and_bulk_reads(self, zt_file):
        """Mixing reader["x"] with load_numpy on the same Reader."""
        path, expected = zt_file

        reader = Reader(path)
        single = reader["small"]
        bulk = reader.load_numpy(copy=False)

        del reader
        gc.collect()

        np.testing.assert_array_equal(single, expected["small"])
        for name, exp in expected.items():
            np.testing.assert_array_equal(bulk[name], exp)


# ============================================================================
# COPY-ON-WRITE
# ============================================================================


class TestCOW:
    """Verify MAP_PRIVATE copy-on-write semantics."""

    def test_write_triggers_cow_not_corruption(self, zt_file):
        """Writing to a zero-copy array must not change the file on disk."""
        path, expected = zt_file

        with open(path, "rb") as f:
            original_bytes = f.read()

        reader = Reader(path)
        arr = reader["small"]

        # .zt files are opened with MAP_PRIVATE, so arrays should be writable
        assert arr.flags[
            "WRITEABLE"
        ], "Zero-copy array from .zt file should be writable (MAP_PRIVATE)"

        # Write sentinel value
        arr[:] = 999.0
        np.testing.assert_array_equal(arr, np.full(100, 999.0, dtype=np.float32))

        del reader
        del arr
        gc.collect()

        # File on disk must be unchanged
        with open(path, "rb") as f:
            after_bytes = f.read()
        assert (
            original_bytes == after_bytes
        ), "File corrupted by writing to zero-copy array"

        # Re-read must return original data
        reader2 = Reader(path)
        np.testing.assert_array_equal(reader2["small"], expected["small"])

    def test_two_readers_cow_isolation(self, zt_file):
        """Writing via one Reader's array must not affect another Reader's view."""
        path, expected = zt_file

        reader1 = Reader(path)
        reader2 = Reader(path)
        arr1 = reader1["small"]
        arr2 = reader2["small"]

        # Mutate arr1
        if arr1.flags["WRITEABLE"]:
            arr1[:] = -1.0

        # arr2 must still see original data (separate mmap)
        np.testing.assert_array_equal(arr2, expected["small"])


# ============================================================================
# TORCH ZERO-COPY
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTorchZeroCopy:
    """Tests for the torch.from_numpy zero-copy chain."""

    def test_torch_zerocopy_survives_reader_deletion(self, zt_file):
        """load_torch(copy=False) tensors must survive Reader deletion."""
        path, expected = zt_file

        reader = Reader(path)
        tensors = reader.load_torch(copy=False)
        del reader
        gc.collect()

        for name, exp in expected.items():
            np.testing.assert_array_equal(tensors[name].numpy(), exp)

    def test_torch_copy_true_independent(self, zt_file):
        """load_torch(copy=True) tensors are fully independent."""
        path, expected = zt_file

        reader = Reader(path)
        tensors = reader.load_torch(copy=True)
        del reader
        gc.collect()

        for name, exp in expected.items():
            np.testing.assert_array_equal(tensors[name].numpy(), exp)

    def test_torch_zerocopy_numpy_roundtrip(self, zt_file):
        """torch tensor → .numpy() → back should maintain data."""
        path, expected = zt_file

        reader = Reader(path)
        tensors = reader.load_torch(copy=False)
        del reader
        gc.collect()

        for name, exp in expected.items():
            # torch → numpy → back to numpy (should still be valid)
            np_arr = tensors[name].numpy()
            np.testing.assert_array_equal(np_arr, exp)


# ============================================================================
# FILE DELETION WHILE ARRAYS ALIVE
# ============================================================================


class TestFileDeletion:
    """Test that arrays survive file deletion (Linux holds inode via mmap)."""

    def test_file_deleted_while_arrays_alive(self):
        """Deleting the .zt file while zero-copy arrays exist should be safe on Linux."""
        fd, path = tempfile.mkstemp(suffix=".zt")
        os.close(fd)

        data = np.arange(1000, dtype=np.float32)
        with Writer(path) as w:
            w.add("tensor", data)

        reader = Reader(path)
        arr = reader["tensor"]

        # Delete the file while mmap is still active
        os.remove(path)
        assert not os.path.exists(path)

        # On Linux, the inode is held open by mmap — data should still be valid
        np.testing.assert_array_equal(arr, data)

        del reader
        gc.collect()

        # Array should still be valid (base keeps PyReader alive, mmap holds inode)
        np.testing.assert_array_equal(arr, data)


# ============================================================================
# MEMORY LEAK DETECTION
# ============================================================================


class TestMemoryLeaks:
    """
    Detect memory leaks using RSS (resident set size).

    tracemalloc can't see mmap allocations, so we use getrusage for RSS.
    RSS-based tests are inherently noisy, so we use large tensors and
    generous-but-meaningful thresholds.
    """

    def test_rss_stable_over_open_close_cycles(self, large_zt_file):
        """RSS should not grow proportionally to iteration count."""
        path, _ = large_zt_file
        per_file_mb = os.path.getsize(path) / (1024 * 1024)

        # Warm up (first mmap may page in Python/numpy internals)
        for _ in range(3):
            r = Reader(path)
            t = r.load_numpy(copy=False)
            del t, r
            gc.collect()

        rss_before = _get_rss_kb()

        for _ in range(20):
            reader = Reader(path)
            tensors = reader.load_numpy(copy=False)
            del tensors
            del reader
            gc.collect()

        rss_after = _get_rss_kb()
        growth_mb = (rss_after - rss_before) / 1024

        # If mmap leaks, 20 iterations of ~4MB file = ~80MB growth.
        # Allow up to 2x the file size for OS/Python overhead.
        assert growth_mb < per_file_mb * 2, (
            f"RSS grew by {growth_mb:.1f} MB over 20 iterations "
            f"(file is {per_file_mb:.1f} MB). Possible mmap leak."
        )

    def test_rss_stable_copy_true(self, large_zt_file):
        """RSS should not grow with copy=True cycles (Rust-allocated memory freed)."""
        path, _ = large_zt_file
        per_file_mb = os.path.getsize(path) / (1024 * 1024)

        # Warm up
        for _ in range(3):
            r = Reader(path)
            t = r.load_numpy(copy=True)
            del t, r
            gc.collect()

        rss_before = _get_rss_kb()

        for _ in range(20):
            reader = Reader(path)
            tensors = reader.load_numpy(copy=True)
            del tensors
            del reader
            gc.collect()

        rss_after = _get_rss_kb()
        growth_mb = (rss_after - rss_before) / 1024

        assert growth_mb < per_file_mb * 2, (
            f"RSS grew by {growth_mb:.1f} MB over 20 copy=True iterations "
            f"(file is {per_file_mb:.1f} MB). Possible leak in copy path."
        )


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Edge cases for memory safety."""

    def test_empty_tensor_zerocopy(self):
        """Zero-copy of a 0-element tensor should not crash."""
        fd, path = tempfile.mkstemp(suffix=".zt")
        os.close(fd)
        try:
            empty = np.array([], dtype=np.float32)
            with Writer(path) as w:
                w.add("empty", empty)

            reader = Reader(path)
            arr = reader["empty"]
            del reader
            gc.collect()

            assert arr.size == 0
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_scalar_tensor_zerocopy(self):
        """Zero-copy of a scalar tensor should work correctly."""
        fd, path = tempfile.mkstemp(suffix=".zt")
        os.close(fd)
        try:
            scalar = np.array(42.0, dtype=np.float32)
            with Writer(path) as w:
                w.add("scalar", scalar)

            reader = Reader(path)
            arr = reader["scalar"]
            del reader
            gc.collect()

            np.testing.assert_equal(float(arr.flat[0]), 42.0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_concurrent_readers_same_file(self, zt_file):
        """Multiple Readers on the same file should not interfere."""
        path, expected = zt_file

        reader1 = Reader(path)
        reader2 = Reader(path)

        arr1 = reader1["small"]
        arr2 = reader2["small"]

        del reader1
        gc.collect()
        np.testing.assert_array_equal(arr2, expected["small"])
        np.testing.assert_array_equal(arr1, expected["small"])

        del reader2
        gc.collect()
        np.testing.assert_array_equal(arr1, expected["small"])
        np.testing.assert_array_equal(arr2, expected["small"])

    def test_multiple_dtypes_zerocopy(self):
        """Zero-copy with mixed dtypes in the same file."""
        fd, path = tempfile.mkstemp(suffix=".zt")
        os.close(fd)
        try:
            tensors = {
                "f32": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "f64": np.array([4.0, 5.0], dtype=np.float64),
                "i32": np.array([10, 20, 30, 40], dtype=np.int32),
                "u8": np.array([0, 128, 255], dtype=np.uint8),
            }
            with Writer(path) as w:
                for name, arr in tensors.items():
                    w.add(name, arr)

            reader = Reader(path)
            loaded = reader.load_numpy(copy=False)
            del reader
            gc.collect()

            for name, exp in tensors.items():
                np.testing.assert_array_equal(loaded[name], exp)
        finally:
            if os.path.exists(path):
                os.remove(path)
