"""
Comprehensive test suite for zTensor Python bindings.

Tests cover:
1. Correctness - data integrity across read/write cycles
2. Edge cases - empty tensors, large tensors, special values
3. Memory leaks - repeated operations shouldn't grow memory
4. Type coverage - all supported dtypes
5. Sparse formats - CSR and COO
6. Compression - zstd encoding
7. Error handling - proper exception raising
"""

import pytest
import numpy as np
import tempfile
import os
import gc
import sys
import tracemalloc

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ml_dtypes import bfloat16 as np_bfloat16
    ML_DTYPES_AVAILABLE = True
except ImportError:
    np_bfloat16 = None
    ML_DTYPES_AVAILABLE = False

import ztensor
from ztensor import Reader, Writer, ZTensorError


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_file():
    """Provides a temporary file path that's cleaned up after the test."""
    fd, path = tempfile.mkstemp(suffix='.zt')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_tensors():
    """Standard test tensors of various shapes."""
    return {
        'scalar': np.array(3.14, dtype=np.float32),
        'vector': np.arange(100, dtype=np.float32),
        'matrix': np.random.randn(32, 64).astype(np.float32),
        'tensor3d': np.random.randn(4, 8, 16).astype(np.float32),
        'large': np.random.randn(1000, 1000).astype(np.float32),
    }


# ============================================================================
# 1. CORRECTNESS TESTS
# ============================================================================

class TestCorrectness:
    """Tests for data integrity and correctness."""

    def test_roundtrip_basic(self, temp_file, sample_tensors):
        """Basic write/read roundtrip preserves data exactly."""
        with Writer(temp_file) as w:
            for name, tensor in sample_tensors.items():
                w.add_tensor(name, tensor)
        
        with Reader(temp_file) as r:
            for name, expected in sample_tensors.items():
                loaded = r.read_tensor(name)
                np.testing.assert_array_equal(loaded, expected, 
                    err_msg=f"Tensor '{name}' data mismatch")

    def test_roundtrip_all_dtypes(self, temp_file):
        """All supported dtypes preserve data correctly."""
        dtypes = [
            np.float64, np.float32, np.float16,
            np.int64, np.int32, np.int16, np.int8,
            np.uint64, np.uint32, np.uint16, np.uint8,
            np.bool_,
        ]
        
        tensors = {}
        for dt in dtypes:
            name = f"tensor_{dt.__name__}"
            if np.issubdtype(dt, np.floating):
                tensors[name] = np.array([1.5, 2.5, -3.5], dtype=dt)
            elif np.issubdtype(dt, np.signedinteger):
                tensors[name] = np.array([1, -2, 3], dtype=dt)
            elif np.issubdtype(dt, np.unsignedinteger):
                tensors[name] = np.array([1, 2, 3], dtype=dt)
            elif dt == np.bool_:
                tensors[name] = np.array([True, False, True], dtype=dt)
        
        with Writer(temp_file) as w:
            for name, tensor in tensors.items():
                w.add_tensor(name, tensor)
        
        with Reader(temp_file) as r:
            for name, expected in tensors.items():
                loaded = r.read_tensor(name)
                np.testing.assert_array_equal(loaded, expected,
                    err_msg=f"Dtype {name} mismatch")

    @pytest.mark.skipif(not ML_DTYPES_AVAILABLE, reason="ml_dtypes not installed")
    def test_bfloat16(self, temp_file):
        """bfloat16 dtype roundtrip."""
        original = np.array([1.0, 2.5, -3.0], dtype=np_bfloat16)
        
        with Writer(temp_file) as w:
            w.add_tensor("bf16", original)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("bf16")
            np.testing.assert_array_equal(loaded, original)

    def test_compression_preserves_data(self, temp_file):
        """Zstd compression doesn't corrupt data."""
        original = np.random.randn(500, 500).astype(np.float32)
        
        with Writer(temp_file) as w:
            w.add_tensor("compressed", original, compress=True)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("compressed")
            np.testing.assert_array_equal(loaded, original)

    def test_multiple_tensors(self, temp_file):
        """Multiple tensors in one file are all correct."""
        tensors = {f"tensor_{i}": np.random.randn(100).astype(np.float32) 
                   for i in range(50)}
        
        with Writer(temp_file) as w:
            for name, t in tensors.items():
                w.add_tensor(name, t)
        
        with Reader(temp_file) as r:
            assert len(r) == 50
            for name, expected in tensors.items():
                loaded = r.read_tensor(name)
                np.testing.assert_array_equal(loaded, expected)


# ============================================================================
# 2. EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_empty_file(self, temp_file):
        """Empty file (no tensors) is valid."""
        with Writer(temp_file) as w:
            pass  # Write nothing
        
        with Reader(temp_file) as r:
            assert len(r) == 0
            assert r.tensor_names == []

    def test_scalar_tensor(self, temp_file):
        """0-dimensional scalar tensor."""
        scalar = np.array(42.0, dtype=np.float32)
        
        with Writer(temp_file) as w:
            w.add_tensor("scalar", scalar)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("scalar")
            # Note: Scalars are stored as 1-element arrays in ztensor
            assert loaded.size == 1
            assert float(loaded.flat[0]) == 42.0

    def test_empty_tensor(self, temp_file):
        """Zero-element tensor."""
        empty = np.array([], dtype=np.float32).reshape(0, 10)
        
        with Writer(temp_file) as w:
            w.add_tensor("empty", empty)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("empty")
            assert loaded.shape == (0, 10)

    def test_single_element(self, temp_file):
        """Single element tensor."""
        single = np.array([3.14], dtype=np.float32)
        
        with Writer(temp_file) as w:
            w.add_tensor("single", single)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("single")
            np.testing.assert_array_equal(loaded, single)

    def test_special_float_values(self, temp_file):
        """NaN, Inf, -Inf are preserved."""
        special = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=np.float32)
        
        with Writer(temp_file) as w:
            w.add_tensor("special", special)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("special")
            assert np.isnan(loaded[0])
            assert np.isposinf(loaded[1])
            assert np.isneginf(loaded[2])
            assert loaded[3] == 0.0
            assert loaded[4] == 0.0

    def test_large_tensor(self, temp_file):
        """Large tensor (~400MB)."""
        large = np.random.randn(10000, 10000).astype(np.float32)  # ~400MB
        
        with Writer(temp_file) as w:
            w.add_tensor("large", large)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("large")
            np.testing.assert_array_equal(loaded, large)

    def test_high_dimensional(self, temp_file):
        """High-dimensional tensor (7D)."""
        high_dim = np.random.randn(2, 3, 4, 5, 6, 7, 8).astype(np.float32)
        
        with Writer(temp_file) as w:
            w.add_tensor("7d", high_dim)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("7d")
            np.testing.assert_array_equal(loaded, high_dim)
            assert loaded.shape == (2, 3, 4, 5, 6, 7, 8)

    def test_unicode_tensor_name(self, temp_file):
        """Unicode characters in tensor name."""
        tensor = np.array([1.0, 2.0], dtype=np.float32)
        name = "层_1.权重_αβγ"
        
        with Writer(temp_file) as w:
            w.add_tensor(name, tensor)
        
        with Reader(temp_file) as r:
            assert name in r.tensor_names
            loaded = r.read_tensor(name)
            np.testing.assert_array_equal(loaded, tensor)

    def test_very_long_name(self, temp_file):
        """Very long tensor name."""
        name = "x" * 1000
        tensor = np.array([1.0], dtype=np.float32)
        
        with Writer(temp_file) as w:
            w.add_tensor(name, tensor)
        
        with Reader(temp_file) as r:
            assert name in r.tensor_names


# ============================================================================
# 3. MEMORY LEAK TESTS
# ============================================================================

class TestMemoryLeaks:
    """Tests to detect memory leaks."""

    def test_repeated_read_no_leak(self, temp_file):
        """Repeated reads should not leak memory."""
        # Create test file
        data = np.random.randn(1000, 1000).astype(np.float32)
        with Writer(temp_file) as w:
            w.add_tensor("data", data)
        
        # Force GC and get baseline
        gc.collect()
        tracemalloc.start()
        
        # Read many times
        with Reader(temp_file) as r:
            for _ in range(100):
                _ = r.read_tensor("data")
                gc.collect()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory should not grow significantly (allow 50MB overhead)
        data_size = data.nbytes
        assert current < data_size * 3, f"Possible memory leak: {current / 1e6:.1f}MB used"

    def test_repeated_write_no_leak(self, temp_file):
        """Repeated file writes should not leak memory."""
        data = np.random.randn(500, 500).astype(np.float32)
        
        gc.collect()
        tracemalloc.start()
        
        for i in range(50):
            path = temp_file + f".{i}"
            with Writer(path) as w:
                w.add_tensor("data", data)
            os.remove(path)
            gc.collect()
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Should not accumulate memory
        assert current < data.nbytes * 3, f"Possible write memory leak: {current / 1e6:.1f}MB"

    def test_reader_context_cleanup(self, temp_file):
        """Reader properly releases resources when exiting context."""
        data = np.random.randn(1000, 1000).astype(np.float32)
        with Writer(temp_file) as w:
            w.add_tensor("data", data)
        
        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
        for _ in range(20):
            with Reader(temp_file) as r:
                _ = r.read_tensor("data")
            gc.collect()
        
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(stat.size_diff for stat in top_stats[:10])
        
        # Allow some overhead but not proportional to iterations
        assert abs(total_diff) < data.nbytes, "Memory not properly released"

    def test_zero_copy_view_lifetime(self, temp_file):
        """Zero-copy view keeps data alive correctly."""
        original = np.random.randn(100, 100).astype(np.float32)
        with Writer(temp_file) as w:
            w.add_tensor("data", original)
        
        # Read and keep reference
        with Reader(temp_file) as r:
            view = r.read_tensor("data")
            # View should be valid within context
            np.testing.assert_array_equal(view, original)
            
            # Take a slice - should also be valid
            slice_view = view[10:20, :]
            assert slice_view.shape == (10, 100)


# ============================================================================
# 4. SPARSE TENSOR TESTS
# ============================================================================

class TestSparseTensors:
    """Tests for sparse tensor formats."""


    def test_sparse_csr_roundtrip(self, temp_file):
        """CSR sparse tensor roundtrip."""
        try:
            from scipy.sparse import csr_matrix
        except ImportError:
            pytest.skip("scipy not installed")
        
        # Create sparse matrix
        dense = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]], dtype=np.float32)
        sparse = csr_matrix(dense)
        
        with Writer(temp_file) as w:
            w.add_sparse_csr("sparse",
                values=sparse.data,
                indices=sparse.indices.astype(np.uint64),
                indptr=sparse.indptr.astype(np.uint64),
                shape=sparse.shape)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("sparse")
            expected = sparse.toarray()
            np.testing.assert_array_equal(loaded.toarray(), expected)


    def test_sparse_coo_roundtrip(self, temp_file):
        """COO sparse tensor roundtrip."""
        try:
            from scipy.sparse import coo_matrix
        except ImportError:
            pytest.skip("scipy not installed")
        
        # Create COO matrix
        row = np.array([0, 1, 2])
        col = np.array([1, 2, 0])
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sparse = coo_matrix((data, (row, col)), shape=(3, 3))
        
        # Convert to SoA format for writing
        coords = np.vstack([sparse.row, sparse.col]).astype(np.uint64).flatten()
        
        with Writer(temp_file) as w:
            w.add_sparse_coo("coo",
                values=sparse.data,
                indices=coords,
                shape=sparse.shape)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("coo")
            np.testing.assert_array_equal(loaded.toarray(), sparse.toarray())

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_sparse_csr_torch_roundtrip(self, temp_file):
        """CSR sparse tensor roundtrip with PyTorch."""
        # Create sparse matrix
        dense = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]], dtype=np.float32)
        
        try:
            from scipy.sparse import csr_matrix
            sparse = csr_matrix(dense)
        except ImportError:
            pytest.skip("scipy not installed")
        
        with Writer(temp_file) as w:
            w.add_sparse_csr("sparse",
                values=sparse.data,
                indices=sparse.indices.astype(np.uint64),
                indptr=sparse.indptr.astype(np.uint64),
                shape=sparse.shape)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("sparse", to='torch')
            assert loaded.is_sparse_csr
            assert loaded.shape == sparse.shape
            torch.testing.assert_close(loaded.to_dense(), torch.from_numpy(dense))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_sparse_coo_torch_roundtrip(self, temp_file):
        """COO sparse tensor roundtrip with PyTorch."""
        # Create COO data
        row = np.array([0, 1, 2])
        col = np.array([1, 2, 0])
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        shape = (3, 3)
        
        # Expected dense result
        expected_dense = np.zeros(shape, dtype=np.float32)
        expected_dense[0, 1] = 1.0
        expected_dense[1, 2] = 2.0
        expected_dense[2, 0] = 3.0
        
        # Convert to SoA format for writing
        coords = np.vstack([row, col]).astype(np.uint64).flatten()
        
        with Writer(temp_file) as w:
            w.add_sparse_coo("coo",
                values=data,
                indices=coords,
                shape=shape)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("coo", to='torch')
            assert loaded.is_sparse
            assert loaded.shape == shape
            torch.testing.assert_close(loaded.to_dense(), torch.from_numpy(expected_dense))


# ============================================================================
# 5. PYTORCH TESTS
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPyTorch:
    """Tests for PyTorch integration."""

    def test_torch_tensor_write_read(self, temp_file):
        """Write PyTorch tensor, read back."""
        original = torch.randn(32, 64)
        
        with Writer(temp_file) as w:
            w.add_tensor("torch", original)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("torch", to='torch')
            torch.testing.assert_close(loaded, original)

    def test_torch_dtypes(self, temp_file):
        """Various PyTorch dtypes."""
        tensors = {
            'f32': torch.randn(10, dtype=torch.float32),
            'f64': torch.randn(10, dtype=torch.float64),
            'f16': torch.randn(10, dtype=torch.float16),
            'bf16': torch.randn(10, dtype=torch.bfloat16),
            'i64': torch.randint(-100, 100, (10,), dtype=torch.int64),
            'i32': torch.randint(-100, 100, (10,), dtype=torch.int32),
        }
        
        with Writer(temp_file) as w:
            for name, t in tensors.items():
                w.add_tensor(name, t)
        
        with Reader(temp_file) as r:
            for name, expected in tensors.items():
                loaded = r.read_tensor(name, to='torch')
                torch.testing.assert_close(loaded, expected)

    def test_torch_zero_copy(self, temp_file):
        """PyTorch tensor uses zero-copy."""
        original = torch.randn(1000, 1000)
        
        with Writer(temp_file) as w:
            w.add_tensor("data", original)
        
        with Reader(temp_file) as r:
            loaded = r.read_tensor("data", to='torch')
            # Should have _owner attribute for zero-copy
            assert hasattr(loaded, '_owner')
            torch.testing.assert_close(loaded, original)


# ============================================================================
# 6. ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for proper error handling."""

    def test_read_nonexistent_tensor(self, temp_file):
        """Reading nonexistent tensor raises error."""
        with Writer(temp_file) as w:
            w.add_tensor("exists", np.array([1.0], dtype=np.float32))
        
        with Reader(temp_file) as r:
            with pytest.raises(ZTensorError):
                r.read_tensor("does_not_exist")

    def test_read_closed_reader(self, temp_file):
        """Reading from closed reader raises error."""
        with Writer(temp_file) as w:
            w.add_tensor("data", np.array([1.0], dtype=np.float32))
        
        r = Reader(temp_file)
        r.__exit__(None, None, None)
        
        with pytest.raises(ZTensorError):
            r.read_tensor("data")

    def test_write_after_finalize(self, temp_file):
        """Writing after finalize raises error."""
        w = Writer(temp_file)
        w.add_tensor("t1", np.array([1.0], dtype=np.float32))
        w.finalize()
        
        with pytest.raises(ZTensorError):
            w.add_tensor("t2", np.array([2.0], dtype=np.float32))

    def test_invalid_file_path(self):
        """Opening nonexistent file raises error."""
        with pytest.raises(ZTensorError):
            Reader("/nonexistent/path/file.zt")

    def test_corrupted_file(self, temp_file):
        """Corrupted file raises error."""
        # Write garbage
        with open(temp_file, 'wb') as f:
            f.write(b"GARBAGE_DATA_NOT_ZTENSOR")
        
        with pytest.raises(ZTensorError):
            Reader(temp_file)


# ============================================================================
# 7. API TESTS
# ============================================================================

class TestAPI:
    """Tests for API functionality."""

    def test_reader_iteration(self, temp_file, sample_tensors):
        """Reader is iterable."""
        with Writer(temp_file) as w:
            for name, t in sample_tensors.items():
                w.add_tensor(name, t)
        
        with Reader(temp_file) as r:
            names = [meta.name for meta in r]
            assert set(names) == set(sample_tensors.keys())

    def test_reader_len(self, temp_file):
        """Reader has correct length."""
        n_tensors = 10
        with Writer(temp_file) as w:
            for i in range(n_tensors):
                w.add_tensor(f"t{i}", np.array([float(i)], dtype=np.float32))
        
        with Reader(temp_file) as r:
            assert len(r) == n_tensors

    def test_reader_contains(self, temp_file):
        """Reader supports 'in' operator."""
        with Writer(temp_file) as w:
            w.add_tensor("exists", np.array([1.0], dtype=np.float32))
        
        with Reader(temp_file) as r:
            assert "exists" in r
            assert "not_exists" not in r

    def test_reader_getitem_by_name(self, temp_file):
        """Reader supports dict-like access by name."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with Writer(temp_file) as w:
            w.add_tensor("data", data)
        
        with Reader(temp_file) as r:
            loaded = r["data"]
            np.testing.assert_array_equal(loaded, data)

    def test_reader_getitem_by_index(self, temp_file):
        """Reader supports index access for metadata."""
        with Writer(temp_file) as w:
            w.add_tensor("first", np.array([1.0], dtype=np.float32))
            w.add_tensor("second", np.array([2.0], dtype=np.float32))
        
        with Reader(temp_file) as r:
            meta = r[0]
            assert meta.name in ["first", "second"]

    def test_metadata_properties(self, temp_file):
        """Metadata object has correct properties."""
        data = np.random.randn(32, 64).astype(np.float32)
        with Writer(temp_file) as w:
            w.add_tensor("test", data)
        
        with Reader(temp_file) as r:
            meta = r.metadata("test")
            assert meta.name == "test"
            assert meta.shape == (32, 64)
            assert meta.dtype_str == "float32"
            assert meta.layout == "dense"

    def test_batch_read(self, temp_file):
        """Batch read multiple tensors."""
        tensors = {f"t{i}": np.random.randn(100).astype(np.float32) for i in range(10)}
        
        with Writer(temp_file) as w:
            for name, t in tensors.items():
                w.add_tensor(name, t)
        
        with Reader(temp_file) as r:
            names = list(tensors.keys())
            loaded = r.read_tensors(names)
            
            assert len(loaded) == len(names)
            for i, name in enumerate(names):
                np.testing.assert_array_equal(loaded[i], tensors[name])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
