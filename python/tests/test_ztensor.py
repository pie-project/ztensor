"""
Comprehensive test suite for zTensor Python bindings (PyO3).

Tests cover:
1. Correctness - data integrity across read/write cycles
2. Edge cases - empty tensors, large tensors, special values
3. Memory leaks - repeated operations shouldn't grow memory
4. Type coverage - all supported dtypes
5. Compression - zstd encoding
6. Error handling - proper exception raising
7. API - dict-like access, context managers, metadata
"""

import pytest
import numpy as np
import tempfile
import os
import gc
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
from ztensor import Reader, Writer

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_file():
    """Provides a temporary file path that's cleaned up after the test."""
    fd, path = tempfile.mkstemp(suffix=".zt")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_dir():
    """Provides a temporary directory that's cleaned up after the test."""
    d = tempfile.mkdtemp()
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_tensors():
    """Standard test tensors of various shapes."""
    return {
        "scalar": np.array(3.14, dtype=np.float32),
        "vector": np.arange(100, dtype=np.float32),
        "matrix": np.random.randn(32, 64).astype(np.float32),
        "tensor3d": np.random.randn(4, 8, 16).astype(np.float32),
        "large": np.random.randn(1000, 1000).astype(np.float32),
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
                w.add(name, tensor)

        reader = Reader(temp_file)
        for name, expected in sample_tensors.items():
            loaded = reader[name]
            np.testing.assert_array_equal(
                loaded, expected, err_msg=f"Tensor '{name}' data mismatch"
            )

    def test_roundtrip_all_dtypes(self, temp_file):
        """All supported dtypes preserve data correctly."""
        dtypes = [
            np.float64,
            np.float32,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint64,
            np.uint32,
            np.uint16,
            np.uint8,
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
                w.add(name, tensor)

        reader = Reader(temp_file)
        for name, expected in tensors.items():
            loaded = reader[name]
            np.testing.assert_array_equal(
                loaded, expected, err_msg=f"Dtype {name} mismatch"
            )

    @pytest.mark.skipif(not ML_DTYPES_AVAILABLE, reason="ml_dtypes not installed")
    def test_bfloat16(self, temp_file):
        """bfloat16 dtype roundtrip."""
        original = np.array([1.0, 2.5, -3.0], dtype=np_bfloat16)

        with Writer(temp_file) as w:
            w.add("bf16", original)

        reader = Reader(temp_file)
        loaded = reader["bf16"]
        np.testing.assert_array_equal(loaded, original)

    def test_compression_preserves_data(self, temp_file):
        """Zstd compression doesn't corrupt data."""
        original = np.random.randn(500, 500).astype(np.float32)

        with Writer(temp_file) as w:
            w.add("compressed", original, compress=True)

        reader = Reader(temp_file)
        loaded = reader["compressed"]
        np.testing.assert_array_equal(loaded, original)

    def test_compression_level(self, temp_file):
        """Specific compression level preserves data."""
        original = np.random.randn(200, 200).astype(np.float32)

        with Writer(temp_file) as w:
            w.add("level5", original, compress=5)

        reader = Reader(temp_file)
        loaded = reader["level5"]
        np.testing.assert_array_equal(loaded, original)

    def test_multiple_tensors(self, temp_file):
        """Multiple tensors in one file are all correct."""
        tensors = {
            f"tensor_{i}": np.random.randn(100).astype(np.float32) for i in range(50)
        }

        with Writer(temp_file) as w:
            for name, t in tensors.items():
                w.add(name, t)

        reader = Reader(temp_file)
        assert len(reader) == 50
        for name, expected in tensors.items():
            loaded = reader[name]
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

        reader = Reader(temp_file)
        assert len(reader) == 0
        assert reader.keys() == []

    def test_scalar_tensor(self, temp_file):
        """0-dimensional scalar tensor."""
        scalar = np.array(42.0, dtype=np.float32)

        with Writer(temp_file) as w:
            w.add("scalar", scalar)

        reader = Reader(temp_file)
        loaded = reader["scalar"]
        # Note: Scalars are stored as 1-element arrays in ztensor
        assert loaded.size == 1
        assert float(loaded.flat[0]) == 42.0

    def test_single_element(self, temp_file):
        """Single element tensor."""
        single = np.array([3.14], dtype=np.float32)

        with Writer(temp_file) as w:
            w.add("single", single)

        reader = Reader(temp_file)
        loaded = reader["single"]
        np.testing.assert_array_equal(loaded, single)

    def test_special_float_values(self, temp_file):
        """NaN, Inf, -Inf are preserved."""
        special = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=np.float32)

        with Writer(temp_file) as w:
            w.add("special", special)

        reader = Reader(temp_file)
        loaded = reader["special"]
        assert np.isnan(loaded[0])
        assert np.isposinf(loaded[1])
        assert np.isneginf(loaded[2])
        assert loaded[3] == 0.0
        assert loaded[4] == 0.0

    def test_large_tensor(self, temp_file):
        """Large tensor (~400MB)."""
        large = np.random.randn(10000, 10000).astype(np.float32)  # ~400MB

        with Writer(temp_file) as w:
            w.add("large", large)

        reader = Reader(temp_file)
        loaded = reader["large"]
        np.testing.assert_array_equal(loaded, large)

    def test_high_dimensional(self, temp_file):
        """High-dimensional tensor (7D)."""
        high_dim = np.random.randn(2, 3, 4, 5, 6, 7, 8).astype(np.float32)

        with Writer(temp_file) as w:
            w.add("7d", high_dim)

        reader = Reader(temp_file)
        loaded = reader["7d"]
        np.testing.assert_array_equal(loaded, high_dim)
        assert loaded.shape == (2, 3, 4, 5, 6, 7, 8)

    def test_unicode_tensor_name(self, temp_file):
        """Unicode characters in tensor name."""
        tensor = np.array([1.0, 2.0], dtype=np.float32)
        name = "层_1.权重_αβγ"

        with Writer(temp_file) as w:
            w.add(name, tensor)

        reader = Reader(temp_file)
        assert name in reader.keys()
        loaded = reader[name]
        np.testing.assert_array_equal(loaded, tensor)

    def test_very_long_name(self, temp_file):
        """Very long tensor name."""
        name = "x" * 1000
        tensor = np.array([1.0], dtype=np.float32)

        with Writer(temp_file) as w:
            w.add(name, tensor)

        reader = Reader(temp_file)
        assert name in reader.keys()


# ============================================================================
# 3. MEMORY LEAK TESTS
# ============================================================================


class TestMemoryLeaks:
    """Tests to detect memory leaks."""

    def test_repeated_read_no_leak(self, temp_file):
        """Repeated reads should not leak memory."""
        data = np.random.randn(1000, 1000).astype(np.float32)
        with Writer(temp_file) as w:
            w.add("data", data)

        gc.collect()
        tracemalloc.start()

        reader = Reader(temp_file)
        for _ in range(100):
            _ = reader["data"]
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        data_size = data.nbytes
        assert (
            current < data_size * 3
        ), f"Possible memory leak: {current / 1e6:.1f}MB used"

    def test_repeated_write_no_leak(self, temp_file):
        """Repeated file writes should not leak memory."""
        data = np.random.randn(500, 500).astype(np.float32)

        gc.collect()
        tracemalloc.start()

        for i in range(50):
            path = temp_file + f".{i}"
            with Writer(path) as w:
                w.add("data", data)
            os.remove(path)
            gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert (
            current < data.nbytes * 3
        ), f"Possible write memory leak: {current / 1e6:.1f}MB"

    def test_reader_cleanup(self, temp_file):
        """Reader properly releases resources."""
        data = np.random.randn(1000, 1000).astype(np.float32)
        with Writer(temp_file) as w:
            w.add("data", data)

        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(20):
            reader = Reader(temp_file)
            _ = reader["data"]
            del reader
            gc.collect()

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_diff = sum(stat.size_diff for stat in top_stats[:10])

        # Allow small overhead from Python/allocator internals (1% margin)
        assert abs(total_diff) < data.nbytes * 1.01, "Memory not properly released"


# ============================================================================
# 4. ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Tests for proper error handling."""

    def test_read_nonexistent_tensor(self, temp_file):
        """Reading nonexistent tensor raises KeyError."""
        with Writer(temp_file) as w:
            w.add("exists", np.array([1.0], dtype=np.float32))

        reader = Reader(temp_file)
        with pytest.raises(KeyError):
            reader["does_not_exist"]

    def test_write_after_finish(self, temp_file):
        """Writing after finish raises error."""
        w = Writer(temp_file)
        w.add("t1", np.array([1.0], dtype=np.float32))
        w.finish()

        with pytest.raises(RuntimeError):
            w.add("t2", np.array([2.0], dtype=np.float32))

    def test_invalid_file_path(self):
        """Opening nonexistent file raises error."""
        with pytest.raises((OSError, RuntimeError)):
            Reader("/nonexistent/path/file.zt")

    def test_corrupted_file(self, temp_file):
        """Corrupted file raises error."""
        with open(temp_file, "wb") as f:
            f.write(b"GARBAGE_DATA_NOT_ZTENSOR")

        with pytest.raises((RuntimeError, OSError)):
            Reader(temp_file)


# ============================================================================
# 5. API TESTS
# ============================================================================


class TestAPI:
    """Tests for API functionality."""

    def test_reader_keys(self, temp_file, sample_tensors):
        """Reader.keys() returns all tensor names."""
        with Writer(temp_file) as w:
            for name, t in sample_tensors.items():
                w.add(name, t)

        reader = Reader(temp_file)
        assert set(reader.keys()) == set(sample_tensors.keys())

    def test_reader_len(self, temp_file):
        """Reader has correct length."""
        n_tensors = 10
        with Writer(temp_file) as w:
            for i in range(n_tensors):
                w.add(f"t{i}", np.array([float(i)], dtype=np.float32))

        reader = Reader(temp_file)
        assert len(reader) == n_tensors

    def test_reader_contains(self, temp_file):
        """Reader supports 'in' operator."""
        with Writer(temp_file) as w:
            w.add("exists", np.array([1.0], dtype=np.float32))

        reader = Reader(temp_file)
        assert "exists" in reader
        assert "not_exists" not in reader

    def test_reader_getitem(self, temp_file):
        """Reader supports dict-like access by name."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with Writer(temp_file) as w:
            w.add("data", data)

        reader = Reader(temp_file)
        loaded = reader["data"]
        np.testing.assert_array_equal(loaded, data)

    def test_reader_context_manager(self, temp_file):
        """Reader works as context manager."""
        data = np.array([1.0, 2.0], dtype=np.float32)
        with Writer(temp_file) as w:
            w.add("data", data)

        with Reader(temp_file) as r:
            loaded = r["data"]
            np.testing.assert_array_equal(loaded, data)

    def test_metadata_properties(self, temp_file):
        """Metadata object has correct properties."""
        data = np.random.randn(32, 64).astype(np.float32)
        with Writer(temp_file) as w:
            w.add("test", data)

        reader = Reader(temp_file)
        meta = reader.metadata("test")
        assert meta.name == "test"
        assert list(meta.shape) == [32, 64]
        assert meta.dtype == "float32"
        assert meta.format == "dense"

    def test_batch_read(self, temp_file):
        """Batch read multiple tensors."""
        tensors = {f"t{i}": np.random.randn(100).astype(np.float32) for i in range(10)}

        with Writer(temp_file) as w:
            for name, t in tensors.items():
                w.add(name, t)

        reader = Reader(temp_file)
        names = list(tensors.keys())
        loaded = reader.read_tensors(names)

        assert len(loaded) == len(names)
        for name in names:
            np.testing.assert_array_equal(loaded[name], tensors[name])

    def test_repr(self, temp_file):
        """Reader and Writer have useful repr."""
        with Writer(temp_file) as w:
            assert "open" in repr(w)
            w.add("t", np.array([1.0], dtype=np.float32))
        assert "finished" in repr(w)

        reader = Reader(temp_file)
        r = repr(reader)
        assert "1 tensor" in r
        assert "zt" in r

    def test_open_function(self, temp_file):
        """ztensor.open() works as entry point."""
        with Writer(temp_file) as w:
            w.add("data", np.array([1.0, 2.0], dtype=np.float32))

        reader = ztensor.open(temp_file)
        assert reader.keys() == ["data"]
        np.testing.assert_array_equal(reader["data"], [1.0, 2.0])


# ============================================================================
# 6. NUMPY CONVENIENCE TESTS
# ============================================================================


class TestNumPyConvenience:
    """Tests for ztensor.numpy module."""

    def test_save_load_file(self, temp_file):
        """numpy save_file/load_file roundtrip."""
        from ztensor.numpy import save_file, load_file

        tensors = {
            "a": np.random.randn(64, 128).astype(np.float32),
            "b": np.zeros(128, dtype=np.float32),
        }
        save_file(tensors, temp_file)
        loaded = load_file(temp_file)
        assert set(loaded.keys()) == {"a", "b"}
        np.testing.assert_array_equal(loaded["a"], tensors["a"])
        np.testing.assert_array_equal(loaded["b"], tensors["b"])

    def test_save_load_bytes(self):
        """numpy save/load bytes roundtrip."""
        from ztensor.numpy import save, load

        tensors = {"x": np.array([1, 2, 3], dtype=np.int32)}
        data = save(tensors)
        loaded = load(data)
        np.testing.assert_array_equal(loaded["x"], [1, 2, 3])

    def test_save_compressed(self, temp_file):
        """numpy save_file with compression."""
        from ztensor.numpy import save_file, load_file

        tensors = {"data": np.random.randn(100, 100).astype(np.float32)}
        save_file(tensors, temp_file, compression=True)
        loaded = load_file(temp_file)
        np.testing.assert_array_equal(loaded["data"], tensors["data"])


# ============================================================================
# 7. PYTORCH CONVENIENCE TESTS
# ============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPyTorchConvenience:
    """Tests for ztensor.torch module."""

    def test_save_load_file(self, temp_file):
        """torch save_file/load_file roundtrip."""
        from ztensor.torch import save_file, load_file

        tensors = {
            "weight": torch.randn(64, 128),
            "bias": torch.zeros(128),
        }
        save_file(tensors, temp_file)
        loaded = load_file(temp_file)
        assert set(loaded.keys()) == {"weight", "bias"}
        torch.testing.assert_close(loaded["weight"], tensors["weight"])
        torch.testing.assert_close(loaded["bias"], tensors["bias"])

    def test_save_load_bytes(self):
        """torch save/load bytes roundtrip."""
        from ztensor.torch import save, load

        tensors = {"x": torch.tensor([1, 2, 3], dtype=torch.int32)}
        data = save(tensors)
        loaded = load(data)
        torch.testing.assert_close(loaded["x"], tensors["x"])

    def test_torch_dtypes(self, temp_file):
        """Various PyTorch dtypes roundtrip."""
        from ztensor.torch import save_file, load_file

        tensors = {
            "f32": torch.randn(10, dtype=torch.float32),
            "f64": torch.randn(10, dtype=torch.float64),
            "f16": torch.randn(10, dtype=torch.float16),
            "i32": torch.tensor([1, 2, 3], dtype=torch.int32),
            "i64": torch.tensor([4, 5, 6], dtype=torch.int64),
            "u8": torch.tensor([7, 8, 9], dtype=torch.uint8),
            "bool": torch.tensor([True, False, True]),
        }
        save_file(tensors, temp_file)
        loaded = load_file(temp_file)
        for name in tensors:
            assert loaded[name].dtype == tensors[name].dtype
            torch.testing.assert_close(loaded[name], tensors[name])

    def test_save_load_model(self, temp_file):
        """save_model/load_model roundtrip."""
        from ztensor.torch import save_model, load_model

        model = torch.nn.Linear(10, 5)
        save_model(model, temp_file)

        model2 = torch.nn.Linear(10, 5)
        missing, unexpected = load_model(model2, temp_file)
        assert len(missing) == 0
        assert len(unexpected) == 0
        torch.testing.assert_close(model.weight, model2.weight)
        torch.testing.assert_close(model.bias, model2.bias)

    def test_shared_tensor_detection(self):
        """save_file rejects shared tensors."""
        from ztensor.torch import save_file

        t = torch.randn(10)
        tensors = {"a": t, "b": t}  # Same storage

        with pytest.raises(ValueError, match="share memory"):
            save_file(tensors, "/tmp/should_not_exist.zt")


# ============================================================================
# 8. CROSS-FORMAT READING TESTS
# ============================================================================

SAFETENSORS_AVAILABLE = False
try:
    import safetensors.numpy

    SAFETENSORS_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.skipif(not SAFETENSORS_AVAILABLE, reason="safetensors not installed")
class TestSafeTensorsReading:
    """Tests for reading HuggingFace SafeTensors files via ztensor."""

    def test_read_safetensors_numpy(self, temp_dir):
        """Read a .safetensors file written by the safetensors library."""
        path = os.path.join(temp_dir, "test.safetensors")
        tensors = {
            "weight": np.random.randn(64, 128).astype(np.float32),
            "bias": np.zeros(128, dtype=np.float32),
        }
        safetensors.numpy.save_file(tensors, path)

        reader = Reader(path)
        assert set(reader.keys()) == {"weight", "bias"}
        np.testing.assert_array_equal(reader["weight"], tensors["weight"])
        np.testing.assert_array_equal(reader["bias"], tensors["bias"])

    def test_read_safetensors_via_load_file(self, temp_dir):
        """ztensor.numpy.load_file reads .safetensors files."""
        from ztensor.numpy import load_file

        path = os.path.join(temp_dir, "test.safetensors")
        tensors = {"data": np.random.randn(100, 100).astype(np.float32)}
        safetensors.numpy.save_file(tensors, path)

        loaded = load_file(path)
        np.testing.assert_array_equal(loaded["data"], tensors["data"])

    def test_read_safetensors_multiple_dtypes(self, temp_dir):
        """Read safetensors with multiple dtypes."""
        path = os.path.join(temp_dir, "multi.safetensors")
        tensors = {
            "f32": np.array([1.0, 2.0], dtype=np.float32),
            "f64": np.array([3.0, 4.0], dtype=np.float64),
            "i32": np.array([5, 6], dtype=np.int32),
            "u8": np.array([7, 8], dtype=np.uint8),
        }
        safetensors.numpy.save_file(tensors, path)

        reader = Reader(path)
        for name in tensors:
            np.testing.assert_array_equal(reader[name], tensors[name])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_read_safetensors_as_torch(self, temp_dir):
        """ztensor.torch.load_file reads .safetensors files."""
        from ztensor.torch import load_file

        path = os.path.join(temp_dir, "torch_st.safetensors")
        tensors = {"w": np.random.randn(32, 64).astype(np.float32)}
        safetensors.numpy.save_file(tensors, path)

        loaded = load_file(path)
        assert isinstance(loaded["w"], torch.Tensor)
        np.testing.assert_allclose(loaded["w"].numpy(), tensors["w"], rtol=1e-6)

    def test_read_safetensors_metadata(self, temp_dir):
        """Metadata is accessible for safetensors tensors."""
        path = os.path.join(temp_dir, "meta.safetensors")
        tensors = {"layer": np.random.randn(10, 20).astype(np.float32)}
        safetensors.numpy.save_file(tensors, path)

        reader = Reader(path)
        meta = reader.metadata("layer")
        assert meta.dtype == "float32"
        assert list(meta.shape) == [10, 20]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestPyTorchFileReading:
    """Tests for reading PyTorch .pt files via ztensor."""

    def test_read_pt_simple(self, temp_dir):
        """Read a .pt file written by torch.save."""
        path = os.path.join(temp_dir, "model.pt")
        state_dict = {
            "weight": torch.randn(32, 64),
            "bias": torch.zeros(32),
        }
        torch.save(state_dict, path)

        reader = Reader(path)
        assert set(reader.keys()) == {"weight", "bias"}
        np.testing.assert_allclose(
            reader["weight"], state_dict["weight"].numpy(), rtol=1e-6
        )
        np.testing.assert_allclose(
            reader["bias"], state_dict["bias"].numpy(), rtol=1e-6
        )

    def test_read_pt_via_load_file(self, temp_dir):
        """ztensor.torch.load_file reads .pt files."""
        from ztensor.torch import load_file

        path = os.path.join(temp_dir, "model.pt")
        state_dict = {"data": torch.randn(100)}
        torch.save(state_dict, path)

        loaded = load_file(path)
        assert isinstance(loaded["data"], torch.Tensor)
        torch.testing.assert_close(loaded["data"], state_dict["data"])

    def test_read_pt_multiple_dtypes(self, temp_dir):
        """Read .pt file with multiple dtypes."""
        path = os.path.join(temp_dir, "dtypes.pt")
        state_dict = {
            "f32": torch.randn(10, dtype=torch.float32),
            "f64": torch.randn(10, dtype=torch.float64),
            "i64": torch.tensor([1, 2, 3], dtype=torch.int64),
        }
        torch.save(state_dict, path)

        reader = Reader(path)
        for name in state_dict:
            expected = state_dict[name].numpy()
            np.testing.assert_allclose(reader[name], expected, rtol=1e-6)

    def test_read_pt_model_state_dict(self, temp_dir):
        """Read a real model's state dict from .pt file."""
        path = os.path.join(temp_dir, "linear.pt")
        model = torch.nn.Linear(10, 5)
        torch.save(model.state_dict(), path)

        reader = Reader(path)
        keys = set(reader.keys())
        assert "weight" in keys
        assert "bias" in keys
        np.testing.assert_allclose(
            reader["weight"], model.weight.detach().numpy(), rtol=1e-6
        )
        np.testing.assert_allclose(
            reader["bias"], model.bias.detach().numpy(), rtol=1e-6
        )

    def test_read_pt_via_numpy_load_file(self, temp_dir):
        """ztensor.numpy.load_file reads .pt files."""
        from ztensor.numpy import load_file

        path = os.path.join(temp_dir, "np_pt.pt")
        state_dict = {"tensor": torch.randn(50)}
        torch.save(state_dict, path)

        loaded = load_file(path)
        assert isinstance(loaded["tensor"], np.ndarray)
        np.testing.assert_allclose(
            loaded["tensor"], state_dict["tensor"].numpy(), rtol=1e-6
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
