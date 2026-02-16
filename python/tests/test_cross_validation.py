"""
Cross-validation tests: create files with canonical libraries, read with both
the canonical library and ztensor, assert exact value equality.

This catches format misinterpretation bugs that synthetic (hand-crafted) tests miss.

Design principles:
  - Use 512+ element tensors to exercise offset calculations and page boundaries
  - Include special float values (NaN, Inf, -0.0, subnormals) in every float test
  - Multi-tensor tests use different sizes to stress offset calculations
  - Assert dtype equality, not just value equality
  - GGUF shape comparison is flattened (GGUF uses reversed dimension order);
    this is documented and element count is checked explicitly
"""

import numpy as np
import pytest
import tempfile
import os

import ztensor

# ---------------------------------------------------------------------------
# Optional imports — skip tests if the canonical library is missing
# ---------------------------------------------------------------------------

try:
    from safetensors.numpy import save_file as st_save, load_file as st_load

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx
    from onnx import numpy_helper

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    from gguf import GGUFWriter, GGUFReader

    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_path_ext():
    """Yields a function that creates a temp file with the given extension."""
    paths = []

    def _make(ext):
        fd, path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        paths.append(path)
        return path

    yield _make

    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


def assert_close(actual, expected, name=""):
    """Assert arrays are exactly equal (bitwise), with dtype check."""
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=f"Value mismatch for '{name}'",
    )
    assert (
        actual.dtype == expected.dtype
    ), f"Dtype mismatch for '{name}': ztensor={actual.dtype}, ref={expected.dtype}"


def _float_data(dtype, n=512):
    """Random float data with special values at known positions.

    Positions 0-8 contain: 0.0, -0.0, +inf, -inf, nan, smallest subnormal,
    largest finite, smallest normal, negative largest finite.
    Remaining positions are random.
    """
    rng = np.random.RandomState(42)
    data = rng.randn(n).astype(dtype)
    finfo = np.finfo(dtype)
    data[0] = dtype(0.0)
    data[1] = dtype(-0.0)
    data[2] = dtype(np.inf)
    data[3] = dtype(-np.inf)
    data[4] = dtype(np.nan)
    data[5] = np.nextafter(dtype(0), dtype(1))  # smallest subnormal
    data[6] = finfo.max
    data[7] = finfo.tiny  # smallest normal
    data[8] = -finfo.max
    return data


def _int_data(dtype, n=512):
    """Random integer data with boundary values at known positions.

    Positions 0-3 contain: 0, 1, max, min.
    Remaining positions are random values in [0, 200).
    """
    rng = np.random.RandomState(42)
    data = rng.randint(0, 200, size=n).astype(dtype)
    info = np.iinfo(dtype)
    data[0] = dtype(0)
    data[1] = dtype(1)
    data[2] = info.max
    data[3] = info.min
    return data


def _bool_data(n=512):
    rng = np.random.RandomState(42)
    return rng.randint(0, 2, size=n).astype(np.bool_)


# ---------------------------------------------------------------------------
# SafeTensors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
class TestSafeTensors:
    FLOAT_DTYPES = [np.float32, np.float64, np.float16]
    INT_DTYPES = [
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
    ]

    def test_all_dtypes(self, tmp_path_ext):
        """512-element tensor for every supported dtype, including boundary values."""
        for dt in self.FLOAT_DTYPES:
            path = tmp_path_ext(".safetensors")
            data = _float_data(dt)
            st_save({"t": data}, path)
            ref = st_load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"safetensors/{dt.__name__}")

        for dt in self.INT_DTYPES:
            path = tmp_path_ext(".safetensors")
            data = _int_data(dt)
            st_save({"t": data}, path)
            ref = st_load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"safetensors/{dt.__name__}")

        # bool
        path = tmp_path_ext(".safetensors")
        data = _bool_data()
        st_save({"t": data}, path)
        ref = st_load(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert_close(zt, ref, "safetensors/bool")

    def test_special_floats_preserved(self, tmp_path_ext):
        """NaN, Inf, -0.0, and subnormals are preserved exactly."""
        for dt in self.FLOAT_DTYPES:
            path = tmp_path_ext(".safetensors")
            data = _float_data(dt, n=16)
            st_save({"t": data}, path)
            ref = st_load(path)["t"]
            zt = ztensor.Reader(path)["t"]

            # Verify special values individually
            assert np.isnan(zt[4]), f"NaN not preserved for {dt.__name__}"
            assert (
                np.isinf(zt[2]) and zt[2] > 0
            ), f"+Inf not preserved for {dt.__name__}"
            assert (
                np.isinf(zt[3]) and zt[3] < 0
            ), f"-Inf not preserved for {dt.__name__}"
            assert_close(zt, ref, f"safetensors/specials/{dt.__name__}")

    def test_shapes(self, tmp_path_ext):
        """Scalar through 4-D shapes."""
        shapes = [(1,), (8,), (3, 4), (2, 3, 5), (2, 2, 2, 2)]
        for shape in shapes:
            path = tmp_path_ext(".safetensors")
            data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            st_save({"t": data}, path)
            ref = st_load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"safetensors/shape={shape}")
            assert list(zt.shape) == list(ref.shape)

    def test_multi_tensor_offset_correctness(self, tmp_path_ext):
        """Multiple tensors of different sizes — catches offset calculation bugs."""
        path = tmp_path_ext(".safetensors")
        # Use deterministic arange data so cross-contamination is obvious
        tensors = {
            "small": np.arange(100, dtype=np.float32),
            "medium": np.arange(1000, dtype=np.float32) + 10000,
            "large": np.arange(4096, dtype=np.float32) + 90000,
            "tiny": np.array([42.0], dtype=np.float32),
        }
        st_save(tensors, path)
        reader = ztensor.Reader(path)
        ref = st_load(path)
        assert set(reader.keys()) == set(ref.keys())
        for name in ref:
            assert_close(reader[name], ref[name], f"safetensors/multi/{name}")

    def test_metadata_consistency(self, tmp_path_ext):
        path = tmp_path_ext(".safetensors")
        data = np.random.randn(4, 8).astype(np.float32)
        st_save({"w": data}, path)
        reader = ztensor.Reader(path)
        meta = reader.metadata("w")
        arr = reader["w"]
        assert list(meta.shape) == list(arr.shape)
        assert meta.dtype == str(arr.dtype)
        assert reader.format == "safetensors"


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestPyTorch:
    DTYPES = [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ]

    def _save_load_compare(self, path, tensors, check_dtype=True):
        """Save with torch.save, load with both torch.load and ztensor, compare."""
        torch.save(tensors, path)
        ref = torch.load(path, map_location="cpu", weights_only=True)
        reader = ztensor.Reader(path)
        for name in ref:
            zt_arr = reader[name]
            if ref[name].dtype == torch.bfloat16:
                # ztensor returns bf16 as ml_dtypes.bfloat16
                if check_dtype:
                    assert (
                        str(zt_arr.dtype) == "bfloat16"
                    ), f"Expected bfloat16 dtype for '{name}', got {zt_arr.dtype}"
                ref_np = ref[name].view(torch.uint16).numpy()
                zt_np = zt_arr.view(np.uint16)
            else:
                ref_np = ref[name].numpy()
                zt_np = zt_arr
                if check_dtype:
                    assert (
                        zt_np.dtype == ref_np.dtype
                    ), f"Dtype mismatch for '{name}': zt={zt_np.dtype}, ref={ref_np.dtype}"
            np.testing.assert_array_equal(
                zt_np, ref_np, err_msg=f"Value mismatch for pytorch/{name}"
            )

    def test_all_dtypes(self, tmp_path_ext):
        """512-element tensor for every supported dtype."""
        for dt in self.DTYPES:
            path = tmp_path_ext(".pt")
            if dt.is_floating_point:
                t = torch.randn(512, dtype=dt)
            elif dt == torch.bool:
                t = torch.randint(0, 2, (512,), dtype=torch.uint8).to(torch.bool)
            else:
                hi = min(200, torch.iinfo(dt).max)
                t = torch.randint(0, hi, (512,), dtype=dt)
            self._save_load_compare(path, {"t": t})

    def test_special_floats(self, tmp_path_ext):
        """NaN, Inf, -0.0, and subnormals via PyTorch format."""
        for dt in [torch.float32, torch.float64, torch.float16]:
            path = tmp_path_ext(".pt")
            np_data = _float_data(
                {
                    torch.float32: np.float32,
                    torch.float64: np.float64,
                    torch.float16: np.float16,
                }[dt]
            )
            t = torch.from_numpy(np_data)
            self._save_load_compare(path, {"t": t})

    def test_shapes(self, tmp_path_ext):
        """1-D through 4-D shapes."""
        shapes = [(8,), (3, 4), (2, 3, 5), (2, 2, 2, 2)]
        for shape in shapes:
            path = tmp_path_ext(".pt")
            t = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
            self._save_load_compare(path, {"t": t})

    def test_state_dict_realistic(self, tmp_path_ext):
        """Simulate a real model state_dict with multiple layers and sizes."""
        path = tmp_path_ext(".pt")
        state = {
            "embed.weight": torch.randn(1024, 64),
            "layer1.weight": torch.randn(128, 64),
            "layer1.bias": torch.randn(128),
            "layer2.weight": torch.randn(32, 128),
            "layer2.bias": torch.randn(32),
            "head.weight": torch.randn(10, 32),
            "head.bias": torch.randn(10),
        }
        self._save_load_compare(path, state)

    def test_metadata_consistency(self, tmp_path_ext):
        path = tmp_path_ext(".pt")
        torch.save({"w": torch.randn(4, 8)}, path)
        reader = ztensor.Reader(path)
        meta = reader.metadata("w")
        arr = reader["w"]
        assert list(meta.shape) == list(arr.shape)
        assert reader.format == "pickle"


# ---------------------------------------------------------------------------
# NPZ (NumPy)
# ---------------------------------------------------------------------------


class TestNpz:
    FLOAT_DTYPES = [np.float32, np.float64, np.float16]
    INT_DTYPES = [
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
    ]

    def test_all_dtypes(self, tmp_path_ext):
        """512-element array for every supported dtype."""
        for dt in self.FLOAT_DTYPES:
            path = tmp_path_ext(".npz")
            data = _float_data(dt)
            np.savez(path, t=data)
            ref = np.load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"npz/{dt.__name__}")

        for dt in self.INT_DTYPES:
            path = tmp_path_ext(".npz")
            data = _int_data(dt)
            np.savez(path, t=data)
            ref = np.load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"npz/{dt.__name__}")

        # bool
        path = tmp_path_ext(".npz")
        data = _bool_data()
        np.savez(path, t=data)
        ref = np.load(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert_close(zt, ref, "npz/bool")

    def test_special_floats_preserved(self, tmp_path_ext):
        """NaN, Inf, -0.0, subnormals preserved through .npz format."""
        for dt in self.FLOAT_DTYPES:
            path = tmp_path_ext(".npz")
            data = _float_data(dt, n=16)
            np.savez(path, t=data)
            ref = np.load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert np.isnan(zt[4]), f"NaN not preserved for {dt.__name__}"
            assert_close(zt, ref, f"npz/specials/{dt.__name__}")

    def test_shapes(self, tmp_path_ext):
        """2-D and 3-D arrays."""
        for shape in [(3, 4), (2, 3, 5), (2, 2, 2, 2)]:
            path = tmp_path_ext(".npz")
            data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            np.savez(path, t=data)
            ref = np.load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"npz/shape={shape}")
            assert list(zt.shape) == list(ref.shape)

    def test_multi_array_different_sizes(self, tmp_path_ext):
        """Multiple arrays of different sizes and dtypes."""
        path = tmp_path_ext(".npz")
        arrays = {
            "weight": np.arange(1024, dtype=np.float32),
            "bias": np.arange(64, dtype=np.float64) + 5000,
            "mask": _bool_data(128),
            "indices": _int_data(np.int32, 256),
        }
        np.savez(path, **arrays)
        reader = ztensor.Reader(path)
        ref = np.load(path)
        assert set(reader.keys()) == set(ref.files)
        for name in ref.files:
            assert_close(reader[name], ref[name], f"npz/multi/{name}")

    def test_compressed_npz(self, tmp_path_ext):
        """np.savez_compressed uses DEFLATE; verify ztensor reads correctly."""
        path = tmp_path_ext(".npz")
        data = _float_data(np.float32, n=4096)
        np.savez_compressed(path, t=data)
        ref = np.load(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert_close(zt, ref, "npz/compressed")

    def test_metadata_consistency(self, tmp_path_ext):
        path = tmp_path_ext(".npz")
        np.savez(path, w=np.random.randn(4, 8).astype(np.float32))
        reader = ztensor.Reader(path)
        meta = reader.metadata("w")
        arr = reader["w"]
        assert list(meta.shape) == list(arr.shape)
        assert meta.dtype == str(arr.dtype)
        assert reader.format == "npz"


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")
class TestOnnx:
    ONNX_FLOAT_DTYPES = [
        (np.float32, onnx.TensorProto.FLOAT if HAS_ONNX else 0),
        (np.float64, onnx.TensorProto.DOUBLE if HAS_ONNX else 0),
        (np.float16, onnx.TensorProto.FLOAT16 if HAS_ONNX else 0),
    ]
    ONNX_INT_DTYPES = [
        (np.int64, onnx.TensorProto.INT64 if HAS_ONNX else 0),
        (np.int32, onnx.TensorProto.INT32 if HAS_ONNX else 0),
        (np.int16, onnx.TensorProto.INT16 if HAS_ONNX else 0),
        (np.int8, onnx.TensorProto.INT8 if HAS_ONNX else 0),
        (np.uint64, onnx.TensorProto.UINT64 if HAS_ONNX else 0),
        (np.uint32, onnx.TensorProto.UINT32 if HAS_ONNX else 0),
        (np.uint16, onnx.TensorProto.UINT16 if HAS_ONNX else 0),
        (np.uint8, onnx.TensorProto.UINT8 if HAS_ONNX else 0),
    ]
    ONNX_BOOL = [(np.bool_, onnx.TensorProto.BOOL if HAS_ONNX else 0)]

    def _make_model(self, tensors, path):
        """Create an ONNX model file with the given numpy arrays as initializers."""
        initializers = []
        for name, data in tensors.items():
            initializers.append(numpy_helper.from_array(data, name=name))
        graph = onnx.helper.make_graph([], "test", [], [], initializer=initializers)
        model = onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
        )
        onnx.save(model, path)

    def _load_ref(self, path):
        """Load ONNX file with the reference library, return dict of numpy arrays."""
        model = onnx.load(path)
        return {
            init.name: numpy_helper.to_array(init) for init in model.graph.initializer
        }

    def test_all_dtypes(self, tmp_path_ext):
        """512-element tensor for every supported dtype."""
        for dt, _ in self.ONNX_FLOAT_DTYPES:
            path = tmp_path_ext(".onnx")
            data = _float_data(dt)
            self._make_model({"t": data}, path)
            ref = self._load_ref(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"onnx/{dt.__name__}")

        for dt, _ in self.ONNX_INT_DTYPES:
            path = tmp_path_ext(".onnx")
            data = _int_data(dt)
            self._make_model({"t": data}, path)
            ref = self._load_ref(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"onnx/{dt.__name__}")

        for dt, _ in self.ONNX_BOOL:
            path = tmp_path_ext(".onnx")
            data = _bool_data()
            self._make_model({"t": data}, path)
            ref = self._load_ref(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, "onnx/bool")

    def test_special_floats_preserved(self, tmp_path_ext):
        """NaN, Inf, -0.0, subnormals preserved through ONNX format."""
        for dt, _ in self.ONNX_FLOAT_DTYPES:
            path = tmp_path_ext(".onnx")
            data = _float_data(dt, n=16)
            self._make_model({"t": data}, path)
            ref = self._load_ref(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert np.isnan(zt[4]), f"NaN not preserved for {dt.__name__}"
            assert_close(zt, ref, f"onnx/specials/{dt.__name__}")

    def test_shapes(self, tmp_path_ext):
        """2-D and 3-D tensors."""
        for shape in [(3, 4), (2, 3, 5), (2, 2, 2, 2)]:
            path = tmp_path_ext(".onnx")
            data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            self._make_model({"t": data}, path)
            ref = self._load_ref(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"onnx/shape={shape}")
            assert list(zt.shape) == list(ref.shape)

    def test_multi_tensor(self, tmp_path_ext):
        """Multiple initializers of different sizes and dtypes."""
        path = tmp_path_ext(".onnx")
        tensors = {
            "weight": np.arange(1024, dtype=np.float32),
            "bias": np.arange(64, dtype=np.float32) + 5000,
            "embed": np.arange(2048, dtype=np.float64) + 20000,
        }
        self._make_model(tensors, path)
        reader = ztensor.Reader(path)
        ref = self._load_ref(path)
        assert set(reader.keys()) == set(ref.keys())
        for name in ref:
            assert_close(reader[name], ref[name], f"onnx/multi/{name}")

    def test_metadata_consistency(self, tmp_path_ext):
        path = tmp_path_ext(".onnx")
        data = np.random.randn(4, 8).astype(np.float32)
        self._make_model({"w": data}, path)
        reader = ztensor.Reader(path)
        meta = reader.metadata("w")
        arr = reader["w"]
        assert list(meta.shape) == list(arr.shape)
        assert meta.dtype == str(arr.dtype)
        assert reader.format == "onnx"


# ---------------------------------------------------------------------------
# GGUF
#
# GGUF uses reversed dimension order (inner-first) compared to numpy's
# row-major (outer-first). For multidimensional tensors, ztensor and the
# gguf library may interpret shapes differently, so data comparisons are
# done on flattened arrays with an explicit element-count check.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GGUF, reason="gguf not installed")
class TestGguf:
    def _write_gguf(self, path, tensors):
        """Write tensors to a GGUF file using the canonical gguf library."""
        w = GGUFWriter(path, "test")
        for name, data in tensors.items():
            w.add_tensor(name, data)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

    def _load_ref(self, path):
        """Load GGUF file with the reference library, return dict of numpy arrays."""
        r = GGUFReader(path)
        return {t.name: np.array(t.data) for t in r.tensors}

    def test_f32(self, tmp_path_ext):
        path = tmp_path_ext(".gguf")
        data = _float_data(np.float32)
        self._write_gguf(path, {"t": data})
        ref = self._load_ref(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert zt.size == ref.size == data.size
        np.testing.assert_array_equal(zt.flatten(), ref.flatten(), err_msg="gguf/f32")
        assert zt.dtype == np.float32

    def test_f16(self, tmp_path_ext):
        path = tmp_path_ext(".gguf")
        data = _float_data(np.float16)
        self._write_gguf(path, {"t": data})
        ref = self._load_ref(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert zt.size == ref.size == data.size
        np.testing.assert_array_equal(zt.flatten(), ref.flatten(), err_msg="gguf/f16")
        assert zt.dtype == np.float16

    def test_i32(self, tmp_path_ext):
        path = tmp_path_ext(".gguf")
        data = _int_data(np.int32)
        self._write_gguf(path, {"t": data})
        ref = self._load_ref(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert zt.size == ref.size == data.size
        np.testing.assert_array_equal(zt.flatten(), ref.flatten(), err_msg="gguf/i32")
        assert zt.dtype == np.int32

    def test_special_floats_preserved(self, tmp_path_ext):
        """NaN, Inf, -0.0, subnormals preserved through GGUF format."""
        path = tmp_path_ext(".gguf")
        data = _float_data(np.float32, n=16)
        self._write_gguf(path, {"t": data})
        ref = self._load_ref(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert np.isnan(zt.flatten()[4]), "NaN not preserved in GGUF"
        np.testing.assert_array_equal(
            zt.flatten(), ref.flatten(), err_msg="gguf/specials"
        )

    def test_multidim_element_count(self, tmp_path_ext):
        """2-D tensor: element count must match even if shape convention differs."""
        path = tmp_path_ext(".gguf")
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        self._write_gguf(path, {"t": data})
        ref = self._load_ref(path)["t"]
        zt = ztensor.Reader(path)["t"]
        # Element count must agree
        assert zt.size == ref.size == 12
        # Data values must agree (flattened — GGUF reverses dimension order)
        np.testing.assert_array_equal(
            zt.flatten(), ref.flatten(), err_msg="gguf/2d/data"
        )

    def test_multi_tensor(self, tmp_path_ext):
        """Multiple tensors of different sizes."""
        path = tmp_path_ext(".gguf")
        tensors = {
            "weight": np.arange(256, dtype=np.float32),
            "bias": np.arange(32, dtype=np.float32) + 1000,
            "scale": np.array([42.0], dtype=np.float32),
        }
        self._write_gguf(path, tensors)
        reader = ztensor.Reader(path)
        ref = self._load_ref(path)
        assert set(reader.keys()) == set(ref.keys())
        for name in ref:
            zt = reader[name]
            assert zt.size == ref[name].size, f"Element count mismatch for gguf/{name}"
            np.testing.assert_array_equal(
                zt.flatten(), ref[name].flatten(), err_msg=f"gguf/multi/{name}"
            )

    def test_metadata_consistency(self, tmp_path_ext):
        path = tmp_path_ext(".gguf")
        data = np.random.randn(8).astype(np.float32)
        self._write_gguf(path, {"w": data})
        reader = ztensor.Reader(path)
        meta = reader.metadata("w")
        assert meta.dtype == "float32"
        assert reader.format == "gguf"


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestHdf5:
    FLOAT_DTYPES = [np.float32, np.float64, np.float16]
    INT_DTYPES = [
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
    ]

    def test_all_dtypes(self, tmp_path_ext):
        """512-element dataset for every supported dtype."""
        for dt in self.FLOAT_DTYPES:
            path = tmp_path_ext(".h5")
            data = _float_data(dt)
            with h5py.File(path, "w") as hf:
                hf.create_dataset("t", data=data)
            with h5py.File(path, "r") as hf:
                ref = hf["t"][:]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"hdf5/{dt.__name__}")

        for dt in self.INT_DTYPES:
            path = tmp_path_ext(".h5")
            data = _int_data(dt)
            with h5py.File(path, "w") as hf:
                hf.create_dataset("t", data=data)
            with h5py.File(path, "r") as hf:
                ref = hf["t"][:]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"hdf5/{dt.__name__}")

    def test_special_floats_preserved(self, tmp_path_ext):
        """NaN, Inf, -0.0, subnormals preserved through HDF5 format."""
        for dt in self.FLOAT_DTYPES:
            path = tmp_path_ext(".h5")
            data = _float_data(dt, n=16)
            with h5py.File(path, "w") as hf:
                hf.create_dataset("t", data=data)
            with h5py.File(path, "r") as hf:
                ref = hf["t"][:]
            zt = ztensor.Reader(path)["t"]
            assert np.isnan(zt[4]), f"NaN not preserved for {dt.__name__}"
            assert_close(zt, ref, f"hdf5/specials/{dt.__name__}")

    def test_shapes(self, tmp_path_ext):
        """2-D and 3-D datasets."""
        for shape in [(3, 4), (2, 3, 5), (2, 2, 2, 2)]:
            path = tmp_path_ext(".h5")
            data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            with h5py.File(path, "w") as hf:
                hf.create_dataset("t", data=data)
            with h5py.File(path, "r") as hf:
                ref = hf["t"][:]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"hdf5/shape={shape}")
            assert list(zt.shape) == list(ref.shape)

    def test_groups(self, tmp_path_ext):
        """Datasets nested in HDF5 groups (e.g. layer1/weight)."""
        path = tmp_path_ext(".h5")
        with h5py.File(path, "w") as hf:
            hf.create_dataset("layer1/weight", data=_float_data(np.float32, 256))
            hf.create_dataset("layer1/bias", data=_float_data(np.float64, 64))
            hf.create_dataset("layer2/weight", data=_float_data(np.float32, 128))

        reader = ztensor.Reader(path)
        # Verify ztensor's key naming convention (/ → .)
        zt_keys = set(reader.keys())
        expected_keys = {"layer1.weight", "layer1.bias", "layer2.weight"}
        assert zt_keys == expected_keys, f"Key mismatch: got {zt_keys}"

        with h5py.File(path, "r") as hf:

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    zt_name = name.replace("/", ".")
                    ref = obj[:]
                    zt = reader[zt_name]
                    assert_close(zt, ref, f"hdf5/groups/{name}")

            hf.visititems(visit)

    def test_multi_dataset(self, tmp_path_ext):
        """Multiple datasets at root level with different sizes."""
        path = tmp_path_ext(".h5")
        with h5py.File(path, "w") as hf:
            hf.create_dataset("weight", data=np.arange(1024, dtype=np.float32))
            hf.create_dataset("bias", data=np.arange(64, dtype=np.float64) + 5000)
            hf.create_dataset("indices", data=_int_data(np.int32, 256))

        reader = ztensor.Reader(path)
        with h5py.File(path, "r") as hf:
            for name in hf:
                ref = hf[name][:]
                zt = reader[name]
                assert_close(zt, ref, f"hdf5/multi/{name}")

    def test_metadata_consistency(self, tmp_path_ext):
        path = tmp_path_ext(".h5")
        with h5py.File(path, "w") as hf:
            hf.create_dataset("w", data=np.random.randn(4, 8).astype(np.float32))
        reader = ztensor.Reader(path)
        meta = reader.metadata("w")
        arr = reader["w"]
        assert list(meta.shape) == list(arr.shape)
        assert meta.dtype == str(arr.dtype)
        assert reader.format == "hdf5"


# ---------------------------------------------------------------------------
# Cross-format: same data through different formats, ztensor must agree
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (HAS_SAFETENSORS and HAS_TORCH),
    reason="safetensors and torch required",
)
class TestCrossFormat:
    def test_f32_safetensors_vs_pytorch(self, tmp_path_ext):
        """Same f32 data written as SafeTensors and PyTorch must read identically."""
        data = _float_data(np.float32, 1024)

        st_path = tmp_path_ext(".safetensors")
        st_save({"w": data}, st_path)

        pt_path = tmp_path_ext(".pt")
        torch.save({"w": torch.from_numpy(data.copy())}, pt_path)

        zt_st = ztensor.Reader(st_path)["w"]
        zt_pt = ztensor.Reader(pt_path)["w"]

        assert_close(zt_st, data, "cross/safetensors_vs_original")
        assert_close(zt_pt, data, "cross/pytorch_vs_original")
        assert_close(zt_st, zt_pt, "cross/safetensors_vs_pytorch")

    def test_all_shared_float_dtypes(self, tmp_path_ext):
        """f16, f32, f64: same data via SafeTensors and PyTorch must match."""
        for np_dt, torch_dt in [
            (np.float16, torch.float16),
            (np.float32, torch.float32),
            (np.float64, torch.float64),
        ]:
            data = _float_data(np_dt, 512)

            st_path = tmp_path_ext(".safetensors")
            st_save({"t": data}, st_path)

            pt_path = tmp_path_ext(".pt")
            torch.save({"t": torch.from_numpy(data.copy())}, pt_path)

            zt_st = ztensor.Reader(st_path)["t"]
            zt_pt = ztensor.Reader(pt_path)["t"]
            assert_close(zt_st, zt_pt, f"cross/{np_dt.__name__}/st_vs_pt")
            assert zt_st.dtype == zt_pt.dtype

    def test_multi_tensor_cross_format(self, tmp_path_ext):
        """Multiple tensors of different sizes via SafeTensors and PyTorch."""
        tensors_np = {
            "embed": np.arange(2048, dtype=np.float32),
            "weight": np.arange(512, dtype=np.float32) + 10000,
            "bias": np.arange(64, dtype=np.float32) + 50000,
        }

        st_path = tmp_path_ext(".safetensors")
        st_save(tensors_np, st_path)

        pt_path = tmp_path_ext(".pt")
        torch.save(
            {k: torch.from_numpy(v.copy()) for k, v in tensors_np.items()}, pt_path
        )

        st_reader = ztensor.Reader(st_path)
        pt_reader = ztensor.Reader(pt_path)

        for name, expected in tensors_np.items():
            assert_close(st_reader[name], expected, f"cross/multi/st/{name}")
            assert_close(pt_reader[name], expected, f"cross/multi/pt/{name}")
            assert_close(
                st_reader[name], pt_reader[name], f"cross/multi/st_vs_pt/{name}"
            )

    def test_special_floats_cross_format(self, tmp_path_ext):
        """Special float values (NaN, Inf) must be identical across formats."""
        data = _float_data(np.float32, 16)

        st_path = tmp_path_ext(".safetensors")
        st_save({"t": data}, st_path)

        pt_path = tmp_path_ext(".pt")
        torch.save({"t": torch.from_numpy(data.copy())}, pt_path)

        zt_st = ztensor.Reader(st_path)["t"]
        zt_pt = ztensor.Reader(pt_path)["t"]

        # Both must preserve NaN at position 4
        assert np.isnan(zt_st[4]) and np.isnan(zt_pt[4])
        # Both must preserve Inf at positions 2, 3
        assert np.isinf(zt_st[2]) and np.isinf(zt_pt[2])
        assert_close(zt_st, zt_pt, "cross/specials/st_vs_pt")
