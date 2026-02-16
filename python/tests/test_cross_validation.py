"""
Cross-validation tests: create files with canonical libraries, read with both
the canonical library and ztensor, assert exact value equality.

This catches format misinterpretation bugs that synthetic (hand-crafted) tests miss.
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
    """Assert arrays are exactly equal (bitwise), with a nice message on failure."""
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=f"Mismatch for tensor '{name}': ztensor vs reference",
    )


# ---------------------------------------------------------------------------
# SafeTensors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_SAFETENSORS, reason="safetensors not installed")
class TestSafeTensors:
    DTYPES = [
        np.float32,
        np.float64,
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

    def test_single_tensor_all_dtypes(self, tmp_path_ext):
        """One tensor per dtype, created by safetensors, read by ztensor."""
        for dt in self.DTYPES:
            path = tmp_path_ext(".safetensors")
            if np.issubdtype(dt, np.floating):
                data = np.array([1.5, -2.25, 0.0, 3.75], dtype=dt)
            elif dt == np.bool_:
                data = np.array([True, False, True, False], dtype=dt)
            else:
                info = np.iinfo(dt)
                data = np.array([0, 1, info.max, info.min], dtype=dt)
            st_save({"t": data}, path)

            ref = st_load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"safetensors/{dt.__name__}")
            assert zt.dtype == ref.dtype

    def test_multidim_shapes(self, tmp_path_ext):
        """Scalar, 1-D, 2-D, 3-D shapes."""
        shapes = [(1,), (8,), (3, 4), (2, 3, 5)]
        for shape in shapes:
            path = tmp_path_ext(".safetensors")
            data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            st_save({"t": data}, path)

            ref = st_load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"safetensors/shape={shape}")
            assert list(zt.shape) == list(ref.shape)

    def test_multi_tensor(self, tmp_path_ext):
        """Multiple tensors in one file."""
        path = tmp_path_ext(".safetensors")
        tensors = {
            "weight": np.random.randn(64, 128).astype(np.float32),
            "bias": np.random.randn(128).astype(np.float32),
            "scale": np.array([0.5], dtype=np.float32),
        }
        st_save(tensors, path)

        reader = ztensor.Reader(path)
        ref = st_load(path)
        assert set(reader.keys()) == set(ref.keys())
        for name in ref:
            assert_close(reader[name], ref[name], f"safetensors/multi/{name}")

    def test_metadata_consistency(self, tmp_path_ext):
        """Verify ztensor metadata matches the actual array."""
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

    def _save_load_compare(self, path, tensors):
        """Save with torch.save, load with both torch.load and ztensor, compare."""
        torch.save(tensors, path)
        ref = torch.load(path, map_location="cpu", weights_only=True)
        reader = ztensor.Reader(path)
        for name in ref:
            ref_np = (
                ref[name].numpy()
                if ref[name].dtype != torch.bfloat16
                else ref[name].view(torch.uint16).numpy()
            )
            zt_np = reader[name]
            if ref[name].dtype == torch.bfloat16:
                # ztensor returns bf16 as ml_dtypes.bfloat16 numpy array;
                # compare raw uint16 view
                zt_np = zt_np.view(np.uint16)
            assert_close(zt_np, ref_np, f"pytorch/{name}")

    def test_single_tensor_all_dtypes(self, tmp_path_ext):
        """One tensor per dtype, created by torch.save, read by ztensor."""
        for dt in self.DTYPES:
            path = tmp_path_ext(".pt")
            if dt.is_floating_point:
                t = torch.tensor([1.5, -2.25, 0.0, 3.75], dtype=dt)
            elif dt == torch.bool:
                t = torch.tensor([True, False, True, False], dtype=dt)
            else:
                t = torch.tensor([0, 1, 42, -1], dtype=dt)
            self._save_load_compare(path, {"t": t})

    def test_multidim_shapes(self, tmp_path_ext):
        """Verify shape handling for 1-D through 4-D."""
        shapes = [(8,), (3, 4), (2, 3, 5), (2, 2, 2, 2)]
        for shape in shapes:
            path = tmp_path_ext(".pt")
            t = torch.arange(int(np.prod(shape)), dtype=torch.float32).reshape(shape)
            self._save_load_compare(path, {"t": t})

    def test_state_dict(self, tmp_path_ext):
        """Simulate a real model state_dict with multiple tensors."""
        path = tmp_path_ext(".pt")
        state = {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
            "layer2.weight": torch.randn(10, 64),
            "layer2.bias": torch.randn(10),
        }
        self._save_load_compare(path, state)

    def test_metadata_consistency(self, tmp_path_ext):
        """Verify ztensor metadata matches the actual array."""
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
    DTYPES = [
        np.float32,
        np.float64,
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

    def test_single_tensor_all_dtypes(self, tmp_path_ext):
        """One array per dtype, created by numpy, read by ztensor."""
        for dt in self.DTYPES:
            path = tmp_path_ext(".npz")
            if np.issubdtype(dt, np.floating):
                data = np.array([1.5, -2.25, 0.0, 3.75], dtype=dt)
            elif dt == np.bool_:
                data = np.array([True, False, True, False], dtype=dt)
            else:
                info = np.iinfo(dt)
                data = np.array([0, 1, info.max, info.min], dtype=dt)
            np.savez(path, t=data)

            ref = np.load(path)["t"]
            reader = ztensor.Reader(path)
            zt = reader["t"]
            assert_close(zt, ref, f"npz/{dt.__name__}")
            assert zt.dtype == ref.dtype

    def test_multidim_shapes(self, tmp_path_ext):
        """2-D and 3-D arrays."""
        for shape in [(3, 4), (2, 3, 5)]:
            path = tmp_path_ext(".npz")
            data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            np.savez(path, t=data)

            ref = np.load(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"npz/shape={shape}")
            assert list(zt.shape) == list(ref.shape)

    def test_multi_array(self, tmp_path_ext):
        """Multiple arrays in one .npz file."""
        path = tmp_path_ext(".npz")
        arrays = {
            "weight": np.random.randn(16, 32).astype(np.float32),
            "bias": np.random.randn(32).astype(np.float64),
            "mask": np.array([True, False, True], dtype=np.bool_),
        }
        np.savez(path, **arrays)

        reader = ztensor.Reader(path)
        ref = np.load(path)
        assert set(reader.keys()) == set(ref.files)
        for name in ref.files:
            assert_close(reader[name], ref[name], f"npz/multi/{name}")

    def test_compressed_npz(self, tmp_path_ext):
        """np.savez_compressed uses DEFLATE; verify ztensor still reads correctly."""
        path = tmp_path_ext(".npz")
        data = np.random.randn(100, 100).astype(np.float32)
        np.savez_compressed(path, t=data)

        ref = np.load(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert_close(zt, ref, "npz/compressed")

    def test_metadata_consistency(self, tmp_path_ext):
        """Verify ztensor metadata matches the actual array."""
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
    ONNX_DTYPES = [
        (np.float32, onnx.TensorProto.FLOAT if HAS_ONNX else 0),
        (np.float64, onnx.TensorProto.DOUBLE if HAS_ONNX else 0),
        (np.float16, onnx.TensorProto.FLOAT16 if HAS_ONNX else 0),
        (np.int64, onnx.TensorProto.INT64 if HAS_ONNX else 0),
        (np.int32, onnx.TensorProto.INT32 if HAS_ONNX else 0),
        (np.int16, onnx.TensorProto.INT16 if HAS_ONNX else 0),
        (np.int8, onnx.TensorProto.INT8 if HAS_ONNX else 0),
        (np.uint64, onnx.TensorProto.UINT64 if HAS_ONNX else 0),
        (np.uint32, onnx.TensorProto.UINT32 if HAS_ONNX else 0),
        (np.uint16, onnx.TensorProto.UINT16 if HAS_ONNX else 0),
        (np.uint8, onnx.TensorProto.UINT8 if HAS_ONNX else 0),
        (np.bool_, onnx.TensorProto.BOOL if HAS_ONNX else 0),
    ]

    def _make_model(self, tensors, path):
        """Create an ONNX model file with the given numpy arrays as initializers."""
        initializers = []
        for name, data in tensors.items():
            initializers.append(numpy_helper.from_array(data, name=name))
        graph = onnx.helper.make_graph(
            [],
            "test",
            [],
            [],
            initializer=initializers,
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )
        onnx.save(model, path)

    def _load_ref(self, path):
        """Load ONNX file with the reference library, return dict of numpy arrays."""
        model = onnx.load(path)
        result = {}
        for init in model.graph.initializer:
            result[init.name] = numpy_helper.to_array(init)
        return result

    def test_single_tensor_all_dtypes(self, tmp_path_ext):
        """One tensor per dtype, created by onnx, read by ztensor."""
        for dt, _ in self.ONNX_DTYPES:
            path = tmp_path_ext(".onnx")
            if np.issubdtype(dt, np.floating):
                data = np.array([1.5, -2.25, 0.0, 3.75], dtype=dt)
            elif dt == np.bool_:
                data = np.array([True, False, True, False], dtype=dt)
            else:
                info = np.iinfo(dt)
                data = np.array([0, 1, info.max, info.min], dtype=dt)
            self._make_model({"t": data}, path)

            ref = self._load_ref(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"onnx/{dt.__name__}")
            assert zt.dtype == ref.dtype

    def test_multidim_shapes(self, tmp_path_ext):
        """2-D and 3-D tensors."""
        for shape in [(3, 4), (2, 3, 5)]:
            path = tmp_path_ext(".onnx")
            data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
            self._make_model({"t": data}, path)

            ref = self._load_ref(path)["t"]
            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"onnx/shape={shape}")
            assert list(zt.shape) == list(ref.shape)

    def test_multi_tensor(self, tmp_path_ext):
        """Multiple initializers in one model."""
        path = tmp_path_ext(".onnx")
        tensors = {
            "weight": np.random.randn(16, 32).astype(np.float32),
            "bias": np.random.randn(32).astype(np.float32),
        }
        self._make_model(tensors, path)

        reader = ztensor.Reader(path)
        ref = self._load_ref(path)
        assert set(reader.keys()) == set(ref.keys())
        for name in ref:
            assert_close(reader[name], ref[name], f"onnx/multi/{name}")

    def test_metadata_consistency(self, tmp_path_ext):
        """Verify ztensor metadata matches the actual array."""
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
        result = {}
        for t in r.tensors:
            result[t.name] = np.array(t.data)
        return result

    def test_f32(self, tmp_path_ext):
        path = tmp_path_ext(".gguf")
        data = np.array([1.5, -2.25, 0.0, 3.75, 100.0], dtype=np.float32)
        self._write_gguf(path, {"t": data})

        ref = self._load_ref(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert_close(zt, ref, "gguf/f32")
        assert zt.dtype == np.float32

    def test_f16(self, tmp_path_ext):
        path = tmp_path_ext(".gguf")
        data = np.array([1.5, -2.25, 0.0, 3.75], dtype=np.float16)
        self._write_gguf(path, {"t": data})

        ref = self._load_ref(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert_close(zt, ref, "gguf/f16")
        assert zt.dtype == np.float16

    def test_i32(self, tmp_path_ext):
        path = tmp_path_ext(".gguf")
        data = np.array([0, 1, -1, 2147483647, -2147483648], dtype=np.int32)
        self._write_gguf(path, {"t": data})

        ref = self._load_ref(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert_close(zt, ref, "gguf/i32")
        assert zt.dtype == np.int32

    def test_multidim(self, tmp_path_ext):
        """2-D tensor in GGUF."""
        path = tmp_path_ext(".gguf")
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        self._write_gguf(path, {"t": data})

        ref = self._load_ref(path)["t"]
        zt = ztensor.Reader(path)["t"]
        assert_close(zt.flatten(), ref.flatten(), "gguf/2d/data")

    def test_multi_tensor(self, tmp_path_ext):
        """Multiple tensors in one GGUF file."""
        path = tmp_path_ext(".gguf")
        tensors = {
            "weight": np.random.randn(64).astype(np.float32),
            "bias": np.random.randn(16).astype(np.float32),
        }
        self._write_gguf(path, tensors)

        reader = ztensor.Reader(path)
        ref = self._load_ref(path)
        assert set(reader.keys()) == set(ref.keys())
        for name in ref:
            assert_close(
                reader[name].flatten(), ref[name].flatten(), f"gguf/multi/{name}"
            )

    def test_metadata_consistency(self, tmp_path_ext):
        """Verify ztensor metadata matches the actual array."""
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
    DTYPES = [
        np.float32,
        np.float64,
        np.float16,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
    ]

    def test_single_tensor_all_dtypes(self, tmp_path_ext):
        """One dataset per dtype, created by h5py, read by ztensor."""
        for dt in self.DTYPES:
            path = tmp_path_ext(".h5")
            if np.issubdtype(dt, np.floating):
                data = np.array([1.5, -2.25, 0.0, 3.75], dtype=dt)
            else:
                info = np.iinfo(dt)
                data = np.array([0, 1, info.max, info.min], dtype=dt)
            with h5py.File(path, "w") as hf:
                hf.create_dataset("t", data=data)

            with h5py.File(path, "r") as hf:
                ref = hf["t"][:]

            zt = ztensor.Reader(path)["t"]
            assert_close(zt, ref, f"hdf5/{dt.__name__}")
            assert zt.dtype == ref.dtype

    def test_multidim_shapes(self, tmp_path_ext):
        """2-D and 3-D datasets."""
        for shape in [(3, 4), (2, 3, 5)]:
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
            hf.create_dataset(
                "layer1/weight", data=np.array([1, 2, 3], dtype=np.float32)
            )
            hf.create_dataset(
                "layer1/bias", data=np.array([0.1, 0.2], dtype=np.float64)
            )
            hf.create_dataset("layer2/weight", data=np.array([4, 5], dtype=np.float32))

        reader = ztensor.Reader(path)
        # ztensor uses '.' as group separator
        with h5py.File(path, "r") as hf:

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    zt_name = name.replace("/", ".")
                    ref = obj[:]
                    zt = reader[zt_name]
                    assert_close(zt, ref, f"hdf5/groups/{name}")

            hf.visititems(visit)

    def test_multi_dataset(self, tmp_path_ext):
        """Multiple datasets at root level."""
        path = tmp_path_ext(".h5")
        with h5py.File(path, "w") as hf:
            hf.create_dataset("weight", data=np.random.randn(16, 32).astype(np.float32))
            hf.create_dataset("bias", data=np.random.randn(32).astype(np.float64))

        reader = ztensor.Reader(path)
        with h5py.File(path, "r") as hf:
            for name in hf:
                ref = hf[name][:]
                zt = reader[name]
                assert_close(zt, ref, f"hdf5/multi/{name}")

    def test_metadata_consistency(self, tmp_path_ext):
        """Verify ztensor metadata matches the actual array."""
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
# Cross-format: write with one library, verify all readers agree
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (HAS_SAFETENSORS and HAS_TORCH),
    reason="safetensors and torch required",
)
class TestCrossFormat:
    def test_same_data_safetensors_vs_pytorch(self, tmp_path_ext):
        """Write identical data as SafeTensors and PyTorch, verify ztensor reads match."""
        data = np.random.randn(32, 64).astype(np.float32)

        st_path = tmp_path_ext(".safetensors")
        st_save({"w": data}, st_path)

        pt_path = tmp_path_ext(".pt")
        torch.save({"w": torch.from_numpy(data.copy())}, pt_path)

        zt_st = ztensor.Reader(st_path)["w"]
        zt_pt = ztensor.Reader(pt_path)["w"]

        assert_close(zt_st, data, "cross/safetensors")
        assert_close(zt_pt, data, "cross/pytorch")
        assert_close(zt_st, zt_pt, "cross/st_vs_pt")
