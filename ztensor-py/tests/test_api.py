"""The Python surface, against the promises it makes.

Run with the extension module built:

    maturin develop -m ztensor-py/Cargo.toml
    pytest ztensor-py/tests

There were no tests here before, which is how a binding ends up reporting an
arbitrary part's dtype and flattening every bf16 tensor to bytes. Every case
below is one of the claims the API makes out loud.
"""

from __future__ import annotations

import builtins
import struct

import pytest

import ztensor

try:
    import numpy
except ImportError:  # the core API does not need it; the shim and DLPack do
    numpy = None

requires_numpy = pytest.mark.skipif(numpy is None, reason="numpy is not installed")


def f32(*vals) -> bytes:
    return struct.pack(f"<{len(vals)}f", *vals)


@pytest.fixture
def simple(tmp_path):
    """A two-tensor canonical file."""
    path = tmp_path / "simple.zt"
    with ztensor.Writer(str(path)) as w:
        w.add("a.weight", f32(1, 2, 3, 4, 5, 6), shape=[2, 3], dtype="f32")
        w.add("b.bias", bytes([7] * 8), shape=[8], dtype="u8")
    return path


# ---- the source is a mapping over tensors ----------------------------------


def test_source_is_iterable_and_indexable(simple):
    with ztensor.open(str(simple)) as src:
        assert len(src) == 2
        assert "a.weight" in src
        assert "nope" not in src
        assert [t.name for t in src] == ["a.weight", "b.bias"]
        assert src.names() == ["a.weight", "b.bias"]
        with pytest.raises(KeyError):
            src["nope"]


def test_closing_is_explicit(simple):
    src = ztensor.open(str(simple))
    assert len(src) == 2
    src.close()
    with pytest.raises(ValueError):
        len(src)


def test_tensor_metadata(simple):
    with ztensor.open(str(simple)) as src:
        t = src["a.weight"]
        assert t.name == "a.weight"
        assert t.shape == [2, 3]
        assert t.dtype == "f32"
        assert t.logical is None
        assert t.layout == "dense"
        assert t.nbytes == 24
        assert t.parts == ["data"]
        assert t.part == "data"


# ---- getting bytes ---------------------------------------------------------


def test_tobytes_round_trips(simple):
    with ztensor.open(str(simple)) as src:
        assert src["a.weight"].tobytes() == f32(1, 2, 3, 4, 5, 6)
        assert src["b.bias"].tobytes() == bytes([7] * 8)


def test_buffer_protocol_is_zero_copy(simple):
    with ztensor.open(str(simple)) as src:
        t = src["a.weight"]
        view = memoryview(t)
        assert view.readonly
        assert len(view) == 24
        assert bytes(view) == f32(1, 2, 3, 4, 5, 6)


@requires_numpy
def test_dlpack_gives_a_typed_array(simple):
    with ztensor.open(str(simple)) as src:
        arr = numpy.from_dlpack(src["a.weight"])
        assert arr.dtype == numpy.float32
        assert arr.shape == (2, 3)
        assert arr[1, 2] == 6.0


@requires_numpy
def test_dlpack_array_outlives_the_handle(simple):
    """A zero-copy export holds the mapping itself.

    Closing the source it came from must not pull the memory out from under an
    array that was already handed over — the reference has to be to the
    mapping, not to the Python object that produced it.
    """
    with ztensor.open(str(simple)) as src:
        arr = numpy.from_dlpack(src["a.weight"])
    del src
    assert arr[0, 0] == 1.0
    assert arr[1, 2] == 6.0


def test_a_memoryview_outlives_the_handle(simple):
    """The same promise, through the buffer protocol."""
    src = ztensor.open(str(simple))
    view = memoryview(src["a.weight"])
    src.close()
    del src
    assert bytes(view) == f32(1, 2, 3, 4, 5, 6)
    view.release()


def test_location_is_an_address(simple):
    with ztensor.open(str(simple)) as src:
        at = src["a.weight"].location
        assert at.path == str(simple)
        assert at.nbytes == 24
        assert at.offset % 65536 == 0, "canonical placement puts it on a page"
        with builtins.open(simple, "rb") as f:
            f.seek(at.offset)
            assert f.read(at.nbytes) == f32(1, 2, 3, 4, 5, 6)



# ---- capabilities ----------------------------------------------------------


def test_caps_match_what_happens(simple):
    with ztensor.open(str(simple)) as src:
        for t in src:
            caps = t.caps
            assert caps.map is t.is_mapped()
            assert caps.verify is t.verify()
            assert caps.alignment >= 65536
            if caps.locate:
                assert t.location.nbytes == t.nbytes
            else:
                with pytest.raises(ValueError):
                    t.location


def test_an_indexed_source_locates_but_does_not_map(simple):
    with ztensor.index(str(simple)) as src:
        t = src["a.weight"]
        assert t.caps.locate
        assert not t.caps.map
        assert t.tobytes() == f32(1, 2, 3, 4, 5, 6)
        with pytest.raises((BufferError, ValueError)):
            memoryview(t)


# ---- parts -----------------------------------------------------------------


def test_parts_are_tensors_too(tmp_path):
    path = tmp_path / "quant.zt"
    with ztensor.Writer(str(path), canonical=False, align=4096) as w:
        w.add("q", bytes([1] * 16), shape=[32], dtype="u8", logical="f4_e2m1")
    with ztensor.open(str(path)) as src:
        t = src["q"]
        assert t.logical == "f4_e2m1"
        assert t["data"].nbytes == 16
        assert t["data"].name == "q"
        assert t["data"].part == "data"
        with pytest.raises(KeyError):
            t["scales"]


@requires_numpy
def test_a_logical_type_numpy_cannot_name_is_refused_not_reinterpreted(tmp_path):
    path = tmp_path / "fp4.zt"
    with ztensor.Writer(str(path), canonical=False, align=4096) as w:
        w.add("q", bytes([1] * 16), shape=[32], dtype="u8", logical="f4_e2m1")
    with ztensor.open(str(path)) as src:
        with pytest.raises(TypeError):
            numpy.from_dlpack(src["q"])
        assert len(src["q"].tobytes()) == 16


# ---- DLPack, in detail -----------------------------------------------------


def capsule_pointer(capsule, name: bytes):
    import ctypes

    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.pythonapi.PyCapsule_GetPointer(capsule, name)


def versioned(capsule):
    """The `DLManagedTensorVersioned` behind a versioned capsule.

    The caller must keep `capsule` alive for as long as it reads the struct:
    dropping it runs the destructor, which frees exactly this memory.
    """
    import ctypes

    class DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class DLDataType(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int),
            ("dtype", DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class DLPackVersion(ctypes.Structure):
        _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]

    class Managed(ctypes.Structure):
        _fields_ = [
            ("version", DLPackVersion),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
            ("flags", ctypes.c_uint64),
            ("dl_tensor", DLTensor),
        ]

    ptr = capsule_pointer(capsule, b"dltensor_versioned")
    return ctypes.cast(ptr, ctypes.POINTER(Managed)).contents


READ_ONLY = 1


def test_a_versioned_consumer_is_told_the_bytes_are_read_only(simple):
    """The whole reason to implement the versioned protocol.

    Legacy DLPack has no way to say read-only, so a framework is free to
    believe it may write into a read-only mapping. The versioned struct can
    say it, so it does.
    """
    with ztensor.open(str(simple)) as src:
        cap = src["a.weight"].__dlpack__(max_version=(1, 0))
        m = versioned(cap)
        assert (m.version.major, m.version.minor) == (1, 0)
        assert m.flags & READ_ONLY, "a mapping is not the consumer's to write"
        assert m.dl_tensor.ndim == 2
        assert [m.dl_tensor.shape[i] for i in range(2)] == [2, 3]
        assert (m.dl_tensor.dtype.code, m.dl_tensor.dtype.bits) == (2, 32)  # kDLFloat


def test_an_older_consumer_still_gets_a_legacy_capsule(simple):
    with ztensor.open(str(simple)) as src:
        cap = src["a.weight"].__dlpack__()
        assert "dltensor" in repr(cap) and "versioned" not in repr(cap)
        assert capsule_pointer(cap, b"dltensor")


def test_copy_true_hands_over_a_buffer_the_consumer_owns(simple):
    with ztensor.open(str(simple)) as src:
        cap = src["a.weight"].__dlpack__(max_version=(1, 0), copy=True)
        m = versioned(cap)
        assert not (m.flags & READ_ONLY), "a copy is the consumer's to write"


def test_another_device_is_refused_not_quietly_served(simple):
    with ztensor.open(str(simple)) as src:
        t = src["a.weight"]
        assert t.__dlpack_device__() == (1, 0)  # kDLCPU
        with pytest.raises(BufferError):
            t.__dlpack__(dl_device=(2, 0))  # kDLCUDA
        with pytest.raises(ValueError):
            t.__dlpack__(stream=object())


@pytest.fixture
def encoded(tmp_path):
    """A tensor whose bytes have to be decoded — no range to point at."""
    path = tmp_path / "encoded.zt"
    with ztensor.Writer(str(path), canonical=False, align=4096) as w:
        w.add(
            "z",
            bytes(4096),
            shape=[4096],
            dtype="u8",
            encoding="zt.zstd-seekable/1",
        )
    return path


def test_an_undecoded_part_is_copied_or_refused_never_faked(encoded):
    with ztensor.open(str(encoded)) as src:
        t = src["z"]
        assert not t.caps.map and not t.caps.locate
        # copy=False cannot be honoured, and saying so is the point.
        with pytest.raises(BufferError):
            t.__dlpack__(max_version=(1, 0), copy=False)
        # Left to choose, it decodes into a buffer the consumer owns.
        cap = t.__dlpack__(max_version=(1, 0))
        m = versioned(cap)
        assert not (m.flags & READ_ONLY)
        assert m.dl_tensor.ndim == 1


@requires_numpy
def test_a_decoded_tensor_still_reaches_numpy(encoded):
    with ztensor.open(str(encoded)) as src:
        arr = numpy.from_dlpack(src["z"])
        assert arr.shape == (4096,)
        assert arr.dtype == numpy.uint8


# ---- several files as one --------------------------------------------------


def test_open_accepts_a_list(tmp_path):
    first, second = tmp_path / "a.zt", tmp_path / "b.zt"
    with ztensor.Writer(str(first)) as w:
        w.add("x", bytes([1] * 4), shape=[4], dtype="u8")
    with ztensor.Writer(str(second)) as w:
        w.add("y", bytes([2] * 4), shape=[4], dtype="u8")
    with ztensor.open([str(first), str(second)]) as src:
        assert src.names() == ["x", "y"]
        assert sorted(src.files()) == sorted([str(first), str(second)])


def test_a_name_in_two_files_is_refused(tmp_path):
    first, second = tmp_path / "a.zt", tmp_path / "b.zt"
    for path in (first, second):
        with ztensor.Writer(str(path)) as w:
            w.add("shared", bytes([1] * 4), shape=[4], dtype="u8")
    with pytest.raises(ValueError, match="shared"):
        ztensor.open([str(first), str(second)])


# ---- writing ---------------------------------------------------------------


def test_canonical_output_is_byte_identical(tmp_path):
    def write(path):
        with ztensor.Writer(str(path)) as w:
            w.add("x", f32(1, 2), shape=[2], dtype="f32")
            w.add("y", bytes([9, 9]), shape=[2], dtype="u8")

    a, b = tmp_path / "1.zt", tmp_path / "2.zt"
    write(a)
    write(b)
    assert a.read_bytes() == b.read_bytes()


def test_publish_leaves_nothing_behind_on_failure(tmp_path):
    path = tmp_path / "published.zt"
    with pytest.raises(RuntimeError):
        with ztensor.Writer(str(path), publish=True) as w:
            w.add("x", bytes([1] * 4), shape=[4], dtype="u8")
            raise RuntimeError("boom")
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_publish_appears_only_when_finished(tmp_path):
    path = tmp_path / "published.zt"
    w = ztensor.Writer(str(path), publish=True)
    w.add("x", bytes([1] * 4), shape=[4], dtype="u8")
    assert not path.exists()
    w.finish()
    assert path.exists()


def test_attributes_round_trip(tmp_path):
    path = tmp_path / "attrs.zt"
    with ztensor.Writer(str(path)) as w:
        w.set_attributes({"producer": "test", "group": 32})
        w.add("x", bytes([1] * 4), shape=[4], dtype="u8")
    with ztensor.open(str(path)) as src:
        assert src.attributes() == {"producer": "test", "group": 32}


# ---- conversion and verification -------------------------------------------


def test_verify_reports_both_halves(simple):
    checked, undigested = ztensor.verify(str(simple))
    assert (checked, undigested) == (2, 0)


def test_convert_adds_what_a_projection_could_not(tmp_path):
    st = tmp_path / "model.safetensors"
    payload = f32(1, 2, 3, 4)
    header = b'{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}'
    st.write_bytes(struct.pack("<Q", len(header)) + header + payload)

    assert ztensor.detect(str(st)) == "safetensors"
    with ztensor.open(str(st)) as src:
        assert not src["w"].caps.verify, "safetensors carries no digests"

    out = tmp_path / "model.zt"
    ztensor.convert(str(st), str(out))
    with ztensor.open(str(out)) as src:
        assert src["w"].caps.verify
        assert src["w"].tobytes() == payload


# ---- the numpy shim --------------------------------------------------------


@requires_numpy
def test_numpy_shim_round_trip(tmp_path):
    import ztensor.numpy as ztnp

    path = tmp_path / "np.zt"
    tensors = {
        "a": numpy.arange(6, dtype=numpy.float32).reshape(2, 3),
        "b": numpy.ones(4, dtype=numpy.uint8),
    }
    ztnp.save_file(tensors, str(path))
    back = ztnp.load_file(str(path))
    assert sorted(back) == ["a", "b"]
    numpy.testing.assert_array_equal(back["a"], tensors["a"])
    numpy.testing.assert_array_equal(back["b"], tensors["b"])


@requires_numpy
def test_numpy_shim_keeps_the_shape_of_types_numpy_cannot_name(tmp_path):
    """The old shim returned a flat uint8 array for bf16, losing the shape."""
    path = tmp_path / "bf16.zt"
    with ztensor.Writer(str(path)) as w:
        w.add("w", bytes(2 * 6), shape=[2, 3], dtype="bf16")

    import ztensor.numpy as ztnp

    arr = ztnp.load_file(str(path))["w"]
    assert arr.shape in [(2, 3), (2, 3, 2)], f"shape was lost: {arr.shape}"
