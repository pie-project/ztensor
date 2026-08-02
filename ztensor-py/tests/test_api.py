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
    """The consumer holds the mapping through the capsule, not through us."""
    with ztensor.open(str(simple)) as src:
        arr = numpy.from_dlpack(src["a.weight"])
    del src
    assert arr[0, 0] == 1.0


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
