"""zTensor: read any tensor format, write canonical .zt.

    import ztensor

    with ztensor.open("model.safetensors") as src:   # or open([shard1, shard2])
        for name in src:                             # a mapping, so it yields names
            print(name, src[name].shape)
        for t in src.values():                       # the tensors themselves
            print(t.name, t.shape, t.dtype, t.caps.map)

        t = src["model.layers.0.mlp.w"]
        t.location                    # (path, offset, nbytes) for your own I/O
        memoryview(t)                 # zero-copy bytes
        # np.from_dlpack(t) / torch.from_dlpack(t): zero-copy, and DLPack can
        # say bfloat16, which the numpy dtype table cannot.
        t["scales"]                   # parts are tensors too

A tensor with one part answers about that part. A tensor with several does
not guess: index the one you mean.

    ztensor.convert("model.safetensors", "model.zt")   # canonical, verifiable
    ztensor.verify("model.zt", deep=True)              # (checked, undigested)

`ztensor.numpy` is a safetensors-shaped convenience layer for migrating
existing code; the API above is the one to write new code against.
"""

from collections.abc import Mapping

from ._ztensor import (  # noqa: F401
    Caps,
    Location,
    Sink,
    Source,
    Tensor,
    Verification,
    Writer,
    convert,
    detect,
    index,
    open,
    page_size,
    verify,
)

# `Source` has the whole read-only mapping surface: keys, values, items, get,
# __len__, __iter__, __contains__ and __getitem__, all keyed by tensor name.
# Registering says so, so `isinstance(src, Mapping)` answers correctly.
Mapping.register(Source)

__all__ = [
    "Caps",
    "Location",
    "Sink",
    "Source",
    "Tensor",
    "Verification",
    "Writer",
    "convert",
    "detect",
    "index",
    "open",
    "page_size",
    "verify",
]
