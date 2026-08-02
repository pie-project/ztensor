"""zTensor: read any tensor format, write canonical .zt.

    import ztensor

    with ztensor.open("model.safetensors") as src:   # or open([shard1, shard2])
        for t in src:
            print(t.name, t.shape, t.dtype, t.caps.map)

        t = src["model.layers.0.mlp.w"]
        t.location                    # (path, offset, nbytes) for your own I/O
        memoryview(t)                 # zero-copy bytes
        # np.from_dlpack(t) / torch.from_dlpack(t) — zero-copy, and DLPack can
        # say bfloat16, which the numpy dtype table cannot.
        t["scales"]                   # parts are tensors too

    ztensor.convert("model.safetensors", "model.zt")   # canonical, verifiable
    ztensor.verify("model.zt", deep=True)

`ztensor.numpy` is a safetensors-shaped convenience layer for migrating
existing code; the API above is the one to write new code against.
"""

from ._ztensor import (  # noqa: F401
    Caps,
    Location,
    Source,
    Tensor,
    Writer,
    convert,
    detect,
    index,
    open,
    page_size,
    verify,
)

__all__ = [
    "Caps",
    "Location",
    "Source",
    "Tensor",
    "Writer",
    "convert",
    "detect",
    "index",
    "open",
    "page_size",
    "verify",
]
