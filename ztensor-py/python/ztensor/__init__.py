"""zTensor: read any tensor format, write canonical .zt.

    import ztensor
    src = ztensor.open("model.safetensors")   # any supported format
    src.keys()                                # tensor names
    src.read("layer.weight")                  # decoded little-endian bytes
    src.caps("layer.weight")["tier"]          # capability ladder tier

    ztensor.convert("model.safetensors", "model.zt")   # canonical, verifiable
    ztensor.verify("model.zt", deep=True)
"""

from ._ztensor import Source, Writer, convert, detect, open, verify  # noqa: F401

__all__ = ["Source", "Writer", "convert", "detect", "open", "verify"]
