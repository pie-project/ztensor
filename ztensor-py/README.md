# ztensor

Read any tensor format, write canonical `.zt`.

```console
pip install ztensor
```

```python
import ztensor, torch

with ztensor.open("model.safetensors") as src:   # or open([shard1, shard2])
    for t in src.values():                       # src itself is a mapping,
        print(t.name, t.shape, t.dtype)          # so iterating it gives names

    t = src["model.layers.0.mlp.w"]
    w = torch.from_dlpack(t)      # zero-copy
    t.location                    # (path, offset, nbytes) for your own I/O
    t["scales"]                   # parts are tensors too
```

A tensor with one part answers about that part. A tensor with several, a
quantized one or a CSR one, does not pick for you: index the part you mean.

Reads `.zt`, `.safetensors`, `.gguf`, `.npz`, PyTorch `.pt`, HDF5 and ONNX,
detected by magic. Writes exactly one format: canonical `.zt`, where every
tensor starts on a 64 KiB page and carries an XXH3 digest.

## No framework modules

Tensors export **DLPack** and the **buffer protocol**, so numpy, torch and jax
read them zero-copy without this package knowing any of them exist:

```python
numpy.from_dlpack(t)
torch.from_dlpack(t)
memoryview(t)
```

DLPack also carries `bfloat16`, which the numpy dtype table cannot, so the
dtype most checkpoints are actually in survives the trip.

A zero-copy export keeps the mapping alive on its own: an array stays valid
after the source it came from is closed.

## It tells you what it can do

```python
t.caps.map       # a zero-copy export will work
t.caps.locate    # .location gives the exact byte range
t.caps.evict     # its pages are its own
t.caps.verify    # there is a digest to check
```

Each is the precondition of the operation it names, so the report cannot
disagree with the behaviour. When only metadata is wanted, `ztensor.index()`
opens without mapping: names, shapes and addresses, for the cost of a header
read.

## Converting

```python
ztensor.convert("model.safetensors", "model.zt")
ztensor.verify("model.zt", deep=True)      # (digest_verified, without_digests)
```

Conversion is the upgrade path: the foreign file has no digests and arbitrary
alignment; the result has both.

## The numpy shim

`ztensor.numpy.load_file` / `save_file` are a safetensors-shaped convenience
layer, kept for code written against that API. New code should use the tensor
handles above, which can say things a dict of arrays cannot.

Linux and macOS. MIT licensed. The format, the Rust crates and the
specification live at
[github.com/pie-project/ztensor](https://github.com/pie-project/ztensor).
