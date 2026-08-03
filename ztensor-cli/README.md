# ztensor-cli

`zt`: inspect, verify, convert and diff tensor files.

```console
$ cargo install ztensor-cli

$ zt ls model.gguf                    # inspect anything
$ zt convert model.gguf model.zt      # canonical, verifiable output
$ zt verify model.zt --deep           # structure + digests + shards
$ zt diff a.safetensors b.zt          # compare across formats
```

Reads `.zt`, `.safetensors`, `.gguf`, `.npz`, `.pt`, `.h5` and `.onnx`; writes
exactly one format, canonical `.zt`.

Two details worth knowing:

- `ls` on a `.zt` root reads only that file, so it still answers when the shard
  files are not beside it. Listing is a question about this file.
- `diff` compares by content, not by bytes, so the same tensors in two
  different formats compare equal.

MIT licensed.
