# ztensor

The `.zt` container format: reading, writing, and the whole of its validation.

```rust
use ztensor::{DType, Source, Writer};

let mut w = Writer::create("model.zt")?;
w.add("weights", [2u64, 2], DType::F32, &[0u8; 16])?;
w.finish()?;

let src = Source::open("model.zt")?;
let t = src.tensor("weights")?;
let bytes = t.map()?;      // borrowed, or an error — never a hidden copy
# Ok::<(), ztensor::Error>(())
```

Every tensor in a canonical file begins on a 64 KiB boundary and carries an
XXH3 digest, so one weight can be mapped, verified, and evicted without
touching its neighbours. Two canonical writes of the same tensors produce
byte-identical files.

Three ways to get at a part's bytes, one per intent:

| | |
| --- | --- |
| `bytes()` | the best the file can do, saying whether it borrowed or copied |
| `map()` | a borrow, or an error |
| `locate()` | the exact byte range, for a caller doing its own I/O |

`caps()` reports which will work, per part, and each of its fields is computed
by the very precondition the matching method checks.

- **Foreign formats** — safetensors, GGUF, `.npz`, `.pt`, HDF5, ONNX — are read
  through [`ztensor-compat`](https://crates.io/crates/ztensor-compat), which
  projects each into this same object model.
- **Command line** — [`ztensor-cli`](https://crates.io/crates/ztensor-cli)
  installs `zt`, for inspecting, verifying, converting and diffing.
- **Specification** — the normative rules live in
  [`spec/ztensor-v2-spec.md`](https://github.com/pie-project/ztensor/blob/main/spec/ztensor-v2-spec.md).

Features: `zstd` enables the `zt.zstd-seekable/1` encoding profile.

MIT licensed.
