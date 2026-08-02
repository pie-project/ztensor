# Layout profiles `gguf.<type>/1`

**Status:** Registry profiles (projection-only) · **Core spec:** zTensor v2 §5.2

GGUF stores quantized tensors as arrays of type-specific **block structs**
— scales, mins, and packed weights interleaved inside one fixed-size
record, with a different internal layout for every type. There is no
faithful way to split such a block into separate `scales`/`weights` parts
without re-encoding it, so these profiles keep a block array exactly as it
appears on disk and describe it, rather than restructuring it.

This is the honest projection: a converted file is byte-identical to the
GGUF payload, and any consumer that already knows ggml's block layout can
use it directly. Consumers that don't must refuse (core §5.2) — no reader
is expected to infer the block internals from this profile.

## Identifier

`gguf.<type>/1`, where `<type>` is the lowercase ggml type name:
`q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1`, `q2_k`, `q3_k`, `q4_k`,
`q5_k`, `q6_k`, `q8_k`, `iq1_s`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`,
`iq3_xxs`, `iq3_s`, `iq4_nl`, `iq4_xs`, `mxfp4`.

Non-quantized ggml types (`f32`, `f16`, `bf16`, `i8`…`i64`, `f64`) do not
use these profiles: they project to the core `dense` layout.

## Parts

| Part | Required | Contents |
| --- | --- | --- |
| `data` | Yes | The block array, `dtype` `u8`, no logical type. Bytes are exactly as stored by GGUF. |

## Attributes

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `elems_per_block` | uint | Yes | Logical elements one block encodes (32 or 256 for current types). |
| `block_bytes` | uint | Yes | On-disk size of one block struct, i.e. `sizeof(block_<type>)` in ggml. |

Both are recorded so a consumer can compute the block count and validate
sizes without a built-in ggml type table — the table is exactly the thing
that drifts between ggml versions.

## Derived quantities

```text
elements = product(shape)      (the LOGICAL shape, not a byte count)
blocks   = elements / elems_per_block
```

The object keeps its **logical** shape: a `[4096, 4096]` Q8_0 tensor has
`shape == [4096, 4096]`, not `[17825792]`. Shape is what the model means;
the byte count is an encoding detail already implied by the attributes.

## Sizes

`decoded_size(data)` MUST equal `blocks × block_bytes`.

## Validation

Metadata rules (at open):

- Exactly one part, named `data`, `dtype` `u8`, no logical type.
- `elems_per_block` ≥ 1 and divides `product(shape)`.
- `block_bytes` ≥ 1.
- The size equation above.

Additionally, GGUF requires the **fastest-varying dimension** to be
divisible by `elems_per_block` (a block never spans two rows); a projection
from a GGUF file MUST enforce that, since a file violating it has no
well-defined block grid.

Data rules: none — the block bytes are opaque to this profile.

## Note on lossless round-tripping

Because the block array is copied verbatim and the logical shape is
preserved, converting GGUF → `.zt` → GGUF reproduces the original tensor
payload byte for byte. Metadata is preserved separately as file
attributes; the ggml type id is recoverable from the profile identifier.
