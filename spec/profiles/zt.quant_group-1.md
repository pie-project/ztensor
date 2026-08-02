# Layout profile `zt.quant_group/1`

**Status:** Registry profile (**parametric**) · **Core spec:** zTensor v2 §5.2

Affine group quantization: low-precision integer codes packed into a wider
word, with a scale — and optionally a zero point — shared by each group of
codes along one axis.

This profile is a **space, not a scheme**. AWQ-int4, GPTQ-int4, MLX's
affine-u4, symmetric int8 and the bias-8 int4 packing are five points in
it, distinguished by their attributes and not by a name. A file that only
said "AWQ" would be telling a reader to go look something up; a file that
says its packing order, scale form and zero-point form has told the reader
everything the decoder needs.

## Parts

| Part | Required | Contents |
| --- | --- | --- |
| `data` | Yes | The packed codes. |
| `scales` | Yes | One scale per group. |
| `zeros` | Conditional | One zero point per group. Required iff `zero_point.form` is `"tensor"`; MUST be absent otherwise. |

## Attributes

Every one of these is **required** unless marked otherwise. A parametric
profile that leaves a decoder parameter unstated invites it to be inferred
from something incidental, which is the failure this profile exists to
avoid.

| Key | Type | Meaning |
| --- | --- | --- |
| `bits` | uint | Width of one code: 2, 3, 4, 5, 6 or 8. |
| `group_size` | uint | Codes sharing one scale, along `axis`. `0` means per-channel: one scale per row of `axis`. |
| `axis` | uint | The axis groups run along. |
| `packing` | map | How codes sit in a storage word — see below. |
| `scale_form` | text | How a consumer must read `scales` — see below. |
| `zero_point` | map | Whether and how a zero point applies — see below. |

### `packing`

| Key | Type | Meaning |
| --- | --- | --- |
| `word` | text | The storage word: `"u8"`, `"u16"`, `"u32"` or `"u64"`. MUST equal `data.dtype` (or its unsigned counterpart when `data.dtype` is signed). |
| `order` | text | `"lsb_first"` — the code for the lowest index occupies the low bits — or `"msb_first"`. |
| `per_word` | uint | Codes per word. MUST equal `8 × width(word) / bits`. |

`bits` MUST divide `8 × width(word)`; a packing that leaves unused bits in
a word is not expressible here, and should be its own profile rather than
an implied convention.

### `scale_form`

What a consumer reads the scale as. Not derivable from `scales.dtype`,
which says how the bytes are *stored*, not how they are *meant*.

| Value | Meaning |
| --- | --- |
| `"f32_factors"` | Multipliers, widened to f32 by the consumer. `scales.dtype` is any float type. |
| `"f16_factors"` | Multipliers consumed at f16/bf16 width. |
| `"e8m0_exponent"` | Exponent bytes: the stored byte `b` denotes `2^(b − 127)`. `scales` MUST be `u8` with logical type `f8_e8m0`. |

### `zero_point`

| Key | Type | Meaning |
| --- | --- | --- |
| `form` | text | `"none"`, `"implied"`, or `"tensor"`. |
| `value` | int | Required iff `form` is `"implied"`: the constant every code is offset by. |
| `packing` | text | Required iff `form` is `"tensor"`: `"same_as_data"` (the `zeros` part is packed exactly like `data`) or `"plain"` (one element per group, unpacked). |

## Derived quantities

```text
along     = shape[axis]
groups    = group_size == 0 ? along : along / group_size
lanes     = product(shape) / along          # rows across the other axes
codes     = product(shape)
```

`group_size`, when nonzero, MUST divide `shape[axis]`: a group never spans
two rows of the quantized axis.

## Sizes

- `decoded_size(data)` MUST equal `codes / per_word × width(word)`.
- `decoded_size(scales)` MUST equal `groups × lanes × width(scales.dtype)`.
- When `zero_point.form` is `"tensor"`:
  - `packing = "plain"`: `decoded_size(zeros)` MUST equal
    `groups × lanes × width(zeros.dtype)`.
  - `packing = "same_as_data"`: `decoded_size(zeros)` MUST equal
    `groups × lanes / per_word × width(word)`.

## Reconstruction

For a code `q` at index `i`, in group `g`:

```text
zero  = form == "tensor"  ? read_zero(g)
      : form == "implied" ? value
      :                     0
value = (q − zero) × scale(g)
```

where `scale(g)` is `scales[g]` read per `scale_form`, and `read_zero(g)`
unpacks `zeros` per its `packing`.

## Validation

Metadata rules (at open):

- Parts are exactly `data`, `scales`, and `zeros` iff the zero-point form
  requires it.
- `data.dtype` is an integer type matching `packing.word`.
- `bits` ∈ {2, 3, 4, 5, 6, 8} and divides `8 × width(word)`;
  `per_word` consistent with both.
- `scales` satisfies its `scale_form`'s dtype requirement.
- `axis < rank`; `group_size` divides `shape[axis]` when nonzero.
- Every size equation above.

Data rules: none. Every bit pattern of a packed word is a valid code.

## Worked points in the space

These are not variants of the profile; they are attribute sets.

**AWQ int4** — packed zero points, f16 scales, group 128:

```text
"attributes": {
  "bits": 4, "group_size": 128, "axis": 0,
  "packing":    { "word": "u32", "order": "lsb_first", "per_word": 8 },
  "scale_form": "f16_factors",
  "zero_point": { "form": "tensor", "packing": "same_as_data" }
}
```

**GPTQ int4** — the same, with the other packing order:

```text
  "packing":    { "word": "u32", "order": "msb_first", "per_word": 8 }
```

**MLX affine-u4** — group 64, unpacked zero points (biases):

```text
"attributes": {
  "bits": 4, "group_size": 64, "axis": 1,
  "packing":    { "word": "u32", "order": "lsb_first", "per_word": 8 },
  "scale_form": "f16_factors",
  "zero_point": { "form": "tensor", "packing": "plain" }
}
```

**Bias-8 int4** — the zero point *is* the 8, so there is no `zeros` part:

```text
"attributes": {
  "bits": 4, "group_size": 32, "axis": 0,
  "packing":    { "word": "u32", "order": "lsb_first", "per_word": 8 },
  "scale_form": "f32_factors",
  "zero_point": { "form": "implied", "value": 8 }
}
```

**Symmetric int8, per channel** — no grouping, no zero point:

```text
"attributes": {
  "bits": 8, "group_size": 0, "axis": 0,
  "packing":    { "word": "u8", "order": "lsb_first", "per_word": 1 },
  "scale_form": "f32_factors",
  "zero_point": { "form": "none" }
}
```

## Relationship to other profiles

`zt.mx/1` is the OCP Microscaling family, whose scale is an exponent shared
by a power-of-two block; it stays separate because its element types are
sub-byte float formats rather than integer codes, and its scale form is
fixed by the OCP specification.

The `gguf.<type>/1` family is **not** expressible here and must not be
forced into it: a ggml block interleaves scales with data inside one
struct, and `q4_k` nests a second level of six-bit scales inside a
super-block. Those are opaque profiles (core §5.2).
