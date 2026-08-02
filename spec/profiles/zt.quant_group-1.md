# Layout profile `zt.quant_group/1`

**Status:** Registry profile · **Core spec:** zTensor v2 §5.2

Group-wise affine quantization of a rank-2 weight matrix — the family
GPTQ and AWQ produce: `bits`-wide integers packed into a wider word, with
one scale and one zero-point per group of `group_size` elements along the
reduction axis.

## Parts

| Part | Required | Contents |
| --- | --- | --- |
| `packed_weight` | Yes | Packed quantized weights, `dtype` `i32` or `u32`. |
| `scales` | Yes | One scale per group, any float `dtype`. |
| `zeros` | No | One zero-point per group. Absent means symmetric quantization (zero-point 0). |

## Attributes

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `bits` | uint | Yes | Width of one quantized value: 2, 3, 4, or 8. |
| `group_size` | uint | Yes | Elements sharing one scale/zero-point, along `axis`. MUST divide `shape[axis]`. |
| `axis` | uint | No | Reduction axis the groups run along. Defaults to 0 (rows), matching GPTQ's `[in_features, out_features]` convention. |
| `packing` | text | Yes | Packing order within a word: `"lsb_first"` or `"msb_first"`. |
| `zero_point_packed` | bool | No | When true, `zeros` is itself packed like `packed_weight` (AWQ style) rather than one element per group. Defaults to false. |

`bits` MUST divide 32: 2, 4, and 8 pack evenly into a 32-bit word (16, 8,
and 4 values). `bits == 3` is permitted only with `group_size` a multiple
of 32, in which case 32 values occupy exactly three words.

## Derived quantities

```text
rows, cols   = shape          (rank MUST be 2)
groups       = shape[axis] / group_size
per_word     = 32 / bits      (bits ∈ {2, 4, 8})
```

## Sizes

With `axis == 0` (the default):

- `decoded_size(packed_weight)` MUST equal
  `(rows / per_word) × cols × 4` bytes.
- `decoded_size(scales)` MUST equal `groups × cols × width(scales.dtype)`.
- `decoded_size(zeros)`, when present and unpacked, MUST equal
  `groups × cols × width(zeros.dtype)`; when `zero_point_packed` is true it
  MUST equal `(groups / per_word) × cols × 4`.

With `axis == 1` the same equations hold with `rows` and `cols` swapped.

## Reconstruction

For an element at `(r, c)` in group `g = r / group_size`:

```text
q      = <the bits-wide field of packed_weight at (r, c), per `packing`>
zero   = zeros.is_some() ? dequant_zero(g, c) : 0
value  = (q − zero) × scales[g, c]
```

`"lsb_first"` places the value for row `r` at bit offset
`(r % per_word) × bits` within its word; `"msb_first"` counts from the top
of the word instead.

## Validation

Metadata rules (at open):

- Rank exactly 2; parts are exactly `packed_weight`, `scales`, and
  optionally `zeros` — no others.
- `packed_weight.dtype` is `i32` or `u32` with no logical type.
- `scales.dtype` is `f16`, `bf16`, `f32`, or `f64`.
- `bits` ∈ {2, 3, 4, 8}; `group_size` ≥ 1 and divides `shape[axis]`;
  `packing` is one of the two registered strings; `axis` < 2.
- All applicable size equations above.

Data rules: none — every bit pattern of a packed word is a valid
quantized value.

## Relationship to `gguf.*` profiles

GGUF's block-quantized types (`q4_k`, `q6_k`, ...) are **not** expressible
here: they interleave scales, mins, and weights inside a single block
struct with type-specific layouts. They project to their own
`gguf.<type>/1` profiles, which keep the block bytes opaque. This profile
is for schemes that already store weights, scales, and zero-points as
separate arrays.
