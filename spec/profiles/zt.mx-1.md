# Layout profile `zt.mx/1`

**Status:** Registry profile · **Core spec:** zTensor v2 §5.2

OCP Microscaling (MX) block formats: a vector of low-precision elements
sharing one power-of-two scale per fixed-size block. Covers MXFP4, MXFP6,
MXFP8, and MXINT8 with one profile — they differ only in the element type.

## Parts

| Part | Required | Contents |
| --- | --- | --- |
| `data` | Yes | The packed elements, in the object's logical order. |
| `scales` | Yes | One `f8_e8m0` scale per block, `dtype` `u8`. |

## Attributes

| Key | Type | Required | Meaning |
| --- | --- | --- | --- |
| `block_size` | uint | Yes | Elements per block. MUST be 32 for OCP MX formats; other values are permitted but are not OCP-conformant. |
| `axis` | uint | No | Dimension the blocks run along. Defaults to the last dimension (`rank − 1`). |

## Derived quantities

```text
elements = product(shape)
blocks   = elements / block_size
```

`shape[axis]` MUST be divisible by `block_size`: a block never spans two
rows of the blocked axis, so a partial trailing block cannot occur.

## Sizes

- `decoded_size(data)` MUST equal the element type's size function at
  `elements`. For `f4_e2m1` that is `⌈elements / 2⌉`; because
  `block_size` is even, the packing has no odd trailing nibble.
- `decoded_size(scales)` MUST equal `blocks`.

## Element types

`data.type` MUST be one of the registered MX element types:

| `type` | `dtype` | Format |
| --- | --- | --- |
| `f4_e2m1` | `u8` | MXFP4 |
| `f8_e4m3fn` | `u8` | MXFP8 (E4M3) |
| `f8_e5m2` | `u8` | MXFP8 (E5M2) |
| `i8` (no logical type) | `i8` | MXINT8 |

A reader that does not recognize `data.type` MUST refuse to decode the
object (core §4.2); it MAY still expose it structurally.

## Reconstruction

For logical element `i` in block `b = i / block_size`:

```text
value(i) = decode(scales[b]) × decode(data[i])
```

`scales[b]` is an `f8_e8m0` biased exponent: the byte `e` denotes `2^(e −
127)`, with `e == 0xFF` reserved for NaN. Element decoding follows the OCP
Microscaling Formats specification for the declared element type.

## Validation

Metadata rules (at open):

- Exactly two parts, named `data` and `scales`.
- `scales.dtype == u8` and `scales.type == "f8_e8m0"`.
- `block_size` present, ≥ 2, and a divisor of `shape[axis]`.
- `axis`, if present, `< rank`.
- Both size equations above.

Data rules: none. A scale byte of `0xFF` (NaN) is a legal encoding, and
this profile does not constrain element bit patterns beyond what the
element type's own rules require.

## Note on hardware

Blackwell-class GPUs consume `data` and `scales` as separate buffers in
exactly this form. Because each part is its own 64 KiB-aligned blob, a
canonical `.zt` file can be memory-mapped and uploaded to such a device
without dequantizing or repacking — which is the reason this profile keeps
the two parts separate rather than interleaving scales into the data.
