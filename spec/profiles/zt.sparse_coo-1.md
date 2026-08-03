# Layout profile `zt.sparse_coo/1`

**Status:** Registry profile · **Core spec:** zTensor v2 §5.2

Coordinate-list sparse tensors of any rank.

## Parts

| Part | Required | Contents |
| --- | --- | --- |
| `values` | Yes | The `nnz` non-zero elements, any registered `dtype`/`type`. |
| `coords` | Yes | `ndim × nnz` indices, plain `u32` or `u64` (no logical type). |

## Derived quantities

`nnz` is derived from `coords`, not from `values`: `coords` has an exact
size function, while a packed sub-byte value type would make the inverse
ill-defined.

```text
ndim  = len(shape)
nnz   = decoded_size(coords) / width(coords.dtype) / ndim
```

## Coordinate order

`coords` is stored **structure-of-arrays**: all `ndim` indices of dimension
0 first, then all of dimension 1, and so on. Element `i` of the tensor has
coordinate `coords[d * nnz + i]` in dimension `d`.

This layout lets a reader slice one dimension's indices contiguously, which
is what index-scan kernels want; the alternative (interleaved tuples)
requires a strided gather for the same access.

## Validation

Metadata rules (checked at open, on every object with this layout):

- `shape` MUST have rank ≥ 1.
- Exactly two parts, named `values` and `coords`.
- `coords.dtype` MUST be `u32` or `u64`, with no logical type.
- `decoded_size(coords)` MUST be divisible by `width(coords.dtype) × ndim`.
- `decoded_size(values)` MUST equal the values type's size function at
  `nnz`.

Data rules (checked when the object is assembled):

- Every coordinate in dimension `d` MUST be `< shape[d]`.
- Coordinates MUST be unique: no two elements may name the same cell.
  (Ordering is otherwise unconstrained, since COO is a set rather than a
  sequence.)

## Attributes

None. A profile that needs sorted coordinates should be a distinct
profile id, not an attribute on this one: a reader must not have to
inspect attributes to know what invariants hold.
