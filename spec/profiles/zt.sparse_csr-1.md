# Layout profile `zt.sparse_csr/1`

**Status:** Registry profile (parametric) · **Core spec:** zTensor v2 §5.2

Compressed Sparse Row matrices.

## Applicability

Rank-2 objects; `shape = [R, C]`.

## Parts

| Part | Required | Contents |
| --- | --- | --- |
| `values` | Yes | The `nnz` non-zero elements, any registered `dtype`/`type`. |
| `indices` | Yes | Column index per value, plain `u32` or `u64` (no logical type). |
| `indptr` | Yes | Row pointers, same `dtype` as `indices`. |

## Derived quantities

`nnz` is derived from `indices`, whose size function is exact; deriving it
from `values` would be ill-defined for a packed sub-byte value type.

```text
nnz = decoded_size(indices) / width(indices.dtype)
```

## Sizes

- `decoded_size(indptr)` MUST equal `(R + 1) × width(indices.dtype)`.
- `decoded_size(values)` MUST equal the values type's size function at `nnz`.

## Validation

Metadata rules (at open):

- Rank exactly 2; parts are exactly `values`, `indices`, `indptr`.
- `indices.dtype` is `u32` or `u64`, with no logical type.
- `indptr.dtype` equals `indices.dtype`, with no logical type.
- Both size equations above.

Data rules (when the object is assembled):

- `indptr[0] == 0`; `indptr` non-decreasing; `indptr[R] == nnz`.
- Within each row, `indices` strictly increasing.
- Every index `< C`.

## Attributes

None.
