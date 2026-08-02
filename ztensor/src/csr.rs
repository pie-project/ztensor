//! Assembling a `zt.sparse_csr/1` object.
//!
//! Deliberately outside the reader: a layout profile is registry vocabulary,
//! so the one this implementation happens to ship gets a module, not a method
//! on [`Tensor`]. A profile added downstream can live exactly here, in its own
//! crate, with the same shape.

use crate::error::{Error, Result, Rule};
use crate::schema::DType;
use crate::source::Tensor;

/// An assembled CSR object with its data-level rules checked.
#[derive(Debug, Clone)]
pub struct Csr {
    pub rows: u64,
    pub cols: u64,
    /// Decoded value bytes: `nnz` elements of `dtype`/`logical`.
    pub values: Vec<u8>,
    pub dtype: DType,
    pub logical: Option<String>,
    /// Column index per value, widened to u64.
    pub indices: Vec<u64>,
    /// Row pointers, `rows + 1` entries.
    pub indptr: Vec<u64>,
}

/// Reads and assembles a `zt.sparse_csr/1` tensor, enforcing the profile's
/// data-level MUSTs: `indptr[0] == 0`, non-decreasing, `indptr[rows] == nnz`,
/// per-row strictly increasing indices, and every index `< cols`.
pub fn read(tensor: &Tensor<'_>) -> Result<Csr> {
    if tensor.layout() != "zt.sparse_csr/1" {
        return Err(Error::Unsupported(format!(
            "{:?} has layout {:?}, not zt.sparse_csr/1",
            tensor.name(),
            tensor.layout()
        )));
    }
    let [rows, cols] = tensor.shape()[..] else {
        return Err(Error::reject(
            Rule::LayoutRule,
            format!("{:?}: sparse_csr requires rank-2 shape", tensor.name()),
        ));
    };

    let idx_part = tensor.part("indices")?;
    let idx_dtype = idx_part.dtype();
    let values_part = tensor.part("values")?;
    let (dtype, logical) = (values_part.dtype(), values_part.logical().map(str::to_string));

    let indices = widen(&idx_part.bytes()?, idx_dtype);
    let indptr = widen(&tensor.part("indptr")?.bytes()?, idx_dtype);
    let values = values_part.bytes()?.into_owned();
    let nnz = indices.len() as u64;

    let name = tensor.name();
    let bad = |detail: String| Err(Error::reject(Rule::LayoutData, detail));
    if indptr.first() != Some(&0) {
        return bad(format!("{name:?}: indptr must start at 0"));
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) {
        return bad(format!("{name:?}: indptr must be non-decreasing"));
    }
    if indptr.last() != Some(&nnz) {
        return bad(format!("{name:?}: indptr must end at nnz ({nnz})"));
    }
    for r in 0..rows as usize {
        let row = &indices[indptr[r] as usize..indptr[r + 1] as usize];
        if row.windows(2).any(|w| w[0] >= w[1]) {
            return bad(format!("{name:?}: row {r} indices not strictly increasing"));
        }
        if row.last().is_some_and(|&c| c >= cols) {
            return bad(format!("{name:?}: row {r} has an index >= cols ({cols})"));
        }
    }

    Ok(Csr {
        rows,
        cols,
        values,
        dtype,
        logical,
        indices,
        indptr,
    })
}

fn widen(bytes: &[u8], dtype: DType) -> Vec<u64> {
    match dtype {
        DType::U32 => bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as u64)
            .collect(),
        _ => bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    }
}
