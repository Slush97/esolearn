// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sparse matrix types: CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column).
//!
//! Designed for NLP/recommender workloads with 50K+ features and >99% zeros.
//! Provides efficient row-oriented (CSR) and column-oriented (CSC) access,
//! plus conversion between formats.

use crate::error::{Result, ScryLearnError};
use std::ops;

// ---------------------------------------------------------------------------
// SparseRow / SparseCol views
// ---------------------------------------------------------------------------

/// View into a single row of a [`CsrMatrix`].
#[derive(Clone, Debug)]
pub struct SparseRow<'a> {
    indices: &'a [usize],
    data: &'a [f64],
}

impl<'a> SparseRow<'a> {
    /// Iterate over `(col_idx, value)` pairs in this row.
    pub fn iter(&self) -> impl Iterator<Item = (usize, f64)> + 'a {
        self.indices.iter().copied().zip(self.data.iter().copied())
    }

    /// Number of non-zero entries in this row.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Column indices of non-zero entries (sorted).
    pub fn indices(&self) -> &[usize] {
        self.indices
    }

    /// Values of non-zero entries (parallel to `indices()`).
    pub fn values(&self) -> &[f64] {
        self.data
    }

    /// Sparse dot product with a dense vector.
    pub fn dot(&self, other: &[f64]) -> f64 {
        self.indices
            .iter()
            .zip(self.data.iter())
            .map(|(&j, &v)| v * other[j])
            .sum()
    }
}

/// View into a single column of a [`CscMatrix`].
#[derive(Clone, Debug)]
pub struct SparseCol<'a> {
    indices: &'a [usize],
    data: &'a [f64],
}

impl<'a> SparseCol<'a> {
    /// Iterate over `(row_idx, value)` pairs in this column.
    pub fn iter(&self) -> impl Iterator<Item = (usize, f64)> + 'a {
        self.indices.iter().copied().zip(self.data.iter().copied())
    }

    /// Number of non-zero entries in this column.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Sparse dot product with a dense vector.
    pub fn dot(&self, other: &[f64]) -> f64 {
        self.indices
            .iter()
            .zip(self.data.iter())
            .map(|(&i, &v)| v * other[i])
            .sum()
    }
}

// ---------------------------------------------------------------------------
// CsrMatrix
// ---------------------------------------------------------------------------

/// Compressed Sparse Row matrix.
///
/// Efficient for row iteration (KNN predict, tree predict).
/// Standard CSR layout: `indptr[i]..indptr[i+1]` gives the range
/// into `indices` and `data` for row `i`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct CsrMatrix {
    /// Row pointers: length `n_rows + 1`.
    indptr: Vec<usize>,
    /// Column indices for each non-zero element.
    indices: Vec<usize>,
    /// Non-zero values.
    data: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
}

impl CsrMatrix {
    /// Build a CSR matrix from COO (triplet) format.
    ///
    /// Duplicate entries at the same `(row, col)` are summed.
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Self> {
        let nnz = rows.len();
        if cols.len() != nnz || vals.len() != nnz {
            return Err(ScryLearnError::InvalidParameter(format!(
                "triplet arrays must have equal length (rows={}, cols={}, vals={})",
                nnz,
                cols.len(),
                vals.len()
            )));
        }

        // Validate indices.
        for i in 0..nnz {
            if rows[i] >= n_rows || cols[i] >= n_cols {
                return Err(ScryLearnError::InvalidParameter(format!(
                    "triplet index ({}, {}) out of bounds for {}x{} matrix",
                    rows[i], cols[i], n_rows, n_cols
                )));
            }
        }

        // Count entries per row.
        let mut row_counts = vec![0usize; n_rows];
        for &r in rows {
            row_counts[r] += 1;
        }

        // Build indptr.
        let mut indptr = vec![0usize; n_rows + 1];
        for i in 0..n_rows {
            indptr[i + 1] = indptr[i] + row_counts[i];
        }

        // Scatter triplets into CSR arrays.
        let total = indptr[n_rows];
        let mut csr_indices = vec![0usize; total];
        let mut csr_data = vec![0.0f64; total];
        let mut offsets = indptr[..n_rows].to_vec();

        for k in 0..nnz {
            let r = rows[k];
            let pos = offsets[r];
            csr_indices[pos] = cols[k];
            csr_data[pos] = vals[k];
            offsets[r] += 1;
        }

        // Sort each row by column index and merge duplicates.
        let mut final_indices = Vec::with_capacity(total);
        let mut final_data = Vec::with_capacity(total);
        let mut new_indptr = vec![0usize; n_rows + 1];

        for i in 0..n_rows {
            let start = indptr[i];
            let end = indptr[i + 1];

            // Sort by column index.
            let mut pairs: Vec<(usize, f64)> = csr_indices[start..end]
                .iter()
                .copied()
                .zip(csr_data[start..end].iter().copied())
                .collect();
            pairs.sort_by_key(|&(c, _)| c);

            // Merge duplicates by summing (only within this row).
            let row_start = final_indices.len();
            for &(col, val) in &pairs {
                // The guard `final_indices.len() > row_start` ensures non-empty.
                if final_indices.len() > row_start
                    && final_indices[final_indices.len() - 1] == col
                {
                    let last = final_data.len() - 1;
                    final_data[last] += val;
                    continue;
                }
                final_indices.push(col);
                final_data.push(val);
            }
            new_indptr[i + 1] = final_indices.len();
        }

        Ok(Self {
            indptr: new_indptr,
            indices: final_indices,
            data: final_data,
            n_rows,
            n_cols,
        })
    }

    /// Convert a dense row-major matrix to CSR (zeros are skipped).
    pub fn from_dense(rows: &[Vec<f64>]) -> Self {
        let n_rows = rows.len();
        let n_cols = if n_rows > 0 { rows[0].len() } else { 0 };

        let mut indptr = vec![0usize; n_rows + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    indices.push(j);
                    data.push(val);
                }
            }
            indptr[i + 1] = indices.len();
        }

        Self {
            indptr,
            indices,
            data,
            n_rows,
            n_cols,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Number of stored non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Fraction of non-zero entries: `nnz / (n_rows * n_cols)`.
    ///
    /// Returns 0.0 for an empty (0×0) matrix.
    #[inline]
    pub fn density(&self) -> f64 {
        let total = self.n_rows * self.n_cols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
    }

    /// View of row `i` as sparse `(col, value)` pairs.
    pub fn row(&self, i: usize) -> SparseRow<'_> {
        let start = self.indptr[i];
        let end = self.indptr[i + 1];
        SparseRow {
            indices: &self.indices[start..end],
            data: &self.data[start..end],
        }
    }

    /// Retrieve a single element. Returns `0.0` if the entry is not stored.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        self.indices[start..end]
            .binary_search(&col)
            .map_or(0.0, |pos| self.data[start + pos])
    }

    /// Convert to CSC format in O(nnz).
    pub fn to_csc(&self) -> CscMatrix {
        let nnz = self.nnz();

        // Count entries per column.
        let mut col_counts = vec![0usize; self.n_cols];
        for &c in &self.indices {
            col_counts[c] += 1;
        }

        let mut indptr = vec![0usize; self.n_cols + 1];
        for j in 0..self.n_cols {
            indptr[j + 1] = indptr[j] + col_counts[j];
        }

        let mut csc_indices = vec![0usize; nnz];
        let mut csc_data = vec![0.0f64; nnz];
        let mut offsets = indptr[..self.n_cols].to_vec();

        for i in 0..self.n_rows {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            for k in start..end {
                let col = self.indices[k];
                let pos = offsets[col];
                csc_indices[pos] = i;
                csc_data[pos] = self.data[k];
                offsets[col] += 1;
            }
        }

        CscMatrix {
            indptr,
            indices: csc_indices,
            data: csc_data,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }

    /// Convert to dense row-major format.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.n_cols]; self.n_rows];
        for (i, row) in dense.iter_mut().enumerate() {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            for k in start..end {
                row[self.indices[k]] = self.data[k];
            }
        }
        dense
    }

    /// Sparse matrix-vector multiply: `y = A * x`.
    pub fn dot_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n_rows];
        for (yi, i) in y.iter_mut().zip(0..self.n_rows) {
            *yi = self.row(i).dot(x);
        }
        y
    }
}

// ---------------------------------------------------------------------------
// CscMatrix
// ---------------------------------------------------------------------------

/// Compressed Sparse Column matrix.
///
/// Efficient for column iteration (tree fit, linear algebra).
/// `indptr[j]..indptr[j+1]` gives the range for column `j`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct CscMatrix {
    /// Column pointers: length `n_cols + 1`.
    indptr: Vec<usize>,
    /// Row indices for each non-zero element.
    indices: Vec<usize>,
    /// Non-zero values.
    data: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
}

impl CscMatrix {
    /// Build a CSC matrix from COO (triplet) format.
    ///
    /// Duplicate entries at the same `(row, col)` are summed.
    pub fn from_triplets(
        rows: &[usize],
        cols: &[usize],
        vals: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<Self> {
        // Build as CSR then transpose — reuses all the validation/dedup logic.
        let csr = CsrMatrix::from_triplets(rows, cols, vals, n_rows, n_cols)?;
        Ok(csr.to_csc())
    }

    /// Convert a column-major dense matrix to CSC (zeros are skipped).
    ///
    /// `cols[j][i]` = value at row `i`, column `j`.
    pub fn from_dense(cols: &[Vec<f64>]) -> Self {
        let n_cols = cols.len();
        let n_rows = if n_cols > 0 { cols[0].len() } else { 0 };

        let mut indptr = vec![0usize; n_cols + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for (j, col) in cols.iter().enumerate() {
            for (i, &val) in col.iter().enumerate() {
                if val != 0.0 {
                    indices.push(i);
                    data.push(val);
                }
            }
            indptr[j + 1] = indices.len();
        }

        Self {
            indptr,
            indices,
            data,
            n_rows,
            n_cols,
        }
    }

    /// Number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Number of stored non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Fraction of non-zero entries.
    #[inline]
    pub fn density(&self) -> f64 {
        let total = self.n_rows * self.n_cols;
        if total == 0 {
            return 0.0;
        }
        self.nnz() as f64 / total as f64
    }

    /// View of column `j` as sparse `(row, value)` pairs.
    pub fn col(&self, j: usize) -> SparseCol<'_> {
        let start = self.indptr[j];
        let end = self.indptr[j + 1];
        SparseCol {
            indices: &self.indices[start..end],
            data: &self.data[start..end],
        }
    }

    /// Retrieve a single element. Returns `0.0` if not stored.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let start = self.indptr[col];
        let end = self.indptr[col + 1];
        self.indices[start..end]
            .binary_search(&row)
            .map_or(0.0, |pos| self.data[start + pos])
    }

    /// Convert to CSR format in O(nnz).
    pub fn to_csr(&self) -> CsrMatrix {
        let nnz = self.nnz();

        // Count entries per row.
        let mut row_counts = vec![0usize; self.n_rows];
        for &r in &self.indices {
            row_counts[r] += 1;
        }

        let mut indptr = vec![0usize; self.n_rows + 1];
        for i in 0..self.n_rows {
            indptr[i + 1] = indptr[i] + row_counts[i];
        }

        let mut csr_indices = vec![0usize; nnz];
        let mut csr_data = vec![0.0f64; nnz];
        let mut offsets = indptr[..self.n_rows].to_vec();

        for j in 0..self.n_cols {
            let start = self.indptr[j];
            let end = self.indptr[j + 1];
            for k in start..end {
                let row = self.indices[k];
                let pos = offsets[row];
                csr_indices[pos] = j;
                csr_data[pos] = self.data[k];
                offsets[row] += 1;
            }
        }

        CsrMatrix {
            indptr,
            indices: csr_indices,
            data: csr_data,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }

    /// Convert to dense row-major format.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        // Build via CSR for clippy-friendly iteration.
        self.to_csr().to_dense()
    }

    /// Sparse matrix-vector multiply: `y = A * x`.
    pub fn dot_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n_rows];
        for (j, &xj) in x.iter().enumerate() {
            let start = self.indptr[j];
            let end = self.indptr[j + 1];
            for k in start..end {
                y[self.indices[k]] += self.data[k] * xj;
            }
        }
        y
    }
}

// ---------------------------------------------------------------------------
// Arithmetic: CsrMatrix + CsrMatrix, CsrMatrix * f64
// ---------------------------------------------------------------------------

impl ops::Add for &CsrMatrix {
    type Output = CsrMatrix;

    /// Element-wise addition of two CSR matrices with the same shape.
    ///
    /// # Panics
    ///
    /// Panics if the matrices have different shapes.
    fn add(self, rhs: &CsrMatrix) -> CsrMatrix {
        assert_eq!(
            (self.n_rows, self.n_cols),
            (rhs.n_rows, rhs.n_cols),
            "CsrMatrix addition requires same shape"
        );

        let mut indptr = vec![0usize; self.n_rows + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for i in 0..self.n_rows {
            let a_start = self.indptr[i];
            let a_end = self.indptr[i + 1];
            let b_start = rhs.indptr[i];
            let b_end = rhs.indptr[i + 1];

            let mut a = a_start;
            let mut b = b_start;

            // Merge two sorted column-index streams.
            while a < a_end && b < b_end {
                match self.indices[a].cmp(&rhs.indices[b]) {
                    std::cmp::Ordering::Less => {
                        indices.push(self.indices[a]);
                        data.push(self.data[a]);
                        a += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        indices.push(rhs.indices[b]);
                        data.push(rhs.data[b]);
                        b += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        let sum = self.data[a] + rhs.data[b];
                        if sum != 0.0 {
                            indices.push(self.indices[a]);
                            data.push(sum);
                        }
                        a += 1;
                        b += 1;
                    }
                }
            }
            while a < a_end {
                indices.push(self.indices[a]);
                data.push(self.data[a]);
                a += 1;
            }
            while b < b_end {
                indices.push(rhs.indices[b]);
                data.push(rhs.data[b]);
                b += 1;
            }

            indptr[i + 1] = indices.len();
        }

        CsrMatrix {
            indptr,
            indices,
            data,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }
}

impl ops::Mul<f64> for &CsrMatrix {
    type Output = CsrMatrix;

    /// Scalar multiplication.
    fn mul(self, rhs: f64) -> CsrMatrix {
        CsrMatrix {
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self.data.iter().map(|&v| v * rhs).collect(),
            n_rows: self.n_rows,
            n_cols: self.n_cols,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_from_triplets_basic() {
        // 3x3 matrix:
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let csr = CsrMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();
        assert_eq!(csr.n_rows(), 3);
        assert_eq!(csr.n_cols(), 3);
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 2), 2.0);
        assert_eq!(csr.get(1, 1), 3.0);
        assert_eq!(csr.get(2, 0), 4.0);
        assert_eq!(csr.get(2, 2), 5.0);
        assert_eq!(csr.get(0, 1), 0.0);
        assert_eq!(csr.get(1, 0), 0.0);
    }

    #[test]
    fn test_duplicate_entries_summed() {
        let rows = vec![0, 0, 0];
        let cols = vec![1, 1, 1];
        let vals = vec![1.0, 2.0, 3.0];

        let csr = CsrMatrix::from_triplets(&rows, &cols, &vals, 2, 3).unwrap();
        assert_eq!(csr.nnz(), 1);
        assert_eq!(csr.get(0, 1), 6.0);
    }

    #[test]
    fn test_csr_csc_roundtrip() {
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let csr = CsrMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();
        let csc = csr.to_csc();
        let csr2 = csc.to_csr();

        assert_eq!(csr.to_dense(), csr2.to_dense());
    }

    #[test]
    fn test_dense_roundtrip() {
        let dense = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 3.0, 0.0],
            vec![4.0, 0.0, 5.0],
        ];

        let csr = CsrMatrix::from_dense(&dense);
        assert_eq!(csr.to_dense(), dense);
    }

    #[test]
    fn test_get_existing_and_missing() {
        let csr = CsrMatrix::from_dense(&[vec![0.0, 7.0], vec![8.0, 0.0]]);
        assert_eq!(csr.get(0, 1), 7.0);
        assert_eq!(csr.get(1, 0), 8.0);
        assert_eq!(csr.get(0, 0), 0.0);
        assert_eq!(csr.get(1, 1), 0.0);
    }

    #[test]
    fn test_dot_vec_csr() {
        // [1 2] * [3] = [1*3+2*4] = [11]
        // [0 3]   [4]   [0*3+3*4]   [12]
        let csr = CsrMatrix::from_dense(&[vec![1.0, 2.0], vec![0.0, 3.0]]);
        let result = csr.dot_vec(&[3.0, 4.0]);
        assert_eq!(result, vec![11.0, 12.0]);
    }

    #[test]
    fn test_dot_vec_csc() {
        let dense = vec![vec![1.0, 2.0], vec![0.0, 3.0]];
        let csr = CsrMatrix::from_dense(&dense);
        let csc = csr.to_csc();
        let result = csc.dot_vec(&[3.0, 4.0]);
        assert_eq!(result, vec![11.0, 12.0]);
    }

    #[test]
    fn test_sparse_row_iteration() {
        let csr = CsrMatrix::from_dense(&[vec![0.0, 5.0, 0.0, 7.0]]);
        let row = csr.row(0);
        let pairs: Vec<(usize, f64)> = row.iter().collect();
        assert_eq!(pairs, vec![(1, 5.0), (3, 7.0)]);
        assert_eq!(row.nnz(), 2);
    }

    #[test]
    fn test_sparse_col_iteration() {
        let csr = CsrMatrix::from_dense(&[vec![1.0, 0.0], vec![0.0, 0.0], vec![3.0, 0.0]]);
        let csc = csr.to_csc();
        let col = csc.col(0);
        let pairs: Vec<(usize, f64)> = col.iter().collect();
        assert_eq!(pairs, vec![(0, 1.0), (2, 3.0)]);
        assert_eq!(col.nnz(), 2);
    }

    #[test]
    fn test_empty_matrix() {
        // 0x0
        let csr = CsrMatrix::from_triplets(&[], &[], &[], 0, 0).unwrap();
        assert_eq!(csr.n_rows(), 0);
        assert_eq!(csr.n_cols(), 0);
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.density(), 0.0);

        // 5x5 with no entries
        let csr = CsrMatrix::from_triplets(&[], &[], &[], 5, 5).unwrap();
        assert_eq!(csr.n_rows(), 5);
        assert_eq!(csr.n_cols(), 5);
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.density(), 0.0);
        assert_eq!(csr.get(2, 3), 0.0);
    }

    #[test]
    fn test_density() {
        // 3x3 with 5 entries → 5/9
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 2, 1, 0, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let csr = CsrMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();
        assert!((csr.density() - 5.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_large_sparse() {
        // 1000x1000 with ~0.1% density.
        let n = 1000;
        let mut rng = fastrand::Rng::with_seed(42);
        let target_nnz = (n * n) / 1000; // 0.1%

        let mut rows = Vec::with_capacity(target_nnz);
        let mut cols = Vec::with_capacity(target_nnz);
        let mut vals = Vec::with_capacity(target_nnz);

        for _ in 0..target_nnz {
            rows.push(rng.usize(..n));
            cols.push(rng.usize(..n));
            vals.push(rng.f64() * 10.0);
        }

        let csr = CsrMatrix::from_triplets(&rows, &cols, &vals, n, n).unwrap();
        assert_eq!(csr.n_rows(), n);
        assert_eq!(csr.n_cols(), n);
        // nnz may be less than target_nnz due to duplicate merging.
        assert!(csr.nnz() <= target_nnz);
        assert!(csr.nnz() > 0);
        assert!(csr.density() < 0.002);

        // Spot-check round-trip.
        let csc = csr.to_csc();
        let csr2 = csc.to_csr();
        assert_eq!(csr.nnz(), csr2.nnz());
    }

    #[test]
    fn test_from_dense_skips_zeros() {
        let dense = vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
        ];
        let csr = CsrMatrix::from_dense(&dense);
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.get(0, 2), 1.0);
        assert_eq!(csr.get(2, 0), 2.0);
    }

    #[test]
    fn test_csr_add() {
        let a = CsrMatrix::from_dense(&[vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 0.0]]);
        let b = CsrMatrix::from_dense(&[vec![0.0, 4.0, 0.0], vec![5.0, 0.0, 6.0]]);
        let c = &a + &b;
        assert_eq!(
            c.to_dense(),
            vec![vec![1.0, 4.0, 2.0], vec![5.0, 3.0, 6.0],]
        );
    }

    #[test]
    fn test_csr_scalar_mul() {
        let a = CsrMatrix::from_dense(&[vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 0.0]]);
        let b = &a * 2.0;
        assert_eq!(
            b.to_dense(),
            vec![vec![2.0, 0.0, 4.0], vec![0.0, 6.0, 0.0],]
        );
    }

    #[test]
    fn test_csc_from_triplets() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let vals = vec![1.0, 2.0, 3.0];
        let csc = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();
        assert_eq!(csc.n_rows(), 3);
        assert_eq!(csc.n_cols(), 3);
        assert_eq!(csc.nnz(), 3);
        assert_eq!(csc.get(0, 0), 1.0);
        assert_eq!(csc.get(1, 1), 2.0);
        assert_eq!(csc.get(2, 2), 3.0);
        assert_eq!(csc.get(0, 1), 0.0);
    }

    #[test]
    fn test_csc_from_dense() {
        // Column-major: cols[j][i]
        let cols = vec![
            vec![1.0, 0.0, 4.0], // column 0
            vec![0.0, 3.0, 0.0], // column 1
            vec![2.0, 0.0, 5.0], // column 2
        ];
        let csc = CscMatrix::from_dense(&cols);
        assert_eq!(csc.n_rows(), 3);
        assert_eq!(csc.n_cols(), 3);
        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.get(0, 0), 1.0);
        assert_eq!(csc.get(2, 0), 4.0);
        assert_eq!(csc.get(1, 1), 3.0);
        assert_eq!(csc.get(0, 2), 2.0);
        assert_eq!(csc.get(2, 2), 5.0);
    }

    #[test]
    fn test_sparse_row_dot() {
        let csr = CsrMatrix::from_dense(&[vec![0.0, 2.0, 3.0]]);
        let row = csr.row(0);
        assert!((row.dot(&[1.0, 10.0, 100.0]) - 320.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_add_cancellation() {
        // When elements cancel to zero, they should be dropped.
        let a = CsrMatrix::from_dense(&[vec![1.0, 2.0]]);
        let b = CsrMatrix::from_dense(&[vec![-1.0, -2.0]]);
        let c = &a + &b;
        assert_eq!(c.nnz(), 0);
        assert_eq!(c.to_dense(), vec![vec![0.0, 0.0]]);
    }

    #[test]
    fn test_from_triplets_cross_row_dedup() {
        // Bug: rows ending with the same column as the next row's start
        // were incorrectly merged across row boundaries.
        // Row 0: col 2 = 1.0
        // Row 1: col 2 = 3.0
        // These must NOT merge.
        let rows = vec![0, 1];
        let cols = vec![2, 2];
        let vals = vec![1.0, 3.0];

        let csr = CsrMatrix::from_triplets(&rows, &cols, &vals, 2, 3).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.get(0, 2), 1.0);
        assert_eq!(csr.get(1, 2), 3.0);
    }

    #[test]
    fn test_from_triplets_intra_row_dedup() {
        // Duplicate entries within the same row should still be summed.
        let rows = vec![0, 0, 1, 1];
        let cols = vec![1, 1, 2, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0];

        let csr = CsrMatrix::from_triplets(&rows, &cols, &vals, 2, 3).unwrap();
        assert_eq!(csr.nnz(), 2);
        assert_eq!(csr.get(0, 1), 3.0); // 1.0 + 2.0
        assert_eq!(csr.get(1, 2), 7.0); // 3.0 + 4.0
    }

    #[test]
    fn test_csc_from_triplets_cross_row_dedup() {
        // Same bug via the CscMatrix path (CSR → transpose).
        let rows = vec![0, 1];
        let cols = vec![2, 2];
        let vals = vec![1.0, 3.0];

        let csc = CscMatrix::from_triplets(&rows, &cols, &vals, 2, 3).unwrap();
        assert_eq!(csc.nnz(), 2);
        assert_eq!(csc.get(0, 2), 1.0);
        assert_eq!(csc.get(1, 2), 3.0);
    }

    #[test]
    fn test_csc_from_triplets_roundtrip_with_dupes() {
        // Build CSC with duplicate entries, convert to CSR and back.
        let rows = vec![0, 0, 1, 2, 2];
        let cols = vec![0, 0, 1, 0, 2]; // row 0, col 0 has duplicates
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let csc = CscMatrix::from_triplets(&rows, &cols, &vals, 3, 3).unwrap();
        assert_eq!(csc.get(0, 0), 3.0); // 1.0 + 2.0
        assert_eq!(csc.get(1, 1), 3.0);
        assert_eq!(csc.get(2, 0), 4.0);
        assert_eq!(csc.get(2, 2), 5.0);

        // Round-trip.
        let csr = csc.to_csr();
        let csc2 = csr.to_csc();
        assert_eq!(csc.to_dense(), csc2.to_dense());
    }

    #[test]
    fn test_sparse_row_accessors() {
        let csr = CsrMatrix::from_dense(&[vec![0.0, 5.0, 0.0, 7.0]]);
        let row = csr.row(0);
        assert_eq!(row.indices(), &[1, 3]);
        assert_eq!(row.values(), &[5.0, 7.0]);
    }
}
