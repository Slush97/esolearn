// SPDX-License-Identifier: MIT OR Apache-2.0
//! Contiguous column-major dense matrix for ML workloads.
//!
//! [`DenseMatrix`] stores all feature data in a single `Vec<f64>` with
//! column-major layout (`data[col * n_rows + row]`), giving cache-friendly
//! column access and eliminating per-column heap allocations.

use crate::error::{Result, ScryLearnError};

/// A contiguous, column-major dense matrix.
///
/// Layout: `data[col * n_rows + row]`.
///
/// This replaces `Vec<Vec<f64>>` for feature storage, providing:
/// - Zero-cost column slicing via [`col`](Self::col)
/// - Single contiguous allocation instead of N+1 heap blocks
/// - Cache-friendly access patterns for column-oriented ML algorithms
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DenseMatrix {
    /// Flat storage in column-major order.
    data: Vec<f64>,
    /// Number of rows (samples).
    n_rows: usize,
    /// Number of columns (features).
    n_cols: usize,
}

impl DenseMatrix {
    /// Create a matrix from a flat column-major buffer.
    ///
    /// Returns an error if `data.len() != n_rows * n_cols`.
    pub fn new(data: Vec<f64>, n_rows: usize, n_cols: usize) -> Result<Self> {
        if data.len() != n_rows * n_cols {
            return Err(ScryLearnError::InvalidParameter(format!(
                "DenseMatrix::new: data.len()={} but n_rows*n_cols={}",
                data.len(),
                n_rows * n_cols,
            )));
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }

    /// Create a matrix of all zeros.
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        Self {
            data: vec![0.0; n_rows * n_cols],
            n_rows,
            n_cols,
        }
    }

    /// Build from column-major `Vec<Vec<f64>>` (each inner vec is one column).
    ///
    /// Returns an error if columns have different lengths.
    pub fn from_col_major(cols: Vec<Vec<f64>>) -> Result<Self> {
        if cols.is_empty() {
            return Ok(Self {
                data: Vec::new(),
                n_rows: 0,
                n_cols: 0,
            });
        }
        let n_rows = cols[0].len();
        let n_cols = cols.len();
        for (i, col) in cols.iter().enumerate() {
            if col.len() != n_rows {
                return Err(ScryLearnError::InvalidParameter(format!(
                    "DenseMatrix::from_col_major: column {i} has {} rows, expected {n_rows}",
                    col.len(),
                )));
            }
        }
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for col in &cols {
            data.extend_from_slice(col);
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }

    /// Build from row-major data, transposing into column-major storage.
    pub fn from_row_major(rows: &[&[f64]], n_rows: usize, n_cols: usize) -> Self {
        let mut data = vec![0.0; n_rows * n_cols];
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[j * n_rows + i] = val;
            }
        }
        Self {
            data,
            n_rows,
            n_cols,
        }
    }

    /// Zero-cost slice of column `j`.
    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        let start = j * self.n_rows;
        &self.data[start..start + self.n_rows]
    }

    /// Mutable slice of column `j`.
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        let start = j * self.n_rows;
        &mut self.data[start..start + self.n_rows]
    }

    /// Get a single element.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[col * self.n_rows + row]
    }

    /// Set a single element.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[col * self.n_rows + row] = val;
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

    /// The raw flat buffer in column-major order.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Iterate over values in row `i` (strided access across columns).
    pub fn row_iter(&self, i: usize) -> impl Iterator<Item = f64> + '_ {
        (0..self.n_cols).map(move |j| self.data[j * self.n_rows + i])
    }

    /// Collect row `i` into a `Vec<f64>`.
    pub fn row_to_vec(&self, i: usize) -> Vec<f64> {
        self.row_iter(i).collect()
    }

    /// Build from a reference to column-major `&[Vec<f64>]` (no ownership transfer).
    ///
    /// Same as [`from_col_major`](Self::from_col_major) but borrows the columns
    /// instead of consuming them, avoiding a clone of the outer `Vec`.
    pub fn from_col_major_ref(cols: &[Vec<f64>]) -> Result<Self> {
        if cols.is_empty() {
            return Ok(Self {
                data: Vec::new(),
                n_rows: 0,
                n_cols: 0,
            });
        }
        let n_rows = cols[0].len();
        let n_cols = cols.len();
        for (i, col) in cols.iter().enumerate() {
            if col.len() != n_rows {
                return Err(ScryLearnError::InvalidParameter(format!(
                    "DenseMatrix::from_col_major_ref: column {i} has {} rows, expected {n_rows}",
                    col.len(),
                )));
            }
        }
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for col in cols {
            data.extend_from_slice(col);
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }

    /// Convert back to `Vec<Vec<f64>>` column-major (backward compat).
    pub fn to_col_vecs(&self) -> Vec<Vec<f64>> {
        (0..self.n_cols).map(|j| self.col(j).to_vec()).collect()
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl From<Vec<Vec<f64>>> for DenseMatrix {
    /// Convert from column-major `Vec<Vec<f64>>`. Panics on ragged input.
    fn from(cols: Vec<Vec<f64>>) -> Self {
        Self::from_col_major(cols).expect("ragged column vectors in DenseMatrix::from")
    }
}

impl From<&[Vec<f64>]> for DenseMatrix {
    fn from(cols: &[Vec<f64>]) -> Self {
        let owned: Vec<Vec<f64>> = cols.to_vec();
        Self::from(owned)
    }
}

// ---------------------------------------------------------------------------
// Serde support
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
impl serde::Serialize for DenseMatrix {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("DenseMatrix", 3)?;
        state.serialize_field("data", &self.data)?;
        state.serialize_field("n_rows", &self.n_rows)?;
        state.serialize_field("n_cols", &self.n_cols)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for DenseMatrix {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Raw {
            data: Vec<f64>,
            n_rows: usize,
            n_cols: usize,
        }
        let raw = Raw::deserialize(deserializer)?;
        Self::new(raw.data, raw.n_rows, raw.n_cols).map_err(serde::de::Error::custom)
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
    fn from_col_major_roundtrip() {
        let cols = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let m = DenseMatrix::from_col_major(cols.clone()).unwrap();
        assert_eq!(m.n_rows(), 3);
        assert_eq!(m.n_cols(), 2);
        assert_eq!(m.to_col_vecs(), cols);
    }

    #[test]
    fn col_correctness() {
        let m = DenseMatrix::from_col_major(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]])
            .unwrap();
        assert_eq!(m.col(0), &[1.0, 2.0]);
        assert_eq!(m.col(1), &[3.0, 4.0]);
        assert_eq!(m.col(2), &[5.0, 6.0]);
    }

    #[test]
    fn row_iter_correctness() {
        let m =
            DenseMatrix::from_col_major(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let row0: Vec<f64> = m.row_iter(0).collect();
        assert_eq!(row0, vec![1.0, 4.0]);
        let row2: Vec<f64> = m.row_iter(2).collect();
        assert_eq!(row2, vec![3.0, 6.0]);
    }

    #[test]
    fn get_set_indexing() {
        let mut m = DenseMatrix::zeros(3, 2);
        m.set(1, 0, 42.0);
        m.set(2, 1, 99.0);
        assert_eq!(m.get(1, 0), 42.0);
        assert_eq!(m.get(2, 1), 99.0);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn from_vec_vec_conversion() {
        let cols = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let m: DenseMatrix = cols.into();
        assert_eq!(m.n_rows(), 2);
        assert_eq!(m.n_cols(), 2);
        assert_eq!(m.get(0, 0), 10.0);
        assert_eq!(m.get(1, 1), 40.0);
    }

    #[test]
    fn from_slice_conversion() {
        let cols = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let m: DenseMatrix = cols.as_slice().into();
        assert_eq!(m.col(0), &[1.0, 2.0]);
    }

    #[test]
    fn empty_matrix() {
        let m = DenseMatrix::from_col_major(vec![]).unwrap();
        assert_eq!(m.n_rows(), 0);
        assert_eq!(m.n_cols(), 0);
        assert_eq!(m.as_slice(), &[] as &[f64]);
    }

    #[test]
    fn zero_row_matrix() {
        let m = DenseMatrix::from_col_major(vec![vec![], vec![]]).unwrap();
        assert_eq!(m.n_rows(), 0);
        assert_eq!(m.n_cols(), 2);
    }

    #[test]
    fn single_column() {
        let m = DenseMatrix::from_col_major(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert_eq!(m.n_cols(), 1);
        assert_eq!(m.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.row_to_vec(1), vec![2.0]);
    }

    #[test]
    fn ragged_error() {
        let result = DenseMatrix::from_col_major(vec![vec![1.0, 2.0], vec![3.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn new_validates_length() {
        assert!(DenseMatrix::new(vec![1.0, 2.0, 3.0], 2, 2).is_err());
        assert!(DenseMatrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).is_ok());
    }

    #[test]
    fn from_row_major_transposes() {
        let rows: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];
        let m = DenseMatrix::from_row_major(&rows, 3, 2);
        // Column 0 should be [1, 3, 5], column 1 should be [2, 4, 6]
        assert_eq!(m.col(0), &[1.0, 3.0, 5.0]);
        assert_eq!(m.col(1), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn col_mut_works() {
        let mut m = DenseMatrix::zeros(3, 2);
        let col = m.col_mut(1);
        col[0] = 10.0;
        col[1] = 20.0;
        col[2] = 30.0;
        assert_eq!(m.col(1), &[10.0, 20.0, 30.0]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip() {
        let m = DenseMatrix::from_col_major(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        let m2: DenseMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(m.as_slice(), m2.as_slice());
        assert_eq!(m.n_rows(), m2.n_rows());
        assert_eq!(m.n_cols(), m2.n_cols());
    }
}
