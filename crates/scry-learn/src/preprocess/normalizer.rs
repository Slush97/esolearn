// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row-wise sample normalization.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::preprocess::Transformer;

/// Norm type for row-wise normalization.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Norm {
    /// Divide each row by the sum of absolute values.
    L1,
    /// Divide each row by its Euclidean (L2) norm.
    L2,
    /// Divide each row by its maximum absolute value.
    Max,
}

/// Normalize samples individually to unit norm.
///
/// Each sample (row) is scaled independently so that its chosen norm
/// equals 1.0. This is useful for text classification or clustering
/// where the direction of the feature vector matters more than magnitude.
///
/// `fit()` is a no-op — normalizer is stateless.
///
/// # Example
///
/// ```ignore
/// let mut norm = Normalizer::new(Norm::L2);
/// norm.transform(&mut ds)?;
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Normalizer {
    norm: Norm,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl Normalizer {
    /// Create a normalizer with the given norm type.
    pub fn new(norm: Norm) -> Self {
        Self {
            norm,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Create a normalizer with L2 norm (default).
    pub fn l2() -> Self {
        Self {
            norm: Norm::L2,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }
}

impl Default for Normalizer {
    fn default() -> Self {
        Self::l2()
    }
}

impl Transformer for Normalizer {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        // No-op: normalizer is stateless.
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        crate::version::check_schema_version(self._schema_version)?;
        let n = data.n_samples();
        let m = data.n_features();

        for i in 0..n {
            // Compute the norm for this row.
            let norm_val = match self.norm {
                Norm::L1 => {
                    let mut s = 0.0_f64;
                    for col in &data.features {
                        s += col[i].abs();
                    }
                    s
                }
                Norm::L2 => {
                    let mut s = 0.0_f64;
                    for col in &data.features {
                        s += col[i] * col[i];
                    }
                    s.sqrt()
                }
                Norm::Max => {
                    let mut mx = 0.0_f64;
                    for col in &data.features {
                        mx = mx.max(col[i].abs());
                    }
                    mx
                }
            };

            if norm_val > 1e-12 {
                for j in 0..m {
                    data.features[j][i] /= norm_val;
                }
            }
        }

        data.sync_matrix();
        Ok(())
    }

    fn inverse_transform(&self, _data: &mut Dataset) -> Result<()> {
        Err(ScryLearnError::InvalidParameter(
            "Normalizer is not invertible (row norms are lost)".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ds(rows: &[Vec<f64>]) -> Dataset {
        let n = rows.len();
        let m = rows[0].len();
        let mut features = vec![vec![0.0; n]; m];
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                features[j][i] = val;
            }
        }
        let names: Vec<String> = (0..m).map(|j| format!("f{j}")).collect();
        Dataset::new(features, vec![0.0; n], names, "y")
    }

    #[test]
    fn test_normalizer_l2_unit_norm() {
        let mut ds = make_ds(&[vec![3.0, 4.0], vec![1.0, 0.0]]);
        let mut norm = Normalizer::new(Norm::L2);
        norm.fit_transform(&mut ds).unwrap();

        // Row 0: [3,4] → norm=5 → [0.6, 0.8]
        assert!((ds.features[0][0] - 0.6).abs() < 1e-10);
        assert!((ds.features[1][0] - 0.8).abs() < 1e-10);

        // Verify unit L2 norm for each row.
        for i in 0..ds.n_samples() {
            let mut sq_sum = 0.0;
            for col in &ds.features {
                sq_sum += col[i] * col[i];
            }
            assert!(
                (sq_sum - 1.0).abs() < 1e-10,
                "row {i} L2 norm² = {sq_sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_normalizer_l1() {
        let mut ds = make_ds(&[vec![1.0, 2.0, 3.0]]);
        let mut norm = Normalizer::new(Norm::L1);
        norm.fit_transform(&mut ds).unwrap();

        // Row 0: sum_abs = 6, so [1/6, 2/6, 3/6]
        let abs_sum: f64 = ds.features.iter().map(|c| c[0].abs()).sum();
        assert!(
            (abs_sum - 1.0).abs() < 1e-10,
            "L1 norm should be 1.0, got {abs_sum}"
        );
    }

    #[test]
    fn test_normalizer_max() {
        let mut ds = make_ds(&[vec![-5.0, 2.0, 3.0]]);
        let mut norm = Normalizer::new(Norm::Max);
        norm.fit_transform(&mut ds).unwrap();

        // max_abs = 5, so [-1, 0.4, 0.6]
        assert!((ds.features[0][0] - (-1.0)).abs() < 1e-10);
        let max_abs: f64 = ds
            .features
            .iter()
            .map(|c| c[0].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            (max_abs - 1.0).abs() < 1e-10,
            "Max norm should be 1.0, got {max_abs}"
        );
    }

    #[test]
    fn test_normalizer_zero_row() {
        // Zero row should be left as-is (no division by zero).
        let mut ds = make_ds(&[vec![0.0, 0.0]]);
        let mut norm = Normalizer::new(Norm::L2);
        norm.fit_transform(&mut ds).unwrap();

        for col in &ds.features {
            assert!((col[0]).abs() < 1e-10);
        }
    }
}
