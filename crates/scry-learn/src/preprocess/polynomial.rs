// SPDX-License-Identifier: MIT OR Apache-2.0
//! Polynomial and interaction feature expansion.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::preprocess::Transformer;

/// Generate polynomial and interaction features.
///
/// Transforms an input feature set `[x1, x2, …]` into all polynomial
/// combinations up to a given `degree`. For example, with `degree=2` and
/// 2 features:
///
/// - `include_bias=true`:  `[1, x1, x2, x1², x1·x2, x2²]`
/// - `interaction_only=true`: `[1, x1, x2, x1·x2]` (no self-powers)
///
/// # Example
///
/// ```ignore
/// let mut poly = PolynomialFeatures::new().degree(2);
/// poly.fit_transform(&mut ds)?;
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct PolynomialFeatures {
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
    /// Stored combo descriptors: each is a Vec of (original_col_idx, power) pairs.
    combos: Vec<Vec<(usize, usize)>>,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl PolynomialFeatures {
    /// Create a new `PolynomialFeatures` with default settings (degree=2).
    pub fn new() -> Self {
        Self {
            degree: 2,
            interaction_only: false,
            include_bias: true,
            combos: Vec::new(),
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the maximum polynomial degree.
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// If true, only interaction features are produced (no self-powers like x²).
    pub fn interaction_only(mut self, v: bool) -> Self {
        self.interaction_only = v;
        self
    }

    /// If true, include a bias (all-ones) column.
    pub fn include_bias(mut self, v: bool) -> Self {
        self.include_bias = v;
        self
    }

    /// Number of output features after transform.
    pub fn n_output_features(&self) -> usize {
        self.combos.len()
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// Recursively enumerate all monomial combinations of `n_features` variables
/// up to `degree`, in graded lexicographic order.
/// Generate combos of exactly `target_deg` total degree,
/// using feature indices >= `start`.
fn gen_combos(
    n_features: usize,
    remaining_deg: usize,
    start: usize,
    interaction_only: bool,
    current: &mut Vec<(usize, usize)>,
    out: &mut Vec<Vec<(usize, usize)>>,
) {
    if remaining_deg == 0 {
        out.push(current.clone());
        return;
    }
    for col in start..n_features {
        let max_power = if interaction_only { 1 } else { remaining_deg };
        // Try powers from highest to lowest so x1^2 comes before x1*x2.
        for power in (1..=max_power).rev() {
            if power > remaining_deg {
                continue;
            }
            current.push((col, power));
            gen_combos(
                n_features,
                remaining_deg - power,
                col + 1,
                interaction_only,
                current,
                out,
            );
            current.pop();
        }
    }
}

fn enumerate_combos(
    n_features: usize,
    degree: usize,
    interaction_only: bool,
    include_bias: bool,
) -> Vec<Vec<(usize, usize)>> {
    let mut result = Vec::new();

    for deg in 0..=degree {
        if deg == 0 {
            if include_bias {
                result.push(Vec::new()); // bias term
            }
        } else if deg == 1 {
            for col in 0..n_features {
                result.push(vec![(col, 1)]);
            }
        } else {
            let mut current = Vec::new();
            gen_combos(
                n_features,
                deg,
                0,
                interaction_only,
                &mut current,
                &mut result,
            );
        }
    }

    result
}

impl Transformer for PolynomialFeatures {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        self.combos = enumerate_combos(
            data.n_features(),
            self.degree,
            self.interaction_only,
            self.include_bias,
        );
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n = data.n_samples();
        let old_features = data.features.clone();

        let mut new_features: Vec<Vec<f64>> = Vec::with_capacity(self.combos.len());
        let mut new_names: Vec<String> = Vec::with_capacity(self.combos.len());

        for combo in &self.combos {
            let mut col = vec![1.0; n];
            let mut name_parts = Vec::new();

            for &(feat_idx, power) in combo {
                #[allow(clippy::cast_possible_wrap)]
                let exp = power as i32;
                for (i, val) in col.iter_mut().enumerate() {
                    *val *= old_features[feat_idx][i].powi(exp);
                }
                let fname = data
                    .feature_names
                    .get(feat_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("x{feat_idx}"));
                if power == 1 {
                    name_parts.push(fname);
                } else {
                    name_parts.push(format!("{fname}^{power}"));
                }
            }

            if name_parts.is_empty() {
                new_names.push("1".into());
            } else {
                new_names.push(name_parts.join("*"));
            }
            new_features.push(col);
        }

        data.features = new_features;
        data.feature_names = new_names;
        data.sync_matrix();
        Ok(())
    }

    fn inverse_transform(&self, _data: &mut Dataset) -> Result<()> {
        Err(ScryLearnError::InvalidParameter(
            "PolynomialFeatures is not invertible".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_degree2_basic() {
        // Input: 2 features, 2 samples: [[1,2],[3,4]]
        let mut ds = Dataset::new(
            vec![vec![1.0, 3.0], vec![2.0, 4.0]],
            vec![0.0, 1.0],
            vec!["x1".into(), "x2".into()],
            "y",
        );
        let mut poly = PolynomialFeatures::new().degree(2).include_bias(true);
        poly.fit_transform(&mut ds).unwrap();

        // Expected columns: [1, x1, x2, x1^2, x1*x2, x2^2]
        assert_eq!(ds.n_features(), 6);

        // Row 0: [1, 1, 2, 1, 2, 4]
        let row0: Vec<f64> = ds.features.iter().map(|c| c[0]).collect();
        assert_eq!(row0, vec![1.0, 1.0, 2.0, 1.0, 2.0, 4.0]);

        // Row 1: [1, 3, 4, 9, 12, 16]
        let row1: Vec<f64> = ds.features.iter().map(|c| c[1]).collect();
        assert_eq!(row1, vec![1.0, 3.0, 4.0, 9.0, 12.0, 16.0]);
    }

    #[test]
    fn test_poly_interaction_only() {
        let mut ds = Dataset::new(
            vec![vec![1.0, 3.0], vec![2.0, 4.0]],
            vec![0.0, 1.0],
            vec!["x1".into(), "x2".into()],
            "y",
        );
        let mut poly = PolynomialFeatures::new()
            .degree(2)
            .interaction_only(true)
            .include_bias(true);
        poly.fit_transform(&mut ds).unwrap();

        // Expected: [1, x1, x2, x1*x2] — no x1^2, x2^2
        assert_eq!(ds.n_features(), 4);

        let row0: Vec<f64> = ds.features.iter().map(|c| c[0]).collect();
        assert_eq!(row0, vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_poly_no_bias() {
        let mut ds = Dataset::new(
            vec![vec![2.0], vec![3.0]],
            vec![0.0],
            vec!["a".into(), "b".into()],
            "y",
        );
        let mut poly = PolynomialFeatures::new().degree(2).include_bias(false);
        poly.fit_transform(&mut ds).unwrap();

        // No bias column, so first col should be a feature, not 1.
        let first_vals = &ds.features[0];
        assert!((first_vals[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_degree3() {
        let mut ds = Dataset::new(vec![vec![2.0]], vec![0.0], vec!["x".into()], "y");
        let mut poly = PolynomialFeatures::new().degree(3).include_bias(true);
        poly.fit_transform(&mut ds).unwrap();

        // [1, x, x^2, x^3] → [1, 2, 4, 8]
        assert_eq!(ds.n_features(), 4);
        let row: Vec<f64> = ds.features.iter().map(|c| c[0]).collect();
        assert_eq!(row, vec![1.0, 2.0, 4.0, 8.0]);
    }

    #[test]
    fn test_poly_not_fitted() {
        let poly = PolynomialFeatures::new();
        let mut ds = Dataset::new(vec![vec![1.0]], vec![0.0], vec!["x".into()], "y");
        assert!(poly.transform(&mut ds).is_err());
    }
}
