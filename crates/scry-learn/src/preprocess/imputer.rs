// SPDX-License-Identifier: MIT OR Apache-2.0
//! Missing-value imputation.
//!
//! [`SimpleImputer`] replaces `NaN` values in a [`Dataset`] with a
//! statistic computed from the non-missing entries of each feature column.
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::preprocess::{SimpleImputer, Strategy, Transformer};
//!
//! let mut imputer = SimpleImputer::new().strategy(Strategy::Mean);
//! imputer.fit_transform(&mut dataset)?;
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::preprocess::Transformer;

/// Strategy for computing the replacement value per feature.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Strategy {
    /// Replace with the arithmetic mean of non-`NaN` values.
    #[default]
    Mean,
    /// Replace with the median of non-`NaN` values.
    Median,
    /// Replace with the most frequent non-`NaN` value (mode).
    MostFrequent,
    /// Replace with a user-specified constant.
    Constant(f64),
}

/// Imputes missing (`NaN`) values in each feature column.
///
/// During [`fit`](Transformer::fit), the imputer computes one fill value
/// per feature using the chosen [`Strategy`]. During
/// [`transform`](Transformer::transform), every `NaN` in a column is
/// replaced with that value.
///
/// # Example
///
/// ```ignore
/// let mut imp = SimpleImputer::new().strategy(Strategy::Median);
/// imp.fit_transform(&mut ds)?;
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SimpleImputer {
    strategy: Strategy,
    fill_values: Vec<f64>,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl SimpleImputer {
    /// Create a new unfitted imputer (defaults to [`Strategy::Mean`]).
    pub fn new() -> Self {
        Self {
            strategy: Strategy::default(),
            fill_values: Vec::new(),
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the imputation strategy.
    pub fn strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Return the per-feature fill values computed during `fit`.
    ///
    /// # Panics
    ///
    /// Panics if the imputer has not been fitted.
    pub fn fill_values(&self) -> &[f64] {
        &self.fill_values
    }
}

impl Default for SimpleImputer {
    fn default() -> Self {
        Self::new()
    }
}

// ── helpers ──────────────────────────────────────────────────────

/// Compute the mean of values that are not NaN.
fn mean_ignore_nan(col: &[f64]) -> f64 {
    let (sum, count) = col
        .iter()
        .filter(|x| !x.is_nan())
        .fold((0.0, 0usize), |(s, c), &v| (s + v, c + 1));
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Compute the median of values that are not NaN.
fn median_ignore_nan(col: &[f64]) -> f64 {
    let mut valid: Vec<f64> = col.iter().copied().filter(|x| !x.is_nan()).collect();
    if valid.is_empty() {
        return 0.0;
    }
    valid.sort_unstable_by(|a, b| a.total_cmp(b));
    let mid = valid.len() / 2;
    if valid.len() % 2 == 0 {
        (valid[mid - 1] + valid[mid]) / 2.0
    } else {
        valid[mid]
    }
}

/// Compute the most frequent value (mode) ignoring NaN.
/// Ties are broken by choosing the smallest value.
fn mode_ignore_nan(col: &[f64]) -> f64 {
    use std::collections::HashMap;

    let mut counts: HashMap<u64, (f64, usize)> = HashMap::new();
    for &v in col {
        if v.is_nan() {
            continue;
        }
        let key = v.to_bits();
        counts
            .entry(key)
            .and_modify(|(_, c)| *c += 1)
            .or_insert((v, 1));
    }
    if counts.is_empty() {
        return 0.0;
    }
    counts
        .into_values()
        .max_by(|(v1, c1), (v2, c2)| c1.cmp(c2).then_with(|| v2.total_cmp(v1)))
        .map_or(0.0, |(v, _)| v)
}

impl Transformer for SimpleImputer {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_no_inf()?;
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.fill_values = Vec::with_capacity(data.n_features());

        for col in &data.features {
            let fill = match &self.strategy {
                Strategy::Mean => mean_ignore_nan(col),
                Strategy::Median => median_ignore_nan(col),
                Strategy::MostFrequent => mode_ignore_nan(col),
                Strategy::Constant(v) => *v,
            };
            self.fill_values.push(fill);
        }
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        for (j, col) in data.features.iter_mut().enumerate() {
            let fill = self.fill_values[j];
            for x in col.iter_mut() {
                if x.is_nan() {
                    *x = fill;
                }
            }
        }
        data.sync_matrix();
        Ok(())
    }

    fn inverse_transform(&self, _data: &mut Dataset) -> Result<()> {
        Err(ScryLearnError::InvalidParameter(
            "SimpleImputer is not invertible".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ds_with_nan() -> Dataset {
        Dataset::new(
            vec![
                vec![1.0, f64::NAN, 3.0, 4.0],
                vec![10.0, 20.0, f64::NAN, 40.0],
            ],
            vec![0.0; 4],
            vec!["a".into(), "b".into()],
            "y",
        )
    }

    #[test]
    fn test_imputer_mean() {
        let mut ds = ds_with_nan();
        let mut imp = SimpleImputer::new().strategy(Strategy::Mean);
        imp.fit_transform(&mut ds).unwrap();

        // col a: mean(1,3,4) = 8/3 ≈ 2.6667
        assert!(!ds.features[0][1].is_nan());
        assert!((ds.features[0][1] - 8.0 / 3.0).abs() < 1e-10);

        // col b: mean(10,20,40) = 70/3 ≈ 23.3333
        assert!(!ds.features[1][2].is_nan());
        assert!((ds.features[1][2] - 70.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_imputer_median() {
        let mut ds = ds_with_nan();
        let mut imp = SimpleImputer::new().strategy(Strategy::Median);
        imp.fit_transform(&mut ds).unwrap();

        // col a: sorted valid = [1,3,4], median = 3
        assert!((ds.features[0][1] - 3.0).abs() < 1e-10);
        // col b: sorted valid = [10,20,40], median = 20
        assert!((ds.features[1][2] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_imputer_most_frequent() {
        let mut ds = Dataset::new(
            vec![vec![1.0, 1.0, f64::NAN, 3.0, 1.0]],
            vec![0.0; 5],
            vec!["a".into()],
            "y",
        );
        let mut imp = SimpleImputer::new().strategy(Strategy::MostFrequent);
        imp.fit_transform(&mut ds).unwrap();

        // mode of [1,1,3,1] = 1
        assert!((ds.features[0][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_imputer_constant() {
        let mut ds = ds_with_nan();
        let mut imp = SimpleImputer::new().strategy(Strategy::Constant(-999.0));
        imp.fit_transform(&mut ds).unwrap();

        assert!((ds.features[0][1] - (-999.0)).abs() < 1e-10);
        assert!((ds.features[1][2] - (-999.0)).abs() < 1e-10);
    }

    #[test]
    fn test_imputer_not_fitted() {
        let imp = SimpleImputer::new();
        let mut ds = ds_with_nan();
        assert!(imp.transform(&mut ds).is_err());
    }

    #[test]
    fn test_imputer_inverse_transform_err() {
        let mut ds = ds_with_nan();
        let mut imp = SimpleImputer::new();
        imp.fit(&ds).unwrap();
        assert!(imp.inverse_transform(&mut ds).is_err());
    }
}
