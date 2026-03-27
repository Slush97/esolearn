// SPDX-License-Identifier: MIT OR Apache-2.0
//! Feature binning for histogram-based gradient boosting.
//!
//! [`FeatureBinner`] quantile-bins continuous `f64` features into `u8`
//! indices (0..=255). Bin 0 is reserved for missing values (`NaN`);
//! valid data maps to bins 1..=255.
//!
//! # Example
//! ```
//! use scry_learn::dataset::Dataset;
//! use scry_learn::tree::FeatureBinner;
//!
//! let features = vec![
//!     vec![1.0, 2.0, f64::NAN, 4.0, 5.0],
//!     vec![10.0, 20.0, 30.0, 40.0, 50.0],
//! ];
//! let target = vec![0.0; 5];
//! let data = Dataset::new(features, target, vec!["a".into(), "b".into()], "y");
//!
//! let mut binner = FeatureBinner::new();
//! binner.fit(&data).unwrap();
//! let binned = binner.transform(&data).unwrap();
//!
//! // NaN → bin 0, valid values → bins 1..=255
//! assert_eq!(binned[0][2], 0);
//! assert!(binned[1][0] >= 1);
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

/// Maximum number of bins (including the missing-value bin 0).
pub const MAX_BINS: usize = 256;

/// Quantile-based feature binner.
///
/// Transforms each feature column into `u8` bin indices using quantile
/// boundaries computed during `fit()`. Missing values (`NaN`) are
/// mapped to bin 0; valid values to bins 1–255.
///
/// The binning is designed for histogram-based gradient boosting where
/// the O(256) histogram scan replaces the O(n log n) sorted-split search.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FeatureBinner {
    /// Bin edges per feature: `bin_edges[feature][edge_idx]`.
    /// For K valid bins there are K−1 edges (upper-exclusive boundaries).
    bin_edges: Vec<Vec<f64>>,
    /// Number of actual valid bins per feature (may be < 255 for
    /// low-cardinality features).
    n_bins_per_feature: Vec<usize>,
    max_bins: usize,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl FeatureBinner {
    /// Create a new binner with the default 256 max bins.
    ///
    /// # Example
    /// ```
    /// use scry_learn::tree::FeatureBinner;
    /// let binner = FeatureBinner::new();
    /// ```
    pub fn new() -> Self {
        Self {
            bin_edges: Vec::new(),
            n_bins_per_feature: Vec::new(),
            max_bins: MAX_BINS,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the maximum number of bins (2..=256, default 256).
    pub fn max_bins(mut self, bins: usize) -> Self {
        self.max_bins = bins.clamp(2, MAX_BINS);
        self
    }

    /// Compute bin edges from training data.
    ///
    /// For each feature column, sorts the non-NaN values and picks
    /// equally-spaced quantile boundaries to create up to `max_bins - 1`
    /// valid bins (bin 0 is reserved for missing).
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_no_inf()?;
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        let n_features = data.n_features();
        let valid_bins = self.max_bins - 1; // reserve bin 0 for NaN

        self.bin_edges = Vec::with_capacity(n_features);
        self.n_bins_per_feature = Vec::with_capacity(n_features);

        for f in 0..n_features {
            let col = &data.features[f];

            // Collect and sort non-NaN values.
            let mut valid: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();
            valid.sort_unstable_by(|a, b| a.total_cmp(b));

            if valid.is_empty() {
                // All NaN — single bin (the missing bin).
                self.bin_edges.push(Vec::new());
                self.n_bins_per_feature.push(1);
                continue;
            }

            // Deduplicate to find unique values.
            valid.dedup();

            let n_unique = valid.len();
            let actual_bins = n_unique.min(valid_bins);

            if actual_bins <= 1 {
                // Constant feature — one valid bin.
                self.bin_edges.push(Vec::new());
                self.n_bins_per_feature.push(1);
                continue;
            }

            // Compute quantile-based bin edges: `actual_bins - 1` thresholds.
            let mut edges = Vec::with_capacity(actual_bins - 1);
            for i in 1..actual_bins {
                let q = i as f64 / actual_bins as f64;
                let pos = q * (valid.len() - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(valid.len() - 1);
                let frac = pos - lo as f64;
                let edge = valid[lo] * (1.0 - frac) + valid[hi] * frac;
                edges.push(edge);
            }

            // Remove duplicate edges (low-cardinality features).
            edges.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

            let n_valid_bins = edges.len() + 1;
            self.n_bins_per_feature.push(n_valid_bins);
            self.bin_edges.push(edges);
        }

        self.fitted = true;
        Ok(())
    }

    /// Map features to `u8` bin indices.
    ///
    /// Returns `binned[feature_idx][sample_idx]`. NaN → 0, valid → 1..=255.
    pub fn transform(&self, data: &Dataset) -> Result<Vec<Vec<u8>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n_features = data.n_features();
        if n_features != self.bin_edges.len() {
            return Err(ScryLearnError::ShapeMismatch {
                expected: self.bin_edges.len(),
                got: n_features,
            });
        }

        let n_samples = data.n_samples();
        let mut result = Vec::with_capacity(n_features);

        for f in 0..n_features {
            let col = &data.features[f];
            let edges = &self.bin_edges[f];
            let mut binned = vec![0u8; n_samples];

            for (i, &val) in col.iter().enumerate() {
                if val.is_nan() {
                    binned[i] = 0; // missing-value bin
                } else {
                    // Binary search for the correct bin.
                    let bin = match edges.binary_search_by(|edge| {
                        edge.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        Ok(pos) => pos + 1, // on edge → next bin
                        Err(pos) => pos,
                    };
                    // Shift by 1 because bin 0 is reserved for NaN.
                    binned[i] = (bin + 1).min(255) as u8;
                }
            }

            result.push(binned);
        }

        Ok(result)
    }

    /// Combined fit + transform.
    pub fn fit_transform(&mut self, data: &Dataset) -> Result<Vec<Vec<u8>>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Number of bins per feature (including the missing-value bin).
    pub fn n_bins_per_feature(&self) -> &[usize] {
        &self.n_bins_per_feature
    }

    /// Bin edges per feature.
    pub fn bin_edges(&self) -> &[Vec<f64>] {
        &self.bin_edges
    }
}

impl Default for FeatureBinner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_dataset() -> Dataset {
        Dataset::new(
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                vec![
                    100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0,
                ],
            ],
            vec![0.0; 10],
            vec!["a".into(), "b".into()],
            "y",
        )
    }

    #[test]
    fn test_fit_transform_basic() {
        let ds = simple_dataset();
        let mut binner = FeatureBinner::new();
        let binned = binner.fit_transform(&ds).unwrap();
        assert_eq!(binned.len(), 2);
        assert_eq!(binned[0].len(), 10);
        // All values should be > 0 (no NaN in data).
        for &b in &binned[0] {
            assert!(b >= 1, "valid values should map to bins >= 1");
        }
        // Monotonicity: sorted input → sorted bins.
        for i in 1..10 {
            assert!(binned[0][i] >= binned[0][i - 1]);
        }
    }

    #[test]
    fn test_nan_handling() {
        let ds = Dataset::new(
            vec![vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]],
            vec![0.0; 5],
            vec!["x".into()],
            "y",
        );
        let mut binner = FeatureBinner::new();
        let binned = binner.fit_transform(&ds).unwrap();
        assert_eq!(binned[0][1], 0, "NaN should map to bin 0");
        assert_eq!(binned[0][3], 0, "NaN should map to bin 0");
        assert!(binned[0][0] >= 1, "valid value should be >= 1");
    }

    #[test]
    fn test_constant_feature() {
        let ds = Dataset::new(
            vec![vec![5.0, 5.0, 5.0, 5.0]],
            vec![0.0; 4],
            vec!["x".into()],
            "y",
        );
        let mut binner = FeatureBinner::new();
        let binned = binner.fit_transform(&ds).unwrap();
        // All should map to the same bin (>= 1).
        let first = binned[0][0];
        for &b in &binned[0] {
            assert_eq!(b, first);
        }
    }

    #[test]
    fn test_max_bins_param() {
        let ds = simple_dataset();
        let mut binner = FeatureBinner::new().max_bins(4);
        let binned = binner.fit_transform(&ds).unwrap();
        // With max_bins=4, valid bins are 1..=3, so max bin index should be <= 3.
        for &b in &binned[0] {
            assert!(b <= 3, "with max_bins=4, bin index should be <= 3, got {b}");
        }
    }

    #[test]
    fn test_not_fitted_error() {
        let ds = simple_dataset();
        let binner = FeatureBinner::new();
        let result = binner.transform(&ds);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_nan_feature() {
        let ds = Dataset::new(
            vec![vec![f64::NAN, f64::NAN, f64::NAN]],
            vec![0.0; 3],
            vec!["x".into()],
            "y",
        );
        let mut binner = FeatureBinner::new();
        let binned = binner.fit_transform(&ds).unwrap();
        for &b in &binned[0] {
            assert_eq!(b, 0, "all-NaN feature should map entirely to bin 0");
        }
    }
}
