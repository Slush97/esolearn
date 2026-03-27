// SPDX-License-Identifier: MIT OR Apache-2.0
//! One-hot encoding for categorical features.
//!
//! Expands integer-encoded categorical columns into binary indicator
//! columns.  Supports `DropStrategy` (avoid multicollinearity) and
//! `UnknownStrategy` (handle unseen categories at transform time).
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::prelude::*;
//!
//! // Assume feature 0 is label-encoded colour: 0=red, 1=green, 2=blue
//! let mut enc = OneHotEncoder::new(vec![0])
//!     .drop(DropStrategy::First)
//!     .handle_unknown(UnknownStrategy::Ignore);
//! enc.fit_transform(&mut dataset)?;
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::preprocess::Transformer;

// ── Public enums ──────────────────────────────────────────────────

/// Strategy for dropping one-hot columns to avoid multicollinearity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DropStrategy {
    /// Keep all categories (default).
    #[default]
    None,
    /// Drop the first category from each feature.
    First,
    /// Drop the first category only for binary (2-category) features.
    IfBinary,
}

/// Strategy for handling categories seen at transform time but not at fit time.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum UnknownStrategy {
    /// Raise an error (default).
    #[default]
    Error,
    /// Encode as all-zeros row.
    Ignore,
}

// ── Struct ────────────────────────────────────────────────────────

/// One-hot encoder for integer-encoded categorical features.
///
/// Replaces each selected column with `n_categories` binary columns
/// (minus any dropped by the [`DropStrategy`]).  Non-selected columns
/// pass through untouched.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct OneHotEncoder {
    feature_indices: Vec<usize>,
    drop_strategy: DropStrategy,
    unknown_strategy: UnknownStrategy,
    // — fitted state —
    /// `categories[i]` = sorted unique values of `feature_indices[i]`.
    categories: Vec<Vec<f64>>,
    /// Original feature names captured at fit time.
    orig_feature_names: Vec<String>,
    fitted: bool,
}

// ── Builder ───────────────────────────────────────────────────────

impl OneHotEncoder {
    /// Create a new encoder for the given feature column indices.
    pub fn new(feature_indices: Vec<usize>) -> Self {
        Self {
            feature_indices,
            drop_strategy: DropStrategy::None,
            unknown_strategy: UnknownStrategy::Error,
            categories: Vec::new(),
            orig_feature_names: Vec::new(),
            fitted: false,
        }
    }

    /// Set the drop strategy.
    pub fn drop(mut self, strategy: DropStrategy) -> Self {
        self.drop_strategy = strategy;
        self
    }

    /// Set the unknown-category strategy.
    pub fn handle_unknown(mut self, strategy: UnknownStrategy) -> Self {
        self.unknown_strategy = strategy;
        self
    }

    // ── Accessors ─────────────────────────────────────────────────

    /// Learned categories for each encoded feature.
    pub fn categories(&self) -> &[Vec<f64>] {
        &self.categories
    }

    /// Compute the output feature names that `transform` would produce.
    pub fn get_feature_names(&self) -> Vec<String> {
        if !self.fitted || self.orig_feature_names.is_empty() {
            return Vec::new();
        }
        let encoded_set: std::collections::HashSet<usize> =
            self.feature_indices.iter().copied().collect();
        let mut names = Vec::new();
        for (j, orig_name) in self.orig_feature_names.iter().enumerate() {
            if encoded_set.contains(&j) {
                let cat_idx = self
                    .feature_indices
                    .iter()
                    .position(|&fi| fi == j)
                    .expect("encoded_set built from feature_indices");
                let cats = &self.categories[cat_idx];
                let skip = self.n_drop(cat_idx);
                for (ci, &cat_val) in cats.iter().enumerate() {
                    if ci < skip {
                        continue;
                    }
                    names.push(format!("{}_{}", orig_name, cat_val as i64));
                }
            } else {
                names.push(orig_name.clone());
            }
        }
        names
    }
}

// ── Helpers ───────────────────────────────────────────────────────

impl OneHotEncoder {
    /// Should we drop a column for feature `idx` (index into `categories`)?
    /// Returns the number of categories to *skip* from the front (0 or 1).
    fn n_drop(&self, cat_idx: usize) -> usize {
        match self.drop_strategy {
            DropStrategy::None => 0,
            DropStrategy::First => 1,
            DropStrategy::IfBinary => usize::from(self.categories[cat_idx].len() == 2),
        }
    }
}

// ── Transformer impl ─────────────────────────────────────────────

impl Transformer for OneHotEncoder {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        for &idx in &self.feature_indices {
            if idx >= data.n_features() {
                return Err(ScryLearnError::InvalidParameter(format!(
                    "feature index {idx} out of range (dataset has {} features)",
                    data.n_features()
                )));
            }
        }

        self.categories.clear();
        self.orig_feature_names.clone_from(&data.feature_names);
        for &idx in &self.feature_indices {
            let mut unique: Vec<f64> = data.features[idx].clone();
            unique.sort_by(|a, b| a.total_cmp(b));
            unique.dedup();
            self.categories.push(unique);
        }
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n = data.n_samples();

        // Build the set of encoded column indices for fast lookup.
        let encoded_set: std::collections::HashSet<usize> =
            self.feature_indices.iter().copied().collect();

        let mut new_features: Vec<Vec<f64>> = Vec::new();
        let mut new_names: Vec<String> = Vec::new();

        for j in 0..data.n_features() {
            if encoded_set.contains(&j) {
                // Find which cat_idx this corresponds to.
                let cat_idx = self
                    .feature_indices
                    .iter()
                    .position(|&fi| fi == j)
                    .ok_or(ScryLearnError::InvalidFeatureIndex(j))?;
                let cats = &self.categories[cat_idx];
                let skip = self.n_drop(cat_idx);
                let orig_name = &data.feature_names[j];

                for (ci, &cat_val) in cats.iter().enumerate() {
                    if ci < skip {
                        continue;
                    }
                    let mut col = Vec::with_capacity(n);
                    for s in 0..n {
                        let val = data.features[j][s];
                        if (val - cat_val).abs() < 1e-10 {
                            col.push(1.0);
                        } else if cats.iter().any(|&c| (val - c).abs() < 1e-10) {
                            col.push(0.0);
                        } else {
                            // Unknown category.
                            match self.unknown_strategy {
                                UnknownStrategy::Error => {
                                    return Err(ScryLearnError::InvalidParameter(format!(
                                        "unknown category {val} in feature '{orig_name}'"
                                    )));
                                }
                                UnknownStrategy::Ignore => {
                                    col.push(0.0);
                                }
                            }
                        }
                    }
                    new_features.push(col);
                    new_names.push(format!("{}_{}", orig_name, cat_val as i64));
                }
            } else {
                // Pass through.
                new_features.push(data.features[j].clone());
                new_names.push(data.feature_names[j].clone());
            }
        }

        data.features = new_features;
        data.feature_names = new_names;
        data.sync_matrix();
        Ok(())
    }

    fn inverse_transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n = data.n_samples();

        // We need to identify the one-hot column groups and collapse them back.
        // Walk through features and feature_names to reconstruct.
        let mut new_features: Vec<Vec<f64>> = Vec::new();
        let mut new_names: Vec<String> = Vec::new();

        let mut j = 0;
        let mut cat_idx = 0;

        // Build a plan: for each original feature, was it encoded?
        // We reconstruct by scanning through the current features.
        // One-hot columns are named "<orig>_<val>".
        // We detect groups by checking consecutive columns whose names
        // share a prefix matching a known encoded feature.

        // For simplicity, use the category counts directly:
        while j < data.n_features() {
            if cat_idx < self.feature_indices.len() {
                let cats = &self.categories[cat_idx];
                let skip = self.n_drop(cat_idx);
                let n_cols = cats.len() - skip;

                // Check if the current block of columns looks like one-hot.
                if j + n_cols <= data.n_features() {
                    // Try to extract the original feature name from the first column name.
                    let first_name = &data.feature_names[j];
                    let prefix = first_name
                        .rfind('_')
                        .map_or(first_name.as_str(), |pos| &first_name[..pos]);

                    // Collapse: for each sample, find which column is 1.
                    let mut col = Vec::with_capacity(n);
                    for s in 0..n {
                        let mut found = false;
                        for (ci, &cat_val) in cats.iter().enumerate().skip(skip) {
                            let col_idx = j + ci - skip;
                            if data.features[col_idx][s] > 0.5 {
                                col.push(cat_val);
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            // If drop was used, the dropped category is the one
                            // where all columns are zero.
                            if skip > 0 {
                                col.push(cats[0]);
                            } else {
                                col.push(f64::NAN);
                            }
                        }
                    }
                    new_features.push(col);
                    new_names.push(prefix.to_string());
                    j += n_cols;
                    cat_idx += 1;
                    continue;
                }
            }

            // Pass through.
            new_features.push(data.features[j].clone());
            new_names.push(data.feature_names[j].clone());
            j += 1;
        }

        data.features = new_features;
        data.feature_names = new_names;
        data.sync_matrix();
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn color_dataset() -> Dataset {
        // Feature 0: color (0=red, 1=green, 2=blue), Feature 1: numeric.
        Dataset::new(
            vec![
                vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ],
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec!["color".into(), "value".into()],
            "target",
        )
    }

    #[test]
    fn onehot_basic_encoding() {
        let mut ds = color_dataset();
        let mut enc = OneHotEncoder::new(vec![0]);
        enc.fit_transform(&mut ds).unwrap();

        // 3 one-hot columns + 1 passthrough = 4 features.
        assert_eq!(ds.n_features(), 4);
        assert_eq!(ds.feature_names[0], "color_0");
        assert_eq!(ds.feature_names[1], "color_1");
        assert_eq!(ds.feature_names[2], "color_2");
        assert_eq!(ds.feature_names[3], "value");

        // Sample 0: color=0 → [1, 0, 0]
        assert_eq!(ds.features[0][0], 1.0);
        assert_eq!(ds.features[1][0], 0.0);
        assert_eq!(ds.features[2][0], 0.0);

        // Sample 2: color=2 → [0, 0, 1]
        assert_eq!(ds.features[0][2], 0.0);
        assert_eq!(ds.features[1][2], 0.0);
        assert_eq!(ds.features[2][2], 1.0);
    }

    #[test]
    fn onehot_drop_first() {
        let mut ds = color_dataset();
        let mut enc = OneHotEncoder::new(vec![0]).drop(DropStrategy::First);
        enc.fit_transform(&mut ds).unwrap();

        // 2 one-hot columns (dropped first) + 1 passthrough = 3 features.
        assert_eq!(ds.n_features(), 3);
        assert_eq!(ds.feature_names[0], "color_1");
        assert_eq!(ds.feature_names[1], "color_2");
    }

    #[test]
    fn onehot_drop_if_binary() {
        // Binary feature: only 2 categories.
        let mut ds = Dataset::new(
            vec![vec![0.0, 1.0, 0.0, 1.0], vec![10.0, 20.0, 30.0, 40.0]],
            vec![0.0; 4],
            vec!["binary".into(), "num".into()],
            "y",
        );
        let mut enc = OneHotEncoder::new(vec![0]).drop(DropStrategy::IfBinary);
        enc.fit_transform(&mut ds).unwrap();

        // Binary → drop first → 1 column + 1 passthrough = 2.
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.feature_names[0], "binary_1");

        // Non-binary (3 cats) should keep all.
        let mut ds3 = color_dataset();
        let mut enc3 = OneHotEncoder::new(vec![0]).drop(DropStrategy::IfBinary);
        enc3.fit_transform(&mut ds3).unwrap();
        assert_eq!(ds3.n_features(), 4); // 3 one-hot + 1 passthrough
    }

    #[test]
    fn onehot_unknown_error() {
        let mut ds = color_dataset();
        let mut enc = OneHotEncoder::new(vec![0]);
        enc.fit(&ds).unwrap();

        // Inject an unknown category.
        ds.features[0][0] = 99.0;
        assert!(enc.transform(&mut ds).is_err());
    }

    #[test]
    fn onehot_unknown_ignore() {
        let mut ds = color_dataset();
        let mut enc = OneHotEncoder::new(vec![0]).handle_unknown(UnknownStrategy::Ignore);
        enc.fit(&ds).unwrap();

        // Inject unknown.
        ds.features[0][0] = 99.0;
        enc.transform(&mut ds).unwrap();

        // Unknown → all zeros.
        assert_eq!(ds.features[0][0], 0.0); // color_0
        assert_eq!(ds.features[1][0], 0.0); // color_1
        assert_eq!(ds.features[2][0], 0.0); // color_2
    }

    #[test]
    fn onehot_roundtrip_inverse() {
        let original = color_dataset();
        let mut ds = original.clone();
        let mut enc = OneHotEncoder::new(vec![0]);
        enc.fit_transform(&mut ds).unwrap();
        enc.inverse_transform(&mut ds).unwrap();

        assert_eq!(ds.n_features(), 2);
        for i in 0..original.n_samples() {
            assert!(
                (ds.features[0][i] - original.features[0][i]).abs() < 1e-10,
                "roundtrip mismatch at sample {i}"
            );
        }
    }

    #[test]
    fn onehot_feature_names() {
        let mut ds = color_dataset();
        let mut enc = OneHotEncoder::new(vec![0]);
        enc.fit_transform(&mut ds).unwrap();

        let names = enc.get_feature_names();
        assert_eq!(names, &["color_0", "color_1", "color_2", "value"]);
    }

    #[test]
    fn onehot_not_fitted_error() {
        let enc = OneHotEncoder::new(vec![0]);
        let mut ds = color_dataset();
        assert!(enc.transform(&mut ds).is_err());
    }

    #[test]
    fn onehot_multiple_features() {
        // Encode two features simultaneously.
        let mut ds = Dataset::new(
            vec![
                vec![0.0, 1.0, 0.0, 1.0], // binary
                vec![0.0, 1.0, 2.0, 0.0], // 3-cat
                vec![5.0, 6.0, 7.0, 8.0], // numeric (passthrough)
            ],
            vec![0.0; 4],
            vec!["a".into(), "b".into(), "num".into()],
            "y",
        );
        let mut enc = OneHotEncoder::new(vec![0, 1]);
        enc.fit_transform(&mut ds).unwrap();

        // 2 one-hot from "a" + 3 one-hot from "b" + 1 passthrough = 6.
        assert_eq!(ds.n_features(), 6);
        assert_eq!(ds.feature_names[0], "a_0");
        assert_eq!(ds.feature_names[1], "a_1");
        assert_eq!(ds.feature_names[2], "b_0");
        assert_eq!(ds.feature_names[3], "b_1");
        assert_eq!(ds.feature_names[4], "b_2");
        assert_eq!(ds.feature_names[5], "num");
    }
}
