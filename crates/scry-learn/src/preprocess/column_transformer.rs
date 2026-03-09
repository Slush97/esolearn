// SPDX-License-Identifier: MIT OR Apache-2.0
//! Column-based transformer composition.
//!
//! [`ColumnTransformer`] applies different transformers to different subsets
//! of feature columns and concatenates the results.
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::preprocess::{ColumnTransformer, StandardScaler, MinMaxScaler};
//!
//! let ct = ColumnTransformer::new()
//!     .add(&[0, 1], StandardScaler::new())
//!     .add(&[2, 3], MinMaxScaler::new());
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::preprocess::Transformer;

/// Internal trait-object wrapper so we can store heterogeneous transformers.
trait BoxedTransformer {
    fn fit(&mut self, data: &Dataset) -> Result<()>;
    fn transform(&self, data: &mut Dataset) -> Result<()>;
}

impl<T: Transformer> BoxedTransformer for T {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        Transformer::fit(self, data)
    }
    fn transform(&self, data: &mut Dataset) -> Result<()> {
        Transformer::transform(self, data)
    }
}

/// A step within the column transformer: column indices + transformer.
struct TransformerStep {
    columns: Vec<usize>,
    transformer: Box<dyn BoxedTransformer>,
}

/// Apply different transformers to different column subsets, then
/// concatenate all transformed outputs.
///
/// # Builder API
///
/// ```ignore
/// let ct = ColumnTransformer::new()
///     .add(&[0, 1], StandardScaler::new())
///     .add(&[2, 3], MinMaxScaler::new());
/// ct.fit_transform(&mut ds)?;
/// ```
#[non_exhaustive]
pub struct ColumnTransformer {
    steps: Vec<TransformerStep>,
    fitted: bool,
}

impl ColumnTransformer {
    /// Create an empty column transformer.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            fitted: false,
        }
    }

    /// Add a transformer to be applied to the given column indices.
    pub fn add<T: Transformer + 'static>(mut self, columns: &[usize], transformer: T) -> Self {
        self.steps.push(TransformerStep {
            columns: columns.to_vec(),
            transformer: Box::new(transformer),
        });
        self
    }
}

impl Default for ColumnTransformer {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract a sub-dataset containing only the specified feature columns.
fn extract_columns(data: &Dataset, cols: &[usize]) -> Dataset {
    let features: Vec<Vec<f64>> = cols.iter().map(|&c| data.features[c].clone()).collect();
    let names: Vec<String> = cols
        .iter()
        .map(|&c| data.feature_names[c].clone())
        .collect();
    Dataset::new(features, data.target.clone(), names, &data.target_name)
}

impl Transformer for ColumnTransformer {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        if data.n_samples() == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        for step in &mut self.steps {
            // Validate column indices.
            for &c in &step.columns {
                if c >= data.n_features() {
                    return Err(ScryLearnError::InvalidColumn(format!(
                        "column index {c} out of range (dataset has {} features)",
                        data.n_features()
                    )));
                }
            }
            let sub = extract_columns(data, &step.columns);
            step.transformer.fit(&sub)?;
        }
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        // Transform each column subset independently, collect results.
        let mut result_cols: Vec<Vec<f64>> = Vec::new();
        let mut result_names: Vec<String> = Vec::new();

        for step in &self.steps {
            let mut sub = extract_columns(data, &step.columns);
            step.transformer.transform(&mut sub)?;
            for (col, name) in sub.features.into_iter().zip(sub.feature_names) {
                result_cols.push(col);
                result_names.push(name);
            }
        }

        // Replace the dataset's features with the concatenated result.
        data.features = result_cols;
        data.feature_names = result_names;
        data.sync_matrix();

        Ok(())
    }

    fn inverse_transform(&self, _data: &mut Dataset) -> Result<()> {
        Err(ScryLearnError::InvalidParameter(
            "ColumnTransformer is not invertible".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::{MinMaxScaler, StandardScaler};

    #[test]
    fn test_column_transformer_basic() {
        // 4 features, apply StandardScaler to [0,1], MinMaxScaler to [2,3]
        let mut ds = Dataset::new(
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],           // col 0
                vec![10.0, 20.0, 30.0, 40.0, 50.0],      // col 1
                vec![100.0, 200.0, 300.0, 400.0, 500.0], // col 2
                vec![5.0, 10.0, 15.0, 20.0, 25.0],       // col 3
            ],
            vec![0.0; 5],
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            "y",
        );

        let mut ct = ColumnTransformer::new()
            .add(&[0, 1], StandardScaler::new())
            .add(&[2, 3], MinMaxScaler::new());

        ct.fit_transform(&mut ds).unwrap();

        assert_eq!(ds.n_features(), 4);

        // StandardScaler'd columns: mean ≈ 0
        let mean_a: f64 = ds.features[0].iter().sum::<f64>() / 5.0;
        assert!(
            mean_a.abs() < 1e-10,
            "col 0 should be zero-mean, got {mean_a}"
        );

        let mean_b: f64 = ds.features[1].iter().sum::<f64>() / 5.0;
        assert!(
            mean_b.abs() < 1e-10,
            "col 1 should be zero-mean, got {mean_b}"
        );

        // MinMaxScaler'd columns: min=0, max=1
        assert!(ds.features[2][0].abs() < 1e-10, "col 2 min should be 0");
        assert!(
            (ds.features[2][4] - 1.0).abs() < 1e-10,
            "col 2 max should be 1"
        );
        assert!(ds.features[3][0].abs() < 1e-10, "col 3 min should be 0");
        assert!(
            (ds.features[3][4] - 1.0).abs() < 1e-10,
            "col 3 max should be 1"
        );
    }

    #[test]
    fn test_column_transformer_not_fitted() {
        let ct = ColumnTransformer::new().add(&[0], StandardScaler::new());
        let mut ds = Dataset::new(vec![vec![1.0, 2.0]], vec![0.0; 2], vec!["x".into()], "y");
        assert!(Transformer::transform(&ct, &mut ds).is_err());
    }

    #[test]
    fn test_column_transformer_invalid_column() {
        let mut ct = ColumnTransformer::new().add(&[99], StandardScaler::new());
        let ds = Dataset::new(vec![vec![1.0, 2.0]], vec![0.0; 2], vec!["x".into()], "y");
        assert!(Transformer::fit(&mut ct, &ds).is_err());
    }
}
