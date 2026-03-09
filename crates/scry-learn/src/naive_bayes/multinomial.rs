// SPDX-License-Identifier: MIT OR Apache-2.0
//! Multinomial Naive Bayes classifier for count/frequency features.
//!
//! Suited for text classification with bag-of-words count vectors,
//! TF-IDF features, or any non-negative count data.
//!
//! # Example
//!
//! ```
//! use scry_learn::naive_bayes::MultinomialNB;
//! use scry_learn::dataset::Dataset;
//!
//! // Simulated word counts: [word_a_count, word_b_count].
//! let data = Dataset::new(
//!     vec![vec![5.0, 5.0, 0.0, 0.0], vec![0.0, 0.0, 5.0, 5.0]],
//!     vec![0.0, 0.0, 1.0, 1.0],
//!     vec!["word_a".into(), "word_b".into()],
//!     "category",
//! );
//!
//! let mut nb = MultinomialNB::new();
//! nb.fit(&data).unwrap();
//! let preds = nb.predict(&[vec![4.0, 0.0]]).unwrap();
//! assert!((preds[0] - 0.0).abs() < 1e-6);
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::sparse::{CscMatrix, CsrMatrix};
use crate::weights::{compute_sample_weights, ClassWeight};

/// Multinomial Naive Bayes — for count/frequency features.
///
/// Models each class as a multinomial distribution over features.
/// Well-suited for document classification with term frequencies.
///
/// Uses Laplace smoothing (additive smoothing) to handle zero counts.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MultinomialNB {
    /// Laplace smoothing parameter.
    alpha: f64,
    /// Class weighting strategy.
    class_weight: ClassWeight,
    /// Log-probabilities of features given class: `log_probs[class][feature]`.
    log_probs: Vec<Vec<f64>>,
    /// Log prior probabilities per class.
    log_priors: Vec<f64>,
    n_classes: usize,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl MultinomialNB {
    /// Create a new Multinomial Naive Bayes classifier.
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            class_weight: ClassWeight::Uniform,
            log_probs: Vec::new(),
            log_priors: Vec::new(),
            n_classes: 0,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set Laplace smoothing parameter (default 1.0).
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Train the model on a dataset.
    ///
    /// Features should be non-negative counts or frequencies.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_classes = data.n_classes();
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);

        // Compute weighted feature sums per class.
        let mut feature_sum = vec![vec![0.0_f64; m]; self.n_classes];
        let mut class_weight_sum = vec![0.0_f64; self.n_classes];

        for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate() {
            let c = target_val as usize;
            if c >= self.n_classes {
                continue;
            }
            class_weight_sum[c] += sw;
            for (j, feat_col) in data.features.iter().enumerate() {
                feature_sum[c][j] += sw * feat_col[i];
            }
        }

        // Compute smoothed log-probabilities.
        // P(x_j | c) = (sum_jc + alpha) / (sum_c + n_features * alpha)
        self.log_probs = vec![vec![0.0; m]; self.n_classes];
        for (c_probs, c_sums) in self.log_probs.iter_mut().zip(feature_sum.iter()) {
            let total: f64 = c_sums.iter().sum::<f64>() + self.alpha * m as f64;
            for (lp, &fs) in c_probs.iter_mut().zip(c_sums.iter()) {
                *lp = ((fs + self.alpha) / total).ln();
            }
        }

        // Log priors.
        let total_weight: f64 = class_weight_sum.iter().sum();
        self.log_priors = class_weight_sum
            .iter()
            .map(|&w| (w / total_weight).ln())
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Fit on sparse features (CSC format) — perfect for TF-IDF.
    ///
    /// Sums non-zero entries per class per feature for count-based likelihood.
    #[allow(clippy::needless_range_loop)]
    pub fn fit_sparse(&mut self, features: &CscMatrix, target: &[f64]) -> Result<()> {
        let n = features.n_rows();
        let m = features.n_cols();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if target.len() != n {
            return Err(ScryLearnError::InvalidParameter(format!(
                "target length {} != n_rows {}",
                target.len(),
                n
            )));
        }

        let max_class = target.iter().map(|&t| t as usize).max().unwrap_or(0);
        self.n_classes = max_class + 1;
        let sample_weights = compute_sample_weights(target, &self.class_weight);

        let mut feature_sum = vec![vec![0.0_f64; m]; self.n_classes];
        let mut class_weight_sum = vec![0.0_f64; self.n_classes];

        for (&sw, &t) in sample_weights.iter().zip(target.iter()) {
            let c = t as usize;
            if c < self.n_classes {
                class_weight_sum[c] += sw;
            }
        }

        // Sum features per class using sparse column iteration.
        for j in 0..m {
            for (row_idx, val) in features.col(j).iter() {
                let c = target[row_idx] as usize;
                if c < self.n_classes {
                    feature_sum[c][j] += sample_weights[row_idx] * val;
                }
            }
        }

        // Smoothed log-probabilities.
        self.log_probs = vec![vec![0.0; m]; self.n_classes];
        for (c_probs, c_sums) in self.log_probs.iter_mut().zip(feature_sum.iter()) {
            let total: f64 = c_sums.iter().sum::<f64>() + self.alpha * m as f64;
            for (lp, &fs) in c_probs.iter_mut().zip(c_sums.iter()) {
                *lp = ((fs + self.alpha) / total).ln();
            }
        }

        let total_weight: f64 = class_weight_sum.iter().sum();
        self.log_priors = class_weight_sum
            .iter()
            .map(|&w| (w / total_weight).ln())
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Predict from sparse features (CSR format).
    pub fn predict_sparse(&self, features: &CsrMatrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let probas = self.predict_proba_sparse(features)?;
        Ok(probas
            .iter()
            .map(|probs| {
                probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0.0, |(idx, _)| idx as f64)
            })
            .collect())
    }

    /// Predict probabilities from sparse features (CSR format).
    ///
    /// Only accumulates log-probability for non-zero features (zero × log_prob = 0).
    pub fn predict_proba_sparse(&self, features: &CsrMatrix) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok((0..features.n_rows())
            .map(|i| {
                let row = features.row(i);
                let mut log_probs: Vec<f64> = (0..self.n_classes)
                    .map(|c| {
                        let mut lp = self.log_priors[c];
                        // Only non-zero features contribute: x_j * log P(x_j | c).
                        for (col, val) in row.iter() {
                            if col < self.log_probs[c].len() {
                                lp += val * self.log_probs[c][col];
                            }
                        }
                        lp
                    })
                    .collect();

                let max_log = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
                for lp in &mut log_probs {
                    *lp = ((*lp - max_log).exp()) / sum;
                }
                log_probs
            })
            .collect())
    }

    /// Predict class labels.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let probas = self.predict_proba(features)?;
        Ok(probas
            .iter()
            .map(|probs| {
                probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0.0, |(idx, _)| idx as f64)
            })
            .collect())
    }

    /// Predict normalized probabilities for each class.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        Ok(features
            .iter()
            .map(|row| {
                let mut log_probs: Vec<f64> = (0..self.n_classes)
                    .map(|c| {
                        let mut lp = self.log_priors[c];
                        for (j, &x) in row.iter().enumerate() {
                            if j >= self.log_probs[c].len() {
                                continue;
                            }
                            // Multinomial likelihood: x_j * log P(x_j | c).
                            lp += x * self.log_probs[c][j];
                        }
                        lp
                    })
                    .collect();

                // Log-sum-exp for numerical stability.
                let max_log = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = log_probs.iter().map(|&lp| (lp - max_log).exp()).sum();
                for lp in &mut log_probs {
                    *lp = ((*lp - max_log).exp()) / sum;
                }
                log_probs
            })
            .collect())
    }
}

impl Default for MultinomialNB {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multinomial_nb_counts() {
        // Class 0: high word_a counts, Class 1: high word_b counts.
        let features = vec![
            vec![5.0, 6.0, 4.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 5.0, 6.0, 4.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(
            features,
            target,
            vec!["word_a".into(), "word_b".into()],
            "class",
        );

        let mut nb = MultinomialNB::new();
        nb.fit(&data).unwrap();

        let preds = nb.predict(&[vec![4.0, 0.0], vec![0.0, 5.0]]).unwrap();
        assert!((preds[0] - 0.0).abs() < 1e-6, "high word_a → class 0");
        assert!((preds[1] - 1.0).abs() < 1e-6, "high word_b → class 1");
    }

    #[test]
    fn test_multinomial_nb_predict_proba() {
        let features = vec![vec![5.0, 5.0, 0.0, 0.0], vec![0.0, 0.0, 5.0, 5.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["f0".into(), "f1".into()], "class");

        let mut nb = MultinomialNB::new();
        nb.fit(&data).unwrap();

        let probas = nb.predict_proba(&[vec![4.0, 0.0]]).unwrap();
        assert_eq!(probas[0].len(), 2);
        let sum: f64 = probas[0].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "probabilities must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_sparse_multinomial_nb_matches_dense() {
        let features = vec![
            vec![5.0, 6.0, 4.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 5.0, 6.0, 4.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(
            features.clone(),
            target.clone(),
            vec!["w_a".into(), "w_b".into()],
            "class",
        );

        let mut nb_dense = MultinomialNB::new();
        nb_dense.fit(&data).unwrap();

        let csc = CscMatrix::from_dense(&features);
        let mut nb_sparse = MultinomialNB::new();
        nb_sparse.fit_sparse(&csc, &target).unwrap();

        let test = vec![vec![4.0, 0.0], vec![0.0, 5.0]];
        let preds_dense = nb_dense.predict(&test).unwrap();
        let csr = CsrMatrix::from_dense(&test);
        let preds_sparse = nb_sparse.predict_sparse(&csr).unwrap();

        for (d, s) in preds_dense.iter().zip(preds_sparse.iter()) {
            assert!((d - s).abs() < 1e-6, "Dense={d} vs Sparse={s}");
        }
    }
}
