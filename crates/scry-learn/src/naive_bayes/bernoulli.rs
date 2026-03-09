// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bernoulli Naive Bayes classifier for binary/boolean features.
//!
//! Each feature is modelled as a Bernoulli distribution (present/absent).
//! An optional `binarize` threshold converts continuous features to binary.
//!
//! # Example
//!
//! ```
//! use scry_learn::naive_bayes::BernoulliNB;
//! use scry_learn::dataset::Dataset;
//!
//! // Binary features: feature 0 and 1 are indicators.
//! let data = Dataset::new(
//!     vec![vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 1.0, 1.0, 0.0]],
//!     vec![1.0, 1.0, 0.0, 0.0],
//!     vec!["has_word_a".into(), "has_word_b".into()],
//!     "spam",
//! );
//!
//! let mut nb = BernoulliNB::new();
//! nb.fit(&data).unwrap();
//! let preds = nb.predict(&[vec![1.0, 0.0]]).unwrap();
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::weights::{compute_sample_weights, ClassWeight};

/// Bernoulli Naive Bayes — for binary/boolean features.
///
/// Models each feature as a Bernoulli random variable (present or absent).
/// Suitable for binary feature vectors like bag-of-words with binary indicators.
///
/// When `binarize` is set (default `Some(0.0)`), continuous features are
/// converted to binary using the threshold before fitting/predicting.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct BernoulliNB {
    /// Laplace smoothing parameter.
    alpha: f64,
    /// Optional binarization threshold. `None` means features are already binary.
    binarize: Option<f64>,
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

impl BernoulliNB {
    /// Create a new Bernoulli Naive Bayes classifier.
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            binarize: Some(0.0),
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

    /// Set binarization threshold. Features > threshold become 1, else 0.
    ///
    /// Pass `None` to disable binarization (assumes features are already binary).
    pub fn binarize(mut self, threshold: Option<f64>) -> Self {
        self.binarize = threshold;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Train the model on a dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_classes = data.n_classes();
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);

        // Compute weighted count of feature=1 per class.
        let mut feature_count = vec![vec![0.0_f64; m]; self.n_classes]; // weighted count of 1s
        let mut class_weight_sum = vec![0.0_f64; self.n_classes];

        for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate() {
            let c = target_val as usize;
            if c >= self.n_classes {
                continue;
            }
            class_weight_sum[c] += sw;
            for (j, feat_col) in data.features.iter().enumerate() {
                let val = feat_col[i];
                let binary = self
                    .binarize
                    .map_or(val, |thresh| if val > thresh { 1.0 } else { 0.0 });
                feature_count[c][j] += sw * binary;
            }
        }

        // Compute smoothed log-probabilities.
        // P(x_j=1 | c) = (N_jc + alpha) / (N_c + 2*alpha)
        self.log_probs = vec![vec![0.0; m]; self.n_classes];
        for c in 0..self.n_classes {
            for (lp, &cnt) in self.log_probs[c].iter_mut().zip(feature_count[c].iter()) {
                *lp = (cnt + self.alpha) / (class_weight_sum[c] + 2.0 * self.alpha);
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
                            let binary =
                                self.binarize
                                    .map_or(x, |thresh| if x > thresh { 1.0 } else { 0.0 });
                            let p = self.log_probs[c][j];
                            if binary > 0.5 {
                                lp += p.ln();
                            } else {
                                lp += (1.0 - p).ln();
                            }
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

impl Default for BernoulliNB {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bernoulli_nb_binary() {
        // Class 0: mostly feature 0 active, Class 1: mostly feature 1 active.
        let features = vec![
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["f0".into(), "f1".into()], "class");

        let mut nb = BernoulliNB::new().binarize(Some(0.5));
        nb.fit(&data).unwrap();

        let preds = nb.predict(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        assert!((preds[0] - 0.0).abs() < 1e-6, "should predict class 0");
        assert!((preds[1] - 1.0).abs() < 1e-6, "should predict class 1");
    }

    #[test]
    fn test_bernoulli_nb_binarize() {
        // Continuous features binarized at threshold 0.5.
        let features = vec![
            vec![0.9, 0.8, 0.7, 0.1, 0.2, 0.3],
            vec![0.1, 0.2, 0.3, 0.9, 0.8, 0.7],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["f0".into(), "f1".into()], "class");

        let mut nb = BernoulliNB::new().binarize(Some(0.5));
        nb.fit(&data).unwrap();

        let preds = nb.predict(&[vec![0.8, 0.1], vec![0.1, 0.9]]).unwrap();
        assert!((preds[0] - 0.0).abs() < 1e-6);
        assert!((preds[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bernoulli_nb_predict_proba() {
        let features = vec![vec![1.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 1.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["f0".into(), "f1".into()], "class");

        let mut nb = BernoulliNB::new().binarize(Some(0.5));
        nb.fit(&data).unwrap();

        let probas = nb.predict_proba(&[vec![1.0, 0.0]]).unwrap();
        assert_eq!(probas[0].len(), 2);
        let sum: f64 = probas[0].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "probabilities must sum to 1.0, got {sum}"
        );
    }
}
