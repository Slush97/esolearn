// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gaussian Naive Bayes classifier.

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::partial_fit::PartialFit;
use crate::sparse::{CscMatrix, CsrMatrix};
use crate::weights::{compute_sample_weights, ClassWeight};

/// Gaussian Naive Bayes — assumes features follow a normal distribution per class.
///
/// Fast to train (single pass), good baseline classifier.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct GaussianNb {
    /// Per-class means: `means[class][feature]`.
    means: Vec<Vec<f64>>,
    /// Per-class variances: `variances[class][feature]`.
    variances: Vec<Vec<f64>>,
    /// Prior probabilities per class.
    priors: Vec<f64>,
    class_weight: ClassWeight,
    /// Additive smoothing: `var_smoothing * max(all_variances)` is added to each
    /// feature variance, matching scikit-learn's behaviour.
    var_smoothing: f64,
    n_classes: usize,
    fitted: bool,
    // Incremental learning state (sufficient statistics for partial_fit).
    partial_count: Vec<f64>,
    partial_sum: Vec<Vec<f64>>,
    partial_sum_sq: Vec<Vec<f64>>,
    n_features_partial: usize,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl GaussianNb {
    /// Create a new Gaussian Naive Bayes classifier.
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            variances: Vec::new(),
            priors: Vec::new(),
            class_weight: ClassWeight::Uniform,
            var_smoothing: 1e-9,
            n_classes: 0,
            fitted: false,
            partial_count: Vec::new(),
            partial_sum: Vec::new(),
            partial_sum_sq: Vec::new(),
            n_features_partial: 0,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Set variance smoothing parameter (default `1e-9`, matching sklearn).
    ///
    /// The smoothing epsilon added to each feature variance is
    /// `var_smoothing × max(all_variances)`, which adapts to the scale
    /// of the data — important for high-dimensional datasets.
    pub fn var_smoothing(mut self, vs: f64) -> Self {
        self.var_smoothing = vs;
        self
    }

    /// Train the model (single-pass computation of per-class mean/variance).
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        if let Some(csc) = data.sparse_csc() {
            return self.fit_sparse(csc, &data.target);
        }
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_classes = data.n_classes();
        self.means = vec![vec![0.0; m]; self.n_classes];
        self.variances = vec![vec![0.0; m]; self.n_classes];

        let mat = data.matrix();
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);
        let mut w_sums = vec![0.0_f64; self.n_classes];

        // Compute weighted means.
        for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate() {
            let c = target_val as usize;
            if c >= self.n_classes {
                continue;
            }
            w_sums[c] += sw;
            for j in 0..m {
                self.means[c][j] += sw * mat.get(i, j);
            }
        }
        for (c_means, &ws) in self.means.iter_mut().zip(w_sums.iter()) {
            if ws > 0.0 {
                for mj in c_means.iter_mut() {
                    *mj /= ws;
                }
            }
        }

        // Compute weighted variances.
        for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate() {
            let c = target_val as usize;
            if c >= self.n_classes {
                continue;
            }
            for j in 0..m {
                let diff = mat.get(i, j) - self.means[c][j];
                self.variances[c][j] += sw * diff * diff;
            }
        }
        for (c_var, &ws) in self.variances.iter_mut().zip(w_sums.iter()) {
            if ws > 0.0 {
                for vj in c_var.iter_mut() {
                    *vj /= ws;
                }
            }
        }

        // Compute max variance across all classes and features (sklearn-style).
        let max_var = self
            .variances
            .iter()
            .flat_map(|cv| cv.iter())
            .copied()
            .fold(0.0_f64, f64::max);
        let epsilon = self.var_smoothing * max_var.max(1e-12);

        // Add scaled smoothing to all variances.
        for c_var in &mut self.variances {
            for vj in c_var.iter_mut() {
                *vj += epsilon;
            }
        }

        // Weighted priors.
        let w_total: f64 = w_sums.iter().sum();
        self.priors = w_sums.iter().map(|&ws| ws / w_total).collect();
        self.fitted = true;
        Ok(())
    }

    /// Per-class means: `class_means()[c][j]` is the mean of feature `j` for class `c`.
    pub fn class_means(&self) -> &[Vec<f64>] {
        &self.means
    }

    /// Per-class variances (smoothed): `class_variances()[c][j]` for class `c`, feature `j`.
    pub fn class_variances(&self) -> &[Vec<f64>] {
        &self.variances
    }

    /// Prior probabilities per class.
    pub fn class_priors(&self) -> &[f64] {
        &self.priors
    }

    /// Fit on sparse features (CSC format).
    ///
    /// Computes per-class mean/variance from sparse columns, correctly
    /// accounting for zero entries.
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
        self.means = vec![vec![0.0; m]; self.n_classes];
        self.variances = vec![vec![0.0; m]; self.n_classes];

        let sample_weights = compute_sample_weights(target, &self.class_weight);
        let mut w_sums = vec![0.0_f64; self.n_classes];

        // Weighted class totals.
        for (i, &t) in target.iter().enumerate() {
            let c = t as usize;
            if c < self.n_classes {
                w_sums[c] += sample_weights[i];
            }
        }

        // Compute weighted means per column: mean[c][j] = Σ sw_i * x_ij / w_sums[c]
        // Only iterate non-zero entries; zeros contribute 0 to the sum.
        for j in 0..m {
            for (row_idx, val) in features.col(j).iter() {
                let c = target[row_idx] as usize;
                if c < self.n_classes {
                    self.means[c][j] += sample_weights[row_idx] * val;
                }
            }
        }
        for (c_means, &ws) in self.means.iter_mut().zip(w_sums.iter()) {
            if ws > 0.0 {
                for mj in c_means.iter_mut() {
                    *mj /= ws;
                }
            }
        }

        // Compute weighted variances: var[c][j] = Σ sw_i * (x_ij - mean[c][j])² / w_sums[c]
        // Non-zero entries contribute (val - mean)², zero entries contribute mean².
        // First handle zero entries: each class c has w_sums[c] total weight.
        // Weight of zero entries for class c, feature j = w_sums[c] - Σ_{nnz in class c} sw_i.
        let mut nnz_weight_per_class_feat = vec![vec![0.0; m]; self.n_classes];

        for j in 0..m {
            for (row_idx, val) in features.col(j).iter() {
                let c = target[row_idx] as usize;
                if c < self.n_classes {
                    let sw = sample_weights[row_idx];
                    let diff = val - self.means[c][j];
                    self.variances[c][j] += sw * diff * diff;
                    nnz_weight_per_class_feat[c][j] += sw;
                }
            }
        }
        // Add contribution of zero entries: (0 - mean)² * weight_of_zeros.
        for c in 0..self.n_classes {
            for j in 0..m {
                let zero_weight = w_sums[c] - nnz_weight_per_class_feat[c][j];
                if zero_weight > 0.0 {
                    self.variances[c][j] += zero_weight * self.means[c][j] * self.means[c][j];
                }
            }
        }
        for (c_var, &ws) in self.variances.iter_mut().zip(w_sums.iter()) {
            if ws > 0.0 {
                for vj in c_var.iter_mut() {
                    *vj /= ws;
                }
            }
        }

        // Variance smoothing.
        let max_var = self
            .variances
            .iter()
            .flat_map(|cv| cv.iter())
            .copied()
            .fold(0.0_f64, f64::max);
        let epsilon = self.var_smoothing * max_var.max(1e-12);
        for c_var in &mut self.variances {
            for vj in c_var.iter_mut() {
                *vj += epsilon;
            }
        }

        // Weighted priors.
        let w_total: f64 = w_sums.iter().sum();
        self.priors = w_sums.iter().map(|&ws| ws / w_total).collect();
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
    #[allow(clippy::needless_range_loop)]
    pub fn predict_proba_sparse(&self, features: &CsrMatrix) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n_feat = self.means[0].len();

        Ok((0..features.n_rows())
            .map(|i| {
                let row = features.row(i);

                // Precompute log-likelihood of zero features per class.
                // For feature j with x=0: -0.5 * (mean²/var + ln(var) + ln(2π))
                let mut log_probs: Vec<f64> = (0..self.n_classes)
                    .map(|c| {
                        let mut log_p = self.priors[c].ln();
                        // Start with all features being zero.
                        for j in 0..n_feat {
                            let mean = self.means[c][j];
                            let var = self.variances[c][j];
                            log_p +=
                                -0.5 * (mean * mean / var + var.ln() + std::f64::consts::TAU.ln());
                        }
                        log_p
                    })
                    .collect();

                // Correct for non-zero features: subtract zero contribution, add actual.
                for (col, val) in row.iter() {
                    if col >= n_feat {
                        continue;
                    }
                    for c in 0..self.n_classes {
                        let mean = self.means[c][col];
                        let var = self.variances[c][col];
                        // Remove zero contribution.
                        log_probs[c] -=
                            -0.5 * (mean * mean / var + var.ln() + std::f64::consts::TAU.ln());
                        // Add actual contribution.
                        log_probs[c] += -0.5
                            * ((val - mean).powi(2) / var + var.ln() + std::f64::consts::TAU.ln());
                    }
                }

                // Log-sum-exp normalization.
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

    /// Predict log-probabilities and return normalized probabilities.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        Ok(features
            .iter()
            .map(|row| {
                let mut log_probs: Vec<f64> = (0..self.n_classes)
                    .map(|c| {
                        let mut log_p = self.priors[c].ln();
                        for (j, &x) in row.iter().enumerate() {
                            if j < self.means[c].len() {
                                let mean = self.means[c][j];
                                let var = self.variances[c][j];
                                // Gaussian log-likelihood.
                                log_p += -0.5
                                    * ((x - mean).powi(2) / var
                                        + var.ln()
                                        + std::f64::consts::TAU.ln());
                            }
                        }
                        log_p
                    })
                    .collect();

                // Log-sum-exp trick for numerical stability.
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

impl Default for GaussianNb {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialFit for GaussianNb {
    /// Accumulate sufficient statistics from a batch and recompute model parameters.
    ///
    /// Uses running weighted sums (sum, sum-of-squares, count) to incrementally
    /// update per-class means and variances. Equivalent to `fit()` on all data
    /// seen so far when class weights are uniform.
    fn partial_fit(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            if self.is_initialized() {
                return Ok(());
            }
            return Err(ScryLearnError::EmptyDataset);
        }

        if !self.is_initialized() {
            self.n_classes = data.n_classes();
            self.n_features_partial = m;
            self.partial_count = vec![0.0; self.n_classes];
            self.partial_sum = vec![vec![0.0; m]; self.n_classes];
            self.partial_sum_sq = vec![vec![0.0; m]; self.n_classes];
        } else if m != self.n_features_partial {
            return Err(ScryLearnError::ShapeMismatch {
                expected: self.n_features_partial,
                got: m,
            });
        }

        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);
        let mat = data.matrix();

        // Accumulate sufficient statistics.
        for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate() {
            let c = target_val as usize;
            // Grow class tracking if a new class appears in a later batch.
            if c >= self.n_classes {
                let new_n = c + 1;
                self.partial_count.resize(new_n, 0.0);
                self.partial_sum.resize(new_n, vec![0.0; m]);
                self.partial_sum_sq.resize(new_n, vec![0.0; m]);
                self.n_classes = new_n;
            }
            self.partial_count[c] += sw;
            for j in 0..m {
                let x = mat.get(i, j);
                self.partial_sum[c][j] += sw * x;
                self.partial_sum_sq[c][j] += sw * x * x;
            }
        }

        // Recompute means and variances from accumulated stats.
        self.means = vec![vec![0.0; m]; self.n_classes];
        self.variances = vec![vec![0.0; m]; self.n_classes];

        for c in 0..self.n_classes {
            if self.partial_count[c] > 0.0 {
                let cnt = self.partial_count[c];
                for j in 0..m {
                    let mean = self.partial_sum[c][j] / cnt;
                    let var = (self.partial_sum_sq[c][j] / cnt - mean * mean).max(0.0);
                    self.means[c][j] = mean;
                    self.variances[c][j] = var;
                }
            }
        }

        // Variance smoothing (matching sklearn).
        let max_var = self
            .variances
            .iter()
            .flat_map(|cv| cv.iter())
            .copied()
            .fold(0.0_f64, f64::max);
        let epsilon = self.var_smoothing * max_var.max(1e-12);
        for c_var in &mut self.variances {
            for vj in c_var.iter_mut() {
                *vj += epsilon;
            }
        }

        // Priors from accumulated class counts.
        let w_total: f64 = self.partial_count.iter().sum();
        self.priors = self.partial_count.iter().map(|&c| c / w_total).collect();
        self.fitted = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        !self.partial_count.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_nb_simple() {
        // Class 0: low values, Class 1: high values.
        let features = vec![
            vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0],
            vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut nb = GaussianNb::new();
        nb.fit(&data).unwrap();

        let preds = nb.predict(&[vec![1.0, 1.0], vec![9.0, 9.0]]).unwrap();
        assert!((preds[0] - 0.0).abs() < 1e-6);
        assert!((preds[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_gaussian_nb_matches_dense() {
        let features = vec![
            vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0],
            vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(
            features.clone(),
            target.clone(),
            vec!["x".into(), "y".into()],
            "class",
        );

        let mut nb_dense = GaussianNb::new();
        nb_dense.fit(&data).unwrap();

        let csc = CscMatrix::from_dense(&features);
        let mut nb_sparse = GaussianNb::new();
        nb_sparse.fit_sparse(&csc, &target).unwrap();

        // Means should match.
        for c in 0..2 {
            for j in 0..2 {
                assert!(
                    (nb_dense.class_means()[c][j] - nb_sparse.class_means()[c][j]).abs() < 1e-6,
                    "Mean mismatch: class={c} feat={j}"
                );
            }
        }

        let test = vec![vec![1.0, 1.0], vec![9.0, 9.0]];
        let preds_dense = nb_dense.predict(&test).unwrap();
        let csr = CsrMatrix::from_dense(&test);
        let preds_sparse = nb_sparse.predict_sparse(&csr).unwrap();

        for (d, s) in preds_dense.iter().zip(preds_sparse.iter()) {
            assert!((d - s).abs() < 1e-6, "Dense={d} vs Sparse={s}");
        }
    }

    #[test]
    fn test_partial_fit_is_initialized() {
        let mut nb = GaussianNb::new();
        assert!(!nb.is_initialized());

        let features = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        let target = vec![0.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");
        nb.partial_fit(&data).unwrap();
        assert!(nb.is_initialized());
    }

    #[test]
    fn test_partial_fit_3_batches_matches_fit() {
        // Build full dataset: 3 samples per class, 2 features.
        let features = vec![
            vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0],
            vec![1.0, 1.5, 2.0, 8.0, 8.5, 9.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let all_data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        // Fit on all data at once.
        let mut nb_full = GaussianNb::new();
        nb_full.fit(&all_data).unwrap();

        // partial_fit in 3 batches of 2 samples each.
        let mut nb_partial = GaussianNb::new();
        for i in 0..3 {
            let s = i * 2;
            let batch = Dataset::new(
                vec![
                    vec![all_data.features[0][s], all_data.features[0][s + 1]],
                    vec![all_data.features[1][s], all_data.features[1][s + 1]],
                ],
                vec![all_data.target[s], all_data.target[s + 1]],
                vec!["x".into(), "y".into()],
                "class",
            );
            nb_partial.partial_fit(&batch).unwrap();
        }

        // Means should match closely.
        for c in 0..2 {
            for j in 0..2 {
                assert!(
                    (nb_full.class_means()[c][j] - nb_partial.class_means()[c][j]).abs() < 1e-10,
                    "Mean mismatch: class={c}, feat={j}: {} vs {}",
                    nb_full.class_means()[c][j],
                    nb_partial.class_means()[c][j],
                );
            }
        }

        // Predictions should match.
        let test = vec![vec![1.0, 1.0], vec![9.0, 9.0]];
        let preds_full = nb_full.predict(&test).unwrap();
        let preds_partial = nb_partial.predict(&test).unwrap();
        assert_eq!(preds_full, preds_partial);
    }

    #[test]
    fn test_partial_fit_classifies_correctly() {
        let mut nb = GaussianNb::new();

        // Batch 1: class 0 samples
        let b1 = Dataset::new(
            vec![vec![1.0, 1.5, 2.0], vec![1.0, 1.5, 2.0]],
            vec![0.0, 0.0, 0.0],
            vec!["x".into(), "y".into()],
            "class",
        );
        // Batch 2: class 1 samples
        let b2 = Dataset::new(
            vec![vec![8.0, 8.5, 9.0], vec![8.0, 8.5, 9.0]],
            vec![1.0, 1.0, 1.0],
            vec!["x".into(), "y".into()],
            "class",
        );

        nb.partial_fit(&b1).unwrap();
        nb.partial_fit(&b2).unwrap();

        let preds = nb.predict(&[vec![1.0, 1.0], vec![9.0, 9.0]]).unwrap();
        assert!((preds[0] - 0.0).abs() < 1e-6, "x=1 should be class 0");
        assert!((preds[1] - 1.0).abs() < 1e-6, "x=9 should be class 1");
    }

    #[test]
    fn test_zero_variance_features() {
        // All features constant across all classes — max_var == 0.
        // With the old floor (1e-300), epsilon ≈ 1e-309 (subnormal territory).
        // With max(1e-12), epsilon = 1e-21 — small but well within normal range.
        let features = vec![
            vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut nb = GaussianNb::new();
        nb.fit(&data).unwrap();

        // Smoothed variances must be non-zero and not subnormal.
        for c_var in nb.class_variances() {
            for &v in c_var {
                assert!(v > 1e-300, "variance {v} is subnormal — smoothing failed");
                assert!(v.is_normal(), "variance must be a normal float");
            }
        }

        // Should predict without NaN/Inf issues.
        let preds = nb.predict(&[vec![5.0, 5.0]]).unwrap();
        assert!(preds[0].is_finite(), "prediction should be finite");

        // Predict on a point away from the constant feature value.
        let preds2 = nb.predict(&[vec![10.0, 10.0]]).unwrap();
        assert!(preds2[0].is_finite(), "off-center prediction should be finite");
    }

    #[test]
    fn test_auto_dispatch_sparse_fit() {
        use crate::sparse::CscMatrix;
        // Create sparse Dataset, call fit() (not fit_sparse).
        let features = vec![vec![1.0, 2.0, 8.0, 9.0], vec![1.0, 2.0, 8.0, 9.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let csc = CscMatrix::from_dense(&features);
        let data = crate::dataset::Dataset::from_sparse(
            csc,
            target,
            vec!["x".into(), "y".into()],
            "class",
        );

        let mut nb = GaussianNb::new();
        nb.fit(&data).unwrap();

        let preds = nb.predict(&[vec![1.5, 1.5], vec![8.5, 8.5]]).unwrap();
        assert!(
            (preds[0] - 0.0).abs() < 1e-6,
            "Auto-dispatch sparse: x=1.5 should be class 0"
        );
        assert!(
            (preds[1] - 1.0).abs() < 1e-6,
            "Auto-dispatch sparse: x=8.5 should be class 1"
        );
    }
}
