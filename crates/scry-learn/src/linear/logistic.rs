// SPDX-License-Identifier: MIT OR Apache-2.0
//! Logistic regression via L-BFGS (default) or gradient descent.
//!
//! Supports configurable [`Penalty`] regularization: `None`, `L1`, `L2` (default),
//! and `ElasticNet(l1_ratio)`. L1 and ElasticNet use proximal gradient descent
//! (soft-thresholding); L-BFGS only supports `L2` and `None`.

use rayon::prelude::*;

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::partial_fit::PartialFit;
use crate::sparse::{CscMatrix, CsrMatrix};
use crate::weights::{compute_sample_weights, ClassWeight};

use super::lbfgs;

/// Regularization penalty for logistic regression.
///
/// Controls the type of regularization applied during training:
/// - `None` — no regularization
/// - `L1` — Lasso penalty (promotes sparsity via proximal gradient descent)
/// - `L2` — Ridge penalty (default, shrinks coefficients)
/// - `ElasticNet(l1_ratio)` — Mix of L1 and L2; `l1_ratio` ∈ \[0, 1\]
///   where 1.0 = pure L1, 0.0 = pure L2
///
/// # Solver compatibility
///
/// | Penalty | GradientDescent | L-BFGS |
/// |---------|:-:|:-:|
/// | `None` | ✓ | ✓ |
/// | `L1` | ✓ | ✗ (error) |
/// | `L2` | ✓ | ✓ |
/// | `ElasticNet` | ✓ | ✗ (error) |
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Penalty {
    /// No regularization.
    None,
    /// L1 (Lasso) penalty — promotes sparse coefficients.
    L1,
    /// L2 (Ridge) penalty — shrinks all coefficients (default).
    #[default]
    L2,
    /// Elastic Net — mix of L1 and L2. The `f64` is the L1 ratio ∈ \[0, 1\].
    ElasticNet(f64),
}

/// Solver algorithm for logistic regression.
///
/// L-BFGS is the default and recommended solver — it converges in ~10-20
/// iterations vs 200+ for gradient descent, matching scikit-learn's default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Solver {
    /// L-BFGS quasi-Newton optimizer (default). Fast, recommended.
    #[default]
    Lbfgs,
    /// Vanilla batch gradient descent. Slower, kept for backward compatibility.
    GradientDescent,
}

/// Logistic regression for binary/multiclass classification.
///
/// Uses L-BFGS (default) or gradient descent with configurable learning rate,
/// iterations, and L2 regularization.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LogisticRegression {
    learning_rate: f64,
    max_iter: usize,
    alpha: f64, // regularization strength
    tolerance: f64,
    class_weight: ClassWeight,
    #[cfg_attr(feature = "serde", serde(default))]
    solver: Solver,
    #[cfg_attr(feature = "serde", serde(default))]
    penalty: Penalty,
    weights: Vec<Vec<f64>>, // [n_classes][n_features + 1] (includes bias)
    n_classes: usize,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl LogisticRegression {
    /// Create a new logistic regression model.
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            max_iter: 1000,
            alpha: 1.0,
            tolerance: crate::constants::STRICT_TOL,
            class_weight: ClassWeight::Uniform,
            solver: Solver::default(),
            penalty: Penalty::default(),
            weights: Vec::new(),
            n_classes: 0,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the learning rate (used by `GradientDescent` solver only).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set regularization strength (equivalent to `1/C` in scikit-learn).
    ///
    /// The meaning depends on the [`Penalty`]:
    /// - `L2` / `L1` — multiplier on the penalty term
    /// - `ElasticNet` — total regularization strength (split by l1_ratio)
    /// - `None` — ignored
    ///
    /// To match scikit-learn's `LogisticRegression(C=x)`, use `alpha(1.0 / x)`.
    /// The default `alpha = 1.0` corresponds to `C = 1.0`.
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set the regularization penalty.
    ///
    /// Default is [`Penalty::L2`]. Use [`Penalty::L1`] for sparse feature selection.
    ///
    /// # Errors
    ///
    /// `L1` and `ElasticNet` are **not** supported with the `Lbfgs` solver — calling
    /// `fit()` will return `Err(InvalidParameter)`. Switch to `GradientDescent`.
    pub fn penalty(mut self, p: Penalty) -> Self {
        self.penalty = p;
        self
    }

    /// Set convergence tolerance.
    pub fn tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }

    /// Alias for [`tolerance`](Self::tolerance) (sklearn convention).
    pub fn tol(self, t: f64) -> Self {
        self.tolerance(t)
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Set the solver algorithm.
    ///
    /// Defaults to `Solver::Lbfgs` which is ~10-20× faster than gradient descent.
    pub fn solver(mut self, s: Solver) -> Self {
        self.solver = s;
        self
    }

    /// Train the model using the configured solver.
    ///
    /// Uses consistent softmax for both training and inference (not one-vs-rest sigmoid).
    ///
    /// # Errors
    ///
    /// Returns `InvalidParameter` if `Penalty::L1` or `Penalty::ElasticNet` is used
    /// with the `Lbfgs` solver (L-BFGS requires a differentiable objective).
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        if let Some(csc) = data.sparse_csc() {
            return self.fit_sparse(csc, &data.target);
        }
        // Classification requires at least 2 distinct classes.
        if data.n_classes() < 2 {
            return Err(ScryLearnError::InvalidParameter(
                "LogisticRegression requires at least 2 distinct classes in the target.".into(),
            ));
        }
        // Validate solver/penalty compatibility.
        if matches!(self.solver, Solver::Lbfgs)
            && matches!(self.penalty, Penalty::L1 | Penalty::ElasticNet(_))
        {
            return Err(ScryLearnError::InvalidParameter(
                "L-BFGS solver does not support L1 or ElasticNet penalties \
                 (non-differentiable). Use Solver::GradientDescent instead."
                    .into(),
            ));
        }
        match self.solver {
            Solver::Lbfgs => self.fit_lbfgs(data),
            Solver::GradientDescent => self.fit_gd(data),
        }
    }

    /// L-BFGS solver: flatten weights, optimize, unflatten.
    ///
    /// Uses vectorized (batch) gradient computation for cache-friendly
    /// column-major access patterns.
    #[allow(clippy::needless_range_loop)]
    fn fit_lbfgs(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_classes = data.n_classes();
        let k = self.n_classes;

        // Binary fast path: use sigmoid instead of softmax (halves parameter count).
        if k == 2 {
            return self.fit_lbfgs_binary(data);
        }

        let dim = m + 1; // features + bias

        // Compute per-sample weights for class imbalance (skip for uniform).
        let uniform = matches!(self.class_weight, ClassWeight::Uniform);
        let sample_weights = if uniform {
            Vec::new()
        } else {
            compute_sample_weights(&data.target, &self.class_weight)
        };

        // Pre-convert targets to usize.
        let target_class: Vec<usize> = data.target.iter().map(|&t| t as usize).collect();

        let alpha = self.alpha;
        let inv_n = 1.0 / n as f64;

        // Flatten initial weights: [class0_bias, class0_w1, ..., class1_bias, ...]
        let total_params = k * dim;
        let mut params = vec![0.0; total_params];

        let config = lbfgs::LbfgsConfig {
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            history_size: 10,
            wolfe: false,
        };

        // Pre-allocate batch buffers — reused every closure call.
        let mut logits = vec![0.0; n * k]; // row-major: logits[i * k + c]
        let mut max_logit = vec![0.0; n];
        let mut sum_exp = vec![0.0; n];
        let use_par = n * m >= crate::constants::LOGREG_PAR_THRESHOLD;
        let mut feature_grad_buf = if use_par { vec![0.0; m * k] } else { Vec::new() };

        lbfgs::minimize(
            &mut params,
            |x, grad| {
                // ── 1. Batch compute logits: logits[i,c] = bias_c + Σ_j w_{c,j} * X_{j,i}
                // Initialize with bias terms.
                for i in 0..n {
                    for c in 0..k {
                        logits[i * k + c] = x[c * dim]; // bias
                    }
                }
                // Accumulate feature contributions column-by-column (cache-friendly).
                for j in 0..m {
                    let feat_col = &data.features[j];
                    for c in 0..k {
                        let w = x[c * dim + j + 1];
                        for i in 0..n {
                            logits[i * k + c] += w * feat_col[i];
                        }
                    }
                }

                // ── 2. Batch softmax + loss computation.
                let mut loss = 0.0;

                // Find max logit per sample (for numerical stability).
                for i in 0..n {
                    let row = &logits[i * k..(i + 1) * k];
                    max_logit[i] = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                }

                // Exponentiate and sum.
                for i in 0..n {
                    let mut se = 0.0;
                    for c in 0..k {
                        let val = (logits[i * k + c] - max_logit[i]).exp();
                        logits[i * k + c] = val; // now stores exp(logit - max)
                        se += val;
                    }
                    sum_exp[i] = se;
                }

                // Cross-entropy loss: sw * (log_sum_exp - logit_tc).
                // logit_tc was overwritten, so reconstruct from max + log(exp_val).
                for i in 0..n {
                    let tc = target_class[i];
                    let log_sum = max_logit[i] + sum_exp[i].ln();
                    let logit_tc_val = max_logit[i] + logits[i * k + tc].ln();
                    let sw = if uniform { 1.0 } else { sample_weights[i] };
                    loss += sw * (log_sum - logit_tc_val);
                }

                // Normalize to probabilities.
                for i in 0..n {
                    let se = sum_exp[i];
                    for c in 0..k {
                        logits[i * k + c] /= se;
                    }
                }

                // ── 3. Batch gradient: grad_{c,j} = Σ_i sw_i * (prob_{i,c} - 1_{tc==c}) * x_{i,j}
                // Zero gradient.
                for g in grad.iter_mut() {
                    *g = 0.0;
                }

                // Compute error = prob - one_hot, weighted by sample weight.
                // Then accumulate bias gradients.
                // We modify logits in-place to store sw * error.
                for i in 0..n {
                    let tc = target_class[i];
                    let sw = if uniform { 1.0 } else { sample_weights[i] };
                    for c in 0..k {
                        let y_i = if tc == c { 1.0 } else { 0.0 };
                        let error = sw * (logits[i * k + c] - y_i);
                        logits[i * k + c] = error; // reuse buffer for weighted errors
                        grad[c * dim] += error; // bias gradient
                    }
                }

                // Accumulate feature gradients column-by-column (cache-friendly).
                if use_par {
                    let errors: &[f64] = &logits;
                    feature_grad_buf.par_chunks_mut(k)
                        .zip(data.features.par_iter())
                        .for_each(|(chunk, feat_col)| {
                            for c in 0..k {
                                let mut acc = 0.0;
                                for i in 0..n { acc += errors[i * k + c] * feat_col[i]; }
                                chunk[c] = acc;
                            }
                        });
                    for j in 0..m {
                        for c in 0..k {
                            grad[c * dim + j + 1] += feature_grad_buf[j * k + c];
                        }
                    }
                } else {
                    for j in 0..m {
                        let feat_col = &data.features[j];
                        for c in 0..k {
                            let grad_idx = c * dim + j + 1;
                            let mut acc = 0.0;
                            for i in 0..n {
                                acc += logits[i * k + c] * feat_col[i];
                            }
                            grad[grad_idx] += acc;
                        }
                    }
                }

                // ── 4. Average over samples + L2 regularization.
                //      sklearn formula: min_w  C * mean(log_loss) + 0.5 * ||w||²
                //      Equivalently:    min_w  mean(log_loss) + 0.5 * (1/C) * ||w||²
                //      Our `alpha` = 1/C, so we scale both loss and penalty by inv_n
                //      to get: mean(log_loss) + 0.5 * alpha * inv_n * ||w||²
                //      This ensures regularization strength scales with dataset size
                //      (matching sklearn's behavior).
                loss *= inv_n;
                for g in grad.iter_mut() {
                    *g *= inv_n;
                }

                if alpha > 0.0 {
                    let reg_scale = alpha * inv_n;
                    for c in 0..k {
                        let base = c * dim;
                        for j in 1..dim {
                            let w = x[base + j];
                            loss += 0.5 * reg_scale * w * w;
                            grad[base + j] += reg_scale * w;
                        }
                    }
                }

                loss
            },
            &config,
        );

        // Unflatten back to [n_classes][dim].
        self.weights = (0..k)
            .map(|c| params[c * dim..(c + 1) * dim].to_vec())
            .collect();

        self.fitted = true;
        Ok(())
    }

    /// Binary L-BFGS fast path: single weight vector with sigmoid.
    ///
    /// For 2-class problems, uses sigmoid(z) instead of softmax over 2 classes,
    /// halving the parameter count and gradient work.
    #[allow(clippy::needless_range_loop)]
    fn fit_lbfgs_binary(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        let dim = m + 1; // features + bias

        let uniform = matches!(self.class_weight, ClassWeight::Uniform);
        let sample_weights = if uniform {
            Vec::new()
        } else {
            compute_sample_weights(&data.target, &self.class_weight)
        };
        let target_bin: Vec<f64> = data.target.iter().map(|&t| if t as usize == 1 { 1.0 } else { 0.0 }).collect();

        let alpha = self.alpha;
        let inv_n = 1.0 / n as f64;

        let mut params = vec![0.0; dim];

        let config = lbfgs::LbfgsConfig {
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            history_size: 10,
            wolfe: false,
        };

        // Pre-allocate buffers reused every closure call.
        let mut prob = vec![0.0; n];
        let use_par = n * m >= crate::constants::LOGREG_PAR_THRESHOLD;

        lbfgs::minimize(
            &mut params,
            |x, grad| {
                // ── 1. Compute z_i = bias + Σ_j w_j * X_{j,i}, then sigmoid.
                for i in 0..n {
                    prob[i] = x[0]; // bias
                }
                for j in 0..m {
                    let w = x[j + 1];
                    let col = &data.features[j];
                    for i in 0..n {
                        prob[i] += w * col[i];
                    }
                }

                // Sigmoid + loss.
                let mut loss = 0.0;
                for i in 0..n {
                    let z = prob[i];
                    // Numerically stable sigmoid and log-loss.
                    let p = if z >= 0.0 {
                        1.0 / (1.0 + (-z).exp())
                    } else {
                        let ez = z.exp();
                        ez / (1.0 + ez)
                    };
                    prob[i] = p;

                    // Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
                    let y = target_bin[i];
                    let log_loss = if z >= 0.0 {
                        (1.0 - y) * z + (-z).exp().ln_1p()
                    } else {
                        -y * z + z.exp().ln_1p()
                    };
                    let sw = if uniform { 1.0 } else { sample_weights[i] };
                    loss += sw * log_loss;
                }

                // ── 2. Gradient: (1/n) * Σ sw_i * (p_i - y_i) * x_i
                for g in grad.iter_mut() {
                    *g = 0.0;
                }

                // Bias gradient.
                let mut bias_grad = 0.0;
                for i in 0..n {
                    let sw = if uniform { 1.0 } else { sample_weights[i] };
                    let err = sw * (prob[i] - target_bin[i]);
                    prob[i] = err; // reuse buffer for weighted errors
                    bias_grad += err;
                }
                grad[0] = bias_grad;

                // Feature gradients column-by-column.
                let errors: &[f64] = &prob;
                if use_par {
                    data.features.par_iter()
                        .zip(grad[1..=m].par_iter_mut())
                        .for_each(|(col, g)| {
                            let mut acc = 0.0;
                            for i in 0..n { acc += errors[i] * col[i]; }
                            *g = acc;
                        });
                } else {
                    for j in 0..m {
                        let col = &data.features[j];
                        let mut acc = 0.0;
                        for i in 0..n { acc += errors[i] * col[i]; }
                        grad[j + 1] = acc;
                    }
                }

                // ── 3. Average + L2 regularization.
                loss *= inv_n;
                for g in grad.iter_mut() {
                    *g *= inv_n;
                }

                if alpha > 0.0 {
                    let reg_scale = alpha * inv_n;
                    for j in 1..dim {
                        let w = x[j];
                        loss += 0.5 * reg_scale * w * w;
                        grad[j] += reg_scale * w;
                    }
                }

                loss
            },
            &config,
        );

        // Unflatten: class 0 = zero weights (reference), class 1 = learned weights.
        // softmax([0, z]) produces the same probabilities as sigmoid(z).
        self.weights = vec![vec![0.0; dim], params];
        self.fitted = true;
        Ok(())
    }

    /// Gradient descent solver (legacy).
    #[allow(clippy::needless_range_loop)]
    fn fit_gd(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        self.n_classes = data.n_classes();
        let dim = m + 1; // features + bias

        // Initialize weights to zero.
        self.weights = vec![vec![0.0; dim]; self.n_classes];

        // Compute per-sample weights for class imbalance.
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);

        // softmax gradient descent.
        let mut probs = vec![0.0; self.n_classes];

        for _epoch in 0..self.max_iter {
            let mut max_grad = 0.0_f64;
            let mut gradient = vec![vec![0.0; dim]; self.n_classes];

            for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate()
            {
                let target_class = target_val as usize;

                // Compute logits for all classes.
                for (c, prob) in probs.iter_mut().enumerate().take(self.n_classes) {
                    let mut z = self.weights[c][0]; // bias
                    for j in 0..m {
                        z += self.weights[c][j + 1] * data.features[j][i];
                    }
                    *prob = z;
                }

                // Softmax.
                let max_s = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for p in &mut probs[..self.n_classes] {
                    *p = (*p - max_s).exp();
                    sum += *p;
                }
                for p in &mut probs[..self.n_classes] {
                    *p /= sum;
                }

                // Gradient: weight_i * (softmax_prob - one_hot_target).
                for (c, (&pc, gc)) in probs
                    .iter()
                    .zip(gradient.iter_mut())
                    .enumerate()
                    .take(self.n_classes)
                {
                    let y_i = if target_class == c { 1.0 } else { 0.0 };
                    let error = sw * (pc - y_i);

                    gc[0] += error; // bias
                    for j in 0..m {
                        gc[j + 1] += error * data.features[j][i];
                    }
                }
            }

            // Compute L2 ratio for the penalty.
            let (l1_ratio, l2_ratio) = match &self.penalty {
                Penalty::None => (0.0, 0.0),
                Penalty::L1 => (1.0, 0.0),
                Penalty::L2 => (0.0, 1.0),
                Penalty::ElasticNet(r) => (*r, 1.0 - *r),
            };

            let inv_n = 1.0 / n as f64;

            // Update weights: gradient step + L2 regularization in gradient.
            for (c_grad, c_w) in gradient
                .iter_mut()
                .zip(self.weights.iter_mut())
                .take(self.n_classes)
            {
                for (j, (g, w)) in c_grad.iter_mut().zip(c_w.iter_mut()).enumerate().take(dim) {
                    *g *= inv_n;
                    if j > 0 {
                        // L2 component goes into the gradient (scaled by inv_n like sklearn).
                        *g += self.alpha * inv_n * l2_ratio * *w;
                    }
                    max_grad = max_grad.max(g.abs());
                    *w -= self.learning_rate * *g;
                }
            }

            // Proximal step for L1 component (soft-thresholding).
            // Applied after the gradient update, only to feature weights (skip bias j=0).
            if l1_ratio > 0.0 {
                let threshold = self.learning_rate * self.alpha * inv_n * l1_ratio;
                for c_w in self.weights.iter_mut().take(self.n_classes) {
                    for w in c_w.iter_mut().skip(1) {
                        let sign = w.signum();
                        *w = sign * (*w * sign - threshold).max(0.0);
                    }
                }
            }

            if max_grad < self.tolerance {
                break;
            }
        }

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

    /// Predict class probabilities.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        Ok(features
            .iter()
            .map(|row| {
                let mut scores: Vec<f64> = self
                    .weights
                    .iter()
                    .map(|w| {
                        let mut z = w[0]; // bias
                        for (j, &x) in row.iter().enumerate() {
                            if j + 1 < w.len() {
                                z += w[j + 1] * x;
                            }
                        }
                        z
                    })
                    .collect();

                // Softmax.
                let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                scores
            })
            .collect())
    }

    /// Fit on sparse features using gradient descent.
    ///
    /// Accepts `CscMatrix` (column-oriented) for efficient gradient computation.
    /// Only supports L2 penalty (or None). Uses gradient descent (not L-BFGS).
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

        // Determine n_classes from target.
        let max_class = target.iter().map(|&t| t as usize).max().unwrap_or(0);
        self.n_classes = max_class + 1;
        if self.n_classes < 2 {
            return Err(ScryLearnError::InvalidParameter(
                "LogisticRegression requires at least 2 distinct classes.".into(),
            ));
        }

        let k = self.n_classes;
        let dim = m + 1;
        let sample_weights = compute_sample_weights(target, &self.class_weight);
        let target_class: Vec<usize> = target.iter().map(|&t| t as usize).collect();

        self.weights = vec![vec![0.0; dim]; k];

        let mut probs = vec![0.0; k];
        let inv_n = 1.0 / n as f64;

        for _epoch in 0..self.max_iter {
            let mut max_grad = 0.0_f64;
            let mut gradient = vec![vec![0.0; dim]; k];

            for i in 0..n {
                let tc = target_class[i];
                let sw = sample_weights[i];

                // Compute logits: bias + sparse dot.
                for c in 0..k {
                    probs[c] = self.weights[c][0]; // bias
                }
                // Accumulate feature contributions from sparse row.
                // We need row access, so iterate all columns and check if row i has an entry.
                // More efficient: convert to CSR, but for fit we iterate columns.
                // Actually, build logits by iterating columns of CSC.
                // But per-sample approach requires iterating all columns for each sample.
                // Better: precompute logits for all samples using column iteration.
                // For simplicity in the per-sample loop, use CSC get which is log(nnz_col).
                for j in 0..m {
                    let xij = features.get(i, j);
                    if xij != 0.0 {
                        for c in 0..k {
                            probs[c] += self.weights[c][j + 1] * xij;
                        }
                    }
                }

                // Softmax.
                let max_s = probs[..k].iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for p in &mut probs[..k] {
                    *p = (*p - max_s).exp();
                    sum += *p;
                }
                for p in &mut probs[..k] {
                    *p /= sum;
                }

                // Gradient.
                for c in 0..k {
                    let y_i = if tc == c { 1.0 } else { 0.0 };
                    let error = sw * (probs[c] - y_i);
                    gradient[c][0] += error; // bias
                    for j in 0..m {
                        let xij = features.get(i, j);
                        if xij != 0.0 {
                            gradient[c][j + 1] += error * xij;
                        }
                    }
                }
            }

            // Update weights.
            for c in 0..k {
                for j in 0..dim {
                    gradient[c][j] *= inv_n;
                    if j > 0 && self.alpha > 0.0 {
                        gradient[c][j] += self.alpha * inv_n * self.weights[c][j];
                    }
                    max_grad = max_grad.max(gradient[c][j].abs());
                    self.weights[c][j] -= self.learning_rate * gradient[c][j];
                }
            }

            if max_grad < self.tolerance {
                break;
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict class labels from sparse features (CSR format).
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

    /// Predict class probabilities from sparse features (CSR format).
    pub fn predict_proba_sparse(&self, features: &CsrMatrix) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok((0..features.n_rows())
            .map(|i| {
                let row = features.row(i);
                let mut scores: Vec<f64> = self
                    .weights
                    .iter()
                    .map(|w| {
                        let mut z = w[0]; // bias
                        for (col, val) in row.iter() {
                            if col + 1 < w.len() {
                                z += w[col + 1] * val;
                            }
                        }
                        z
                    })
                    .collect();

                // Softmax.
                let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }
                scores
            })
            .collect())
    }

    /// Get learned weights (coefficients + bias) for each class.
    pub fn weights(&self) -> &[Vec<f64>] {
        &self.weights
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialFit for LogisticRegression {
    /// Run one pass of gradient descent on the given batch.
    ///
    /// On the first call, initializes weights from the data dimensions and
    /// class count. Subsequent calls preserve weights and continue updating.
    #[allow(clippy::needless_range_loop)]
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
            if data.n_classes() < 2 {
                return Err(ScryLearnError::InvalidParameter(
                    "LogisticRegression requires at least 2 distinct classes.".into(),
                ));
            }
            self.n_classes = data.n_classes();
            let dim = m + 1;
            self.weights = vec![vec![0.0; dim]; self.n_classes];
        }

        let dim = m + 1;
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);

        // Pre-scan for new classes and grow weights if needed.
        let max_class = data.target.iter().map(|&t| t as usize).max().unwrap_or(0);
        if max_class >= self.n_classes {
            let new_n = max_class + 1;
            self.weights.resize(new_n, vec![0.0; dim]);
            self.n_classes = new_n;
        }

        let mut probs = vec![0.0; self.n_classes];
        let mut gradient = vec![vec![0.0; dim]; self.n_classes];

        for (i, (&sw, &target_val)) in sample_weights.iter().zip(data.target.iter()).enumerate() {
            let target_class = target_val as usize;

            // Compute logits for all classes.
            for (c, prob) in probs.iter_mut().enumerate().take(self.n_classes) {
                let mut z = self.weights[c][0]; // bias
                for j in 0..m {
                    z += self.weights[c][j + 1] * data.features[j][i];
                }
                *prob = z;
            }

            // Softmax.
            let max_s = probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for p in &mut probs[..self.n_classes] {
                *p = (*p - max_s).exp();
                sum += *p;
            }
            for p in &mut probs[..self.n_classes] {
                *p /= sum;
            }

            // Accumulate gradient.
            for (c, (&pc, gc)) in probs
                .iter()
                .zip(gradient.iter_mut())
                .enumerate()
                .take(self.n_classes)
            {
                let y_i = if target_class == c { 1.0 } else { 0.0 };
                let error = sw * (pc - y_i);
                gc[0] += error;
                for j in 0..m {
                    gc[j + 1] += error * data.features[j][i];
                }
            }
        }

        // Penalty ratios.
        let (l1_ratio, l2_ratio) = match &self.penalty {
            Penalty::None => (0.0, 0.0),
            Penalty::L1 => (1.0, 0.0),
            Penalty::L2 => (0.0, 1.0),
            Penalty::ElasticNet(r) => (*r, 1.0 - *r),
        };

        let inv_n = 1.0 / n as f64;

        // Update weights with L2 gradient and learning rate.
        for (c_grad, c_w) in gradient
            .iter_mut()
            .zip(self.weights.iter_mut())
            .take(self.n_classes)
        {
            for (j, (g, w)) in c_grad.iter_mut().zip(c_w.iter_mut()).enumerate().take(dim) {
                *g *= inv_n;
                if j > 0 {
                    *g += self.alpha * inv_n * l2_ratio * *w;
                }
                *w -= self.learning_rate * *g;
            }
        }

        // Proximal step for L1 component (soft-thresholding).
        if l1_ratio > 0.0 {
            let threshold = self.learning_rate * self.alpha * inv_n * l1_ratio;
            for c_w in self.weights.iter_mut().take(self.n_classes) {
                for w in c_w.iter_mut().skip(1) {
                    let sign = w.signum();
                    *w = sign * (*w * sign - threshold).max(0.0);
                }
            }
        }

        self.fitted = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        !self.weights.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_linearly_separable() {
        // Class 0: x < 5, Class 1: x >= 5
        let features = vec![(0..20).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut lr = LogisticRegression::new().alpha(0.0).max_iter(200);
        lr.fit(&data).unwrap();

        let matrix = data.feature_matrix();
        let preds = lr.predict(&matrix).unwrap();
        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.85,
            "expected ≥85% accuracy, got {:.1}%",
            acc * 100.0
        );
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let features = vec![vec![1.0, 2.0, 3.0]];
        let target = vec![0.0, 1.0, 0.0];
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut lr = LogisticRegression::new().max_iter(100);
        lr.fit(&data).unwrap();

        let probas = lr.predict_proba(&[vec![2.0]]).unwrap();
        let sum: f64 = probas[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gd_solver_still_works() {
        let features = vec![(0..20).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .learning_rate(0.1)
            .max_iter(1000);
        lr.fit(&data).unwrap();

        let matrix = data.feature_matrix();
        let preds = lr.predict(&matrix).unwrap();
        let acc = preds
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / data.n_samples() as f64;

        assert!(
            acc >= 0.85,
            "GD solver: expected ≥85% accuracy, got {:.1}%",
            acc * 100.0
        );
    }

    #[test]
    fn test_lbfgs_is_default() {
        let lr = LogisticRegression::new();
        assert_eq!(lr.solver, Solver::Lbfgs);
    }

    #[test]
    fn test_l1_sparsity() {
        // 4 features: x0 = signal, x1 = signal, x2 = noise, x3 = noise.
        // Signal features have clear class separation; noise features are random.
        let n = 200;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        for i in 0..n {
            // Strong signal: class 0 centered at -3, class 1 centered at +3
            let class = i32::from(i >= n / 2);
            let offset = if class == 0 { -3.0 } else { 3.0 };
            f0.push(offset + (i % 7) as f64 * 0.1);
            f1.push(offset * 0.5 + (i % 5) as f64 * 0.05);
            // Noise: same distribution regardless of class
            f2.push((i % 3) as f64 * 0.01);
            f3.push((i % 5) as f64 * 0.01);
            target.push(class as f64);
        }

        let data = Dataset::new(
            vec![f0, f1, f2, f3],
            target,
            vec![
                "sig0".into(),
                "sig1".into(),
                "noise0".into(),
                "noise1".into(),
            ],
            "class",
        );

        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .penalty(Penalty::L1)
            .alpha(0.1)
            .learning_rate(0.1)
            .max_iter(3000);
        lr.fit(&data).unwrap();

        // Noise coefficients should be driven toward zero.
        let w = &lr.weights()[0];
        let noise_mag = w[3].abs() + w[4].abs(); // indices 3,4 = features 2,3 (skip bias)
        let signal_mag = w[1].abs() + w[2].abs();
        assert!(
            signal_mag > 0.01,
            "L1: signal coefficients should be nonzero, got {signal_mag:.6}"
        );
        assert!(
            noise_mag < signal_mag * 0.3,
            "L1: noise coefficients ({noise_mag:.4}) should be much smaller than signal ({signal_mag:.4})"
        );
    }

    #[test]
    fn test_l2_no_sparsity() {
        let n = 200;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        for i in 0..n {
            let x = i as f64 / n as f64;
            f0.push(x);
            f1.push(x * 2.0);
            f2.push(0.5 + (i % 3) as f64 * 0.01);
            f3.push(0.5 - (i % 5) as f64 * 0.01);
            target.push(if x < 0.5 { 0.0 } else { 1.0 });
        }

        let data = Dataset::new(
            vec![f0, f1, f2, f3],
            target,
            vec![
                "sig0".into(),
                "sig1".into(),
                "noise0".into(),
                "noise1".into(),
            ],
            "class",
        );

        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .penalty(Penalty::L2)
            .alpha(0.01)
            .learning_rate(0.5)
            .max_iter(2000);
        lr.fit(&data).unwrap();

        // L2 should keep ALL coefficients nonzero (no sparsity).
        let w = &lr.weights()[0];
        for (j, &wj) in w.iter().enumerate().skip(1) {
            assert!(
                wj.abs() > 1e-6,
                "L2: coefficient w[{j}] = {wj:.6} should be nonzero"
            );
        }
    }

    #[test]
    fn test_elasticnet_middle_ground() {
        let n = 200;
        let mut f0 = Vec::with_capacity(n);
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut f3 = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);

        for i in 0..n {
            let class = i32::from(i >= n / 2);
            let offset = if class == 0 { -3.0 } else { 3.0 };
            f0.push(offset + (i % 7) as f64 * 0.1);
            f1.push(offset * 0.5 + (i % 5) as f64 * 0.05);
            f2.push((i % 3) as f64 * 0.01);
            f3.push((i % 5) as f64 * 0.01);
            target.push(class as f64);
        }

        let data = Dataset::new(
            vec![f0, f1, f2, f3],
            target,
            vec![
                "sig0".into(),
                "sig1".into(),
                "noise0".into(),
                "noise1".into(),
            ],
            "class",
        );

        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .penalty(Penalty::ElasticNet(0.5))
            .alpha(0.1)
            .learning_rate(0.1)
            .max_iter(3000);
        lr.fit(&data).unwrap();

        // ElasticNet: signal coefficients should remain present.
        let w = &lr.weights()[0];
        let signal_mag = w[1].abs() + w[2].abs();
        assert!(
            signal_mag > 0.01,
            "ElasticNet: signal coefficients should remain nonzero, got {signal_mag:.6}"
        );
    }

    #[test]
    fn test_lbfgs_rejects_l1() {
        let features = vec![vec![1.0, 2.0, 3.0]];
        let target = vec![0.0, 1.0, 0.0];
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut lr = LogisticRegression::new()
            .solver(Solver::Lbfgs)
            .penalty(Penalty::L1)
            .alpha(0.1);
        let result = lr.fit(&data);
        assert!(result.is_err(), "L-BFGS should reject L1 penalty");

        // Also reject ElasticNet.
        let mut lr2 = LogisticRegression::new()
            .solver(Solver::Lbfgs)
            .penalty(Penalty::ElasticNet(0.5))
            .alpha(0.1);
        let result2 = lr2.fit(&data);
        assert!(result2.is_err(), "L-BFGS should reject ElasticNet penalty");
    }

    #[test]
    fn test_partial_fit_is_initialized() {
        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .learning_rate(0.1);
        assert!(!lr.is_initialized());

        let features = vec![(0..20).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "class");
        lr.partial_fit(&data).unwrap();
        assert!(lr.is_initialized());
    }

    #[test]
    fn test_partial_fit_convergence_10_batches() {
        // Linearly separable: class 0 = low x, class 1 = high x.
        // 10 batches of 100 samples each.
        let mut lr = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .learning_rate(0.1)
            .alpha(0.0);

        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..10 {
            let mut feats = Vec::with_capacity(100);
            let mut tgt = Vec::with_capacity(100);
            for _ in 0..50 {
                feats.push(rng.f64() * 3.0); // class 0: [0, 3)
                tgt.push(0.0);
            }
            for _ in 0..50 {
                feats.push(7.0 + rng.f64() * 3.0); // class 1: [7, 10)
                tgt.push(1.0);
            }
            let batch = Dataset::new(vec![feats], tgt, vec!["x".into()], "class");
            lr.partial_fit(&batch).unwrap();
        }

        // Test on held-out points.
        let preds = lr.predict(&[vec![1.0], vec![9.0]]).unwrap();
        assert!(
            (preds[0] - 0.0).abs() < f64::EPSILON,
            "expected class 0 for x=1"
        );
        assert!(
            (preds[1] - 1.0).abs() < f64::EPSILON,
            "expected class 1 for x=9"
        );
    }

    #[test]
    fn test_partial_fit_single_batch_approximates_fit() {
        // Normalized features to avoid large gradient magnitudes.
        let features = vec![(0..40).map(|i| i as f64 / 40.0).collect()];
        let target: Vec<f64> = (0..40).map(|i| if i < 20 { 0.0 } else { 1.0 }).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        // partial_fit many passes on same data
        let mut lr_partial = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .learning_rate(1.0)
            .alpha(0.0);
        for _ in 0..500 {
            lr_partial.partial_fit(&data).unwrap();
        }

        // Full fit with same settings
        let mut lr_full = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .learning_rate(1.0)
            .alpha(0.0)
            .max_iter(500);
        lr_full.fit(&data).unwrap();

        // Both should classify correctly
        let matrix = data.feature_matrix();
        let preds_partial = lr_partial.predict(&matrix).unwrap();
        let preds_full = lr_full.predict(&matrix).unwrap();

        let acc_partial = preds_partial
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / 40.0;
        let acc_full = preds_full
            .iter()
            .zip(data.target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / 40.0;

        assert!(
            acc_partial >= 0.85,
            "partial_fit accuracy {:.1}% too low",
            acc_partial * 100.0
        );
        assert!(
            acc_full >= 0.85,
            "full fit accuracy {:.1}% too low",
            acc_full * 100.0
        );
    }

    #[test]
    fn test_sparse_fit_predict_matches_dense() {
        let features = vec![(0..20).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let data = Dataset::new(features.clone(), target.clone(), vec!["x".into()], "class");

        let mut lr_dense = LogisticRegression::new()
            .solver(Solver::GradientDescent)
            .alpha(0.0)
            .learning_rate(0.1)
            .max_iter(500);
        lr_dense.fit(&data).unwrap();

        let csc = CscMatrix::from_dense(&features);
        let mut lr_sparse = LogisticRegression::new()
            .alpha(0.0)
            .learning_rate(0.1)
            .max_iter(500);
        lr_sparse.fit_sparse(&csc, &target).unwrap();

        let matrix = data.feature_matrix();
        let preds_dense = lr_dense.predict(&matrix).unwrap();
        let csr = CsrMatrix::from_dense(&matrix);
        let preds_sparse = lr_sparse.predict_sparse(&csr).unwrap();

        let acc_dense: usize = preds_dense
            .iter()
            .zip(target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count();
        let acc_sparse: usize = preds_sparse
            .iter()
            .zip(target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count();

        assert!(acc_dense >= 17, "Dense accuracy too low: {acc_dense}/20");
        assert!(acc_sparse >= 17, "Sparse accuracy too low: {acc_sparse}/20");
    }

    #[test]
    fn test_binary_sigmoid_matches_predictions() {
        // Verify binary sigmoid fast path produces correct classifications.
        let features = vec![(0..40).map(|i| i as f64).collect()];
        let target: Vec<f64> = (0..40).map(|i| if i < 20 { 0.0 } else { 1.0 }).collect();
        let data = Dataset::new(features, target.clone(), vec!["x".into()], "class");

        let mut lr = LogisticRegression::new().alpha(0.01).max_iter(200);
        lr.fit(&data).unwrap();

        // Verify weights structure: class 0 should be all zeros (reference class).
        assert_eq!(lr.weights().len(), 2, "should have 2 weight vectors");
        assert!(
            lr.weights()[0].iter().all(|&w| w == 0.0),
            "class 0 weights should all be zero (reference class)"
        );
        assert!(
            lr.weights()[1].iter().any(|&w| w != 0.0),
            "class 1 weights should be non-zero"
        );

        // Verify predictions are correct.
        let matrix = data.feature_matrix();
        let preds = lr.predict(&matrix).unwrap();
        let acc = preds
            .iter()
            .zip(target.iter())
            .filter(|(p, t)| (*p - *t).abs() < 1e-6)
            .count() as f64
            / 40.0;
        assert!(
            acc >= 0.90,
            "binary sigmoid: expected ≥90% accuracy, got {:.1}%",
            acc * 100.0
        );

        // Verify probabilities sum to 1.
        let probas = lr.predict_proba(&[vec![5.0], vec![35.0]]).unwrap();
        for (idx, probs) in probas.iter().enumerate() {
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "probabilities for sample {idx} should sum to 1, got {sum}"
            );
        }
        // Low x should predict class 0, high x should predict class 1.
        assert!(
            probas[0][0] > probas[0][1],
            "x=5 should have higher prob for class 0"
        );
        assert!(
            probas[1][1] > probas[1][0],
            "x=35 should have higher prob for class 1"
        );
    }
}
