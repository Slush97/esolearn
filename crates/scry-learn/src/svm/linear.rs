// SPDX-License-Identifier: MIT OR Apache-2.0
//! Linear SVM classifier and regressor via Pegasos SGD.
//!
//! [`LinearSVC`] uses hinge loss with L2 penalty for classification.
//! [`LinearSVR`] uses ε-insensitive loss with L2 penalty for regression.
//! Both solve the SVM objective via stochastic sub-gradient descent
//! (Pegasos algorithm).

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::sparse::{CscMatrix, CsrMatrix};
use crate::weights::{compute_sample_weights, ClassWeight};

// ─────────────────────────────────────────────────────────────────
// LinearSVC
// ─────────────────────────────────────────────────────────────────

/// Linear Support Vector Classifier.
///
/// Uses the Pegasos SGD algorithm to minimize hinge loss with L2
/// regularisation. Binary problems use a single weight vector;
/// multiclass problems use one-vs-rest (one weight vector per class,
/// prediction = argmax of decision function scores).
///
/// # Example
///
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::svm::LinearSVC;
///
/// let features = vec![
///     vec![0.0, 0.0, 10.0, 10.0],
///     vec![0.0, 0.0, 10.0, 10.0],
/// ];
/// let target = vec![0.0, 0.0, 1.0, 1.0];
/// let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");
///
/// let mut svc = LinearSVC::new().c(1.0).max_iter(500);
/// svc.fit(&data).unwrap();
///
/// let preds = svc.predict(&[vec![1.0, 1.0]]).unwrap();
/// assert_eq!(preds[0] as usize, 0);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LinearSVC {
    c: f64,
    max_iter: usize,
    tol: f64,
    class_weight: ClassWeight,
    probability: bool,
    /// One weight vector per class (OVR). Each vector has length
    /// `n_features + 1` (last element is the bias).
    weights: Vec<Vec<f64>>,
    /// Platt scaling parameters (A, B) per OVR model.
    platt_params: Vec<(f64, f64)>,
    n_classes: usize,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl LinearSVC {
    /// Create a new `LinearSVC` with default parameters.
    ///
    /// Defaults: `C = 1.0`, `max_iter = 1000`, `tol = 1e-4`.
    pub fn new() -> Self {
        Self {
            c: 1.0,
            max_iter: 1000,
            tol: crate::constants::DEFAULT_TOL,
            class_weight: ClassWeight::Uniform,
            probability: false,
            weights: Vec::new(),
            platt_params: Vec::new(),
            n_classes: 0,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the regularisation parameter `C`.
    ///
    /// Larger values penalise misclassification more (tighter margin).
    pub fn c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the maximum number of SGD epochs.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set convergence tolerance on the max weight change per epoch.
    pub fn tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    /// Set class weighting strategy for imbalanced datasets.
    pub fn class_weight(mut self, cw: ClassWeight) -> Self {
        self.class_weight = cw;
        self
    }

    /// Enable Platt scaling for probability estimates.
    ///
    /// When `true`, [`predict_proba`](Self::predict_proba) returns
    /// calibrated class probabilities after fitting.
    pub fn probability(mut self, enable: bool) -> Self {
        self.probability = enable;
        self
    }

    /// Train the SVM on the given dataset.
    ///
    /// Uses Pegasos-style SGD with one-vs-rest decomposition for
    /// multiclass problems (≥ 3 classes). Auto-dispatches to sparse
    /// kernels when the dataset uses sparse storage.
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
        if self.c <= 0.0 || !self.c.is_finite() {
            return Err(ScryLearnError::InvalidParameter(
                "C must be finite and positive".into(),
            ));
        }

        self.n_classes = data.n_classes();
        let sample_weights = compute_sample_weights(&data.target, &self.class_weight);

        // One-vs-rest: train one binary sub-problem per class.
        // For each class k the binary target is +1 / -1.
        self.weights = Vec::with_capacity(self.n_classes);
        self.platt_params = Vec::with_capacity(self.n_classes);

        for cls in 0..self.n_classes {
            let binary_target: Vec<f64> = data
                .target
                .iter()
                .map(|&t| if t as usize == cls { 1.0 } else { -1.0 })
                .collect();

            let w = pegasos_train(
                &data.features,
                &binary_target,
                &sample_weights,
                m,
                n,
                self.c,
                self.max_iter,
                self.tol,
            );

            // Platt scaling: fit sigmoid on decision values.
            let ab = if self.probability {
                let dvals: Vec<f64> = (0..n)
                    .map(|i| {
                        let mut score = w[m]; // bias
                        for (j, feat_col) in data.features.iter().enumerate().take(m) {
                            score += w[j] * feat_col[i];
                        }
                        score
                    })
                    .collect();
                platt_fit(&dvals, &binary_target)
            } else {
                (0.0, 0.0)
            };
            self.platt_params.push(ab);
            self.weights.push(w);
        }

        self.fitted = true;
        Ok(())
    }

    /// Train on sparse data (CSC format).
    fn fit_sparse(&mut self, csc: &CscMatrix, target: &[f64]) -> Result<()> {
        let csr = csc.to_csr();
        let n = csr.n_rows();
        let m = csc.n_cols();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.c <= 0.0 || !self.c.is_finite() {
            return Err(ScryLearnError::InvalidParameter(
                "C must be finite and positive".into(),
            ));
        }

        self.n_classes = {
            let mut max_class = 0usize;
            for &t in target {
                let c = t as usize;
                if c > max_class {
                    max_class = c;
                }
            }
            max_class + 1
        };
        let sample_weights = compute_sample_weights(target, &self.class_weight);

        self.weights = Vec::with_capacity(self.n_classes);
        self.platt_params = Vec::with_capacity(self.n_classes);

        for cls in 0..self.n_classes {
            let binary_target: Vec<f64> = target
                .iter()
                .map(|&t| if t as usize == cls { 1.0 } else { -1.0 })
                .collect();

            let w = pegasos_train_sparse(
                &csr,
                &binary_target,
                &sample_weights,
                m,
                n,
                self.c,
                self.max_iter,
                self.tol,
            );

            let ab = if self.probability {
                let dvals: Vec<f64> = (0..n)
                    .map(|i| {
                        let row = csr.row(i);
                        let mut score = w[m]; // bias
                        for (col, val) in row.iter() {
                            score += w[col] * val;
                        }
                        score
                    })
                    .collect();
                platt_fit(&dvals, &binary_target)
            } else {
                (0.0, 0.0)
            };
            self.platt_params.push(ab);
            self.weights.push(w);
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict class labels from sparse input (CSR format).
    pub fn predict_sparse(&self, csr: &CsrMatrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n = csr.n_rows();
        let mut preds = Vec::with_capacity(n);
        for i in 0..n {
            let row = csr.row(i);
            let mut best_cls = 0usize;
            let mut best_score = f64::NEG_INFINITY;
            for (cls, w) in self.weights.iter().enumerate() {
                let m = w.len() - 1;
                let mut score = w[m]; // bias
                for (col, val) in row.iter() {
                    if col < m {
                        score += w[col] * val;
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_cls = cls;
                }
            }
            preds.push(best_cls as f64);
        }
        Ok(preds)
    }

    /// Predict class labels for the given row-major feature matrix.
    ///
    /// Returns the class whose OVR decision function is largest.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        let scores = self.decision_function(features)?;
        Ok(scores
            .into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0.0, |(idx, _)| idx as f64)
            })
            .collect())
    }

    /// Compute the raw decision function score for each class.
    ///
    /// Returns `scores[sample][class]` = `w · x + b` for each OVR
    /// sub-problem.
    ///
    /// # Example
    ///
    /// ```
    /// use scry_learn::dataset::Dataset;
    /// use scry_learn::svm::LinearSVC;
    ///
    /// let features = vec![
    ///     vec![0.0, 0.0, 10.0, 10.0],
    ///     vec![0.0, 0.0, 10.0, 10.0],
    /// ];
    /// let target = vec![0.0, 0.0, 1.0, 1.0];
    /// let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");
    ///
    /// let mut svc = LinearSVC::new();
    /// svc.fit(&data).unwrap();
    ///
    /// let scores = svc.decision_function(&[vec![1.0, 1.0]]).unwrap();
    /// assert_eq!(scores[0].len(), 2); // two classes
    /// ```
    pub fn decision_function(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(features
            .iter()
            .map(|row| {
                self.weights
                    .iter()
                    .map(|w| {
                        let m = w.len() - 1;
                        let mut score = w[m]; // bias
                        for (j, &x) in row.iter().enumerate().take(m) {
                            score += w[j] * x;
                        }
                        score
                    })
                    .collect()
            })
            .collect())
    }

    /// Predict class probabilities using Platt scaling.
    ///
    /// Requires `.probability(true)` to have been set before fitting.
    /// Returns `probabilities[sample][class]` normalised to sum to 1.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        if !self.probability {
            return Err(ScryLearnError::InvalidParameter(
                "call .probability(true) before fit to enable predict_proba".into(),
            ));
        }
        let scores = self.decision_function(features)?;
        Ok(scores
            .into_iter()
            .map(|row| {
                let raw: Vec<f64> = row
                    .iter()
                    .zip(self.platt_params.iter())
                    .map(|(&dv, &(a, b))| platt_predict(dv, a, b))
                    .collect();
                let sum: f64 = raw.iter().sum();
                if sum > f64::EPSILON {
                    raw.iter().map(|&p| p / sum).collect()
                } else {
                    vec![1.0 / raw.len() as f64; raw.len()]
                }
            })
            .collect())
    }
}

impl Default for LinearSVC {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// LinearSVR
// ─────────────────────────────────────────────────────────────────

/// Linear Support Vector Regressor.
///
/// Uses ε-insensitive loss with L2 penalty, solved by SGD.
/// Predictions within `epsilon` of the true value incur no loss.
///
/// # Example
///
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::svm::LinearSVR;
///
/// let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let data = Dataset::new(features, target, vec!["x".into()], "y");
///
/// let mut svr = LinearSVR::new().c(1.0).epsilon(0.1);
/// svr.fit(&data).unwrap();
///
/// let preds = svr.predict(&[vec![3.0]]).unwrap();
/// assert!((preds[0] - 6.0).abs() < 1.0);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LinearSVR {
    c: f64,
    epsilon: f64,
    max_iter: usize,
    tol: f64,
    /// `w[0..m]` = feature weights, `w[m]` = bias.
    weights: Vec<f64>,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl LinearSVR {
    /// Create a new `LinearSVR` with default parameters.
    ///
    /// Defaults: `C = 1.0`, `epsilon = 0.1`, `max_iter = 1000`, `tol = 1e-4`.
    pub fn new() -> Self {
        Self {
            c: 1.0,
            epsilon: 0.1,
            max_iter: 1000,
            tol: crate::constants::DEFAULT_TOL,
            weights: Vec::new(),
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the regularisation parameter `C`.
    pub fn c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set the epsilon tube width.
    ///
    /// Predictions within `epsilon` of the true value incur zero loss.
    pub fn epsilon(mut self, e: f64) -> Self {
        self.epsilon = e;
        self
    }

    /// Set the maximum number of SGD epochs.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set convergence tolerance on the max weight change per epoch.
    pub fn tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    /// Train the SVR on the given dataset.
    ///
    /// Auto-dispatches to sparse kernels when the dataset uses sparse storage.
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
        if self.c <= 0.0 || !self.c.is_finite() {
            return Err(ScryLearnError::InvalidParameter(
                "C must be finite and positive".into(),
            ));
        }

        let lambda = 1.0 / (self.c * n as f64);
        // w has m+1 elements: m feature weights + 1 bias.
        let mut w = vec![0.0; m + 1];
        let mut t = 1.0_f64;

        let mut prev_w = w.clone();

        for _epoch in 0..self.max_iter {
            for i in 0..n {
                let eta = 1.0 / (lambda * t);
                t += 1.0;

                // Compute prediction: w·x + b
                let mut pred = w[m]; // bias
                for (wj, feat_col) in w.iter().zip(data.features.iter()) {
                    pred += wj * feat_col[i];
                }

                let residual = pred - data.target[i];

                let sign = if residual > self.epsilon {
                    1.0
                } else if residual < -self.epsilon {
                    -1.0
                } else {
                    0.0
                };

                for (wj, feat_col) in w.iter_mut().zip(data.features.iter()) {
                    *wj = (1.0 - eta * lambda) * *wj - eta * sign * feat_col[i];
                }
                w[m] -= eta * sign;
            }

            let max_delta = w
                .iter()
                .zip(prev_w.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_delta < self.tol {
                break;
            }
            prev_w.copy_from_slice(&w);
        }

        self.weights = w;
        self.fitted = true;
        Ok(())
    }

    /// Train on sparse data (CSC format).
    fn fit_sparse(&mut self, csc: &CscMatrix, target: &[f64]) -> Result<()> {
        let csr = csc.to_csr();
        let n = csr.n_rows();
        let m = csc.n_cols();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.c <= 0.0 || !self.c.is_finite() {
            return Err(ScryLearnError::InvalidParameter(
                "C must be finite and positive".into(),
            ));
        }

        let lambda = 1.0 / (self.c * n as f64);
        let mut w = vec![0.0; m + 1];
        let mut t = 1.0_f64;
        let mut prev_w = w.clone();

        for _epoch in 0..self.max_iter {
            for i in 0..n {
                let eta = 1.0 / (lambda * t);
                t += 1.0;

                let row = csr.row(i);
                let mut pred = w[m]; // bias
                for (col, val) in row.iter() {
                    pred += w[col] * val;
                }

                let residual = pred - target[i];
                let sign = if residual > self.epsilon {
                    1.0
                } else if residual < -self.epsilon {
                    -1.0
                } else {
                    0.0
                };

                // Regularise all weights, then update non-zero entries.
                let decay = 1.0 - eta * lambda;
                for wj in w.iter_mut().take(m) {
                    *wj *= decay;
                }
                for (col, val) in row.iter() {
                    w[col] -= eta * sign * val;
                }
                w[m] -= eta * sign;
            }

            let max_delta = w
                .iter()
                .zip(prev_w.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if max_delta < self.tol {
                break;
            }
            prev_w.copy_from_slice(&w);
        }

        self.weights = w;
        self.fitted = true;
        Ok(())
    }

    /// Predict continuous target values.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let m = self.weights.len() - 1;
        Ok(features
            .iter()
            .map(|row| {
                let mut pred = self.weights[m]; // bias
                for (j, &x) in row.iter().enumerate().take(m) {
                    pred += self.weights[j] * x;
                }
                pred
            })
            .collect())
    }

    /// Predict continuous target values from sparse input (CSR format).
    pub fn predict_sparse(&self, csr: &CsrMatrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let m = self.weights.len() - 1;
        let n = csr.n_rows();
        let mut preds = Vec::with_capacity(n);
        for i in 0..n {
            let row = csr.row(i);
            let mut pred = self.weights[m]; // bias
            for (col, val) in row.iter() {
                if col < m {
                    pred += self.weights[col] * val;
                }
            }
            preds.push(pred);
        }
        Ok(preds)
    }
}

impl Default for LinearSVR {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// Pegasos SGD helper (used by LinearSVC)
// ─────────────────────────────────────────────────────────────────

/// Train a single binary SVM via sub-gradient descent.
///
/// Uses full batch gradient with a fixed-then-decay learning rate.
/// `binary_target` values are +1 / -1.
/// Returns weight vector of length `m + 1` (last = bias).
#[allow(clippy::too_many_arguments)]
fn pegasos_train(
    features: &[Vec<f64>],  // [n_features][n_samples] (column-major)
    binary_target: &[f64],  // [n_samples], +1/-1
    sample_weights: &[f64], // [n_samples]
    m: usize,               // n_features
    n: usize,               // n_samples
    c: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let lambda = 1.0 / (c * n as f64);
    let mut w = vec![0.0; m + 1]; // w[0..m] = features, w[m] = bias
    let mut best_w = w.clone();
    let mut best_loss = f64::INFINITY;

    let mut prev_w = w.clone();

    for epoch in 0..max_iter {
        // Decaying learning rate with a floor to avoid stalling.
        let eta = 1.0 / (1.0 + crate::constants::PEGASOS_LR_DECAY * epoch as f64);

        // Batch sub-gradient.
        let mut grad = vec![0.0; m + 1];
        let mut hinge_loss = 0.0_f64;

        for i in 0..n {
            let mut score = w[m]; // bias
            for j in 0..m {
                score += w[j] * features[j][i];
            }

            let y = binary_target[i];
            let sw = sample_weights[i];
            let margin = y * score;

            if margin < 1.0 {
                let loss_contrib = sw * (1.0 - margin);
                hinge_loss += loss_contrib;
                for j in 0..m {
                    grad[j] -= sw * y * features[j][i];
                }
                grad[m] -= sw * y;
            }
        }

        // Average hinge gradient + L2 penalty on weights (not bias).
        for j in 0..m {
            grad[j] = grad[j] / n as f64 + lambda * w[j];
        }
        grad[m] /= n as f64;

        // Update.
        for j in 0..=m {
            w[j] -= eta * grad[j];
        }

        // Track best weights by total loss.
        let total_loss =
            hinge_loss / n as f64 + 0.5 * lambda * w.iter().take(m).map(|x| x * x).sum::<f64>();
        if total_loss < best_loss {
            best_loss = total_loss;
            best_w.copy_from_slice(&w);
        }

        // Convergence: max weight change.
        let max_delta = w
            .iter()
            .zip(prev_w.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if max_delta < tol {
            break;
        }
        prev_w.copy_from_slice(&w);
    }

    best_w
}

// ─────────────────────────────────────────────────────────────────
// Sparse Pegasos SGD helper (used by LinearSVC sparse path)
// ─────────────────────────────────────────────────────────────────

/// Train a single binary SVM on sparse data via batch sub-gradient descent.
///
/// Mirrors `pegasos_train` but operates on a CSR matrix for efficient
/// row access. Returns weight vector of length `m + 1` (last = bias).
#[allow(clippy::too_many_arguments)]
fn pegasos_train_sparse(
    csr: &CsrMatrix,
    binary_target: &[f64],
    sample_weights: &[f64],
    m: usize,
    n: usize,
    c: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let lambda = 1.0 / (c * n as f64);
    let mut w = vec![0.0; m + 1];
    let mut best_w = w.clone();
    let mut best_loss = f64::INFINITY;
    let mut prev_w = w.clone();

    for epoch in 0..max_iter {
        let eta = 1.0 / (1.0 + crate::constants::PEGASOS_LR_DECAY * epoch as f64);

        let mut grad = vec![0.0; m + 1];
        let mut hinge_loss = 0.0_f64;

        for i in 0..n {
            let row = csr.row(i);
            let mut score = w[m]; // bias
            for (col, val) in row.iter() {
                score += w[col] * val;
            }

            let y = binary_target[i];
            let sw = sample_weights[i];
            let margin = y * score;

            if margin < 1.0 {
                hinge_loss += sw * (1.0 - margin);
                for (col, val) in row.iter() {
                    grad[col] -= sw * y * val;
                }
                grad[m] -= sw * y;
            }
        }

        // Average hinge gradient + L2 penalty.
        for j in 0..m {
            grad[j] = grad[j] / n as f64 + lambda * w[j];
        }
        grad[m] /= n as f64;

        for j in 0..=m {
            w[j] -= eta * grad[j];
        }

        let total_loss =
            hinge_loss / n as f64 + 0.5 * lambda * w.iter().take(m).map(|x| x * x).sum::<f64>();
        if total_loss < best_loss {
            best_loss = total_loss;
            best_w.copy_from_slice(&w);
        }

        let max_delta = w
            .iter()
            .zip(prev_w.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if max_delta < tol {
            break;
        }
        prev_w.copy_from_slice(&w);
    }

    best_w
}

// ─────────────────────────────────────────────────────────────────
// Platt scaling (shared with kernel.rs)
// ─────────────────────────────────────────────────────────────────

/// Fit Platt sigmoid parameters (A, B) on decision values.
fn platt_fit(decision_values: &[f64], labels: &[f64]) -> (f64, f64) {
    let n = decision_values.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    let n_pos = labels.iter().filter(|&&y| y > 0.0).count() as f64;
    let n_neg = n as f64 - n_pos;

    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);
    let targets: Vec<f64> = labels
        .iter()
        .map(|&y| if y > 0.0 { t_pos } else { t_neg })
        .collect();

    let mut a = 0.0_f64;
    let mut b = ((n_neg + 1.0) / (n_pos + 1.0)).ln();
    let sigma = crate::constants::PLATT_HESSIAN_REG;

    for _ in 0..100 {
        let mut g1 = 0.0_f64;
        let mut g2 = 0.0_f64;
        let mut h11 = sigma;
        let mut h22 = sigma;
        let mut h21 = 0.0_f64;

        for i in 0..n {
            let fval = decision_values[i] * a + b;
            let p = 1.0 / (1.0 + (-fval).exp());
            let d = p - targets[i];
            let s = p * (1.0 - p);

            g1 += d * decision_values[i];
            g2 += d;
            h11 += s * decision_values[i] * decision_values[i];
            h22 += s;
            h21 += s * decision_values[i];
        }

        let det = h11 * h22 - h21 * h21;
        if det.abs() < crate::constants::PLATT_SINGULAR_DET {
            break;
        }
        let da = -(h22 * g1 - h21 * g2) / det;
        let db = -(h11 * g2 - h21 * g1) / det;

        if da.abs() < crate::constants::PLATT_CONVERGENCE && db.abs() < crate::constants::PLATT_CONVERGENCE {
            break;
        }

        a += da;
        b += db;
    }

    (a, b)
}

/// Predict probability from a single decision value via Platt sigmoid.
#[inline]
fn platt_predict(dv: f64, a: f64, b: f64) -> f64 {
    1.0 / (1.0 + (a * dv + b).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_svc_binary() {
        // Two linearly separable clusters.
        let features = vec![
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = LinearSVC::new().c(1.0).max_iter(500);
        svc.fit(&data).unwrap();

        let preds = svc.predict(&[vec![1.0, 1.0], vec![9.0, 9.0]]).unwrap();
        assert_eq!(preds[0] as usize, 0);
        assert_eq!(preds[1] as usize, 1);
    }

    #[test]
    fn test_linear_svc_decision_function() {
        let features = vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = LinearSVC::new();
        svc.fit(&data).unwrap();

        let scores = svc.decision_function(&[vec![1.0, 1.0]]).unwrap();
        assert_eq!(scores[0].len(), 2);
    }

    #[test]
    fn test_linear_svc_not_fitted() {
        let svc = LinearSVC::new();
        assert!(svc.predict(&[vec![1.0]]).is_err());
        assert!(svc.decision_function(&[vec![1.0]]).is_err());
    }

    #[test]
    fn test_linear_svc_invalid_c() {
        let features = vec![vec![1.0]];
        let target = vec![0.0];
        let data = Dataset::new(features, target, vec!["x".into()], "class");

        let mut svc = LinearSVC::new().c(-1.0);
        assert!(svc.fit(&data).is_err());
    }

    #[test]
    fn test_linear_svr_simple() {
        // y = 2x
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut svr = LinearSVR::new().c(10.0).epsilon(0.1).max_iter(2000);
        svr.fit(&data).unwrap();

        let preds = svr.predict(&[vec![3.0], vec![5.0]]).unwrap();
        assert!(
            (preds[0] - 6.0).abs() < 2.0,
            "Expected ~6.0, got {}",
            preds[0]
        );
        assert!(
            (preds[1] - 10.0).abs() < 2.0,
            "Expected ~10.0, got {}",
            preds[1]
        );
    }

    #[test]
    fn test_linear_svr_not_fitted() {
        let svr = LinearSVR::new();
        assert!(svr.predict(&[vec![1.0]]).is_err());
    }

    #[test]
    fn test_linear_svc_predict_proba() {
        let features = vec![
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = LinearSVC::new().c(1.0).max_iter(500).probability(true);
        svc.fit(&data).unwrap();

        let proba = svc
            .predict_proba(&[vec![1.0, 1.0], vec![9.0, 9.0]])
            .unwrap();
        for row in &proba {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "probabilities should sum to 1, got {sum}"
            );
            for &p in row {
                assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
            }
        }
    }

    #[test]
    fn test_linear_svc_predict_proba_not_enabled() {
        let features = vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = LinearSVC::new();
        svc.fit(&data).unwrap();
        assert!(svc.predict_proba(&[vec![1.0, 1.0]]).is_err());
    }
}
