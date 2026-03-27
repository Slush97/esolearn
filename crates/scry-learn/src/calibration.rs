// SPDX-License-Identifier: MIT OR Apache-2.0
//! Probability calibration for classifiers.
//!
//! Many classifiers (especially tree-based ones) produce poorly-calibrated
//! probability estimates. This module provides methods to map raw classifier
//! outputs to well-calibrated probabilities.
//!
//! # Methods
//!
//! - [`PlattScaling`] — Fits a logistic sigmoid `P(y=1|f) = 1/(1+exp(Af+B))`
//!   using the Platt (1999) algorithm. Best for small datasets.
//! - [`IsotonicRegression`] — Non-parametric calibration using the Pool
//!   Adjacent Violators (PAV) algorithm. More flexible but needs more data.
//! - [`CalibratedClassifierCV`] — Wraps any classifier and calibrates its
//!   `predict_proba` output using cross-validation.
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::calibration::{CalibratedClassifierCV, CalibrationMethod};
//! use scry_learn::tree::RandomForestClassifier;
//!
//! let cal = CalibratedClassifierCV::new(
//!     RandomForestClassifier::new().n_estimators(50),
//!     CalibrationMethod::Isotonic,
//! );
//! ```

use crate::error::{Result, ScryLearnError};

// ---------------------------------------------------------------------------
// Platt Scaling
// ---------------------------------------------------------------------------

/// Sigmoid calibration using Platt's method.
///
/// Fits the parameters A and B of the sigmoid:
/// `P(y=1 | f) = 1 / (1 + exp(A·f + B))`
///
/// Uses the improved algorithm from Platt (1999) with modified target
/// values to avoid saturation:
/// `t+ = (N+ + 1) / (N+ + 2)` and `t- = 1 / (N- + 2)`.
///
/// # Example
///
/// ```
/// use scry_learn::calibration::PlattScaling;
///
/// let mut platt = PlattScaling::new();
/// // decision_values: raw SVM or tree output
/// // labels: 0.0 or 1.0
/// platt.fit(&[2.0, 1.5, -0.5, -1.0], &[1.0, 1.0, 0.0, 0.0]).unwrap();
/// let probs = platt.predict(&[1.0, -1.0]);
/// assert!(probs[0] > 0.5);
/// assert!(probs[1] < 0.5);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct PlattScaling {
    /// Sigmoid parameter A (slope).
    a: f64,
    /// Sigmoid parameter B (intercept).
    b: f64,
    /// Maximum iterations for Newton's method.
    max_iter: usize,
    /// Whether the model has been fitted.
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl PlattScaling {
    /// Create a new unfitted Platt scaler with default parameters.
    pub fn new() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            max_iter: 100,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the maximum number of Newton iterations (default: 100).
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// The fitted A parameter (sigmoid slope).
    pub fn a(&self) -> f64 {
        self.a
    }

    /// The fitted B parameter (sigmoid intercept).
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Fit the sigmoid parameters to decision values and binary labels.
    ///
    /// `decision_values`: raw classifier output (e.g. distance to hyperplane).
    /// `labels`: binary ground truth, each element must be 0.0 or 1.0.
    pub fn fit(&mut self, decision_values: &[f64], labels: &[f64]) -> Result<()> {
        let n = decision_values.len();
        if n != labels.len() {
            return Err(ScryLearnError::InvalidParameter(
                "decision_values and labels must have the same length".into(),
            ));
        }
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        // Count positives and negatives.
        let n_pos = labels.iter().filter(|&&y| y > 0.5).count();
        let n_neg = n - n_pos;
        if n_pos == 0 || n_neg == 0 {
            return Err(ScryLearnError::InvalidParameter(
                "labels must contain both positive and negative samples".into(),
            ));
        }

        // Bail early if all decision values are identical — the sigmoid
        // cannot separate anything and Newton will produce a singular Hessian.
        let dv_min = decision_values.iter().copied().fold(f64::INFINITY, f64::min);
        let dv_max = decision_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (dv_max - dv_min).abs() < f64::EPSILON {
            return Err(ScryLearnError::InvalidData(
                "all decision values are identical — Platt scaling cannot calibrate".into(),
            ));
        }

        // Modified target values to avoid saturation (Platt 1999).
        let t_pos = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
        let t_neg = 1.0 / (n_neg as f64 + 2.0);

        let t: Vec<f64> = labels
            .iter()
            .map(|&y| if y > 0.5 { t_pos } else { t_neg })
            .collect();

        // Newton's method to minimize the negative log-likelihood:
        //   L = -sum[ t_i * log(p_i) + (1-t_i) * log(1-p_i) ]
        // where p_i = 1 / (1 + exp(A*f_i + B))
        let mut a = 0.0_f64;
        let mut b = ((n_neg as f64 + 1.0) / (n_pos as f64 + 1.0)).ln();

        let min_step = crate::constants::PLATT_MIN_STEP;
        let sigma = crate::constants::PLATT_HESSIAN_REG;

        for _ in 0..self.max_iter {
            // Compute gradient and Hessian.
            let mut g1 = 0.0_f64; // dL/dA
            let mut g2 = 0.0_f64; // dL/dB
            let mut h11 = sigma; // d²L/dA²
            let mut h22 = sigma; // d²L/dB²
            let mut h21 = 0.0_f64; // d²L/dAdB

            for i in 0..n {
                let fval = decision_values[i] * a + b;
                let p = sigmoid(fval);
                let d = p - t[i];
                let w = p * (1.0 - p).max(crate::constants::SINGULAR_THRESHOLD);
                let fi = decision_values[i];

                g1 += fi * d;
                g2 += d;
                h11 += fi * fi * w;
                h22 += w;
                h21 += fi * w;
            }

            // Solve the 2×2 system: H * [dA, dB]' = -[g1, g2]'
            let det = h11 * h22 - h21 * h21;
            if det.abs() < crate::constants::PLATT_SINGULAR_DET {
                return Err(ScryLearnError::ConvergenceFailure {
                    iterations: self.max_iter,
                    tolerance: crate::constants::PLATT_SINGULAR_DET,
                });
            }
            let da = -(h22 * g1 - h21 * g2) / det;
            let db = -(h11 * g2 - h21 * g1) / det;

            // Line search with step halving.
            let mut step = 1.0;
            let old_nll = neg_log_likelihood(decision_values, &t, a, b);
            loop {
                let new_a = a + step * da;
                let new_b = b + step * db;
                let new_nll = neg_log_likelihood(decision_values, &t, new_a, new_b);
                if new_nll < old_nll + crate::constants::ARMIJO_C * step * (g1 * da + g2 * db) {
                    a = new_a;
                    b = new_b;
                    break;
                }
                step *= 0.5;
                if step < min_step {
                    a += step * da;
                    b += step * db;
                    break;
                }
            }

            // Check convergence.
            if (da * step).abs() < crate::constants::PLATT_CONVERGENCE && (db * step).abs() < crate::constants::PLATT_CONVERGENCE {
                break;
            }
        }

        self.a = a;
        self.b = b;
        self.fitted = true;
        Ok(())
    }

    /// Transform decision values into calibrated probabilities.
    pub fn predict(&self, decision_values: &[f64]) -> Vec<f64> {
        decision_values
            .iter()
            .map(|&f| sigmoid(self.a * f + self.b))
            .collect()
    }
}

impl Default for PlattScaling {
    fn default() -> Self {
        Self::new()
    }
}

/// Sigmoid function: 1 / (1 + exp(-x)).
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Negative log-likelihood for Platt scaling.
fn neg_log_likelihood(f: &[f64], t: &[f64], a: f64, b: f64) -> f64 {
    let mut nll = 0.0;
    for i in 0..f.len() {
        let p = sigmoid(a * f[i] + b);
        let p_clamped = p.clamp(crate::constants::NEAR_ZERO, 1.0 - crate::constants::NEAR_ZERO);
        nll -= t[i] * p_clamped.ln() + (1.0 - t[i]) * (1.0 - p_clamped).ln();
    }
    nll
}

// ---------------------------------------------------------------------------
// Isotonic Regression
// ---------------------------------------------------------------------------

/// Non-parametric calibration using isotonic (monotone) regression.
///
/// Uses the Pool Adjacent Violators (PAV) algorithm to fit a non-decreasing
/// step function to the data. Predictions use linear interpolation between
/// fitted values.
///
/// # Example
///
/// ```
/// use scry_learn::calibration::IsotonicRegression;
///
/// let mut iso = IsotonicRegression::new();
/// iso.fit(&[0.1, 0.4, 0.6, 0.9], &[0.0, 0.0, 1.0, 1.0]).unwrap();
/// let p = iso.predict(&[0.5]);
/// assert!(p[0] >= 0.0 && p[0] <= 1.0);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct IsotonicRegression {
    /// Fitted x-values (sorted).
    xs: Vec<f64>,
    /// Fitted y-values (non-decreasing, corresponding to xs).
    ys: Vec<f64>,
    /// Whether the model has been fitted.
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl IsotonicRegression {
    /// Create a new unfitted isotonic regression model.
    pub fn new() -> Self {
        Self {
            xs: Vec::new(),
            ys: Vec::new(),
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Fit the isotonic regression to (x, y) pairs.
    ///
    /// `x`: predictor values (e.g. uncalibrated probabilities).
    /// `y`: response values (e.g. 0/1 labels or true probabilities).
    pub fn fit(&mut self, x: &[f64], y: &[f64]) -> Result<()> {
        let n = x.len();
        if n != y.len() {
            return Err(ScryLearnError::InvalidParameter(
                "x and y must have the same length".into(),
            ));
        }
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }

        // Sort by x.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_x: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
        let sorted_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

        // Pool Adjacent Violators (PAV) algorithm.
        // Each block has (sum_y, count, start_idx).
        let mut blocks: Vec<(f64, usize, usize)> = Vec::with_capacity(n);

        for i in 0..n {
            blocks.push((sorted_y[i], 1, i));

            // Merge backwards while the last block violates monotonicity.
            while blocks.len() >= 2 {
                let len = blocks.len();
                let mean_last = blocks[len - 1].0 / blocks[len - 1].1 as f64;
                let mean_prev = blocks[len - 2].0 / blocks[len - 2].1 as f64;
                if mean_prev <= mean_last {
                    break;
                }
                // Merge — loop guard above ensures blocks.len() >= 2.
                let last = blocks.pop().expect("loop guard: blocks.len() >= 2");
                let prev = blocks.last_mut().expect("loop guard: blocks.len() >= 2");
                prev.0 += last.0;
                prev.1 += last.1;
            }
        }

        // Build the fitted (x, y) pairs — one pair per unique x value.
        // For each block, use the block's mean and the mean x of the block.
        let mut fit_x = Vec::with_capacity(blocks.len());
        let mut fit_y = Vec::with_capacity(blocks.len());

        let mut idx = 0;
        for &(sum_y, count, _) in &blocks {
            let mean_y = sum_y / count as f64;
            // Use the first and last x of this block's range.
            let block_start = idx;
            let block_end = idx + count;
            // Use the mid-x of the block as representative.
            let mean_x: f64 =
                sorted_x[block_start..block_end].iter().sum::<f64>() / count as f64;
            fit_x.push(mean_x);
            fit_y.push(mean_y);
            idx = block_end;
        }

        self.xs = fit_x;
        self.ys = fit_y;
        self.fitted = true;
        Ok(())
    }

    /// Predict calibrated values using linear interpolation.
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.interpolate(v)).collect()
    }

    /// Linear interpolation between fitted points.
    fn interpolate(&self, x: f64) -> f64 {
        if self.xs.is_empty() {
            return 0.5;
        }
        if self.xs.len() == 1 {
            return self.ys[0];
        }

        // Clamp to range.
        if x <= self.xs[0] {
            return self.ys[0];
        }
        // xs.len() >= 2 is guaranteed by the early returns above.
        if x >= self.xs[self.xs.len() - 1] {
            return self.ys[self.ys.len() - 1];
        }

        // Binary search for the interval.
        let mut lo = 0;
        let mut hi = self.xs.len() - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.xs[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Linear interpolation.
        let dx = self.xs[hi] - self.xs[lo];
        if dx.abs() < crate::constants::NEAR_ZERO {
            return (self.ys[lo] + self.ys[hi]) / 2.0;
        }
        let t = (x - self.xs[lo]) / dx;
        self.ys[lo] + t * (self.ys[hi] - self.ys[lo])
    }
}

impl Default for IsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Calibration method enum
// ---------------------------------------------------------------------------

/// Calibration method for [`CalibratedClassifierCV`].
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CalibrationMethod {
    /// Platt's sigmoid-based calibration.
    #[default]
    Sigmoid,
    /// Isotonic regression (non-parametric).
    Isotonic,
}

// ---------------------------------------------------------------------------
// CalibratedClassifierCV
// ---------------------------------------------------------------------------

/// A calibrated classifier wrapper.
///
/// Uses internal cross-validation to produce calibrated probability
/// estimates from any classifier that supports `predict_proba`.
///
/// During `fit`, the data is split into `n_folds` folds. For each fold,
/// the base classifier is trained on the training portion, predictions are
/// made on the held-out portion, and those predictions are used to fit a
/// calibration model (per class in the OVR scheme). At predict time, the
/// full model's raw probabilities are transformed through the calibrator.
///
/// # Example
///
/// ```ignore
/// use scry_learn::calibration::{CalibratedClassifierCV, CalibrationMethod};
/// use scry_learn::tree::DecisionTreeClassifier;
/// use scry_learn::dataset::Dataset;
///
/// let data = Dataset::from_csv("iris.csv", "species").unwrap();
/// let mut cal = CalibratedClassifierCV::new(
///     DecisionTreeClassifier::new(),
///     CalibrationMethod::Isotonic,
/// ).n_folds(5);
///
/// cal.fit(&data).unwrap();
/// let probs = cal.predict_proba(&data.feature_matrix()).unwrap();
/// ```
#[non_exhaustive]
pub struct CalibratedClassifierCV {
    /// The base classifier (boxed for heterogeneity).
    base: Box<dyn CalibrableClassifier>,
    /// Calibration method.
    method: CalibrationMethod,
    /// Number of cross-validation folds.
    n_folds: usize,
    /// Per-class calibrators (fitted during `.fit()`).
    calibrators: Vec<CalibratorKind>,
    /// Whether the model has been fitted.
    fitted: bool,
}

/// Internal enum wrapping a fitted calibrator.
enum CalibratorKind {
    Platt(PlattScaling),
    Isotonic(IsotonicRegression),
}

impl CalibratorKind {
    fn predict(&self, values: &[f64]) -> Vec<f64> {
        match self {
            Self::Platt(p) => p.predict(values),
            Self::Isotonic(iso) => iso.predict(values),
        }
    }
}

/// Trait for classifiers that can be calibrated.
///
/// Any classifier with `fit`, `predict`, and `predict_proba` methods
/// that returns `Vec<Vec<f64>>` for probabilities.
pub trait CalibrableClassifier {
    /// Train on a dataset.
    fn fit(&mut self, data: &crate::dataset::Dataset) -> Result<()>;
    /// Predict class labels.
    fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>>;
    /// Predict class probabilities. Returns `[n_samples][n_classes]`.
    fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>>;
    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn CalibrableClassifier>;
}

// Implement CalibrableClassifier for common classifiers.
macro_rules! impl_calibrable {
    ($($ty:ty),* $(,)?) => {
        $(
            impl CalibrableClassifier for $ty {
                fn fit(&mut self, data: &crate::dataset::Dataset) -> Result<()> {
                    self.fit(data)
                }
                fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
                    self.predict(features)
                }
                fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
                    self.predict_proba(features)
                }
                fn clone_box(&self) -> Box<dyn CalibrableClassifier> {
                    Box::new(self.clone())
                }
            }
        )*
    };
}

impl_calibrable! {
    crate::tree::DecisionTreeClassifier,
    crate::tree::RandomForestClassifier,
    crate::tree::GradientBoostingClassifier,
    crate::tree::HistGradientBoostingClassifier,
    crate::linear::LogisticRegression,
    crate::naive_bayes::GaussianNb,
    crate::naive_bayes::BernoulliNB,
    crate::naive_bayes::MultinomialNB,
    crate::svm::LinearSVC,
    crate::neighbors::KnnClassifier,
}

#[cfg(feature = "experimental")]
impl_calibrable! {
    crate::svm::KernelSVC,
}

impl CalibratedClassifierCV {
    /// Create a new calibrated classifier wrapper.
    pub fn new<C: CalibrableClassifier + 'static>(
        classifier: C,
        method: CalibrationMethod,
    ) -> Self {
        Self {
            base: Box::new(classifier),
            method,
            n_folds: 5,
            calibrators: Vec::new(),
            fitted: false,
        }
    }

    /// Set the number of cross-validation folds (default: 5).
    pub fn n_folds(mut self, n: usize) -> Self {
        self.n_folds = n;
        self
    }

    /// Fit the calibrated classifier.
    ///
    /// 1. Splits data into `n_folds` stratified folds.
    /// 2. For each fold, trains a clone of the base classifier on the
    ///    training portion and collects `predict_proba` on held-out.
    /// 3. Fits a per-class calibrators on the aggregated out-of-fold predictions.
    /// 4. Re-trains the base classifier on the full dataset.
    pub fn fit(&mut self, data: &crate::dataset::Dataset) -> Result<()> {
        let n = data.n_samples();
        if n < self.n_folds {
            return Err(ScryLearnError::InvalidParameter(format!(
                "n_folds ({}) must be ≤ n_samples ({n})",
                self.n_folds
            )));
        }

        let features = data.feature_matrix(); // row-major
        let targets = &data.target;

        // Determine number of classes.
        let n_classes = {
            let mut max_class = 0usize;
            for &t in targets {
                let c = t as usize;
                if c > max_class {
                    max_class = c;
                }
            }
            max_class + 1
        };

        // Hold-out predictions: proba[i][c] for sample i, class c.
        let mut oof_proba: Vec<Vec<f64>> = vec![vec![0.0; n_classes]; n];

        // Simple k-fold splitting (deterministic, stratified-like via interleaving).
        let fold_indices = k_fold_indices(n, self.n_folds);

        for fold in 0..self.n_folds {
            let val_mask = &fold_indices[fold];
            let train_indices: Vec<usize> = (0..n).filter(|i| !val_mask.contains(i)).collect();
            let val_indices: Vec<usize> = val_mask.clone();

            // Build training dataset.
            let train_features: Vec<Vec<f64>> = data
                .features
                .iter()
                .map(|col| train_indices.iter().map(|&i| col[i]).collect())
                .collect();
            let train_target: Vec<f64> = train_indices.iter().map(|&i| targets[i]).collect();
            let train_data = crate::dataset::Dataset::new(
                train_features,
                train_target,
                data.feature_names.clone(),
                &data.target_name,
            );

            // Train a clone of the base classifier.
            let mut clf = self.base.clone_box();
            clf.fit(&train_data)?;

            // Predict probabilities on validation fold.
            let val_features: Vec<Vec<f64>> = val_indices
                .iter()
                .map(|&i| features[i].clone())
                .collect();

            let proba = clf.predict_proba(&val_features)?;

            // Store out-of-fold predictions.
            for (j, &val_idx) in val_indices.iter().enumerate() {
                if j < proba.len() {
                    for c in 0..n_classes.min(proba[j].len()) {
                        oof_proba[val_idx][c] = proba[j][c];
                    }
                }
            }
        }

        // Fit per-class calibrators using OOF predictions.
        self.calibrators = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let proba_c: Vec<f64> = oof_proba.iter().map(|p| p[c]).collect();
            let labels_c: Vec<f64> = targets
                .iter()
                .map(|&t| if (t as usize) == c { 1.0 } else { 0.0 })
                .collect();

            let cal = match &self.method {
                CalibrationMethod::Sigmoid => {
                    let mut platt = PlattScaling::new();
                    platt.fit(&proba_c, &labels_c)?;
                    CalibratorKind::Platt(platt)
                }
                CalibrationMethod::Isotonic => {
                    let mut iso = IsotonicRegression::new();
                    iso.fit(&proba_c, &labels_c)?;
                    CalibratorKind::Isotonic(iso)
                }
            };
            self.calibrators.push(cal);
        }

        // Re-train the base classifier on the full dataset.
        self.base.fit(data)?;
        self.fitted = true;
        Ok(())
    }

    /// Predict calibrated probabilities.
    ///
    /// Returns `[n_samples][n_classes]` with calibrated probabilities
    /// that sum to 1 for each sample.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }

        // Get raw probabilities from the base classifier.
        let raw = self.base.predict_proba(features)?;
        let n_classes = self.calibrators.len();

        // Calibrate each class independently, then normalize.
        let mut result = Vec::with_capacity(raw.len());
        for row in &raw {
            let mut calibrated = Vec::with_capacity(n_classes);
            for c in 0..n_classes {
                let raw_p = if c < row.len() { row[c] } else { 0.0 };
                let cal_p = self.calibrators[c].predict(&[raw_p])[0];
                calibrated.push(cal_p.max(0.0));
            }

            // Normalize to sum to 1.
            let sum: f64 = calibrated.iter().sum();
            if sum > 0.0 {
                for p in &mut calibrated {
                    *p /= sum;
                }
            } else {
                // Uniform fallback.
                let uniform = 1.0 / n_classes as f64;
                calibrated.fill(uniform);
            }
            result.push(calibrated);
        }
        Ok(result)
    }

    /// Predict class labels (argmax of calibrated probabilities).
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        let proba = self.predict_proba(features)?;
        Ok(proba
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0.0, |(i, _)| i as f64)
            })
            .collect())
    }
}

/// Simple k-fold index splitting: returns `k` vecs of sample indices.
fn k_fold_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut folds: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    for i in 0..n {
        folds[i % k].push(i);
    }
    folds
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_basic_separation() {
        let mut platt = PlattScaling::new();
        let dv = vec![3.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -3.0];
        let labels = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        platt.fit(&dv, &labels).unwrap();

        let probs = platt.predict(&[2.0, -2.0]);
        assert!(probs[0] > 0.7, "positive should have high prob: {}", probs[0]);
        assert!(probs[1] < 0.3, "negative should have low prob: {}", probs[1]);
    }

    #[test]
    fn test_platt_monotone() {
        let mut platt = PlattScaling::new();
        let dv: Vec<f64> = (-10..=10).map(|x| x as f64).collect();
        let labels: Vec<f64> = dv.iter().map(|&x| if x >= 0.0 { 1.0 } else { 0.0 }).collect();
        platt.fit(&dv, &labels).unwrap();

        let test_vals = vec![-5.0, -2.0, 0.0, 2.0, 5.0];
        let probs = platt.predict(&test_vals);
        for w in probs.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-6,
                "probabilities should be monotone: {} < {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_isotonic_monotone_output() {
        let mut iso = IsotonicRegression::new();
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let y = vec![0.0, 0.0, 0.3, 0.2, 0.5, 0.4, 0.8, 0.9, 1.0];
        iso.fit(&x, &y).unwrap();

        let pred = iso.predict(&x);
        for w in pred.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-10,
                "isotonic output must be non-decreasing: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_isotonic_perfect_data() {
        let mut iso = IsotonicRegression::new();
        let x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let y = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        iso.fit(&x, &y).unwrap();

        let pred = iso.predict(&[0.0, 0.5, 1.0]);
        assert!((pred[0] - 0.0).abs() < 0.05);
        assert!((pred[1] - 0.5).abs() < 0.05);
        assert!((pred[2] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_isotonic_clamp_extrapolation() {
        let mut iso = IsotonicRegression::new();
        iso.fit(&[0.2, 0.5, 0.8], &[0.1, 0.5, 0.9]).unwrap();

        let pred = iso.predict(&[0.0, 1.0]);
        // Should clamp to boundary values.
        assert!((pred[0] - 0.1).abs() < 1e-6);
        assert!((pred[1] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_calibrated_classifier_cv_smoke() {
        use crate::dataset::Dataset;
        use crate::tree::DecisionTreeClassifier;

        let features = vec![
            vec![0.0, 0.5, 1.0, 1.5, 5.0, 5.5, 6.0, 6.5],
            vec![0.0, 0.5, 1.0, 1.5, 5.0, 5.5, 6.0, 6.5],
        ];
        let target = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut cal = CalibratedClassifierCV::new(
            DecisionTreeClassifier::new(),
            CalibrationMethod::Isotonic,
        )
        .n_folds(2);

        cal.fit(&data).unwrap();
        let proba = cal.predict_proba(&data.feature_matrix()).unwrap();

        assert_eq!(proba.len(), 8);
        for row in &proba {
            assert_eq!(row.len(), 2);
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "probabilities should sum to 1, got {sum}"
            );
        }

        let preds = cal.predict(&data.feature_matrix()).unwrap();
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_calibrated_classifier_cv_sigmoid() {
        use crate::dataset::Dataset;
        use crate::tree::DecisionTreeClassifier;

        let features = vec![
            vec![0.0, 0.5, 1.0, 1.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
            vec![0.0, 0.5, 1.0, 1.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
        ];
        let target = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut cal = CalibratedClassifierCV::new(
            DecisionTreeClassifier::new(),
            CalibrationMethod::Sigmoid,
        )
        .n_folds(2);

        cal.fit(&data).unwrap();
        let proba = cal.predict_proba(&data.feature_matrix()).unwrap();

        // All probabilities should be valid.
        for row in &proba {
            for &p in row {
                assert!((0.0..=1.0).contains(&p), "prob out of range: {p}");
            }
        }
    }

    #[test]
    fn test_platt_constant_decision_values() {
        let mut platt = PlattScaling::new();
        let dv = vec![1.0; 10];
        let labels = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = platt.fit(&dv, &labels);
        assert!(result.is_err(), "constant decision values should fail");
        let err = result.unwrap_err();
        assert!(
            matches!(err, ScryLearnError::InvalidData(_)),
            "expected InvalidData, got {err:?}"
        );
    }

    #[test]
    fn test_platt_near_singular() {
        let mut platt = PlattScaling::new().max_iter(5);
        // Decision values with near-zero variance — likely to produce singular Hessian.
        let dv = vec![1.0, 1.0 + 1e-18, 1.0 - 1e-18, 1.0, 1.0 + 1e-18, 1.0 - 1e-18];
        let labels = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let result = platt.fit(&dv, &labels);
        // This should either hit the InvalidData check (values within EPSILON)
        // or the ConvergenceFailure from singular det.
        assert!(result.is_err(), "near-singular should fail");
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                ScryLearnError::InvalidData(_) | ScryLearnError::ConvergenceFailure { .. }
            ),
            "expected InvalidData or ConvergenceFailure, got {err:?}"
        );
    }
}
