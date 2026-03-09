// SPDX-License-Identifier: MIT OR Apache-2.0
//! Kernel SVM classifier via Sequential Minimal Optimization (SMO).
//!
//! [`KernelSVC`] supports linear, RBF, and polynomial kernels. It
//! solves the dual SVM problem using a simplified SMO algorithm
//! (Platt 1998) and handles multiclass via one-vs-rest.
//!
//! ## Probability estimates
//!
//! Enable `.probability(true)` to fit Platt scaling after SMO,
//! providing calibrated probabilities via [`KernelSVC::predict_proba`].

use rayon::prelude::*;

use crate::constants::SVM_KERNEL_PAR_THRESHOLD;
use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

// ─────────────────────────────────────────────────────────────────
// Kernel enum
// ─────────────────────────────────────────────────────────────────

/// Kernel function for non-linear SVM.
///
/// # Example
///
/// ```
/// use scry_learn::svm::Kernel;
///
/// let rbf = Kernel::RBF { gamma: 0.5 };
/// let poly = Kernel::Polynomial { degree: 3, coef0: 1.0 };
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Kernel {
    /// Linear kernel: `K(x, y) = x · y`.
    Linear,
    /// Radial Basis Function: `K(x, y) = exp(-gamma · ||x - y||²)`.
    RBF {
        /// Kernel coefficient. Common default: `1 / n_features`.
        gamma: f64,
    },
    /// Polynomial: `K(x, y) = (x · y + coef0)^degree`.
    Polynomial {
        /// Polynomial degree.
        degree: usize,
        /// Independent term.
        coef0: f64,
    },
}

impl Default for Kernel {
    fn default() -> Self {
        Self::RBF { gamma: 1.0 }
    }
}

/// Strategy for computing the RBF gamma parameter.
///
/// # Variants
///
/// - `Scale` — `1 / (n_features × feature_variance)` (sklearn default).
/// - `Auto` — `1 / n_features`.
/// - `Value(f64)` — user-specified constant.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Gamma {
    /// `1.0 / (n_features * X.var())` — sklearn default.
    Scale,
    /// `1.0 / n_features`.
    Auto,
    /// A user-specified gamma value.
    Value(f64),
}

impl Gamma {
    /// Resolve gamma given training data dimensions and variance.
    pub(crate) fn resolve(&self, n_features: usize, feature_variance: f64) -> f64 {
        match self {
            Gamma::Scale => {
                let denom = n_features as f64 * feature_variance;
                if denom > f64::EPSILON {
                    1.0 / denom
                } else {
                    1.0
                }
            }
            Gamma::Auto => {
                if n_features > 0 {
                    1.0 / n_features as f64
                } else {
                    1.0
                }
            }
            Gamma::Value(v) => *v,
        }
    }
}

impl Kernel {
    /// Evaluate the kernel function for two feature vectors.
    #[inline]
    pub(crate) fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            Kernel::Linear => dot(a, b),
            Kernel::RBF { gamma } => {
                let sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                (-gamma * sq).exp()
            }
            #[allow(clippy::cast_possible_wrap)]
            Kernel::Polynomial { degree, coef0 } => (dot(a, b) + coef0).powi(*degree as i32),
        }
    }
}

#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────
// KernelSVC
// ─────────────────────────────────────────────────────────────────

/// Kernel Support Vector Classifier.
///
/// Uses Sequential Minimal Optimization (SMO) to solve the dual SVM
/// problem. Multiclass via one-vs-rest: one binary classifier per
/// class, prediction = argmax decision function.
///
/// # Example
///
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::svm::{KernelSVC, Kernel};
///
/// let features = vec![
///     vec![0.0, 0.0, 10.0, 10.0],
///     vec![0.0, 0.0, 10.0, 10.0],
/// ];
/// let target = vec![0.0, 0.0, 1.0, 1.0];
/// let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");
///
/// let mut svc = KernelSVC::new()
///     .kernel(Kernel::RBF { gamma: 0.1 })
///     .c(1.0);
/// svc.fit(&data).unwrap();
///
/// let preds = svc.predict(&[vec![1.0, 1.0]]).unwrap();
/// assert_eq!(preds[0] as usize, 0);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KernelSVC {
    kernel: Kernel,
    c: f64,
    tol: f64,
    max_iter: usize,
    gamma_strategy: Option<Gamma>,
    probability: bool,
    /// One binary model per class (OVR).
    models: Vec<BinarySMO>,
    /// Platt scaling parameters (A, B) per OVR model.
    platt_params: Vec<(f64, f64)>,
    n_classes: usize,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

/// Internal binary SMO model for one OVR sub-problem.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub(crate) struct BinarySMO {
    /// Dual variables.
    pub(crate) alphas: Vec<f64>,
    /// Bias term.
    pub(crate) b: f64,
    /// Training support vectors (row-major).
    pub(crate) support_vectors: Vec<Vec<f64>>,
    /// Binary labels (+1 / -1) for each SV.
    pub(crate) labels: Vec<f64>,
}

impl KernelSVC {
    /// Create a new `KernelSVC` with default parameters.
    ///
    /// Defaults: RBF with `Gamma::Scale`, `C = 1.0`, `tol = 1e-3`, `max_iter = 1000`.
    /// Gamma is resolved from data variance during [`fit`](Self::fit),
    /// matching sklearn's `SVC(kernel='rbf')` behaviour.
    pub fn new() -> Self {
        Self {
            kernel: Kernel::default(),
            c: 1.0,
            tol: crate::constants::SMO_TOL,
            max_iter: 1000,
            gamma_strategy: Some(Gamma::Scale),
            probability: false,
            models: Vec::new(),
            platt_params: Vec::new(),
            n_classes: 0,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the kernel function.
    ///
    /// Setting a non-RBF kernel clears any gamma strategy so it is not
    /// silently overwritten to RBF during [`fit`](Self::fit).
    pub fn kernel(mut self, k: Kernel) -> Self {
        if !matches!(k, Kernel::RBF { .. }) {
            self.gamma_strategy = None;
        }
        self.kernel = k;
        self
    }

    /// Set the regularisation parameter `C`.
    pub fn c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }

    /// Set convergence tolerance for SMO.
    pub fn tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    /// Set the maximum number of SMO passes.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set the gamma strategy for RBF kernels.
    ///
    /// When set, the gamma value is computed during [`fit`](Self::fit)
    /// and overrides any gamma specified in the `Kernel::RBF` variant.
    pub fn gamma(mut self, g: Gamma) -> Self {
        self.gamma_strategy = Some(g);
        self
    }

    /// Enable Platt scaling for probability estimates.
    ///
    /// When `true`, [`predict_proba`](Self::predict_proba) returns
    /// calibrated class probabilities fitted via Platt's sigmoid method.
    pub fn probability(mut self, enable: bool) -> Self {
        self.probability = enable;
        self
    }

    /// Train the kernel SVM using SMO (one-vs-rest for multiclass).
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.c <= 0.0 || !self.c.is_finite() {
            return Err(ScryLearnError::InvalidParameter(
                "C must be finite and positive".into(),
            ));
        }

        // Resolve gamma strategy if set.
        if let Some(ref gs) = self.gamma_strategy {
            let m = data.n_features();
            let rows = data.feature_matrix();
            let variance = compute_feature_variance(&rows, m);
            let g = gs.resolve(m, variance);
            self.kernel = Kernel::RBF { gamma: g };
        }

        self.n_classes = data.n_classes();

        // Build row-major feature matrix once.
        let rows = data.feature_matrix();

        if self.n_classes > 2 {
            let results: Vec<(BinarySMO, (f64, f64))> = (0..self.n_classes)
                .into_par_iter()
                .map(|cls| {
                    let binary_y: Vec<f64> = data
                        .target
                        .iter()
                        .map(|&t| if t as usize == cls { 1.0 } else { -1.0 })
                        .collect();

                    let model = smo_train(
                        &rows,
                        &binary_y,
                        &self.kernel,
                        self.c,
                        self.tol,
                        self.max_iter,
                    );

                    let ab = if self.probability {
                        let dvals: Vec<f64> = rows
                            .iter()
                            .map(|x| smo_decision(&model, x, &self.kernel))
                            .collect();
                        platt_fit(&dvals, &binary_y)
                    } else {
                        (0.0, 0.0)
                    };
                    (model, ab)
                })
                .collect();

            self.models = Vec::with_capacity(self.n_classes);
            self.platt_params = Vec::with_capacity(self.n_classes);
            for (model, ab) in results {
                self.models.push(model);
                self.platt_params.push(ab);
            }
        } else {
            self.models = Vec::with_capacity(self.n_classes);
            self.platt_params = Vec::with_capacity(self.n_classes);

            for cls in 0..self.n_classes {
                let binary_y: Vec<f64> = data
                    .target
                    .iter()
                    .map(|&t| if t as usize == cls { 1.0 } else { -1.0 })
                    .collect();

                let model = smo_train(
                    &rows,
                    &binary_y,
                    &self.kernel,
                    self.c,
                    self.tol,
                    self.max_iter,
                );

                let ab = if self.probability {
                    let dvals: Vec<f64> = rows
                        .iter()
                        .map(|x| smo_decision(&model, x, &self.kernel))
                        .collect();
                    platt_fit(&dvals, &binary_y)
                } else {
                    (0.0, 0.0)
                };
                self.platt_params.push(ab);
                self.models.push(model);
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict class labels.
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
    /// # Example
    ///
    /// ```
    /// use scry_learn::dataset::Dataset;
    /// use scry_learn::svm::{KernelSVC, Kernel};
    ///
    /// let features = vec![
    ///     vec![0.0, 0.0, 10.0, 10.0],
    ///     vec![0.0, 0.0, 10.0, 10.0],
    /// ];
    /// let target = vec![0.0, 0.0, 1.0, 1.0];
    /// let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");
    ///
    /// let mut svc = KernelSVC::new().kernel(Kernel::Linear);
    /// svc.fit(&data).unwrap();
    ///
    /// let scores = svc.decision_function(&[vec![1.0, 1.0]]).unwrap();
    /// assert_eq!(scores[0].len(), 2);
    /// ```
    pub fn decision_function(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(features
            .iter()
            .map(|x| {
                self.models
                    .iter()
                    .map(|model| smo_decision(model, x, &self.kernel))
                    .collect()
            })
            .collect())
    }

    /// Predict class probabilities using Platt scaling.
    ///
    /// Requires `.probability(true)` to have been set before fitting.
    /// Returns `probabilities[sample][class]` normalised to sum to 1.
    ///
    /// # Errors
    ///
    /// Returns [`ScryLearnError::NotFitted`] if the model has not been
    /// fitted, or [`ScryLearnError::InvalidParameter`] if probability
    /// estimation was not enabled.
    pub fn predict_proba(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        if !self.probability {
            return Err(ScryLearnError::InvalidParameter(
                "call .probability(true) before fit to enable predict_proba".into(),
            ));
        }
        Ok(features
            .iter()
            .map(|x| {
                let raw: Vec<f64> = self
                    .models
                    .iter()
                    .zip(self.platt_params.iter())
                    .map(|(model, &(a, b))| {
                        let dv = smo_decision(model, x, &self.kernel);
                        platt_predict(dv, a, b)
                    })
                    .collect();
                // Normalise to sum to 1.
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

impl Default for KernelSVC {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// Simplified SMO solver
// ─────────────────────────────────────────────────────────────────

/// Train a binary SVM using simplified SMO (Platt 1998, simplified variant).
pub(crate) fn smo_train(
    x: &[Vec<f64>], // [n_samples][n_features] row-major
    y: &[f64],      // [n_samples], +1/-1
    kernel: &Kernel,
    c: f64,
    tol: f64,
    max_passes: usize,
) -> BinarySMO {
    let n = x.len();

    // Guard: with 0 or 1 sample, SMO cannot select a pair (j index
    // computation uses `n - 1` as divisor → div-by-zero when n == 1).
    if n <= 1 {
        return BinarySMO {
            alphas: if n == 1 { vec![c.min(1.0)] } else { Vec::new() },
            b: 0.0,
            support_vectors: x.to_vec(),
            labels: y.to_vec(),
        };
    }

    let mut alphas = vec![0.0; n];
    let mut b = 0.0_f64;

    // Pre-compute kernel matrix for efficiency (O(n²) memory, fine for
    // typical SVM datasets).
    let mut k_matrix = vec![vec![0.0; n]; n];
    if n * n >= SVM_KERNEL_PAR_THRESHOLD {
        k_matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
            for j in 0..n {
                row[j] = kernel.eval(&x[i], &x[j]);
            }
        });
    } else {
        for i in 0..n {
            for j in i..n {
                let val = kernel.eval(&x[i], &x[j]);
                k_matrix[i][j] = val;
                k_matrix[j][i] = val;
            }
        }
    }

    let mut passes = 0_usize;
    let mut total_iter = 0_usize;
    let hard_cap = max_passes * n;

    while passes < max_passes && total_iter < hard_cap {
        let mut num_changed = 0_usize;
        total_iter += 1;

        for i in 0..n {
            // Error for sample i.
            let e_i = smo_predict_raw(&alphas, y, &k_matrix[i], b) - y[i];

            if (y[i] * e_i < -tol && alphas[i] < c) || (y[i] * e_i > tol && alphas[i] > 0.0) {
                // Select j ≠ i randomly via simple deterministic heuristic.
                let j = (i + 1 + (passes % (n - 1))) % n;

                let e_j = smo_predict_raw(&alphas, y, &k_matrix[j], b) - y[j];

                let alpha_i_old = alphas[i];
                let alpha_j_old = alphas[j];

                // Compute bounds L, H.
                let (l, h) = if (y[i] - y[j]).abs() > f64::EPSILON {
                    // y_i ≠ y_j
                    (
                        f64::max(0.0, alphas[j] - alphas[i]),
                        f64::min(c, c + alphas[j] - alphas[i]),
                    )
                } else {
                    // y_i == y_j
                    (
                        f64::max(0.0, alphas[i] + alphas[j] - c),
                        f64::min(c, alphas[i] + alphas[j]),
                    )
                };

                if (l - h).abs() < crate::constants::SMO_BOUNDS_EQ {
                    continue;
                }

                // Second derivative (eta).
                let eta = 2.0 * k_matrix[i][j] - k_matrix[i][i] - k_matrix[j][j];
                if eta >= 0.0 {
                    continue;
                }

                // Update alpha_j.
                alphas[j] -= y[j] * (e_i - e_j) / eta;
                alphas[j] = alphas[j].clamp(l, h);

                if (alphas[j] - alpha_j_old).abs() < crate::constants::SMO_ALPHA_CHANGE_THRESH {
                    continue;
                }

                // Update alpha_i.
                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j]);

                // Update bias.
                let b1 = b
                    - e_i
                    - y[i] * (alphas[i] - alpha_i_old) * k_matrix[i][i]
                    - y[j] * (alphas[j] - alpha_j_old) * k_matrix[i][j];
                let b2 = b
                    - e_j
                    - y[i] * (alphas[i] - alpha_i_old) * k_matrix[i][j]
                    - y[j] * (alphas[j] - alpha_j_old) * k_matrix[j][j];

                b = if alphas[i] > 0.0 && alphas[i] < c {
                    b1
                } else if alphas[j] > 0.0 && alphas[j] < c {
                    b2
                } else {
                    (b1 + b2) / 2.0
                };

                num_changed += 1;
            }
        }

        if num_changed == 0 {
            passes += 1;
        } else {
            passes = 0;
        }
    }

    // Keep only support vectors (alpha > 0) for compact storage.
    let mut sv_list = Vec::new();
    let mut sv_labels = Vec::new();
    let mut sv_alphas = Vec::new();
    for i in 0..n {
        if alphas[i] > crate::constants::SV_ALPHA_THRESH {
            sv_alphas.push(alphas[i]);
            sv_labels.push(y[i]);
            sv_list.push(x[i].clone());
        }
    }

    BinarySMO {
        alphas: sv_alphas,
        b,
        support_vectors: sv_list,
        labels: sv_labels,
    }
}

/// Compute raw decision value: Σ αᵢ yᵢ K(xᵢ, x) + b.
#[inline]
fn smo_predict_raw(alphas: &[f64], y: &[f64], k_row: &[f64], b: f64) -> f64 {
    let mut sum = b;
    for ((&a, &yi), &ki) in alphas.iter().zip(y.iter()).zip(k_row.iter()) {
        sum += a * yi * ki;
    }
    sum
}

/// Decision function for a trained binary SMO model on a new sample.
pub(crate) fn smo_decision(model: &BinarySMO, x: &[f64], kernel: &Kernel) -> f64 {
    let mut sum = model.b;
    for i in 0..model.alphas.len() {
        sum += model.alphas[i] * model.labels[i] * kernel.eval(&model.support_vectors[i], x);
    }
    sum
}

// ─────────────────────────────────────────────────────────────────
// Platt scaling
// ─────────────────────────────────────────────────────────────────

/// Fit Platt sigmoid parameters (A, B) on decision values.
///
/// Minimises -Σ [tᵢ log pᵢ + (1-tᵢ) log(1-pᵢ)] where
/// pᵢ = 1 / (1 + exp(A·fᵢ + B)) and tᵢ are smoothed targets.
fn platt_fit(decision_values: &[f64], labels: &[f64]) -> (f64, f64) {
    let n = decision_values.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    let n_pos = labels.iter().filter(|&&y| y > 0.0).count() as f64;
    let n_neg = n as f64 - n_pos;

    // Smoothed targets (Platt 2000).
    let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
    let t_neg = 1.0 / (n_neg + 2.0);
    let targets: Vec<f64> = labels
        .iter()
        .map(|&y| if y > 0.0 { t_pos } else { t_neg })
        .collect();

    // Newton's method for A and B.
    let mut a = 0.0_f64;
    let mut b = ((n_neg + 1.0) / (n_pos + 1.0)).ln();

    let max_iter = 100;
    let min_step = crate::constants::PLATT_MIN_STEP;
    let sigma = crate::constants::PLATT_HESSIAN_REG;

    for _ in 0..max_iter {
        let mut g1 = 0.0_f64; // dL/dA
        let mut g2 = 0.0_f64; // dL/dB
        let mut h11 = sigma; // d²L/dA²
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

        if da.abs() < min_step && db.abs() < min_step {
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

// ─────────────────────────────────────────────────────────────────
// Helper: feature variance
// ─────────────────────────────────────────────────────────────────

/// Mean variance across all features (for Gamma::Scale).
fn compute_feature_variance(rows: &[Vec<f64>], n_features: usize) -> f64 {
    let n = rows.len() as f64;
    if n <= 1.0 || n_features == 0 {
        return 1.0;
    }
    let mut total_var = 0.0;
    for j in 0..n_features {
        let mean = rows.iter().map(|r| r[j]).sum::<f64>() / n;
        let var = rows.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n;
        total_var += var;
    }
    total_var / n_features as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_svc_linear() {
        // Two linearly separable clusters.
        let features = vec![
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = KernelSVC::new().kernel(Kernel::Linear).c(1.0);
        svc.fit(&data).unwrap();

        let preds = svc.predict(&[vec![1.0, 1.0], vec![9.0, 9.0]]).unwrap();
        assert_eq!(preds[0] as usize, 0);
        assert_eq!(preds[1] as usize, 1);
    }

    #[test]
    fn test_kernel_svc_rbf_xor() {
        // XOR: not linearly separable, but RBF should handle it.
        let features = vec![vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]];
        let target = vec![0.0, 1.0, 1.0, 0.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = KernelSVC::new()
            .kernel(Kernel::RBF { gamma: 5.0 })
            .c(10.0)
            .max_iter(500);
        svc.fit(&data).unwrap();

        let preds = svc
            .predict(&[
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ])
            .unwrap();
        // Should get at least 3/4 correct.
        let correct = preds
            .iter()
            .zip([0.0, 1.0, 1.0, 0.0].iter())
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count();
        assert!(
            correct >= 3,
            "RBF should solve XOR (got {correct}/4 correct)"
        );
    }

    #[test]
    fn test_kernel_svc_not_fitted() {
        let svc = KernelSVC::new();
        assert!(svc.predict(&[vec![1.0]]).is_err());
    }

    #[test]
    fn test_kernel_svc_decision_function() {
        let features = vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = KernelSVC::new().kernel(Kernel::Linear);
        svc.fit(&data).unwrap();

        let scores = svc.decision_function(&[vec![1.0, 1.0]]).unwrap();
        assert_eq!(scores[0].len(), 2);
    }

    #[test]
    fn test_kernel_svc_predict_proba() {
        let features = vec![
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = KernelSVC::new()
            .kernel(Kernel::Linear)
            .c(1.0)
            .probability(true);
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
                assert!(p >= 0.0 && p <= 1.0, "probability out of range: {p}");
            }
        }
    }

    #[test]
    fn test_kernel_svc_predict_proba_not_enabled() {
        let features = vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = KernelSVC::new().kernel(Kernel::Linear);
        svc.fit(&data).unwrap();
        assert!(svc.predict_proba(&[vec![1.0, 1.0]]).is_err());
    }

    #[test]
    fn test_gamma_auto() {
        let features = vec![
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
        ];
        let target = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let data = Dataset::new(
            features,
            target.clone(),
            vec!["x".into(), "y".into(), "z".into()],
            "class",
        );

        // Auto = 1/n_features = 1/3
        let mut svc = KernelSVC::new().gamma(Gamma::Auto).c(1.0);
        svc.fit(&data).unwrap();

        // After fit, the resolved kernel should be RBF with gamma=1/3.
        match &svc.kernel {
            Kernel::RBF { gamma } => {
                assert!(
                    (*gamma - 1.0 / 3.0).abs() < 1e-10,
                    "Gamma::Auto should give 1/n_features, got {gamma}",
                );
            }
            other => panic!("expected RBF kernel, got {:?}", other),
        }
    }

    #[test]
    fn test_gamma_scale() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 3.0, 4.0, 5.0]];
        let target = vec![0.0, 0.0, 1.0, 1.0];
        let data = Dataset::new(features, target, vec!["a".into(), "b".into()], "class");

        let mut svc = KernelSVC::new().gamma(Gamma::Scale).c(1.0);
        svc.fit(&data).unwrap();

        // gamma should be 1 / (n_features * var)
        match &svc.kernel {
            Kernel::RBF { gamma } => {
                assert!(*gamma > 0.0, "gamma should be positive, got {gamma}");
            }
            other => panic!("expected RBF kernel, got {:?}", other),
        }
    }

    #[test]
    fn test_kernel_svc_single_sample() {
        // Single-sample input should not panic (previously caused div-by-zero
        // in smo_train at `passes % (n - 1)` when n == 1).
        let features = vec![vec![1.0], vec![2.0]];
        let target = vec![0.0];
        let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

        let mut svc = KernelSVC::new().kernel(Kernel::Linear).c(1.0);
        svc.fit(&data).unwrap();

        let preds = svc.predict(&[vec![1.0, 2.0]]).unwrap();
        assert_eq!(preds.len(), 1);
    }
}
