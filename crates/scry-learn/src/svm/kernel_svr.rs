// SPDX-License-Identifier: MIT OR Apache-2.0
//! Kernel Support Vector Regressor via SMO.
//!
//! [`KernelSVR`] uses ε-insensitive loss with kernel functions, solved
//! by an SMO-style algorithm adapted for regression. It shares the
//! [`Kernel`](super::Kernel) enum with [`KernelSVC`](super::KernelSVC).
//!
//! # Example
//!
//! ```
//! use scry_learn::dataset::Dataset;
//! use scry_learn::svm::{KernelSVR, Kernel};
//!
//! let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
//! let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
//! let data = Dataset::new(features, target, vec!["x".into()], "y");
//!
//! let mut svr = KernelSVR::new()
//!     .kernel(Kernel::Linear)
//!     .c(10.0)
//!     .epsilon(0.1);
//! svr.fit(&data).unwrap();
//!
//! let preds = svr.predict(&[vec![3.0]]).unwrap();
//! assert!((preds[0] - 6.0).abs() < 2.0);
//! ```

use super::kernel::{Gamma, Kernel};
use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};

// ─────────────────────────────────────────────────────────────────
// KernelSVR
// ─────────────────────────────────────────────────────────────────

/// Kernel Support Vector Regressor.
///
/// Solves the dual SVR problem using an SMO-style algorithm with
/// ε-insensitive loss. Supports linear, RBF, and polynomial kernels.
///
/// # Example
///
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::svm::{KernelSVR, Kernel};
///
/// let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let data = Dataset::new(features, target, vec!["x".into()], "y");
///
/// let mut svr = KernelSVR::new()
///     .kernel(Kernel::Linear)
///     .c(10.0)
///     .epsilon(0.1);
/// svr.fit(&data).unwrap();
///
/// let preds = svr.predict(&[vec![3.0]]).unwrap();
/// assert!((preds[0] - 6.0).abs() < 2.0);
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KernelSVR {
    kernel: Kernel,
    c: f64,
    epsilon: f64,
    tol: f64,
    max_iter: usize,
    gamma_strategy: Option<Gamma>,
    /// Bias term.
    b: f64,
    /// Training support vectors (row-major).
    sv_x: Vec<Vec<f64>>,
    /// Corresponding α_i - α*_i for support vectors.
    sv_coeff: Vec<f64>,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl KernelSVR {
    /// Create a new `KernelSVR` with default parameters.
    ///
    /// Defaults: RBF with `Gamma::Scale`, `C = 1.0`, `epsilon = 0.1`,
    /// `tol = 1e-3`, `max_iter = 1000`. Gamma is resolved from data
    /// variance during [`fit`](Self::fit), matching sklearn's `SVR(kernel='rbf')`.
    pub fn new() -> Self {
        Self {
            kernel: Kernel::default(),
            c: 1.0,
            epsilon: 0.1,
            tol: 1e-3,
            max_iter: 1000,
            gamma_strategy: Some(Gamma::Scale),
            b: 0.0,
            sv_x: Vec::new(),
            sv_coeff: Vec::new(),
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

    /// Set the epsilon tube width.
    ///
    /// Predictions within `epsilon` of the true value incur zero loss.
    pub fn epsilon(mut self, e: f64) -> Self {
        self.epsilon = e;
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

    /// Train the kernel SVR using SMO.
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

        let rows = data.feature_matrix();
        let y = &data.target;

        // SMO for SVR (ε-SVR dual formulation).
        //
        // The dual has 2n variables: α_i and α*_i.
        // We work with a = α_i - α*_i for each sample.
        // Constraints: -C <= a_i <= C and Σ a_i = 0.
        //
        // The prediction is: f(x) = Σ a_i K(x_i, x) + b

        // Pre-compute kernel matrix.
        let mut k_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i..n {
                let val = self.kernel.eval(&rows[i], &rows[j]);
                k_matrix[i][j] = val;
                k_matrix[j][i] = val;
            }
        }

        let mut a = vec![0.0_f64; n]; // α - α*
        let mut b = 0.0_f64;

        let mut passes = 0_usize;
        let mut total_iter = 0_usize;
        let hard_cap = self.max_iter * n;

        while passes < self.max_iter && total_iter < hard_cap {
            let mut num_changed = 0_usize;
            total_iter += 1;

            for i in 0..n {
                // Prediction for sample i.
                let f_i = svr_predict_raw(&a, &k_matrix[i], b);
                let r_i = f_i - y[i]; // residual

                // Check KKT violations for both α_i and α*_i.
                // For α_i (positive side): if r_i > ε and a_i < C, or r_i < ε and a_i > 0
                // For α*_i (negative side): if -r_i > ε and -a_i < C, or -r_i < ε and -a_i > 0
                let violates_pos = (r_i > self.epsilon + self.tol && a[i] < self.c)
                    || (r_i < self.epsilon - self.tol && a[i] > 0.0);
                let violates_neg = (-r_i > self.epsilon + self.tol && -a[i] < self.c)
                    || (-r_i < self.epsilon - self.tol && -a[i] > 0.0);

                if !violates_pos && !violates_neg {
                    continue;
                }

                // Select j ≠ i.
                let j = (i + 1 + (passes % n.saturating_sub(1).max(1))) % n;
                if j == i {
                    continue;
                }

                let f_j = svr_predict_raw(&a, &k_matrix[j], b);
                let r_j = f_j - y[j];

                let eta = k_matrix[i][i] + k_matrix[j][j] - 2.0 * k_matrix[i][j];
                if eta < 1e-12 {
                    continue;
                }

                let a_i_old = a[i];
                let a_j_old = a[j];

                // Update a_i based on gradient.
                // gradient_i = r_i + ε·sign(a_i) if outside tube, r_i - ε·sign(a_i) otherwise
                let delta_i = if r_i > self.epsilon {
                    r_i - self.epsilon
                } else if r_i < -self.epsilon {
                    r_i + self.epsilon
                } else {
                    0.0
                };

                let delta_j = if r_j > self.epsilon {
                    r_j - self.epsilon
                } else if r_j < -self.epsilon {
                    r_j + self.epsilon
                } else {
                    0.0
                };

                // Coordinate descent update.
                let new_a_i = a[i] - (delta_i - delta_j) / eta;
                let new_a_i = new_a_i.clamp(-self.c, self.c);

                if (new_a_i - a_i_old).abs() < 1e-8 {
                    continue;
                }

                a[i] = new_a_i;
                // Maintain constraint: update a_j to compensate.
                a[j] = a_j_old + (a_i_old - new_a_i);
                a[j] = a[j].clamp(-self.c, self.c);

                // Update bias.
                let b1 =
                    y[i] - self.epsilon * a[i].signum() - svr_predict_raw_no_b(&a, &k_matrix[i]);
                let b2 =
                    y[j] - self.epsilon * a[j].signum() - svr_predict_raw_no_b(&a, &k_matrix[j]);
                b = (b1 + b2) / 2.0;

                num_changed += 1;
            }

            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }

        // Store only support vectors.
        self.sv_x = Vec::new();
        self.sv_coeff = Vec::new();
        for i in 0..n {
            if a[i].abs() > 1e-10 {
                self.sv_x.push(rows[i].clone());
                self.sv_coeff.push(a[i]);
            }
        }
        self.b = b;
        self.fitted = true;
        Ok(())
    }

    /// Predict continuous target values.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(features
            .iter()
            .map(|x| {
                let mut sum = self.b;
                for (sv, &coeff) in self.sv_x.iter().zip(self.sv_coeff.iter()) {
                    sum += coeff * self.kernel.eval(sv, x);
                }
                sum
            })
            .collect())
    }
}

impl Default for KernelSVR {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────

/// Compute f(x_i) = Σ a_j K(x_j, x_i) + b using pre-computed kernel row.
#[inline]
fn svr_predict_raw(a: &[f64], k_row: &[f64], b: f64) -> f64 {
    let mut sum = b;
    for (&ai, &ki) in a.iter().zip(k_row.iter()) {
        sum += ai * ki;
    }
    sum
}

/// Compute Σ a_j K(x_j, x_i) without bias.
#[inline]
fn svr_predict_raw_no_b(a: &[f64], k_row: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (&ai, &ki) in a.iter().zip(k_row.iter()) {
        sum += ai * ki;
    }
    sum
}

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
    fn test_kernel_svr_linear() {
        // y = 2x
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut svr = KernelSVR::new()
            .kernel(Kernel::Linear)
            .c(100.0)
            .epsilon(0.1)
            .max_iter(2000);
        svr.fit(&data).unwrap();

        let preds = svr.predict(&[vec![3.0], vec![5.0]]).unwrap();
        assert!(
            (preds[0] - 6.0).abs() < 3.0,
            "Expected ~6.0, got {}",
            preds[0]
        );
        assert!(
            (preds[1] - 10.0).abs() < 3.0,
            "Expected ~10.0, got {}",
            preds[1]
        );
    }

    #[test]
    fn test_kernel_svr_rbf() {
        // Non-linear: y = x^2 on [-3, 3]
        let n = 30;
        let x: Vec<f64> = (0..n)
            .map(|i| -3.0 + 6.0 * i as f64 / (n - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let data = Dataset::new(vec![x.clone()], y, vec!["x".into()], "y");

        let mut svr = KernelSVR::new()
            .kernel(Kernel::RBF { gamma: 0.5 })
            .c(100.0)
            .epsilon(0.1)
            .max_iter(2000);
        svr.fit(&data).unwrap();

        // Test predictions.
        let test_x = vec![vec![0.0], vec![1.0], vec![-1.0]];
        let preds = svr.predict(&test_x).unwrap();

        // y = x^2: 0^2=0, 1^2=1, (-1)^2=1
        // Allow tolerance of 2.0 since this is a small dataset.
        assert!(preds[0].abs() < 2.0, "Expected ~0, got {}", preds[0]);
        assert!(
            (preds[1] - 1.0).abs() < 2.0,
            "Expected ~1, got {}",
            preds[1]
        );
        assert!(
            (preds[2] - 1.0).abs() < 2.0,
            "Expected ~1, got {}",
            preds[2]
        );
    }

    #[test]
    fn test_kernel_svr_not_fitted() {
        let svr = KernelSVR::new();
        assert!(svr.predict(&[vec![1.0]]).is_err());
    }

    #[test]
    fn test_kernel_svr_invalid_c() {
        let features = vec![vec![1.0]];
        let target = vec![0.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut svr = KernelSVR::new().c(-1.0);
        assert!(svr.fit(&data).is_err());
    }

    #[test]
    fn test_kernel_svr_gamma_auto() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2.0, 3.0, 4.0, 5.0, 6.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let data = Dataset::new(features, target, vec!["a".into(), "b".into()], "y");

        let mut svr = KernelSVR::new().gamma(Gamma::Auto).c(10.0);
        svr.fit(&data).unwrap();

        match &svr.kernel {
            Kernel::RBF { gamma } => {
                assert!(
                    (*gamma - 0.5).abs() < 1e-10,
                    "Gamma::Auto should give 1/n_features=0.5, got {gamma}",
                );
            }
            other => panic!("expected RBF kernel, got {:?}", other),
        }
    }
}
