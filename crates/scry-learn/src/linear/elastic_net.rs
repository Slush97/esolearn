// SPDX-License-Identifier: MIT OR Apache-2.0
//! ElasticNet regression via coordinate descent (L1 + L2 regularization).
//!
//! Combines L1 (Lasso) and L2 (Ridge) penalties, controlled by `l1_ratio`:
//! - `l1_ratio = 1.0` → pure Lasso
//! - `l1_ratio = 0.0` → pure Ridge
//! - `0 < l1_ratio < 1` → mixed penalty

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::sparse::{CscMatrix, CsrMatrix};

/// ElasticNet regression (mixed L1 + L2 regularization).
///
/// Uses coordinate descent to minimize:
///
/// ```text
/// (1 / 2n) ‖y − Xβ − β₀‖² + α × l1_ratio × ‖β‖₁ + α × (1 − l1_ratio) / 2 × ‖β‖²
/// ```
///
/// # Example
/// ```
/// use scry_learn::dataset::Dataset;
/// use scry_learn::linear::ElasticNet;
///
/// let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
/// let target = vec![2.1, 4.0, 5.9, 8.1, 10.0];
/// let data = Dataset::new(features, target, vec!["x".into()], "y");
///
/// let mut model = ElasticNet::new().alpha(0.1).l1_ratio(0.5);
/// model.fit(&data).unwrap();
/// let preds = model.predict(&[vec![3.0]]).unwrap();
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ElasticNet {
    /// Overall regularization strength.
    alpha: f64,
    /// Mix between L1 and L2 (0.0 = pure Ridge, 1.0 = pure Lasso).
    l1_ratio: f64,
    /// Maximum coordinate descent iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f64,
    /// Learned coefficients (one per feature).
    coefficients: Vec<f64>,
    /// Learned intercept.
    intercept: f64,
    /// Whether the model has been fitted.
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl ElasticNet {
    /// Create a new ElasticNet with default parameters.
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.5,
            max_iter: 1000,
            tol: crate::constants::DEFAULT_TOL,
            coefficients: Vec::new(),
            intercept: 0.0,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the overall regularization strength.
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set the L1/L2 mix ratio (0.0 = Ridge, 1.0 = Lasso).
    pub fn l1_ratio(mut self, r: f64) -> Self {
        self.l1_ratio = r;
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set convergence tolerance.
    pub fn tol(mut self, t: f64) -> Self {
        self.tol = t;
        self
    }

    /// Fit the ElasticNet model using coordinate descent.
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
        if self.alpha < 0.0 {
            return Err(ScryLearnError::InvalidParameter(
                "alpha must be >= 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(ScryLearnError::InvalidParameter(
                "l1_ratio must be in [0, 1]".into(),
            ));
        }

        let y = &data.target;

        let mut beta = vec![0.0; m];
        let mut intercept = y.iter().sum::<f64>() / n as f64;

        // Precompute ‖X_j‖² / n (using column-major data directly).
        let mut col_norm_sq: Vec<f64> = vec![0.0; m];
        for j in 0..m {
            let col = &data.features[j];
            let mut sq = 0.0;
            for &x in col {
                sq += x * x;
            }
            col_norm_sq[j] = sq / n as f64;
        }

        let n_f64 = n as f64;
        let l1_pen = self.alpha * self.l1_ratio;
        let l2_pen = self.alpha * (1.0 - self.l1_ratio);

        // Initialize residuals: r_i = y_i - intercept
        let mut residuals: Vec<f64> = y.iter().map(|&yi| yi - intercept).collect();

        for _iter in 0..self.max_iter {
            let mut max_change = 0.0_f64;

            // Update intercept: shift by mean of residuals.
            let r_mean = residuals.iter().sum::<f64>() / n_f64;
            let new_intercept = intercept + r_mean;
            max_change = max_change.max((new_intercept - intercept).abs());
            for r in &mut residuals {
                *r -= r_mean;
            }
            intercept = new_intercept;

            // Coordinate descent over each feature.
            for j in 0..m {
                if col_norm_sq[j] < crate::constants::NEAR_ZERO {
                    continue;
                }

                let old_beta_j = beta[j];
                let col = &data.features[j];

                // Add back current j contribution to residuals.
                if old_beta_j != 0.0 {
                    for i in 0..n {
                        residuals[i] += col[i] * old_beta_j;
                    }
                }

                // ρ = (1/n) Σ x_ij * r_i
                let mut rho = 0.0;
                for i in 0..n {
                    rho += col[i] * residuals[i];
                }
                rho /= n_f64;

                // ElasticNet update: β_j = S(ρ, l1_pen) / (‖X_j‖²/n + l2_pen)
                let new_beta_j = soft_threshold(rho, l1_pen) / (col_norm_sq[j] + l2_pen);
                max_change = max_change.max((new_beta_j - old_beta_j).abs());
                beta[j] = new_beta_j;

                // Remove new j contribution from residuals.
                if new_beta_j != 0.0 {
                    for i in 0..n {
                        residuals[i] -= col[i] * new_beta_j;
                    }
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        self.coefficients = beta;
        self.intercept = intercept;
        self.fitted = true;
        Ok(())
    }

    /// Predict target values for new samples.
    ///
    /// `features` is row-major: `features[sample_idx][feature_idx]`.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(features
            .iter()
            .map(|row| {
                row.iter()
                    .zip(self.coefficients.iter())
                    .map(|(x, b)| x * b)
                    .sum::<f64>()
                    + self.intercept
            })
            .collect())
    }

    /// Fit on sparse features using coordinate descent.
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
        if self.alpha < 0.0 {
            return Err(ScryLearnError::InvalidParameter(
                "alpha must be >= 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return Err(ScryLearnError::InvalidParameter(
                "l1_ratio must be in [0, 1]".into(),
            ));
        }

        let n_f64 = n as f64;
        let l1_pen = self.alpha * self.l1_ratio;
        let l2_pen = self.alpha * (1.0 - self.l1_ratio);

        let mut beta = vec![0.0; m];
        let mut intercept = target.iter().sum::<f64>() / n_f64;

        let mut col_norm_sq = vec![0.0; m];
        for j in 0..m {
            let mut sq_sum = 0.0;
            for (_, val) in features.col(j).iter() {
                sq_sum += val * val;
            }
            col_norm_sq[j] = sq_sum / n_f64;
        }

        let mut residuals: Vec<f64> = target.iter().map(|&y| y - intercept).collect();

        for _iter in 0..self.max_iter {
            let mut max_change = 0.0_f64;

            let r_mean = residuals.iter().sum::<f64>() / n_f64;
            let new_intercept = intercept + r_mean;
            max_change = max_change.max((new_intercept - intercept).abs());
            for r in &mut residuals {
                *r -= r_mean;
            }
            intercept = new_intercept;

            for j in 0..m {
                if col_norm_sq[j] < crate::constants::NEAR_ZERO {
                    continue;
                }

                let old_beta_j = beta[j];

                if old_beta_j != 0.0 {
                    for (row_idx, val) in features.col(j).iter() {
                        residuals[row_idx] += val * old_beta_j;
                    }
                }

                let mut rho = 0.0;
                for (row_idx, val) in features.col(j).iter() {
                    rho += val * residuals[row_idx];
                }
                rho /= n_f64;

                let new_beta_j = soft_threshold(rho, l1_pen) / (col_norm_sq[j] + l2_pen);
                max_change = max_change.max((new_beta_j - old_beta_j).abs());
                beta[j] = new_beta_j;

                if new_beta_j != 0.0 {
                    for (row_idx, val) in features.col(j).iter() {
                        residuals[row_idx] -= val * new_beta_j;
                    }
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        self.coefficients = beta;
        self.intercept = intercept;
        self.fitted = true;
        Ok(())
    }

    /// Predict from sparse features (CSR format).
    pub fn predict_sparse(&self, features: &CsrMatrix) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok((0..features.n_rows())
            .map(|i| {
                let mut y = self.intercept;
                for (col, val) in features.row(i).iter() {
                    if col < self.coefficients.len() {
                        y += self.coefficients[col] * val;
                    }
                }
                y
            })
            .collect())
    }

    /// Get learned coefficients.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Get learned intercept.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl Default for ElasticNet {
    fn default() -> Self {
        Self::new()
    }
}

/// Soft-thresholding operator: S(z, γ) = sign(z) max(|z| - γ, 0).
#[inline]
fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_net_fit_predict() {
        // y = 2x + 1
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let target = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut en = ElasticNet::new().alpha(0.01).l1_ratio(0.5).max_iter(5000);
        en.fit(&data).unwrap();

        let preds = en.predict(&[vec![3.0]]).unwrap();
        assert!(
            (preds[0] - 7.0).abs() < 0.5,
            "expected ~7.0, got {}",
            preds[0]
        );
    }

    #[test]
    fn test_elastic_net_pure_lasso() {
        // l1_ratio=1.0 should behave like Lasso
        let n = 50;
        let mut rng = crate::rng::FastRng::new(42);
        let mut x1 = Vec::with_capacity(n);
        let mut x2 = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let v1 = rng.f64() * 10.0;
            let v2 = rng.f64() * 10.0;
            x1.push(v1);
            x2.push(v2);
            y.push(3.0 * v1 + 1.0); // x2 is irrelevant
        }

        let data = Dataset::new(vec![x1, x2], y, vec!["x1".into(), "x2".into()], "y");

        let mut en = ElasticNet::new().alpha(0.5).l1_ratio(1.0).max_iter(5000);
        en.fit(&data).unwrap();

        let coefs = en.coefficients();
        assert!(
            coefs[1].abs() < 0.1,
            "x2 should be ~0 with pure lasso, got {}",
            coefs[1]
        );
    }

    #[test]
    fn test_elastic_net_not_fitted() {
        let en = ElasticNet::new();
        assert!(en.predict(&[vec![1.0]]).is_err());
    }

    #[test]
    fn test_sparse_elastic_net_matches_dense() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let target = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let data = Dataset::new(features.clone(), target.clone(), vec!["x".into()], "y");

        let mut en_dense = ElasticNet::new().alpha(0.01).l1_ratio(0.5).max_iter(5000);
        en_dense.fit(&data).unwrap();

        let csc = CscMatrix::from_dense(&features);
        let mut en_sparse = ElasticNet::new().alpha(0.01).l1_ratio(0.5).max_iter(5000);
        en_sparse.fit_sparse(&csc, &target).unwrap();

        assert!(
            (en_dense.coefficients()[0] - en_sparse.coefficients()[0]).abs() < 0.1,
            "Dense={} vs Sparse={}",
            en_dense.coefficients()[0],
            en_sparse.coefficients()[0]
        );
    }
}
