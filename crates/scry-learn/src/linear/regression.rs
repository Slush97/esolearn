// SPDX-License-Identifier: MIT OR Apache-2.0
//! Linear regression via OLS (Ordinary Least Squares).

use crate::accel;
use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::sparse::{CscMatrix, CsrMatrix};

/// Solver strategy for linear regression.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum LinRegSolver {
    /// Normal equation: (X^T X + aI)^-1 X^T y. Fast but numerically fragile.
    Normal,
    /// QR decomposition. More robust than Normal, faster than SVD.
    Qr,
    /// SVD (pseudoinverse). Most robust, handles rank-deficient and wide matrices.
    Svd,
    /// Auto: use Normal for well-conditioned problems, fall back to SVD otherwise.
    #[default]
    Auto,
}

/// Linear regression model.
///
/// Uses the **OLS** closed-form normal equations solution by default:
/// `β = (XᵀX + αI)⁻¹ Xᵀy`. Set `alpha > 0` for Ridge (L2) regularization.
///
/// Alternative solvers (QR, SVD) provide better numerical stability for
/// ill-conditioned or rank-deficient problems.
///
/// When the `gpu` feature is enabled and the dataset is large enough,
/// the XᵀX/Xᵀy computation is automatically offloaded to GPU compute shaders.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LinearRegression {
    /// Learned coefficients (one per feature).
    coefficients: Vec<f64>,
    /// Learned intercept (bias term).
    intercept: f64,
    /// L2 regularization strength (0.0 = OLS, >0 = Ridge).
    alpha: f64,
    /// Solver strategy.
    #[cfg_attr(feature = "serde", serde(skip))]
    solver: LinRegSolver,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl LinearRegression {
    /// Create a new linear regression model.
    pub fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            intercept: 0.0,
            alpha: 0.0,
            solver: LinRegSolver::Auto,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set L2 regularization strength (Ridge regression).
    pub fn alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self
    }

    /// Set the solver strategy.
    pub fn solver(mut self, s: LinRegSolver) -> Self {
        self.solver = s;
        self
    }

    /// Train the model on the given dataset.
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

        match &self.solver {
            LinRegSolver::Normal => self.fit_normal(data),
            LinRegSolver::Qr => self.fit_qr(data),
            LinRegSolver::Svd => self.fit_svd(data),
            LinRegSolver::Auto => {
                // Normal equations are fast (~350µs). Fall back to SVD for
                // underdetermined systems or singular matrices.
                if m >= n {
                    return self.fit_svd(data);
                }
                match self.fit_normal(data) {
                    Ok(()) => Ok(()),
                    Err(_) => self.fit_svd(data),
                }
            }
        }
    }

    /// Normal equation solver (existing code path).
    fn fit_normal(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        let m = data.n_features();
        let dim = m + 1;

        let backend = accel::auto();
        let mat = data.matrix();
        let (mut xtx, mut xty) = backend.xtx_xty_contiguous(mat.as_slice(), &data.target, n, m);

        for j in 1..dim {
            xtx[j * dim + j] += self.alpha;
        }

        let beta = solve_linear(dim, &mut xtx, &mut xty)?;

        self.intercept = beta[0];
        self.coefficients = beta[1..].to_vec();
        self.fitted = true;
        Ok(())
    }

    /// Build augmented column-major feature matrix [1, x1, x2, ...].
    fn build_augmented(data: &Dataset) -> (Vec<f64>, usize, usize) {
        let n = data.n_samples();
        let m = data.n_features();
        let dim = m + 1;
        let mat = data.matrix();
        let mut x = vec![0.0; n * dim];
        for i in 0..n {
            x[i] = 1.0;
        }
        for j in 0..m {
            let offset = (j + 1) * n;
            x[offset..offset + n].copy_from_slice(mat.col(j));
        }
        (x, n, dim)
    }

    /// Build augmented matrix with Ridge regularization rows appended.
    fn build_regularized(data: &Dataset, alpha: f64) -> (Vec<f64>, Vec<f64>, usize, usize) {
        let n = data.n_samples();
        let m = data.n_features();
        let dim = m + 1;
        let mat = data.matrix();
        let sqrt_a = alpha.sqrt();
        let aug_rows = n + m;
        let mut x_aug = vec![0.0; aug_rows * dim];
        let mut y_aug = vec![0.0; aug_rows];

        for i in 0..n {
            x_aug[i] = 1.0;
        }
        for j in 0..m {
            let offset = (j + 1) * aug_rows;
            x_aug[offset..offset + n].copy_from_slice(mat.col(j));
        }
        y_aug[..n].copy_from_slice(&data.target);

        for j in 0..m {
            x_aug[(j + 1) * aug_rows + n + j] = sqrt_a;
        }

        (x_aug, y_aug, aug_rows, dim)
    }

    /// QR decomposition solver.
    fn fit_qr(&mut self, data: &Dataset) -> Result<()> {
        if self.alpha > 0.0 {
            let (x_aug, y_aug, aug_rows, dim) = Self::build_regularized(data, self.alpha);
            let beta = super::qr::qr_solve(&x_aug, &y_aug, aug_rows, dim)?;
            self.intercept = beta[0];
            self.coefficients = beta[1..].to_vec();
        } else {
            let (x, n, dim) = Self::build_augmented(data);
            let beta = super::qr::qr_solve(&x, &data.target, n, dim)?;
            self.intercept = beta[0];
            self.coefficients = beta[1..].to_vec();
        }

        self.fitted = true;
        Ok(())
    }

    /// SVD solver.
    fn fit_svd(&mut self, data: &Dataset) -> Result<()> {
        if self.alpha > 0.0 {
            let (x_aug, y_aug, aug_rows, dim) = Self::build_regularized(data, self.alpha);
            let result = super::svd::svd_solve(&x_aug, &y_aug, aug_rows, dim)?;
            self.intercept = result.coefficients[0];
            self.coefficients = result.coefficients[1..].to_vec();
        } else {
            let (x, n, dim) = Self::build_augmented(data);
            let result = super::svd::svd_solve(&x, &data.target, n, dim)?;
            self.intercept = result.coefficients[0];
            self.coefficients = result.coefficients[1..].to_vec();
        }

        self.fitted = true;
        Ok(())
    }

    /// Predict target values.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(features
            .iter()
            .map(|row| {
                let mut y = self.intercept;
                for (j, &coeff) in self.coefficients.iter().enumerate() {
                    if j < row.len() {
                        y += coeff * row[j];
                    }
                }
                y
            })
            .collect())
    }

    /// Fit on sparse features (CSC format for column-oriented access).
    ///
    /// Builds XᵀX and Xᵀy by iterating only over non-zero entries.
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

        let dim = m + 1; // intercept + features

        // Build XᵀX (dim×dim row-major) and Xᵀy (dim) with intercept column.
        let mut xtx = vec![0.0; dim * dim];
        let mut xty = vec![0.0; dim];

        // Intercept-intercept: XᵀX[0][0] = n
        xtx[0] = n as f64;

        // Intercept-target: Xᵀy[0] = Σ y_i
        xty[0] = target.iter().sum();

        // Intercept-feature cross terms: XᵀX[0][j+1] = XᵀX[j+1][0] = Σ x_ij
        for j in 0..m {
            let col = features.col(j);
            let sum: f64 = col.iter().map(|(_, v)| v).sum();
            xtx[j + 1] = sum;
            xtx[(j + 1) * dim] = sum;

            // Xᵀy[j+1] = Σ x_ij * y_i
            let mut dot = 0.0;
            for (row_idx, val) in col.iter() {
                dot += val * target[row_idx];
            }
            xty[j + 1] = dot;
        }

        // Feature-feature: XᵀX[i+1][j+1] = Σ_k x_ki * x_kj (only non-zero entries)
        // For efficiency, use scatter approach: for each column j, scatter into a dense vector,
        // then dot with each column i.
        let mut dense_col = vec![0.0; n];
        for j in 0..m {
            // Scatter column j into dense.
            for (row_idx, val) in features.col(j).iter() {
                dense_col[row_idx] = val;
            }

            // Diagonal: XᵀX[j+1][j+1]
            let mut diag = 0.0;
            for (row_idx, val) in features.col(j).iter() {
                diag += val * dense_col[row_idx];
            }
            xtx[(j + 1) * dim + j + 1] = diag;

            // Off-diagonal with columns i < j.
            for i in 0..j {
                let mut dot = 0.0;
                for (row_idx, val) in features.col(i).iter() {
                    dot += val * dense_col[row_idx];
                }
                xtx[(i + 1) * dim + j + 1] = dot;
                xtx[(j + 1) * dim + i + 1] = dot;
            }

            // Clear dense_col.
            for (row_idx, _) in features.col(j).iter() {
                dense_col[row_idx] = 0.0;
            }
        }

        // Add Ridge regularization.
        for j in 1..dim {
            xtx[j * dim + j] += self.alpha;
        }

        let beta = solve_linear(dim, &mut xtx, &mut xty)?;
        self.intercept = beta[0];
        self.coefficients = beta[1..].to_vec();
        self.fitted = true;
        Ok(())
    }

    /// Predict from sparse features (CSR format for row-oriented access).
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

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Gauss-Jordan elimination for Ax = b.
fn solve_linear(n: usize, a: &mut [f64], b: &mut [f64]) -> Result<Vec<f64>> {
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let val = a[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < crate::constants::SINGULAR_THRESHOLD {
            return Err(ScryLearnError::InvalidParameter(
                "singular matrix — features may be linearly dependent".into(),
            ));
        }

        if max_row != col {
            for k in 0..n {
                a.swap(col * n + k, max_row * n + k);
            }
            b.swap(col, max_row);
        }

        let pivot = a[col * n + col];
        for k in col..n {
            a[col * n + k] /= pivot;
        }
        b[col] /= pivot;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            for k in col..n {
                a[row * n + k] -= factor * a[col * n + k];
            }
            b[row] -= factor * b[col];
        }
    }

    Ok(b.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_y_equals_x() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr = LinearRegression::new();
        lr.fit(&data).unwrap();

        assert!(
            (lr.coefficients()[0] - 2.0).abs() < 1e-6,
            "coefficient should be ~2.0, got {}",
            lr.coefficients()[0]
        );
        assert!(
            (lr.intercept() - 3.0).abs() < 1e-6,
            "intercept should be ~3.0, got {}",
            lr.intercept()
        );
    }

    #[test]
    fn test_ridge_regression() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr = LinearRegression::new().alpha(1.0);
        lr.fit(&data).unwrap();

        assert!(lr.coefficients()[0] < 2.0);
        assert!(lr.coefficients()[0] > 1.0);
    }

    #[test]
    fn test_svd_solver_matches_normal() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr_normal = LinearRegression::new();
        lr_normal.fit(&data).unwrap();

        let mut lr_svd = LinearRegression::new().solver(LinRegSolver::Svd);
        lr_svd.fit(&data).unwrap();

        assert!(
            (lr_normal.coefficients()[0] - lr_svd.coefficients()[0]).abs() < 1e-6,
            "Normal={} vs SVD={}",
            lr_normal.coefficients()[0],
            lr_svd.coefficients()[0]
        );
        assert!(
            (lr_normal.intercept() - lr_svd.intercept()).abs() < 1e-6,
            "Normal intercept={} vs SVD={}",
            lr_normal.intercept(),
            lr_svd.intercept()
        );
    }

    #[test]
    fn test_qr_solver_matches_normal() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr_normal = LinearRegression::new();
        lr_normal.fit(&data).unwrap();

        let mut lr_qr = LinearRegression::new().solver(LinRegSolver::Qr);
        lr_qr.fit(&data).unwrap();

        assert!(
            (lr_normal.coefficients()[0] - lr_qr.coefficients()[0]).abs() < 1e-6,
            "Normal={} vs QR={}",
            lr_normal.coefficients()[0],
            lr_qr.coefficients()[0]
        );
        assert!(
            (lr_normal.intercept() - lr_qr.intercept()).abs() < 1e-6,
            "Normal intercept={} vs QR={}",
            lr_normal.intercept(),
            lr_qr.intercept()
        );
    }

    #[test]
    fn test_svd_handles_ill_conditioned() {
        let n = 5;
        let mut features = vec![vec![0.0; n]; n];
        for j in 0..n {
            for i in 0..n {
                features[j][i] = 1.0 / (i + j + 1) as f64;
            }
        }
        let true_beta = vec![1.0; n];
        let target: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| features[j][i] * true_beta[j]).sum())
            .collect();
        let names: Vec<String> = (0..n).map(|j| format!("f{j}")).collect();
        let data = Dataset::new(features, target, names, "y");

        let mut lr = LinearRegression::new().solver(LinRegSolver::Svd);
        lr.fit(&data).unwrap();

        for (i, &c) in lr.coefficients().iter().enumerate() {
            assert!(
                (c - 1.0).abs() < 0.5,
                "SVD Hilbert coeff[{}] = {}, expected ~1.0",
                i,
                c
            );
        }
    }

    #[test]
    fn test_ridge_with_svd() {
        let features = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr_normal = LinearRegression::new().alpha(1.0);
        lr_normal.fit(&data).unwrap();

        let mut lr_svd = LinearRegression::new().alpha(1.0).solver(LinRegSolver::Svd);
        lr_svd.fit(&data).unwrap();

        assert!(
            (lr_normal.coefficients()[0] - lr_svd.coefficients()[0]).abs() < 0.1,
            "Ridge Normal={} vs SVD={}",
            lr_normal.coefficients()[0],
            lr_svd.coefficients()[0]
        );
    }

    #[test]
    fn test_auto_solver() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr = LinearRegression::new().solver(LinRegSolver::Auto);
        lr.fit(&data).unwrap();

        assert!(
            (lr.coefficients()[0] - 2.0).abs() < 1e-6,
            "Auto solver coefficient should be ~2.0, got {}",
            lr.coefficients()[0]
        );
    }

    #[test]
    fn test_sparse_fit_matches_dense() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features.clone(), target.clone(), vec!["x".into()], "y");

        let mut lr_dense = LinearRegression::new();
        lr_dense.fit(&data).unwrap();

        let csc = CscMatrix::from_dense(&features);
        let mut lr_sparse = LinearRegression::new();
        lr_sparse.fit_sparse(&csc, &target).unwrap();

        assert!(
            (lr_dense.coefficients()[0] - lr_sparse.coefficients()[0]).abs() < 1e-6,
            "Dense={} vs Sparse={}",
            lr_dense.coefficients()[0],
            lr_sparse.coefficients()[0]
        );
        assert!(
            (lr_dense.intercept() - lr_sparse.intercept()).abs() < 1e-6,
            "Dense intercept={} vs Sparse={}",
            lr_dense.intercept(),
            lr_sparse.intercept()
        );
    }

    #[test]
    fn test_sparse_predict_matches_dense() {
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let data = Dataset::new(features, target, vec!["x".into()], "y");

        let mut lr = LinearRegression::new();
        lr.fit(&data).unwrap();

        let test_rows = vec![vec![3.0], vec![10.0], vec![15.0]];
        let preds_dense = lr.predict(&test_rows).unwrap();

        let csr = CsrMatrix::from_dense(&test_rows);
        let preds_sparse = lr.predict_sparse(&csr).unwrap();

        for (d, s) in preds_dense.iter().zip(preds_sparse.iter()) {
            assert!((d - s).abs() < 1e-6, "Dense pred={d} vs Sparse pred={s}");
        }
    }

    #[test]
    fn test_auto_dispatch_sparse_fit() {
        // Create a sparse Dataset, call fit() (not fit_sparse), verify it works.
        let features = vec![(0..20).map(|i| i as f64).collect::<Vec<f64>>()];
        let target: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let csc = CscMatrix::from_dense(&features);
        let data = crate::dataset::Dataset::from_sparse(csc, target, vec!["x".into()], "y");

        let mut lr = LinearRegression::new();
        lr.fit(&data).unwrap();

        assert!(
            (lr.coefficients()[0] - 2.0).abs() < 1e-4,
            "Auto-dispatched sparse fit: coefficient should be ~2.0, got {}",
            lr.coefficients()[0]
        );
    }
}

#[cfg(all(test, feature = "scry-gpu"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn gpu_linear_regression_matches_cpu() {
        let n = 500;
        let m = 50;
        let mut features = Vec::with_capacity(m);
        for j in 0..m {
            let col: Vec<f64> = (0..n).map(|i| ((i * (j + 1)) % 97) as f64 * 0.1).collect();
            features.push(col);
        }
        let target: Vec<f64> = (0..n)
            .map(|i| features[0][i] * 2.0 + features[1][i] * 0.5 + features[2][i] + 3.0)
            .collect();
        let names: Vec<String> = (0..m).map(|j| format!("f{j}")).collect();
        let data = Dataset::new(features, target, names, "y");

        let mut lr = LinearRegression::new().alpha(0.1);
        lr.fit(&data).unwrap();

        assert!(lr.coefficients().len() == m);
        let preds = lr.predict(&[vec![1.0; m]]).unwrap();
        assert!(
            preds[0].is_finite(),
            "prediction must be finite, got {}",
            preds[0]
        );
    }
}
