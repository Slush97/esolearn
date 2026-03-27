// SPDX-License-Identifier: MIT OR Apache-2.0
//! Principal Component Analysis (PCA) — pure-Rust eigendecomposition.
//!
//! Reduces dimensionality by projecting data onto the directions of
//! maximum variance.  Uses Jacobi rotation for eigendecomposition of
//! the covariance matrix — no BLAS / LAPACK required.
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::prelude::*;
//!
//! let mut pca = Pca::with_n_components(2).whiten(true);
//! pca.fit_transform(&mut dataset)?;
//!
//! // Inspect variance explained
//! println!("{:?}", pca.explained_variance_ratio());
//! ```

use crate::dataset::Dataset;
use crate::error::{Result, ScryLearnError};
use crate::preprocess::Transformer;

// ── Public types ──────────────────────────────────────────────────

/// Principal Component Analysis.
///
/// Projects data onto the top-k eigenvectors of the covariance matrix.
/// Optionally whitens the output so each component has unit variance.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Pca {
    n_components: Option<usize>,
    do_whiten: bool,
    // — fitted state —
    mean: Vec<f64>,
    /// Rows = components (top-k eigenvectors), each of length n_features.
    components: Vec<Vec<f64>>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    total_variance: f64,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

// ── Builder ───────────────────────────────────────────────────────

impl Pca {
    /// Create a PCA that retains **all** components.
    pub fn new() -> Self {
        Self {
            n_components: None,
            do_whiten: false,
            mean: Vec::new(),
            components: Vec::new(),
            explained_variance: Vec::new(),
            explained_variance_ratio: Vec::new(),
            total_variance: 0.0,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Create a PCA that retains the top `k` components.
    pub fn with_n_components(k: usize) -> Self {
        Self {
            n_components: Some(k),
            ..Self::new()
        }
    }

    /// Enable whitening (scale components to unit variance).
    pub fn whiten(mut self, yes: bool) -> Self {
        self.do_whiten = yes;
        self
    }

    // ── Accessors ─────────────────────────────────────────────────

    /// Fraction of total variance explained by each retained component.
    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.explained_variance_ratio
    }

    /// Absolute variance (eigenvalue) of each retained component.
    pub fn explained_variance(&self) -> &[f64] {
        &self.explained_variance
    }

    /// Principal axes in feature space — `[n_components][n_features]`.
    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }

    /// Number of components actually retained after fitting.
    pub fn n_components_fitted(&self) -> usize {
        self.components.len()
    }
}

impl Default for Pca {
    fn default() -> Self {
        Self::new()
    }
}

// ── Transformer impl ─────────────────────────────────────────────

impl Transformer for Pca {
    fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if m == 0 {
            return Err(ScryLearnError::InvalidParameter(
                "dataset has no features".into(),
            ));
        }

        let k = self.n_components.unwrap_or(m).min(m);

        // 1. Column means — contiguous column slices from DenseMatrix.
        let mat = data.matrix();
        let mut mean = vec![0.0; m];
        for j in 0..m {
            let col = mat.col(j);
            let s: f64 = col.iter().sum();
            mean[j] = s / n as f64;
        }

        // 2. Covariance matrix (m × m), stored flat row-major.
        //    cov[i*m+j] = (1/(n-1)) * Σ (x_ij - μ_j)(x_ik - μ_k)
        let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
        let mut cov = vec![0.0; m * m];
        for i in 0..m {
            let col_i = mat.col(i);
            for j in i..m {
                let col_j = mat.col(j);
                let mut s = 0.0;
                for s_idx in 0..n {
                    s += (col_i[s_idx] - mean[i]) * (col_j[s_idx] - mean[j]);
                }
                let v = s / denom;
                cov[i * m + j] = v;
                cov[j * m + i] = v;
            }
        }

        // 3. Jacobi eigendecomposition → eigenvalues + eigenvectors.
        let (eigenvalues, eigenvectors) = jacobi_eigen(m, &mut cov);

        // 4. Sort by descending eigenvalue.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| eigenvalues[b].total_cmp(&eigenvalues[a]));

        let total: f64 = eigenvalues.iter().filter(|&&v| v > 0.0).sum();

        self.mean = mean;
        self.total_variance = total;
        self.explained_variance = order[..k]
            .iter()
            .map(|&i| eigenvalues[i].max(0.0))
            .collect();
        self.explained_variance_ratio = if total > crate::constants::NEAR_ZERO {
            self.explained_variance.iter().map(|v| v / total).collect()
        } else {
            vec![0.0; k]
        };
        self.components = order[..k]
            .iter()
            .map(|&i| {
                // eigenvector column i → row in components
                (0..m).map(|r| eigenvectors[r * m + i]).collect()
            })
            .collect();
        self.fitted = true;
        Ok(())
    }

    fn transform(&self, data: &mut Dataset) -> Result<()> {
        const BLOCK: usize = 32;
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n = data.n_samples();
        let m = self.mean.len();
        let k = self.components.len();

        // ── Cache-friendly blocked matrix multiply ──
        //
        // Pre-center into contiguous row-major buffer, transpose components
        // for column-major access, then blocked matmul with slice-based
        // bound elision for SIMD (iter_mut().zip() proves equal lengths →
        // LLVM emits vfmadd231pd).
        let mat = data.matrix();
        let mut centered = vec![0.0; n * m];
        for j in 0..m {
            let col = mat.col(j);
            let mean_j = self.mean[j];
            for i in 0..n {
                centered[i * m + j] = col[i] - mean_j;
            }
        }

        // 2. Transpose components to column-major layout: comp_t[j*k + c]
        //    so that the inner dot-product loop reads contiguously.
        let mut comp_t = vec![0.0; m * k];
        for (c, comp) in self.components.iter().enumerate() {
            for (j, &v) in comp.iter().enumerate() {
                comp_t[j * k + c] = v;
            }
        }

        // 3. Blocked matmul: centered(n×m) × comp_t(m×k) → result(n×k)
        //    Slice extraction elides bounds checks → LLVM vectorises to FMA.
        let mut result = vec![0.0; n * k];
        for ib in (0..n).step_by(BLOCK) {
            let i_end = (ib + BLOCK).min(n);
            for jb in (0..m).step_by(BLOCK) {
                let j_end = (jb + BLOCK).min(m);
                for i in ib..i_end {
                    let r_row = i * k;
                    for j in jb..j_end {
                        let c_val = centered[i * m + j];

                        let r_slice = &mut result[r_row..r_row + k];
                        let c_slice = &comp_t[j * k..j * k + k];

                        // iter_mut().zip() proves equal lengths → no
                        // bounds checks → LLVM vectorises to vfmadd231pd.
                        for (r, &w) in r_slice.iter_mut().zip(c_slice) {
                            *r += c_val * w;
                        }
                    }
                }
            }
        }

        // 4. Apply whitening if enabled, then split into column-major features.
        let mut new_features: Vec<Vec<f64>> = Vec::with_capacity(k);
        for c in 0..k {
            let scale = if self.do_whiten {
                let ev = self.explained_variance[c];
                if ev > crate::constants::NEAR_ZERO {
                    1.0 / ev.sqrt()
                } else {
                    1.0
                }
            } else {
                1.0
            };
            let mut col = Vec::with_capacity(n);
            for i in 0..n {
                col.push(result[i * k + c] * scale);
            }
            new_features.push(col);
        }

        // Replace dataset features.
        data.features = new_features;
        data.feature_names = (0..k).map(|i| format!("PC{}", i + 1)).collect();
        data.invalidate_matrix();
        Ok(())
    }

    fn inverse_transform(&self, data: &mut Dataset) -> Result<()> {
        const BLOCK: usize = 64;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        let n = data.n_samples();
        let k = self.components.len();
        let m = self.mean.len();

        // ── Cache-friendly blocked inverse transform ──
        //
        // Reconstructs X_recon ≈ Y · W + μ using contiguous flat buffers
        // and blocked matmul with slice-based bound elision for SIMD.

        // Step 1: Flatten scores from column-major Vec<Vec<f64>> into
        //         row-major n×k buffer, un-whitening inline.
        let mut scores_flat = vec![0.0; n * k];
        for c in 0..k {
            let col = &data.features[c];
            let scale = if self.do_whiten {
                self.explained_variance[c].sqrt()
            } else {
                1.0
            };
            for i in 0..n {
                scores_flat[i * k + c] = col[i] * scale;
            }
        }

        // Step 2: Flatten components (k × m) — natively row-major,
        //         so copy_from_slice lowers to a fast memcpy.
        let mut comp_flat = vec![0.0; k * m];
        for (c, comp) in self.components.iter().enumerate() {
            comp_flat[c * m..c * m + m].copy_from_slice(comp);
        }

        // Step 3: Pre-fill row-major n×m recon buffer with means.
        //         Avoids a separate addition pass in the hot loop.
        let mut recon_flat = vec![0.0; n * m];
        for i in 0..n {
            recon_flat[i * m..i * m + m].copy_from_slice(&self.mean);
        }

        // Step 4: Blocked DAXPY matmul — scores_flat(n×k) · comp_flat(k×m)
        //         Loop order ib→jb→cb keeps a BLOCK×BLOCK recon tile in L1.
        //         Slice extraction elides bounds checks → LLVM emits SIMD FMA.
        for ib in (0..n).step_by(BLOCK) {
            let i_end = (ib + BLOCK).min(n);
            for jb in (0..m).step_by(BLOCK) {
                let j_end = (jb + BLOCK).min(m);
                for cb in (0..k).step_by(BLOCK) {
                    let c_end = (cb + BLOCK).min(k);

                    for i in ib..i_end {
                        for c in cb..c_end {
                            let y_val = scores_flat[i * k + c];

                            let recon_slice = &mut recon_flat[i * m + jb..i * m + j_end];
                            let comp_slice = &comp_flat[c * m + jb..c * m + j_end];

                            // iter_mut().zip() proves equal lengths → no
                            // bounds checks → LLVM vectorises to vfmadd231pd.
                            for (r, &w) in recon_slice.iter_mut().zip(comp_slice) {
                                *r += y_val * w;
                            }
                        }
                    }
                }
            }
        }

        // Step 5: Blocked scatter from row-major flat → column-major Vec<Vec<f64>>.
        let mut reconstructed: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
        for ib in (0..n).step_by(BLOCK) {
            let i_end = (ib + BLOCK).min(n);
            for jb in (0..m).step_by(BLOCK) {
                let j_end = (jb + BLOCK).min(m);
                for j in jb..j_end {
                    let col = &mut reconstructed[j];
                    for i in ib..i_end {
                        col[i] = recon_flat[i * m + j];
                    }
                }
            }
        }

        data.features = reconstructed;
        data.feature_names = (0..m).map(|i| format!("x{i}")).collect();
        data.invalidate_matrix();
        Ok(())
    }
}

// ── Jacobi eigendecomposition ─────────────────────────────────────
//
// For a real symmetric n×n matrix, iterates 2×2 rotations to
// diagonalise it.  Returns (eigenvalues, eigenvectors_flat).
// eigenvectors_flat is row-major n×n where column j is eigenvector j.

fn jacobi_eigen(n: usize, a: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    // Identity matrix for eigenvectors (row-major).
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_sweeps = crate::constants::JACOBI_MAX_SWEEPS;
    let tol = crate::constants::JACOBI_TOL;

    for _sweep in 0..max_sweeps {
        // Off-diagonal Frobenius norm.
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off < tol {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < crate::constants::NEAR_ZERO {
                    continue;
                }

                let diff = a[q * n + q] - a[p * n + p];
                let t = if diff.abs() < crate::constants::NEAR_ZERO {
                    // θ = π/4 → t = 1
                    1.0
                } else {
                    let tau = diff / (2.0 * apq);
                    // Pick the smaller root for stability.
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update matrix A.
                let tau_val = s / (1.0 + c);

                a[p * n + p] -= t * apq;
                a[q * n + q] += t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Rotate rows/columns (only upper triangle elements are needed
                // but we keep it symmetric for simplicity).
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = arp - s * (arq + tau_val * arp);
                    a[p * n + r] = a[r * n + p];
                    a[r * n + q] = arq + s * (arp - tau_val * arq);
                    a[q * n + r] = a[r * n + q];
                }

                // Rotate eigenvector columns.
                for r in 0..n {
                    let vp = v[r * n + p];
                    let vq = v[r * n + q];
                    v[r * n + p] = vp - s * (vq + tau_val * vp);
                    v[r * n + q] = vq + s * (vp - tau_val * vq);
                }
            }
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn iris_4d_subset() -> Dataset {
        // 12 samples from Iris (3 classes × 4 samples), 4 features.
        Dataset::new(
            vec![
                vec![5.1, 4.9, 4.7, 4.6, 7.0, 6.4, 6.9, 5.5, 6.3, 5.8, 7.1, 6.3],
                vec![3.5, 3.0, 3.2, 3.1, 3.2, 3.2, 3.1, 2.3, 3.3, 2.7, 3.0, 2.9],
                vec![1.4, 1.4, 1.3, 1.5, 4.7, 4.5, 4.9, 4.0, 6.0, 5.1, 5.9, 5.6],
                vec![0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 2.5, 1.9, 2.1, 1.8],
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            vec![
                "sepal_length".into(),
                "sepal_width".into(),
                "petal_length".into(),
                "petal_width".into(),
            ],
            "species",
        )
    }

    #[test]
    fn pca_identity_no_reduction() {
        let ds = iris_4d_subset();
        let mut pca = Pca::new();
        pca.fit(&ds).unwrap();
        assert_eq!(pca.n_components_fitted(), 4);
    }

    #[test]
    fn pca_variance_explained_sums_to_one() {
        let ds = iris_4d_subset();
        let mut pca = Pca::new();
        pca.fit(&ds).unwrap();
        let sum: f64 = pca.explained_variance_ratio().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "variance ratios should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn pca_reduces_dimensions() {
        let mut ds = iris_4d_subset();
        let mut pca = Pca::with_n_components(2);
        pca.fit_transform(&mut ds).unwrap();
        assert_eq!(ds.n_features(), 2);
        assert_eq!(ds.feature_names[0], "PC1");
        assert_eq!(ds.feature_names[1], "PC2");
    }

    #[test]
    fn pca_roundtrip_inverse() {
        let original = iris_4d_subset();
        let mut ds = original.clone();
        let mut pca = Pca::new(); // keep all → perfect roundtrip.
        pca.fit_transform(&mut ds).unwrap();
        pca.inverse_transform(&mut ds).unwrap();

        for j in 0..original.n_features() {
            for i in 0..original.n_samples() {
                assert!(
                    (ds.features[j][i] - original.features[j][i]).abs() < 1e-6,
                    "roundtrip mismatch at feature {j}, sample {i}: {} vs {}",
                    ds.features[j][i],
                    original.features[j][i],
                );
            }
        }
    }

    #[test]
    fn pca_whiten_unit_variance() {
        let mut ds = iris_4d_subset();
        let mut pca = Pca::with_n_components(2).whiten(true);
        pca.fit_transform(&mut ds).unwrap();

        // Each component should have variance ≈ 1.
        for j in 0..2 {
            let col = &ds.features[j];
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (col.len() - 1) as f64;
            assert!(
                (var - 1.0).abs() < 0.15,
                "whitened PC{} variance should be ~1.0, got {var}",
                j + 1,
            );
        }
    }

    #[test]
    fn pca_not_fitted_error() {
        let pca = Pca::new();
        let mut ds = iris_4d_subset();
        assert!(pca.transform(&mut ds).is_err());
    }

    #[test]
    fn pca_empty_dataset_error() {
        let ds = Dataset::new(vec![vec![]], vec![], vec!["x".into()], "y");
        let mut pca = Pca::new();
        assert!(pca.fit(&ds).is_err());
    }

    #[test]
    fn pca_components_orthogonal() {
        let ds = iris_4d_subset();
        let mut pca = Pca::new();
        pca.fit(&ds).unwrap();

        let comps = pca.components();
        let k = comps.len();
        for i in 0..k {
            for j in (i + 1)..k {
                let dot: f64 = comps[i]
                    .iter()
                    .zip(comps[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(
                    dot.abs() < 1e-6,
                    "components {i} and {j} should be orthogonal, dot = {dot}"
                );
            }
            // Unit norm.
            let norm: f64 = comps[i].iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "component {i} should have unit norm, got {norm}"
            );
        }
    }
}
