// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compute acceleration backends for linear algebra operations.
//!
//! Provides a [`ComputeBackend`] abstraction with CPU and optional GPU
//! implementations. The GPU backend uses wgpu compute shaders for
//! matrix multiplication and pairwise distance computation.
//!
//! # Runtime auto-detection
//!
//! Use [`auto()`] to get the fastest available backend. With the `gpu`
//! feature enabled, this tries to initialize wgpu and falls back to CPU
//! if no suitable GPU adapter is found.
//!
//! ```ignore
//! use scry_learn::accel;
//!
//! let backend = accel::auto();
//! let c = backend.matmul(&a, &b, m, k, n);
//! ```

mod cpu;
#[cfg(feature = "gpu")]
mod gpu;

pub use cpu::CpuBackend;
#[cfg(feature = "gpu")]
pub use gpu::GpuBackend;

/// Linear algebra compute backend.
///
/// Implementations provide accelerated matrix operations used by
/// model training and prediction.
#[allow(dead_code)]
pub trait ComputeBackend {
    /// Matrix multiply: C = A × B.
    ///
    /// - `a`: row-major `m × k` matrix (length `m * k`)
    /// - `b`: row-major `k × n` matrix (length `k * n`)
    /// - Returns: row-major `m × n` matrix (length `m * n`)
    fn matmul(&self, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64>;

    /// Compute XᵀX for a column-major feature matrix.
    ///
    /// - `features`: slice of column vectors, each of length `n_samples`
    /// - Returns: row-major `(n_features+1) × (n_features+1)` matrix (with intercept column)
    ///
    /// This is the dominant cost in linear regression fitting.
    fn xtx_xty(&self, features: &[Vec<f64>], target: &[f64]) -> (Vec<f64>, Vec<f64>);

    /// Pairwise Euclidean distances between query points and training points.
    ///
    /// - `queries`: row-major `n_q × dim` matrix
    /// - `train`: row-major `n_t × dim` matrix
    /// - Returns: row-major `n_q × n_t` distance matrix (squared distances)
    fn pairwise_distances_squared(
        &self,
        queries: &[f64],
        train: &[f64],
        n_q: usize,
        n_t: usize,
        dim: usize,
    ) -> Vec<f64>;

    /// Compute XᵀX and Xᵀy from a contiguous column-major feature buffer.
    ///
    /// - `data`: flat column-major buffer of length `n_samples * n_features`
    /// - `target`: target vector of length `n_samples`
    /// - `n_samples`: number of rows
    /// - `n_features`: number of feature columns
    /// - Returns: same as [`xtx_xty`] — `(XᵀX, Xᵀy)` with intercept column
    ///
    /// Default implementation rebuilds `Vec<Vec<f64>>` and delegates to [`xtx_xty`].
    /// Backends may override for better cache locality on contiguous data.
    fn xtx_xty_contiguous(
        &self,
        data: &[f64],
        target: &[f64],
        n_samples: usize,
        n_features: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let features: Vec<Vec<f64>> = (0..n_features)
            .map(|j| data[j * n_samples..(j + 1) * n_samples].to_vec())
            .collect();
        self.xtx_xty(&features, target)
    }

    /// Returns the backend name for diagnostics.
    fn name(&self) -> &'static str;

    /// Build gradient/hessian histograms for histogram-based GBT.
    ///
    /// - `binned`: column-major binned features `[n_features][n_samples]` as u8
    /// - `gradients`: per-sample gradients
    /// - `hessians`: per-sample hessians
    /// - `sample_indices`: active sample indices for this node
    /// - `n_features`: number of features
    /// - `n_bins`: max number of bins (typically 256)
    /// - Returns: `[n_features][n_bins]` histogram bins as `(grad_sum, hess_sum, count)`
    fn build_histograms(
        &self,
        binned: &[Vec<u8>],
        gradients: &[f64],
        hessians: &[f64],
        sample_indices: &[usize],
        n_features: usize,
        n_bins: usize,
    ) -> Vec<Vec<(f64, f64, f64)>> {
        // Default CPU implementation
        let mut histograms = vec![vec![(0.0_f64, 0.0_f64, 0.0_f64); n_bins]; n_features];
        for &idx in sample_indices {
            let g = gradients[idx];
            let h = hessians[idx];
            for f in 0..n_features {
                let bin = binned[f][idx] as usize;
                if bin < n_bins {
                    histograms[f][bin].0 += g;
                    histograms[f][bin].1 += h;
                    histograms[f][bin].2 += 1.0;
                }
            }
        }
        histograms
    }
}

/// Get the fastest available compute backend.
///
/// With the `gpu` feature enabled, attempts wgpu initialization
/// and falls back to [`CpuBackend`] if no GPU is available.
/// Without the `gpu` feature, always returns [`CpuBackend`].
pub fn auto() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "gpu")]
    {
        match GpuBackend::new() {
            Ok(gpu) => return Box::new(gpu),
            Err(_e) => {
                // Silently fall back to CPU
            }
        }
    }
    Box::new(CpuBackend)
}

/// Get the CPU compute backend (always available).
#[allow(dead_code)]
pub fn cpu() -> CpuBackend {
    CpuBackend
}
