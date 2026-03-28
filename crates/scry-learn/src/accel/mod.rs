// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compute acceleration backends for linear algebra operations.
//!
//! Provides a [`ComputeBackend`] abstraction with CPU and optional GPU
//! implementations. The `scry-gpu` feature enables Vulkan compute
//! acceleration for matrix multiply and pairwise distance.
//!
//! # Runtime auto-detection
//!
//! Use [`auto()`] to get the fastest available backend. With the `scry-gpu`
//! feature, this uses Vulkan compute via scry-gpu. Without it, returns
//! [`CpuBackend`].
//!
//! ```ignore
//! use scry_learn::accel;
//!
//! let backend = accel::auto();
//! let c = backend.matmul(&a, &b, m, k, n);
//! ```

mod cpu;
#[cfg(feature = "scry-gpu")]
mod scry_gpu;

#[cfg(feature = "scry-gpu")]
pub use self::scry_gpu::ScryGpuBackend;
pub use cpu::CpuBackend;

/// A tensor that may live on GPU or CPU.
///
/// Used by the `gpu_*` methods on [`ComputeBackend`] to keep data on-device
/// across multiple operations (matmul, bias add, activation) without
/// round-tripping through CPU after every op.
///
/// The `Cpu` variant is the fallback when no GPU is available. The `Gpu`
/// variant holds a `scry_gpu::Buffer<f32>` that stays device-resident.
pub enum GpuTensor {
    /// CPU fallback: data as f64, with shape (rows, cols).
    Cpu(Vec<f64>, usize, usize),
    /// GPU-resident f32 buffer with shape (rows, cols).
    #[cfg(feature = "scry-gpu")]
    Gpu(::scry_gpu::Buffer<f32>, usize, usize),
}

impl GpuTensor {
    /// Shape as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Cpu(_, r, c) => (*r, *c),
            #[cfg(feature = "scry-gpu")]
            Self::Gpu(_, r, c) => (*r, *c),
        }
    }

    /// Convert to a `Cpu` variant tensor (download if GPU-resident).
    #[allow(dead_code)]
    pub fn to_cpu_tensor(&self) -> Self {
        let (rows, cols) = self.shape();
        Self::Cpu(self.to_cpu(), rows, cols)
    }

    /// Download to CPU f64 vec. For `Cpu` variant, clones the data.
    /// For `Gpu` variant, downloads from device and converts f32 → f64.
    pub fn to_cpu(&self) -> Vec<f64> {
        match self {
            Self::Cpu(data, _, _) => data.clone(),
            #[cfg(feature = "scry-gpu")]
            Self::Gpu(buf, _, _) => buf
                .download()
                .unwrap_or_default()
                .iter()
                .map(|&v| f64::from(v))
                .collect(),
        }
    }
}

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

    // ── GPU-resident tensor operations ──
    //
    // These methods keep data on-device across multiple operations.
    // Default implementations fall back to CPU (GpuTensor::Cpu variant).
    // GPU backends override to use real device buffers.

    /// Upload f64 data to a persistent GPU buffer (f32 on device).
    ///
    /// Returns a [`GpuTensor`] that stays on the GPU until explicitly
    /// downloaded. Default: wraps in `GpuTensor::Cpu`.
    fn gpu_upload(&self, data: &[f64], rows: usize, cols: usize) -> GpuTensor {
        GpuTensor::Cpu(data.to_vec(), rows, cols)
    }

    /// GPU-to-GPU matrix multiply. Result stays on device.
    ///
    /// `a` is `m × k`, `b` is `k × n`, result is `m × n`.
    fn gpu_matmul(&self, a: &GpuTensor, b: &GpuTensor, m: usize, k: usize, n: usize) -> GpuTensor {
        let a_data = a.to_cpu();
        let b_data = b.to_cpu();
        GpuTensor::Cpu(self.matmul(&a_data, &b_data, m, k, n), m, n)
    }

    /// GPU-resident bias add: `z[i,j] += bias[j]`.
    ///
    /// `z` is `rows × cols`, `bias` is `1 × cols`.
    fn gpu_bias_add(&self, z: &GpuTensor, bias: &GpuTensor, rows: usize, cols: usize) -> GpuTensor {
        let mut data = z.to_cpu();
        let b = bias.to_cpu();
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] += b[j];
            }
        }
        GpuTensor::Cpu(data, rows, cols)
    }

    /// GPU-resident ReLU activation.
    fn gpu_relu(&self, x: &GpuTensor) -> GpuTensor {
        let (rows, cols) = x.shape();
        let mut data = x.to_cpu();
        for v in &mut data {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        GpuTensor::Cpu(data, rows, cols)
    }

    /// GPU-resident tanh activation.
    fn gpu_tanh(&self, x: &GpuTensor) -> GpuTensor {
        let (rows, cols) = x.shape();
        let mut data = x.to_cpu();
        for v in &mut data {
            *v = v.tanh();
        }
        GpuTensor::Cpu(data, rows, cols)
    }

    /// GPU-resident sigmoid activation.
    fn gpu_sigmoid(&self, x: &GpuTensor) -> GpuTensor {
        let (rows, cols) = x.shape();
        let mut data = x.to_cpu();
        for v in &mut data {
            *v = if *v >= 0.0 {
                1.0 / (1.0 + (-*v).exp())
            } else {
                let ex = v.exp();
                ex / (1.0 + ex)
            };
        }
        GpuTensor::Cpu(data, rows, cols)
    }

    /// Download GPU tensor to CPU f64 vec.
    fn gpu_download(&self, t: &GpuTensor) -> Vec<f64> {
        t.to_cpu()
    }

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
/// With the `scry-gpu` feature enabled, attempts Vulkan GPU initialization
/// and falls back to [`CpuBackend`] if no GPU is available.
pub fn auto() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "scry-gpu")]
    {
        match ScryGpuBackend::new() {
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
