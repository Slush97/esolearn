// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU compute backend via scry-gpu (Vulkan compute shaders).
//!
//! Drop-in replacement for the wgpu backend with significantly lower
//! per-dispatch overhead. All data is converted f64 -> f32 for GPU
//! compute, same as the wgpu path.

use std::mem::ManuallyDrop;
use std::sync::{Mutex, OnceLock};

use super::ComputeBackend;

/// Maximum GPU buffer size in bytes (128 MiB).
const MAX_GPU_BUFFER_BYTES: u64 = 128 * 1024 * 1024;

// ---------------------------------------------------------------------------
// GPU context — cached device and compiled kernels
// ---------------------------------------------------------------------------

struct ScryCtx {
    /// Mutex serializes Vulkan dispatches (the Device reuses a single
    /// fence/command buffer internally, so concurrent access is unsafe).
    inner: Mutex<ScryCtxInner>,
}

struct ScryCtxInner {
    dev: ::scry_gpu::Device,
    matmul: ::scry_gpu::Kernel,
    distance: ::scry_gpu::Kernel,
}

/// Leaked on purpose: Vulkan teardown during static destructor ordering
/// causes SIGSEGV. The OS reclaims all GPU resources on process exit.
static GPU_CTX: OnceLock<Option<ManuallyDrop<ScryCtx>>> = OnceLock::new();

fn get_ctx() -> Option<&'static ScryCtx> {
    GPU_CTX
        .get_or_init(|| match init_ctx() {
            Ok(ctx) => Some(ManuallyDrop::new(ctx)),
            Err(e) => {
                eprintln!("[scry-learn] scry-gpu init failed, falling back to CPU: {e}");
                None
            }
        })
        .as_ref()
        .map(|md| &**md)
}

fn init_ctx() -> Result<ScryCtx, String> {
    let dev = ::scry_gpu::Device::auto().map_err(|e| format!("scry-gpu: {e}"))?;
    let matmul = dev
        .compile(::scry_gpu::shaders::matmul::COARSE_64X64)
        .map_err(|e| format!("scry-gpu: matmul shader: {e}"))?;
    let distance = dev
        .compile(::scry_gpu::shaders::distance::PAIRWISE_EUCLIDEAN)
        .map_err(|e| format!("scry-gpu: distance shader: {e}"))?;
    Ok(ScryCtx {
        inner: Mutex::new(ScryCtxInner {
            dev,
            matmul,
            distance,
        }),
    })
}

// ---------------------------------------------------------------------------
// ScryGpuBackend — public API
// ---------------------------------------------------------------------------

/// GPU-accelerated compute backend using scry-gpu (Vulkan compute).
///
/// Uses the coarsened 64x64 matmul shader and pairwise Euclidean distance
/// shader from `scry_gpu::shaders`. Falls back to [`super::CpuBackend`]
/// for small inputs or when the GPU is unavailable.
#[non_exhaustive]
pub struct ScryGpuBackend;

impl ScryGpuBackend {
    /// Create a new scry-gpu backend, initializing the GPU context.
    ///
    /// # Errors
    ///
    /// Returns an error string if no compatible Vulkan device is found.
    pub fn new() -> Result<Self, String> {
        get_ctx().map(|_| Self).ok_or_else(|| "scry-gpu: initialization failed".into())
    }
}

impl ComputeBackend for ScryGpuBackend {
    fn matmul(&self, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);

        if m == 0 || k == 0 || n == 0 {
            return vec![0.0; m * n];
        }

        // Size threshold: GPU overhead not worth it for small matrices
        if m * k * n < 4096 {
            return super::CpuBackend.matmul(a, b, m, k, n);
        }

        // Buffer size guard
        let a_bytes = (m * k * 4) as u64;
        let b_bytes = (k * n * 4) as u64;
        let c_bytes = (m * n * 4) as u64;
        if a_bytes > MAX_GPU_BUFFER_BYTES
            || b_bytes > MAX_GPU_BUFFER_BYTES
            || c_bytes > MAX_GPU_BUFFER_BYTES
        {
            return super::CpuBackend.matmul(a, b, m, k, n);
        }

        let Some(ctx) = get_ctx() else {
            return super::CpuBackend.matmul(a, b, m, k, n);
        };

        let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&v| v as f32).collect();

        let result = (|| {
            let gpu = ctx.inner.lock().ok()?;
            let sa = gpu.dev.upload(&a_f32).ok()?;
            let sb = gpu.dev.upload(&b_f32).ok()?;
            let sc = gpu.dev.alloc::<f32>(m * n).ok()?;

            let dims: [u32; 3] = [m as u32, n as u32, k as u32];
            gpu.dev
                .run_configured(
                    &gpu.matmul,
                    &[&sa, &sb, &sc],
                    [(n as u32).div_ceil(64), (m as u32).div_ceil(64), 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;

            let c_f32: Vec<f32> = sc.download().ok()?;
            Some(c_f32.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>())
        })();

        result.unwrap_or_else(|| super::CpuBackend.matmul(a, b, m, k, n))
    }

    fn xtx_xty(&self, features: &[Vec<f64>], target: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n_samples = target.len();
        let n_features = features.len();

        if n_samples * n_features * n_features < 50_000 {
            return super::CpuBackend.xtx_xty(features, target);
        }

        // Build augmented X matrix: [1, x1, x2, ...] row-major
        let dim = n_features + 1;
        let mut x_f32 = Vec::with_capacity(n_samples * dim);
        for i in 0..n_samples {
            x_f32.push(1.0f32);
            for feat in features {
                x_f32.push(feat[i] as f32);
            }
        }

        // Transpose: Xt is dim x n_samples
        let mut xt_f32 = vec![0.0f32; dim * n_samples];
        for i in 0..n_samples {
            for j in 0..dim {
                xt_f32[j * n_samples + i] = x_f32[i * dim + j];
            }
        }

        // XtX = matmul(Xt, X) via f64 path (delegates back to GPU matmul)
        let xtx = self.matmul(
            &xt_f32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &x_f32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            dim,
            n_samples,
            dim,
        );

        // Xty = matmul(Xt, y)
        let xty = self.matmul(
            &xt_f32.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            target,
            dim,
            n_samples,
            1,
        );

        (xtx, xty)
    }

    fn pairwise_distances_squared(
        &self,
        queries: &[f64],
        train: &[f64],
        n_q: usize,
        n_t: usize,
        dim: usize,
    ) -> Vec<f64> {
        debug_assert_eq!(queries.len(), n_q * dim);
        debug_assert_eq!(train.len(), n_t * dim);

        if n_q == 0 || n_t == 0 || dim == 0 {
            return vec![0.0; n_q * n_t];
        }

        if n_q * n_t < 1024 {
            return super::CpuBackend.pairwise_distances_squared(queries, train, n_q, n_t, dim);
        }

        let out_size = n_q * n_t;
        let out_bytes = (out_size * 4) as u64;
        let q_bytes = (n_q * dim * 4) as u64;
        let t_bytes = (n_t * dim * 4) as u64;
        if out_bytes > MAX_GPU_BUFFER_BYTES
            || q_bytes > MAX_GPU_BUFFER_BYTES
            || t_bytes > MAX_GPU_BUFFER_BYTES
        {
            return super::CpuBackend.pairwise_distances_squared(queries, train, n_q, n_t, dim);
        }

        let Some(ctx) = get_ctx() else {
            return super::CpuBackend.pairwise_distances_squared(queries, train, n_q, n_t, dim);
        };

        let q_f32: Vec<f32> = queries.iter().map(|&v| v as f32).collect();
        let t_f32: Vec<f32> = train.iter().map(|&v| v as f32).collect();

        let result = (|| {
            let gpu = ctx.inner.lock().ok()?;
            let sq = gpu.dev.upload(&q_f32).ok()?;
            let st = gpu.dev.upload(&t_f32).ok()?;
            let sd = gpu.dev.alloc::<f32>(out_size).ok()?;

            let dims: [u32; 3] = [n_q as u32, n_t as u32, dim as u32];
            gpu.dev
                .run_configured(
                    &gpu.distance,
                    &[&sq, &st, &sd],
                    [(out_size as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;

            let d_f32: Vec<f32> = sd.download().ok()?;
            Some(d_f32.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>())
        })();

        result.unwrap_or_else(|| {
            super::CpuBackend.pairwise_distances_squared(queries, train, n_q, n_t, dim)
        })
    }

    fn name(&self) -> &'static str {
        "gpu (scry-gpu)"
    }

    fn build_histograms(
        &self,
        binned: &[Vec<u8>],
        gradients: &[f64],
        hessians: &[f64],
        sample_indices: &[usize],
        n_features: usize,
        n_bins: usize,
    ) -> Vec<Vec<(f64, f64, f64)>> {
        // CPU implementation — histogram shader activation is a future task
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

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gpu() -> Option<ScryGpuBackend> {
        ScryGpuBackend::new().ok()
    }

    #[test]
    fn scry_gpu_matmul_identity() {
        let Some(gpu) = try_gpu() else { return };

        let n = 64;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let b = a.clone();

        let c = gpu.matmul(&a, &b, n, n, n);
        assert_eq!(c.len(), n * n);
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (c[i * n + j] - expected).abs() < 1e-5,
                    "c[{i}][{j}] = {}, expected {expected}",
                    c[i * n + j]
                );
            }
        }
    }

    #[test]
    fn scry_gpu_matmul_known_result() {
        let Some(gpu) = try_gpu() else { return };

        let m = 32;
        let k = 32;
        let n = 32;
        let a: Vec<f64> = (0..m * k).map(|i| (i % 7) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 5) as f64).collect();

        let gpu_result = gpu.matmul(&a, &b, m, k, n);
        let cpu_result = super::super::CpuBackend.matmul(&a, &b, m, k, n);

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1.0,
                "matmul mismatch at {i}: gpu={g}, cpu={c}"
            );
        }
    }

    #[test]
    fn scry_gpu_pairwise_distances() {
        let Some(gpu) = try_gpu() else { return };

        let n_q = 50;
        let n_t = 50;
        let dim = 10;
        let queries: Vec<f64> = (0..n_q * dim).map(|i| (i % 13) as f64 * 0.1).collect();
        let train: Vec<f64> = (0..n_t * dim).map(|i| (i % 11) as f64 * 0.1).collect();

        let gpu_result = gpu.pairwise_distances_squared(&queries, &train, n_q, n_t, dim);
        let cpu_result =
            super::super::CpuBackend.pairwise_distances_squared(&queries, &train, n_q, n_t, dim);

        assert_eq!(gpu_result.len(), cpu_result.len());
        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert!(
                (g - c).abs() < 0.1,
                "distance mismatch at {i}: gpu={g}, cpu={c}"
            );
        }
    }

    #[test]
    fn scry_gpu_backend_name() {
        let Some(gpu) = try_gpu() else { return };
        assert_eq!(gpu.name(), "gpu (scry-gpu)");
    }
}
