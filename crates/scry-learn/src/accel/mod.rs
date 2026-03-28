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

// ── Batch dispatch types ──

/// Activation function tag for batched GPU operations.
///
/// Mirrors [`crate::neural::Activation`] without creating a dependency
/// from the accel module on the neural module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuActivation {
    /// Identity (pass-through).
    Identity,
    /// ReLU: max(0, x).
    Relu,
    /// Logistic sigmoid.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
}

/// Per-layer descriptor for a batched GPU forward pass.
pub struct GpuForwardLayer<'a> {
    /// Transposed weight matrix W^T `[in_size, out_size]` on GPU.
    pub weights_t: &'a GpuTensor,
    /// Bias vector `[1, out_size]` on GPU.
    pub bias: &'a GpuTensor,
    /// Activation function for this layer.
    pub activation: GpuActivation,
    /// Input dimension.
    pub in_size: usize,
    /// Output dimension.
    pub out_size: usize,
}

/// Per-layer cache produced by a batched forward pass, consumed by backward.
pub struct GpuLayerCache {
    /// Cached input: `[batch, in_size]`.
    pub input: GpuTensor,
    /// Cached pre-activation (after bias add): `[batch, out_size]`.
    pub z: GpuTensor,
    /// Cached post-activation: `[batch, out_size]`.
    pub a: GpuTensor,
    /// Batch size.
    pub batch: usize,
}

/// Per-layer descriptor for a batched GPU backward pass.
///
/// Layers are passed in reverse order (last network layer first).
pub struct GpuBackwardLayer<'a> {
    /// Pre-activation cache: `[batch, out_size]`.
    pub z_cache: &'a GpuTensor,
    /// Post-activation cache: `[batch, out_size]`.
    pub a_cache: &'a GpuTensor,
    /// Input cache: `[batch, in_size]`.
    pub input_cache: &'a GpuTensor,
    /// Original weight matrix W `[out_size, in_size]` on GPU.
    pub weights_w: &'a GpuTensor,
    /// Activation function for this layer.
    pub activation: GpuActivation,
    /// Batch size.
    pub batch: usize,
    /// Input dimension.
    pub in_size: usize,
    /// Output dimension.
    pub out_size: usize,
}

/// Default implementation of batched GPU forward pass using individual
/// `gpu_*` calls. Used as the trait default and as fallback when the
/// batched device path is unavailable (e.g. cuBLAS matmul).
pub(crate) fn gpu_forward_batch_default<B: ComputeBackend + ?Sized>(
    backend: &B,
    input: &GpuTensor,
    batch: usize,
    layers: &[GpuForwardLayer<'_>],
    training: bool,
) -> (GpuTensor, Vec<GpuLayerCache>) {
    let mut caches = Vec::new();
    let mut current = backend.gpu_copy(input);

    for layer in layers {
        let layer_input = current;

        let z = backend.gpu_matmul(
            &layer_input,
            layer.weights_t,
            batch,
            layer.in_size,
            layer.out_size,
        );
        let z = backend.gpu_bias_add(&z, layer.bias, batch, layer.out_size);

        let a = match layer.activation {
            GpuActivation::Identity => {
                if training {
                    let z_cache = backend.gpu_copy(&z);
                    let a_cache = backend.gpu_copy(&z);
                    caches.push(GpuLayerCache {
                        input: layer_input,
                        z: z_cache,
                        a: a_cache,
                        batch,
                    });
                }
                z
            }
            act => {
                let a = match act {
                    GpuActivation::Relu => backend.gpu_relu(&z),
                    GpuActivation::Sigmoid => backend.gpu_sigmoid(&z),
                    GpuActivation::Tanh => backend.gpu_tanh(&z),
                    GpuActivation::Identity => unreachable!(),
                };
                if training {
                    caches.push(GpuLayerCache {
                        input: layer_input,
                        z,
                        a: backend.gpu_copy(&a),
                        batch,
                    });
                }
                a
            }
        };

        current = a;
    }

    (current, caches)
}

/// Default implementation of batched GPU backward pass using individual
/// `gpu_*` calls.
pub(crate) fn gpu_backward_batch_default<B: ComputeBackend + ?Sized>(
    backend: &B,
    grad_output: &GpuTensor,
    layers: &[GpuBackwardLayer<'_>],
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut grads = Vec::with_capacity(layers.len());
    let mut current_grad = backend.gpu_copy(grad_output);

    for layer in layers {
        let batch = layer.batch;

        // 1. Activation backward
        let delta = match layer.activation {
            GpuActivation::Identity => current_grad,
            GpuActivation::Relu => backend.gpu_relu_backward(&current_grad, layer.z_cache),
            GpuActivation::Sigmoid => backend.gpu_sigmoid_backward(&current_grad, layer.a_cache),
            GpuActivation::Tanh => backend.gpu_tanh_backward(&current_grad, layer.a_cache),
        };

        // 2. Bias gradient: db = reduce_cols(delta) / batch
        let db_gpu = backend.gpu_reduce_cols(&delta, batch, layer.out_size, 1.0 / batch as f64);
        let db = backend.gpu_download(&db_gpu);

        // 3. Weight gradient: dW = delta^T · input / batch
        let delta_t = backend.gpu_transpose(&delta, batch, layer.out_size);
        let dw_gpu = backend.gpu_matmul(
            &delta_t,
            layer.input_cache,
            layer.out_size,
            batch,
            layer.in_size,
        );
        let dw_gpu = backend.gpu_scale(&dw_gpu, 1.0 / batch as f64);
        let dw = backend.gpu_download(&dw_gpu);

        // 4. Input gradient for previous layer
        current_grad = backend.gpu_matmul(
            &delta,
            layer.weights_w,
            batch,
            layer.out_size,
            layer.in_size,
        );

        grads.push((dw, db));
    }

    grads
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

    /// GPU-to-GPU copy of a tensor. Returns an independent copy.
    ///
    /// Default: clones the CPU data.
    fn gpu_copy(&self, x: &GpuTensor) -> GpuTensor {
        let (rows, cols) = x.shape();
        GpuTensor::Cpu(x.to_cpu(), rows, cols)
    }

    // ── Backward-pass GPU operations ──

    /// ReLU backward: `out[i] = grad[i] * (z[i] > 0 ? 1 : 0)`.
    ///
    /// `grad` and `z` must have the same shape.
    fn gpu_relu_backward(&self, grad: &GpuTensor, z: &GpuTensor) -> GpuTensor {
        let (rows, cols) = grad.shape();
        let g = grad.to_cpu();
        let zv = z.to_cpu();
        let out: Vec<f64> = g
            .iter()
            .zip(zv.iter())
            .map(|(&gi, &zi)| if zi > 0.0 { gi } else { 0.0 })
            .collect();
        GpuTensor::Cpu(out, rows, cols)
    }

    /// Sigmoid backward: `out[i] = grad[i] * a[i] * (1 - a[i])`.
    ///
    /// `activated` is the post-sigmoid output.
    fn gpu_sigmoid_backward(&self, grad: &GpuTensor, activated: &GpuTensor) -> GpuTensor {
        let (rows, cols) = grad.shape();
        let g = grad.to_cpu();
        let a = activated.to_cpu();
        let out: Vec<f64> = g
            .iter()
            .zip(a.iter())
            .map(|(&gi, &ai)| gi * ai * (1.0 - ai))
            .collect();
        GpuTensor::Cpu(out, rows, cols)
    }

    /// Tanh backward: `out[i] = grad[i] * (1 - a[i]^2)`.
    ///
    /// `activated` is the post-tanh output.
    fn gpu_tanh_backward(&self, grad: &GpuTensor, activated: &GpuTensor) -> GpuTensor {
        let (rows, cols) = grad.shape();
        let g = grad.to_cpu();
        let a = activated.to_cpu();
        let out: Vec<f64> = g
            .iter()
            .zip(a.iter())
            .map(|(&gi, &ai)| gi * (1.0 - ai * ai))
            .collect();
        GpuTensor::Cpu(out, rows, cols)
    }

    /// Transpose a `[rows, cols]` matrix to `[cols, rows]`.
    fn gpu_transpose(&self, m: &GpuTensor, rows: usize, cols: usize) -> GpuTensor {
        let data = m.to_cpu();
        let mut t = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = data[i * cols + j];
            }
        }
        GpuTensor::Cpu(t, cols, rows)
    }

    /// Element-wise scale: `out[i] = x[i] * alpha`.
    fn gpu_scale(&self, x: &GpuTensor, alpha: f64) -> GpuTensor {
        let (rows, cols) = x.shape();
        let out: Vec<f64> = x.to_cpu().iter().map(|&v| v * alpha).collect();
        GpuTensor::Cpu(out, rows, cols)
    }

    /// Column-wise reduction: `out[j] = sum_i(x[i * cols + j]) * scale`.
    ///
    /// Sums over the row dimension, producing a `[1, cols]` vector.
    fn gpu_reduce_cols(&self, x: &GpuTensor, rows: usize, cols: usize, scale: f64) -> GpuTensor {
        let data = x.to_cpu();
        let mut out = vec![0.0; cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j] += data[i * cols + j];
            }
        }
        for v in &mut out {
            *v *= scale;
        }
        GpuTensor::Cpu(out, 1, cols)
    }

    // ── Batched GPU operations ──

    /// Batched forward pass: chains matmul → bias_add → activation across
    /// all layers in a single GPU submission (one fence wait).
    ///
    /// Returns the final output tensor and per-layer training caches.
    /// Default implementation calls individual `gpu_*` methods.
    fn gpu_forward_batch(
        &self,
        input: &GpuTensor,
        batch: usize,
        layers: &[GpuForwardLayer<'_>],
        training: bool,
    ) -> (GpuTensor, Vec<GpuLayerCache>) {
        gpu_forward_batch_default(self, input, batch, layers, training)
    }

    /// Batched backward pass: chains activation_backward → reduce_cols →
    /// transpose → matmul (dW) → scale → matmul (grad_input) across all
    /// layers in a single GPU submission.
    ///
    /// Layers are in reverse order (last network layer first).
    /// Returns `(dw, db)` per layer in the same order as `layers`.
    fn gpu_backward_batch(
        &self,
        grad_output: &GpuTensor,
        layers: &[GpuBackwardLayer<'_>],
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        gpu_backward_batch_default(self, grad_output, layers)
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
