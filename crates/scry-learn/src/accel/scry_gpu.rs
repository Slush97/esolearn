// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU compute backend via scry-gpu.
//!
//! Drop-in replacement for the CPU backend with significantly lower
//! per-dispatch overhead. All data is converted f64 -> f32 for GPU
//! compute, same as the wgpu path.
//!
//! With the `cuda` feature, matmul uses cuBLAS SGEMM (~2x faster than
//! the best Vulkan compute shader) and distance uses an NVRTC-compiled
//! CUDA C kernel. Without `cuda`, dispatches go through Vulkan WGSL
//! shaders as before.

use std::mem::ManuallyDrop;
use std::sync::{Mutex, OnceLock};

use super::ComputeBackend;

/// Maximum GPU buffer size in bytes (128 MiB).
const MAX_GPU_BUFFER_BYTES: u64 = 128 * 1024 * 1024;

// ---------------------------------------------------------------------------
// GPU context — cached device and compiled kernels
// ---------------------------------------------------------------------------

/// Matmul dispatch strategy: cuBLAS on CUDA, WGSL kernel on Vulkan.
enum MatmulStrategy {
    Wgsl(::scry_gpu::Kernel),
    #[cfg(feature = "cuda")]
    CuBlas,
}

/// Pre-compiled element-wise kernels for GPU-resident tensor ops.
struct ElementwiseKernels {
    bias_add: ::scry_gpu::Kernel,
    relu: ::scry_gpu::Kernel,
    tanh: ::scry_gpu::Kernel,
    sigmoid: ::scry_gpu::Kernel,
}

/// Pre-compiled backward + utility kernels for GPU-resident backpropagation.
struct BackwardKernels {
    relu_backward: ::scry_gpu::Kernel,
    sigmoid_backward: ::scry_gpu::Kernel,
    tanh_backward: ::scry_gpu::Kernel,
    transpose: ::scry_gpu::Kernel,
    scale: ::scry_gpu::Kernel,
    reduce_cols: ::scry_gpu::Kernel,
}

struct ScryCtx {
    /// Mutex serializes GPU dispatches (the Device reuses internal
    /// synchronization objects, so concurrent access must be serialized).
    inner: Mutex<ScryCtxInner>,
}

struct ScryCtxInner {
    dev: ::scry_gpu::Device,
    matmul: MatmulStrategy,
    distance: ::scry_gpu::Kernel,
    elementwise: ElementwiseKernels,
    backward: BackwardKernels,
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

    // Try CUDA path: cuBLAS for matmul, NVRTC-compiled CUDA C for distance + elementwise.
    #[cfg(feature = "cuda")]
    if dev.backend_kind() == ::scry_gpu::BackendKind::Cuda {
        match init_cuda_kernels(&dev) {
            Ok((distance, elementwise, backward)) => {
                return Ok(ScryCtx {
                    inner: Mutex::new(ScryCtxInner {
                        dev,
                        matmul: MatmulStrategy::CuBlas,
                        distance,
                        elementwise,
                        backward,
                    }),
                });
            }
            Err(e) => {
                eprintln!("[scry-learn] CUDA kernel compile failed ({e}), trying Vulkan");
            }
        }
        // CUDA selected but kernel compilation failed — try Vulkan explicitly.
        let dev = ::scry_gpu::Device::with_backend(::scry_gpu::BackendKind::Vulkan)
            .map_err(|e| format!("scry-gpu vulkan fallback: {e}"))?;
        return init_vulkan(dev);
    }

    init_vulkan(dev)
}

#[cfg(feature = "cuda")]
fn init_cuda_kernels(
    dev: &::scry_gpu::Device,
) -> Result<(::scry_gpu::Kernel, ElementwiseKernels, BackwardKernels), String> {
    use ::scry_gpu::shaders::{backward, distance, elementwise};

    let distance = dev
        .compile_cuda(distance::PAIRWISE_EUCLIDEAN_CUDA, "pairwise_euclidean", 3, [256, 1, 1])
        .map_err(|e| format!("distance: {e}"))?;
    let bias_add = dev
        .compile_cuda(elementwise::BIAS_ADD_CUDA, "bias_add", 3, [256, 1, 1])
        .map_err(|e| format!("bias_add: {e}"))?;
    let relu = dev
        .compile_cuda(elementwise::RELU_CUDA, "relu", 2, [256, 1, 1])
        .map_err(|e| format!("relu: {e}"))?;
    let tanh = dev
        .compile_cuda(elementwise::TANH_CUDA, "tanh_fwd", 2, [256, 1, 1])
        .map_err(|e| format!("tanh: {e}"))?;
    let sigmoid = dev
        .compile_cuda(elementwise::SIGMOID_CUDA, "sigmoid", 2, [256, 1, 1])
        .map_err(|e| format!("sigmoid: {e}"))?;

    let relu_backward = dev
        .compile_cuda(backward::RELU_BACKWARD_CUDA, "relu_backward", 3, [256, 1, 1])
        .map_err(|e| format!("relu_backward: {e}"))?;
    let sigmoid_backward = dev
        .compile_cuda(backward::SIGMOID_BACKWARD_CUDA, "sigmoid_backward", 3, [256, 1, 1])
        .map_err(|e| format!("sigmoid_backward: {e}"))?;
    let tanh_backward = dev
        .compile_cuda(backward::TANH_BACKWARD_CUDA, "tanh_backward", 3, [256, 1, 1])
        .map_err(|e| format!("tanh_backward: {e}"))?;
    let transpose = dev
        .compile_cuda(backward::TRANSPOSE_CUDA, "transpose_2d", 2, [256, 1, 1])
        .map_err(|e| format!("transpose: {e}"))?;
    let scale = dev
        .compile_cuda(backward::SCALE_CUDA, "scale_fwd", 2, [256, 1, 1])
        .map_err(|e| format!("scale: {e}"))?;
    let reduce_cols = dev
        .compile_cuda(backward::REDUCE_COLS_CUDA, "reduce_cols", 2, [256, 1, 1])
        .map_err(|e| format!("reduce_cols: {e}"))?;

    Ok((
        distance,
        ElementwiseKernels { bias_add, relu, tanh, sigmoid },
        BackwardKernels { relu_backward, sigmoid_backward, tanh_backward, transpose, scale, reduce_cols },
    ))
}

fn init_vulkan(dev: ::scry_gpu::Device) -> Result<ScryCtx, String> {
    use ::scry_gpu::shaders::{
        backward as bwd_shaders, distance as dist_shaders, elementwise,
        matmul as matmul_shaders,
    };

    let matmul = dev
        .compile(matmul_shaders::COARSE_64X64)
        .map_err(|e| format!("scry-gpu: matmul shader: {e}"))?;
    let distance = dev
        .compile(dist_shaders::PAIRWISE_EUCLIDEAN)
        .map_err(|e| format!("scry-gpu: distance shader: {e}"))?;
    let bias_add = dev
        .compile(elementwise::BIAS_ADD)
        .map_err(|e| format!("scry-gpu: bias_add shader: {e}"))?;
    let relu = dev
        .compile(elementwise::RELU)
        .map_err(|e| format!("scry-gpu: relu shader: {e}"))?;
    let tanh = dev
        .compile(elementwise::TANH)
        .map_err(|e| format!("scry-gpu: tanh shader: {e}"))?;
    let sigmoid = dev
        .compile(elementwise::SIGMOID)
        .map_err(|e| format!("scry-gpu: sigmoid shader: {e}"))?;

    let relu_backward = dev
        .compile(bwd_shaders::RELU_BACKWARD)
        .map_err(|e| format!("scry-gpu: relu_backward shader: {e}"))?;
    let sigmoid_backward = dev
        .compile(bwd_shaders::SIGMOID_BACKWARD)
        .map_err(|e| format!("scry-gpu: sigmoid_backward shader: {e}"))?;
    let tanh_backward = dev
        .compile(bwd_shaders::TANH_BACKWARD)
        .map_err(|e| format!("scry-gpu: tanh_backward shader: {e}"))?;
    let transpose = dev
        .compile(bwd_shaders::TRANSPOSE)
        .map_err(|e| format!("scry-gpu: transpose shader: {e}"))?;
    let scale = dev
        .compile(bwd_shaders::SCALE)
        .map_err(|e| format!("scry-gpu: scale shader: {e}"))?;
    let reduce_cols = dev
        .compile(bwd_shaders::REDUCE_COLS)
        .map_err(|e| format!("scry-gpu: reduce_cols shader: {e}"))?;

    Ok(ScryCtx {
        inner: Mutex::new(ScryCtxInner {
            dev,
            matmul: MatmulStrategy::Wgsl(matmul),
            distance,
            elementwise: ElementwiseKernels { bias_add, relu, tanh, sigmoid },
            backward: BackwardKernels {
                relu_backward,
                sigmoid_backward,
                tanh_backward,
                transpose,
                scale,
                reduce_cols,
            },
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
        get_ctx()
            .map(|_| Self)
            .ok_or_else(|| "scry-gpu: initialization failed".into())
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

            let c_f32: Vec<f32> = match &gpu.matmul {
                MatmulStrategy::Wgsl(kernel) => {
                    let sc = gpu.dev.alloc::<f32>(m * n).ok()?;
                    let dims: [u32; 3] = [m as u32, n as u32, k as u32];
                    gpu.dev
                        .run_configured(
                            kernel,
                            &[&sa, &sb, &sc],
                            [(n as u32).div_ceil(64), (m as u32).div_ceil(64), 1],
                            Some(bytemuck::bytes_of(&dims)),
                        )
                        .ok()?;
                    sc.download().ok()?
                }
                #[cfg(feature = "cuda")]
                MatmulStrategy::CuBlas => {
                    let mut sc = gpu.dev.alloc::<f32>(m * n).ok()?;
                    gpu.dev
                        .cublas_matmul(&sa, &sb, &mut sc, m as u32, n as u32, k as u32)
                        .ok()?;
                    sc.download().ok()?
                }
            };

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
        if let Some(ctx) = get_ctx() {
            if let Ok(gpu) = ctx.inner.lock() {
                if gpu.dev.backend_kind() == ::scry_gpu::BackendKind::Cuda {
                    return "gpu (scry-gpu/cuda)";
                }
            }
        }
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

    // ── GPU-resident tensor operations ──

    fn gpu_upload(&self, data: &[f64], rows: usize, cols: usize) -> super::GpuTensor {
        let Some(ctx) = get_ctx() else {
            return super::GpuTensor::Cpu(data.to_vec(), rows, cols);
        };
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let Ok(gpu) = ctx.inner.lock() else {
            return super::GpuTensor::Cpu(data.to_vec(), rows, cols);
        };
        match gpu.dev.upload(&f32_data) {
            Ok(buf) => super::GpuTensor::Gpu(buf, rows, cols),
            Err(_) => super::GpuTensor::Cpu(data.to_vec(), rows, cols),
        }
    }

    fn gpu_matmul(
        &self,
        a: &super::GpuTensor,
        b: &super::GpuTensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> super::GpuTensor {
        // Both must be GPU tensors, otherwise fall back to CPU default.
        let (super::GpuTensor::Gpu(a_buf, ..), super::GpuTensor::Gpu(b_buf, ..)) = (a, b) else {
            let a_data = a.to_cpu();
            let b_data = b.to_cpu();
            return super::GpuTensor::Cpu(self.matmul(&a_data, &b_data, m, k, n), m, n);
        };

        let Some(ctx) = get_ctx() else {
            let a_data = a.to_cpu();
            let b_data = b.to_cpu();
            return super::GpuTensor::Cpu(self.matmul(&a_data, &b_data, m, k, n), m, n);
        };

        let result = (|| {
            let gpu = ctx.inner.lock().ok()?;
            match &gpu.matmul {
                MatmulStrategy::Wgsl(kernel) => {
                    let c = gpu.dev.alloc::<f32>(m * n).ok()?;
                    let dims: [u32; 3] = [m as u32, n as u32, k as u32];
                    gpu.dev
                        .run_configured(
                            kernel,
                            &[a_buf, b_buf, &c],
                            [(n as u32).div_ceil(64), (m as u32).div_ceil(64), 1],
                            Some(bytemuck::bytes_of(&dims)),
                        )
                        .ok()?;
                    Some(super::GpuTensor::Gpu(c, m, n))
                }
                #[cfg(feature = "cuda")]
                MatmulStrategy::CuBlas => {
                    let mut c = gpu.dev.alloc::<f32>(m * n).ok()?;
                    gpu.dev
                        .cublas_matmul(a_buf, b_buf, &mut c, m as u32, n as u32, k as u32)
                        .ok()?;
                    Some(super::GpuTensor::Gpu(c, m, n))
                }
            }
        })();

        result.unwrap_or_else(|| {
            let a_data = a.to_cpu();
            let b_data = b.to_cpu();
            super::GpuTensor::Cpu(self.matmul(&a_data, &b_data, m, k, n), m, n)
        })
    }

    fn gpu_bias_add(
        &self,
        z: &super::GpuTensor,
        bias: &super::GpuTensor,
        rows: usize,
        cols: usize,
    ) -> super::GpuTensor {
        let (super::GpuTensor::Gpu(z_buf, ..), super::GpuTensor::Gpu(b_buf, ..)) = (z, bias)
        else {
            let mut data = z.to_cpu();
            let b = bias.to_cpu();
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] += b[j];
                }
            }
            return super::GpuTensor::Cpu(data, rows, cols);
        };

        let n = rows * cols;
        let result = (|| {
            let gpu = ctx_lock()?;
            let out = gpu.dev.alloc::<f32>(n).ok()?;
            let dims: [u32; 2] = [n as u32, cols as u32];
            gpu.dev
                .run_configured(
                    &gpu.elementwise.bias_add,
                    &[z_buf, b_buf, &out],
                    [(n as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(super::GpuTensor::Gpu(out, rows, cols))
        })();

        result.unwrap_or_else(|| {
            let mut data = z.to_cpu();
            let b = bias.to_cpu();
            for i in 0..rows {
                for j in 0..cols {
                    data[i * cols + j] += b[j];
                }
            }
            super::GpuTensor::Cpu(data, rows, cols)
        })
    }

    fn gpu_relu(&self, x: &super::GpuTensor) -> super::GpuTensor {
        dispatch_unary(x, |gpu, buf, n| {
            let out = gpu.dev.alloc::<f32>(n).ok()?;
            let dims: [u32; 1] = [n as u32];
            gpu.dev
                .run_configured(
                    &gpu.elementwise.relu,
                    &[buf, &out],
                    [(n as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(out)
        })
    }

    fn gpu_tanh(&self, x: &super::GpuTensor) -> super::GpuTensor {
        dispatch_unary(x, |gpu, buf, n| {
            let out = gpu.dev.alloc::<f32>(n).ok()?;
            let dims: [u32; 1] = [n as u32];
            gpu.dev
                .run_configured(
                    &gpu.elementwise.tanh,
                    &[buf, &out],
                    [(n as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(out)
        })
    }

    fn gpu_sigmoid(&self, x: &super::GpuTensor) -> super::GpuTensor {
        dispatch_unary(x, |gpu, buf, n| {
            let out = gpu.dev.alloc::<f32>(n).ok()?;
            let dims: [u32; 1] = [n as u32];
            gpu.dev
                .run_configured(
                    &gpu.elementwise.sigmoid,
                    &[buf, &out],
                    [(n as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(out)
        })
    }

    fn gpu_download(&self, t: &super::GpuTensor) -> Vec<f64> {
        match t {
            super::GpuTensor::Gpu(buf, _, _) => buf
                .download()
                .unwrap_or_default()
                .iter()
                .map(|&v| f64::from(v))
                .collect(),
            super::GpuTensor::Cpu(data, _, _) => data.clone(),
        }
    }

    fn gpu_copy(&self, x: &super::GpuTensor) -> super::GpuTensor {
        match x {
            super::GpuTensor::Gpu(buf, rows, cols) => {
                let result = (|| {
                    let gpu = ctx_lock()?;
                    gpu.dev.copy_buffer(buf).ok()
                })();
                match result {
                    Some(copy) => super::GpuTensor::Gpu(copy, *rows, *cols),
                    None => x.to_cpu_tensor(),
                }
            }
            super::GpuTensor::Cpu(data, rows, cols) => {
                super::GpuTensor::Cpu(data.clone(), *rows, *cols)
            }
        }
    }

    fn gpu_relu_backward(&self, grad: &super::GpuTensor, z: &super::GpuTensor) -> super::GpuTensor {
        dispatch_binary(grad, z, |gpu, g_buf, z_buf, n| {
            let out = gpu.dev.alloc::<f32>(n).ok()?;
            let dims: [u32; 1] = [n as u32];
            gpu.dev
                .run_configured(
                    &gpu.backward.relu_backward,
                    &[g_buf, z_buf, &out],
                    [(n as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(out)
        })
    }

    fn gpu_sigmoid_backward(
        &self,
        grad: &super::GpuTensor,
        activated: &super::GpuTensor,
    ) -> super::GpuTensor {
        dispatch_binary(grad, activated, |gpu, g_buf, a_buf, n| {
            let out = gpu.dev.alloc::<f32>(n).ok()?;
            let dims: [u32; 1] = [n as u32];
            gpu.dev
                .run_configured(
                    &gpu.backward.sigmoid_backward,
                    &[g_buf, a_buf, &out],
                    [(n as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(out)
        })
    }

    fn gpu_tanh_backward(
        &self,
        grad: &super::GpuTensor,
        activated: &super::GpuTensor,
    ) -> super::GpuTensor {
        dispatch_binary(grad, activated, |gpu, g_buf, a_buf, n| {
            let out = gpu.dev.alloc::<f32>(n).ok()?;
            let dims: [u32; 1] = [n as u32];
            gpu.dev
                .run_configured(
                    &gpu.backward.tanh_backward,
                    &[g_buf, a_buf, &out],
                    [(n as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(out)
        })
    }

    fn gpu_transpose(&self, m: &super::GpuTensor, rows: usize, cols: usize) -> super::GpuTensor {
        let super::GpuTensor::Gpu(buf, ..) = m else {
            return super::CpuBackend.gpu_transpose(m, rows, cols);
        };

        let n = rows * cols;
        let result = (|| {
            let gpu = ctx_lock()?;
            let out = gpu.dev.alloc::<f32>(n).ok()?;
            let dims: [u32; 2] = [rows as u32, cols as u32];
            gpu.dev
                .run_configured(
                    &gpu.backward.transpose,
                    &[buf, &out],
                    [(n as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(super::GpuTensor::Gpu(out, cols, rows))
        })();

        result.unwrap_or_else(|| {
            super::CpuBackend.gpu_transpose(m, rows, cols)
        })
    }

    fn gpu_scale(&self, x: &super::GpuTensor, alpha: f64) -> super::GpuTensor {
        let super::GpuTensor::Gpu(buf, rows, cols) = x else {
            return super::CpuBackend.gpu_scale(x, alpha);
        };

        let n = rows * cols;
        let result = (|| {
            let gpu = ctx_lock()?;
            let out = gpu.dev.alloc::<f32>(n).ok()?;
            let dims: [u32; 2] = [n as u32, (alpha as f32).to_bits()];
            gpu.dev
                .run_configured(
                    &gpu.backward.scale,
                    &[buf, &out],
                    [(n as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(super::GpuTensor::Gpu(out, *rows, *cols))
        })();

        result.unwrap_or_else(|| {
            super::CpuBackend.gpu_scale(x, alpha)
        })
    }

    fn gpu_reduce_cols(
        &self,
        x: &super::GpuTensor,
        rows: usize,
        cols: usize,
        scale: f64,
    ) -> super::GpuTensor {
        let super::GpuTensor::Gpu(buf, ..) = x else {
            return super::CpuBackend.gpu_reduce_cols(x, rows, cols, scale);
        };

        let result = (|| {
            let gpu = ctx_lock()?;
            let out = gpu.dev.alloc::<f32>(cols).ok()?;
            let dims: [u32; 3] = [rows as u32, cols as u32, (scale as f32).to_bits()];
            gpu.dev
                .run_configured(
                    &gpu.backward.reduce_cols,
                    &[buf, &out],
                    [(cols as u32).div_ceil(256), 1, 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            Some(super::GpuTensor::Gpu(out, 1, cols))
        })();

        result.unwrap_or_else(|| {
            super::CpuBackend.gpu_reduce_cols(x, rows, cols, scale)
        })
    }
}

// ── Helpers for GPU-resident dispatch ──

fn ctx_lock() -> Option<std::sync::MutexGuard<'static, ScryCtxInner>> {
    get_ctx()?.inner.lock().ok()
}

/// Dispatch a binary element-wise kernel on two GPU tensors (e.g. backward
/// activations). Falls back to CPU if either tensor isn't GPU-resident.
fn dispatch_binary(
    a: &super::GpuTensor,
    b: &super::GpuTensor,
    f: impl FnOnce(
        &ScryCtxInner,
        &::scry_gpu::Buffer<f32>,
        &::scry_gpu::Buffer<f32>,
        usize,
    ) -> Option<::scry_gpu::Buffer<f32>>,
) -> super::GpuTensor {
    let (super::GpuTensor::Gpu(a_buf, rows, cols), super::GpuTensor::Gpu(b_buf, ..)) = (a, b)
    else {
        return a.to_cpu_tensor();
    };

    let n = rows * cols;
    let result = (|| {
        let gpu = ctx_lock()?;
        f(&gpu, a_buf, b_buf, n)
    })();

    match result {
        Some(out) => super::GpuTensor::Gpu(out, *rows, *cols),
        None => a.to_cpu_tensor(),
    }
}

/// Dispatch a unary element-wise kernel on a GPU tensor. Falls back to CPU
/// via `GpuTensor::to_cpu()` + trait default if the tensor isn't GPU-resident
/// or if the dispatch fails.
fn dispatch_unary(
    x: &super::GpuTensor,
    f: impl FnOnce(&ScryCtxInner, &::scry_gpu::Buffer<f32>, usize) -> Option<::scry_gpu::Buffer<f32>>,
) -> super::GpuTensor {
    let super::GpuTensor::Gpu(buf, rows, cols) = x else {
        // CPU fallback — should not happen in the normal GPU forward path,
        // but safe to handle.
        return x.to_cpu_tensor();
    };

    let n = rows * cols;
    let result = (|| {
        let gpu = ctx_lock()?;
        f(&gpu, buf, n)
    })();

    match result {
        Some(out) => super::GpuTensor::Gpu(out, *rows, *cols),
        None => x.to_cpu_tensor(),
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
        assert!(
            gpu.name().starts_with("gpu (scry-gpu"),
            "unexpected backend name: {}",
            gpu.name()
        );
    }
}
