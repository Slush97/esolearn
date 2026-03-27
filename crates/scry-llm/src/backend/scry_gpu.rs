//! scry-gpu compute backend — GPU-accelerated matmul via scry-gpu.
//!
//! Only `matmul` runs on GPU; all other `MathBackend` methods delegate to
//! `CpuBackend` for simplicity.
//!
//! Because `MathBackend` trait methods are static (no `&self`), we store
//! the GPU context in a `OnceLock` initialized on first use.

use std::sync::OnceLock;

use crate::backend::cpu::CpuBackend;
use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;

/// Minimum M*K*N product before engaging GPU (below this, CPU/BLAS is faster
/// due to per-dispatch overhead: buffer creation, submission, readback).
const GPU_MIN_ELEMENTS: usize = 65_536;

/// Maximum GPU buffer size in bytes (128 MiB).
const MAX_GPU_BUFFER_BYTES: u64 = 128 * 1024 * 1024;

// ---------------------------------------------------------------------------
// GPU context — cached device and compiled kernel
// ---------------------------------------------------------------------------

struct ScryCtx {
    dev: ::scry_gpu::Device,
    matmul: ::scry_gpu::Kernel,
}

// Safety: scry_gpu::Device and Kernel are Send+Sync
unsafe impl Send for ScryCtx {}
unsafe impl Sync for ScryCtx {}

/// Global GPU context, initialized on first matmul call.
static GPU_CTX: OnceLock<Option<ScryCtx>> = OnceLock::new();

fn get_ctx() -> Option<&'static ScryCtx> {
    GPU_CTX
        .get_or_init(|| {
            match init_scry_context() {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    eprintln!("[scry-llm] scry-gpu init failed, falling back to CPU: {e}");
                    None
                }
            }
        })
        .as_ref()
}

fn init_scry_context() -> Result<ScryCtx, String> {
    let dev = ::scry_gpu::Device::auto().map_err(|e| format!("scry-gpu: {e}"))?;
    let matmul = dev
        .compile(::scry_gpu::shaders::matmul::COARSE_64X64)
        .map_err(|e| format!("scry-gpu: shader compile: {e}"))?;
    Ok(ScryCtx { dev, matmul })
}

// ---------------------------------------------------------------------------
// GPU matmul dispatch
// ---------------------------------------------------------------------------

/// Run a single matmul on GPU. Returns None if GPU is unavailable.
fn gpu_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    let ctx = get_ctx()?;

    let sa = ctx.dev.upload(a).ok()?;
    let sb = ctx.dev.upload(b).ok()?;
    let sc = ctx.dev.alloc::<f32>(m * n).ok()?;

    let dims: [u32; 3] = [m as u32, n as u32, k as u32];
    ctx.dev
        .run_configured(
            &ctx.matmul,
            &[&sa, &sb, &sc],
            [(n as u32).div_ceil(64), (m as u32).div_ceil(64), 1],
            Some(bytemuck::bytes_of(&dims)),
        )
        .ok()?;

    sc.download().ok()
}

/// Transpose [rows x cols] -> [cols x rows] on CPU.
fn transpose_cpu(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Check if a matmul is worth sending to GPU.
fn should_use_gpu(m: usize, k: usize, n: usize) -> bool {
    if m * k * n < GPU_MIN_ELEMENTS {
        return false;
    }
    let a_bytes = (m * k * 4) as u64;
    let b_bytes = (k * n * 4) as u64;
    let c_bytes = (m * n * 4) as u64;
    a_bytes <= MAX_GPU_BUFFER_BYTES
        && b_bytes <= MAX_GPU_BUFFER_BYTES
        && c_bytes <= MAX_GPU_BUFFER_BYTES
}

/// Matmul with GPU acceleration: handles transpose and size thresholds.
fn matmul_gpu_or_cpu(
    a: &Vec<f32>,
    b: &Vec<f32>,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    if !should_use_gpu(m, k, n) {
        return CpuBackend::matmul(a, b, m, k, n, trans_a, trans_b);
    }

    // Handle transposes: the shader expects row-major A[M×K] × B[K×N]
    let a_rm;
    let a_data: &[f32] = if trans_a {
        a_rm = transpose_cpu(a, k, m);
        &a_rm
    } else {
        a
    };

    let b_rm;
    let b_data: &[f32] = if trans_b {
        b_rm = transpose_cpu(b, n, k);
        &b_rm
    } else {
        b
    };

    gpu_matmul(a_data, b_data, m, k, n)
        .unwrap_or_else(|| CpuBackend::matmul(a, b, m, k, n, trans_a, trans_b))
}

// ---------------------------------------------------------------------------
// ScryGpuBackend — public type
// ---------------------------------------------------------------------------

/// GPU-accelerated backend for scry-llm using scry-gpu compute shaders.
///
/// Matmul dispatches to the GPU via a global `OnceLock` context;
/// all other ops use `CpuBackend`. Storage is `Vec<f32>` (CPU-resident).
pub struct ScryGpuBackend;

impl DeviceBackend for ScryGpuBackend {
    type Storage = Vec<f32>;
    type Stream = ();
    #[cfg(feature = "quantize")]
    type I8Storage = Vec<i8>;

    #[cfg(feature = "quantize")]
    fn i8_from_vec(data: Vec<i8>) -> Vec<i8> { data }
    #[cfg(feature = "quantize")]
    fn i8_to_vec(storage: &Vec<i8>) -> Vec<i8> { storage.clone() }

    fn zeros(shape: &Shape) -> Vec<f32> { CpuBackend::zeros(shape) }
    fn ones(shape: &Shape) -> Vec<f32> { CpuBackend::ones(shape) }
    fn from_vec(data: Vec<f32>, shape: &Shape) -> Vec<f32> { CpuBackend::from_vec(data, shape) }
    fn to_vec(storage: &Vec<f32>) -> Vec<f32> { CpuBackend::to_vec(storage) }
    fn clone_storage(storage: &Vec<f32>) -> Vec<f32> { CpuBackend::clone_storage(storage) }
}

impl MathBackend for ScryGpuBackend {
    fn matmul(
        a: &Vec<f32>,
        b: &Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Vec<f32> {
        matmul_gpu_or_cpu(a, b, m, k, n, trans_a, trans_b)
    }

    fn add(a: &Vec<f32>, b: &Vec<f32>, a_shape: &Shape, b_shape: &Shape, out_shape: &Shape) -> Vec<f32> {
        CpuBackend::add(a, b, a_shape, b_shape, out_shape)
    }

    fn softmax(input: &Vec<f32>, shape: &Shape) -> Vec<f32> {
        CpuBackend::softmax(input, shape)
    }

    fn layernorm(input: &Vec<f32>, gamma: &Vec<f32>, beta: &Vec<f32>, shape: &Shape, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        CpuBackend::layernorm(input, gamma, beta, shape, eps)
    }

    fn gelu(input: &Vec<f32>) -> Vec<f32> {
        CpuBackend::gelu(input)
    }

    fn embedding(weight: &Vec<f32>, indices: &[usize], vocab: usize, dim: usize) -> Vec<f32> {
        CpuBackend::embedding(weight, indices, vocab, dim)
    }

    fn sum(input: &Vec<f32>) -> f32 {
        CpuBackend::sum(input)
    }

    fn mul_elementwise(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        CpuBackend::mul_elementwise(a, b)
    }

    fn scale(a: &Vec<f32>, scalar: f32) -> Vec<f32> {
        CpuBackend::scale(a, scalar)
    }

    fn concat_rows(a: &Vec<f32>, b: &Vec<f32>, a_rows: usize, b_rows: usize, cols: usize) -> Vec<f32> {
        CpuBackend::concat_rows(a, b, a_rows, b_rows, cols)
    }

    fn rmsnorm(input: &Vec<f32>, weight: &Vec<f32>, shape: &Shape, eps: f32) -> Vec<f32> {
        CpuBackend::rmsnorm(input, weight, shape, eps)
    }

    fn rope(input: &Vec<f32>, shape: &Shape, pos: usize, head_dim: usize, theta: f32) -> Vec<f32> {
        CpuBackend::rope(input, shape, pos, head_dim, theta)
    }

    fn rope_with_freqs_preloaded(
        input: &Vec<f32>, seq: usize, n_heads: usize, head_dim: usize,
        start_pos: usize, freqs: &Vec<f32>,
    ) -> Vec<f32> {
        CpuBackend::rope_with_freqs_preloaded(input, seq, n_heads, head_dim, start_pos, freqs)
    }

    fn swiglu(gate: &Vec<f32>, up: &Vec<f32>) -> Vec<f32> {
        CpuBackend::swiglu(gate, up)
    }

    fn repeat_kv(input: &Vec<f32>, n_kv_heads: usize, n_q_heads: usize, seq: usize, d_head: usize) -> Vec<f32> {
        CpuBackend::repeat_kv(input, n_kv_heads, n_q_heads, seq, d_head)
    }
}
