//! scry-gpu compute backend — GPU-accelerated matmul via scry-gpu.
//!
//! Storage is an enum [`ScryGpuStorage`] that can hold either CPU or
//! GPU-resident data. Today, every `MathBackend` op materializes inputs
//! to CPU before computing and returns a CPU variant — this commit only
//! introduces the type and explicit transfer helpers. Subsequent commits
//! will keep results on-device when inputs already are.
//!
//! With the `scry-gpu-cuda` feature, matmul uses cuBLAS SGEMM (~2x faster
//! than the best Vulkan compute shader). Without it, dispatches go through
//! Vulkan WGSL shaders.
//!
//! Because `MathBackend` trait methods are static (no `&self`), we store
//! the GPU context in a `OnceLock` initialized on first use.

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};

use scry_gpu::Buffer;

use crate::backend::cpu::CpuBackend;
use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;

/// Minimum M*K*N product before engaging GPU (below this, CPU/BLAS is faster
/// due to per-dispatch overhead: buffer creation, submission, readback).
const GPU_MIN_ELEMENTS: usize = 65_536;

/// Maximum GPU buffer size in bytes (128 MiB).
const MAX_GPU_BUFFER_BYTES: u64 = 128 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Storage enum — CPU Vec or GPU-resident buffer
// ---------------------------------------------------------------------------

/// Storage for [`ScryGpuBackend`] tensors. Either a CPU `Vec<f32>` or a
/// reference-counted GPU buffer.
///
/// `Clone` on the GPU variant is cheap (`Arc::clone`); on the CPU variant
/// it clones the underlying `Vec`.
#[derive(Clone)]
pub enum ScryGpuStorage {
    /// Host-resident data.
    Cpu(Vec<f32>),
    /// Device-resident buffer. The Arc lets multiple tensors share the
    /// same allocation (e.g. after a clone) without re-uploading.
    Gpu { buf: Arc<Buffer<f32>>, len: usize },
}

impl ScryGpuStorage {
    /// Number of f32 elements held, regardless of residency.
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(v) => v.len(),
            Self::Gpu { len, .. } => *len,
        }
    }

    /// True if no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True if data lives on the GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu { .. })
    }

    /// Copy CPU data into a fresh `Vec` or download GPU data.
    fn materialize(&self) -> Vec<f32> {
        match self {
            Self::Cpu(v) => v.clone(),
            Self::Gpu { buf, .. } => buf
                .download()
                .expect("scry-gpu: download failed in materialize"),
        }
    }

    /// Borrow CPU data, downloading first if GPU-resident. Returns
    /// `Cow<Vec<f32>>` (rather than `Cow<[f32]>`) so callers can deref
    /// to `&Vec<f32>` for `CpuBackend` ops without an extra clone.
    #[allow(clippy::owned_cow)]
    fn as_vec(&self) -> Cow<'_, Vec<f32>> {
        match self {
            Self::Cpu(v) => Cow::Borrowed(v),
            Self::Gpu { .. } => Cow::Owned(self.materialize()),
        }
    }
}

impl std::fmt::Debug for ScryGpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu(v) => f
                .debug_struct("ScryGpuStorage::Cpu")
                .field("len", &v.len())
                .finish(),
            Self::Gpu { len, .. } => f
                .debug_struct("ScryGpuStorage::Gpu")
                .field("len", len)
                .finish_non_exhaustive(),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU context — cached device and compiled kernel
// ---------------------------------------------------------------------------

/// Matmul dispatch strategy: cuBLAS on CUDA, WGSL kernel on Vulkan.
enum MatmulStrategy {
    Wgsl(::scry_gpu::Kernel),
    #[cfg(feature = "scry-gpu-cuda")]
    CuBlas,
}

struct ScryCtx {
    dev: ::scry_gpu::Device,
    matmul: MatmulStrategy,
    /// On-device transpose kernel — populated only on the WGSL path.
    /// cuBLAS dispatch transposes on CPU until a CUDA transpose kernel
    /// is wired up.
    transpose: Option<::scry_gpu::Kernel>,
}

// Safety: scry_gpu::Device and Kernel are Send+Sync
unsafe impl Send for ScryCtx {}
unsafe impl Sync for ScryCtx {}

/// Global GPU context, initialized on first matmul call.
static GPU_CTX: OnceLock<Option<ScryCtx>> = OnceLock::new();

fn get_ctx() -> Option<&'static ScryCtx> {
    GPU_CTX
        .get_or_init(|| match init_scry_context() {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("[scry-llm] scry-gpu init failed, falling back to CPU: {e}");
                None
            }
        })
        .as_ref()
}

fn init_scry_context() -> Result<ScryCtx, String> {
    let dev = ::scry_gpu::Device::auto().map_err(|e| format!("scry-gpu: {e}"))?;

    // Try CUDA path: cuBLAS for matmul (no kernel compilation needed).
    #[cfg(feature = "scry-gpu-cuda")]
    if dev.backend_kind() == ::scry_gpu::BackendKind::Cuda {
        return Ok(ScryCtx {
            dev,
            matmul: MatmulStrategy::CuBlas,
            transpose: None,
        });
    }

    let matmul = dev
        .compile(::scry_gpu::shaders::matmul::COARSE_64X64)
        .map_err(|e| format!("scry-gpu: shader compile: {e}"))?;
    let transpose = dev
        .compile(::scry_gpu::shaders::backward::TRANSPOSE)
        .map_err(|e| format!("scry-gpu: transpose compile: {e}"))?;
    Ok(ScryCtx {
        dev,
        matmul: MatmulStrategy::Wgsl(matmul),
        transpose: Some(transpose),
    })
}

// ---------------------------------------------------------------------------
// GPU matmul dispatch
// ---------------------------------------------------------------------------

/// Run a single matmul on GPU. Returns None if GPU is unavailable.
fn gpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    let ctx = get_ctx()?;

    let sa = ctx.dev.upload(a).ok()?;
    let sb = ctx.dev.upload(b).ok()?;

    match &ctx.matmul {
        MatmulStrategy::Wgsl(kernel) => {
            let sc = ctx.dev.alloc::<f32>(m * n).ok()?;
            let dims: [u32; 3] = [m as u32, n as u32, k as u32];
            ctx.dev
                .run_configured(
                    kernel,
                    &[&sa, &sb, &sc],
                    [(n as u32).div_ceil(64), (m as u32).div_ceil(64), 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
            sc.download().ok()
        }
        #[cfg(feature = "scry-gpu-cuda")]
        MatmulStrategy::CuBlas => {
            let mut sc = ctx.dev.alloc::<f32>(m * n).ok()?;
            ctx.dev
                .cublas_matmul(&sa, &sb, &mut sc, m as u32, n as u32, k as u32)
                .ok()?;
            sc.download().ok()
        }
    }
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
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    if !should_use_gpu(m, k, n) {
        return CpuBackend::matmul(&a.to_vec(), &b.to_vec(), m, k, n, trans_a, trans_b);
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
        .unwrap_or_else(|| CpuBackend::matmul(&a.to_vec(), &b.to_vec(), m, k, n, trans_a, trans_b))
}

// ---------------------------------------------------------------------------
// GPU-resident matmul — keeps result on device when possible
// ---------------------------------------------------------------------------

/// Acquire a GPU buffer view of `storage`. Uploads if currently CPU-resident.
/// Returns `None` if the GPU is unavailable.
fn as_gpu_buffer(storage: &ScryGpuStorage) -> Option<Arc<Buffer<f32>>> {
    match storage {
        ScryGpuStorage::Gpu { buf, .. } => Some(Arc::clone(buf)),
        ScryGpuStorage::Cpu(v) => {
            let ctx = get_ctx()?;
            let buf = ctx.dev.upload::<f32>(v).ok()?;
            Some(Arc::new(buf))
        }
    }
}

/// Run the on-device transpose kernel. Returns a fresh buffer of shape `[cols, rows]`.
fn gpu_transpose(input: &Buffer<f32>, rows: usize, cols: usize) -> Option<Buffer<f32>> {
    let ctx = get_ctx()?;
    let kernel = ctx.transpose.as_ref()?;
    let out = ctx.dev.alloc::<f32>(rows * cols).ok()?;
    let dims: [u32; 2] = [rows as u32, cols as u32];
    let groups = ((rows * cols) as u32).div_ceil(256);
    ctx.dev
        .run_configured(
            kernel,
            &[input, &out],
            [groups, 1, 1],
            Some(bytemuck::bytes_of(&dims)),
        )
        .ok()?;
    Some(out)
}

/// GPU-resident matmul: takes `ScryGpuStorage` inputs, returns a `Gpu`-variant
/// output without round-tripping through CPU. Returns `None` if the GPU path
/// is unavailable for the given inputs (caller should fall back to CPU).
///
/// Preconditions enforced by caller via [`should_use_gpu`].
fn gpu_matmul_persistent(
    a: &ScryGpuStorage,
    b: &ScryGpuStorage,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    // cuBLAS path doesn't have a CUDA transpose kernel wired yet; defer to
    // the legacy materialize-and-download path for now.
    // Single-arm without scry-gpu-cuda — clippy doesn't see the cfg arm.
    #[allow(clippy::infallible_destructuring_match)]
    let kernel = match &ctx.matmul {
        MatmulStrategy::Wgsl(k) => k,
        #[cfg(feature = "scry-gpu-cuda")]
        MatmulStrategy::CuBlas => return None,
    };

    let buf_a = as_gpu_buffer(a)?;
    let buf_b = as_gpu_buffer(b)?;

    // Transpose on-device when needed. The shader expects row-major M×K and K×N.
    let a_t;
    let buf_a_ref: &Buffer<f32> = if trans_a {
        a_t = gpu_transpose(&buf_a, k, m)?;
        &a_t
    } else {
        &buf_a
    };
    let b_t;
    let buf_b_ref: &Buffer<f32> = if trans_b {
        b_t = gpu_transpose(&buf_b, n, k)?;
        &b_t
    } else {
        &buf_b
    };

    let out = ctx.dev.alloc::<f32>(m * n).ok()?;
    let dims: [u32; 3] = [m as u32, n as u32, k as u32];
    ctx.dev
        .run_configured(
            kernel,
            &[buf_a_ref, buf_b_ref, &out],
            [(n as u32).div_ceil(64), (m as u32).div_ceil(64), 1],
            Some(bytemuck::bytes_of(&dims)),
        )
        .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: Arc::new(out),
        len: m * n,
    })
}

// ---------------------------------------------------------------------------
// ScryGpuBackend — public type
// ---------------------------------------------------------------------------

/// GPU-accelerated backend for scry-llm using scry-gpu compute shaders.
///
/// Storage is [`ScryGpuStorage`], which can hold either CPU or GPU-resident
/// data. Use [`ScryGpuBackend::to_gpu`] / [`ScryGpuBackend::to_cpu`] to move
/// tensor data between residencies explicitly.
pub struct ScryGpuBackend;

impl ScryGpuBackend {
    /// Move storage to GPU residency. No-op if already on GPU.
    /// Returns `Err` if scry-gpu is unavailable.
    pub fn to_gpu(storage: &ScryGpuStorage) -> Result<ScryGpuStorage, String> {
        match storage {
            ScryGpuStorage::Gpu { .. } => Ok(storage.clone()),
            ScryGpuStorage::Cpu(v) => {
                let ctx = get_ctx().ok_or_else(|| "scry-gpu unavailable".to_string())?;
                let buf = ctx
                    .dev
                    .upload::<f32>(v)
                    .map_err(|e| format!("scry-gpu upload: {e}"))?;
                let len = v.len();
                Ok(ScryGpuStorage::Gpu {
                    buf: Arc::new(buf),
                    len,
                })
            }
        }
    }

    /// Move storage to CPU residency. No-op if already on CPU.
    pub fn to_cpu(storage: &ScryGpuStorage) -> ScryGpuStorage {
        match storage {
            ScryGpuStorage::Cpu(_) => storage.clone(),
            ScryGpuStorage::Gpu { buf, .. } => {
                let v = buf.download().expect("scry-gpu: download failed in to_cpu");
                ScryGpuStorage::Cpu(v)
            }
        }
    }
}

impl DeviceBackend for ScryGpuBackend {
    type Storage = ScryGpuStorage;
    type Stream = ();
    #[cfg(feature = "quantize")]
    type I8Storage = Vec<i8>;

    #[cfg(feature = "quantize")]
    fn i8_from_vec(data: Vec<i8>) -> Vec<i8> {
        data
    }
    #[cfg(feature = "quantize")]
    fn i8_to_vec(storage: &[i8]) -> Vec<i8> {
        storage.to_vec()
    }

    fn zeros(shape: &Shape) -> ScryGpuStorage {
        ScryGpuStorage::Cpu(CpuBackend::zeros(shape))
    }
    fn ones(shape: &Shape) -> ScryGpuStorage {
        ScryGpuStorage::Cpu(CpuBackend::ones(shape))
    }
    fn from_vec(data: Vec<f32>, shape: &Shape) -> ScryGpuStorage {
        ScryGpuStorage::Cpu(CpuBackend::from_vec(data, shape))
    }
    fn to_vec(storage: &ScryGpuStorage) -> Vec<f32> {
        storage.materialize()
    }
    fn into_vec(storage: ScryGpuStorage) -> Vec<f32> {
        match storage {
            ScryGpuStorage::Cpu(v) => v,
            ScryGpuStorage::Gpu { buf, .. } => buf
                .download()
                .expect("scry-gpu: download failed in into_vec"),
        }
    }
    fn as_slice(storage: &ScryGpuStorage) -> Cow<'_, [f32]> {
        match storage {
            ScryGpuStorage::Cpu(v) => Cow::Borrowed(v.as_slice()),
            ScryGpuStorage::Gpu { .. } => Cow::Owned(storage.materialize()),
        }
    }
    fn clone_storage(storage: &ScryGpuStorage) -> ScryGpuStorage {
        storage.clone()
    }
}

/// Wrap a CPU result in `ScryGpuStorage::Cpu`.
fn cpu(v: Vec<f32>) -> ScryGpuStorage {
    ScryGpuStorage::Cpu(v)
}

impl MathBackend for ScryGpuBackend {
    fn matmul(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> ScryGpuStorage {
        // Try the GPU-resident path first when the workload clears the
        // size threshold. Either input being already on-device tilts the
        // tradeoff further toward keeping the result on-device.
        if should_use_gpu(m, k, n) {
            if let Some(gpu_out) = gpu_matmul_persistent(a, b, m, k, n, trans_a, trans_b) {
                return gpu_out;
            }
        }
        // Fallback: CPU compute (or cuBLAS via the legacy materialize path).
        let av = a.as_vec();
        let bv = b.as_vec();
        cpu(matmul_gpu_or_cpu(&av, &bv, m, k, n, trans_a, trans_b))
    }

    fn add(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> ScryGpuStorage {
        cpu(CpuBackend::add(
            &a.as_vec(),
            &b.as_vec(),
            a_shape,
            b_shape,
            out_shape,
        ))
    }

    fn softmax(input: &ScryGpuStorage, shape: &Shape) -> ScryGpuStorage {
        cpu(CpuBackend::softmax(&input.as_vec(), shape))
    }

    fn layernorm(
        input: &ScryGpuStorage,
        gamma: &ScryGpuStorage,
        beta: &ScryGpuStorage,
        shape: &Shape,
        eps: f32,
    ) -> (ScryGpuStorage, ScryGpuStorage, ScryGpuStorage) {
        let (out, mean, rstd) =
            CpuBackend::layernorm(&input.as_vec(), &gamma.as_vec(), &beta.as_vec(), shape, eps);
        (cpu(out), cpu(mean), cpu(rstd))
    }

    fn gelu(input: &ScryGpuStorage) -> ScryGpuStorage {
        cpu(CpuBackend::gelu(&input.as_vec()))
    }

    fn embedding(
        weight: &ScryGpuStorage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> ScryGpuStorage {
        cpu(CpuBackend::embedding(&weight.as_vec(), indices, vocab, dim))
    }

    fn sum(input: &ScryGpuStorage) -> f32 {
        CpuBackend::sum(&input.as_vec())
    }

    fn mul_elementwise(a: &ScryGpuStorage, b: &ScryGpuStorage) -> ScryGpuStorage {
        cpu(CpuBackend::mul_elementwise(&a.as_vec(), &b.as_vec()))
    }

    fn scale(a: &ScryGpuStorage, scalar: f32) -> ScryGpuStorage {
        cpu(CpuBackend::scale(&a.as_vec(), scalar))
    }

    fn concat_rows(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        a_rows: usize,
        b_rows: usize,
        cols: usize,
    ) -> ScryGpuStorage {
        cpu(CpuBackend::concat_rows(
            &a.as_vec(),
            &b.as_vec(),
            a_rows,
            b_rows,
            cols,
        ))
    }

    fn rmsnorm(
        input: &ScryGpuStorage,
        weight: &ScryGpuStorage,
        shape: &Shape,
        eps: f32,
    ) -> ScryGpuStorage {
        cpu(CpuBackend::rmsnorm(
            &input.as_vec(),
            &weight.as_vec(),
            shape,
            eps,
        ))
    }

    fn rope(
        input: &ScryGpuStorage,
        shape: &Shape,
        pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> ScryGpuStorage {
        cpu(CpuBackend::rope(
            &input.as_vec(),
            shape,
            pos,
            head_dim,
            theta,
        ))
    }

    fn rope_with_freqs_preloaded(
        input: &ScryGpuStorage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &ScryGpuStorage,
    ) -> ScryGpuStorage {
        cpu(CpuBackend::rope_with_freqs_preloaded(
            &input.as_vec(),
            seq,
            n_heads,
            head_dim,
            start_pos,
            &freqs.as_vec(),
        ))
    }

    fn swiglu(gate: &ScryGpuStorage, up: &ScryGpuStorage) -> ScryGpuStorage {
        cpu(CpuBackend::swiglu(&gate.as_vec(), &up.as_vec()))
    }

    fn repeat_kv(
        input: &ScryGpuStorage,
        n_kv_heads: usize,
        n_q_heads: usize,
        seq: usize,
        d_head: usize,
    ) -> ScryGpuStorage {
        cpu(CpuBackend::repeat_kv(
            &input.as_vec(),
            n_kv_heads,
            n_q_heads,
            seq,
            d_head,
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_storage_round_trip() {
        let s = ScryGpuStorage::Cpu(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(!s.is_gpu());
        assert_eq!(s.len(), 4);
        let cpu_again = ScryGpuBackend::to_cpu(&s);
        assert!(matches!(cpu_again, ScryGpuStorage::Cpu(_)));
        assert_eq!(ScryGpuBackend::to_vec(&cpu_again), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gpu_round_trip_preserves_data() {
        // Skips gracefully if no GPU is present.
        let original: Vec<f32> = (0..64).map(|i| i as f32 * 0.5 - 1.0).collect();
        let s = ScryGpuStorage::Cpu(original.clone());

        let gpu = match ScryGpuBackend::to_gpu(&s) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_round_trip_preserves_data: {e}");
                return;
            }
        };
        assert!(gpu.is_gpu(), "to_gpu should produce Gpu variant");
        assert_eq!(gpu.len(), original.len());

        let back = ScryGpuBackend::to_cpu(&gpu);
        assert!(matches!(back, ScryGpuStorage::Cpu(_)));
        assert_eq!(ScryGpuBackend::to_vec(&back), original);
    }

    #[test]
    fn to_gpu_is_idempotent() {
        let s = ScryGpuStorage::Cpu(vec![1.0; 8]);
        let gpu = match ScryGpuBackend::to_gpu(&s) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping to_gpu_is_idempotent: {e}");
                return;
            }
        };
        // Calling to_gpu on a GPU-resident storage should not re-upload.
        let gpu2 = ScryGpuBackend::to_gpu(&gpu).expect("idempotent to_gpu");
        assert!(gpu2.is_gpu());
        assert_eq!(gpu2.len(), 8);
    }

    #[test]
    fn large_matmul_returns_gpu_resident_result() {
        // Big enough to clear should_use_gpu (M*K*N >= 65536).
        // 64 * 64 * 32 = 131,072 elements.
        let m = 64;
        let k = 64;
        let n = 32;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.002).collect();
        let a_storage = ScryGpuStorage::Cpu(a);
        let b_storage = ScryGpuStorage::Cpu(b);

        // If the GPU is available we expect a Gpu-variant result; if not,
        // gpu_matmul_persistent returns None and we fall back to Cpu.
        let result = ScryGpuBackend::matmul(&a_storage, &b_storage, m, k, n, false, false);
        if get_ctx().is_some() {
            assert!(
                result.is_gpu(),
                "matmul above threshold with GPU available should return Gpu variant, got {result:?}"
            );
            assert_eq!(result.len(), m * n);
        } else {
            eprintln!("skipping gpu-residency assertion: no GPU");
        }
    }

    #[test]
    fn chained_matmuls_stay_on_gpu() {
        // Two matmuls in a row: (A @ B) @ C. The intermediate must remain
        // GPU-resident so we don't pay a download/upload between them.
        let m = 64;
        let k = 64;
        let n = 64;
        let p = 32;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.002).collect();
        let c: Vec<f32> = (0..n * p).map(|i| (i as f32) * 0.003).collect();

        let a_s = ScryGpuStorage::Cpu(a);
        let b_s = ScryGpuStorage::Cpu(b);
        let c_s = ScryGpuStorage::Cpu(c);

        let ab = ScryGpuBackend::matmul(&a_s, &b_s, m, k, n, false, false);
        if get_ctx().is_none() {
            eprintln!("skipping chained_matmuls_stay_on_gpu: no GPU");
            return;
        }
        assert!(ab.is_gpu(), "intermediate AB should be Gpu-resident");

        let abc = ScryGpuBackend::matmul(&ab, &c_s, m, n, p, false, false);
        assert!(abc.is_gpu(), "final ABC should also be Gpu-resident");
        assert_eq!(abc.len(), m * p);

        // Compare against pure CPU baseline.
        let cpu_ab = CpuBackend::matmul(
            &ScryGpuBackend::to_vec(&a_s),
            &ScryGpuBackend::to_vec(&b_s),
            m,
            k,
            n,
            false,
            false,
        );
        let cpu_abc = CpuBackend::matmul(
            &cpu_ab,
            &ScryGpuBackend::to_vec(&c_s),
            m,
            n,
            p,
            false,
            false,
        );
        let gpu_abc = ScryGpuBackend::to_vec(&abc);
        assert_eq!(cpu_abc.len(), gpu_abc.len());
        for (i, (e, g)) in cpu_abc.iter().zip(gpu_abc.iter()).enumerate() {
            // Relative tolerance: two chained 64×64 matmuls accumulate ~4k
            // fp32 multiply-adds per output, so absolute differences scale
            // with magnitude.
            let tol = 1e-4 * e.abs().max(1.0);
            assert!(
                (e - g).abs() < tol,
                "mismatch at {i}: cpu={e} gpu={g} (tol={tol})"
            );
        }
    }

    #[test]
    fn matmul_with_transpose_a_on_gpu_matches_cpu() {
        // Tests that the on-device transpose helper plumbs correctly.
        let m = 64;
        let k = 64;
        let n = 32;
        // Provide A in transposed layout [k, m]; matmul will transpose to [m, k].
        let a_transposed: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.002).collect();
        let a_storage = ScryGpuStorage::Cpu(a_transposed.clone());
        let b_storage = ScryGpuStorage::Cpu(b.clone());

        let gpu_result = ScryGpuBackend::matmul(&a_storage, &b_storage, m, k, n, true, false);
        if get_ctx().is_none() {
            eprintln!("skipping matmul_with_transpose_a_on_gpu_matches_cpu: no GPU");
            return;
        }
        let cpu_result = CpuBackend::matmul(&a_transposed, &b, m, k, n, true, false);
        let g = ScryGpuBackend::to_vec(&gpu_result);
        assert_eq!(cpu_result.len(), g.len());
        for (i, (c, gv)) in cpu_result.iter().zip(g.iter()).enumerate() {
            let tol = 1e-4 * c.abs().max(1.0);
            assert!(
                (c - gv).abs() < tol,
                "mismatch at {i}: cpu={c} gpu={gv} (tol={tol})"
            );
        }
    }

    #[test]
    fn matmul_through_gpu_resident_inputs_matches_cpu() {
        // Even though matmul currently materializes inputs to CPU, this proves
        // the API accepts Gpu-variant inputs and produces correct results.
        let a: Vec<f32> = (0..8 * 16).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..16 * 4).map(|i| (i as f32) * 0.02).collect();

        let a_storage = ScryGpuStorage::Cpu(a.clone());
        let b_storage = ScryGpuStorage::Cpu(b.clone());

        let cpu_result = ScryGpuBackend::matmul(&a_storage, &b_storage, 8, 16, 4, false, false);

        // Try with GPU-resident inputs; skip if no GPU.
        let a_gpu = match ScryGpuBackend::to_gpu(&a_storage) {
            Ok(g) => g,
            Err(_) => return,
        };
        let b_gpu = ScryGpuBackend::to_gpu(&b_storage).expect("upload b");
        let gpu_inputs_result = ScryGpuBackend::matmul(&a_gpu, &b_gpu, 8, 16, 4, false, false);

        let cv = ScryGpuBackend::to_vec(&cpu_result);
        let gv = ScryGpuBackend::to_vec(&gpu_inputs_result);
        assert_eq!(cv.len(), gv.len());
        for (i, (c, g)) in cv.iter().zip(gv.iter()).enumerate() {
            assert!((c - g).abs() < 1e-4, "mismatch at {i}: cpu={c} gpu={g}");
        }
    }
}
