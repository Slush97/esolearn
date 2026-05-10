//! scry-gpu compute backend — GPU-accelerated matmul and elementwise ops
//! via scry-gpu.
//!
//! Storage is an enum [`ScryGpuStorage`] that can hold either CPU or
//! GPU-resident data. Ops with on-device kernels (matmul, GELU) keep
//! results on the GPU when the workload clears their size threshold;
//! ops without a kernel yet materialize inputs to CPU and return a
//! CPU variant. Use [`ScryGpuBackend::to_gpu`] / [`ScryGpuBackend::to_cpu`]
//! for explicit transfers.
//!
//! With the `scry-gpu-cuda` feature, matmul uses cuBLAS SGEMM (~2x faster
//! than the best Vulkan compute shader). Transpose and GELU dispatch to
//! NVRTC-compiled CUDA kernels so the chain stays GPU-resident.
//!
//! Because `MathBackend` trait methods are static (no `&self`), we store
//! the GPU context in a `OnceLock` initialized on first use.

use std::borrow::Cow;
#[cfg(any(feature = "scry-gpu-bf16", feature = "scry-gpu-cudnn"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "scry-gpu-bf16")]
use std::sync::OnceLock as BufOnceLock;
use std::sync::{Arc, OnceLock};

use scry_gpu::Buffer;

use crate::backend::cpu::CpuBackend;
use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;

/// Minimum M*K*N product before engaging GPU (below this, CPU/BLAS is faster
/// due to per-dispatch overhead: buffer creation, submission, readback).
///
/// Empirical crossover on RTX 5070 Ti / cuBLAS / async dispatch is ~50K FMAs
/// for an isolated matmul; in chains the GPU wins from much lower sizes
/// because the kernel dispatch is the only cost (no upload, no sync until
/// the chain ends). 32_768 splits the difference: side ≥ 32 goes GPU,
/// which loses ~6 µs/op at exactly side 32 in single-op mode but wins
/// big in chains. See `benches/threshold_sweep.rs`.
const GPU_MIN_ELEMENTS: usize = 32_768;

/// Maximum GPU buffer size in bytes (128 MiB).
const MAX_GPU_BUFFER_BYTES: u64 = 128 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Storage enum — CPU Vec or GPU-resident buffer
// ---------------------------------------------------------------------------

/// GPU tensor storage: an `Arc<Buffer<f32>>` plus an optional lazily-cached
/// bf16 shadow.
///
/// The bf16 shadow is materialized on first access via
/// [`as_gpu_buffer_bf16`] and reused on subsequent calls — this is what
/// keeps weight tensors from being re-cast on every matmul once the bf16
/// fast-path is engaged. For activation tensors, the shadow is created
/// once per Buffer and dropped when the surrounding Arc drops; cost is
/// the same as the per-call cast we'd otherwise do.
pub struct GpuTensorStorage {
    pub(crate) f32: Arc<Buffer<f32>>,
    #[cfg(feature = "scry-gpu-bf16")]
    pub(crate) bf16: BufOnceLock<Arc<Buffer<half::bf16>>>,
}

impl GpuTensorStorage {
    fn from_owned(buf: Buffer<f32>) -> Arc<Self> {
        Self::from_arc(Arc::new(buf))
    }

    fn from_arc(buf: Arc<Buffer<f32>>) -> Arc<Self> {
        Arc::new(Self {
            f32: buf,
            #[cfg(feature = "scry-gpu-bf16")]
            bf16: BufOnceLock::new(),
        })
    }
}

/// Storage for [`ScryGpuBackend`] tensors. Either a CPU `Vec<f32>` or a
/// reference-counted GPU buffer (with optional bf16 shadow).
///
/// `Clone` on the GPU variant is cheap (`Arc::clone`); on the CPU variant
/// it clones the underlying `Vec`.
#[derive(Clone)]
pub enum ScryGpuStorage {
    /// Host-resident data.
    Cpu(Vec<f32>),
    /// Device-resident buffer. The Arc lets multiple tensors share the
    /// same allocation (e.g. after a clone) without re-uploading, and
    /// carries the optional bf16 shadow alongside.
    Gpu {
        buf: Arc<GpuTensorStorage>,
        len: usize,
    },
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
                .f32
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
    /// On-device transpose kernel. Populated for both WGSL and CUDA paths.
    transpose: Option<::scry_gpu::Kernel>,
    /// On-device GELU (tanh approximation). Populated for both paths.
    gelu: Option<::scry_gpu::Kernel>,
    /// On-device ReLU. CUDA-only — Vulkan path is `None` and callers fall
    /// back to CPU. Used by `scry-vision`'s `relu` to keep ResNet activations
    /// device-resident between conv layers.
    relu: Option<::scry_gpu::Kernel>,
    /// On-device SiLU / Swish (`x * sigmoid(x)`). CUDA-only — Vulkan path is
    /// `None` and callers fall back to CPU. Used by SD UNet ResBlocks; same
    /// elementwise shape as [`Self::gelu`].
    silu: Option<::scry_gpu::Kernel>,
    /// On-device exact (erf-based) GELU. CUDA-only — Vulkan path is `None`
    /// and callers fall back to CPU. Used by SD UNet's GeGLU MLP, which is
    /// the single largest cost in the UNet forward (47% at 512×512 / bf16);
    /// the tanh approximation in [`Self::gelu`] drifts enough across 16
    /// transformer blocks to break the M6 1e-3 parity gate.
    gelu_exact: Option<::scry_gpu::Kernel>,
    /// On-device row-wise softmax. CUDA-only; Vulkan path is `None` and
    /// callers fall back to CPU.
    softmax: Option<::scry_gpu::Kernel>,
    /// On-device fused scaled-softmax: `softmax(scale · x)` along the last
    /// axis in one kernel. Replaces the trait default's two-pass dispatch
    /// (separate `scale` then `softmax`), saving the `(read + write)` of
    /// the intermediate scaled tensor — at SD's deepest self-attn stage
    /// (`[B=8, n=4096, n=4096]`) that's 1.07 GB of memory traffic per call.
    /// CUDA-only; Vulkan path is `None`.
    scaled_softmax: Option<::scry_gpu::Kernel>,
    /// On-device row-wise layernorm with affine gamma/beta. CUDA-only; Vulkan
    /// path is `None` and callers fall back to CPU.
    layernorm: Option<::scry_gpu::Kernel>,
    /// On-device 2D batch-normalization (inference) with stored running stats.
    /// CUDA-only; Vulkan path is `None` and callers fall back to CPU.
    batchnorm: Option<::scry_gpu::Kernel>,
    /// On-device im2col lowering for 2D convolution. CUDA-only; Vulkan path
    /// is `None` and callers fall back to CPU.
    im2col: Option<::scry_gpu::Kernel>,
    /// On-device column-broadcast bias add (`out[r,c] = a[r,c] + bias[r]`).
    /// Used to keep Conv2d's bias add GPU-resident after the matmul. CUDA-only.
    add_row_bias: Option<::scry_gpu::Kernel>,
    /// On-device row-broadcast bias add (`out[r,c] = a[r,c] + bias[c]`).
    /// Used by `MathBackend::matmul_bias` after the on-device matmul, so
    /// the result stays on the device instead of round-tripping through
    /// host for the bias loop. CUDA-only. The `BIAS_ADD_CUDA` shader has
    /// existed in scry-gpu since the original wiring; this slot connects
    /// it to the trait method, which had no GPU override and was firing
    /// ~104 host roundtrips per SD UNet forward (FF × 48, attn out_proj
    /// × 32, resblock time_emb_proj × 22, time_embed × 2).
    bias_add: Option<::scry_gpu::Kernel>,
    /// On-device same-shape elementwise add (`out[i] = a[i] + b[i]`). Used
    /// for ResNet residual adds and other identical-shape sums where the
    /// row-bias broadcast doesn't apply. CUDA-only.
    add_elementwise: Option<::scry_gpu::Kernel>,
    /// On-device same-shape elementwise multiply (`out[i] = a[i] * b[i]`).
    /// Used by SD UNet's GeGLU `values * gelu(gate)` step — the previous
    /// trait default downloaded both operands, multiplied on host, and
    /// uploaded back. CUDA-only.
    mul_elementwise: Option<::scry_gpu::Kernel>,
    /// On-device 2D max-pool with fixed kernel/stride/padding. CUDA-only;
    /// Vulkan path is `None` and callers fall back to CPU.
    max_pool: Option<::scry_gpu::Kernel>,
    /// On-device adaptive 2D average-pool to a fixed `[h_out, w_out]`.
    /// CUDA-only; Vulkan path is `None` and callers fall back to CPU.
    adaptive_avg_pool: Option<::scry_gpu::Kernel>,
    /// On-device 2D nearest-neighbor upsample by an integer factor.
    /// CUDA-only; Vulkan path is `None` and callers fall back to CPU.
    /// Used by SD UNet UpBlocks and the VAE decoder.
    upsample_nearest: Option<::scry_gpu::Kernel>,
    /// On-device group normalization (inference) with per-channel affine.
    /// CUDA-only; Vulkan path is `None`. Used by SD UNet ResBlocks and the
    /// VAE decoder; SD always uses `num_groups = 32`.
    group_norm: Option<::scry_gpu::Kernel>,
    /// On-device fused-QKV split + per-head reshape for transformer
    /// attention. CUDA-only; Vulkan path is `None`.
    split_qkv_heads: Option<::scry_gpu::Kernel>,
    /// On-device reshape `[n_heads, seq, d_head]` → `[seq, n_heads*d_head]`.
    /// CUDA-only; Vulkan path is `None`.
    reshape_from_heads: Option<::scry_gpu::Kernel>,
    /// On-device reshape `[seq, n_heads*d_head]` → `[n_heads, seq, d_head]`.
    /// Forward direction of [`Self::reshape_from_heads`]. CUDA-only.
    /// Used by SD UNet attention to permute Q/K/V before the per-head
    /// matmul batch — replaces a per-head `gather_columns` loop that
    /// went through host on every dispatch.
    reshape_to_heads: Option<::scry_gpu::Kernel>,
    /// On-device row concatenation `[a_rows, cols] + [b_rows, cols] →
    /// [a_rows + b_rows, cols]`. CUDA-only. Used by SD UNet's UpBlock
    /// skip-concat (12×/forward) and by the Llama KV cache —
    /// replaces a `concat_rows` default that round-tripped both inputs
    /// through host.
    concat_rows: Option<::scry_gpu::Kernel>,
    /// On-device elementwise scale `out[i] = in[i] * alpha`. Used to keep
    /// `scaled_softmax`'s pre-softmax scale step on-device. CUDA-only.
    scale: Option<::scry_gpu::Kernel>,
    /// On-device embedding lookup: gathers rows of a `[vocab, dim]` weight
    /// table by index. CUDA-only; Vulkan path is `None` and callers fall
    /// back to CPU. Used by `ScryGpuBackend::embedding` so SD CLIP's token
    /// and positional lookups don't download the full token table to host
    /// on every text-encode (~145 MiB at SD 1.5).
    embedding: Option<::scry_gpu::Kernel>,
    /// On-device contiguous-column gather (`out[r, c] = in[r, col_start + c]`).
    /// CUDA-only. Powers the per-head Q/K/V split inside
    /// `CausalSelfAttention::forward` — 432 calls per CLIP text-encode at
    /// SD 1.5, each round-tripping the full `[77, 2304]` qkv tensor on the
    /// trait default before this override existed.
    gather_columns: Option<::scry_gpu::Kernel>,
    /// On-device additive contiguous-column scatter
    /// (`dst[r, col_start + c] += src[r, c]`). CUDA-only. Mirror of
    /// [`Self::gather_columns`]; used to merge per-head attention outputs
    /// back into a `[seq, d_model]` accumulator. 144 calls per CLIP encode
    /// at SD 1.5; trait default downloaded both `dst` and `src` per call.
    scatter_columns: Option<::scry_gpu::Kernel>,
    /// On-device causal-mask + scale fused over a `[seq, seq]` score
    /// matrix. CUDA-only. Replaces the trait default's host loop so the
    /// scaled-and-masked scores stay device-resident through the softmax
    /// → matmul chain. 144 calls per CLIP encode at SD 1.5.
    apply_causal_mask_and_scale: Option<::scry_gpu::Kernel>,
    /// f32 → bf16 elementwise cast. Used to feed the bf16 GemmEx fast-path.
    /// CUDA + `scry-gpu-bf16` only. The reverse (bf16 → f32) cast is no
    /// longer needed on the matmul path — `cublas_matmul_bf16_in_f32_out_async`
    /// writes the fp32 accumulator directly. The CAST_BF16_F32_CUDA shader
    /// is still in scry-gpu for future bf16-storage paths.
    #[cfg(feature = "scry-gpu-bf16")]
    cast_f32_bf16: Option<::scry_gpu::Kernel>,
    /// Opt-in flag for routing every matmul through the bf16 GemmEx fast-path
    /// (cast→GemmEx→cast). Initialized from `SCRY_GPU_MATMUL_BF16` and
    /// runtime-flippable via [`ScryGpuBackend::set_bf16_matmul`] for
    /// side-by-side benches.
    #[cfg(feature = "scry-gpu-bf16")]
    bf16_matmul_enabled: AtomicBool,
    /// Opt-out flag for the cuDNN conv path. Defaults to `true` when the
    /// `scry-gpu-cudnn` feature is on. Lets benches toggle between cuDNN
    /// implicit-GEMM and the legacy im2col + cuBLAS chain on the same model
    /// instance for clean A/B numbers.
    #[cfg(feature = "scry-gpu-cudnn")]
    cudnn_conv_enabled: AtomicBool,
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

    // CUDA path: cuBLAS for matmul + NVRTC-compiled helper kernels so the
    // persistent chain stays GPU-resident.
    #[cfg(feature = "scry-gpu-cuda")]
    if dev.backend_kind() == ::scry_gpu::BackendKind::Cuda {
        let transpose = dev
            .compile_cuda(
                ::scry_gpu::shaders::backward::TRANSPOSE_CUDA,
                "transpose_2d",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: transpose_cuda compile: {e}"))?;
        let gelu = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::GELU_CUDA,
                "gelu",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: gelu_cuda compile: {e}"))?;
        let relu = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::RELU_CUDA,
                "relu",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: relu_cuda compile: {e}"))?;
        let silu = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::SILU_CUDA,
                "silu",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: silu_cuda compile: {e}"))?;
        let gelu_exact = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::GELU_EXACT_CUDA,
                "gelu_exact",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: gelu_exact_cuda compile: {e}"))?;
        let softmax = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::SOFTMAX_ROWWISE_CUDA,
                "softmax_rowwise",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: softmax_cuda compile: {e}"))?;
        let scaled_softmax = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::SCALED_SOFTMAX_ROWWISE_CUDA,
                "scaled_softmax_rowwise",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: scaled_softmax_cuda compile: {e}"))?;
        let layernorm = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::LAYERNORM_ROWWISE_CUDA,
                "layernorm_rowwise",
                6,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: layernorm_cuda compile: {e}"))?;
        let batchnorm = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::BATCHNORM_INFERENCE_CUDA,
                "batchnorm_inference",
                6,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: batchnorm_cuda compile: {e}"))?;
        let im2col = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::IM2COL_NCHW_CUDA,
                "im2col_nchw",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: im2col_cuda compile: {e}"))?;
        let add_row_bias = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::ADD_ROW_BIAS_CUDA,
                "add_row_bias",
                3,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: add_row_bias_cuda compile: {e}"))?;
        let bias_add = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::BIAS_ADD_CUDA,
                "bias_add",
                3,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: bias_add_cuda compile: {e}"))?;
        let add_elementwise = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::ADD_ELEMENTWISE_CUDA,
                "add_elementwise",
                3,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: add_elementwise_cuda compile: {e}"))?;
        let mul_elementwise = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::MUL_ELEMENTWISE_CUDA,
                "mul_elementwise",
                3,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: mul_elementwise_cuda compile: {e}"))?;
        let max_pool = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::MAXPOOL_2D_CUDA,
                "maxpool_2d",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: maxpool_2d_cuda compile: {e}"))?;
        let adaptive_avg_pool = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::ADAPTIVE_AVG_POOL_2D_CUDA,
                "adaptive_avg_pool_2d",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: adaptive_avg_pool_2d_cuda compile: {e}"))?;
        let upsample_nearest = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::UPSAMPLE_2D_NEAREST_CUDA,
                "upsample_2d_nearest",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: upsample_2d_nearest_cuda compile: {e}"))?;
        let group_norm = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::GROUP_NORM_CUDA,
                "group_norm",
                4,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: group_norm_cuda compile: {e}"))?;
        let split_qkv_heads = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::SPLIT_QKV_RESHAPE_HEADS_CUDA,
                "split_qkv_reshape_heads",
                4,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: split_qkv_reshape_heads_cuda compile: {e}"))?;
        let reshape_from_heads = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::RESHAPE_FROM_HEADS_CUDA,
                "reshape_from_heads",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: reshape_from_heads_cuda compile: {e}"))?;
        let reshape_to_heads = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::RESHAPE_TO_HEADS_CUDA,
                "reshape_to_heads",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: reshape_to_heads_cuda compile: {e}"))?;
        let concat_rows = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::CONCAT_ROWS_CUDA,
                "concat_rows",
                3,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: concat_rows_cuda compile: {e}"))?;
        let scale = dev
            .compile_cuda(
                ::scry_gpu::shaders::backward::SCALE_CUDA,
                "scale_fwd",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: scale_cuda compile: {e}"))?;
        let embedding = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::EMBEDDING_FWD_CUDA,
                "embedding_fwd",
                3,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: embedding_fwd_cuda compile: {e}"))?;
        let gather_columns = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::GATHER_COLUMNS_CUDA,
                "gather_columns",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: gather_columns_cuda compile: {e}"))?;
        let scatter_columns = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::SCATTER_COLUMNS_ADD_CUDA,
                "scatter_columns_add",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: scatter_columns_add_cuda compile: {e}"))?;
        let apply_causal_mask_and_scale = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::APPLY_CAUSAL_MASK_AND_SCALE_CUDA,
                "apply_causal_mask_and_scale",
                1,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: apply_causal_mask_and_scale_cuda compile: {e}"))?;
        #[cfg(feature = "scry-gpu-bf16")]
        let cast_f32_bf16 = dev
            .compile_cuda(
                ::scry_gpu::shaders::elementwise::CAST_F32_BF16_CUDA,
                "cast_f32_bf16",
                2,
                [256, 1, 1],
            )
            .map_err(|e| format!("scry-gpu: cast_f32_bf16 compile: {e}"))?;
        #[cfg(feature = "scry-gpu-bf16")]
        let bf16_matmul_enabled = std::env::var("SCRY_GPU_MATMUL_BF16")
            .ok()
            .as_deref()
            .is_some_and(|v| matches!(v, "1" | "true" | "TRUE" | "on" | "ON"));
        return Ok(ScryCtx {
            dev,
            matmul: MatmulStrategy::CuBlas,
            transpose: Some(transpose),
            gelu: Some(gelu),
            relu: Some(relu),
            silu: Some(silu),
            gelu_exact: Some(gelu_exact),
            softmax: Some(softmax),
            scaled_softmax: Some(scaled_softmax),
            layernorm: Some(layernorm),
            batchnorm: Some(batchnorm),
            im2col: Some(im2col),
            add_row_bias: Some(add_row_bias),
            bias_add: Some(bias_add),
            add_elementwise: Some(add_elementwise),
            mul_elementwise: Some(mul_elementwise),
            max_pool: Some(max_pool),
            adaptive_avg_pool: Some(adaptive_avg_pool),
            upsample_nearest: Some(upsample_nearest),
            group_norm: Some(group_norm),
            split_qkv_heads: Some(split_qkv_heads),
            reshape_from_heads: Some(reshape_from_heads),
            reshape_to_heads: Some(reshape_to_heads),
            concat_rows: Some(concat_rows),
            scale: Some(scale),
            embedding: Some(embedding),
            gather_columns: Some(gather_columns),
            scatter_columns: Some(scatter_columns),
            apply_causal_mask_and_scale: Some(apply_causal_mask_and_scale),
            #[cfg(feature = "scry-gpu-bf16")]
            cast_f32_bf16: Some(cast_f32_bf16),
            #[cfg(feature = "scry-gpu-bf16")]
            bf16_matmul_enabled: AtomicBool::new(bf16_matmul_enabled),
            #[cfg(feature = "scry-gpu-cudnn")]
            cudnn_conv_enabled: AtomicBool::new(true),
        });
    }

    let matmul = dev
        .compile(::scry_gpu::shaders::matmul::COARSE_64X64)
        .map_err(|e| format!("scry-gpu: shader compile: {e}"))?;
    let transpose = dev
        .compile(::scry_gpu::shaders::backward::TRANSPOSE)
        .map_err(|e| format!("scry-gpu: transpose compile: {e}"))?;
    let gelu = dev
        .compile(::scry_gpu::shaders::elementwise::GELU)
        .map_err(|e| format!("scry-gpu: gelu compile: {e}"))?;
    Ok(ScryCtx {
        dev,
        matmul: MatmulStrategy::Wgsl(matmul),
        transpose: Some(transpose),
        gelu: Some(gelu),
        relu: None,
        silu: None,
        gelu_exact: None,
        softmax: None,
        scaled_softmax: None,
        layernorm: None,
        batchnorm: None,
        im2col: None,
        add_row_bias: None,
        bias_add: None,
        add_elementwise: None,
        mul_elementwise: None,
        max_pool: None,
        adaptive_avg_pool: None,
        upsample_nearest: None,
        group_norm: None,
        split_qkv_heads: None,
        reshape_from_heads: None,
        reshape_to_heads: None,
        concat_rows: None,
        scale: None,
        embedding: None,
        gather_columns: None,
        scatter_columns: None,
        apply_causal_mask_and_scale: None,
        #[cfg(feature = "scry-gpu-bf16")]
        cast_f32_bf16: None,
        #[cfg(feature = "scry-gpu-bf16")]
        bf16_matmul_enabled: AtomicBool::new(false),
        #[cfg(feature = "scry-gpu-cudnn")]
        cudnn_conv_enabled: AtomicBool::new(false),
    })
}

// ---------------------------------------------------------------------------
// GPU matmul dispatch
// ---------------------------------------------------------------------------

/// Dispatch a kernel on the active backend, skipping the per-call host sync
/// when running on CUDA. The next `Buffer::download` (or any explicit fence)
/// observes the result; in tight chains this elides one stream sync per op.
/// On Vulkan we keep the synchronous path — its async wins live behind the
/// batched-dispatch port, which is off the active backlog (per
/// `HACKING_GPU_BREAKDOWN.md`, post-cuBLAS section).
fn dispatch_kernel(
    ctx: &ScryCtx,
    kernel: &::scry_gpu::Kernel,
    buffers: &[&dyn ::scry_gpu::GpuBuf],
    workgroups: [u32; 3],
    push_constants: Option<&[u8]>,
) -> ::scry_gpu::Result<()> {
    #[cfg(feature = "scry-gpu-cuda")]
    if matches!(ctx.matmul, MatmulStrategy::CuBlas) {
        return ctx
            .dev
            .run_configured_async(kernel, buffers, workgroups, push_constants);
    }
    ctx.dev
        .run_configured(kernel, buffers, workgroups, push_constants)
}

/// Run a single matmul on GPU. Returns None if GPU is unavailable.
fn gpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Option<Vec<f32>> {
    let ctx = get_ctx()?;

    let sa = ctx.dev.upload(a).ok()?;
    let sb = ctx.dev.upload(b).ok()?;

    match &ctx.matmul {
        MatmulStrategy::Wgsl(kernel) => {
            let sc = ctx.dev.alloc_uninit::<f32>(m * n).ok()?;
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
            let mut sc = ctx.dev.alloc_uninit::<f32>(m * n).ok()?;
            ctx.dev
                .cublas_matmul_async(&sa, &sb, &mut sc, m as u32, n as u32, k as u32)
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
        ScryGpuStorage::Gpu { buf, .. } => Some(Arc::clone(&buf.f32)),
        ScryGpuStorage::Cpu(v) => {
            let ctx = get_ctx()?;
            let buf = ctx.dev.upload::<f32>(v).ok()?;
            Some(Arc::new(buf))
        }
    }
}

/// Acquire a bf16 GPU buffer view of `storage`, lazily materializing the
/// shadow on first call and reusing it on subsequent calls.
///
/// This is what makes weight tensors free in the bf16 fast-path: once a
/// `Conv2d::weight` has been cast to bf16, the shadow is cached on the
/// `GpuTensorStorage` Arc and every later forward pass picks it up via
/// `OnceLock::get`. Activation buffers also benefit (their cast happens
/// once per buffer rather than per matmul) but the win is single-use:
/// the next forward allocates fresh activations.
///
/// Returns `None` if scry-gpu, the bf16 feature, or the cast kernel is
/// unavailable, or if `storage` is `Cpu`-resident (callers should upload
/// to a `Gpu` variant first).
#[cfg(feature = "scry-gpu-bf16")]
fn as_gpu_buffer_bf16(storage: &ScryGpuStorage) -> Option<Arc<Buffer<half::bf16>>> {
    let ScryGpuStorage::Gpu { buf, .. } = storage else {
        return None;
    };
    if let Some(cached) = buf.bf16.get() {
        return Some(Arc::clone(cached));
    }
    let ctx = get_ctx()?;
    let cast_down = ctx.cast_f32_bf16.as_ref()?;

    let n = buf.f32.len();
    let bf16_buf = ctx.dev.alloc_uninit::<half::bf16>(n).ok()?;
    let n_pc: u32 = n as u32;
    dispatch_kernel(
        ctx,
        cast_down,
        &[&*buf.f32, &bf16_buf],
        [n_pc.div_ceil(256), 1, 1],
        Some(bytemuck::bytes_of(&n_pc)),
    )
    .ok()?;
    let bf16_arc = Arc::new(bf16_buf);

    // OnceLock::set returns Err if another thread won the race; in that
    // case our freshly-cast bf16 buffer drops, and we use theirs (the
    // contents are identical up to bit-exact casting).
    match buf.bf16.set(Arc::clone(&bf16_arc)) {
        Ok(()) => Some(bf16_arc),
        Err(_) => Some(Arc::clone(buf.bf16.get().expect("just lost a race; cache populated"))),
    }
}

/// GPU-resident concat along axis 0. Returns `None` when the GPU path
/// is unavailable, the kernel slot is missing (Vulkan), or the
/// workload is below `GPU_ELEMENTWISE_MIN`. Used by SD UNet UpBlock
/// skip-concat and by the Llama KV cache.
fn gpu_concat_rows_persistent(
    a: &ScryGpuStorage,
    b: &ScryGpuStorage,
    a_total: usize,
    b_total: usize,
) -> Option<ScryGpuStorage> {
    let total = a_total.checked_add(b_total)?;
    if total < GPU_ELEMENTWISE_MIN {
        return None;
    }
    if a.len() != a_total || b.len() != b_total {
        return None;
    }
    let ctx = get_ctx()?;
    let kernel = ctx.concat_rows.as_ref()?;
    let buf_a = as_gpu_buffer(a)?;
    let buf_b = as_gpu_buffer(b)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;
    let dims_pc: [u32; 2] = [a_total as u32, b_total as u32];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_a, &*buf_b, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Top-level wrapper for `MathBackend::transpose_2d`: dispatch the
/// CUDA transpose kernel on a `ScryGpuStorage` and return GPU-resident
/// output. Returns `None` when the GPU path is unavailable, the kernel
/// slot is missing, or the workload is below `GPU_ELEMENTWISE_MIN`
/// (matmul callers route through `gpu_transpose` which assumes the
/// GPU path is wanted; this wrapper is for callers that want graceful
/// fallback to CPU). Used by SD UNet's `SpatialTransformer` to permute
/// `[C, H*W] ↔ [H*W, C]` around the transformer-block stack.
fn gpu_transpose_2d_persistent(
    input: &ScryGpuStorage,
    rows: usize,
    cols: usize,
) -> Option<ScryGpuStorage> {
    let total = rows.checked_mul(cols)?;
    if total < GPU_ELEMENTWISE_MIN {
        return None;
    }
    if input.len() != total {
        return None;
    }
    let buf_in = as_gpu_buffer(input)?;
    let out = gpu_transpose(&buf_in, rows, cols)?;
    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Run the on-device transpose kernel. Returns a fresh buffer of shape `[cols, rows]`.
fn gpu_transpose(input: &Buffer<f32>, rows: usize, cols: usize) -> Option<Buffer<f32>> {
    let ctx = get_ctx()?;
    let kernel = ctx.transpose.as_ref()?;
    // Transpose kernel writes every output element; zero-init is wasted work.
    let out = ctx.dev.alloc_uninit::<f32>(rows * cols).ok()?;
    let dims: [u32; 2] = [rows as u32, cols as u32];
    let groups = ((rows * cols) as u32).div_ceil(256);
    dispatch_kernel(
        ctx,
        kernel,
        &[input, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims)),
    )
    .ok()?;
    Some(out)
}

/// Below this element count, elementwise ops are not worth the GPU dispatch
/// overhead.
///
/// Empirical crossover (n=1024 CPU 6 µs / GPU 7.7 µs; n=2048 CPU 12 µs /
/// GPU 8.7 µs) is around 1.5K elements on async-dispatched CUDA. Set just
/// above the breakeven so single-op GELU never regresses; chain-mode GPU
/// wins from any size. The previous 16_384 cutoff was set when GELU
/// dispatch carried a wasted `cuMemsetD8Async` (now skipped via uninit
/// alloc) and per-call sync (now async). See `benches/threshold_sweep.rs`.
const GPU_ELEMENTWISE_MIN: usize = 2_048;

/// Run a single-input elementwise kernel (RELU/GELU/etc) on `input`.
/// Returns a fresh GPU buffer of the same length.
fn run_unary_elementwise(
    kernel: &::scry_gpu::Kernel,
    input: &Buffer<f32>,
    n: usize,
) -> Option<Buffer<f32>> {
    let ctx = get_ctx()?;
    // Elementwise kernels dispatch one thread per output element; every byte
    // is overwritten before any read.
    let out = ctx.dev.alloc_uninit::<f32>(n).ok()?;
    let dims: [u32; 1] = [n as u32];
    let groups = (n as u32).div_ceil(256);
    dispatch_kernel(
        ctx,
        kernel,
        &[input, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims)),
    )
    .ok()?;
    Some(out)
}

/// GPU-resident GELU. Returns `None` when the GPU path is unavailable so the
/// caller can fall back to CPU.
fn gpu_gelu_persistent(input: &ScryGpuStorage) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.gelu.as_ref()?;
    let n = input.len();
    if n < GPU_ELEMENTWISE_MIN {
        return None;
    }
    let buf_in = as_gpu_buffer(input)?;
    let out = run_unary_elementwise(kernel, &buf_in, n)?;
    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: n,
    })
}

/// GPU-resident `ReLU`. Returns `None` when the GPU path is unavailable
/// (e.g. Vulkan, where the kernel slot is `None`) or the workload is below
/// `GPU_ELEMENTWISE_MIN`, so the caller falls back to CPU.
fn gpu_relu_persistent(input: &ScryGpuStorage) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.relu.as_ref()?;
    let n = input.len();
    if n < GPU_ELEMENTWISE_MIN {
        return None;
    }
    let buf_in = as_gpu_buffer(input)?;
    let out = run_unary_elementwise(kernel, &buf_in, n)?;
    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: n,
    })
}

/// GPU-resident SiLU / Swish (`x * sigmoid(x)`). Returns `None` when the
/// GPU path is unavailable (e.g. Vulkan, where the kernel slot is `None`)
/// or the workload is below `GPU_ELEMENTWISE_MIN`, so the caller falls back
/// to CPU. Same dispatch shape as [`gpu_gelu_persistent`].
fn gpu_silu_persistent(input: &ScryGpuStorage) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.silu.as_ref()?;
    let n = input.len();
    if n < GPU_ELEMENTWISE_MIN {
        return None;
    }
    let buf_in = as_gpu_buffer(input)?;
    let out = run_unary_elementwise(kernel, &buf_in, n)?;
    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: n,
    })
}

/// GPU-resident exact (erf-based) GELU. Returns `None` when the GPU path
/// is unavailable or the workload is below `GPU_ELEMENTWISE_MIN`, so the
/// caller falls back to CPU. Same dispatch shape as
/// [`gpu_gelu_persistent`]; the only kernel difference is `erff(x/sqrt(2))`
/// instead of the tanh-approximation polynomial.
///
/// This is the hot-path replacement for `unet/common.rs::exact_gelu`,
/// which used to round-trip every gate tensor through host. The SD UNet's
/// GeGLU MLP calls this 16× per forward — at 512×512 the deepest stage's
/// gate tensor is `[4096, 5120]` = 80 MB per call.
fn gpu_gelu_exact_persistent(input: &ScryGpuStorage) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.gelu_exact.as_ref()?;
    let n = input.len();
    if n < GPU_ELEMENTWISE_MIN {
        return None;
    }
    let buf_in = as_gpu_buffer(input)?;
    let out = run_unary_elementwise(kernel, &buf_in, n)?;
    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: n,
    })
}

/// Minimum row count before engaging the GPU softmax path. Below this, CPU
/// rayon parallelism + cache locality wins; the per-launch grid setup
/// dominates a small handful of rows.
const GPU_SOFTMAX_MIN_ROWS: usize = 32;

/// GPU-resident row-wise softmax over the last dimension. CUDA-only — the
/// Vulkan path returns `None` so the caller falls back to CPU. Returns `None`
/// also when the GPU path is unavailable, the workload is below threshold,
/// or the input is empty / 1-D with d == 0.
fn gpu_softmax_persistent(input: &ScryGpuStorage, shape: &Shape) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.softmax.as_ref()?;
    let dims = shape.dims();
    let d = *dims.last()?;
    if d == 0 {
        return None;
    }
    let total = input.len();
    if total == 0 || total % d != 0 {
        return None;
    }
    let n_rows = total / d;
    if n_rows < GPU_SOFTMAX_MIN_ROWS {
        return None;
    }
    let buf_in = as_gpu_buffer(input)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;
    let dims_pc: [u32; 2] = [n_rows as u32, d as u32];
    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [n_rows as u32, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;
    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// GPU-resident fused scaled-softmax. CUDA-only — the Vulkan path returns
/// `None` so the caller falls back to the trait default (separate scale +
/// softmax). Shares [`GPU_SOFTMAX_MIN_ROWS`] threshold semantics.
fn gpu_scaled_softmax_persistent(
    input: &ScryGpuStorage,
    scale: f32,
    shape: &Shape,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.scaled_softmax.as_ref()?;
    let dims = shape.dims();
    let d = *dims.last()?;
    if d == 0 {
        return None;
    }
    let total = input.len();
    if total == 0 || total % d != 0 {
        return None;
    }
    let n_rows = total / d;
    if n_rows < GPU_SOFTMAX_MIN_ROWS {
        return None;
    }
    let buf_in = as_gpu_buffer(input)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;
    // Push constants: [n_rows: u32, d: u32, scale: f32-as-bits].
    // Same packing pattern as `gpu_layernorm_persistent`'s `eps` arg.
    let pc: [u32; 3] = [n_rows as u32, d as u32, scale.to_bits()];
    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [n_rows as u32, 1, 1],
        Some(bytemuck::bytes_of(&pc)),
    )
    .ok()?;
    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Minimum row count before engaging the GPU layernorm path. Below this, CPU
/// rayon parallelism wins; the per-launch grid setup dominates a small handful
/// of rows. Set to match `GPU_SOFTMAX_MIN_ROWS` since the kernel shape is the
/// same (one block per row, two reductions).
const GPU_LAYERNORM_MIN_ROWS: usize = 32;

/// GPU-resident row-wise layernorm with affine gamma/beta. CUDA-only — the
/// Vulkan path returns `None` so the caller falls back to CPU. Returns `None`
/// also when the GPU path is unavailable, the workload is below threshold,
/// the input is empty / 1-D with d == 0, or `gamma`/`beta` don't match the
/// last-dim length.
fn gpu_layernorm_persistent(
    input: &ScryGpuStorage,
    gamma: &ScryGpuStorage,
    beta: &ScryGpuStorage,
    shape: &Shape,
    eps: f32,
) -> Option<(ScryGpuStorage, ScryGpuStorage, ScryGpuStorage)> {
    let ctx = get_ctx()?;
    let kernel = ctx.layernorm.as_ref()?;
    let dims = shape.dims();
    let d = *dims.last()?;
    if d == 0 {
        return None;
    }
    let total = input.len();
    if total == 0 || total % d != 0 {
        return None;
    }
    let n_rows = total / d;
    if n_rows < GPU_LAYERNORM_MIN_ROWS {
        return None;
    }
    if gamma.len() != d || beta.len() != d {
        return None;
    }

    let buf_in = as_gpu_buffer(input)?;
    let buf_g = as_gpu_buffer(gamma)?;
    let buf_b = as_gpu_buffer(beta)?;

    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;
    let means = ctx.dev.alloc_uninit::<f32>(n_rows).ok()?;
    let rstds = ctx.dev.alloc_uninit::<f32>(n_rows).ok()?;

    // Push constants pack [n_rows: u32, d: u32, eps: f32]. The CUDA dispatch
    // path treats push-constant bytes as a stream of u32 kernel args, so the
    // f32 bits are passed transparently and the kernel reads them per its
    // signature. See `crates/scry-gpu/src/backend/cuda.rs::launch_on_stream`.
    let dims_pc: [u32; 3] = [n_rows as u32, d as u32, eps.to_bits()];

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &*buf_g, &*buf_b, &out, &means, &rstds],
        [n_rows as u32, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some((
        ScryGpuStorage::Gpu {
            buf: GpuTensorStorage::from_owned(out),
            len: total,
        },
        ScryGpuStorage::Gpu {
            buf: GpuTensorStorage::from_owned(means),
            len: n_rows,
        },
        ScryGpuStorage::Gpu {
            buf: GpuTensorStorage::from_owned(rstds),
            len: n_rows,
        },
    ))
}

/// Minimum channel count before engaging the GPU batchnorm path. Below this,
/// CPU rayon parallelism wins; the per-launch grid setup dominates a small
/// number of channel planes. ResNet's first stage has 64 channels, so this
/// engages from the network entry onward.
const GPU_BATCHNORM_MIN_CHANNELS: usize = 16;

/// GPU-resident 2D batchnorm inference with stored running stats. CUDA-only —
/// the Vulkan path returns `None` so the caller falls back to CPU. Returns
/// `None` also when the GPU path is unavailable, the workload is below
/// threshold, the input total doesn't divide evenly into `channels * spatial`
/// planes, or `weight`/`bias`/`running_mean`/`running_var` lengths don't match
/// `channels`.
fn gpu_batchnorm_persistent(
    input: &ScryGpuStorage,
    weight: &ScryGpuStorage,
    bias: &ScryGpuStorage,
    running_mean: &ScryGpuStorage,
    running_var: &ScryGpuStorage,
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.batchnorm.as_ref()?;
    if channels < GPU_BATCHNORM_MIN_CHANNELS || spatial == 0 {
        return None;
    }
    let total = input.len();
    let plane = channels * spatial;
    if plane == 0 || total % plane != 0 {
        return None;
    }
    let n_batch = total / plane;
    if n_batch == 0 {
        return None;
    }
    if weight.len() != channels
        || bias.len() != channels
        || running_mean.len() != channels
        || running_var.len() != channels
    {
        return None;
    }

    let buf_in = as_gpu_buffer(input)?;
    let buf_w = as_gpu_buffer(weight)?;
    let buf_b = as_gpu_buffer(bias)?;
    let buf_m = as_gpu_buffer(running_mean)?;
    let buf_v = as_gpu_buffer(running_var)?;

    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    // Push constants pack [channels: u32, spatial: u32, eps: f32]. Same f32
    // bit-pun trick as layernorm — see `dispatch_kernel` and
    // `crates/scry-gpu/src/backend/cuda.rs::launch_on_stream`.
    let dims_pc: [u32; 3] = [channels as u32, spatial as u32, eps.to_bits()];

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &*buf_w, &*buf_b, &*buf_m, &*buf_v, &out],
        [channels as u32, n_batch as u32, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Minimum number of output elements before engaging the GPU im2col path.
/// Below this, the CPU loop (with rayon-friendly memory layout and warm cache)
/// beats the GPU launch + upload + download chain for a one-shot conv. The
/// kernel is one thread per output element, so this is a memory-traffic
/// crossover, not compute. Set to match `GPU_MIN_ELEMENTS` (32_768): ResNet's
/// first layer produces 3*7*7 * 112*112 = 1.85M output elements, well above.
const GPU_IM2COL_MIN_OUTPUT_ELEMENTS: usize = 32_768;

/// GPU-resident im2col lowering for a 2D convolution.
///
/// `input` is `[c_in, h_in, w_in]`; output is `[c_in*kh*kw, h_out*w_out]` in
/// row-major layout. Returns `None` when the GPU path is unavailable, the
/// workload is below threshold, or any dimension is zero.
#[allow(clippy::too_many_arguments)]
fn gpu_im2col_persistent(
    input: &ScryGpuStorage,
    c_in: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    h_out: usize,
    w_out: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.im2col.as_ref()?;
    let col_rows = c_in.checked_mul(kh)?.checked_mul(kw)?;
    let spatial_out = h_out.checked_mul(w_out)?;
    let total = col_rows.checked_mul(spatial_out)?;
    if col_rows == 0 || spatial_out == 0 || total < GPU_IM2COL_MIN_OUTPUT_ELEMENTS {
        return None;
    }
    if input.len() != c_in * h_in * w_in {
        return None;
    }

    let buf_in = as_gpu_buffer(input)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    // Push constants pack 9 u32s = 36 bytes, well under the 128-byte limit.
    let dims_pc: [u32; 9] = [
        c_in as u32,
        h_in as u32,
        w_in as u32,
        kh as u32,
        kw as u32,
        stride as u32,
        padding as u32,
        h_out as u32,
        w_out as u32,
    ];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Minimum element count for the GPU scale path. ViT attention scores at
/// 12×197×197 = 466K easily clear it; tiny tensors just keep using CPU.
const GPU_SCALE_MIN: usize = 2_048;

/// GPU-resident elementwise scale: `out[i] = a[i] * scalar`. Keeps the
/// pre-softmax scale step in `scaled_softmax` on-device for transformer
/// attention.
fn gpu_scale_persistent(a: &ScryGpuStorage, scalar: f32) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.scale.as_ref()?;
    let total = a.len();
    if total < GPU_SCALE_MIN {
        return None;
    }
    let buf_in = as_gpu_buffer(a)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    // Push constants: u32 N, f32 alpha. The dispatcher packs each 4-byte
    // value as a u32; the kernel reads `alpha` as `f32` (transparent pun).
    let mut pc = [0u32; 2];
    pc[0] = total as u32;
    pc[1] = scalar.to_bits();
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Minimum output element count for the GPU split_qkv path. ViT-B/16 has
/// 12 heads × 197 tokens × 64 d_head = 151_296 per output → well above; even
/// tiny test configs of 4 heads × 32 × 16 = 2_048 hit it.
const GPU_SPLIT_QKV_MIN: usize = 2_048;

/// GPU-resident fused-QKV split into per-head Q/K/V. Reads `[seq, 3*d_model]`,
/// writes three `[n_heads, seq, d_head]` tensors via a single kernel launch.
/// Replaces the trait default's full host round-trip in transformer
/// attention.
fn gpu_split_qkv_persistent(
    qkv: &ScryGpuStorage,
    seq: usize,
    n_heads: usize,
    d_head: usize,
) -> Option<(ScryGpuStorage, ScryGpuStorage, ScryGpuStorage)> {
    let ctx = get_ctx()?;
    let kernel = ctx.split_qkv_heads.as_ref()?;
    let d_model = n_heads.checked_mul(d_head)?;
    let total = seq.checked_mul(n_heads)?.checked_mul(d_head)?;
    if total < GPU_SPLIT_QKV_MIN {
        return None;
    }
    if qkv.len() != seq * 3 * d_model {
        return None;
    }

    let buf_in = as_gpu_buffer(qkv)?;
    let q_out = ctx.dev.alloc_uninit::<f32>(total).ok()?;
    let k_out = ctx.dev.alloc_uninit::<f32>(total).ok()?;
    let v_out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    let dims_pc: [u32; 3] = [seq as u32, n_heads as u32, d_head as u32];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &q_out, &k_out, &v_out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some((
        ScryGpuStorage::Gpu {
            buf: GpuTensorStorage::from_owned(q_out),
            len: total,
        },
        ScryGpuStorage::Gpu {
            buf: GpuTensorStorage::from_owned(k_out),
            len: total,
        },
        ScryGpuStorage::Gpu {
            buf: GpuTensorStorage::from_owned(v_out),
            len: total,
        },
    ))
}

/// GPU-resident reshape `[n_heads, seq, d_head]` → `[seq, n_heads*d_head]`.
/// Single kernel launch, one thread per output element.
fn gpu_reshape_from_heads_persistent(
    input: &ScryGpuStorage,
    seq: usize,
    n_heads: usize,
    d_head: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.reshape_from_heads.as_ref()?;
    let total = seq.checked_mul(n_heads)?.checked_mul(d_head)?;
    if total < GPU_SPLIT_QKV_MIN {
        return None;
    }
    if input.len() != total {
        return None;
    }
    let buf_in = as_gpu_buffer(input)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    let dims_pc: [u32; 3] = [seq as u32, n_heads as u32, d_head as u32];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Forward direction of [`gpu_reshape_from_heads_persistent`]:
/// `[seq, n_heads*d_head]` → `[n_heads, seq, d_head]`. Returns `None`
/// when the GPU path is unavailable (Vulkan kernel slot is `None`),
/// the workload is below `GPU_SPLIT_QKV_MIN`, or the input length
/// doesn't match `seq * n_heads * d_head`.
fn gpu_reshape_to_heads_persistent(
    input: &ScryGpuStorage,
    seq: usize,
    n_heads: usize,
    d_head: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.reshape_to_heads.as_ref()?;
    let total = seq.checked_mul(n_heads)?.checked_mul(d_head)?;
    if total < GPU_SPLIT_QKV_MIN {
        return None;
    }
    if input.len() != total {
        return None;
    }
    let buf_in = as_gpu_buffer(input)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    let dims_pc: [u32; 3] = [seq as u32, n_heads as u32, d_head as u32];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Minimum total output element count before engaging the cuDNN conv path.
/// Same scale as the im2col threshold — at smaller sizes, cuDNN's per-call
/// dispatch overhead (~30–50 µs) is comparable to a small CPU conv. ResNet
/// stem (1.84M outputs) and every internal stage (≥200K) clear this easily.
#[cfg(feature = "scry-gpu-cudnn")]
const GPU_CUDNN_CONV2D_MIN_OUTPUT_ELEMENTS: usize = 32_768;

/// GPU-resident 2D conv forward via cuDNN. Implicit-GEMM (or Winograd / FFT —
/// the cuDNN heuristic picks per shape) fused conv that skips the im2col +
/// cuBLAS round-trip the default path uses.
///
/// Returns `None` if scry-gpu, the cuDNN feature, or the cuDNN handle are
/// unavailable, the workload is below threshold, or the input/weight aren't
/// already on-device — caller should fall back to the im2col + matmul path.
#[cfg(feature = "scry-gpu-cudnn")]
#[allow(clippy::too_many_arguments)]
fn gpu_conv2d_cudnn_persistent(
    input: &ScryGpuStorage,
    weight: &ScryGpuStorage,
    in_channels: usize,
    h_in: usize,
    w_in: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    // cuDNN is CUDA-only.
    if !matches!(ctx.matmul, MatmulStrategy::CuBlas) {
        return None;
    }
    // Runtime opt-out — benches flip this to compare against the im2col path.
    if !ctx.cudnn_conv_enabled.load(Ordering::Relaxed) {
        return None;
    }
    let h_out = (h_in + 2 * padding).checked_sub(kernel_h)? / stride + 1;
    let w_out = (w_in + 2 * padding).checked_sub(kernel_w)? / stride + 1;
    let out_len = out_channels.checked_mul(h_out)?.checked_mul(w_out)?;
    if out_len < GPU_CUDNN_CONV2D_MIN_OUTPUT_ELEMENTS {
        return None;
    }
    if input.len() != in_channels * h_in * w_in {
        return None;
    }
    if weight.len() != out_channels * in_channels * kernel_h * kernel_w {
        return None;
    }

    let buf_in = as_gpu_buffer(input)?;
    let buf_w = as_gpu_buffer(weight)?;
    // cuDNN writes every output element — zero-init is wasted work.
    let mut out = ctx.dev.alloc_uninit::<f32>(out_len).ok()?;

    ctx.dev
        .cudnn_conv2d_forward_async(
            &buf_in,
            &buf_w,
            &mut out,
            1,
            in_channels as u32,
            h_in as u32,
            w_in as u32,
            out_channels as u32,
            kernel_h as u32,
            kernel_w as u32,
            padding as u32,
            padding as u32,
            stride as u32,
            stride as u32,
        )
        .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: out_len,
    })
}

/// Minimum total element count before engaging the GPU column-broadcast bias
/// add. Same scale as `GPU_ELEMENTWISE_MIN` since the kernel is also one
/// thread per output element with a single load + store + L1-cached bias[r].
const GPU_ADD_ROW_BIAS_MIN: usize = 2_048;

/// Minimum total element count before engaging the GPU same-shape add. Same
/// scale as `GPU_ELEMENTWISE_MIN` — the kernel is one thread per output
/// element with two coalesced loads + one store. ResNet's smallest residual
/// add (stage 4 of R50, `2048 * 7 * 7 = 100K` elements) is well above; only
/// truly tiny tensors fall back to CPU.
const GPU_ADD_ELEMENTWISE_MIN: usize = 2_048;

/// GPU-resident `[rows, cols] + bias[rows]` (column-broadcast). Bias buffer
/// length must equal `rows`; an `[rows, 1]` shape is also accepted by the
/// caller. Returns `None` when the GPU path is unavailable or the workload is
/// below threshold.
fn gpu_add_row_bias_persistent(
    a: &ScryGpuStorage,
    bias: &ScryGpuStorage,
    rows: usize,
    cols: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.add_row_bias.as_ref()?;
    if rows == 0 || cols == 0 {
        return None;
    }
    let total = rows.checked_mul(cols)?;
    if total < GPU_ADD_ROW_BIAS_MIN {
        return None;
    }
    if a.len() != total || bias.len() != rows {
        return None;
    }

    let buf_a = as_gpu_buffer(a)?;
    let buf_b = as_gpu_buffer(bias)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    let dims_pc: [u32; 2] = [rows as u32, cols as u32];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_a, &*buf_b, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// GPU-resident column-broadcast bias add: `out[r, c] = a[r, c] + bias[c]`.
/// `a` is `[rows, cols]` row-major, `bias` is `[cols]`. Returns `None`
/// when the GPU path is unavailable, the kernel slot is missing
/// (Vulkan), or the workload is below `GPU_ADD_ROW_BIAS_MIN`.
///
/// Powers `MathBackend::matmul_bias` on `ScryGpuBackend` — every dense
/// linear layer in SD calls this via the `matmul_bias` override.
fn gpu_bias_add_persistent(
    a: &ScryGpuStorage,
    bias: &ScryGpuStorage,
    rows: usize,
    cols: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.bias_add.as_ref()?;
    if rows == 0 || cols == 0 {
        return None;
    }
    let total = rows.checked_mul(cols)?;
    if total < GPU_ADD_ROW_BIAS_MIN {
        return None;
    }
    if a.len() != total || bias.len() != cols {
        return None;
    }

    let buf_a = as_gpu_buffer(a)?;
    let buf_b = as_gpu_buffer(bias)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    // BIAS_ADD_CUDA push consts: (N = total, cols).
    let dims_pc: [u32; 2] = [total as u32, cols as u32];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_a, &*buf_b, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// GPU-resident same-shape elementwise add (`out[i] = a[i] + b[i]`). Both
/// inputs must have length `n`. Returns `None` when the GPU path is
/// unavailable, the workload is below threshold, or either operand isn't
/// device-resident (shape-driven branching is the caller's job).
fn gpu_add_elementwise_persistent(
    a: &ScryGpuStorage,
    b: &ScryGpuStorage,
    n: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.add_elementwise.as_ref()?;
    if n == 0 || n < GPU_ADD_ELEMENTWISE_MIN {
        return None;
    }
    if a.len() != n || b.len() != n {
        return None;
    }

    let buf_a = as_gpu_buffer(a)?;
    let buf_b = as_gpu_buffer(b)?;
    let out = ctx.dev.alloc_uninit::<f32>(n).ok()?;

    let dims_pc: [u32; 1] = [n as u32];
    let groups = (n as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_a, &*buf_b, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: n,
    })
}

/// GPU-resident same-shape elementwise multiply (`out[i] = a[i] * b[i]`).
/// Mirror of [`gpu_add_elementwise_persistent`]. Used by GeGLU's
/// `values * gelu(gate)` step in SD UNet's feed-forward — the trait
/// default downloaded both operands to host before the matmul_bias fix
/// of 2026-05-08, then re-uploaded the product. Single GPU kernel
/// dispatch instead.
fn gpu_mul_elementwise_persistent(
    a: &ScryGpuStorage,
    b: &ScryGpuStorage,
    n: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.mul_elementwise.as_ref()?;
    if n == 0 || n < GPU_ADD_ELEMENTWISE_MIN {
        return None;
    }
    if a.len() != n || b.len() != n {
        return None;
    }

    let buf_a = as_gpu_buffer(a)?;
    let buf_b = as_gpu_buffer(b)?;
    let out = ctx.dev.alloc_uninit::<f32>(n).ok()?;

    let dims_pc: [u32; 1] = [n as u32];
    let groups = (n as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_a, &*buf_b, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: n,
    })
}

/// Minimum output element count for the GPU embedding path. SD 1.5 CLIP is
/// `seq_len=77 × dim=768 = 59_136` outputs, well above. Tiny tests below the
/// threshold bypass the GPU and use the host loop, where the index-decode
/// overhead matters more than the gather bandwidth.
const GPU_EMBEDDING_MIN_OUTPUT_ELEMENTS: usize = 32_768;

/// GPU-resident embedding lookup: gathers rows of `weight[vocab, dim]` by
/// index. The previous trait override downloaded the entire weight table to
/// host before the gather (~145 MiB for SD 1.5 CLIP's `[49408, 768]` token
/// table), defeating `to_device` residency on every text-encode. Single
/// kernel launch keeps both the table and the result on-device.
///
/// `indices` arrive as `&[usize]` from the trait; uploaded as `Vec<u32>` so
/// the kernel can index with native 32-bit lane width. Returns `None` when
/// the GPU is unavailable, the workload is below threshold, or `weight`'s
/// length disagrees with `vocab * dim`.
fn gpu_embedding_persistent(
    weight: &ScryGpuStorage,
    indices: &[usize],
    vocab: usize,
    dim: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.embedding.as_ref()?;
    let n_indices = indices.len();
    let total = n_indices.checked_mul(dim)?;
    if total == 0 || total < GPU_EMBEDDING_MIN_OUTPUT_ELEMENTS {
        return None;
    }
    if weight.len() != vocab.checked_mul(dim)? {
        return None;
    }

    let buf_w = as_gpu_buffer(weight)?;
    let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let buf_idx = ctx.dev.upload::<u32>(&indices_u32).ok()?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    let dims_pc: [u32; 2] = [n_indices as u32, dim as u32];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_w, &buf_idx, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Minimum output element count for the GPU column-gather/scatter path. The
/// trait default downloads the entire `[rows, total_cols]` source on every
/// call, so even small outputs are worth a kernel launch — using the same
/// 2_048 floor as `gpu_split_qkv_persistent`. SD CLIP per-head Q/K/V
/// (77 × 64 = 4_928 outputs) clears it.
const GPU_COLUMN_GS_MIN_OUTPUT_ELEMENTS: usize = 2_048;

/// GPU-resident contiguous-column gather. Single kernel launch reads
/// `input[r, col_start..col_start + col_count]` for every row and writes the
/// `[rows, col_count]` output device-side. Replaces the trait default's
/// `to_vec` → CPU loop → `from_vec` chain, which fired 432 times per CLIP
/// text-encode at SD 1.5.
fn gpu_gather_columns_persistent(
    input: &ScryGpuStorage,
    rows: usize,
    total_cols: usize,
    col_start: usize,
    col_count: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.gather_columns.as_ref()?;
    if rows == 0 || col_count == 0 || total_cols == 0 {
        return None;
    }
    if col_start.checked_add(col_count)? > total_cols {
        return None;
    }
    let total = rows.checked_mul(col_count)?;
    if total < GPU_COLUMN_GS_MIN_OUTPUT_ELEMENTS {
        return None;
    }
    if input.len() != rows.checked_mul(total_cols)? {
        return None;
    }

    let buf_in = as_gpu_buffer(input)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    let dims_pc: [u32; 4] = [
        rows as u32,
        total_cols as u32,
        col_start as u32,
        col_count as u32,
    ];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// GPU-resident additive contiguous-column scatter. Mutates `dst` in place
/// — the kernel does `dst[r, col_start + c] += src[r, c]`. Used to merge
/// per-head attention outputs back into a `[seq, d_model]` accumulator;
/// 144 calls per CLIP encode. Returns `None` (and the caller falls back
/// to the host loop) when the GPU is unavailable, the workload is below
/// threshold, the column range overflows, or `src.len() != rows * col_count`.
fn gpu_scatter_columns_persistent(
    dst: &mut ScryGpuStorage,
    src: &ScryGpuStorage,
    rows: usize,
    total_cols: usize,
    col_start: usize,
    col_count: usize,
) -> Option<()> {
    let ctx = get_ctx()?;
    let kernel = ctx.scatter_columns.as_ref()?;
    if rows == 0 || col_count == 0 || total_cols == 0 {
        return None;
    }
    if col_start.checked_add(col_count)? > total_cols {
        return None;
    }
    let total = rows.checked_mul(col_count)?;
    if total < GPU_COLUMN_GS_MIN_OUTPUT_ELEMENTS {
        return None;
    }
    let dst_total = rows.checked_mul(total_cols)?;
    if dst.len() != dst_total || src.len() != total {
        return None;
    }

    // Both buffers must be device-resident so the scatter writes back to
    // the caller's `dst`. The trait default's `B::zeros` returns a Cpu
    // variant, so promote on first call — replace `*dst` with a Gpu
    // variant pointing at the freshly-uploaded buffer, and write through
    // it. Subsequent scatters in the per-head loop see the Gpu variant
    // directly.
    let dst_buf = match dst {
        ScryGpuStorage::Gpu { buf, .. } => Arc::clone(&buf.f32),
        ScryGpuStorage::Cpu(v) => {
            let buf = ctx.dev.upload::<f32>(v).ok()?;
            let arc = Arc::new(buf);
            *dst = ScryGpuStorage::Gpu {
                buf: GpuTensorStorage::from_arc(Arc::clone(&arc)),
                len: dst_total,
            };
            arc
        }
    };
    let src_buf = as_gpu_buffer(src)?;

    let dims_pc: [u32; 4] = [
        rows as u32,
        total_cols as u32,
        col_start as u32,
        col_count as u32,
    ];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*src_buf, &*dst_buf],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(())
}

/// GPU-resident causal-mask + scale fused over a `[seq, seq]` score block.
/// Mutates `scores` in place: `scores[s, t] = (t > s) ? mask : scores[s,t]*scale`.
/// 144 calls per CLIP encode at SD 1.5; trait default round-tripped each
/// `[77, 77]` block through host. Returns `None` when the GPU is
/// unavailable or `scores` is currently CPU-resident (the scatter
/// write-back would otherwise be lost).
fn gpu_apply_causal_mask_and_scale_persistent(
    scores: &mut ScryGpuStorage,
    seq: usize,
    scale: f32,
    mask_value: f32,
) -> Option<()> {
    let ctx = get_ctx()?;
    let kernel = ctx.apply_causal_mask_and_scale.as_ref()?;
    if seq == 0 {
        return None;
    }
    let total = seq.checked_mul(seq)?;
    if scores.len() != total {
        return None;
    }
    let scores_buf = match scores {
        ScryGpuStorage::Gpu { buf, .. } => Arc::clone(&buf.f32),
        ScryGpuStorage::Cpu(_) => return None,
    };

    // Push constants pack [seq: u32, scale: f32, mask_value: f32]. Same
    // bit-pun trick as `gpu_scale_persistent` — the dispatcher reads each
    // 4-byte slot as u32 and the kernel reinterprets the float-typed slots.
    let dims_pc: [u32; 3] = [seq as u32, scale.to_bits(), mask_value.to_bits()];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*scores_buf],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(())
}

/// Minimum number of output elements before engaging the GPU pool path. Below
/// this, CPU rayon parallelism plus warm-cache locality wins; the per-launch
/// grid setup dominates a small handful of output elements. Same scale as
/// `GPU_IM2COL_MIN_OUTPUT_ELEMENTS`. ResNet's 56×56 stem max-pool produces
/// 64*56*56 = 200K output elements, well above; global avg pool at the end
/// produces only `channels` outputs (e.g. 2048) so it falls back to CPU there
/// — fine, that work is cheap on host.
const GPU_POOL_MIN_OUTPUT_ELEMENTS: usize = 32_768;

/// GPU-resident 2D max-pool. CUDA-only — the Vulkan path returns `None` so the
/// caller falls back to CPU. Returns `None` also when the GPU path is
/// unavailable, the workload is below threshold, or any dimension is zero.
#[allow(clippy::too_many_arguments)]
fn gpu_max_pool_persistent(
    input: &ScryGpuStorage,
    channels: usize,
    h_in: usize,
    w_in: usize,
    kernel_sz: usize,
    stride: usize,
    padding: usize,
    h_out: usize,
    w_out: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.max_pool.as_ref()?;
    if channels == 0 || h_out == 0 || w_out == 0 || kernel_sz == 0 || stride == 0 {
        return None;
    }
    let total = channels.checked_mul(h_out)?.checked_mul(w_out)?;
    if total < GPU_POOL_MIN_OUTPUT_ELEMENTS {
        return None;
    }
    if input.len() != channels * h_in * w_in {
        return None;
    }

    let buf_in = as_gpu_buffer(input)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    // 9 u32s = 36 bytes, well under the 128-byte push-constant limit.
    let dims_pc: [u32; 9] = [
        channels as u32,
        h_in as u32,
        w_in as u32,
        kernel_sz as u32,
        kernel_sz as u32,
        stride as u32,
        padding as u32,
        h_out as u32,
        w_out as u32,
    ];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// GPU-resident adaptive 2D average-pool. CUDA-only — the Vulkan path returns
/// `None` so the caller falls back to CPU. Returns `None` also when the GPU
/// path is unavailable, the workload is below threshold, or any dimension is
/// zero.
fn gpu_adaptive_avg_pool_persistent(
    input: &ScryGpuStorage,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.adaptive_avg_pool.as_ref()?;
    if channels == 0 || h_out == 0 || w_out == 0 || h_in == 0 || w_in == 0 {
        return None;
    }
    let total = channels.checked_mul(h_out)?.checked_mul(w_out)?;
    if total < GPU_POOL_MIN_OUTPUT_ELEMENTS {
        return None;
    }
    if input.len() != channels * h_in * w_in {
        return None;
    }

    let buf_in = as_gpu_buffer(input)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    let dims_pc: [u32; 5] = [
        channels as u32,
        h_in as u32,
        w_in as u32,
        h_out as u32,
        w_out as u32,
    ];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Minimum group count before engaging the GPU group_norm path. Below this,
/// CPU rayon parallelism wins; the per-launch grid setup dominates a small
/// number of `(batch, group)` blocks. SD always uses `num_groups = 32`, so
/// even at batch=1 this engages from the network entry. Same scale as
/// `GPU_LAYERNORM_MIN_ROWS` (the kernel shape is the same — one block per
/// reduction unit, two block-wide reductions).
const GPU_GROUP_NORM_MIN_BLOCKS: usize = 32;

/// GPU-resident group normalization (inference) with per-channel affine.
/// CUDA-only — the Vulkan path returns `None` so the caller falls back to
/// CPU. Returns `None` also when the GPU path is unavailable, the workload
/// is below threshold (`num_groups * batch < GPU_GROUP_NORM_MIN_BLOCKS`),
/// `channels` doesn't divide evenly into `num_groups`, the input total
/// doesn't divide evenly into `channels * spatial` planes, or
/// `weight`/`bias` lengths don't match `channels`.
#[allow(clippy::too_many_arguments)]
fn gpu_group_norm_persistent(
    input: &ScryGpuStorage,
    weight: &ScryGpuStorage,
    bias: &ScryGpuStorage,
    num_groups: usize,
    channels: usize,
    spatial: usize,
    eps: f32,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.group_norm.as_ref()?;
    if num_groups == 0 || channels == 0 || spatial == 0 {
        return None;
    }
    if channels % num_groups != 0 {
        return None;
    }
    if weight.len() != channels || bias.len() != channels {
        return None;
    }
    let total = input.len();
    let plane = channels * spatial;
    if plane == 0 || total % plane != 0 {
        return None;
    }
    let n_batch = total / plane;
    if n_batch == 0 {
        return None;
    }
    if num_groups.checked_mul(n_batch)? < GPU_GROUP_NORM_MIN_BLOCKS {
        return None;
    }

    let buf_in = as_gpu_buffer(input)?;
    let buf_w = as_gpu_buffer(weight)?;
    let buf_b = as_gpu_buffer(bias)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    // Push constants pack [channels: u32, spatial: u32, num_groups: u32, eps: f32].
    // f32 bit-pun trick — see `dispatch_kernel` and
    // `crates/scry-gpu/src/backend/cuda.rs::launch_on_stream`.
    let dims_pc: [u32; 4] = [
        channels as u32,
        spatial as u32,
        num_groups as u32,
        eps.to_bits(),
    ];

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &*buf_w, &*buf_b, &out],
        [num_groups as u32, n_batch as u32, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// Minimum output-element count before engaging the GPU upsample path. Below
/// this, the CPU loop wins on a one-shot call. Same scale as the pool kernels
/// since the dispatch shape is identical (one thread per output element).
const GPU_UPSAMPLE_MIN_OUTPUT_ELEMENTS: usize = 32_768;

/// GPU-resident 2D nearest-neighbor upsample by an integer factor. CUDA-only —
/// the Vulkan path returns `None` so the caller falls back to CPU. Returns
/// `None` also when the GPU path is unavailable, the workload is below
/// threshold, the input length doesn't match `channels * h_in * w_in`, or
/// any dimension (or `scale`) is zero.
fn gpu_upsample_2d_nearest_persistent(
    input: &ScryGpuStorage,
    channels: usize,
    h_in: usize,
    w_in: usize,
    scale: usize,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    let kernel = ctx.upsample_nearest.as_ref()?;
    if channels == 0 || h_in == 0 || w_in == 0 || scale == 0 {
        return None;
    }
    let h_out = h_in.checked_mul(scale)?;
    let w_out = w_in.checked_mul(scale)?;
    let total = channels.checked_mul(h_out)?.checked_mul(w_out)?;
    if total < GPU_UPSAMPLE_MIN_OUTPUT_ELEMENTS {
        return None;
    }
    if input.len() != channels * h_in * w_in {
        return None;
    }

    let buf_in = as_gpu_buffer(input)?;
    let out = ctx.dev.alloc_uninit::<f32>(total).ok()?;

    let dims_pc: [u32; 4] = [channels as u32, h_in as u32, w_in as u32, scale as u32];
    let groups = (total as u32).div_ceil(256);

    dispatch_kernel(
        ctx,
        kernel,
        &[&*buf_in, &out],
        [groups, 1, 1],
        Some(bytemuck::bytes_of(&dims_pc)),
    )
    .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total,
    })
}

/// GPU-resident matmul: takes `ScryGpuStorage` inputs, returns a `Gpu`-variant
/// output without round-tripping through CPU. Returns `None` if the GPU path
/// is unavailable for the given inputs (caller should fall back to CPU).
///
/// On CUDA, dispatches to cuBLAS SGEMM (≈45% of f32 peak on RTX 5070 Ti).
/// On Vulkan, dispatches to the WGSL coarse 4×4 tiled matmul (~19%).
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

    let buf_a = as_gpu_buffer(a)?;
    let buf_b = as_gpu_buffer(b)?;

    // Transpose on-device when needed. The matmul kernels (WGSL and cuBLAS)
    // both expect row-major M×K and K×N.
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

    // `mut` only needed on the cuBLAS arm; Vulkan path doesn't mutate the binding.
    // Both matmul kernels (WGSL and cuBLAS SGEMM) write the entire C[m*n]
    // matrix, so zero-init is wasted work.
    #[allow(unused_mut)]
    let mut out = ctx.dev.alloc_uninit::<f32>(m * n).ok()?;
    match &ctx.matmul {
        MatmulStrategy::Wgsl(kernel) => {
            let dims: [u32; 3] = [m as u32, n as u32, k as u32];
            ctx.dev
                .run_configured(
                    kernel,
                    &[buf_a_ref, buf_b_ref, &out],
                    [(n as u32).div_ceil(64), (m as u32).div_ceil(64), 1],
                    Some(bytemuck::bytes_of(&dims)),
                )
                .ok()?;
        }
        #[cfg(feature = "scry-gpu-cuda")]
        MatmulStrategy::CuBlas => {
            ctx.dev
                .cublas_matmul_async(
                    buf_a_ref,
                    buf_b_ref,
                    &mut out,
                    m as u32,
                    n as u32,
                    k as u32,
                )
                .ok()?;
        }
    }

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: m * n,
    })
}

/// GPU-resident strided batched matmul via cuBLAS `sgemm_strided_batched`.
///
/// Single launch covers all `batch` per-head matmuls — vs the default impl
/// which downloads to CPU, loops with `from_vec`/`to_vec` per iteration, and
/// re-uploads the result. ViT attention hits this twice per block (Q@Kᵀ
/// and attn@V); 12 blocks × 2 = 24 calls per forward where this matters.
///
/// CUDA-only — Vulkan and pre-cuBLAS paths return `None` and the caller
/// falls back to the trait default.
#[cfg(feature = "scry-gpu-cuda")]
#[allow(clippy::too_many_arguments)]
fn gpu_matmul_strided_batched_persistent(
    a: &ScryGpuStorage,
    b: &ScryGpuStorage,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    if !matches!(ctx.matmul, MatmulStrategy::CuBlas) {
        return None;
    }
    let total_a = batch * m * k;
    let total_b = batch * k * n;
    let total_c = batch * m * n;
    if total_a == 0 || total_b == 0 || total_c == 0 {
        return None;
    }
    if a.len() != total_a || b.len() != total_b {
        return None;
    }
    let buf_a = as_gpu_buffer(a)?;
    let buf_b = as_gpu_buffer(b)?;
    let mut out = ctx.dev.alloc_uninit::<f32>(total_c).ok()?;

    ctx.dev
        .cublas_strided_batched_matmul_async(
            &buf_a,
            &buf_b,
            &mut out,
            batch as u32,
            m as u32,
            n as u32,
            k as u32,
            trans_a,
            trans_b,
        )
        .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total_c,
    })
}

/// GPU-resident strided batched matmul through cuBLAS GemmStridedBatchedEx
/// (bf16 inputs, fp32 accumulate, fp32 output).
///
/// Mirrors [`gpu_matmul_strided_batched_persistent`] but routes through the
/// bf16 path. Unlike [`gpu_matmul_persistent_bf16`] this passes trans flags
/// directly to cuBLAS (the strided wrapper accepts them), so no
/// transpose-then-cast pre-pass is needed — Q@Kᵀ takes `trans_b=true`
/// straight through.
///
/// Inputs are typically attention activations (Q/K/V from
/// `split_qkv_reshape_heads`) which are first-touch fp32 buffers per
/// forward — the bf16 shadow cache rarely fires here, so each call casts
/// fresh. The cast is cheap relative to the GEMM at attention shapes
/// (12 × 197 × 64 ≈ 151K elements vs 12 × 197 × 197 × 64 = 30M FMAs).
///
/// CUDA-only — Vulkan and pre-cuBLAS paths return `None` and the caller
/// falls back to the fp32 strided path.
#[cfg(feature = "scry-gpu-bf16")]
#[allow(clippy::too_many_arguments)]
fn gpu_matmul_strided_batched_persistent_bf16(
    a: &ScryGpuStorage,
    b: &ScryGpuStorage,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;
    if !matches!(ctx.matmul, MatmulStrategy::CuBlas) {
        return None;
    }
    let total_a = batch * m * k;
    let total_b = batch * k * n;
    let total_c = batch * m * n;
    if total_a == 0 || total_b == 0 || total_c == 0 {
        return None;
    }
    if a.len() != total_a || b.len() != total_b {
        return None;
    }

    let a_bf16: Arc<Buffer<half::bf16>> = if let Some(cached) = as_gpu_buffer_bf16(a) {
        cached
    } else {
        let a_buf = as_gpu_buffer(a)?;
        Arc::new(cast_f32_to_bf16(ctx, &a_buf, total_a)?)
    };
    let b_bf16: Arc<Buffer<half::bf16>> = if let Some(cached) = as_gpu_buffer_bf16(b) {
        cached
    } else {
        let b_buf = as_gpu_buffer(b)?;
        Arc::new(cast_f32_to_bf16(ctx, &b_buf, total_b)?)
    };

    let mut out = ctx.dev.alloc_uninit::<f32>(total_c).ok()?;

    ctx.dev
        .cublas_strided_batched_matmul_bf16_in_f32_out_async(
            &a_bf16,
            &b_bf16,
            &mut out,
            batch as u32,
            m as u32,
            n as u32,
            k as u32,
            trans_a,
            trans_b,
        )
        .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: total_c,
    })
}

/// GPU-resident matmul through cuBLAS GemmEx (bf16 inputs, fp32 accumulate,
/// fp32 output).
///
/// f32 inputs are cast down to bf16, the GEMM runs on tensor cores, and the
/// fp32 accumulator is written **directly** to the output buffer via
/// `cublasGemmEx` with `C_type = CUDA_R_32F` — no cast-back kernel needed.
/// The downstream graph sees fp32 storage exactly as before.
///
/// Returns `None` if scry-gpu, the bf16 feature, or the cast/matmul kernels
/// are unavailable — caller should fall through to the f32 path.
#[cfg(feature = "scry-gpu-bf16")]
#[allow(clippy::cast_possible_truncation)]
fn gpu_matmul_persistent_bf16(
    a: &ScryGpuStorage,
    b: &ScryGpuStorage,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Option<ScryGpuStorage> {
    let ctx = get_ctx()?;

    // Transpose first if requested. Transpose returns a fresh `Buffer<f32>`
    // with no `GpuTensorStorage` wrapper, so the bf16 cache only fires for
    // the un-transposed (typical) case. Conv2d weights and activations
    // are both passed `trans_a=trans_b=false`, so this is the hot path.
    let a_bf16: Arc<Buffer<half::bf16>> = if trans_a {
        let src = as_gpu_buffer(a)?;
        let tmp = gpu_transpose(&src, k, m)?;
        Arc::new(cast_f32_to_bf16(ctx, &tmp, m * k)?)
    } else if let Some(cached) = as_gpu_buffer_bf16(a) {
        cached
    } else {
        // Fallback: storage was Cpu-resident. Upload, then cast — no
        // shadow caching since we don't have a `GpuTensorStorage` to
        // attach to. Caller should `to_device` for free caching.
        let a_buf = as_gpu_buffer(a)?;
        Arc::new(cast_f32_to_bf16(ctx, &a_buf, m * k)?)
    };

    let b_bf16: Arc<Buffer<half::bf16>> = if trans_b {
        let src = as_gpu_buffer(b)?;
        let tmp = gpu_transpose(&src, n, k)?;
        Arc::new(cast_f32_to_bf16(ctx, &tmp, k * n)?)
    } else if let Some(cached) = as_gpu_buffer_bf16(b) {
        cached
    } else {
        let b_buf = as_gpu_buffer(b)?;
        Arc::new(cast_f32_to_bf16(ctx, &b_buf, k * n)?)
    };

    // GemmEx: bf16 × bf16 → fp32 (directly, no cast-up). Same tensor-core
    // path as the bf16-output variant; the fp32 accumulator is dropped to
    // the output buffer in one kernel.
    let mut out = ctx.dev.alloc_uninit::<f32>(m * n).ok()?;
    ctx.dev
        .cublas_matmul_bf16_in_f32_out_async(
            &a_bf16,
            &b_bf16,
            &mut out,
            m as u32,
            n as u32,
            k as u32,
        )
        .ok()?;

    Some(ScryGpuStorage::Gpu {
        buf: GpuTensorStorage::from_owned(out),
        len: m * n,
    })
}

/// Allocate a bf16 buffer of `n` elements and dispatch the f32→bf16 cast
/// kernel. Used for the transpose-then-cast slow path and the Cpu-fallback
/// path; the hot path goes through [`as_gpu_buffer_bf16`]'s OnceLock cache.
#[cfg(feature = "scry-gpu-bf16")]
fn cast_f32_to_bf16(
    ctx: &ScryCtx,
    src: &Buffer<f32>,
    n: usize,
) -> Option<Buffer<half::bf16>> {
    let cast_down = ctx.cast_f32_bf16.as_ref()?;
    let dst = ctx.dev.alloc_uninit::<half::bf16>(n).ok()?;
    let n_pc: u32 = n as u32;
    dispatch_kernel(
        ctx,
        cast_down,
        &[src, &dst],
        [n_pc.div_ceil(256), 1, 1],
        Some(bytemuck::bytes_of(&n_pc)),
    )
    .ok()?;
    Some(dst)
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
                    buf: GpuTensorStorage::from_owned(buf),
                    len,
                })
            }
        }
    }

    /// Block until every previously-issued GPU dispatch has completed.
    ///
    /// Bench-only and diagnostic use: the persistent path runs async on
    /// CUDA so that chains of dispatches don't pay a sync per op. A
    /// benchmark needs to wait for the work to actually finish before
    /// stopping the timer, otherwise it measures host-side launch latency
    /// rather than GPU execution time. Production callers either trigger
    /// sync via `Buffer::download` (which already syncs internally) or
    /// don't need to observe completion at all.
    ///
    /// Returns `Err` only if scry-gpu is unavailable.
    pub fn synchronize() -> Result<(), String> {
        let ctx = get_ctx().ok_or_else(|| "scry-gpu unavailable".to_string())?;
        ctx.dev
            .synchronize()
            .map_err(|e| format!("scry-gpu synchronize: {e}"))
    }

    /// Move storage to CPU residency. No-op if already on CPU.
    pub fn to_cpu(storage: &ScryGpuStorage) -> ScryGpuStorage {
        match storage {
            ScryGpuStorage::Cpu(_) => storage.clone(),
            ScryGpuStorage::Gpu { buf, .. } => {
                let v = buf
                    .f32
                    .download()
                    .expect("scry-gpu: download failed in to_cpu");
                ScryGpuStorage::Cpu(v)
            }
        }
    }

    /// Bench-only: run the persistent GPU matmul without the
    /// [`GPU_MIN_ELEMENTS`] gate, so a sweep can probe sub-threshold sizes.
    /// Returns `None` if the GPU is unavailable. Production callers should go
    /// through [`MathBackend::matmul`].
    pub fn matmul_force_gpu_for_bench(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<ScryGpuStorage> {
        gpu_matmul_persistent(a, b, m, k, n, false, false)
    }

    /// Toggle the cuDNN convolution fast-path at runtime.
    ///
    /// Defaults to `true` when the `scry-gpu-cudnn` feature is enabled. The
    /// bench harness flips it off to measure the legacy im2col + cuBLAS
    /// chain on the same model instance and back on for the cuDNN row.
    /// Returns `Err` only if scry-gpu is unavailable.
    #[cfg(feature = "scry-gpu-cudnn")]
    pub fn set_cudnn_conv(enable: bool) -> Result<(), String> {
        let ctx = get_ctx().ok_or_else(|| "scry-gpu unavailable".to_string())?;
        ctx.cudnn_conv_enabled.store(enable, Ordering::Relaxed);
        Ok(())
    }

    /// Toggle the bf16 GemmEx fast-path for matmul at runtime.
    ///
    /// Initial value is taken from `SCRY_GPU_MATMUL_BF16` at first ctx
    /// access; this method lets benches and tests flip the routing between
    /// rows without re-spawning the process. Returns `Err` only if scry-gpu
    /// is unavailable.
    #[cfg(feature = "scry-gpu-bf16")]
    pub fn set_bf16_matmul(enable: bool) -> Result<(), String> {
        let ctx = get_ctx().ok_or_else(|| "scry-gpu unavailable".to_string())?;
        ctx.bf16_matmul_enabled.store(enable, Ordering::Relaxed);
        Ok(())
    }

    /// Bench-only: run the bf16 GemmEx fast-path regardless of the
    /// `SCRY_GPU_MATMUL_BF16` env var. Returns `None` if the GPU or the
    /// `scry-gpu-bf16` feature is unavailable.
    #[cfg(feature = "scry-gpu-bf16")]
    pub fn matmul_force_bf16_for_bench(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<ScryGpuStorage> {
        gpu_matmul_persistent_bf16(a, b, m, k, n, false, false)
    }

    /// Bench-only: run the persistent GPU GELU without the
    /// [`GPU_ELEMENTWISE_MIN`] gate. Returns `None` if the GPU is unavailable.
    pub fn gelu_force_gpu_for_bench(input: &ScryGpuStorage) -> Option<ScryGpuStorage> {
        let ctx = get_ctx()?;
        let kernel = ctx.gelu.as_ref()?;
        let n = input.len();
        let buf_in = as_gpu_buffer(input)?;
        let out = run_unary_elementwise(kernel, &buf_in, n)?;
        Some(ScryGpuStorage::Gpu {
            buf: GpuTensorStorage::from_owned(out),
            len: n,
        })
    }

    /// Experimental: record `matmul` + `gelu` into a single `Device::batch()`
    /// and submit with one fence wait, instead of two synchronous
    /// `run_configured` round trips.
    ///
    /// Used by the `gpu_batched_poc` bench to validate whether eliminating
    /// one fence per chain wins enough to justify a broader batched-dispatch
    /// port. Not yet wired into the `MathBackend` impl.
    ///
    /// Returns `None` when the GPU path is unavailable for any reason
    /// (no device, cuBLAS path, sub-threshold size, transposes requested).
    /// Inputs must be `ScryGpuStorage::Gpu` or upload-able to it.
    /// No transpose support — caller pre-transposes if needed.
    pub fn matmul_then_gelu_batched_for_bench(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        m: usize,
        k: usize,
        n: usize,
    ) -> Option<ScryGpuStorage> {
        let ctx = get_ctx()?;
        // cuBLAS path can't be recorded into a Vulkan-style batch.
        #[allow(clippy::infallible_destructuring_match)]
        let mm_kernel = match &ctx.matmul {
            MatmulStrategy::Wgsl(k) => k,
            #[cfg(feature = "scry-gpu-cuda")]
            MatmulStrategy::CuBlas => return None,
        };
        let gelu_kernel = ctx.gelu.as_ref()?;

        let buf_a = as_gpu_buffer(a)?;
        let buf_b = as_gpu_buffer(b)?;

        let c_buf = ctx.dev.alloc_uninit::<f32>(m * n).ok()?;
        let g_buf = ctx.dev.alloc_uninit::<f32>(m * n).ok()?;

        let mm_dims: [u32; 3] = [m as u32, n as u32, k as u32];
        let gelu_dims: [u32; 1] = [(m * n) as u32];
        let gelu_groups = ((m * n) as u32).div_ceil(256);

        let mut batch = ctx.dev.batch().ok()?;
        batch
            .run_configured(
                mm_kernel,
                &[&*buf_a, &*buf_b, &c_buf],
                [(n as u32).div_ceil(64), (m as u32).div_ceil(64), 1],
                Some(bytemuck::bytes_of(&mm_dims)),
            )
            .ok()?;
        // c_buf is read by gelu — barrier is required for correct ordering.
        batch.barrier();
        batch
            .run_configured(
                gelu_kernel,
                &[&c_buf, &g_buf],
                [gelu_groups, 1, 1],
                Some(bytemuck::bytes_of(&gelu_dims)),
            )
            .ok()?;
        batch.submit().ok()?;

        Some(ScryGpuStorage::Gpu {
            buf: GpuTensorStorage::from_owned(g_buf),
            len: m * n,
        })
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
                .f32
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

    fn to_device_in_place(storage: &mut ScryGpuStorage) {
        // Already on device — nothing to do.
        if matches!(storage, ScryGpuStorage::Gpu { .. }) {
            return;
        }
        // GPU unavailable: keep the CPU storage so the call stays a soft no-op.
        if let Ok(gpu) = Self::to_gpu(storage) {
            *storage = gpu;
        }
    }
}

/// Wrap a CPU result in `ScryGpuStorage::Cpu`.
fn cpu(v: Vec<f32>) -> ScryGpuStorage {
    ScryGpuStorage::Cpu(v)
}

impl MathBackend for ScryGpuBackend {
    const PREFERS_IM2COL_OVER_WINOGRAD: bool = true;

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
            // bf16 / fp32-accumulate GemmEx fast-path. Opt-in via
            // `SCRY_GPU_MATMUL_BF16=1`; only fires when the bf16 ctx is
            // initialized (CUDA + scry-gpu-bf16 feature).
            #[cfg(feature = "scry-gpu-bf16")]
            if get_ctx().is_some_and(|c| c.bf16_matmul_enabled.load(Ordering::Relaxed)) {
                if let Some(gpu_out) =
                    gpu_matmul_persistent_bf16(a, b, m, k, n, trans_a, trans_b)
                {
                    return gpu_out;
                }
            }
            if let Some(gpu_out) = gpu_matmul_persistent(a, b, m, k, n, trans_a, trans_b) {
                return gpu_out;
            }
        }
        // Fallback: CPU compute (or cuBLAS via the legacy materialize path).
        let av = a.as_vec();
        let bv = b.as_vec();
        cpu(matmul_gpu_or_cpu(&av, &bv, m, k, n, trans_a, trans_b))
    }

    fn matmul_bias(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        bias: &ScryGpuStorage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> ScryGpuStorage {
        // Try the on-device path: matmul → column-broadcast bias add.
        // The trait default downloads the matmul output to host, runs a
        // scalar bias loop, and returns Cpu storage — the next op then
        // re-uploads. Per SD UNet forward that's ~104 round trips. The
        // GPU path keeps everything on-device.
        let c = Self::matmul(a, b, m, k, n, trans_a, trans_b);
        if let Some(out) = gpu_bias_add_persistent(&c, bias, m, n) {
            return out;
        }
        // Fall through to the trait default's host bias loop. Reached
        // when the workload is below `GPU_ADD_ROW_BIAS_MIN` or the GPU
        // path is unavailable.
        let c_vec = Self::to_vec(&c);
        let bias_vec = Self::to_vec(bias);
        let mut out = c_vec;
        for row in 0..m {
            for col in 0..n {
                out[row * n + col] += bias_vec[col];
            }
        }
        Self::from_vec(out, &Shape::new(&[m, n]))
    }

    fn split_qkv_reshape_heads(
        qkv: &ScryGpuStorage,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> (ScryGpuStorage, ScryGpuStorage, ScryGpuStorage) {
        if let Some(triple) = gpu_split_qkv_persistent(qkv, seq, n_heads, d_head) {
            return triple;
        }
        // Fallback to default impl (CPU-bouncing). Default expects qkv as
        // a Storage, so just call through.
        let (q, k, v) = {
            let d_model = n_heads * d_head;
            let head_len = n_heads * seq * d_head;
            let data = Self::to_vec(qkv);
            let mut q = vec![0.0f32; head_len];
            let mut k = vec![0.0f32; head_len];
            let mut v = vec![0.0f32; head_len];
            for s in 0..seq {
                let row = s * 3 * d_model;
                for h in 0..n_heads {
                    for d in 0..d_head {
                        let dst = (h * seq + s) * d_head + d;
                        let src_col = h * d_head + d;
                        q[dst] = data[row + src_col];
                        k[dst] = data[row + d_model + src_col];
                        v[dst] = data[row + 2 * d_model + src_col];
                    }
                }
            }
            let shape = Shape::new(&[n_heads, seq, d_head]);
            (
                Self::from_vec(q, &shape),
                Self::from_vec(k, &shape),
                Self::from_vec(v, &shape),
            )
        };
        (q, k, v)
    }

    fn reshape_from_heads(
        storage: &ScryGpuStorage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> ScryGpuStorage {
        // GPU path is `batch=1` only — that's the ViT shape. Anything else
        // (multi-batch transformers, future) falls back to the default
        // host impl until the kernel grows a batch dim.
        if batch == 1 {
            if let Some(out) = gpu_reshape_from_heads_persistent(storage, seq, n_heads, d_head) {
                return out;
            }
        }
        // Default impl (CPU bouncing).
        let data = Self::to_vec(storage);
        let d_model = n_heads * d_head;
        let total = batch * seq * d_model;
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..d_head {
                        out[(b * seq + s) * d_model + h * d_head + d] =
                            data[(b * n_heads + h) * seq * d_head + s * d_head + d];
                    }
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[batch * seq, d_model]))
    }

    fn transpose_2d(input: &ScryGpuStorage, rows: usize, cols: usize) -> ScryGpuStorage {
        if let Some(out) = gpu_transpose_2d_persistent(input, rows, cols) {
            return out;
        }
        // Fall through to the trait default's scalar permute on host.
        let v = Self::to_vec(input);
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = v[r * cols + c];
            }
        }
        cpu(out)
    }

    fn reshape_for_heads(
        storage: &ScryGpuStorage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> ScryGpuStorage {
        // GPU path is `batch=1` only — same shape constraint as
        // `reshape_from_heads`. Anything else falls back to the host
        // default until the kernel grows a batch dim.
        if batch == 1 {
            if let Some(out) = gpu_reshape_to_heads_persistent(storage, seq, n_heads, d_head) {
                return out;
            }
        }
        // Default impl (CPU bouncing) — same scalar permute the trait does.
        let data = Self::to_vec(storage);
        let d_model = n_heads * d_head;
        let total = batch * n_heads * seq * d_head;
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..d_head {
                        out[(b * n_heads + h) * seq * d_head + s * d_head + d] =
                            data[(b * seq + s) * d_model + h * d_head + d];
                    }
                }
            }
        }
        Self::from_vec(out, &Shape::new(&[batch * n_heads, seq, d_head]))
    }

    fn matmul_strided_batched(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> ScryGpuStorage {
        // Try cuBLAS strided batched first. Falls through to the trait
        // default (per-batch loop with to_vec/from_vec) on Vulkan, sub-
        // threshold workloads, or shape mismatches.
        #[cfg(feature = "scry-gpu-bf16")]
        if get_ctx().is_some_and(|c| c.bf16_matmul_enabled.load(Ordering::Relaxed)) {
            if let Some(gpu_out) = gpu_matmul_strided_batched_persistent_bf16(
                a,
                b,
                batch_count,
                m,
                k,
                n,
                trans_a,
                trans_b,
            ) {
                return gpu_out;
            }
        }
        #[cfg(feature = "scry-gpu-cuda")]
        if let Some(gpu_out) = gpu_matmul_strided_batched_persistent(
            a,
            b,
            batch_count,
            m,
            k,
            n,
            trans_a,
            trans_b,
        ) {
            return gpu_out;
        }
        let a_stride = m * k;
        let b_stride = k * n;
        let c_stride = m * n;
        let total = batch_count * c_stride;
        let av = Self::to_vec(a);
        let bv = Self::to_vec(b);
        let mut cv = vec![0.0f32; total];
        for i in 0..batch_count {
            let a_slice = Self::from_vec(
                av[i * a_stride..(i + 1) * a_stride].to_vec(),
                &Shape::new(&[m, k]),
            );
            let b_slice = Self::from_vec(
                bv[i * b_stride..(i + 1) * b_stride].to_vec(),
                &Shape::new(&[k, n]),
            );
            let c_slice = Self::matmul(&a_slice, &b_slice, m, k, n, trans_a, trans_b);
            let c_data = Self::to_vec(&c_slice);
            cv[i * c_stride..(i + 1) * c_stride].copy_from_slice(&c_data);
        }
        Self::from_vec(cv, &Shape::new(&[batch_count * m, n]))
    }

    fn add(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> ScryGpuStorage {
        let a_dims = a_shape.dims();
        let b_dims = b_shape.dims();
        let out_dims = out_shape.dims();

        // Fast path: same-shape add — ResNet residual adds, attention output
        // sums, etc. Most common case; checked first so we don't fall through
        // the row-bias branch.
        if a_dims == out_dims && b_dims == out_dims {
            if let Some(gpu_out) = gpu_add_elementwise_persistent(a, b, out_shape.numel()) {
                return gpu_out;
            }
        }

        // Fast path: column broadcast `[rows, cols] + [rows, 1]` — Conv2d's
        // bias add post-matmul. Mirrors the same pattern in `CpuBackend::add`
        // (and its swapped-operand twin). Kept on-device so the conv result
        // chains into the next layer without a round trip.
        if out_dims.len() == 2 {
            let (rows, cols) = (out_dims[0], out_dims[1]);
            if a_dims == out_dims && b_dims == [rows, 1] {
                if let Some(gpu_out) = gpu_add_row_bias_persistent(a, b, rows, cols) {
                    return gpu_out;
                }
            } else if b_dims == out_dims && a_dims == [rows, 1] {
                if let Some(gpu_out) = gpu_add_row_bias_persistent(b, a, rows, cols) {
                    return gpu_out;
                }
            }

            // Fast path: row broadcast `[rows, cols] + [1, cols]` — every
            // transformer attention block adds a `[1, 3*d_model]` qkv-bias
            // and a `[1, d_model]` proj-bias in this shape. Reuses the
            // `bias_add` kernel that already powers `matmul_bias` (a
            // `[cols]`-shaped bias is the same gather pattern as a
            // `[1, cols]` one). 24 calls per CLIP encode at SD 1.5.
            if a_dims == out_dims && b_dims == [1, cols] {
                if let Some(gpu_out) = gpu_bias_add_persistent(a, b, rows, cols) {
                    return gpu_out;
                }
            } else if b_dims == out_dims && a_dims == [1, cols] {
                if let Some(gpu_out) = gpu_bias_add_persistent(b, a, rows, cols) {
                    return gpu_out;
                }
            }
        }
        cpu(CpuBackend::add(
            &a.as_vec(),
            &b.as_vec(),
            a_shape,
            b_shape,
            out_shape,
        ))
    }

    fn conv2d_forward(
        input: &ScryGpuStorage,
        weight: &ScryGpuStorage,
        in_channels: usize,
        h_in: usize,
        w_in: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> ScryGpuStorage {
        // cuDNN: fused implicit-GEMM, skips the im2col HBM pass. Falls
        // through to the default (im2col + matmul) path when unavailable
        // (Vulkan backend, sub-threshold workload, weights still CPU-side).
        #[cfg(feature = "scry-gpu-cudnn")]
        if let Some(out) = gpu_conv2d_cudnn_persistent(
            input,
            weight,
            in_channels,
            h_in,
            w_in,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
        ) {
            return out;
        }

        // Default: same `im2col_2d` + `matmul` composition the trait does,
        // but kept inline so we hit `Self::im2col_2d` (which routes to the
        // GPU im2col kernel) and `Self::matmul` (which routes to cuBLAS).
        let h_out = (h_in + 2 * padding - kernel_h) / stride + 1;
        let w_out = (w_in + 2 * padding - kernel_w) / stride + 1;
        let lowered = Self::im2col_2d(
            input,
            in_channels,
            h_in,
            w_in,
            kernel_h,
            kernel_w,
            stride,
            padding,
        );
        Self::matmul(
            weight,
            &lowered,
            out_channels,
            in_channels * kernel_h * kernel_w,
            h_out * w_out,
            false,
            false,
        )
    }

    fn im2col_2d(
        input: &ScryGpuStorage,
        in_channels: usize,
        h_in: usize,
        w_in: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    ) -> ScryGpuStorage {
        let h_out = (h_in + 2 * padding - kernel_h) / stride + 1;
        let w_out = (w_in + 2 * padding - kernel_w) / stride + 1;
        if let Some(gpu_out) = gpu_im2col_persistent(
            input,
            in_channels,
            h_in,
            w_in,
            kernel_h,
            kernel_w,
            stride,
            padding,
            h_out,
            w_out,
        ) {
            return gpu_out;
        }
        cpu(CpuBackend::im2col_2d(
            &input.as_vec(),
            in_channels,
            h_in,
            w_in,
            kernel_h,
            kernel_w,
            stride,
            padding,
        ))
    }

    fn softmax(input: &ScryGpuStorage, shape: &Shape) -> ScryGpuStorage {
        if let Some(gpu_out) = gpu_softmax_persistent(input, shape) {
            return gpu_out;
        }
        cpu(CpuBackend::softmax(&input.as_vec(), shape))
    }

    fn scaled_softmax(input: &ScryGpuStorage, scale: f32, shape: &Shape) -> ScryGpuStorage {
        if let Some(gpu_out) = gpu_scaled_softmax_persistent(input, scale, shape) {
            return gpu_out;
        }
        // Fall through to the trait default — Self::scale already has a CUDA
        // override, so even on the slow path we stay device-resident; the
        // only cost vs the fused kernel is the standalone scale dispatch.
        let scaled = Self::scale(input, scale);
        Self::softmax(&scaled, shape)
    }

    fn gather_columns(
        storage: &ScryGpuStorage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> ScryGpuStorage {
        if let Some(gpu_out) =
            gpu_gather_columns_persistent(storage, rows, total_cols, col_start, col_count)
        {
            return gpu_out;
        }
        // Trait-default body: download, scalar gather, re-upload.
        let data = Self::to_vec(storage);
        let mut out = vec![0.0f32; rows * col_count];
        for r in 0..rows {
            for c in 0..col_count {
                out[r * col_count + c] = data[r * total_cols + col_start + c];
            }
        }
        Self::from_vec(out, &Shape::new(&[rows, col_count]))
    }

    fn scatter_columns(
        dst: &mut ScryGpuStorage,
        src: &ScryGpuStorage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) {
        if gpu_scatter_columns_persistent(dst, src, rows, total_cols, col_start, col_count)
            .is_some()
        {
            return;
        }
        // Trait-default body: download, scalar scatter, re-upload.
        let mut dst_vec = Self::to_vec(dst);
        let src_vec = Self::to_vec(src);
        for r in 0..rows {
            for c in 0..col_count {
                dst_vec[r * total_cols + col_start + c] += src_vec[r * col_count + c];
            }
        }
        *dst = Self::from_vec(dst_vec, &Shape::new(&[rows, total_cols]));
    }

    fn apply_causal_mask_and_scale(
        scores: &mut ScryGpuStorage,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        if gpu_apply_causal_mask_and_scale_persistent(scores, seq_len, scale, mask_value).is_some()
        {
            return;
        }
        // Trait-default body: download, scalar mask + scale, re-upload.
        let mut data = Self::to_vec(scores);
        for s in 0..seq_len {
            for t in 0..seq_len {
                if t > s {
                    data[s * seq_len + t] = mask_value;
                } else {
                    data[s * seq_len + t] *= scale;
                }
            }
        }
        *scores = Self::from_vec(data, &Shape::new(&[seq_len, seq_len]));
    }

    fn layernorm(
        input: &ScryGpuStorage,
        gamma: &ScryGpuStorage,
        beta: &ScryGpuStorage,
        shape: &Shape,
        eps: f32,
    ) -> (ScryGpuStorage, ScryGpuStorage, ScryGpuStorage) {
        if let Some(triple) = gpu_layernorm_persistent(input, gamma, beta, shape, eps) {
            return triple;
        }
        let (out, mean, rstd) =
            CpuBackend::layernorm(&input.as_vec(), &gamma.as_vec(), &beta.as_vec(), shape, eps);
        (cpu(out), cpu(mean), cpu(rstd))
    }

    fn group_norm(
        input: &ScryGpuStorage,
        weight: &ScryGpuStorage,
        bias: &ScryGpuStorage,
        num_groups: usize,
        channels: usize,
        spatial: usize,
        eps: f32,
    ) -> ScryGpuStorage {
        if let Some(gpu_out) = gpu_group_norm_persistent(
            input, weight, bias, num_groups, channels, spatial, eps,
        ) {
            return gpu_out;
        }
        // CpuBackend has no `group_norm` op — go through the trait default
        // impl on `CpuBackend` directly. Materializes any Gpu inputs to host
        // first.
        let input_v = input.as_vec();
        let weight_v = weight.as_vec();
        let bias_v = bias.as_vec();
        cpu(<CpuBackend as MathBackend>::group_norm(
            &input_v,
            &weight_v,
            &bias_v,
            num_groups,
            channels,
            spatial,
            eps,
        ))
    }

    fn batchnorm_2d_inference(
        input: &ScryGpuStorage,
        weight: &ScryGpuStorage,
        bias: &ScryGpuStorage,
        running_mean: &ScryGpuStorage,
        running_var: &ScryGpuStorage,
        channels: usize,
        spatial: usize,
        eps: f32,
    ) -> ScryGpuStorage {
        if let Some(gpu_out) = gpu_batchnorm_persistent(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            channels,
            spatial,
            eps,
        ) {
            return gpu_out;
        }
        cpu(CpuBackend::batchnorm_2d_inference(
            &input.as_vec(),
            &weight.as_vec(),
            &bias.as_vec(),
            &running_mean.as_vec(),
            &running_var.as_vec(),
            channels,
            spatial,
            eps,
        ))
    }

    fn gelu(input: &ScryGpuStorage) -> ScryGpuStorage {
        if let Some(gpu_out) = gpu_gelu_persistent(input) {
            return gpu_out;
        }
        cpu(CpuBackend::gelu(&input.as_vec()))
    }

    fn relu(input: &ScryGpuStorage) -> ScryGpuStorage {
        if let Some(gpu_out) = gpu_relu_persistent(input) {
            return gpu_out;
        }
        // CpuBackend has no `relu` op — inline the same scalar map the
        // default trait impl uses, but reuse the borrowed Vec to avoid an
        // extra allocation when we're already on host.
        let v = input.as_vec();
        let out: Vec<f32> = v.iter().map(|x| x.max(0.0)).collect();
        cpu(out)
    }

    fn silu(input: &ScryGpuStorage) -> ScryGpuStorage {
        if let Some(gpu_out) = gpu_silu_persistent(input) {
            return gpu_out;
        }
        // CpuBackend has no `silu` op — inline the same scalar map the
        // default trait impl uses, reusing the borrowed Vec on host.
        let v = input.as_vec();
        let out: Vec<f32> = v.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        cpu(out)
    }

    fn gelu_exact(input: &ScryGpuStorage) -> ScryGpuStorage {
        if let Some(gpu_out) = gpu_gelu_exact_persistent(input) {
            return gpu_out;
        }
        // Fall through to MathBackend's CPU default (erf-based, runs on host).
        cpu(CpuBackend::gelu_exact(&input.as_vec()))
    }

    fn max_pool_2d(
        input: &ScryGpuStorage,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> ScryGpuStorage {
        let h_out = (h_in + 2 * padding - kernel) / stride + 1;
        let w_out = (w_in + 2 * padding - kernel) / stride + 1;
        if let Some(gpu_out) = gpu_max_pool_persistent(
            input, channels, h_in, w_in, kernel, stride, padding, h_out, w_out,
        ) {
            return gpu_out;
        }
        cpu(CpuBackend::max_pool_2d(
            &input.as_vec(),
            channels,
            h_in,
            w_in,
            kernel,
            stride,
            padding,
        ))
    }

    fn adaptive_avg_pool_2d(
        input: &ScryGpuStorage,
        channels: usize,
        h_in: usize,
        w_in: usize,
        h_out: usize,
        w_out: usize,
    ) -> ScryGpuStorage {
        if let Some(gpu_out) =
            gpu_adaptive_avg_pool_persistent(input, channels, h_in, w_in, h_out, w_out)
        {
            return gpu_out;
        }
        cpu(CpuBackend::adaptive_avg_pool_2d(
            &input.as_vec(),
            channels,
            h_in,
            w_in,
            h_out,
            w_out,
        ))
    }

    fn upsample_2d_nearest(
        input: &ScryGpuStorage,
        channels: usize,
        h_in: usize,
        w_in: usize,
        scale: usize,
    ) -> ScryGpuStorage {
        if let Some(gpu_out) =
            gpu_upsample_2d_nearest_persistent(input, channels, h_in, w_in, scale)
        {
            return gpu_out;
        }
        // CpuBackend has no `upsample_2d_nearest` op — fall through to the
        // trait default by going through `as_vec` and the per-element loop.
        // Mirror of the default impl in `MathBackend::upsample_2d_nearest`.
        let h_out = h_in * scale;
        let w_out = w_in * scale;
        let v = input.as_vec();
        let mut out = vec![0.0f32; channels * h_out * w_out];
        for c in 0..channels {
            let in_plane = c * h_in * w_in;
            let out_plane = c * h_out * w_out;
            for oh in 0..h_out {
                let ih = oh / scale;
                for ow in 0..w_out {
                    let iw = ow / scale;
                    out[out_plane + oh * w_out + ow] = v[in_plane + ih * w_in + iw];
                }
            }
        }
        cpu(out)
    }

    fn embedding(
        weight: &ScryGpuStorage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> ScryGpuStorage {
        if let Some(out) = gpu_embedding_persistent(weight, indices, vocab, dim) {
            return out;
        }
        cpu(CpuBackend::embedding(&weight.as_vec(), indices, vocab, dim))
    }

    fn sum(input: &ScryGpuStorage) -> f32 {
        CpuBackend::sum(&input.as_vec())
    }

    fn mul_elementwise(a: &ScryGpuStorage, b: &ScryGpuStorage) -> ScryGpuStorage {
        let n = a.len();
        if n == b.len() {
            if let Some(out) = gpu_mul_elementwise_persistent(a, b, n) {
                return out;
            }
        }
        cpu(CpuBackend::mul_elementwise(&a.as_vec(), &b.as_vec()))
    }

    fn scale(a: &ScryGpuStorage, scalar: f32) -> ScryGpuStorage {
        if let Some(out) = gpu_scale_persistent(a, scalar) {
            return out;
        }
        cpu(CpuBackend::scale(&a.as_vec(), scalar))
    }

    fn concat_rows(
        a: &ScryGpuStorage,
        b: &ScryGpuStorage,
        a_rows: usize,
        b_rows: usize,
        cols: usize,
    ) -> ScryGpuStorage {
        let a_total = a_rows * cols;
        let b_total = b_rows * cols;
        if let Some(out) = gpu_concat_rows_persistent(a, b, a_total, b_total) {
            return out;
        }
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
#[allow(clippy::manual_let_else)] // Match arms keep the error message for skip diagnostics.
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
    fn gpu_relu_matches_reference_within_tolerance() {
        // Above the elementwise threshold so the GPU path engages. Mix of
        // negatives, zeros, and positives across the input range.
        let n = 32_768;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 16.0).collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_relu_matches_reference_within_tolerance: {e}");
                return;
            }
        };
        let gpu_out = ScryGpuBackend::relu(&gpu);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "relu over Gpu input should stay Gpu on CUDA, got {gpu_out:?}"
        );
        let g = ScryGpuBackend::to_vec(&gpu_out);
        let reference: Vec<f32> = input.iter().map(|x| x.max(0.0)).collect();
        assert_eq!(g.len(), reference.len());
        for (i, (gv, rv)) in g.iter().zip(reference.iter()).enumerate() {
            // ReLU is exact in f32 (no transcendentals); equality holds.
            assert!((gv - rv).abs() == 0.0, "idx={i}: gpu={gv} ref={rv}");
        }
    }

    #[test]
    fn small_relu_falls_back_to_cpu() {
        // Below GPU_ELEMENTWISE_MIN — same threshold as gelu since both use
        // `run_unary_elementwise`. Tensor stays/lands on host.
        let n = 1024;
        assert!(n < GPU_ELEMENTWISE_MIN);
        let input: Vec<f32> = (0..n).map(|i| (i as f32) - 512.0).collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping small_relu_falls_back_to_cpu: {e}");
                return;
            }
        };
        let out = ScryGpuBackend::relu(&gpu);
        assert!(!out.is_gpu(), "small relu should fall back to Cpu");
        let v = ScryGpuBackend::to_vec(&out);
        for (i, (got, x)) in v.iter().zip(input.iter()).enumerate() {
            assert!((got - x.max(0.0)).abs() == 0.0, "idx={i}: got={got} x={x}");
        }
    }

    #[test]
    fn gpu_gelu_matches_cpu_within_tolerance() {
        // Above the elementwise threshold so the GPU path engages.
        let n = 32_768;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 16.0).collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_gelu_matches_cpu_within_tolerance: {e}");
                return;
            }
        };
        let gpu_out = ScryGpuBackend::gelu(&gpu);
        assert!(
            gpu_out.is_gpu(),
            "gelu over Gpu input should stay Gpu, got {gpu_out:?}"
        );
        let g = ScryGpuBackend::to_vec(&gpu_out);
        let c = CpuBackend::gelu(&input);
        assert_eq!(g.len(), c.len());
        for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
            // GPU runs in f32, CPU in f64-then-cast. Allow 1e-5 absolute.
            assert!((gv - cv).abs() < 1e-5, "mismatch at {i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_softmax_matches_cpu_within_tolerance() {
        // Sized above GPU_SOFTMAX_MIN_ROWS so the kernel engages, with d
        // both small (=64) and larger-than-block (=512) cases.
        for (rows, d) in [(64usize, 64usize), (32, 512)] {
            let input: Vec<f32> = (0..rows * d).map(|i| ((i % 97) as f32) * 0.05 - 2.0).collect();
            let shape = Shape::new(&[rows, d]);
            let storage = ScryGpuStorage::Cpu(input.clone());
            let gpu = match ScryGpuBackend::to_gpu(&storage) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping gpu_softmax_matches_cpu_within_tolerance: {e}");
                    return;
                }
            };
            let gpu_out = ScryGpuBackend::softmax(&gpu, &shape);
            // On CUDA the kernel must engage; falling back to CPU here means
            // the kernel compile or dispatch failed silently.
            #[cfg(feature = "scry-gpu-cuda")]
            assert!(
                gpu_out.is_gpu(),
                "softmax over Gpu input should stay Gpu on CUDA, got {gpu_out:?}"
            );
            let g = ScryGpuBackend::to_vec(&gpu_out);
            let c = CpuBackend::softmax(&input, &shape);
            assert_eq!(g.len(), c.len());
            for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
                // CPU softmax accumulates in f64 then casts; GPU runs entirely in
                // f32 with a 256-thread tree reduction. 1e-5 absolute holds for
                // the tested ranges.
                assert!(
                    (gv - cv).abs() < 1e-5,
                    "rows={rows} d={d} idx={i}: gpu={gv} cpu={cv}"
                );
            }
            // Each row should sum to 1.
            for r in 0..rows {
                let s: f32 = g[r * d..(r + 1) * d].iter().sum();
                assert!(
                    (s - 1.0).abs() < 1e-4,
                    "row {r} did not sum to 1: got {s}"
                );
            }
        }
    }

    #[test]
    fn gpu_scaled_softmax_matches_cpu_within_tolerance() {
        // Mirror of `gpu_softmax_matches_cpu_within_tolerance` but exercising
        // the fused `scale + softmax` path. Tests both positive and negative
        // scale to confirm the max-shift trick stays correct (max is taken
        // over scaled values so any scale works).
        for (rows, d, scale) in [
            (64usize, 64usize, 0.1f32),
            (32, 512, -0.25),
            (32, 512, 1.0 / (40.0_f32).sqrt()), // SD self-attn deepest stage
        ] {
            let input: Vec<f32> = (0..rows * d).map(|i| ((i % 97) as f32) * 0.05 - 2.0).collect();
            let shape = Shape::new(&[rows, d]);
            let storage = ScryGpuStorage::Cpu(input.clone());
            let gpu = match ScryGpuBackend::to_gpu(&storage) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping gpu_scaled_softmax_matches_cpu_within_tolerance: {e}");
                    return;
                }
            };
            let gpu_out = ScryGpuBackend::scaled_softmax(&gpu, scale, &shape);
            #[cfg(feature = "scry-gpu-cuda")]
            assert!(
                gpu_out.is_gpu(),
                "scaled_softmax over Gpu input should stay Gpu on CUDA, got {gpu_out:?}"
            );
            let g = ScryGpuBackend::to_vec(&gpu_out);
            let c = CpuBackend::scaled_softmax(&input, scale, &shape);
            assert_eq!(g.len(), c.len());
            for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
                assert!(
                    (gv - cv).abs() < 1e-5,
                    "rows={rows} d={d} scale={scale} idx={i}: gpu={gv} cpu={cv}"
                );
            }
            for r in 0..rows {
                let s: f32 = g[r * d..(r + 1) * d].iter().sum();
                assert!(
                    (s - 1.0).abs() < 1e-4,
                    "rows={rows} d={d} scale={scale} row {r} did not sum to 1: got {s}"
                );
            }
        }
    }

    #[test]
    fn gpu_layernorm_matches_cpu_within_tolerance() {
        // Sized above GPU_LAYERNORM_MIN_ROWS so the kernel engages, with d
        // both small (=64) and larger-than-block (=512) cases.
        for (rows, d) in [(64usize, 64usize), (32, 512)] {
            let input: Vec<f32> = (0..rows * d).map(|i| ((i % 97) as f32) * 0.05 - 2.0).collect();
            let gamma: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.01).collect();
            let beta: Vec<f32> = (0..d).map(|i| (i as f32) * 0.005 - 0.5).collect();
            let shape = Shape::new(&[rows, d]);
            let eps = 1e-5_f32;

            let in_storage = ScryGpuStorage::Cpu(input.clone());
            let g_storage = ScryGpuStorage::Cpu(gamma.clone());
            let b_storage = ScryGpuStorage::Cpu(beta.clone());

            let gpu_in = match ScryGpuBackend::to_gpu(&in_storage) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping gpu_layernorm_matches_cpu_within_tolerance: {e}");
                    return;
                }
            };

            let (out, means, rstds) =
                ScryGpuBackend::layernorm(&gpu_in, &g_storage, &b_storage, &shape, eps);
            // On CUDA the kernel must engage; falling back to CPU here means
            // the kernel compile or dispatch failed silently.
            #[cfg(feature = "scry-gpu-cuda")]
            {
                assert!(
                    out.is_gpu(),
                    "layernorm output should stay Gpu on CUDA, got {out:?}"
                );
                assert!(
                    means.is_gpu(),
                    "layernorm means should stay Gpu on CUDA, got {means:?}"
                );
                assert!(
                    rstds.is_gpu(),
                    "layernorm rstds should stay Gpu on CUDA, got {rstds:?}"
                );
            }

            let g_out = ScryGpuBackend::to_vec(&out);
            let g_means = ScryGpuBackend::to_vec(&means);
            let g_rstds = ScryGpuBackend::to_vec(&rstds);

            let (c_out, c_means, c_rstds) =
                CpuBackend::layernorm(&input, &gamma, &beta, &shape, eps);

            assert_eq!(g_out.len(), c_out.len());
            assert_eq!(g_means.len(), rows);
            assert_eq!(g_rstds.len(), rows);

            // Per-row mean and rstd: f32 sums vs CPU's f64 reductions.
            // 1e-3 absolute easily holds for both d values (mean magnitudes
            // are ~few units; rstd ~O(1)).
            for r in 0..rows {
                let dm = (g_means[r] - c_means[r]).abs();
                let dr = (g_rstds[r] - c_rstds[r]).abs();
                assert!(
                    dm < 1e-3,
                    "rows={rows} d={d} row {r} mean mismatch: gpu={} cpu={} delta={dm}",
                    g_means[r],
                    c_means[r]
                );
                assert!(
                    dr < 1e-3,
                    "rows={rows} d={d} row {r} rstd mismatch: gpu={} cpu={} delta={dr}",
                    g_rstds[r],
                    c_rstds[r]
                );
            }

            // Output: relative tolerance — the affine scale lets row values
            // grow with gamma/beta, and f32 accumulation drift in the
            // reductions feeds through to every output element.
            for (i, (gv, cv)) in g_out.iter().zip(c_out.iter()).enumerate() {
                let tol = 1e-4 * cv.abs().max(1.0);
                assert!(
                    (gv - cv).abs() < tol,
                    "rows={rows} d={d} idx={i}: gpu={gv} cpu={cv} (tol={tol})"
                );
            }
        }
    }

    #[test]
    fn small_layernorm_falls_back_to_cpu() {
        // Below GPU_LAYERNORM_MIN_ROWS: result should come back as Cpu variant.
        let rows = 4;
        let d = 32;
        let input: Vec<f32> = (0..rows * d).map(|i| (i as f32) * 0.01).collect();
        let gamma: Vec<f32> = vec![1.0; d];
        let beta: Vec<f32> = vec![0.0; d];
        let shape = Shape::new(&[rows, d]);
        let in_s = ScryGpuStorage::Cpu(input);
        let g_s = ScryGpuStorage::Cpu(gamma);
        let b_s = ScryGpuStorage::Cpu(beta);
        let (out, means, rstds) = ScryGpuBackend::layernorm(&in_s, &g_s, &b_s, &shape, 1e-5);
        assert!(!out.is_gpu(), "small layernorm output should fall back to Cpu");
        assert!(!means.is_gpu(), "small layernorm means should fall back to Cpu");
        assert!(!rstds.is_gpu(), "small layernorm rstds should fall back to Cpu");
    }

    #[test]
    fn small_softmax_falls_back_to_cpu() {
        // Below GPU_SOFTMAX_MIN_ROWS: result should come back as Cpu variant.
        let rows = 4;
        let d = 32;
        let input: Vec<f32> = (0..rows * d).map(|i| (i as f32) * 0.01).collect();
        let shape = Shape::new(&[rows, d]);
        let storage = ScryGpuStorage::Cpu(input);
        let out = ScryGpuBackend::softmax(&storage, &shape);
        assert!(!out.is_gpu(), "small softmax should fall back to Cpu");
    }

    #[test]
    fn small_gelu_falls_back_to_cpu() {
        // Below the elementwise threshold: GPU dispatch isn't worth it,
        // result should come back as Cpu variant.
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let storage = ScryGpuStorage::Cpu(input);
        let out = ScryGpuBackend::gelu(&storage);
        assert!(!out.is_gpu(), "small gelu should fall back to Cpu");
    }

    #[test]
    fn gpu_silu_matches_cpu_within_tolerance() {
        // Above the elementwise threshold so the GPU path engages. Same
        // input distribution as the gelu test — span across the negative
        // saturation region and the linear-ish regime.
        let n = 32_768;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 16.0).collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_silu_matches_cpu_within_tolerance: {e}");
                return;
            }
        };
        let gpu_out = ScryGpuBackend::silu(&gpu);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "silu over Gpu input should stay Gpu on CUDA, got {gpu_out:?}"
        );
        let g = ScryGpuBackend::to_vec(&gpu_out);
        // Reference: same scalar formula the trait default uses.
        let reference: Vec<f32> = input.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        assert_eq!(g.len(), reference.len());
        for (i, (gv, rv)) in g.iter().zip(reference.iter()).enumerate() {
            // expf is the only transcendental; at the tested range f32 round-off
            // stays well under 1e-5 absolute.
            assert!((gv - rv).abs() < 1e-5, "mismatch at {i}: gpu={gv} ref={rv}");
        }
    }

    #[test]
    fn small_silu_falls_back_to_cpu() {
        // Below the elementwise threshold: GPU dispatch isn't worth it,
        // result should come back as Cpu variant.
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let storage = ScryGpuStorage::Cpu(input);
        let out = ScryGpuBackend::silu(&storage);
        assert!(!out.is_gpu(), "small silu should fall back to Cpu");
    }

    #[test]
    fn gpu_gelu_exact_matches_cpu_within_tolerance() {
        // Spans negative saturation, the linear regime through zero,
        // and positive saturation — same coverage as the silu/gelu tests.
        // The GPU path uses CUDA's `erff`; the CPU reference uses the
        // Abramowitz–Stegun 7.1.26 polynomial in f64. Both are within ~1.5e-7
        // of true erf, so fp32 round-off dominates the diff.
        let n = 32_768;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 16.0).collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_gelu_exact_matches_cpu_within_tolerance: {e}");
                return;
            }
        };
        let gpu_out = ScryGpuBackend::gelu_exact(&gpu);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "gelu_exact over Gpu input should stay Gpu on CUDA, got {gpu_out:?}"
        );
        let g = ScryGpuBackend::to_vec(&gpu_out);
        let reference = CpuBackend::gelu_exact(&input);
        assert_eq!(g.len(), reference.len());
        for (i, (gv, rv)) in g.iter().zip(reference.iter()).enumerate() {
            // erff vs Abramowitz–Stegun: both ~1.5e-7 vs true erf, plus
            // fp32 round-off in the multiply. Generous 1e-5 envelope.
            assert!(
                (gv - rv).abs() < 1e-5,
                "mismatch at {i}: gpu={gv} ref={rv}"
            );
        }
    }

    #[test]
    fn small_gelu_exact_falls_back_to_cpu() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let storage = ScryGpuStorage::Cpu(input);
        let out = ScryGpuBackend::gelu_exact(&storage);
        assert!(!out.is_gpu(), "small gelu_exact should fall back to Cpu");
    }

    #[test]
    fn gpu_batchnorm_matches_cpu_within_tolerance() {
        // Channel counts above GPU_BATCHNORM_MIN_CHANNELS, with both small and
        // larger-than-block spatial dims to exercise the per-thread strided loop.
        for (channels, spatial) in [(64usize, 49usize), (32, 256), (128, 1024)] {
            let total = channels * spatial;
            let input: Vec<f32> = (0..total).map(|i| ((i % 113) as f32) * 0.05 - 2.5).collect();
            let weight: Vec<f32> = (0..channels).map(|c| 0.5 + (c as f32) * 0.01).collect();
            let bias: Vec<f32> = (0..channels).map(|c| (c as f32) * 0.005 - 0.25).collect();
            let mean: Vec<f32> = (0..channels).map(|c| (c as f32) * 0.02 - 0.5).collect();
            let var: Vec<f32> = (0..channels).map(|c| 0.5 + (c as f32) * 0.01).collect();
            let eps = 1e-5_f32;

            let in_s = ScryGpuStorage::Cpu(input.clone());
            let w_s = ScryGpuStorage::Cpu(weight.clone());
            let b_s = ScryGpuStorage::Cpu(bias.clone());
            let m_s = ScryGpuStorage::Cpu(mean.clone());
            let v_s = ScryGpuStorage::Cpu(var.clone());

            let gpu_in = match ScryGpuBackend::to_gpu(&in_s) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping gpu_batchnorm_matches_cpu_within_tolerance: {e}");
                    return;
                }
            };

            let out =
                ScryGpuBackend::batchnorm_2d_inference(&gpu_in, &w_s, &b_s, &m_s, &v_s, channels, spatial, eps);
            #[cfg(feature = "scry-gpu-cuda")]
            assert!(
                out.is_gpu(),
                "batchnorm output should stay Gpu on CUDA, got {out:?}"
            );

            let g_out = ScryGpuBackend::to_vec(&out);
            let c_out = CpuBackend::batchnorm_2d_inference(
                &input, &weight, &bias, &mean, &var, channels, spatial, eps,
            );
            assert_eq!(g_out.len(), c_out.len());

            // Pure elementwise op — no reductions, so f32 round-off is small
            // and 1e-5 absolute tolerance holds across the tested ranges.
            for (i, (gv, cv)) in g_out.iter().zip(c_out.iter()).enumerate() {
                assert!(
                    (gv - cv).abs() < 1e-5,
                    "channels={channels} spatial={spatial} idx={i}: gpu={gv} cpu={cv}"
                );
            }
        }
    }

    #[test]
    fn gpu_im2col_matches_cpu_within_tolerance() {
        // Two configurations: ResNet first-layer (3×224×224, 7×7 stride-2 pad-3
        // → 3*7*7=147 rows × 112*112=12_544 cols, 1.84M output elements) and a
        // mid-network 3×3 conv (64×56×56, kernel-3 stride-1 pad-1 → 576 × 3136,
        // 1.81M elements). Both clear GPU_IM2COL_MIN_OUTPUT_ELEMENTS so the
        // kernel must engage on CUDA; falling back here means a silent compile
        // or dispatch failure.
        let cases: &[(usize, usize, usize, usize, usize, usize)] = &[
            // (c_in, h, w, kernel, stride, padding)
            (3, 224, 224, 7, 2, 3),
            (64, 56, 56, 3, 1, 1),
        ];
        for &(c_in, h_in, w_in, k, stride, padding) in cases {
            let total = c_in * h_in * w_in;
            let input: Vec<f32> = (0..total).map(|i| ((i % 113) as f32) * 0.05 - 2.5).collect();
            let storage = ScryGpuStorage::Cpu(input.clone());
            let gpu = match ScryGpuBackend::to_gpu(&storage) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping gpu_im2col_matches_cpu_within_tolerance: {e}");
                    return;
                }
            };
            let gpu_out =
                ScryGpuBackend::im2col_2d(&gpu, c_in, h_in, w_in, k, k, stride, padding);
            #[cfg(feature = "scry-gpu-cuda")]
            assert!(
                gpu_out.is_gpu(),
                "im2col output should stay Gpu on CUDA, got {gpu_out:?}"
            );
            let g = ScryGpuBackend::to_vec(&gpu_out);
            let c = CpuBackend::im2col_2d(&input, c_in, h_in, w_in, k, k, stride, padding);
            assert_eq!(g.len(), c.len(), "len mismatch for case ({c_in},{h_in},{w_in},{k},{stride},{padding})");
            // Pure copy/zero kernel — values must match exactly.
            for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
                assert!(
                    (gv - cv).abs() < 1e-6,
                    "case ({c_in},{h_in},{w_in},{k},{stride},{padding}) idx={i}: gpu={gv} cpu={cv}"
                );
            }
        }
    }

    #[cfg(feature = "scry-gpu-cudnn")]
    #[test]
    fn gpu_conv2d_forward_cudnn_matches_cpu_within_tolerance() {
        // ResNet stem (3×224×224, 7×7s2p3, 64 out) and a 3×3-s1-p1 stage
        // (64×56×56, 64 out) — both above GPU_CUDNN_CONV2D_MIN_OUTPUT_ELEMENTS
        // so the cuDNN path must engage. Validates that conv2d_forward on
        // ScryGpuBackend produces the same flat output (without bias) as
        // CpuBackend's default impl (= im2col + matmul).
        let cases: &[(usize, usize, usize, usize, usize, usize, usize)] = &[
            // (c_in, h, w, c_out, kernel, stride, padding)
            (3, 224, 224, 64, 7, 2, 3),
            (64, 56, 56, 64, 3, 1, 1),
        ];
        for &(c_in, h_in, w_in, c_out, k, stride, padding) in cases {
            let total_in = c_in * h_in * w_in;
            let total_w = c_out * c_in * k * k;
            let input: Vec<f32> = (0..total_in)
                .map(|i| ((i % 113) as f32) * 0.05 - 2.5)
                .collect();
            let weight: Vec<f32> = (0..total_w)
                .map(|i| ((i % 17) as f32) * 0.03 - 0.25)
                .collect();

            let in_storage = ScryGpuStorage::Cpu(input.clone());
            let w_storage = ScryGpuStorage::Cpu(weight.clone());
            let in_gpu = match ScryGpuBackend::to_gpu(&in_storage) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping cudnn conv test: {e}");
                    return;
                }
            };
            let w_gpu = ScryGpuBackend::to_gpu(&w_storage).expect("upload weight");

            let gpu_out = ScryGpuBackend::conv2d_forward(
                &in_gpu, &w_gpu, c_in, h_in, w_in, c_out, k, k, stride, padding,
            );
            assert!(
                gpu_out.is_gpu(),
                "conv2d_forward output should stay Gpu on CUDA"
            );

            let g = ScryGpuBackend::to_vec(&gpu_out);
            let c = CpuBackend::conv2d_forward(
                &input, &weight, c_in, h_in, w_in, c_out, k, k, stride, padding,
            );
            assert_eq!(g.len(), c.len());
            // cuDNN's implicit-GEMM accumulator order may differ from im2col +
            // SGEMM, so allow a small fp32 rounding tolerance scaled by the
            // reduction width.
            let tol = 1e-3 * (c_in * k * k) as f32;
            for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
                assert!(
                    (gv - cv).abs() < tol,
                    "case ({c_in},{h_in},{w_in},{c_out},{k},{stride},{padding}) idx={i}: gpu={gv} cpu={cv}"
                );
            }
        }
    }

    #[test]
    fn small_im2col_falls_back_to_cpu() {
        // Below GPU_IM2COL_MIN_OUTPUT_ELEMENTS (3*3*3*4*4 = 432 elements):
        // result should come back as Cpu variant.
        let c_in = 3;
        let (h, w, k, stride, padding) = (4, 4, 3, 1, 1);
        let input: Vec<f32> = (0..c_in * h * w).map(|i| (i as f32) * 0.01).collect();
        let storage = ScryGpuStorage::Cpu(input);
        let out = ScryGpuBackend::im2col_2d(&storage, c_in, h, w, k, k, stride, padding);
        assert!(!out.is_gpu(), "small im2col should fall back to Cpu");
    }

    #[test]
    fn gpu_split_qkv_reshape_heads_matches_cpu_within_tolerance() {
        // ViT-B/16 attention shape: seq=197 (CLS + 14×14 patches),
        // n_heads=12, d_head=64. Total per output = 12*197*64 = 151_296 — well
        // above GPU_SPLIT_QKV_MIN.
        let seq = 197;
        let n_heads = 12;
        let d_head = 64;
        let d_model = n_heads * d_head;
        let total_in = seq * 3 * d_model;
        let qkv: Vec<f32> = (0..total_in)
            .map(|i| ((i % 211) as f32 - 100.0) * 0.011)
            .collect();
        let storage = ScryGpuStorage::Cpu(qkv.clone());
        let gpu_in = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_split_qkv test: {e}");
                return;
            }
        };

        let (q, k, v) = ScryGpuBackend::split_qkv_reshape_heads(&gpu_in, seq, n_heads, d_head);
        #[cfg(feature = "scry-gpu-cuda")]
        {
            assert!(q.is_gpu(), "q should stay Gpu on CUDA");
            assert!(k.is_gpu(), "k should stay Gpu on CUDA");
            assert!(v.is_gpu(), "v should stay Gpu on CUDA");
        }

        let (qc, kc, vc) =
            CpuBackend::split_qkv_reshape_heads(&qkv, seq, n_heads, d_head);
        let qg = ScryGpuBackend::to_vec(&q);
        let kg = ScryGpuBackend::to_vec(&k);
        let vg = ScryGpuBackend::to_vec(&v);
        // Pure permutation kernel — values must match exactly.
        for (i, (g, c)) in qg.iter().zip(qc.iter()).enumerate() {
            assert!((g - c).abs() < 1e-6, "q idx={i}: gpu={g} cpu={c}");
        }
        for (i, (g, c)) in kg.iter().zip(kc.iter()).enumerate() {
            assert!((g - c).abs() < 1e-6, "k idx={i}: gpu={g} cpu={c}");
        }
        for (i, (g, c)) in vg.iter().zip(vc.iter()).enumerate() {
            assert!((g - c).abs() < 1e-6, "v idx={i}: gpu={g} cpu={c}");
        }
    }

    #[test]
    fn gpu_reshape_from_heads_matches_cpu_within_tolerance() {
        // Same ViT attention shape, batch=1.
        let seq = 197;
        let n_heads = 12;
        let d_head = 64;
        let d_model = n_heads * d_head;
        let total = n_heads * seq * d_head;
        let input: Vec<f32> = (0..total).map(|i| ((i % 173) as f32 - 80.0) * 0.013).collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu_in = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_reshape_from_heads test: {e}");
                return;
            }
        };
        let out = ScryGpuBackend::reshape_from_heads(&gpu_in, 1, seq, n_heads, d_head);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(out.is_gpu(), "reshape_from_heads should stay Gpu on CUDA");

        let cpu_out = CpuBackend::reshape_from_heads(&input, 1, seq, n_heads, d_head);
        let g = ScryGpuBackend::to_vec(&out);
        assert_eq!(g.len(), seq * d_model);
        for (i, (gv, cv)) in g.iter().zip(cpu_out.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_reshape_for_heads_matches_cpu_within_tolerance() {
        // Forward direction: [seq, n_heads*d_head] -> [n_heads, seq, d_head].
        // Same ViT attention shape, batch=1.
        let seq = 197;
        let n_heads = 12;
        let d_head = 64;
        let d_model = n_heads * d_head;
        let total = seq * d_model;
        let input: Vec<f32> = (0..total).map(|i| ((i % 173) as f32 - 80.0) * 0.013).collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu_in = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_reshape_for_heads test: {e}");
                return;
            }
        };
        let out = ScryGpuBackend::reshape_for_heads(&gpu_in, 1, seq, n_heads, d_head);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(out.is_gpu(), "reshape_for_heads should stay Gpu on CUDA");

        let cpu_out = CpuBackend::reshape_for_heads(&input, 1, seq, n_heads, d_head);
        let g = ScryGpuBackend::to_vec(&out);
        assert_eq!(g.len(), n_heads * seq * d_head);
        for (i, (gv, cv)) in g.iter().zip(cpu_out.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_transpose_2d_matches_cpu() {
        // SD spatial-transformer reshape size at stage 0: [320, 4096].
        // Use a smaller shape to keep the test quick but still well above
        // GPU_ELEMENTWISE_MIN so the kernel path engages.
        let rows = 320;
        let cols = 1024;
        let total = rows * cols;
        let input: Vec<f32> = (0..total).map(|i| ((i % 211) as f32 - 100.0) * 0.011).collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu_in = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_transpose_2d_matches_cpu: {e}");
                return;
            }
        };
        let out = ScryGpuBackend::transpose_2d(&gpu_in, rows, cols);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(out.is_gpu(), "transpose_2d should stay Gpu on CUDA");

        let g = ScryGpuBackend::to_vec(&out);
        // Pure permutation — values must match exactly.
        for r in 0..rows {
            for c in 0..cols {
                let src = input[r * cols + c];
                let dst = g[c * rows + r];
                assert!(
                    (src - dst).abs() < 1e-6,
                    "transpose mismatch at (r={r}, c={c}): src={src} dst={dst}"
                );
            }
        }
    }

    #[test]
    fn gpu_matmul_bias_matches_cpu_within_tolerance() {
        // Dense linear layer shape — matches an SD FF inner matmul:
        // [N=2048, K=320] · [K=320, M=320] + [M=320]. Above
        // GPU_ADD_ROW_BIAS_MIN so the kernel path engages.
        let m = 2048;
        let k = 320;
        let n = 320;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 211) as f32 - 100.0) * 0.011).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 173) as f32 - 80.0) * 0.013).collect();
        let bias: Vec<f32> = (0..n).map(|i| ((i % 41) as f32 - 20.0) * 0.07).collect();
        let a_storage = ScryGpuStorage::Cpu(a.clone());
        let b_storage = ScryGpuStorage::Cpu(b.clone());
        let bias_storage = ScryGpuStorage::Cpu(bias.clone());
        let a_gpu = match ScryGpuBackend::to_gpu(&a_storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_matmul_bias test: {e}");
                return;
            }
        };
        let b_gpu = ScryGpuBackend::to_gpu(&b_storage).unwrap();
        let bias_gpu = ScryGpuBackend::to_gpu(&bias_storage).unwrap();

        let out = ScryGpuBackend::matmul_bias(&a_gpu, &b_gpu, &bias_gpu, m, k, n, false, false);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            out.is_gpu(),
            "matmul_bias should stay Gpu on CUDA, got {out:?}"
        );

        let cpu_out = CpuBackend::matmul_bias(&a, &b, &bias, m, k, n, false, false);
        let g = ScryGpuBackend::to_vec(&out);
        assert_eq!(g.len(), cpu_out.len());
        // f32 matmul + bias add — fp32 round-off envelope.
        let max_diff = g
            .iter()
            .zip(cpu_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 5e-4,
            "matmul_bias max abs diff {max_diff:.3e} exceeds 5e-4"
        );
    }

    #[test]
    fn gpu_concat_rows_matches_cpu() {
        // Two row-major matrices with the same number of cols, well above
        // GPU_ELEMENTWISE_MIN so the kernel path engages.
        let cols = 256;
        let a_rows = 192;
        let b_rows = 64;
        let a: Vec<f32> = (0..a_rows * cols)
            .map(|i| ((i % 211) as f32 - 100.0) * 0.011)
            .collect();
        let b: Vec<f32> = (0..b_rows * cols)
            .map(|i| ((i % 173) as f32 - 80.0) * 0.013)
            .collect();
        let a_storage = ScryGpuStorage::Cpu(a.clone());
        let b_storage = ScryGpuStorage::Cpu(b.clone());
        let a_gpu = match ScryGpuBackend::to_gpu(&a_storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_concat_rows test: {e}");
                return;
            }
        };
        let b_gpu = ScryGpuBackend::to_gpu(&b_storage).unwrap();
        let out = ScryGpuBackend::concat_rows(&a_gpu, &b_gpu, a_rows, b_rows, cols);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(out.is_gpu(), "concat_rows should stay Gpu on CUDA");

        let cpu_out = CpuBackend::concat_rows(&a, &b, a_rows, b_rows, cols);
        let g = ScryGpuBackend::to_vec(&out);
        assert_eq!(g.len(), cpu_out.len());
        // Pure memcpy — exact match expected.
        for (i, (gv, cv)) in g.iter().zip(cpu_out.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-9, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn small_transpose_2d_falls_back_to_cpu() {
        let input: Vec<f32> = (0..16 * 16).map(|i| i as f32 * 0.01).collect();
        let storage = ScryGpuStorage::Cpu(input);
        let out = ScryGpuBackend::transpose_2d(&storage, 16, 16);
        assert!(!out.is_gpu(), "small transpose_2d should fall back to Cpu");
    }

    #[test]
    fn gpu_reshape_round_trip_is_identity() {
        // reshape_for_heads then reshape_from_heads should restore the
        // original tensor byte-for-byte. Catches index-decode bugs in
        // either kernel.
        let seq = 64;
        let n_heads = 8;
        let d_head = 80;
        let d_model = n_heads * d_head;
        let total = seq * d_model;
        let input: Vec<f32> = (0..total).map(|i| ((i % 251) as f32 - 120.0) * 0.017).collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu_in = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_reshape_round_trip test: {e}");
                return;
            }
        };
        let permuted = ScryGpuBackend::reshape_for_heads(&gpu_in, 1, seq, n_heads, d_head);
        let restored = ScryGpuBackend::reshape_from_heads(&permuted, 1, seq, n_heads, d_head);
        let r = ScryGpuBackend::to_vec(&restored);
        assert_eq!(r.len(), input.len());
        for (i, (rv, iv)) in r.iter().zip(input.iter()).enumerate() {
            assert!((rv - iv).abs() < 1e-6, "round trip idx={i}: r={rv} in={iv}");
        }
    }

    #[test]
    fn gpu_matmul_strided_batched_matches_cpu_within_tolerance() {
        // ViT attention shape Q@Kᵀ: batch=12 heads, m=k=197, n=64 (or
        // permutations thereof). Use a smaller batch+shape to keep the test
        // quick while clearing the strided-batched path.
        let batch = 4;
        let m = 32;
        let k = 16;
        let n = 24;
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| ((i % 89) as f32 - 40.0) * 0.013)
            .collect();
        let b: Vec<f32> = (0..batch * k * n)
            .map(|i| ((i % 67) as f32 - 30.0) * 0.011)
            .collect();
        let a_storage = ScryGpuStorage::Cpu(a.clone());
        let b_storage = ScryGpuStorage::Cpu(b.clone());
        let a_gpu = match ScryGpuBackend::to_gpu(&a_storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_matmul_strided_batched test: {e}");
                return;
            }
        };
        let b_gpu = ScryGpuBackend::to_gpu(&b_storage).expect("upload b");

        let gpu_out =
            ScryGpuBackend::matmul_strided_batched(&a_gpu, &b_gpu, batch, m, k, n, false, false);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(gpu_out.is_gpu(), "strided batched output should stay Gpu");

        let cpu_out =
            CpuBackend::matmul_strided_batched(&a, &b, batch, m, k, n, false, false);
        let g = ScryGpuBackend::to_vec(&gpu_out);
        for (i, (gv, cv)) in g.iter().zip(cpu_out.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-3, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_add_elementwise_matches_cpu_within_tolerance() {
        // Same-shape add — ResNet residual case. Use a 3D shape sized like a
        // mid-network feature map (above GPU_ADD_ELEMENTWISE_MIN so the
        // kernel engages) to exercise the actual call site.
        let shape = Shape::new(&[256, 14, 14]);
        let total = shape.numel();
        let a: Vec<f32> = (0..total).map(|i| ((i % 113) as f32) * 0.03 - 1.5).collect();
        let b: Vec<f32> = (0..total).map(|i| ((i % 71) as f32) * 0.07 + 0.5).collect();

        let gpu_a = match ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(a.clone())) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_add_elementwise_matches_cpu_within_tolerance: {e}");
                return;
            }
        };
        let gpu_b = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(b.clone()))
            .expect("upload b after upload a succeeded");

        let gpu_out = ScryGpuBackend::add(&gpu_a, &gpu_b, &shape, &shape, &shape);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "same-shape add over Gpu inputs should stay Gpu on CUDA, got {gpu_out:?}"
        );

        let g = ScryGpuBackend::to_vec(&gpu_out);
        let c = CpuBackend::add(&a, &b, &shape, &shape, &shape);
        assert_eq!(g.len(), c.len());
        for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_add_elementwise_below_threshold_falls_back_to_cpu() {
        // Below GPU_ADD_ELEMENTWISE_MIN — even with both inputs Gpu-resident,
        // the helper should return None and the path should fall through to
        // the CPU add.
        let shape = Shape::new(&[16, 16]);
        let total = shape.numel();
        assert!(total < GPU_ADD_ELEMENTWISE_MIN);
        let a: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..total).map(|i| i as f32 * 0.2).collect();

        let gpu_a = match ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(a.clone())) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_add_elementwise_below_threshold_falls_back_to_cpu: {e}");
                return;
            }
        };
        let gpu_b = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(b.clone()))
            .expect("upload b after upload a succeeded");
        let out = ScryGpuBackend::add(&gpu_a, &gpu_b, &shape, &shape, &shape);
        assert!(!out.is_gpu(), "small same-shape add should fall back to Cpu");
    }

    #[test]
    fn gpu_mul_elementwise_matches_cpu_within_tolerance() {
        // SD UNet's GeGLU gate shape — `[H*W, d_ff]` at a mid-network stage.
        // Sized above GPU_ADD_ELEMENTWISE_MIN so the kernel engages.
        let n = 1024 * 16;
        let a: Vec<f32> = (0..n).map(|i| ((i % 113) as f32) * 0.03 - 1.5).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i % 71) as f32) * 0.07 + 0.5).collect();

        let gpu_a = match ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(a.clone())) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_mul_elementwise_matches_cpu_within_tolerance: {e}");
                return;
            }
        };
        let gpu_b = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(b.clone()))
            .expect("upload b after upload a succeeded");

        let gpu_out = ScryGpuBackend::mul_elementwise(&gpu_a, &gpu_b);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "mul_elementwise over Gpu inputs should stay Gpu on CUDA, got {gpu_out:?}"
        );

        let g = ScryGpuBackend::to_vec(&gpu_out);
        let c = CpuBackend::mul_elementwise(&a, &b);
        assert_eq!(g.len(), c.len());
        for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_embedding_matches_cpu_and_stays_gpu() {
        // SD 1.5 CLIP token-table shape: vocab=49408, dim=768. Sized down
        // here so the test is fast but well above GPU_EMBEDDING_MIN_OUTPUT_ELEMENTS
        // — 128 indices × 768 dim = 98_304 outputs.
        let vocab = 1024;
        let dim = 768;
        let n_indices = 128;
        let weight: Vec<f32> = (0..vocab * dim)
            .map(|i| ((i % 257) as f32) * 0.013 - 1.5)
            .collect();
        let indices: Vec<usize> = (0..n_indices).map(|i| (i * 7) % vocab).collect();

        let gpu_w = match ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(weight.clone())) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_embedding_matches_cpu_and_stays_gpu: {e}");
                return;
            }
        };
        let gpu_out = ScryGpuBackend::embedding(&gpu_w, &indices, vocab, dim);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "embedding over Gpu weight should stay Gpu on CUDA, got {gpu_out:?}"
        );

        let g = ScryGpuBackend::to_vec(&gpu_out);
        let c = CpuBackend::embedding(&weight, &indices, vocab, dim);
        assert_eq!(g.len(), c.len());
        for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_embedding_below_threshold_falls_back_to_cpu() {
        // Below GPU_EMBEDDING_MIN_OUTPUT_ELEMENTS — host loop wins.
        let vocab = 32;
        let dim = 16;
        let n_indices = 8;
        assert!(n_indices * dim < GPU_EMBEDDING_MIN_OUTPUT_ELEMENTS);
        let weight: Vec<f32> = (0..vocab * dim).map(|i| i as f32 * 0.01).collect();
        let indices: Vec<usize> = vec![0, 5, 31, 10, 2, 17, 0, 8];

        let gpu_w = match ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(weight)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_embedding_below_threshold_falls_back_to_cpu: {e}");
                return;
            }
        };
        let out = ScryGpuBackend::embedding(&gpu_w, &indices, vocab, dim);
        assert!(!out.is_gpu(), "small embedding should fall back to Cpu");
    }

    #[test]
    fn gpu_mul_elementwise_below_threshold_falls_back_to_cpu() {
        // Below GPU_ADD_ELEMENTWISE_MIN — same threshold as add_elementwise,
        // since the kernel cost profile is identical.
        let n = 256;
        assert!(n < GPU_ADD_ELEMENTWISE_MIN);
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();

        let gpu_a = match ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(a.clone())) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_mul_elementwise_below_threshold_falls_back_to_cpu: {e}");
                return;
            }
        };
        let gpu_b = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(b.clone()))
            .expect("upload b after upload a succeeded");
        let out = ScryGpuBackend::mul_elementwise(&gpu_a, &gpu_b);
        assert!(!out.is_gpu(), "small mul_elementwise should fall back to Cpu");
    }

    #[test]
    fn gpu_add_row_bias_matches_cpu_within_tolerance() {
        // [rows, cols] + [rows, 1] broadcast — Conv2d's bias add. Sized above
        // GPU_ADD_ROW_BIAS_MIN so the kernel engages.
        let rows = 64;
        let cols = 256;
        let total = rows * cols;
        let a: Vec<f32> = (0..total).map(|i| ((i % 97) as f32) * 0.05 - 2.0).collect();
        let bias: Vec<f32> = (0..rows).map(|r| (r as f32) * 0.1 - 1.0).collect();

        let a_s = ScryGpuStorage::Cpu(a.clone());
        let b_s = ScryGpuStorage::Cpu(bias.clone());
        let gpu_a = match ScryGpuBackend::to_gpu(&a_s) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_add_row_bias_matches_cpu_within_tolerance: {e}");
                return;
            }
        };

        let a_shape = Shape::new(&[rows, cols]);
        let b_shape = Shape::new(&[rows, 1]);
        let gpu_out = ScryGpuBackend::add(&gpu_a, &b_s, &a_shape, &b_shape, &a_shape);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "row-bias add should stay Gpu on CUDA, got {gpu_out:?}"
        );

        let g = ScryGpuBackend::to_vec(&gpu_out);
        let c = CpuBackend::add(&a, &bias, &a_shape, &b_shape, &a_shape);
        assert_eq!(g.len(), c.len());
        for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_add_row_broadcast_matches_cpu_within_tolerance() {
        // [seq, cols] + [1, cols] — every transformer attention block adds
        // a [1, 3*d_model] qkv-bias and a [1, d_model] proj-bias in this
        // shape. CLIP-L sized: seq=77, cols=2304.
        let rows = 77;
        let cols = 2304;
        let total = rows * cols;
        let a: Vec<f32> = (0..total).map(|i| ((i % 113) as f32) * 0.013 - 1.5).collect();
        let bias: Vec<f32> = (0..cols).map(|c| (c as f32) * 0.0007 - 0.8).collect();

        let a_s = ScryGpuStorage::Cpu(a.clone());
        let b_s = ScryGpuStorage::Cpu(bias.clone());
        let gpu_a = match ScryGpuBackend::to_gpu(&a_s) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_add_row_broadcast_matches_cpu_within_tolerance: {e}");
                return;
            }
        };
        let gpu_b = ScryGpuBackend::to_gpu(&b_s).expect("upload bias after upload a");

        let a_shape = Shape::new(&[rows, cols]);
        let b_shape = Shape::new(&[1, cols]);
        let gpu_out = ScryGpuBackend::add(&gpu_a, &gpu_b, &a_shape, &b_shape, &a_shape);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "row-broadcast add should stay Gpu on CUDA, got {gpu_out:?}"
        );

        let g = ScryGpuBackend::to_vec(&gpu_out);
        let c = CpuBackend::add(&a, &bias, &a_shape, &b_shape, &a_shape);
        assert_eq!(g.len(), c.len());
        for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "idx={i}: gpu={gv} cpu={cv}");
        }

        // Swapped operands ([1, cols] + [rows, cols]) — same fast path.
        let gpu_swapped = ScryGpuBackend::add(&gpu_b, &gpu_a, &b_shape, &a_shape, &a_shape);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_swapped.is_gpu(),
            "swapped row-broadcast add should stay Gpu on CUDA"
        );
        let g_sw = ScryGpuBackend::to_vec(&gpu_swapped);
        for (i, (gv, cv)) in g_sw.iter().zip(c.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "swapped idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_gather_columns_matches_cpu_and_stays_gpu() {
        // SD CLIP per-head Q gather: [seq=77, total_cols=2304] → [77, 64]
        // starting at col 0. Sized above GPU_COLUMN_GS_MIN_OUTPUT_ELEMENTS
        // (4_928 outputs >> 2_048 floor).
        let rows = 77;
        let total_cols = 2304;
        let col_count = 64;
        let col_start = 768;
        let total_in = rows * total_cols;
        let input: Vec<f32> = (0..total_in)
            .map(|i| ((i % 191) as f32) * 0.011 - 1.0)
            .collect();

        let gpu_in = match ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(input.clone())) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_gather_columns_matches_cpu_and_stays_gpu: {e}");
                return;
            }
        };
        let gpu_out =
            ScryGpuBackend::gather_columns(&gpu_in, rows, total_cols, col_start, col_count);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "gather_columns over Gpu input should stay Gpu on CUDA, got {gpu_out:?}"
        );

        let g = ScryGpuBackend::to_vec(&gpu_out);
        let c = CpuBackend::gather_columns(&input, rows, total_cols, col_start, col_count);
        assert_eq!(g.len(), c.len());
        for (i, (gv, cv)) in g.iter().zip(c.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_scatter_columns_promotes_cpu_dst_and_accumulates() {
        // Two non-overlapping head writes into a `[seq, d_model]` accumulator
        // — mirrors the per-head attention concat. First call promotes a
        // Cpu zeros() dst to Gpu; second call hits the Gpu fast path.
        let rows = 77;
        let total_cols = 768;
        let col_count = 64;

        let mut dst_cpu = vec![0.0f32; rows * total_cols];
        let src1: Vec<f32> = (0..rows * col_count)
            .map(|i| ((i % 73) as f32) * 0.07 - 0.5)
            .collect();
        let src2: Vec<f32> = (0..rows * col_count)
            .map(|i| ((i * 13 + 5) as f32) * 0.013 + 0.2)
            .collect();
        let mut dst = match ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(vec![0.0f32; rows * total_cols])) {
            // Use a Cpu zeros to mirror what the trait does, then forget the
            // Gpu attempt — we want the override to do the promotion.
            Ok(_) => ScryGpuStorage::Cpu(vec![0.0f32; rows * total_cols]),
            Err(e) => {
                eprintln!("skipping gpu_scatter_columns_promotes_cpu_dst_and_accumulates: {e}");
                return;
            }
        };
        let gpu_src1 = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(src1.clone()))
            .expect("upload src1");
        let gpu_src2 = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(src2.clone()))
            .expect("upload src2");

        ScryGpuBackend::scatter_columns(&mut dst, &gpu_src1, rows, total_cols, 0, col_count);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(dst.is_gpu(), "scatter promoted Cpu dst to Gpu");
        ScryGpuBackend::scatter_columns(&mut dst, &gpu_src2, rows, total_cols, 64, col_count);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(dst.is_gpu(), "second scatter stays on Gpu");

        // Build CPU reference.
        CpuBackend::scatter_columns(&mut dst_cpu, &src1, rows, total_cols, 0, col_count);
        CpuBackend::scatter_columns(&mut dst_cpu, &src2, rows, total_cols, 64, col_count);

        let g = ScryGpuBackend::to_vec(&dst);
        for (i, (gv, cv)) in g.iter().zip(dst_cpu.iter()).enumerate() {
            assert!((gv - cv).abs() < 1e-6, "idx={i}: gpu={gv} cpu={cv}");
        }
    }

    #[test]
    fn gpu_apply_causal_mask_and_scale_matches_cpu() {
        // CLIP causal-mask shape: seq=77.
        let seq = 77;
        let total = seq * seq;
        let scale = 1.0_f32 / (64.0_f32).sqrt();
        let mask_value = f32::NEG_INFINITY;
        let scores: Vec<f32> = (0..total).map(|i| ((i % 89) as f32) * 0.04 - 1.5).collect();

        let mut gpu_scores =
            match ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(scores.clone())) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping gpu_apply_causal_mask_and_scale_matches_cpu: {e}");
                    return;
                }
            };
        ScryGpuBackend::apply_causal_mask_and_scale(&mut gpu_scores, seq, scale, mask_value);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(gpu_scores.is_gpu(), "mask+scale over Gpu input stays Gpu");

        let mut cpu_scores = scores;
        CpuBackend::apply_causal_mask_and_scale(&mut cpu_scores, seq, scale, mask_value);

        let g = ScryGpuBackend::to_vec(&gpu_scores);
        for (i, (gv, cv)) in g.iter().zip(cpu_scores.iter()).enumerate() {
            // -INF compares with itself fine; finite cells should match.
            if cv.is_finite() {
                assert!(
                    (gv - cv).abs() < 1e-6,
                    "idx={i}: gpu={gv} cpu={cv}"
                );
            } else {
                assert!(gv.is_infinite() && gv.is_sign_negative(), "idx={i}: expected -INF got {gv}");
            }
        }
    }

    #[test]
    fn gpu_max_pool_matches_cpu_within_tolerance() {
        // Two cases: ResNet stem (64×112×112 → 64×56×56 with k=3, s=2, p=1)
        // and a no-padding case (32×64×64 → 32×32×32 with k=2, s=2, p=0).
        // Both produce >> GPU_POOL_MIN_OUTPUT_ELEMENTS so the kernel engages.
        let cases: [(usize, usize, usize, usize, usize, usize); 2] = [
            (64, 112, 112, 3, 2, 1),
            (32, 64, 64, 2, 2, 0),
        ];
        for (c, h, w, k, stride, padding) in cases {
            let total_in = c * h * w;
            // Use a deterministic non-monotonic pattern so max isn't trivially
            // the corner element.
            let input: Vec<f32> = (0..total_in)
                .map(|i| ((i * 1664525 + 1013904223) as u32 as f32 / u32::MAX as f32) * 4.0 - 2.0)
                .collect();
            let storage = ScryGpuStorage::Cpu(input.clone());
            let gpu = match ScryGpuBackend::to_gpu(&storage) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping gpu_max_pool_matches_cpu_within_tolerance: {e}");
                    return;
                }
            };
            let gpu_out = ScryGpuBackend::max_pool_2d(&gpu, c, h, w, k, stride, padding);
            #[cfg(feature = "scry-gpu-cuda")]
            assert!(
                gpu_out.is_gpu(),
                "case ({c},{h},{w},{k},{stride},{padding}): max_pool over Gpu input should stay Gpu, got {gpu_out:?}"
            );

            let g = ScryGpuBackend::to_vec(&gpu_out);
            let cref = CpuBackend::max_pool_2d(&input, c, h, w, k, stride, padding);
            assert_eq!(g.len(), cref.len());
            for (i, (gv, cv)) in g.iter().zip(cref.iter()).enumerate() {
                // Pure max: bit-equality expected.
                assert!(
                    (gv - cv).abs() < 1e-6,
                    "case ({c},{h},{w},{k},{stride},{padding}) idx={i}: gpu={gv} cpu={cv}"
                );
            }
        }
    }

    #[test]
    fn gpu_adaptive_avg_pool_matches_cpu_within_tolerance() {
        // ResNet's spatial-7×7 → global-1×1 produces only `channels` outputs
        // (e.g. 2048), so the global case below threshold is correct
        // behaviour — exercise a downsample case instead, sized above
        // GPU_POOL_MIN_OUTPUT_ELEMENTS. 64 channels × 28×28 → 64 × 14×14
        // = 12,544 outputs is below threshold; 256 × 28×28 → 256 × 14×14
        // = 50,176 clears it.
        let (c, h, w, ho, wo) = (256usize, 28usize, 28usize, 14usize, 14usize);
        let total_in = c * h * w;
        let input: Vec<f32> = (0..total_in)
            .map(|i| ((i % 113) as f32) * 0.07 - 3.5)
            .collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let gpu = match ScryGpuBackend::to_gpu(&storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping gpu_adaptive_avg_pool_matches_cpu_within_tolerance: {e}");
                return;
            }
        };
        let gpu_out = ScryGpuBackend::adaptive_avg_pool_2d(&gpu, c, h, w, ho, wo);
        #[cfg(feature = "scry-gpu-cuda")]
        assert!(
            gpu_out.is_gpu(),
            "adaptive_avg_pool over Gpu input should stay Gpu, got {gpu_out:?}"
        );

        let g = ScryGpuBackend::to_vec(&gpu_out);
        let cref = CpuBackend::adaptive_avg_pool_2d(&input, c, h, w, ho, wo);
        assert_eq!(g.len(), cref.len());
        for (i, (gv, cv)) in g.iter().zip(cref.iter()).enumerate() {
            // Sum/divide of a few cells per output — f32 accumulation order
            // differs slightly. 1e-5 absolute tolerance is plenty.
            assert!(
                (gv - cv).abs() < 1e-5,
                "idx={i}: gpu={gv} cpu={cv}"
            );
        }
    }

    #[test]
    fn gpu_upsample_2d_nearest_matches_cpu_within_tolerance() {
        // SD UNet middle resolution: 320 channels × 16×16 → 320 × 32×32 with
        // scale=2 produces 320*32*32 = 327,680 outputs, well above the
        // GPU_UPSAMPLE_MIN_OUTPUT_ELEMENTS threshold. Also exercise scale=4
        // (less common in SD but worth checking the integer-divide path).
        for (channels, h_in, w_in, scale) in [(320usize, 16usize, 16usize, 2usize), (32, 16, 16, 4)]
        {
            let total_in = channels * h_in * w_in;
            let input: Vec<f32> = (0..total_in)
                .map(|i| ((i % 113) as f32) * 0.07 - 3.5)
                .collect();
            let storage = ScryGpuStorage::Cpu(input.clone());
            let gpu = match ScryGpuBackend::to_gpu(&storage) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping gpu_upsample_2d_nearest_matches_cpu_within_tolerance: {e}");
                    return;
                }
            };
            let gpu_out = ScryGpuBackend::upsample_2d_nearest(&gpu, channels, h_in, w_in, scale);
            #[cfg(feature = "scry-gpu-cuda")]
            assert!(
                gpu_out.is_gpu(),
                "upsample over Gpu input should stay Gpu, got {gpu_out:?}"
            );
            let g = ScryGpuBackend::to_vec(&gpu_out);
            // Reference: trait default impl semantics — `out[c, oh, ow] =
            // in[c, oh/scale, ow/scale]`. Pure index permutation, no
            // arithmetic, so equality is exact.
            let h_out = h_in * scale;
            let w_out = w_in * scale;
            assert_eq!(g.len(), channels * h_out * w_out);
            for c in 0..channels {
                let in_plane = c * h_in * w_in;
                let out_plane = c * h_out * w_out;
                for oh in 0..h_out {
                    let ih = oh / scale;
                    for ow in 0..w_out {
                        let iw = ow / scale;
                        let gv = g[out_plane + oh * w_out + ow];
                        let rv = input[in_plane + ih * w_in + iw];
                        assert!(
                            (gv - rv).abs() == 0.0,
                            "channels={channels} scale={scale} c={c} oh={oh} ow={ow}: gpu={gv} ref={rv}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn small_upsample_falls_back_to_cpu() {
        // Below GPU_UPSAMPLE_MIN_OUTPUT_ELEMENTS: 4 channels × 8×8 with
        // scale=2 produces 4 * 16 * 16 = 1024 outputs. Falls back to CPU.
        let channels = 4;
        let h_in = 8;
        let w_in = 8;
        let scale = 2;
        let input: Vec<f32> = (0..channels * h_in * w_in)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let storage = ScryGpuStorage::Cpu(input.clone());
        let out = ScryGpuBackend::upsample_2d_nearest(&storage, channels, h_in, w_in, scale);
        assert!(!out.is_gpu(), "small upsample should fall back to Cpu");
        let v = ScryGpuBackend::to_vec(&out);
        let h_out = h_in * scale;
        let w_out = w_in * scale;
        assert_eq!(v.len(), channels * h_out * w_out);
        // Spot-check a couple of indices on the CPU fallback path.
        for c in 0..channels {
            let out_plane = c * h_out * w_out;
            let in_plane = c * h_in * w_in;
            // (oh=0, ow=0) → (ih=0, iw=0)
            assert_eq!(v[out_plane], input[in_plane]);
            // (oh=3, ow=5) → (ih=1, iw=2)
            assert_eq!(
                v[out_plane + 3 * w_out + 5],
                input[in_plane + 1 * w_in + 2]
            );
        }
    }

    #[test]
    fn small_pool_falls_back_to_cpu() {
        // Below GPU_POOL_MIN_OUTPUT_ELEMENTS: max-pool 8×16×16 → 8×8×8 = 512
        // outputs; adaptive-avg-pool 4×16×16 → 4×1×1 = 4 outputs. Both fall
        // back to CPU.
        let max_input: Vec<f32> = (0..8 * 16 * 16).map(|i| (i as f32) * 0.01).collect();
        let max_s = ScryGpuStorage::Cpu(max_input);
        let max_out = ScryGpuBackend::max_pool_2d(&max_s, 8, 16, 16, 2, 2, 0);
        assert!(!max_out.is_gpu(), "small max_pool should fall back to Cpu");

        let avg_input: Vec<f32> = (0..4 * 16 * 16).map(|i| (i as f32) * 0.01).collect();
        let avg_s = ScryGpuStorage::Cpu(avg_input);
        let avg_out = ScryGpuBackend::adaptive_avg_pool_2d(&avg_s, 4, 16, 16, 1, 1);
        assert!(!avg_out.is_gpu(), "small adaptive_avg_pool should fall back to Cpu");
    }

    #[test]
    fn gpu_group_norm_matches_cpu_within_tolerance() {
        // SD UNet uses num_groups=32 everywhere. Three shapes: small spatial
        // (matches a 64-channel ResBlock at high resolution), mid (320 channels
        // at 32×32), large (1280 channels at 8×8). Each clears the
        // GPU_GROUP_NORM_MIN_BLOCKS = 32 threshold (num_groups * batch ≥ 32).
        // Per-block reduction extent (cpg * spatial) varies across the cases so
        // we exercise both the in-block-bounded (cpg*spatial ≤ 256) and
        // strided-loop regimes.
        for &(batch, channels, spatial, num_groups) in &[
            (1usize, 64usize, 64usize, 32usize),    // cpg=2, gsize=128
            (1, 320, 32 * 32, 32),                   // cpg=10, gsize=10240
            (1, 1280, 8 * 8, 32),                    // cpg=40, gsize=2560
        ] {
            let total = batch * channels * spatial;
            let input: Vec<f32> = (0..total).map(|i| ((i % 113) as f32) * 0.05 - 2.5).collect();
            let weight: Vec<f32> = (0..channels).map(|c| 0.5 + (c as f32) * 0.01).collect();
            let bias: Vec<f32> = (0..channels).map(|c| (c as f32) * 0.005 - 0.25).collect();
            let eps = 1e-5_f32;

            let in_s = ScryGpuStorage::Cpu(input.clone());
            let w_s = ScryGpuStorage::Cpu(weight.clone());
            let b_s = ScryGpuStorage::Cpu(bias.clone());

            let gpu_in = match ScryGpuBackend::to_gpu(&in_s) {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("skipping gpu_group_norm_matches_cpu_within_tolerance: {e}");
                    return;
                }
            };

            let out = ScryGpuBackend::group_norm(
                &gpu_in, &w_s, &b_s, num_groups, channels, spatial, eps,
            );
            #[cfg(feature = "scry-gpu-cuda")]
            assert!(
                out.is_gpu(),
                "group_norm output should stay Gpu on CUDA, got {out:?}"
            );

            let g_out = ScryGpuBackend::to_vec(&out);
            let c_out = <CpuBackend as MathBackend>::group_norm(
                &input, &weight, &bias, num_groups, channels, spatial, eps,
            );
            assert_eq!(g_out.len(), c_out.len());

            // GPU reduces in f32 with a 256-thread tree; CPU default impl
            // accumulates in f64 to keep the reference clean. For SD-class
            // group sizes (`gsize` up to ~10K elements), absolute tolerance
            // 1e-3 holds — this is the same envelope batchnorm uses for
            // affine outputs (the affine step lets row values grow with
            // `weight`/`bias`).
            for (i, (gv, cv)) in g_out.iter().zip(c_out.iter()).enumerate() {
                let tol = 1e-3 * cv.abs().max(1.0);
                assert!(
                    (gv - cv).abs() < tol,
                    "batch={batch} channels={channels} spatial={spatial} num_groups={num_groups} idx={i}: gpu={gv} cpu={cv} (tol={tol})"
                );
            }
        }
    }

    #[test]
    fn small_group_norm_falls_back_to_cpu() {
        // num_groups=8, batch=1 → 8 blocks < GPU_GROUP_NORM_MIN_BLOCKS=32.
        // Output should land in the Cpu variant.
        let (batch, channels, spatial, num_groups) = (1usize, 16usize, 8usize, 8usize);
        let total = batch * channels * spatial;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = vec![1.0; channels];
        let bias: Vec<f32> = vec![0.0; channels];
        let in_s = ScryGpuStorage::Cpu(input);
        let w_s = ScryGpuStorage::Cpu(weight);
        let b_s = ScryGpuStorage::Cpu(bias);
        let out = ScryGpuBackend::group_norm(
            &in_s, &w_s, &b_s, num_groups, channels, spatial, 1e-5,
        );
        assert!(!out.is_gpu(), "small group_norm should fall back to Cpu");
    }

    #[test]
    fn small_batchnorm_falls_back_to_cpu() {
        // Below GPU_BATCHNORM_MIN_CHANNELS: result should come back as Cpu variant.
        let channels = 8;
        let spatial = 64;
        let total = channels * spatial;
        let input: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = vec![1.0; channels];
        let bias: Vec<f32> = vec![0.0; channels];
        let mean: Vec<f32> = vec![0.0; channels];
        let var: Vec<f32> = vec![1.0; channels];
        let in_s = ScryGpuStorage::Cpu(input);
        let w_s = ScryGpuStorage::Cpu(weight);
        let b_s = ScryGpuStorage::Cpu(bias);
        let m_s = ScryGpuStorage::Cpu(mean);
        let v_s = ScryGpuStorage::Cpu(var);
        let out = ScryGpuBackend::batchnorm_2d_inference(
            &in_s, &w_s, &b_s, &m_s, &v_s, channels, spatial, 1e-5,
        );
        assert!(!out.is_gpu(), "small batchnorm should fall back to Cpu");
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

    #[cfg(feature = "scry-gpu-bf16")]
    #[test]
    fn gpu_matmul_strided_batched_bf16_matches_cpu_within_relaxed_tolerance() {
        // ViT-B/16 attention shape: 12 heads × seq=197 × d_head=64. Q@Kᵀ
        // is the worst-case for bf16 drift in attention (output magnitudes
        // grow with k=64, so absolute error scales with sqrt(k)). 5%
        // relative is the conservative envelope per the bf16 GemmEx test.
        let batch = 12;
        let m = 197;
        let k = 64;
        let n = 197;

        // Bound inputs to ~[-1, 1] so output magnitudes ≈ sqrt(k) ≈ 8;
        // outsized magnitudes would dilute the relative-error check.
        let mut state = 0xb_f16_5_71_d_d_u64;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((state >> 32) as i32 as f32) / (i32::MAX as f32)
        };
        let a: Vec<f32> = (0..batch * m * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|_| next()).collect();
        let a_storage = ScryGpuStorage::Cpu(a.clone());
        let b_storage = ScryGpuStorage::Cpu(b.clone());
        let a_gpu = match ScryGpuBackend::to_gpu(&a_storage) {
            Ok(g) => g,
            Err(e) => {
                eprintln!(
                    "skipping gpu_matmul_strided_batched_bf16_matches_cpu_within_relaxed_tolerance: {e}"
                );
                return;
            }
        };
        let b_gpu = ScryGpuBackend::to_gpu(&b_storage).expect("upload b");

        let bf16_out = match gpu_matmul_strided_batched_persistent_bf16(
            &a_gpu, &b_gpu, batch, m, k, n, false, false,
        ) {
            Some(o) => o,
            None => {
                eprintln!(
                    "skipping gpu_matmul_strided_batched_bf16_matches_cpu_within_relaxed_tolerance: bf16 path unavailable"
                );
                return;
            }
        };
        assert!(
            bf16_out.is_gpu(),
            "bf16 strided batched output should stay Gpu"
        );

        let cpu_out = CpuBackend::matmul_strided_batched(&a, &b, batch, m, k, n, false, false);
        let bf16_vec = ScryGpuBackend::to_vec(&bf16_out);
        assert_eq!(bf16_vec.len(), cpu_out.len());

        let mut max_rel_err: f32 = 0.0;
        for (i, (cpu_v, bf_v)) in cpu_out.iter().zip(bf16_vec.iter()).enumerate() {
            let mag = cpu_v.abs().max(1.0);
            let rel = (cpu_v - bf_v).abs() / mag;
            max_rel_err = max_rel_err.max(rel);
            assert!(
                rel < 5e-2,
                "bf16 strided batched drift at {i}: cpu={cpu_v} bf16={bf_v} rel={rel:.4}"
            );
        }
        eprintln!(
            "bf16 strided batched 12×197×64×197 max relative error: {max_rel_err:.5}"
        );
    }

    #[cfg(feature = "scry-gpu-bf16")]
    #[test]
    fn bf16_matmul_matches_cpu_within_relaxed_tolerance() {
        // 64×64 × 64×64 — clears the size threshold and exercises tensor
        // cores. bf16's 7-bit mantissa puts the realistic envelope at ~3% of
        // the output magnitude.
        let m = 64;
        let k = 64;
        let n = 64;

        // Bound inputs to [-1, 1] so output magnitudes stay sane (each
        // output sums k=64 products → typical magnitude ≈ sqrt(k) ≈ 8 by
        // random walk).
        let mut state = 0xb_f16_d_15_u64;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((state >> 32) as i32 as f32) / (i32::MAX as f32)
        };
        let a: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| next()).collect();

        let a_s = ScryGpuStorage::Cpu(a.clone());
        let b_s = ScryGpuStorage::Cpu(b.clone());

        let bf16_out = match ScryGpuBackend::matmul_force_bf16_for_bench(&a_s, &b_s, m, k, n) {
            Some(o) => o,
            None => {
                eprintln!("skipping bf16_matmul_matches_cpu_within_relaxed_tolerance: no GPU");
                return;
            }
        };

        let cpu_out = CpuBackend::matmul(&a, &b, m, k, n, false, false);
        let bf16_vec = ScryGpuBackend::to_vec(&bf16_out);
        assert_eq!(bf16_vec.len(), cpu_out.len());

        let mut max_rel_err: f32 = 0.0;
        for (i, (cpu_v, bf_v)) in cpu_out.iter().zip(bf16_vec.iter()).enumerate() {
            let mag = cpu_v.abs().max(1.0);
            let rel = (cpu_v - bf_v).abs() / mag;
            max_rel_err = max_rel_err.max(rel);
            // 5% relative is the conservative envelope for chained-rounding
            // bf16; in practice this test sees ~1% on uniform random inputs.
            assert!(
                rel < 5e-2,
                "bf16 matmul drift at {i}: cpu={cpu_v} bf16={bf_v} rel={rel:.4}"
            );
        }
        eprintln!("bf16 64×64×64 max relative error: {max_rel_err:.5}");
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
