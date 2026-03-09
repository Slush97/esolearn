use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

/// All custom CUDA kernels compiled once at init time.
pub struct KernelCache {
    pub gelu_fwd: CudaFunction,
    pub add_broadcast_2d: CudaFunction,
    pub softmax_fwd: CudaFunction,
    pub layernorm_fwd: CudaFunction,
    pub embedding_fwd: CudaFunction,
    pub scale_kernel: CudaFunction,
    pub mul_elementwise: CudaFunction,
    pub add_inplace_kernel: CudaFunction,
    pub gather_columns: CudaFunction,
    pub scatter_columns: CudaFunction,
    pub causal_mask_scale: CudaFunction,
    pub reduce_sum: CudaFunction,
    pub reshape_bsh_to_bnsh: CudaFunction,
    pub reshape_bnsh_to_bsh: CudaFunction,
    pub batched_causal_mask_scale: CudaFunction,

    // Llama-specific kernels
    pub rmsnorm_fwd: CudaFunction,
    pub rope_fwd: CudaFunction,
    pub rope_with_freqs_fwd: CudaFunction,
    pub rope_with_freqs_scaled_fwd: CudaFunction,
    pub swiglu_fwd: CudaFunction,
    pub repeat_kv_kernel: CudaFunction,
    pub gather_reshape_repeat_kv: CudaFunction,

    // BF16 kernels
    #[cfg(feature = "bf16")]
    pub embedding_bf16_fwd: CudaFunction,
    #[cfg(feature = "bf16")]
    pub f32_to_bf16: CudaFunction,
    #[cfg(feature = "bf16")]
    pub bf16_to_f32: CudaFunction,
    #[cfg(feature = "bf16")]
    pub rmsnorm_fwd_with_bf16: CudaFunction,
    #[cfg(feature = "bf16")]
    pub swiglu_fwd_with_bf16: CudaFunction,
}

impl KernelCache {
    pub fn compile(ctx: &Arc<cudarc::driver::CudaContext>) -> Self {
        let opts = CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SOURCE, opts)
            .expect("failed to compile CUDA kernels");
        let module: Arc<CudaModule> = ctx.load_module(ptx)
            .expect("failed to load CUDA module");

        #[cfg(feature = "bf16")]
        let bf16_module = {
            let bf16_opts = CompileOptions {
                use_fast_math: Some(true),
                ..Default::default()
            };
            let bf16_ptx = compile_ptx_with_opts(BF16_KERNEL_SOURCE, bf16_opts)
                .expect("failed to compile BF16 CUDA kernels");
            ctx.load_module(bf16_ptx)
                .expect("failed to load BF16 CUDA module")
        };

        Self {
            gelu_fwd: module.load_function("gelu_fwd").unwrap(),
            add_broadcast_2d: module.load_function("add_broadcast_2d").unwrap(),
            softmax_fwd: module.load_function("softmax_fwd").unwrap(),
            layernorm_fwd: module.load_function("layernorm_fwd").unwrap(),
            embedding_fwd: module.load_function("embedding_fwd").unwrap(),
            scale_kernel: module.load_function("scale_kernel").unwrap(),
            mul_elementwise: module.load_function("mul_elementwise").unwrap(),
            add_inplace_kernel: module.load_function("add_inplace_kernel").unwrap(),
            gather_columns: module.load_function("gather_columns").unwrap(),
            scatter_columns: module.load_function("scatter_columns").unwrap(),
            causal_mask_scale: module.load_function("causal_mask_scale").unwrap(),
            reduce_sum: module.load_function("reduce_sum").unwrap(),
            reshape_bsh_to_bnsh: module.load_function("reshape_bsh_to_bnsh").unwrap(),
            reshape_bnsh_to_bsh: module.load_function("reshape_bnsh_to_bsh").unwrap(),
            batched_causal_mask_scale: module.load_function("batched_causal_mask_scale").unwrap(),

            // Llama-specific
            rmsnorm_fwd: module.load_function("rmsnorm_fwd").unwrap(),
            rope_fwd: module.load_function("rope_fwd").unwrap(),
            rope_with_freqs_fwd: module.load_function("rope_with_freqs_fwd").unwrap(),
            rope_with_freqs_scaled_fwd: module.load_function("rope_with_freqs_scaled_fwd").unwrap(),
            swiglu_fwd: module.load_function("swiglu_fwd").unwrap(),
            repeat_kv_kernel: module.load_function("repeat_kv_kernel").unwrap(),
            gather_reshape_repeat_kv: module.load_function("gather_reshape_repeat_kv").unwrap(),

            #[cfg(feature = "bf16")]
            embedding_bf16_fwd: bf16_module.load_function("embedding_bf16_fwd").unwrap(),
            #[cfg(feature = "bf16")]
            f32_to_bf16: bf16_module.load_function("f32_to_bf16").unwrap(),
            #[cfg(feature = "bf16")]
            bf16_to_f32: bf16_module.load_function("bf16_to_f32").unwrap(),
            #[cfg(feature = "bf16")]
            rmsnorm_fwd_with_bf16: bf16_module.load_function("rmsnorm_fwd_with_bf16").unwrap(),
            #[cfg(feature = "bf16")]
            swiglu_fwd_with_bf16: bf16_module.load_function("swiglu_fwd_with_bf16").unwrap(),
        }
    }
}

const KERNEL_SOURCE: &str = r#"
// ============================================================
// Simple element-wise kernels
// ============================================================

extern "C" __global__ void gelu_fwd(float* out, const float* x, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
    out[i] = 0.5f * v * (1.0f + tanhf(inner));
}

extern "C" __global__ void scale_kernel(float* out, const float* a, float scalar, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * scalar;
}

extern "C" __global__ void mul_elementwise(float* out, const float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] * b[i];
}

extern "C" __global__ void add_inplace_kernel(float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

// ============================================================
// Broadcast add: a[rows, cols] + b[1, cols] = out[rows, cols]
// ============================================================

extern "C" __global__ void add_broadcast_2d(
    float* out, const float* a, const float* b,
    size_t rows, size_t cols
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    size_t col = idx % cols;
    out[idx] = a[idx] + b[col];
}

// ============================================================
// Softmax: row-wise with shared memory reduction
// ============================================================

extern "C" __global__ void softmax_fwd(float* out, const float* inp, size_t rows, size_t cols) {
    extern __shared__ float sdata[];

    size_t row = blockIdx.x;
    if (row >= rows) return;
    const float* row_in = inp + row * cols;
    float* row_out = out + row * cols;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_max = -1e30f;
    for (size_t j = tid; j < cols; j += stride) {
        float v = row_in[j];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (size_t j = tid; j < cols; j += stride) {
        float e = expf(row_in[j] - max_val);
        row_out[j] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];
    __syncthreads();

    for (size_t j = tid; j < cols; j += stride) {
        row_out[j] /= sum;
    }
}

// ============================================================
// LayerNorm: one block per row
// ============================================================

extern "C" __global__ void layernorm_fwd(
    float* out, float* mean_out, float* rstd_out,
    const float* inp, const float* gamma, const float* beta,
    size_t rows, size_t d, float eps
) {
    extern __shared__ float sdata[];

    size_t row = blockIdx.x;
    if (row >= rows) return;
    size_t off = row * d;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_sum = 0.0f;
    for (size_t j = tid; j < d; j += stride) {
        local_sum += inp[off + j];
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)d;
    __syncthreads();

    float local_var = 0.0f;
    for (size_t j = tid; j < d; j += stride) {
        float diff = inp[off + j] - mean;
        local_var += diff * diff;
    }
    sdata[tid] = local_var;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float var = sdata[0] / (float)d;
    float rstd = rsqrtf(var + eps);
    __syncthreads();

    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    for (size_t j = tid; j < d; j += stride) {
        float norm = (inp[off + j] - mean) * rstd;
        out[off + j] = norm * gamma[j] + beta[j];
    }
}

// ============================================================
// Embedding
// ============================================================

extern "C" __global__ void embedding_fwd(
    float* out, const float* weight, const unsigned int* indices,
    size_t n_indices, size_t dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_indices * dim) return;
    size_t token = idx / dim;
    size_t d = idx % dim;
    out[token * dim + d] = weight[indices[token] * dim + d];
}

// ============================================================
// Attention helpers
// ============================================================

extern "C" __global__ void gather_columns(
    float* out, const float* src,
    size_t rows, size_t total_cols, size_t col_start, size_t col_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * col_count) return;
    size_t r = idx / col_count;
    size_t c = idx % col_count;
    out[idx] = src[r * total_cols + col_start + c];
}

extern "C" __global__ void scatter_columns(
    float* dst, const float* src,
    size_t rows, size_t total_cols, size_t col_start, size_t col_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * col_count) return;
    size_t r = idx / col_count;
    size_t c = idx % col_count;
    dst[r * total_cols + col_start + c] += src[idx];
}

extern "C" __global__ void causal_mask_scale(
    float* scores, size_t seq_len, float scale, float mask_value
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * seq_len) return;
    size_t s = idx / seq_len;
    size_t t = idx % seq_len;
    if (t > s) {
        scores[idx] = mask_value;
    } else {
        scores[idx] *= scale;
    }
}

// ============================================================
// Reductions
// ============================================================

extern "C" __global__ void reduce_sum(float* out, const float* x, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_sum = 0.0f;
    for (size_t i = tid; i < n; i += stride) {
        local_sum += x[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[0] = sdata[0];
}

// ============================================================
// Reshape kernels for batched multi-head attention
// ============================================================

extern "C" __global__ void reshape_bsh_to_bnsh(
    float* dst, const float* src,
    size_t batch, size_t seq, size_t n_heads, size_t d_head
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n_heads * seq * d_head;
    if (idx >= total) return;
    size_t d = idx % d_head;
    size_t tmp = idx / d_head;
    size_t s = tmp % seq;
    size_t bh = tmp / seq;
    size_t h = bh % n_heads;
    size_t b = bh / n_heads;
    size_t D = n_heads * d_head;
    dst[idx] = src[(b * seq + s) * D + h * d_head + d];
}

extern "C" __global__ void reshape_bnsh_to_bsh(
    float* dst, const float* src,
    size_t batch, size_t seq, size_t n_heads, size_t d_head
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * n_heads * seq * d_head;
    if (idx >= total) return;
    size_t d = idx % d_head;
    size_t tmp = idx / d_head;
    size_t s = tmp % seq;
    size_t bh = tmp / seq;
    size_t h = bh % n_heads;
    size_t b = bh / n_heads;
    size_t D = n_heads * d_head;
    dst[(b * seq + s) * D + h * d_head + d] = src[idx];
}

extern "C" __global__ void batched_causal_mask_scale(
    float* scores, size_t num_matrices, size_t seq_len, float scale, float mask_value
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_matrices * seq_len * seq_len;
    if (idx >= total) return;
    size_t local = idx % (seq_len * seq_len);
    size_t s = local / seq_len;
    size_t t = local % seq_len;
    if (t > s) {
        scores[idx] = mask_value;
    } else {
        scores[idx] *= scale;
    }
}

// ============================================================
// RMSNorm: one block per row
// out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
// ============================================================

extern "C" __global__ void rmsnorm_fwd(
    float* out, const float* inp, const float* weight,
    size_t rows, size_t d, float eps
) {
    extern __shared__ float sdata[];

    size_t row = blockIdx.x;
    if (row >= rows) return;
    size_t off = row * d;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    // Compute sum of squares
    float local_sum_sq = 0.0f;
    for (size_t j = tid; j < d; j += stride) {
        float v = inp[off + j];
        local_sum_sq += v * v;
    }
    sdata[tid] = local_sum_sq;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean_sq = sdata[0] / (float)d;
    float rstd = rsqrtf(mean_sq + eps);
    __syncthreads();

    // Normalize and scale
    for (size_t j = tid; j < d; j += stride) {
        out[off + j] = inp[off + j] * rstd * weight[j];
    }
}

// ============================================================
// RoPE: Rotary Position Embeddings
// Processes pairs (2i, 2i+1) applying rotation by theta = pos / base^(2i/d)
// ============================================================

extern "C" __global__ void rope_fwd(
    float* out, const float* inp,
    size_t n, size_t pos, size_t head_dim, float theta
) {
    size_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_pairs = n / 2;
    if (pair_idx >= n_pairs) return;

    // Find which dimension pair within the head
    size_t dim_pair = pair_idx % (head_dim / 2);

    // Compute rotation angle
    float freq = 1.0f / powf(theta, 2.0f * (float)dim_pair / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    size_t idx0 = pair_idx * 2 - (pair_idx % (head_dim / 2)) * 2 + dim_pair * 2;
    // Simpler: map pair_idx to the actual element indices
    // pair_idx tells us which pair globally. Within each head_dim,
    // pairs are (0,1), (2,3), ..., (head_dim-2, head_dim-1)
    // Across heads, they continue linearly.
    size_t elem0 = pair_idx * 2;
    size_t elem1 = pair_idx * 2 + 1;

    // But we need to check that elem0 and elem1 are within the same head
    // Since head_dim elements are contiguous and pairs are (2i, 2i+1) within head,
    // pair_idx = global_element / 2, and dim_pair = (global_element % head_dim) / 2
    // This is correct since elements are laid out as [head0_dim0, head0_dim1, ..., head1_dim0, ...]

    float x0 = inp[elem0];
    float x1 = inp[elem1];
    out[elem0] = x0 * cos_val - x1 * sin_val;
    out[elem1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================
// SwiGLU: out[i] = silu(gate[i]) * up[i]
// silu(x) = x / (1 + exp(-x))
// ============================================================

extern "C" __global__ void swiglu_fwd(
    float* out, const float* gate, const float* up, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float silu = g / (1.0f + expf(-g));
    out[i] = silu * up[i];
}

// ============================================================
// Repeat KV: expand [n_kv_heads, seq, d_head] to [n_q_heads, seq, d_head]
// ============================================================

extern "C" __global__ void repeat_kv_kernel(
    float* out, const float* inp,
    size_t n_kv_heads, size_t n_q_heads, size_t seq, size_t d_head
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_q_heads * seq * d_head;
    if (idx >= total) return;

    size_t n_rep = n_q_heads / n_kv_heads;
    size_t head_size = seq * d_head;

    // Decompose output index
    size_t q_head = idx / head_size;
    size_t within_head = idx % head_size;

    // Map query head to KV head
    size_t kv_head = q_head / n_rep;

    out[idx] = inp[kv_head * head_size + within_head];
}

// ============================================================
// Fused gather + reshape + repeat_kv
// Reads directly from pre-allocated cache [max_seq, n_kv*hd]
// and produces [n_q_heads, cached_len, hd] in one pass.
// ============================================================

extern "C" __global__ void gather_reshape_repeat_kv(
    float* out, const float* cache,
    size_t cached_len, size_t n_kv_heads, size_t n_q_heads, size_t head_dim, size_t kv_dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_q_heads * cached_len * head_dim;
    if (idx >= total) return;

    size_t n_rep = n_q_heads / n_kv_heads;
    size_t d = idx % head_dim;
    size_t s = (idx / head_dim) % cached_len;
    size_t q_head = idx / (cached_len * head_dim);
    size_t kv_head = q_head / n_rep;

    out[idx] = cache[s * kv_dim + kv_head * head_dim + d];
}

// ============================================================
// RoPE with pre-computed frequencies
// Input: [seq, n_heads * head_dim], freqs: [head_dim/2]
// Each thread handles one (x0, x1) pair
// ============================================================

extern "C" __global__ void rope_with_freqs_fwd(
    float* out, const float* inp, const float* freqs,
    size_t total_pairs, size_t n_heads, size_t head_dim, size_t start_pos
) {
    size_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;

    size_t half_hd = head_dim / 2;
    size_t total_dim = n_heads * head_dim;
    size_t total_pairs_per_row = n_heads * half_hd;

    // Which row (sequence position) and which pair within that row
    size_t row = pair_idx / total_pairs_per_row;
    size_t pair_in_row = pair_idx % total_pairs_per_row;

    // Which dimension pair within the head
    size_t dim_pair = pair_in_row % half_hd;

    // Position for this row
    size_t pos = start_pos + row;

    // Compute rotation angle from pre-computed frequency
    float freq = freqs[dim_pair];
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Map to element indices
    size_t head_in_row = pair_in_row / half_hd;
    size_t elem0 = row * total_dim + head_in_row * head_dim + dim_pair * 2;
    size_t elem1 = elem0 + 1;

    float x0 = inp[elem0];
    float x1 = inp[elem1];
    out[elem0] = x0 * cos_val - x1 * sin_val;
    out[elem1] = x0 * sin_val + x1 * cos_val;
}

// ============================================================
// RoPE with pre-computed frequencies + output scaling
// Fuses the Q pre-scaling (1/√d) into the RoPE write to eliminate
// a separate scale kernel launch per layer.
// ============================================================

extern "C" __global__ void rope_with_freqs_scaled_fwd(
    float* out, const float* inp, const float* freqs,
    size_t total_pairs, size_t n_heads, size_t head_dim,
    size_t start_pos, float output_scale
) {
    size_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;

    size_t half_hd = head_dim / 2;
    size_t total_dim = n_heads * head_dim;
    size_t total_pairs_per_row = n_heads * half_hd;

    size_t row = pair_idx / total_pairs_per_row;
    size_t pair_in_row = pair_idx % total_pairs_per_row;

    size_t dim_pair = pair_in_row % half_hd;

    size_t pos = start_pos + row;

    float freq = freqs[dim_pair];
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    size_t head_in_row = pair_in_row / half_hd;
    size_t elem0 = row * total_dim + head_in_row * head_dim + dim_pair * 2;
    size_t elem1 = elem0 + 1;

    float x0 = inp[elem0];
    float x1 = inp[elem1];
    out[elem0] = (x0 * cos_val - x1 * sin_val) * output_scale;
    out[elem1] = (x0 * sin_val + x1 * cos_val) * output_scale;
}
"#;

#[cfg(feature = "bf16")]
const BF16_KERNEL_SOURCE: &str = r#"
typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_float(bf16_t x) {
    unsigned int bits = ((unsigned int)x) << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

__device__ __forceinline__ bf16_t float_to_bf16(float x) {
    unsigned int bits;
    memcpy(&bits, &x, 4);
    return (bf16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

// ============================================================
// Fused RMSNorm + bf16 cast: writes both f32 and bf16 output in one pass
// ============================================================

extern "C" __global__ void rmsnorm_fwd_with_bf16(
    float* out, bf16_t* out_bf16, const float* inp, const float* weight,
    size_t rows, size_t d, float eps
) {
    extern __shared__ float sdata[];

    size_t row = blockIdx.x;
    if (row >= rows) return;
    size_t off = row * d;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    float local_sum_sq = 0.0f;
    for (size_t j = tid; j < d; j += stride) {
        float v = inp[off + j];
        local_sum_sq += v * v;
    }
    sdata[tid] = local_sum_sq;
    __syncthreads();
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean_sq = sdata[0] / (float)d;
    float rstd = rsqrtf(mean_sq + eps);
    __syncthreads();

    for (size_t j = tid; j < d; j += stride) {
        float val = inp[off + j] * rstd * weight[j];
        out[off + j] = val;
        out_bf16[off + j] = float_to_bf16(val);
    }
}

// ============================================================
// Fused SwiGLU + bf16 cast: writes both f32 and bf16 output in one pass
// ============================================================

extern "C" __global__ void swiglu_fwd_with_bf16(
    float* out, bf16_t* out_bf16, const float* gate, const float* up, size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float silu = g / (1.0f + expf(-g));
    float val = silu * up[i];
    out[i] = val;
    out_bf16[i] = float_to_bf16(val);
}

// Embedding lookup from bf16 weights → f32 output
extern "C" __global__ void embedding_bf16_fwd(
    float* out, const bf16_t* weight, const unsigned int* indices,
    size_t n_indices, size_t dim
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_indices * dim) return;
    size_t token = idx / dim;
    size_t d = idx % dim;
    out[token * dim + d] = bf16_to_float(weight[indices[token] * dim + d]);
}

extern "C" __global__ void f32_to_bf16(bf16_t* out, const float* in, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = float_to_bf16(in[i]);
}

extern "C" __global__ void bf16_to_f32(float* out, const bf16_t* in, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bf16_to_float(in[i]);
}
"#;
