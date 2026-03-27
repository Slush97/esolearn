pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "scry-gpu")]
pub mod scry_gpu;

use crate::tensor::shape::Shape;

/// Memory management trait — allocate, copy, transfer between host/device.
pub trait DeviceBackend: Sized {
    type Storage: Clone;
    type Stream;
    #[cfg(feature = "quantize")]
    type I8Storage: Clone;

    #[cfg(feature = "quantize")]
    fn i8_from_vec(data: Vec<i8>) -> Self::I8Storage;
    #[cfg(feature = "quantize")]
    fn i8_to_vec(storage: &Self::I8Storage) -> Vec<i8>;

    fn zeros(shape: &Shape) -> Self::Storage;
    fn ones(shape: &Shape) -> Self::Storage;
    fn from_vec(data: Vec<f32>, shape: &Shape) -> Self::Storage;
    /// Create storage from f32 data with optional raw bf16 bytes for direct upload.
    /// Default ignores the bf16 bytes and falls back to `from_vec`.
    #[cfg(feature = "bf16")]
    fn from_vec_with_bf16(data: Vec<f32>, bf16_bytes: &[u8], shape: &Shape) -> Self::Storage {
        let _ = bf16_bytes;
        Self::from_vec(data, shape)
    }
    fn to_vec(storage: &Self::Storage) -> Vec<f32>;
    /// Consume storage and return the underlying `Vec<f32>`.
    /// Default clones via `to_vec`; backends with `Storage = Vec<f32>` override to move.
    fn into_vec(storage: Self::Storage) -> Vec<f32> {
        Self::to_vec(&storage)
    }
    /// Borrow storage as a `&[f32]` slice without cloning.
    /// Default clones via `to_vec` — backends with `Storage = Vec<f32>` should override.
    fn as_slice(storage: &Self::Storage) -> std::borrow::Cow<'_, [f32]> {
        std::borrow::Cow::Owned(Self::to_vec(storage))
    }
    fn clone_storage(storage: &Self::Storage) -> Self::Storage;
}

/// Math operations trait — all operations needed for forward passes.
pub trait MathBackend: DeviceBackend {
    // ---- Core forward ops ----

    /// General matrix multiply: `C = alpha * op(A) * op(B) + beta * C`
    fn matmul(
        a: &Self::Storage,
        b: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Self::Storage;

    /// Fused matrix multiply + bias add: `C = A @ B + bias` (bias broadcast along rows).
    /// Avoids a separate allocation for the matmul result.
    fn matmul_bias(
        a: &Self::Storage,
        b: &Self::Storage,
        bias: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Self::Storage {
        // Default: matmul then add bias in-place
        let mut c = Self::matmul(a, b, m, k, n, trans_a, trans_b);
        let c_vec = Self::to_vec(&c);
        let bias_vec = Self::to_vec(bias);
        let mut out = c_vec;
        for row in 0..m {
            for col in 0..n {
                out[row * n + col] += bias_vec[col];
            }
        }
        c = Self::from_vec(out, &Shape::new(&[m, n]));
        c
    }

    /// Elementwise add with broadcasting.
    fn add(
        a: &Self::Storage,
        b: &Self::Storage,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> Self::Storage;

    /// In-place elementwise add: `dst[i] += src[i]` (same shape, no broadcast).
    /// Eliminates an allocation compared to `add` + reassign.
    fn add_inplace(dst: &mut Self::Storage, src: &Self::Storage) {
        let mut d = Self::to_vec(dst);
        let s = Self::to_vec(src);
        for (di, si) in d.iter_mut().zip(s.iter()) {
            *di += si;
        }
        let len = d.len();
        *dst = Self::from_vec(d, &Shape::new(&[len]));
    }

    /// Softmax along the last axis.
    fn softmax(input: &Self::Storage, shape: &Shape) -> Self::Storage;

    /// Fused scale + softmax: applies `x * scale` then softmax along the last axis.
    /// Eliminates the separate `scale()` allocation.
    fn scaled_softmax(input: &Self::Storage, scale: f32, shape: &Shape) -> Self::Storage {
        let scaled = Self::scale(input, scale);
        Self::softmax(&scaled, shape)
    }

    /// Layer normalization along the last axis.
    /// Returns `(output, mean, rstd)`.
    fn layernorm(
        input: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> (Self::Storage, Self::Storage, Self::Storage);

    /// Inference-only layer normalization — returns only the output, no mean/rstd.
    fn layernorm_inference(
        input: &Self::Storage,
        gamma: &Self::Storage,
        beta: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> Self::Storage {
        let (output, _, _) = Self::layernorm(input, gamma, beta, shape, eps);
        output
    }

    /// GELU activation (tanh approximation).
    fn gelu(input: &Self::Storage) -> Self::Storage;

    /// Embedding lookup: gather rows by indices.
    fn embedding(
        weight: &Self::Storage,
        indices: &[usize],
        vocab: usize,
        dim: usize,
    ) -> Self::Storage;

    /// Sum all elements to scalar.
    fn sum(input: &Self::Storage) -> f32;

    /// Elementwise multiply: `a * b` (same shape, no broadcast).
    fn mul_elementwise(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;

    /// Scale all elements: `a * scalar`.
    fn scale(a: &Self::Storage, scalar: f32) -> Self::Storage;

    /// Concatenate two row-major matrices along rows (axis 0).
    fn concat_rows(
        a: &Self::Storage,
        b: &Self::Storage,
        a_rows: usize,
        b_rows: usize,
        cols: usize,
    ) -> Self::Storage;

    // ---- Llama-specific ops ----

    /// RMS normalization: `out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
    fn rmsnorm(
        input: &Self::Storage,
        weight: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> Self::Storage;

    /// Fused RMS normalization + bf16 cast.
    /// Returns f32 output with bf16 shadow pre-populated (avoids a separate f32→bf16 kernel).
    /// Default falls back to `rmsnorm` (no bf16 fusion).
    #[cfg(feature = "bf16")]
    fn rmsnorm_with_bf16(
        input: &Self::Storage,
        weight: &Self::Storage,
        shape: &Shape,
        eps: f32,
    ) -> Self::Storage {
        Self::rmsnorm(input, weight, shape, eps)
    }

    /// Rotary position embeddings.
    fn rope(
        input: &Self::Storage,
        shape: &Shape,
        pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> Self::Storage;

    /// Rotary position embeddings with pre-computed frequencies (GPU-resident).
    ///
    /// The `freqs` storage is already on-device (uploaded once at model load),
    /// eliminating per-token host→device transfers.
    fn rope_with_freqs_preloaded(
        input: &Self::Storage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &Self::Storage,
    ) -> Self::Storage;

    /// Rotary position embeddings with pre-computed frequencies and output scaling (GPU-resident).
    ///
    /// Fuses RoPE and scalar multiply into a single kernel: `out = RoPE(input) * scale`.
    /// Default implementation calls `rope_with_freqs_preloaded` then `scale`.
    fn rope_with_freqs_scaled_preloaded(
        input: &Self::Storage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &Self::Storage,
        scale: f32,
    ) -> Self::Storage {
        let out = Self::rope_with_freqs_preloaded(input, seq, n_heads, head_dim, start_pos, freqs);
        Self::scale(&out, scale)
    }

    /// Rotary position embeddings with pre-computed frequencies (host-side).
    ///
    /// Converts `&[f64]` freqs to `Self::Storage` then delegates to `rope_with_freqs_preloaded`.
    /// Prefer `_preloaded` variant in hot paths to avoid per-call uploads.
    fn rope_with_freqs(
        input: &Self::Storage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &[f64],
    ) -> Self::Storage {
        let freqs_f32: Vec<f32> = freqs.iter().map(|&f| f as f32).collect();
        let freqs_storage = Self::from_vec(freqs_f32, &Shape::new(&[head_dim / 2]));
        Self::rope_with_freqs_preloaded(input, seq, n_heads, head_dim, start_pos, &freqs_storage)
    }

    /// Rotary position embeddings with pre-computed frequencies and output scaling (host-side).
    ///
    /// Converts `&[f64]` freqs to `Self::Storage` then delegates to `rope_with_freqs_scaled_preloaded`.
    fn rope_with_freqs_scaled(
        input: &Self::Storage,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &[f64],
        scale: f32,
    ) -> Self::Storage {
        let freqs_f32: Vec<f32> = freqs.iter().map(|&f| f as f32).collect();
        let freqs_storage = Self::from_vec(freqs_f32, &Shape::new(&[head_dim / 2]));
        Self::rope_with_freqs_scaled_preloaded(input, seq, n_heads, head_dim, start_pos, &freqs_storage, scale)
    }

    /// `SwiGLU`: `silu(gate) * up` where `silu(x) = x / (1 + exp(-x))`
    fn swiglu(gate: &Self::Storage, up: &Self::Storage) -> Self::Storage;

    /// Fused SwiGLU + bf16 cast.
    /// Returns f32 output with bf16 shadow pre-populated.
    /// Default falls back to `swiglu` (no bf16 fusion).
    #[cfg(feature = "bf16")]
    fn swiglu_with_bf16(gate: &Self::Storage, up: &Self::Storage) -> Self::Storage {
        Self::swiglu(gate, up)
    }

    /// Repeat KV heads for GQA: expand `[n_kv_heads, seq, d_head]` to `[n_q_heads, seq, d_head]`.
    fn repeat_kv(
        input: &Self::Storage,
        n_kv_heads: usize,
        n_q_heads: usize,
        seq: usize,
        d_head: usize,
    ) -> Self::Storage;

    /// Fused gather + reshape + repeat_kv: read directly from pre-allocated KV cache
    /// `[max_seq, n_kv_heads * head_dim]` and produce `[n_q_heads, cached_len, head_dim]`
    /// in a single pass. Eliminates separate gather_rows, reshape_for_heads, and repeat_kv.
    fn gather_reshape_repeat_kv(
        cache: &Self::Storage,
        max_seq: usize,
        cached_len: usize,
        n_kv_heads: usize,
        n_q_heads: usize,
        head_dim: usize,
    ) -> Self::Storage {
        // Default: fall back to the 3-step path
        let _ = max_seq;
        let kv_dim = n_kv_heads * head_dim;
        let gathered = Self::gather_rows(cache, max_seq, kv_dim, 0, cached_len);
        let reshaped = Self::reshape_for_heads(&gathered, 1, cached_len, n_kv_heads, head_dim);
        Self::repeat_kv(&reshaped, n_kv_heads, n_q_heads, cached_len, head_dim)
    }

    // ---- Attention helpers ----

    /// Extract columns `[col_start..col_start+col_count)` from a `[rows, total_cols]` matrix.
    fn gather_columns(
        storage: &Self::Storage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) -> Self::Storage {
        let data = Self::to_vec(storage);
        let mut out = vec![0.0f32; rows * col_count];
        for r in 0..rows {
            for c in 0..col_count {
                out[r * col_count + c] = data[r * total_cols + col_start + c];
            }
        }
        Self::from_vec(out, &Shape::new(&[rows, col_count]))
    }

    /// Scatter (additive) a `[rows, col_count]` source into columns of a destination.
    fn scatter_columns(
        dst: &mut Self::Storage,
        src: &Self::Storage,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_count: usize,
    ) {
        let mut dst_vec = Self::to_vec(dst);
        let src_vec = Self::to_vec(src);
        for r in 0..rows {
            for c in 0..col_count {
                dst_vec[r * total_cols + col_start + c] += src_vec[r * col_count + c];
            }
        }
        *dst = Self::from_vec(dst_vec, &Shape::new(&[rows, total_cols]));
    }

    /// Extract rows `[row_start..row_start+row_count)` from a `[total_rows, cols]` matrix.
    fn gather_rows(
        storage: &Self::Storage,
        total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) -> Self::Storage {
        let _ = total_rows;
        let data = Self::to_vec(storage);
        let start = row_start * cols;
        let end = start + row_count * cols;
        Self::from_vec(data[start..end].to_vec(), &Shape::new(&[row_count, cols]))
    }

    /// Write rows into a destination matrix (overwrite).
    fn scatter_rows(
        dst: &mut Self::Storage,
        src: &Self::Storage,
        total_rows: usize,
        cols: usize,
        row_start: usize,
        row_count: usize,
    ) {
        let mut dst_vec = Self::to_vec(dst);
        let src_vec = Self::to_vec(src);
        let start = row_start * cols;
        dst_vec[start..start + row_count * cols].copy_from_slice(&src_vec[..row_count * cols]);
        *dst = Self::from_vec(dst_vec, &Shape::new(&[total_rows, cols]));
    }

    /// Apply causal mask and scale to a `[seq_len, seq_len]` matrix in-place.
    fn apply_causal_mask_and_scale(
        scores: &mut Self::Storage,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
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

    /// Fused QKV split + reshape for multi-head attention.
    ///
    /// Input: `[seq, 3*d_model]` containing concatenated Q, K, V.
    /// Returns `(Q_heads, K_heads, V_heads)` each shaped `[n_heads, seq, d_head]`.
    /// Eliminates separate split + 3x from_vec + 3x reshape_for_heads.
    fn split_qkv_reshape_heads(
        qkv: &Self::Storage,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> (Self::Storage, Self::Storage, Self::Storage) {
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
    }

    /// Reshape `[B*S, H*d_head]` → `[B*H, S, d_head]` for batched multi-head attention.
    fn reshape_for_heads(
        storage: &Self::Storage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Self::Storage {
        // When B=1, S=1 the reshape is an identity permutation — skip the transpose.
        if batch == 1 && seq == 1 {
            return Self::clone_storage(storage);
        }
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

    /// Reshape host `&[f32]` data `[B*S, H*d_head]` → `[B*H, S, d_head]` for batched attention.
    /// Avoids clone + from_vec for CpuBackend where Storage = Vec<f32>.
    fn reshape_for_heads_from_host(
        data: &[f32],
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Self::Storage {
        let v = data.to_vec();
        let storage = Self::from_vec(v, &Shape::new(&[batch * seq, n_heads * d_head]));
        Self::reshape_for_heads(&storage, batch, seq, n_heads, d_head)
    }

    /// Reshape `[B*H, S, d_head]` → `[B*S, H*d_head]` (reverse of `reshape_for_heads`).
    fn reshape_from_heads(
        storage: &Self::Storage,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Self::Storage {
        // When B=1, S=1 the reshape is an identity permutation — skip the transpose.
        if batch == 1 && seq == 1 {
            return Self::clone_storage(storage);
        }
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

    // ---- INT8 quantized ops ----

    /// Matrix multiply with i8 weights: `C = A_f32 @ dequant(B_i8)`.
    ///
    /// `B_i8` is `[k, n]` in i8, `scale` converts i8→f32.
    /// Result is `[m, n]` in f32.
    ///
    /// Default: dequantize to f32 buffer then call `matmul`.
    #[cfg(feature = "quantize")]
    fn matmul_i8_f32(
        a: &Self::Storage,
        b_q: &Self::I8Storage,
        scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Storage {
        let b_i8 = Self::i8_to_vec(b_q);
        let b_f32: Vec<f32> = b_i8.iter().map(|&q| f32::from(q) * scale).collect();
        let b_storage = Self::from_vec(b_f32, &Shape::new(&[k, n]));
        Self::matmul(a, &b_storage, m, k, n, false, false)
    }

    /// Fused i8 matmul + bias: `C = A_f32 @ dequant(B_i8) + bias`.
    #[cfg(feature = "quantize")]
    fn matmul_i8_f32_bias(
        a: &Self::Storage,
        b_q: &Self::I8Storage,
        scale: f32,
        bias: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
    ) -> Self::Storage {
        let mut c = Self::matmul_i8_f32(a, b_q, scale, m, k, n);
        let c_vec = Self::to_vec(&c);
        let bias_vec = Self::to_vec(bias);
        let mut out = c_vec;
        for row in 0..m {
            for col in 0..n {
                out[row * n + col] += bias_vec[col];
            }
        }
        c = Self::from_vec(out, &Shape::new(&[m, n]));
        c
    }

    /// Strided batched matrix multiply.
    #[allow(clippy::too_many_arguments)]
    fn matmul_strided_batched(
        a: &Self::Storage,
        b: &Self::Storage,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Self::Storage {
        let a_stride = m * k;
        let b_stride = k * n;
        let c_stride = m * n;
        let total = batch_count * c_stride;
        let a_vec = Self::to_vec(a);
        let b_vec = Self::to_vec(b);
        let mut c_vec = vec![0.0f32; total];
        for i in 0..batch_count {
            let a_slice = Self::from_vec(
                a_vec[i * a_stride..(i + 1) * a_stride].to_vec(),
                &Shape::new(&[m, k]),
            );
            let b_slice = Self::from_vec(
                b_vec[i * b_stride..(i + 1) * b_stride].to_vec(),
                &Shape::new(&[k, n]),
            );
            let c_slice = Self::matmul(&a_slice, &b_slice, m, k, n, trans_a, trans_b);
            let c_data = Self::to_vec(&c_slice);
            c_vec[i * c_stride..(i + 1) * c_stride].copy_from_slice(&c_data);
        }
        Self::from_vec(c_vec, &Shape::new(&[batch_count * m, n]))
    }

    /// Apply causal mask and scale to batched matrices.
    fn apply_batched_causal_mask_and_scale(
        scores: &mut Self::Storage,
        num_matrices: usize,
        seq_len: usize,
        scale: f32,
        mask_value: f32,
    ) {
        let _ = num_matrices;
        let mut data = Self::to_vec(scores);
        let mat_size = seq_len * seq_len;
        let total = data.len() / mat_size;
        for m in 0..total {
            let off = m * mat_size;
            for s in 0..seq_len {
                for t in 0..seq_len {
                    let idx = off + s * seq_len + t;
                    if t > s {
                        data[idx] = mask_value;
                    } else {
                        data[idx] *= scale;
                    }
                }
            }
        }
        let n = data.len();
        *scores = Self::from_vec(data, &Shape::new(&[n]));
    }
}
