use crate::backend::MathBackend;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Matrix multiply: `C = op(A) @ op(B)`
/// `A`: `[M, K]` (or `[K, M]` if `trans_a`), `B`: `[K, N]` (or `[N, K]` if `trans_b`)
/// Output: `[M, N]`
pub fn matmul<B: MathBackend>(
    a: &Tensor<B>,
    b: &Tensor<B>,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Tensor<B> {
    let data = B::matmul(&a.data, &b.data, m, k, n, trans_a, trans_b);
    Tensor::new(data, Shape::new(&[m, n]))
}

/// Fused matrix multiply + bias add: `C = A @ B + bias`.
/// Bias is broadcast along rows. Saves one allocation vs separate matmul + add.
pub fn matmul_bias<B: MathBackend>(
    a: &Tensor<B>,
    b: &Tensor<B>,
    bias: &Tensor<B>,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Tensor<B> {
    let data = B::matmul_bias(&a.data, &b.data, &bias.data, m, k, n, trans_a, trans_b);
    Tensor::new(data, Shape::new(&[m, n]))
}

/// Elementwise add with broadcasting.
pub fn add<B: MathBackend>(a: &Tensor<B>, b: &Tensor<B>) -> Tensor<B> {
    let out_shape = Shape::broadcast(&a.shape, &b.shape).expect("broadcast failed in add");
    let data = B::add(&a.data, &b.data, &a.shape, &b.shape, &out_shape);
    Tensor::new(data, out_shape)
}

/// Softmax along the last axis.
pub fn softmax<B: MathBackend>(input: &Tensor<B>) -> Tensor<B> {
    let data = B::softmax(&input.data, &input.shape);
    Tensor::new(data, input.shape.clone())
}

/// Fused scale + softmax along the last axis.
pub fn scaled_softmax<B: MathBackend>(input: &Tensor<B>, scale: f32) -> Tensor<B> {
    let data = B::scaled_softmax(&input.data, scale, &input.shape);
    Tensor::new(data, input.shape.clone())
}

/// Layer normalization along the last axis.
/// Returns the normalized output (discards mean/rstd since no backward needed).
pub fn layernorm<B: MathBackend>(
    input: &Tensor<B>,
    gamma: &Tensor<B>,
    beta: &Tensor<B>,
    eps: f32,
) -> Tensor<B> {
    let (data, _mean, _rstd) =
        B::layernorm(&input.data, &gamma.data, &beta.data, &input.shape, eps);
    Tensor::new(data, input.shape.clone())
}

/// Inference-only layer normalization — skips mean/rstd allocation.
pub fn layernorm_inference<B: MathBackend>(
    input: &Tensor<B>,
    gamma: &Tensor<B>,
    beta: &Tensor<B>,
    eps: f32,
) -> Tensor<B> {
    let data = B::layernorm_inference(&input.data, &gamma.data, &beta.data, &input.shape, eps);
    Tensor::new(data, input.shape.clone())
}

/// GELU activation (tanh approximation).
pub fn gelu<B: MathBackend>(input: &Tensor<B>) -> Tensor<B> {
    let data = B::gelu(&input.data);
    Tensor::new(data, input.shape.clone())
}

/// Embedding lookup: gather rows by indices.
/// `weight`: `[V, D]`, `indices`: `[N]` → output: `[N, D]`
pub fn embedding<B: MathBackend>(
    weight: &Tensor<B>,
    indices: &[usize],
    vocab: usize,
    dim: usize,
) -> Tensor<B> {
    let data = B::embedding(&weight.data, indices, vocab, dim);
    Tensor::new(data, Shape::new(&[indices.len(), dim]))
}

/// RMS normalization along the last axis.
/// `out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`
///
/// When `bf16` feature is enabled, uses fused RMSNorm+bf16 cast to avoid
/// a separate f32→bf16 conversion kernel before the subsequent matmul.
pub fn rmsnorm<B: MathBackend>(
    input: &Tensor<B>,
    weight: &Tensor<B>,
    eps: f32,
) -> Tensor<B> {
    #[cfg(feature = "bf16")]
    {
        let data = B::rmsnorm_with_bf16(&input.data, &weight.data, &input.shape, eps);
        return Tensor::new(data, input.shape.clone());
    }
    #[cfg(not(feature = "bf16"))]
    {
        let data = B::rmsnorm(&input.data, &weight.data, &input.shape, eps);
        Tensor::new(data, input.shape.clone())
    }
}

/// Rotary position embeddings.
/// Applies rotation to dimension pairs at the given position.
pub fn rope<B: MathBackend>(
    input: &Tensor<B>,
    pos: usize,
    head_dim: usize,
    theta: f32,
) -> Tensor<B> {
    let data = B::rope(&input.data, &input.shape, pos, head_dim, theta);
    Tensor::new(data, input.shape.clone())
}

/// `SwiGLU` activation: `silu(gate) * up`
pub fn swiglu<B: MathBackend>(gate: &Tensor<B>, up: &Tensor<B>) -> Tensor<B> {
    let data = B::swiglu(&gate.data, &up.data);
    Tensor::new(data, gate.shape.clone())
}

/// Repeat KV heads to match query head count for GQA.
/// `input`: `[n_kv_heads * seq * d_head]` → `[n_q_heads * seq * d_head]`
pub fn repeat_kv<B: MathBackend>(
    input: &Tensor<B>,
    n_kv_heads: usize,
    n_q_heads: usize,
    seq: usize,
    d_head: usize,
) -> Tensor<B> {
    let data = B::repeat_kv(&input.data, n_kv_heads, n_q_heads, seq, d_head);
    Tensor::new(data, Shape::new(&[n_q_heads * seq, d_head]))
}
