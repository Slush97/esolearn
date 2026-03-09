use scry_llm::backend::MathBackend;
use scry_llm::nn::layernorm::LayerNormModule;
use scry_llm::nn::linear::Linear;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::model::conv1d::Conv1d;

/// Whisper audio encoder.
///
/// Architecture:
///   1. Two `Conv1D` layers downsample mel spectrogram by 2x
///   2. Sinusoidal positional embedding added
///   3. N Transformer encoder blocks (self-attention + MLP)
///
/// Input: mel spectrogram `[n_mels, n_frames]`
/// Output: encoder hidden states `[n_frames/2, d_model]`
pub struct WhisperEncoder<B: MathBackend> {
    /// First conv: `[n_mels, d_model, kernel=3, stride=1, pad=1]`.
    pub conv1: Conv1d<B>,
    /// Second conv: `[d_model, d_model, kernel=3, stride=2, pad=1]`.
    pub conv2: Conv1d<B>,
    /// Sinusoidal positional embedding: `[n_audio_ctx, d_model]`.
    pub positional_embedding: Tensor<B>,
    /// Pre-materialized positional embedding data in row-major `[n_audio_ctx, d_model]`.
    /// Cached to avoid `B::to_vec()` on every forward pass.
    pub pos_data: Vec<f32>,
    /// Transformer encoder blocks.
    pub blocks: Vec<EncoderBlock<B>>,
    /// Final layer norm.
    pub ln_post: LayerNormModule<B>,
    /// Model dimension.
    pub d_model: usize,
}

/// Single Transformer encoder block (self-attention + MLP).
pub struct EncoderBlock<B: MathBackend> {
    /// Pre-attention layer norm.
    pub attn_ln: LayerNormModule<B>,
    /// Multi-head self-attention (non-causal for encoder).
    pub attn: EncoderSelfAttention<B>,
    /// Pre-MLP layer norm.
    pub mlp_ln: LayerNormModule<B>,
    /// MLP: linear → GELU → linear.
    pub mlp_fc1: Linear<B>,
    /// MLP second projection.
    pub mlp_fc2: Linear<B>,
}

/// Encoder self-attention (bidirectional — no causal mask).
pub struct EncoderSelfAttention<B: MathBackend> {
    /// Combined Q, K, V projection: `[d_model, 3 * d_model]`.
    pub qkv_weight: Tensor<B>,
    /// Combined Q, K, V bias: `[3 * d_model]`.
    pub qkv_bias: Tensor<B>,
    /// Output projection: `[d_model, d_model]`.
    pub out_weight: Tensor<B>,
    /// Output bias: `[d_model]`.
    pub out_bias: Tensor<B>,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Per-head dimension.
    pub d_head: usize,
}

impl<B: MathBackend> WhisperEncoder<B> {
    /// Create a new encoder with random initialization.
    pub fn new(
        n_mels: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        n_audio_ctx: usize,
        rng: &mut fastrand::Rng,
    ) -> Self {
        let conv1 = Conv1d::new(n_mels, d_model, 3, 1, 1, rng);
        let conv2 = Conv1d::new(d_model, d_model, 3, 2, 1, rng);

        // Sinusoidal positional embeddings
        let pos_data = sinusoidal_embedding(n_audio_ctx, d_model);
        let positional_embedding =
            Tensor::from_vec(pos_data.clone(), Shape::new(&[n_audio_ctx, d_model]));

        let blocks = (0..n_layers)
            .map(|_| EncoderBlock::new(d_model, n_heads, rng))
            .collect();

        let ln_post = LayerNormModule::new(d_model);

        Self {
            conv1,
            conv2,
            positional_embedding,
            pos_data,
            blocks,
            ln_post,
            d_model,
        }
    }

    /// Forward pass: mel spectrogram → encoder hidden states.
    ///
    /// `mel`: `[n_mels, n_frames]` → returns `[n_frames/2, d_model]`.
    pub fn forward(&self, mel: &Tensor<B>) -> Tensor<B> {
        // Conv1: [n_mels, n_frames] → [d_model, n_frames] + GELU
        let x = self.conv1.forward(mel);
        let x = scry_llm::ops::gelu(&x);

        // Conv2: [d_model, n_frames] → [d_model, n_frames/2] + GELU
        let x = self.conv2.forward(&x);
        let x = scry_llm::ops::gelu(&x);

        // Fused transpose [d_model, out_len] → [out_len, d_model] + positional embedding add
        let out_len = x.shape.dims()[1];
        let x_vec = x.to_vec();
        let mut fused = vec![0.0f32; out_len * self.d_model];
        for t in 0..out_len {
            let pos_row = t * self.d_model;
            for c in 0..self.d_model {
                fused[pos_row + c] = x_vec[c * out_len + t] + self.pos_data[pos_row + c];
            }
        }
        let mut x = Tensor::<B>::from_vec(fused, Shape::new(&[out_len, self.d_model]));

        // Transformer encoder blocks
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // Final layer norm
        self.ln_post.forward(&x)
    }
}

impl<B: MathBackend> EncoderBlock<B> {
    fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            attn_ln: LayerNormModule::new(d_model),
            attn: EncoderSelfAttention::new(d_model, n_heads, rng),
            mlp_ln: LayerNormModule::new(d_model),
            mlp_fc1: Linear::new(d_model, d_model * 4, rng),
            mlp_fc2: Linear::new(d_model * 4, d_model, rng),
        }
    }

    fn forward(&self, x: &Tensor<B>) -> Tensor<B> {
        // Self-attention with residual
        let attn_in = self.attn_ln.forward(x);
        let attn_out = self.attn.forward(&attn_in);
        let x = scry_llm::ops::add(x, &attn_out);

        // MLP with residual
        let mlp_in = self.mlp_ln.forward(&x);
        let h = self.mlp_fc1.forward(&mlp_in);
        let h = scry_llm::ops::gelu(&h);
        let mlp_out = self.mlp_fc2.forward(&h);
        scry_llm::ops::add(&x, &mlp_out)
    }
}

impl<B: MathBackend> EncoderSelfAttention<B> {
    fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        let d_head = d_model / n_heads;
        let std_dev = 0.02;
        let mut rand_vec = |size: usize| -> Vec<f32> {
            (0..size)
                .map(|_| ((rng.f64() * 2.0 - 1.0) * std_dev) as f32)
                .collect()
        };

        Self {
            qkv_weight: Tensor::from_vec(
                rand_vec(d_model * 3 * d_model),
                Shape::new(&[d_model, 3 * d_model]),
            ),
            qkv_bias: Tensor::from_vec(vec![0.0; 3 * d_model], Shape::new(&[3 * d_model])),
            out_weight: Tensor::from_vec(
                rand_vec(d_model * d_model),
                Shape::new(&[d_model, d_model]),
            ),
            out_bias: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            n_heads,
            d_model,
            d_head,
        }
    }

    /// Bidirectional self-attention (no causal mask).
    ///
    /// Uses batched matmul across all heads simultaneously instead of a
    /// per-head loop, eliminating gather/scatter overhead and enabling
    /// BLAS to amortise call overhead across heads.
    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let seq_len = input.shape.dims()[0];
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let d_head = self.d_head;

        // QKV = input @ W_qkv + b_qkv → [seq_len, 3 * d_model]
        let qkv = scry_llm::ops::matmul_bias(
            input,
            &self.qkv_weight,
            &self.qkv_bias,
            seq_len,
            d_model,
            3 * d_model,
            false,
            false,
        );

        // Fused QKV split + reshape → [n_heads, seq_len, d_head] each
        let (q_heads, k_heads, v_heads) =
            B::split_qkv_reshape_heads(&qkv.data, seq_len, n_heads, d_head);

        // Batched scores = Q_heads @ K_heads^T → [n_heads * seq_len, seq_len]
        let scores = B::matmul_strided_batched(
            &q_heads, &k_heads, n_heads, seq_len, d_head, seq_len, false, true,
        );

        // Fused scale + softmax
        let scale = 1.0 / (d_head as f32).sqrt();
        let attn = B::scaled_softmax(
            &scores,
            scale,
            &Shape::new(&[n_heads * seq_len, seq_len]),
        );

        // Batched out = attn @ V_heads → [n_heads * seq_len, d_head]
        let out_heads = B::matmul_strided_batched(
            &attn, &v_heads, n_heads, seq_len, seq_len, d_head, false, false,
        );

        // Reshape [n_heads * seq_len, d_head] → [seq_len, d_model]
        let head_concat = B::reshape_from_heads(&out_heads, 1, seq_len, n_heads, d_head);

        // Output projection (fused matmul + bias)
        let hc = Tensor::<B>::new(head_concat, Shape::new(&[seq_len, d_model]));
        scry_llm::ops::matmul_bias(
            &hc,
            &self.out_weight,
            &self.out_bias,
            seq_len,
            d_model,
            d_model,
            false,
            false,
        )
    }
}

impl<B: MathBackend> Module<B> for EncoderSelfAttention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.qkv_weight, &self.qkv_bias, &self.out_weight, &self.out_bias]
    }
}

impl<B: MathBackend> Module<B> for EncoderBlock<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.attn_ln.parameters();
        params.extend(self.attn.parameters());
        params.extend(self.mlp_ln.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }
}

impl<B: MathBackend> Module<B> for WhisperEncoder<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params.push(&self.positional_embedding);
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_post.parameters());
        params
    }
}

/// Generate sinusoidal positional embeddings for the encoder.
fn sinusoidal_embedding(max_len: usize, d_model: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; max_len * d_model];
    for pos in 0..max_len {
        for i in 0..d_model {
            let angle =
                pos as f64 / 10_000.0_f64.powf(2.0 * (i / 2) as f64 / d_model as f64);
            data[pos * d_model + i] = if i % 2 == 0 {
                angle.sin() as f32
            } else {
                angle.cos() as f32
            };
        }
    }
    data
}
