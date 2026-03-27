// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vision Transformer (ViT) backbone.
//!
//! Implements non-causal multi-head self-attention for vision.
//! Input: `[3, H, W]` → Output: `[embed_dim]` (CLS token embedding).

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::nn::PatchEmbedding;

/// ViT configuration.
#[derive(Clone, Debug)]
pub struct VitConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub mlp_ratio: f32,
    pub in_channels: usize,
}

impl VitConfig {
    /// ViT-B/32 (CLIP default).
    pub fn vit_b32() -> Self {
        Self {
            image_size: 224,
            patch_size: 32,
            embed_dim: 768,
            num_heads: 12,
            num_layers: 12,
            mlp_ratio: 4.0,
            in_channels: 3,
        }
    }

    /// ViT-B/16.
    pub fn vit_b16() -> Self {
        Self { patch_size: 16, ..Self::vit_b32() }
    }

    /// ViT-L/14.
    pub fn vit_l14() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            embed_dim: 1024,
            num_heads: 16,
            num_layers: 24,
            mlp_ratio: 4.0,
            in_channels: 3,
        }
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    pub fn d_head(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    pub fn d_ff(&self) -> usize {
        (self.embed_dim as f32 * self.mlp_ratio) as usize
    }
}

/// Non-causal multi-head self-attention for ViT.
///
/// Unlike GPT's causal attention, this has no mask — every token attends
/// to every other token (bidirectional).
pub struct VitAttention<B: MathBackend> {
    /// Fused QKV projection: `[embed_dim, 3 * embed_dim]`.
    pub qkv_weight: Tensor<B>,
    /// QKV bias: `[3 * embed_dim]`.
    pub qkv_bias: Tensor<B>,
    /// Output projection: `[embed_dim, embed_dim]`.
    pub proj_weight: Tensor<B>,
    /// Output projection bias: `[embed_dim]`.
    pub proj_bias: Tensor<B>,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_model: usize,
}

impl<B: MathBackend> VitAttention<B> {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        let d_head = d_model / n_heads;
        Self {
            qkv_weight: Tensor::from_vec(
                vec![0.0; d_model * 3 * d_model],
                Shape::new(&[d_model, 3 * d_model]),
            ),
            qkv_bias: Tensor::from_vec(vec![0.0; 3 * d_model], Shape::new(&[3 * d_model])),
            proj_weight: Tensor::from_vec(
                vec![0.0; d_model * d_model],
                Shape::new(&[d_model, d_model]),
            ),
            proj_bias: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            n_heads,
            d_head,
            d_model,
        }
    }

    /// Forward: `[seq_len, d_model]` → `[seq_len, d_model]`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let seq = input.shape.dims()[0];

        // QKV projection: [seq, d_model] @ [d_model, 3*d_model] → [seq, 3*d_model]
        let qkv_storage = B::matmul(
            &input.data,
            &self.qkv_weight.data,
            seq,
            self.d_model,
            3 * self.d_model,
            false,
            false,
        );
        // Add bias
        let qkv_shape = Shape::new(&[seq, 3 * self.d_model]);
        let bias_shape = Shape::new(&[1, 3 * self.d_model]);
        let qkv_biased =
            B::add(&qkv_storage, &self.qkv_bias.data, &qkv_shape, &bias_shape, &qkv_shape);

        // Split Q,K,V and reshape to [n_heads, seq, d_head]
        let (q, k, v) = B::split_qkv_reshape_heads(&qkv_biased, seq, self.n_heads, self.d_head);

        // Attention: Q @ K^T / sqrt(d_head)
        let scores = B::matmul_strided_batched(
            &q,
            &k,
            self.n_heads,
            seq,
            self.d_head,
            seq,
            false,
            true,
        );

        // Scaled softmax (no causal mask)
        let scale = 1.0 / (self.d_head as f32).sqrt();
        let scores_shape = Shape::new(&[self.n_heads, seq, seq]);
        let attn = B::scaled_softmax(&scores, scale, &scores_shape);

        // Context: attn @ V → [n_heads, seq, d_head]
        let context = B::matmul_strided_batched(
            &attn,
            &v,
            self.n_heads,
            seq,
            seq,
            self.d_head,
            false,
            false,
        );

        // Reshape back: [n_heads, seq, d_head] → [seq, d_model]
        let context_flat = B::reshape_from_heads(&context, 1, seq, self.n_heads, self.d_head);

        // Output projection: [seq, d_model] @ [d_model, d_model] → [seq, d_model]
        let proj = B::matmul(
            &context_flat,
            &self.proj_weight.data,
            seq,
            self.d_model,
            self.d_model,
            false,
            false,
        );
        let out_shape = Shape::new(&[seq, self.d_model]);
        let proj_bias_shape = Shape::new(&[1, self.d_model]);
        let result = B::add(&proj, &self.proj_bias.data, &out_shape, &proj_bias_shape, &out_shape);

        Tensor::new(result, out_shape)
    }
}

/// Feed-forward MLP for ViT blocks.
pub struct VitMlp<B: MathBackend> {
    pub fc1_weight: Tensor<B>,
    pub fc1_bias: Tensor<B>,
    pub fc2_weight: Tensor<B>,
    pub fc2_bias: Tensor<B>,
    pub d_model: usize,
    pub d_ff: usize,
}

impl<B: MathBackend> VitMlp<B> {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            fc1_weight: Tensor::from_vec(
                vec![0.0; d_model * d_ff],
                Shape::new(&[d_model, d_ff]),
            ),
            fc1_bias: Tensor::from_vec(vec![0.0; d_ff], Shape::new(&[d_ff])),
            fc2_weight: Tensor::from_vec(
                vec![0.0; d_ff * d_model],
                Shape::new(&[d_ff, d_model]),
            ),
            fc2_bias: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            d_model,
            d_ff,
        }
    }

    /// Forward: `[seq, d_model]` → `[seq, d_model]`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let seq = input.shape.dims()[0];

        // fc1: [seq, d_model] @ [d_model, d_ff] + bias → [seq, d_ff]
        let hidden = B::matmul(
            &input.data,
            &self.fc1_weight.data,
            seq,
            self.d_model,
            self.d_ff,
            false,
            false,
        );
        let hidden_shape = Shape::new(&[seq, self.d_ff]);
        let bias1_shape = Shape::new(&[1, self.d_ff]);
        let hidden =
            B::add(&hidden, &self.fc1_bias.data, &hidden_shape, &bias1_shape, &hidden_shape);

        // GELU activation
        let hidden = B::gelu(&hidden);

        // fc2: [seq, d_ff] @ [d_ff, d_model] + bias → [seq, d_model]
        let out = B::matmul(&hidden, &self.fc2_weight.data, seq, self.d_ff, self.d_model, false, false);
        let out_shape = Shape::new(&[seq, self.d_model]);
        let bias2_shape = Shape::new(&[1, self.d_model]);
        let result = B::add(&out, &self.fc2_bias.data, &out_shape, &bias2_shape, &out_shape);

        Tensor::new(result, out_shape)
    }
}

/// A single ViT transformer block (pre-norm).
///
/// `x → LN → Attn → + → LN → MLP → +`
pub struct VitBlock<B: MathBackend> {
    pub ln1_gamma: Tensor<B>,
    pub ln1_beta: Tensor<B>,
    pub attn: VitAttention<B>,
    pub ln2_gamma: Tensor<B>,
    pub ln2_beta: Tensor<B>,
    pub mlp: VitMlp<B>,
    pub d_model: usize,
}

impl<B: MathBackend> VitBlock<B> {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        Self {
            ln1_gamma: Tensor::from_vec(vec![1.0; d_model], Shape::new(&[d_model])),
            ln1_beta: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            attn: VitAttention::new(d_model, n_heads),
            ln2_gamma: Tensor::from_vec(vec![1.0; d_model], Shape::new(&[d_model])),
            ln2_beta: Tensor::from_vec(vec![0.0; d_model], Shape::new(&[d_model])),
            mlp: VitMlp::new(d_model, d_ff),
            d_model,
        }
    }

    /// Forward: `[seq, d_model]` → `[seq, d_model]`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let shape = input.shape.clone();

        // LN → Attn → residual
        let normed = B::layernorm_inference(
            &input.data,
            &self.ln1_gamma.data,
            &self.ln1_beta.data,
            &shape,
            1e-5,
        );
        let attn_out = self.attn.forward(&Tensor::new(normed, shape.clone()));
        let x = B::add(&input.data, &attn_out.data, &shape, &shape, &shape);

        // LN → MLP → residual
        let normed2 = B::layernorm_inference(
            &x,
            &self.ln2_gamma.data,
            &self.ln2_beta.data,
            &shape,
            1e-5,
        );
        let mlp_out = self.mlp.forward(&Tensor::new(normed2, shape.clone()));
        let result = B::add(&x, &mlp_out.data, &shape, &shape, &shape);

        Tensor::new(result, shape)
    }
}

/// Vision Transformer backbone.
///
/// Input: `[3, H, W]` → Output: `[embed_dim]` (CLS token embedding).
pub struct Vit<B: MathBackend> {
    pub patch_embed: PatchEmbedding<B>,
    /// CLS token: `[1, embed_dim]`.
    pub cls_token: Tensor<B>,
    /// Positional embeddings: `[1 + num_patches, embed_dim]`.
    pub pos_embed: Tensor<B>,
    pub blocks: Vec<VitBlock<B>>,
    /// Final layer norm.
    pub ln_post_gamma: Tensor<B>,
    pub ln_post_beta: Tensor<B>,
    pub config: VitConfig,
}

impl<B: MathBackend> Vit<B> {
    /// Create a zero-initialized ViT.
    pub fn new(config: VitConfig) -> Self {
        let num_patches = config.num_patches();
        let d = config.embed_dim;
        let d_ff = config.d_ff();

        let blocks = (0..config.num_layers)
            .map(|_| VitBlock::new(d, config.num_heads, d_ff))
            .collect();

        Self {
            patch_embed: PatchEmbedding::new(config.in_channels, d, config.patch_size),
            cls_token: Tensor::from_vec(vec![0.0; d], Shape::new(&[1, d])),
            pos_embed: Tensor::from_vec(
                vec![0.0; (1 + num_patches) * d],
                Shape::new(&[1 + num_patches, d]),
            ),
            blocks,
            ln_post_gamma: Tensor::from_vec(vec![1.0; d], Shape::new(&[d])),
            ln_post_beta: Tensor::from_vec(vec![0.0; d], Shape::new(&[d])),
            config,
        }
    }

    /// Forward pass: `[3, H, W]` → `[embed_dim]`.
    ///
    /// Returns the CLS token embedding after all transformer blocks + final LN.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let d = self.config.embed_dim;

        // Patch embedding: [3, H, W] → [num_patches, embed_dim]
        let patches = self.patch_embed.forward(input);
        let num_patches = patches.shape.dims()[0];
        let seq = 1 + num_patches; // CLS + patches

        // Prepend CLS token: [1 + num_patches, embed_dim]
        let cls_data = self.cls_token.to_vec();
        let patch_data = patches.to_vec();
        let mut seq_data = Vec::with_capacity(seq * d);
        seq_data.extend_from_slice(&cls_data);
        seq_data.extend_from_slice(&patch_data);

        // Add positional embeddings
        let seq_shape = Shape::new(&[seq, d]);
        let seq_storage = B::from_vec(seq_data, &seq_shape);
        let x = B::add(&seq_storage, &self.pos_embed.data, &seq_shape, &seq_shape, &seq_shape);

        // Transformer blocks
        let mut x_tensor = Tensor::new(x, seq_shape);
        for block in &self.blocks {
            x_tensor = block.forward(&x_tensor);
        }

        // Final layer norm
        let normed = B::layernorm_inference(
            &x_tensor.data,
            &self.ln_post_gamma.data,
            &self.ln_post_beta.data,
            &x_tensor.shape,
            1e-5,
        );

        // Extract CLS token (first row): [embed_dim]
        let all_data = B::to_vec(&normed);
        let cls_embedding = all_data[..d].to_vec();
        Tensor::from_vec(cls_embedding, Shape::new(&[d]))
    }
}

impl<B: MathBackend> Module<B> for Vit<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters());
        params.push(&self.cls_token);
        params.push(&self.pos_embed);
        for block in &self.blocks {
            params.push(&block.ln1_gamma);
            params.push(&block.ln1_beta);
            params.push(&block.attn.qkv_weight);
            params.push(&block.attn.qkv_bias);
            params.push(&block.attn.proj_weight);
            params.push(&block.attn.proj_bias);
            params.push(&block.ln2_gamma);
            params.push(&block.ln2_beta);
            params.push(&block.mlp.fc1_weight);
            params.push(&block.mlp.fc1_bias);
            params.push(&block.mlp.fc2_weight);
            params.push(&block.mlp.fc2_bias);
        }
        params.push(&self.ln_post_gamma);
        params.push(&self.ln_post_beta);
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn vit_attention_output_shape() {
        let attn = VitAttention::<CpuBackend>::new(768, 12);
        let input = Tensor::from_vec(vec![0.0; 50 * 768], Shape::new(&[50, 768]));
        let output = attn.forward(&input);
        assert_eq!(output.shape.dims(), &[50, 768]);
    }

    #[test]
    fn vit_mlp_output_shape() {
        let mlp = VitMlp::<CpuBackend>::new(768, 3072);
        let input = Tensor::from_vec(vec![0.0; 50 * 768], Shape::new(&[50, 768]));
        let output = mlp.forward(&input);
        assert_eq!(output.shape.dims(), &[50, 768]);
    }

    #[test]
    fn vit_block_output_shape() {
        let block = VitBlock::<CpuBackend>::new(768, 12, 3072);
        let input = Tensor::from_vec(vec![0.0; 50 * 768], Shape::new(&[50, 768]));
        let output = block.forward(&input);
        assert_eq!(output.shape.dims(), &[50, 768]);
    }

    #[test]
    fn vit_b32_output_shape() {
        let config = VitConfig::vit_b32();
        let model = Vit::<CpuBackend>::new(config);
        let input = Tensor::from_vec(vec![0.0; 3 * 224 * 224], Shape::new(&[3, 224, 224]));
        let output = model.forward(&input);
        // CLS token embedding
        assert_eq!(output.shape.dims(), &[768]);
    }

    #[test]
    fn vit_parameter_count() {
        let config = VitConfig {
            image_size: 32,
            patch_size: 16,
            embed_dim: 64,
            num_heads: 4,
            num_layers: 2,
            mlp_ratio: 4.0,
            in_channels: 3,
        };
        let model = Vit::<CpuBackend>::new(config);
        let params = model.parameters();
        // patch_embed (2) + cls (1) + pos_embed (1) + 2 blocks × 12 params + ln_post (2) = 30
        assert_eq!(params.len(), 2 + 1 + 1 + 2 * 12 + 2);
    }
}
