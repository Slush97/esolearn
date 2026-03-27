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
    /// Optional pre-LN applied after positional embedding, before transformer blocks.
    /// Present in OpenCLIP models, absent in vanilla ViT/DeiT.
    pub ln_pre_gamma: Option<Tensor<B>>,
    pub ln_pre_beta: Option<Tensor<B>>,
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
            ln_pre_gamma: None,
            ln_pre_beta: None,
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
        let mut x = B::add(&seq_storage, &self.pos_embed.data, &seq_shape, &seq_shape, &seq_shape);

        // Optional pre-LN (present in OpenCLIP, absent in vanilla ViT)
        if let (Some(gamma), Some(beta)) = (&self.ln_pre_gamma, &self.ln_pre_beta) {
            x = B::layernorm_inference(&x, &gamma.data, &beta.data, &seq_shape, 1e-5);
        }

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

#[cfg(feature = "safetensors")]
impl<B: MathBackend> Vit<B> {
    /// Load a ViT from safetensors using OpenAI CLIP naming.
    ///
    /// `prefix` selects the key namespace:
    /// - `"visual."` — for full CLIP checkpoints (`visual.conv1.weight`, ...)
    /// - `""` — for standalone ViT checkpoints (`conv1.weight`, ...)
    ///
    /// # Weight layout
    ///
    /// Linear weights in PyTorch are `[out, in]`; we store `[in, out]` for
    /// direct matmul, so all Linear weights are transposed during loading.
    ///
    /// # Naming convention (with `prefix = "visual."`)
    ///
    /// ```text
    /// visual.conv1.weight                                [d, C, P, P]
    /// visual.class_embedding                             [d]
    /// visual.positional_embedding                        [1+N, d]
    /// visual.transformer.resblocks.{i}.ln_1.{weight,bias}
    /// visual.transformer.resblocks.{i}.attn.in_proj_weight   [3d, d]
    /// visual.transformer.resblocks.{i}.attn.in_proj_bias     [3d]
    /// visual.transformer.resblocks.{i}.attn.out_proj.{weight,bias}
    /// visual.transformer.resblocks.{i}.ln_2.{weight,bias}
    /// visual.transformer.resblocks.{i}.mlp.c_fc.{weight,bias}
    /// visual.transformer.resblocks.{i}.mlp.c_proj.{weight,bias}
    /// visual.ln_post.{weight,bias}
    /// ```
    pub fn from_safetensors(
        config: VitConfig,
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
    ) -> crate::error::Result<Self> {
        use crate::checkpoint::{load_f32, load_tensor, load_tensor_transposed};

        let d = config.embed_dim;
        let d_ff = config.d_ff();
        let n_heads = config.num_heads;
        let num_patches = config.num_patches();
        let in_ch = config.in_channels;
        let p = config.patch_size;
        let patch_len = in_ch * p * p;

        // ── Patch embedding ──
        // CLIP: conv1.weight [d, C, P, P] → our proj [d, C*P*P] (same flat layout)
        let proj = load_tensor(tensors, &format!("{prefix}conv1.weight"), &[d, patch_len])?;
        // CLIP doesn't store a separate patch embed bias; use zeros.
        let patch_bias = Tensor::from_vec(vec![0.0; d], Shape::new(&[d]));
        let patch_embed = PatchEmbedding {
            proj,
            bias: patch_bias,
            patch_size: p,
            in_channels: in_ch,
            embed_dim: d,
        };

        // ── CLS token ──
        // CLIP: class_embedding [d] → our cls_token [1, d]
        let cls_data = load_f32(tensors, &format!("{prefix}class_embedding"))?;
        let cls_token = Tensor::from_vec(cls_data, Shape::new(&[1, d]));

        // ── Positional embedding ──
        let pos_embed = load_tensor(
            tensors,
            &format!("{prefix}positional_embedding"),
            &[1 + num_patches, d],
        )?;

        // ── Transformer blocks ──
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let bp = format!("{prefix}transformer.resblocks.{i}");

            let ln1_gamma = load_tensor(tensors, &format!("{bp}.ln_1.weight"), &[d])?;
            let ln1_beta = load_tensor(tensors, &format!("{bp}.ln_1.bias"), &[d])?;

            // QKV: in_proj_weight [3d, d] → our qkv_weight [d, 3d] (transpose)
            let qkv_weight = load_tensor_transposed(
                tensors, &format!("{bp}.attn.in_proj_weight"), 3 * d, d,
            )?;
            let qkv_bias = load_tensor(tensors, &format!("{bp}.attn.in_proj_bias"), &[3 * d])?;

            // Output projection: out_proj.weight [d, d] → our proj_weight [d, d] (transpose)
            let proj_weight = load_tensor_transposed(
                tensors, &format!("{bp}.attn.out_proj.weight"), d, d,
            )?;
            let proj_bias = load_tensor(tensors, &format!("{bp}.attn.out_proj.bias"), &[d])?;

            let attn = VitAttention {
                qkv_weight,
                qkv_bias,
                proj_weight,
                proj_bias,
                n_heads,
                d_head: d / n_heads,
                d_model: d,
            };

            let ln2_gamma = load_tensor(tensors, &format!("{bp}.ln_2.weight"), &[d])?;
            let ln2_beta = load_tensor(tensors, &format!("{bp}.ln_2.bias"), &[d])?;

            // MLP: c_fc.weight [d_ff, d] → our fc1_weight [d, d_ff] (transpose)
            let fc1_weight = load_tensor_transposed(
                tensors, &format!("{bp}.mlp.c_fc.weight"), d_ff, d,
            )?;
            let fc1_bias = load_tensor(tensors, &format!("{bp}.mlp.c_fc.bias"), &[d_ff])?;

            // c_proj.weight [d, d_ff] → our fc2_weight [d_ff, d] (transpose)
            let fc2_weight = load_tensor_transposed(
                tensors, &format!("{bp}.mlp.c_proj.weight"), d, d_ff,
            )?;
            let fc2_bias = load_tensor(tensors, &format!("{bp}.mlp.c_proj.bias"), &[d])?;

            let mlp = VitMlp { fc1_weight, fc1_bias, fc2_weight, fc2_bias, d_model: d, d_ff };

            blocks.push(VitBlock {
                ln1_gamma, ln1_beta, attn,
                ln2_gamma, ln2_beta, mlp,
                d_model: d,
            });
        }

        // ── Optional pre-LN (OpenCLIP has it, vanilla ViT doesn't) ──
        let ln_pre_key = format!("{prefix}ln_pre.weight");
        let (ln_pre_gamma, ln_pre_beta) = if tensors.tensor(&ln_pre_key).is_ok() {
            (
                Some(load_tensor(tensors, &ln_pre_key, &[d])?),
                Some(load_tensor(tensors, &format!("{prefix}ln_pre.bias"), &[d])?),
            )
        } else {
            (None, None)
        };

        // ── Final layer norm ──
        let ln_post_gamma = load_tensor(tensors, &format!("{prefix}ln_post.weight"), &[d])?;
        let ln_post_beta = load_tensor(tensors, &format!("{prefix}ln_post.bias"), &[d])?;

        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            ln_pre_gamma,
            ln_pre_beta,
            blocks,
            ln_post_gamma,
            ln_post_beta,
            config,
        })
    }
}

impl<B: MathBackend> Module<B> for Vit<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters());
        params.push(&self.cls_token);
        params.push(&self.pos_embed);
        if let Some(ref g) = self.ln_pre_gamma { params.push(g); }
        if let Some(ref b) = self.ln_pre_beta { params.push(b); }
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

    #[cfg(feature = "safetensors")]
    mod safetensors_tests {
        use super::*;
        use std::borrow::Cow;
        use std::collections::HashMap;

        struct F32View {
            data: Vec<u8>,
            shape: Vec<usize>,
        }

        impl F32View {
            fn new(values: &[f32], shape: &[usize]) -> Self {
                let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Self { data, shape: shape.to_vec() }
            }
        }

        impl safetensors::View for F32View {
            fn dtype(&self) -> safetensors::Dtype { safetensors::Dtype::F32 }
            fn shape(&self) -> &[usize] { &self.shape }
            fn data(&self) -> Cow<'_, [u8]> { Cow::Borrowed(&self.data) }
            fn data_len(&self) -> usize { self.data.len() }
        }

        /// Serialize a Vit into safetensors bytes using OpenAI CLIP naming.
        fn serialize_vit(
            model: &Vit<CpuBackend>,
            config: &VitConfig,
            prefix: &str,
        ) -> Vec<u8> {
            let d = config.embed_dim;
            let d_ff = config.d_ff();
            let mut entries: Vec<(String, F32View)> = Vec::new();

            // Patch embed: proj [d, C*P*P] — same flat layout as conv1.weight [d, C, P, P]
            entries.push((
                format!("{prefix}conv1.weight"),
                F32View::new(&model.patch_embed.proj.to_vec(), model.patch_embed.proj.shape.dims()),
            ));

            // CLS token: our [1, d] → CLIP stores as [d]
            entries.push((
                format!("{prefix}class_embedding"),
                F32View::new(&model.cls_token.to_vec(), &[d]),
            ));

            // Positional embedding
            entries.push((
                format!("{prefix}positional_embedding"),
                F32View::new(&model.pos_embed.to_vec(), model.pos_embed.shape.dims()),
            ));

            // Blocks
            for (i, block) in model.blocks.iter().enumerate() {
                let bp = format!("{prefix}transformer.resblocks.{i}");

                entries.push((format!("{bp}.ln_1.weight"), F32View::new(&block.ln1_gamma.to_vec(), &[d])));
                entries.push((format!("{bp}.ln_1.bias"), F32View::new(&block.ln1_beta.to_vec(), &[d])));

                // qkv_weight: our [d, 3d] → CLIP in_proj_weight [3d, d] (transpose)
                let qkv_t = crate::checkpoint::transpose_2d(&block.attn.qkv_weight.to_vec(), d, 3 * d);
                entries.push((format!("{bp}.attn.in_proj_weight"), F32View::new(&qkv_t, &[3 * d, d])));
                entries.push((format!("{bp}.attn.in_proj_bias"), F32View::new(&block.attn.qkv_bias.to_vec(), &[3 * d])));

                // proj_weight: our [d, d] → CLIP out_proj.weight [d, d] (transpose)
                let proj_t = crate::checkpoint::transpose_2d(&block.attn.proj_weight.to_vec(), d, d);
                entries.push((format!("{bp}.attn.out_proj.weight"), F32View::new(&proj_t, &[d, d])));
                entries.push((format!("{bp}.attn.out_proj.bias"), F32View::new(&block.attn.proj_bias.to_vec(), &[d])));

                entries.push((format!("{bp}.ln_2.weight"), F32View::new(&block.ln2_gamma.to_vec(), &[d])));
                entries.push((format!("{bp}.ln_2.bias"), F32View::new(&block.ln2_beta.to_vec(), &[d])));

                // fc1: our [d, d_ff] → CLIP c_fc.weight [d_ff, d] (transpose)
                let fc1_t = crate::checkpoint::transpose_2d(&block.mlp.fc1_weight.to_vec(), d, d_ff);
                entries.push((format!("{bp}.mlp.c_fc.weight"), F32View::new(&fc1_t, &[d_ff, d])));
                entries.push((format!("{bp}.mlp.c_fc.bias"), F32View::new(&block.mlp.fc1_bias.to_vec(), &[d_ff])));

                // fc2: our [d_ff, d] → CLIP c_proj.weight [d, d_ff] (transpose)
                let fc2_t = crate::checkpoint::transpose_2d(&block.mlp.fc2_weight.to_vec(), d_ff, d);
                entries.push((format!("{bp}.mlp.c_proj.weight"), F32View::new(&fc2_t, &[d, d_ff])));
                entries.push((format!("{bp}.mlp.c_proj.bias"), F32View::new(&block.mlp.fc2_bias.to_vec(), &[d])));
            }

            // LN post
            entries.push((format!("{prefix}ln_post.weight"), F32View::new(&model.ln_post_gamma.to_vec(), &[d])));
            entries.push((format!("{prefix}ln_post.bias"), F32View::new(&model.ln_post_beta.to_vec(), &[d])));

            let info: Option<HashMap<String, String>> = None;
            safetensors::serialize(entries, &info).unwrap()
        }

        fn tensors_equal(a: &Tensor<CpuBackend>, b: &Tensor<CpuBackend>, name: &str) {
            assert_eq!(a.shape.dims(), b.shape.dims(), "shape mismatch for {name}");
            for (i, (x, y)) in a.to_vec().iter().zip(b.to_vec().iter()).enumerate() {
                assert!((x - y).abs() < 1e-5, "{name}[{i}]: {x} vs {y}");
            }
        }

        #[test]
        fn roundtrip_vit_small() {
            let config = VitConfig {
                image_size: 32,
                patch_size: 16,
                embed_dim: 64,
                num_heads: 4,
                num_layers: 2,
                mlp_ratio: 4.0,
                in_channels: 3,
            };
            let original = Vit::<CpuBackend>::new(config.clone());

            let bytes = serialize_vit(&original, &config, "");
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let loaded = Vit::<CpuBackend>::from_safetensors(config, &tensors, "").unwrap();

            // Compare key tensors
            tensors_equal(&original.patch_embed.proj, &loaded.patch_embed.proj, "patch_embed.proj");
            tensors_equal(&original.cls_token, &loaded.cls_token, "cls_token");
            tensors_equal(&original.pos_embed, &loaded.pos_embed, "pos_embed");
            tensors_equal(&original.ln_post_gamma, &loaded.ln_post_gamma, "ln_post.gamma");
            tensors_equal(&original.ln_post_beta, &loaded.ln_post_beta, "ln_post.beta");

            for (i, (a, b)) in original.blocks.iter().zip(loaded.blocks.iter()).enumerate() {
                tensors_equal(&a.attn.qkv_weight, &b.attn.qkv_weight, &format!("block{i}.qkv_w"));
                tensors_equal(&a.attn.proj_weight, &b.attn.proj_weight, &format!("block{i}.proj_w"));
                tensors_equal(&a.mlp.fc1_weight, &b.mlp.fc1_weight, &format!("block{i}.fc1_w"));
                tensors_equal(&a.mlp.fc2_weight, &b.mlp.fc2_weight, &format!("block{i}.fc2_w"));
                tensors_equal(&a.ln1_gamma, &b.ln1_gamma, &format!("block{i}.ln1_g"));
                tensors_equal(&a.ln2_gamma, &b.ln2_gamma, &format!("block{i}.ln2_g"));
            }
        }

        #[test]
        fn roundtrip_vit_with_prefix() {
            // Simulate CLIP visual.* prefix
            let config = VitConfig {
                image_size: 32,
                patch_size: 16,
                embed_dim: 64,
                num_heads: 4,
                num_layers: 1,
                mlp_ratio: 4.0,
                in_channels: 3,
            };
            let original = Vit::<CpuBackend>::new(config.clone());

            let bytes = serialize_vit(&original, &config, "visual.");
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let loaded = Vit::<CpuBackend>::from_safetensors(config, &tensors, "visual.").unwrap();

            tensors_equal(&original.patch_embed.proj, &loaded.patch_embed.proj, "proj");
            tensors_equal(&original.cls_token, &loaded.cls_token, "cls");
            tensors_equal(&original.blocks[0].attn.qkv_weight, &loaded.blocks[0].attn.qkv_weight, "qkv");
        }
    }
}
