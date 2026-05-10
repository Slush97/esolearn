// SPDX-License-Identifier: MIT OR Apache-2.0
//! CLIP-ViT-L/14 text encoder (SD 1.5 / 2.x).
//!
//! Architecture: token embedding + learned positional embedding → 12
//! transformer blocks (causal mask, pre-LN, QuickGELU MLP) → final
//! LayerNorm. Output is the per-token embedding `[77, 768]`. SD 1.5
//! conditions the UNet on the full sequence, not the pooled embedding.
//!
//! Differences from `scry_vision::models::vit::Vit`:
//!
//! - Causal attention mask — every position attends only to itself and
//!   earlier positions. ViT is bidirectional.
//! - Token embeddings instead of patch embeddings (input is `[seq]` of
//!   token IDs, not `[3, H, W]`).
//! - Learned positional embeddings (`[77, 768]` table looked up per
//!   position; ViT also has learned embeddings — same mechanism).
//! - QuickGELU MLP (`x * sigmoid(1.702 * x)`) instead of exact GELU —
//!   needed for 1e-4 numerical match against HF.
//! - No CLS token, no projection head — SD wants the token-level output,
//!   not a pooled vector.
//!
//! HF SD 1.5 ships the text encoder in `text_encoder/model.safetensors`
//! with 197 keys total — 2 embeddings (`token_embedding` and
//! `position_embedding`), 1 unused buffer (`position_ids`), 12 layers ×
//! 16 keys per layer, and 2 final-LN keys. The `from_safetensors`
//! loader consumes 196 of those (skipping `position_ids`) and asserts
//! no extra keys, so a missed mapping fails loudly at load time rather
//! than silently zero-initializing a tensor.

use scry_llm::backend::MathBackend;
use scry_llm::nn::attention::CausalSelfAttention;
use scry_llm::nn::layernorm::LayerNormModule;
use scry_llm::ops;
use scry_llm::tensor::Tensor;

use super::TextEncoder;
use crate::conditioning::Conditioning;
use crate::error::{Error, Result};
use crate::ops::quick_gelu;

/// CLIP-ViT-L/14 text encoder configuration.
#[derive(Debug, Clone)]
pub struct ClipTextConfig {
    /// Vocabulary size (49,408 for CLIP).
    pub vocab_size: usize,
    /// Maximum sequence length (77 for CLIP).
    pub max_seq_len: usize,
    /// Embedding dimension (768 for CLIP-L, 1024 for OpenCLIP-H, 1280 for
    /// OpenCLIP-bigG).
    pub d_model: usize,
    /// Number of transformer layers (12 for CLIP-L, 23 for OpenCLIP-H,
    /// 32 for OpenCLIP-bigG).
    pub num_layers: usize,
    /// Number of attention heads (12 for CLIP-L, 16 for OpenCLIP-H, 20 for
    /// OpenCLIP-bigG).
    pub num_heads: usize,
    /// MLP hidden multiplier (4× — CLIP convention).
    pub mlp_ratio: f32,
}

impl ClipTextConfig {
    /// CLIP-ViT-L/14 (SD 1.5 / 2.x).
    pub fn clip_vit_l() -> Self {
        Self {
            vocab_size: 49_408,
            max_seq_len: 77,
            d_model: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
        }
    }

    /// OpenCLIP-ViT-H/14 (SD 2.x alt).
    pub fn open_clip_h() -> Self {
        Self {
            d_model: 1024,
            num_layers: 23,
            num_heads: 16,
            ..Self::clip_vit_l()
        }
    }

    /// OpenCLIP-ViT-bigG/14 (SDXL).
    pub fn open_clip_big_g() -> Self {
        Self {
            d_model: 1280,
            num_layers: 32,
            num_heads: 20,
            ..Self::clip_vit_l()
        }
    }

    /// MLP hidden width = `d_model * mlp_ratio`. 3072 for CLIP-L.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn d_ff(&self) -> usize {
        (self.d_model as f32 * self.mlp_ratio) as usize
    }
}

/// One CLIP transformer block. Reuses scry-llm's `CausalSelfAttention` and
/// `LayerNormModule` directly, but holds its own MLP weights since CLIP
/// uses QuickGELU rather than the exact GELU baked into `scry_llm::nn::Mlp`.
struct ClipBlock<B: MathBackend> {
    ln1: LayerNormModule<B>,
    attn: CausalSelfAttention<B>,
    ln2: LayerNormModule<B>,
    /// MLP `fc1.weight`, scry-llm Linear convention `[d_model, d_ff]`.
    fc1_weight: Tensor<B>,
    fc1_bias: Tensor<B>,
    /// MLP `fc2.weight`, `[d_ff, d_model]`.
    fc2_weight: Tensor<B>,
    fc2_bias: Tensor<B>,
}

/// CLIP-ViT-L/14 text encoder.
pub struct ClipTextEncoder<B: MathBackend> {
    /// Architecture configuration.
    pub config: ClipTextConfig,
    /// Token embedding table, `[vocab_size, d_model]`.
    token_embed: Tensor<B>,
    /// Learned absolute positional embedding, `[max_seq_len, d_model]`.
    pos_embed: Tensor<B>,
    /// Stacked transformer blocks (12 for CLIP-L).
    blocks: Vec<ClipBlock<B>>,
    /// LayerNorm applied after the last block.
    final_ln: LayerNormModule<B>,
}

impl<B: MathBackend> ClipBlock<B> {
    fn to_device(&mut self) {
        self.ln1.to_device();
        self.attn.to_device();
        self.ln2.to_device();
        B::to_device_in_place(&mut self.fc1_weight.data);
        B::to_device_in_place(&mut self.fc1_bias.data);
        B::to_device_in_place(&mut self.fc2_weight.data);
        B::to_device_in_place(&mut self.fc2_bias.data);
    }
}

impl<B: MathBackend> ClipTextEncoder<B> {
    /// Pre-upload every parameter tensor in the CLIP text encoder to the
    /// backend's device-resident form. No-op on `CpuBackend`; idempotent on
    /// any backend.
    pub fn to_device(&mut self) {
        B::to_device_in_place(&mut self.token_embed.data);
        B::to_device_in_place(&mut self.pos_embed.data);
        for b in &mut self.blocks {
            b.to_device();
        }
        self.final_ln.to_device();
    }

    /// Output embedding dimension. Convenience accessor.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Number of transformer layers. Convenience accessor.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Forward pass on a sequence of `seq_len ≤ max_seq_len` token IDs.
    ///
    /// Returns `[seq_len, d_model]`. The pipeline always pads to
    /// `max_seq_len = 77`, so the typical shape here is `[77, 768]`.
    pub fn forward(&self, tokens: &[u32]) -> Result<Tensor<B>> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err(Error::Llm("clip text encoder: empty token sequence".into()));
        }
        if seq_len > self.config.max_seq_len {
            return Err(Error::Llm(format!(
                "clip text encoder: seq_len={seq_len} exceeds max_seq_len={}",
                self.config.max_seq_len
            )));
        }

        // Token embedding + positional embedding. `B::embedding` takes
        // `&[usize]`, so widen the u32 input.
        let token_ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        let pos_ids: Vec<usize> = (0..seq_len).collect();
        let tok = ops::embedding(
            &self.token_embed,
            &token_ids,
            self.config.vocab_size,
            self.config.d_model,
        );
        let pos = ops::embedding(
            &self.pos_embed,
            &pos_ids,
            self.config.max_seq_len,
            self.config.d_model,
        );
        let mut x = ops::add(&tok, &pos);

        for block in &self.blocks {
            x = forward_block(block, &x, self.config.d_ff(), self.config.d_model);
        }

        Ok(self.final_ln.forward(&x))
    }
}

fn forward_block<B: MathBackend>(
    block: &ClipBlock<B>,
    x: &Tensor<B>,
    d_ff: usize,
    d_model: usize,
) -> Tensor<B> {
    let seq_len = x.shape.dims()[0];

    // Attention sub-block: LN -> causal self-attn -> +residual.
    let h = block.ln1.forward(x);
    let attn_out = block.attn.forward(&h);
    let x1 = ops::add(x, &attn_out);

    // MLP sub-block: LN -> fc1 -> QuickGELU -> fc2 -> +residual.
    let h = block.ln2.forward(&x1);
    let h = ops::matmul_bias(
        &h,
        &block.fc1_weight,
        &block.fc1_bias,
        seq_len,
        d_model,
        d_ff,
        false,
        false,
    );
    let h = quick_gelu(&h);
    let h = ops::matmul_bias(
        &h,
        &block.fc2_weight,
        &block.fc2_bias,
        seq_len,
        d_ff,
        d_model,
        false,
        false,
    );
    ops::add(&x1, &h)
}

impl<B: MathBackend> TextEncoder<B> for ClipTextEncoder<B> {
    fn encode(&mut self, tokens: &[u32]) -> Result<Conditioning<B>> {
        let embeddings = self.forward(tokens)?;
        Ok(Conditioning {
            embeddings,
            extras: None,
        })
    }

    fn d_model(&self) -> usize {
        self.config.d_model
    }

    fn to_device(&mut self) {
        ClipTextEncoder::to_device(self);
    }
}

// -----------------------------------------------------------------------
// Safetensors loader (HF `text_model.*` -> our struct).
// -----------------------------------------------------------------------

/// HF text encoder weight loader.
///
/// Consumes every key in `text_encoder/model.safetensors` except the
/// unused `text_model.embeddings.position_ids` int buffer (PyTorch
/// bookkeeping; we materialize positions from `0..seq_len` at forward
/// time). A missed key triggers a load failure rather than a silent
/// zero-init.
#[cfg(feature = "safetensors")]
use scry_llm::tensor::shape::Shape;

#[cfg(feature = "safetensors")]
impl<B: MathBackend> ClipTextEncoder<B> {
    /// Load CLIP text encoder weights from a HF safetensors checkpoint.
    ///
    /// Each block load is unrolled inline so a missed key surfaces with
    /// the offending HF tensor name in the error message — splitting
    /// into helpers buys nothing and obscures the error context.
    #[allow(clippy::too_many_lines)]
    pub fn from_safetensors(
        config: ClipTextConfig,
        ckpt: &crate::weights::SafetensorsCheckpoint,
    ) -> Result<Self> {
        use scry_vision::checkpoint::{load_f32, load_tensor};

        let view = ckpt.tensors()?;
        let d = config.d_model;
        let d_ff = config.d_ff();
        let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut consume = |key: &str| {
            consumed.insert(key.to_string());
        };

        // ---- Embeddings ----
        let token_embed_key = "text_model.embeddings.token_embedding.weight";
        let token_embed = load_tensor::<B>(&view, token_embed_key, &[config.vocab_size, d])
            .map_err(|e| Error::Llm(format!("load {token_embed_key}: {e}")))?;
        consume(token_embed_key);

        let pos_embed_key = "text_model.embeddings.position_embedding.weight";
        let pos_embed = load_tensor::<B>(&view, pos_embed_key, &[config.max_seq_len, d])
            .map_err(|e| Error::Llm(format!("load {pos_embed_key}: {e}")))?;
        consume(pos_embed_key);

        // PyTorch ships a literal [0..max_seq_len] int buffer alongside
        // the embedding to feed into nn.Embedding. We don't need it —
        // `forward` materializes the positions on the fly.
        consume("text_model.embeddings.position_ids");

        // ---- Transformer blocks ----
        let mut blocks = Vec::with_capacity(config.num_layers);
        for layer_i in 0..config.num_layers {
            let prefix = format!("text_model.encoder.layers.{layer_i}");

            // LN1.
            let ln1_w_key = format!("{prefix}.layer_norm1.weight");
            let ln1_b_key = format!("{prefix}.layer_norm1.bias");
            let ln1 = LayerNormModule {
                gamma: load_tensor::<B>(&view, &ln1_w_key, &[d])
                    .map_err(|e| Error::Llm(format!("load {ln1_w_key}: {e}")))?,
                beta: load_tensor::<B>(&view, &ln1_b_key, &[d])
                    .map_err(|e| Error::Llm(format!("load {ln1_b_key}: {e}")))?,
                eps: 1e-5,
            };
            consume(&ln1_w_key);
            consume(&ln1_b_key);

            // Attention: HF stores Q/K/V as three separate `[d, d]`
            // matrices (PyTorch [out, in] convention). scry-llm's
            // `CausalSelfAttention` wants a fused `qkv_weight: [d, 3*d]`
            // in [in, out] convention, so we both transpose and concat
            // in one pass.
            let q_w_key = format!("{prefix}.self_attn.q_proj.weight");
            let k_w_key = format!("{prefix}.self_attn.k_proj.weight");
            let v_w_key = format!("{prefix}.self_attn.v_proj.weight");
            let q_w_hf = load_f32(&view, &q_w_key)
                .map_err(|e| Error::Llm(format!("load {q_w_key}: {e}")))?;
            let k_w_hf = load_f32(&view, &k_w_key)
                .map_err(|e| Error::Llm(format!("load {k_w_key}: {e}")))?;
            let v_w_hf = load_f32(&view, &v_w_key)
                .map_err(|e| Error::Llm(format!("load {v_w_key}: {e}")))?;
            for (key, raw) in [
                (&q_w_key, &q_w_hf),
                (&k_w_key, &k_w_hf),
                (&v_w_key, &v_w_hf),
            ] {
                if raw.len() != d * d {
                    return Err(Error::Llm(format!(
                        "{key}: expected {} elements, got {}",
                        d * d,
                        raw.len()
                    )));
                }
            }
            let mut qkv = vec![0.0f32; d * 3 * d];
            for in_i in 0..d {
                for out_i in 0..d {
                    qkv[in_i * (3 * d) + out_i] = q_w_hf[out_i * d + in_i];
                    qkv[in_i * (3 * d) + d + out_i] = k_w_hf[out_i * d + in_i];
                    qkv[in_i * (3 * d) + 2 * d + out_i] = v_w_hf[out_i * d + in_i];
                }
            }
            consume(&q_w_key);
            consume(&k_w_key);
            consume(&v_w_key);

            let q_b_key = format!("{prefix}.self_attn.q_proj.bias");
            let k_b_key = format!("{prefix}.self_attn.k_proj.bias");
            let v_b_key = format!("{prefix}.self_attn.v_proj.bias");
            let mut qkv_b = Vec::with_capacity(3 * d);
            qkv_b.extend(
                load_f32(&view, &q_b_key)
                    .map_err(|e| Error::Llm(format!("load {q_b_key}: {e}")))?,
            );
            qkv_b.extend(
                load_f32(&view, &k_b_key)
                    .map_err(|e| Error::Llm(format!("load {k_b_key}: {e}")))?,
            );
            qkv_b.extend(
                load_f32(&view, &v_b_key)
                    .map_err(|e| Error::Llm(format!("load {v_b_key}: {e}")))?,
            );
            if qkv_b.len() != 3 * d {
                return Err(Error::Llm(format!(
                    "{prefix} qkv biases concat: expected {}, got {}",
                    3 * d,
                    qkv_b.len()
                )));
            }
            consume(&q_b_key);
            consume(&k_b_key);
            consume(&v_b_key);

            // Output projection: [d, d] HF [out, in] -> [in, out] for us.
            let o_w_key = format!("{prefix}.self_attn.out_proj.weight");
            let o_w_hf = load_f32(&view, &o_w_key)
                .map_err(|e| Error::Llm(format!("load {o_w_key}: {e}")))?;
            if o_w_hf.len() != d * d {
                return Err(Error::Llm(format!(
                    "{o_w_key}: expected {} elements, got {}",
                    d * d,
                    o_w_hf.len()
                )));
            }
            let mut proj_w = vec![0.0f32; d * d];
            for in_i in 0..d {
                for out_i in 0..d {
                    proj_w[in_i * d + out_i] = o_w_hf[out_i * d + in_i];
                }
            }
            consume(&o_w_key);
            let o_b_key = format!("{prefix}.self_attn.out_proj.bias");
            let proj_b = load_f32(&view, &o_b_key)
                .map_err(|e| Error::Llm(format!("load {o_b_key}: {e}")))?;
            consume(&o_b_key);

            let attn = CausalSelfAttention {
                qkv_weight: Tensor::from_vec(qkv, Shape::new(&[d, 3 * d])),
                qkv_bias: Tensor::from_vec(qkv_b, Shape::new(&[3 * d])),
                proj_weight: Tensor::from_vec(proj_w, Shape::new(&[d, d])),
                proj_bias: Tensor::from_vec(proj_b, Shape::new(&[d])),
                n_heads: config.num_heads,
                d_model: d,
                d_head: d / config.num_heads,
            };

            // LN2.
            let ln2_w_key = format!("{prefix}.layer_norm2.weight");
            let ln2_b_key = format!("{prefix}.layer_norm2.bias");
            let ln2 = LayerNormModule {
                gamma: load_tensor::<B>(&view, &ln2_w_key, &[d])
                    .map_err(|e| Error::Llm(format!("load {ln2_w_key}: {e}")))?,
                beta: load_tensor::<B>(&view, &ln2_b_key, &[d])
                    .map_err(|e| Error::Llm(format!("load {ln2_b_key}: {e}")))?,
                eps: 1e-5,
            };
            consume(&ln2_w_key);
            consume(&ln2_b_key);

            // MLP: HF stores [out, in], we want [in, out].
            let fc1_w_key = format!("{prefix}.mlp.fc1.weight");
            let fc1_b_key = format!("{prefix}.mlp.fc1.bias");
            let fc1_w_hf = load_f32(&view, &fc1_w_key)
                .map_err(|e| Error::Llm(format!("load {fc1_w_key}: {e}")))?;
            if fc1_w_hf.len() != d_ff * d {
                return Err(Error::Llm(format!(
                    "{fc1_w_key}: expected {} elements, got {}",
                    d_ff * d,
                    fc1_w_hf.len()
                )));
            }
            let mut fc1_w = vec![0.0f32; d * d_ff];
            for in_i in 0..d {
                for out_i in 0..d_ff {
                    fc1_w[in_i * d_ff + out_i] = fc1_w_hf[out_i * d + in_i];
                }
            }
            let fc1_weight = Tensor::from_vec(fc1_w, Shape::new(&[d, d_ff]));
            let fc1_bias = load_tensor::<B>(&view, &fc1_b_key, &[d_ff])
                .map_err(|e| Error::Llm(format!("load {fc1_b_key}: {e}")))?;
            consume(&fc1_w_key);
            consume(&fc1_b_key);

            let fc2_w_key = format!("{prefix}.mlp.fc2.weight");
            let fc2_b_key = format!("{prefix}.mlp.fc2.bias");
            let fc2_w_hf = load_f32(&view, &fc2_w_key)
                .map_err(|e| Error::Llm(format!("load {fc2_w_key}: {e}")))?;
            if fc2_w_hf.len() != d * d_ff {
                return Err(Error::Llm(format!(
                    "{fc2_w_key}: expected {} elements, got {}",
                    d * d_ff,
                    fc2_w_hf.len()
                )));
            }
            let mut fc2_w = vec![0.0f32; d_ff * d];
            for in_i in 0..d_ff {
                for out_i in 0..d {
                    fc2_w[in_i * d + out_i] = fc2_w_hf[out_i * d_ff + in_i];
                }
            }
            let fc2_weight = Tensor::from_vec(fc2_w, Shape::new(&[d_ff, d]));
            let fc2_bias = load_tensor::<B>(&view, &fc2_b_key, &[d])
                .map_err(|e| Error::Llm(format!("load {fc2_b_key}: {e}")))?;
            consume(&fc2_w_key);
            consume(&fc2_b_key);

            blocks.push(ClipBlock {
                ln1,
                attn,
                ln2,
                fc1_weight,
                fc1_bias,
                fc2_weight,
                fc2_bias,
            });
        }

        // ---- Final LN ----
        let final_ln_w_key = "text_model.final_layer_norm.weight";
        let final_ln_b_key = "text_model.final_layer_norm.bias";
        let final_ln = LayerNormModule {
            gamma: load_tensor::<B>(&view, final_ln_w_key, &[d])
                .map_err(|e| Error::Llm(format!("load {final_ln_w_key}: {e}")))?,
            beta: load_tensor::<B>(&view, final_ln_b_key, &[d])
                .map_err(|e| Error::Llm(format!("load {final_ln_b_key}: {e}")))?,
            eps: 1e-5,
        };
        consume(final_ln_w_key);
        consume(final_ln_b_key);

        // ---- Verify 100% key consumption ----
        let all: std::collections::HashSet<String> = view.names().into_iter().cloned().collect();
        let missing: Vec<String> = all.difference(&consumed).cloned().collect();
        if !missing.is_empty() {
            let mut sorted = missing;
            sorted.sort();
            return Err(Error::Llm(format!(
                "clip text encoder: {} HF keys not consumed by loader: {}{}",
                sorted.len(),
                sorted
                    .iter()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                if sorted.len() > 8 { ", ..." } else { "" }
            )));
        }

        Ok(Self {
            config,
            token_embed,
            pos_embed,
            blocks,
            final_ln,
        })
    }
}
