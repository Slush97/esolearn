// SPDX-License-Identifier: MIT OR Apache-2.0
//! Spatial transformer block — self-attention + cross-attention + GeGLU MLP.
//!
//! Each cross-attention block in SD's UNet is a "spatial transformer" stack:
//!
//! ```text
//!     in (NCHW) ──► reshape to (N, H*W, C) ──┐
//!                                             │
//!     ┌─ LayerNorm → self-attn(Q=K=V=x) ─ + ─┤        (image attends to itself)
//!     │                                       │
//!     ├─ LayerNorm → cross-attn(Q=x, K/V=ctx) ┤        (image attends to text)
//!     │                                       │
//!     └─ LayerNorm → GeGLU MLP ─────────── + ─┘
//!                                             │
//!     reshape back to (N, C, H, W)  ◄─────────┘
//! ```
//!
//! - **Self-attention** is just `scry-vision`'s ViT attention used at a
//!   different shape (image tokens instead of patches).
//! - **Cross-attention** has Q from the latent and K/V from the conditioning
//!   embeddings — the only shape change vs self-attention is that K/V come
//!   from a different tensor with a different sequence length.
//! - **GeGLU** = chunk dim in two, gate one half by GELU(other half) — needs
//!   a small MathBackend helper or two existing matmul + GELU calls.
//!
//! `transformer_layers_per_block` (from `UnetConfig`) controls how many of
//! these stacks are nested per attention block. SD 1.5 = 1; SDXL deepest
//! stage = 10 (the bulk of SDXL's parameter count lives here).
//!
//! HF naming convention (mirrors `diffusers/models/attention_processor.py`
//! and `attention.py::Transformer2DModel`):
//!
//! - `proj_in` / `proj_out`: 1×1 Conv2d (SD 1.5 uses `use_linear_projection=False`).
//! - `attn1` is self-attention with `to_q`, `to_k`, `to_v` (no bias) and
//!   `to_out.0` (with bias). `attn2` is cross-attention with the same
//!   layout but K/V projecting from the conditioning dim.
//! - `ff.net.0.proj` is the GEGLU first linear (`d -> 2*d_ff`). `ff.net.1`
//!   is dropout (no params, ignored at inference). `ff.net.2` is the
//!   second linear (`d_ff -> d`).

use scry_llm::backend::MathBackend;
use scry_llm::nn::layernorm::LayerNormModule;
use scry_llm::tensor::Tensor;
use scry_vision::nn::conv2d::Conv2d;

use super::common::GroupNormParams;
use crate::conditioning::Conditioning;
use crate::error::Result;

/// Self- or cross-attention layer. SD's UNet uses the same `Attention`
/// shape for both `attn1` (self) and `attn2` (cross); the only difference
/// is whether K/V project from `d_model` (self) or `cross_dim` (cross).
///
/// Q/K/V have no bias in SD 1.5; the output projection (`to_out.0`) does.
/// Weights are stored in scry-llm `[in, out]` convention.
pub(crate) struct Attention<B: MathBackend> {
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) d_model: usize,
    pub(crate) cross_dim: usize,

    /// `[d_model, inner_dim]` where `inner_dim = num_heads * head_dim`.
    pub(crate) q_weight: Tensor<B>,
    /// `[cross_dim, inner_dim]`.
    pub(crate) k_weight: Tensor<B>,
    /// `[cross_dim, inner_dim]`.
    pub(crate) v_weight: Tensor<B>,
    /// `[inner_dim, d_model]`.
    pub(crate) out_weight: Tensor<B>,
    /// `[d_model]`.
    pub(crate) out_bias: Tensor<B>,
}

/// GeGLU feed-forward block (`ff.*` in HF). Two linears with a GELU gate
/// in between: `proj_in` produces `[d_model -> 2*d_ff]`, the result is
/// chunked into `(x, gate)` along the last dim, multiplied by `gelu(gate)`,
/// and projected back to `d_model` via `proj_out`.
pub(crate) struct GeGluFf<B: MathBackend> {
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,

    /// `proj_in`: `[d_model, 2*d_ff]`, scry-llm `[in, out]` convention.
    pub(crate) proj_in_weight: Tensor<B>,
    /// `[2*d_ff]`.
    pub(crate) proj_in_bias: Tensor<B>,
    /// `proj_out`: `[d_ff, d_model]`.
    pub(crate) proj_out_weight: Tensor<B>,
    /// `[d_model]`.
    pub(crate) proj_out_bias: Tensor<B>,
}

/// One transformer-2D layer (self + cross + MLP, all with pre-LN).
pub struct BasicTransformerBlock<B: MathBackend> {
    /// Inner channel count.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Cross-attention conditioning dim (768 SD 1.5, 2048 SDXL).
    pub cross_attention_dim: usize,

    pub(crate) norm1: LayerNormModule<B>,
    pub(crate) attn1: Attention<B>,
    pub(crate) norm2: LayerNormModule<B>,
    pub(crate) attn2: Attention<B>,
    pub(crate) norm3: LayerNormModule<B>,
    pub(crate) ff: GeGluFf<B>,
}

impl<B: MathBackend> BasicTransformerBlock<B> {
    /// Forward pass: latent ⊕ self-attn → ⊕ cross-attn(text) → ⊕ MLP.
    pub fn forward(
        &mut self,
        latent: &Tensor<B>,
        conditioning: &Conditioning<B>,
    ) -> Result<Tensor<B>> {
        let _ = (latent, conditioning);
        todo!("M6: pre-LN self-attn, pre-LN cross-attn, pre-LN GeGLU MLP — all residual-added")
    }
}

/// Spatial transformer block — wraps `transformer_layers_per_block` stacks
/// with the NCHW ↔ (N, H*W, C) reshape and a 1×1 conv on entry/exit.
pub struct SpatialTransformer<B: MathBackend> {
    /// Channel count of the feature map entering the block.
    pub channels: usize,
    /// Stacks of `BasicTransformerBlock` to apply (= `transformer_layers_per_block`).
    pub transformer_blocks: Vec<BasicTransformerBlock<B>>,

    pub(crate) norm: GroupNormParams<B>,
    pub(crate) proj_in: Conv2d<B>,
    pub(crate) proj_out: Conv2d<B>,
}

impl<B: MathBackend> SpatialTransformer<B> {
    /// Forward: 1×1 conv in → reshape to seq → N × `BasicTransformerBlock` →
    /// reshape back → 1×1 conv out → residual add.
    pub fn forward(
        &mut self,
        feature_map: &Tensor<B>,
        conditioning: &Conditioning<B>,
    ) -> Result<Tensor<B>> {
        let _ = (feature_map, conditioning);
        todo!("M6: spatial transformer — 1x1 in, NCHW->(N,seq,C), N transformer blocks, reshape, 1x1 out, residual")
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> Attention<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        d_model: usize,
        cross_dim: usize,
        num_heads: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use super::common::load_linear;
        use crate::error::Error;

        if d_model % num_heads != 0 {
            return Err(Error::Llm(format!(
                "{prefix}: d_model={d_model} not divisible by num_heads={num_heads}"
            )));
        }
        let head_dim = d_model / num_heads;
        let inner_dim = num_heads * head_dim;

        // SD attention has bias=False on Q/K/V; bias=True on to_out.0.
        let (q_weight, _) = load_linear::<B>(
            view,
            &format!("{prefix}.to_q"),
            d_model,
            inner_dim,
            false,
            consume,
        )?;
        let (k_weight, _) = load_linear::<B>(
            view,
            &format!("{prefix}.to_k"),
            cross_dim,
            inner_dim,
            false,
            consume,
        )?;
        let (v_weight, _) = load_linear::<B>(
            view,
            &format!("{prefix}.to_v"),
            cross_dim,
            inner_dim,
            false,
            consume,
        )?;
        // HF `to_out` is a `nn.ModuleList([Linear(inner, d), Dropout])`
        // — the linear is at `to_out.0` and the dropout has no params.
        let (out_weight, out_bias) = load_linear::<B>(
            view,
            &format!("{prefix}.to_out.0"),
            inner_dim,
            d_model,
            true,
            consume,
        )?;

        Ok(Self {
            num_heads,
            head_dim,
            d_model,
            cross_dim,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            out_bias,
        })
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> GeGluFf<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        d_model: usize,
        d_ff: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use super::common::load_linear;

        let (proj_in_weight, proj_in_bias) = load_linear::<B>(
            view,
            &format!("{prefix}.net.0.proj"),
            d_model,
            2 * d_ff,
            true,
            consume,
        )?;
        let (proj_out_weight, proj_out_bias) = load_linear::<B>(
            view,
            &format!("{prefix}.net.2"),
            d_ff,
            d_model,
            true,
            consume,
        )?;
        Ok(Self {
            d_model,
            d_ff,
            proj_in_weight,
            proj_in_bias,
            proj_out_weight,
            proj_out_bias,
        })
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> BasicTransformerBlock<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        d_model: usize,
        num_heads: usize,
        cross_attention_dim: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use scry_vision::checkpoint::load_tensor;

        use crate::error::Error;

        // SD 1.5 BasicTransformerBlock follows pre-LN / parallel stack:
        //   norm1 -> attn1(self)        -> +residual
        //   norm2 -> attn2(cross,ctx)   -> +residual
        //   norm3 -> ff(GeGLU)          -> +residual
        let norm1 = LayerNormModule {
            gamma: load_tensor::<B>(view, &format!("{prefix}.norm1.weight"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm1.weight: {e}")))?,
            beta: load_tensor::<B>(view, &format!("{prefix}.norm1.bias"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm1.bias: {e}")))?,
            eps: 1e-5,
        };
        consume(&format!("{prefix}.norm1.weight"));
        consume(&format!("{prefix}.norm1.bias"));

        let attn1 = Attention::from_safetensors(
            view,
            &format!("{prefix}.attn1"),
            d_model,
            d_model,
            num_heads,
            consume,
        )?;

        let norm2 = LayerNormModule {
            gamma: load_tensor::<B>(view, &format!("{prefix}.norm2.weight"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm2.weight: {e}")))?,
            beta: load_tensor::<B>(view, &format!("{prefix}.norm2.bias"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm2.bias: {e}")))?,
            eps: 1e-5,
        };
        consume(&format!("{prefix}.norm2.weight"));
        consume(&format!("{prefix}.norm2.bias"));

        let attn2 = Attention::from_safetensors(
            view,
            &format!("{prefix}.attn2"),
            d_model,
            cross_attention_dim,
            num_heads,
            consume,
        )?;

        let norm3 = LayerNormModule {
            gamma: load_tensor::<B>(view, &format!("{prefix}.norm3.weight"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm3.weight: {e}")))?,
            beta: load_tensor::<B>(view, &format!("{prefix}.norm3.bias"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm3.bias: {e}")))?,
            eps: 1e-5,
        };
        consume(&format!("{prefix}.norm3.weight"));
        consume(&format!("{prefix}.norm3.bias"));

        // SD 1.5 fixes ff_inner_dim = 4 * d_model (FeedForward `mult=4`).
        let d_ff = 4 * d_model;
        let ff = GeGluFf::from_safetensors(view, &format!("{prefix}.ff"), d_model, d_ff, consume)?;

        Ok(Self {
            d_model,
            num_heads,
            cross_attention_dim,
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
        })
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> SpatialTransformer<B> {
    /// Load one `Transformer2DModel` block at `prefix.*` (e.g.
    /// `down_blocks.0.attentions.0`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        channels: usize,
        num_heads: usize,
        cross_attention_dim: usize,
        transformer_layers: usize,
        num_norm_groups: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use scry_vision::checkpoint::load_conv2d_with_bias;

        use super::common::load_group_norm;
        use crate::error::Error;

        // Outer norm + 1×1 in/out projections (SD 1.5 uses Conv2d, not Linear).
        let norm = load_group_norm::<B>(
            view,
            &format!("{prefix}.norm"),
            channels,
            num_norm_groups,
            consume,
        )?;
        let proj_in = load_conv2d_with_bias::<B>(
            view,
            &format!("{prefix}.proj_in"),
            channels,
            channels,
            1,
            1,
            1,
            0,
        )
        .map_err(|e| Error::Llm(format!("load {prefix}.proj_in: {e}")))?;
        consume(&format!("{prefix}.proj_in.weight"));
        consume(&format!("{prefix}.proj_in.bias"));
        let proj_out = load_conv2d_with_bias::<B>(
            view,
            &format!("{prefix}.proj_out"),
            channels,
            channels,
            1,
            1,
            1,
            0,
        )
        .map_err(|e| Error::Llm(format!("load {prefix}.proj_out: {e}")))?;
        consume(&format!("{prefix}.proj_out.weight"));
        consume(&format!("{prefix}.proj_out.bias"));

        let mut transformer_blocks = Vec::with_capacity(transformer_layers);
        for i in 0..transformer_layers {
            transformer_blocks.push(BasicTransformerBlock::from_safetensors(
                view,
                &format!("{prefix}.transformer_blocks.{i}"),
                channels,
                num_heads,
                cross_attention_dim,
                consume,
            )?);
        }

        Ok(Self {
            channels,
            transformer_blocks,
            norm,
            proj_in,
            proj_out,
        })
    }
}
