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
//! Reference: HF `diffusers/src/diffusers/models/attention.py::Transformer2DModel`
//! and `BasicTransformerBlock`.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use crate::conditioning::Conditioning;
use crate::error::Result;

/// One transformer-2D layer (self + cross + MLP, all with pre-LN).
pub struct BasicTransformerBlock<B: MathBackend> {
    /// Inner channel count.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Cross-attention conditioning dim (768 SD 1.5, 2048 SDXL).
    pub cross_attention_dim: usize,
    // M6: norm1 (LN), self_attn, norm2 (LN), cross_attn, norm3 (LN), ff (GeGLU MLP).
    _backend: std::marker::PhantomData<B>,
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
