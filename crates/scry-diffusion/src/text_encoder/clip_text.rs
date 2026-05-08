// SPDX-License-Identifier: MIT OR Apache-2.0
//! CLIP-ViT-L/14 text encoder (SD 1.5 / 2.x).
//!
//! Architecture: token embedding + positional embedding → 12 transformer
//! blocks (causal mask, GELU activation, pre-LN) → final LayerNorm. Output
//! is the per-token embedding `[77, 768]`. SD 1.5 conditions the UNet on
//! the full sequence, not the pooled embedding.
//!
//! Differences from `scry_vision::models::vit::Vit`:
//!   - **Causal attention mask** — every position can only attend to itself
//!     and earlier positions. ViT is bidirectional.
//!   - **Token embeddings** instead of patch embeddings (input is `[seq]`
//!     of token IDs, not `[3, H, W]`).
//!   - **Learned positional embeddings** (`[77, 768]` table looked up per
//!     position; ViT also has learned embeddings — same mechanism).
//!   - **No CLS token / no projection head** — SD wants the token-level
//!     output, not a pooled vector.
//!
//! The transformer block itself (LN → MHA → residual → LN → MLP → residual)
//! reuses `scry_llm`'s `transformer` module if it has a causal-mask path,
//! or scry-vision's `VitBlock` adapted for the causal mask.

use scry_llm::backend::MathBackend;

use super::TextEncoder;
use crate::conditioning::Conditioning;
use crate::error::Result;

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
}

/// CLIP text encoder.
pub struct ClipTextEncoder<B: MathBackend> {
    /// Architecture configuration.
    pub config: ClipTextConfig,
    // M3: token embedding table, positional embedding, transformer blocks,
    // final LN. See module doc for layout.
    _backend: std::marker::PhantomData<B>,
}

impl<B: MathBackend> ClipTextEncoder<B> {
    /// Construct an encoder with zero-initialized weights. Real weights are
    /// applied via [`Self::load_safetensors`] (M3).
    pub fn new(config: ClipTextConfig) -> Self {
        Self {
            config,
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: MathBackend> TextEncoder<B> for ClipTextEncoder<B> {
    fn encode(&mut self, tokens: &[u32]) -> Result<Conditioning<B>> {
        let _ = tokens;
        todo!("M3: token embed + positional embed + N transformer blocks (causal) + final LN")
    }

    fn d_model(&self) -> usize {
        self.config.d_model
    }
}
