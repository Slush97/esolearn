// SPDX-License-Identifier: MIT OR Apache-2.0
//! Conditioning fed to the UNet at every denoising step.
//!
//! For SD 1.5 this is just token-level text embeddings. For SDXL it adds a
//! pooled embedding plus size / crop / target-size micro-conditioning that
//! gets folded into the timestep embedding. The SDXL extension is additive:
//! UNet code reads [`Conditioning::embeddings`] always and consults
//! [`Conditioning::extras`] only when the SDXL features are present.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

/// Conditioning passed to UNet attention layers and timestep embeddings.
pub struct Conditioning<B: MathBackend> {
    /// Token-level text embeddings, shape `[seq_len, d_model]`.
    ///
    /// - SD 1.5 / 2.x: `[77, 768]` from CLIP-ViT-L/14.
    /// - SDXL: `[77, 2048]` (CLIP-ViT-L 768 ++ OpenCLIP-ViT-bigG 1280).
    pub embeddings: Tensor<B>,

    /// SDXL-only extras. `None` for SD 1.5 / 2.x.
    pub extras: Option<SdxlExtras<B>>,
}

/// Extra conditioning carried by SDXL: pooled embedding plus
/// micro-conditioning that gets added into the timestep embedding.
pub struct SdxlExtras<B: MathBackend> {
    /// Pooled OpenCLIP-ViT-bigG output, shape `[1280]`.
    pub pooled: Tensor<B>,
    /// Original sample size `(h, w)` before any crop.
    pub original_size: (u32, u32),
    /// Crop offset `(top, left)` of the original sample.
    pub crop_top_left: (u32, u32),
    /// Target output size `(h, w)`.
    pub target_size: (u32, u32),
}
