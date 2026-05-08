// SPDX-License-Identifier: MIT OR Apache-2.0
//! UNet predicting noise (or v-prediction) per denoising step.
//!
//! Topology:
//! ```text
//!     conv_in
//!         │
//!     down_blocks[0..4]  ── DownBlock { 2× ResBlock + optional self+cross-attn + downsample }
//!         │
//!     mid_block          ── ResBlock + self+cross-attn + ResBlock
//!         │
//!     up_blocks[0..4]    ── UpBlock   { 3× ResBlock + optional self+cross-attn + upsample }
//!         │
//!     conv_norm_out (GroupNorm) → SiLU → conv_out
//! ```
//!
//! For SD 1.5: 4 down blocks, 1 mid block, 4 up blocks, channel multiplier
//! `[1, 2, 4, 4]` × `block_out_channels=320` (first block has 320, then 640,
//! 1280, 1280). Cross-attention sits in down blocks 1..3 and up blocks 0..2,
//! with `attention_head_dim = 8` and 8 heads per attention block.
//!
//! For SDXL: same shape but 4 up/down (one is identity), bigger channels
//! (`[1, 2, 4]` × 320), more transformer layers per attention block, and
//! the timestep MLP also takes the SDXL extras (size, crop, target).
//!
//! Submodules:
//! - [`config`] — `UnetConfig` (parameterized so SDXL is a config swap).
//! - [`resblock`] — Residual block (GroupNorm + SiLU + Conv2d sandwich).
//! - [`attention`] — Spatial transformer block (self-attention + cross-attention).
//! - [`blocks`] — DownBlock / MidBlock / UpBlock orchestration.

pub mod attention;
pub mod blocks;
pub mod config;
pub mod resblock;

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use crate::conditioning::Conditioning;
use crate::error::Result;

pub use config::UnetConfig;

/// SD UNet predicting per-step noise (or v-prediction, depending on schedule).
pub struct Unet<B: MathBackend> {
    /// Architecture config.
    pub config: UnetConfig,
    // M5/M6: conv_in, time_embed MLP, down_blocks, mid_block, up_blocks,
    // conv_norm_out, conv_out. Field layout follows HF naming so the
    // safetensors loader maps cleanly.
    _backend: std::marker::PhantomData<B>,
}

impl<B: MathBackend> Unet<B> {
    /// Construct a UNet with zero-initialized weights.
    pub fn new(config: UnetConfig) -> Self {
        Self {
            config,
            _backend: std::marker::PhantomData,
        }
    }

    /// Forward pass: `(noisy_latent, timestep, conditioning) → predicted_noise`.
    ///
    /// `noisy_latent` is `[batch, in_channels, h, w]` — for SD 1.5 latents
    /// are `[batch, 4, height/8, width/8]`. With CFG enabled the pipeline
    /// passes `batch=2` (uncond + cond) and combines outputs after the call.
    pub fn forward(
        &mut self,
        noisy_latent: &Tensor<B>,
        timestep: f32,
        conditioning: &Conditioning<B>,
    ) -> Result<Tensor<B>> {
        let _ = (noisy_latent, timestep, conditioning);
        todo!(
            "M5/M6: conv_in → time_embed → down_blocks → mid_block → up_blocks (with skip \
             concats) → conv_norm_out + SiLU + conv_out"
        )
    }
}
