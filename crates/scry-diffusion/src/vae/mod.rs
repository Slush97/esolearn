// SPDX-License-Identifier: MIT OR Apache-2.0
//! VAE — encode RGB ↔ 4-channel latent space.
//!
//! For txt2img (M9) we only need the **decoder**: latent `[batch, 4, H/8, W/8]`
//! → RGB `[batch, 3, H, W]`. The encoder is needed for img2img / inpainting
//! later (M10+) but is structurally similar (mirror network).
//!
//! Architecture (decoder, SD 1.5):
//! ```text
//!     latent [4, H/8, W/8]
//!         │
//!     post_quant_conv (1×1)
//!         │
//!     conv_in (3×3, 512 ch)
//!         │
//!     mid_block: ResBlock → spatial_attn(self only) → ResBlock
//!         │
//!     up_blocks[0..3]: 3× ResBlock + (no attn) + 2× nearest upsample
//!         │
//!     conv_norm_out (GroupNorm) → SiLU → conv_out (3×3, 3 ch)
//!         │
//!     image [3, H, W] in (-1, 1)
//! ```
//!
//! No cross-attention (no text conditioning); the mid-block has a
//! self-attention layer. Reuses [`crate::unet::resblock::ResBlock`] but with
//! `time_embed_dim = 0` (the VAE doesn't take a timestep).

pub mod decoder;

pub use decoder::VaeDecoder;
