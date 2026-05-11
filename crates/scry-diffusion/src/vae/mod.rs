// SPDX-License-Identifier: MIT OR Apache-2.0
//! VAE — encode RGB ↔ 4-channel latent space.
//!
//! Two mirror networks share the same residual + mid-block self-attention
//! flavor (see [`blocks`]). The **decoder** maps a latent
//! `[4, H/8, W/8]` → RGB `[3, H, W]`; the **encoder** is the reverse,
//! mapping `[3, H, W]` → `(mean, logvar)` of a diagonal Gaussian latent
//! distribution (and lives behind `cfg(feature = "safetensors")` until
//! its forward pass lands).
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
//! Architecture (encoder, SD 1.5):
//! ```text
//!     image [3, H, W]
//!         │
//!     conv_in (3×3, 128 ch)
//!         │
//!     down_blocks[0..3]: 2× ResBlock + (asymmetric-pad stride-2 conv on 0..2)
//!         │
//!     mid_block: ResBlock → spatial_attn(self only) → ResBlock
//!         │
//!     conv_norm_out (GroupNorm) → SiLU → conv_out (3×3, 8 ch)
//!         │
//!     quant_conv (1×1, 8 → 8)
//!         │
//!     (mean, logvar) — each [4, H/8, W/8]
//! ```
//!
//! No cross-attention (no text conditioning); the mid-block has a
//! self-attention layer. Reuses [`crate::unet::resblock::ResBlock`] but with
//! `time_embed_dim = 0` (the VAE doesn't take a timestep).

mod blocks;
pub mod decoder;
pub mod encoder;

pub use decoder::VaeDecoder;
pub use encoder::VaeEncoder;
