// SPDX-License-Identifier: MIT OR Apache-2.0
//! UNet architecture configuration. Parameterized to keep SD 1.5, SD 2.x,
//! and SDXL as preset constructors over a single struct rather than
//! divergent code paths.

/// SD UNet architecture configuration.
#[derive(Debug, Clone)]
pub struct UnetConfig {
    /// Latent channel count entering `conv_in`. Always 4 for the SD VAE.
    pub in_channels: usize,
    /// Latent channel count leaving `conv_out`. Always 4 for the SD VAE
    /// (predicting noise in latent space).
    pub out_channels: usize,
    /// Base channel width — actual block widths are this × `channel_mult`.
    pub block_out_channels: Vec<usize>,
    /// Whether each down/up block stage has cross-attention. SD 1.5 has
    /// `[true, true, true, false]` (no attention at the deepest stage —
    /// 8×8 spatial would be too coarse). SDXL is `[false, true, true]`.
    pub attention_per_block: Vec<bool>,
    /// Number of attention heads per cross-attention block. SD 1.5 = 8;
    /// SDXL varies per stage.
    pub num_attention_heads: Vec<usize>,
    /// Number of transformer layers in each cross-attention block. SD 1.5 = 1
    /// per attention; SDXL = 2 (deepest stage = 10 — the bulk of SDXL's
    /// size growth lives here).
    pub transformer_layers_per_block: Vec<usize>,
    /// Number of ResBlocks per down block (SD 1.5 = 2, SDXL = 2).
    pub layers_per_block: usize,
    /// Cross-attention conditioning dim. SD 1.5 = 768 (CLIP-L). SDXL = 2048
    /// (CLIP-L 768 ++ OpenCLIP-bigG 1280).
    pub cross_attention_dim: usize,
    /// Number of GroupNorm groups (32 across all SD variants).
    pub num_norm_groups: usize,
    /// Timestep embedding dim (typically `block_out_channels[0] * 4`).
    pub time_embed_dim: usize,
    /// Whether the timestep MLP also consumes SDXL extras (size, crop,
    /// target_size). Set this `true` to enable SDXL conditioning.
    pub uses_sdxl_extras: bool,
}

impl UnetConfig {
    /// Stable Diffusion 1.5.
    pub fn sd_1_5() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280, 1280],
            attention_per_block: vec![true, true, true, false],
            num_attention_heads: vec![8, 8, 8, 8],
            transformer_layers_per_block: vec![1, 1, 1, 1],
            layers_per_block: 2,
            cross_attention_dim: 768,
            num_norm_groups: 32,
            time_embed_dim: 1280,
            uses_sdxl_extras: false,
        }
    }

    /// Stable Diffusion 1.5 inpainting. Identical to [`Self::sd_1_5`]
    /// except `conv_in` widens to 9 input channels: the 4 noisy-latent
    /// channels are concatenated with 4 masked-latent channels (VAE-encoded
    /// `image * (1 - mask)`) and 1 mask channel (downsampled to latent
    /// resolution). `conv_out` still emits 4 channels — the masked-latent
    /// and mask are inputs only.
    pub fn sd_1_5_inpainting() -> Self {
        Self {
            in_channels: 9,
            ..Self::sd_1_5()
        }
    }

    /// Stable Diffusion XL base. Heavier UNet (~2.6B params).
    pub fn sdxl_base() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![320, 640, 1280],
            attention_per_block: vec![false, true, true],
            num_attention_heads: vec![5, 10, 20],
            transformer_layers_per_block: vec![1, 2, 10],
            layers_per_block: 2,
            cross_attention_dim: 2048,
            num_norm_groups: 32,
            time_embed_dim: 1280,
            uses_sdxl_extras: true,
        }
    }
}
