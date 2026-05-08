// SPDX-License-Identifier: MIT OR Apache-2.0
//! VAE decoder: latent `[B, 4, H/8, W/8]` → image `[B, 3, H, W]` in (-1, 1).

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use crate::error::Result;

/// VAE decoder configuration.
#[derive(Debug, Clone)]
pub struct VaeDecoderConfig {
    /// Latent channel count entering the decoder. SD VAE = 4.
    pub in_channels: usize,
    /// Output image channel count (3 for RGB).
    pub out_channels: usize,
    /// Channel widths per up block. SD VAE = `[128, 256, 512, 512]`
    /// (deepest first; the decoder builds channels DOWN as it upsamples).
    pub block_out_channels: Vec<usize>,
    /// ResBlocks per up block (3 in SD).
    pub layers_per_block: usize,
    /// Number of GroupNorm groups (32 in SD).
    pub num_norm_groups: usize,
    /// Latent scaling factor — multiply the input latent by this before
    /// `post_quant_conv`. SD 1.5 = `1.0 / 0.18215`. SDXL = `1.0 / 0.13025`.
    pub scaling_factor: f32,
}

impl VaeDecoderConfig {
    /// SD 1.5 / 2.x VAE.
    pub fn sd_1_5() -> Self {
        Self {
            in_channels: 4,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 3,
            num_norm_groups: 32,
            scaling_factor: 1.0 / 0.18215,
        }
    }

    /// SDXL VAE — same architecture, different scaling factor.
    pub fn sdxl() -> Self {
        Self {
            scaling_factor: 1.0 / 0.13025,
            ..Self::sd_1_5()
        }
    }
}

/// VAE decoder.
pub struct VaeDecoder<B: MathBackend> {
    /// Architecture config.
    pub config: VaeDecoderConfig,
    // M4: post_quant_conv, conv_in, mid_block (ResBlock + self-attn + ResBlock),
    // up_blocks (3× ResBlock + 2× upsample per stage, no cross-attn),
    // conv_norm_out, conv_out.
    _backend: std::marker::PhantomData<B>,
}

impl<B: MathBackend> VaeDecoder<B> {
    /// Construct a zero-initialized decoder.
    pub fn new(config: VaeDecoderConfig) -> Self {
        Self {
            config,
            _backend: std::marker::PhantomData,
        }
    }

    /// Decode a latent into pixels in `[-1, 1]`. Caller is responsible for
    /// clamping and rescaling to `[0, 1]` for image output.
    pub fn decode(&mut self, latent: &Tensor<B>) -> Result<Tensor<B>> {
        let _ = latent;
        todo!(
            "M4: scale × scaling_factor → post_quant_conv → conv_in → mid_block → \
             4 up_blocks → conv_norm_out + SiLU + conv_out"
        )
    }
}
