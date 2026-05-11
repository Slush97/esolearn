// SPDX-License-Identifier: MIT OR Apache-2.0
//! VAE decoder: latent `[B, 4, H/8, W/8]` → image `[B, 3, H, W]` in (-1, 1).
//!
//! Architecture (SD 1.5 / 2.x — `block_out_channels = [128, 256, 512, 512]`):
//!
//! ```text
//! latent [4, 64, 64]
//!     │
//!     × scaling_factor (1 / 0.18215)
//!     │
//! post_quant_conv (1×1, 4 → 4)
//!     │
//! conv_in (3×3, 4 → 512)
//!     │
//! mid_block:  ResNet → SelfAttn (1-head, d=512) → ResNet
//!     │
//! up_blocks[0]: 3× ResNet (512→512) → 2× nearest upsample → conv (3×3)
//!     │
//! up_blocks[1]: 3× ResNet (512→512) → 2× nearest upsample → conv (3×3)
//!     │
//! up_blocks[2]: 3× ResNet (first 512→256, rest 256→256) → 2× nearest → conv
//!     │
//! up_blocks[3]: 3× ResNet (first 256→128, rest 128→128) → no upsampler
//!     │
//! conv_norm_out (GroupNorm 32-group on 128 ch) → SiLU → conv_out (3×3, 128 → 3)
//!     │
//! image [3, 512, 512]   in (-1, 1)
//! ```
//!
//! Reuses M1 kernels (GroupNorm, SiLU, upsample_2d_nearest) and
//! `scry_vision::nn::Conv2d` for the convolutions. The mid-block
//! attention is HF's legacy 1-head spatial self-attention (`query`,
//! `key`, `value`, `proj_attn` keys — not the `q_proj`/`k_proj`/etc
//! naming the UNet uses), single-head with `head_dim = channels = 512`
//! and scale `1 / sqrt(512)`.
//!
//! Reference: `diffusers/src/diffusers/models/autoencoders/vae.py::Decoder`
//! and `models/attention_processor.py::AttnProcessor` with `heads=1`.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_vision::nn::conv2d::Conv2d;

use crate::error::{Error, Result};
use crate::vae::blocks::{clone_tensor, GroupNormParams, VaeMidAttention, VaeResnetBlock};
#[cfg(feature = "safetensors")]
use crate::vae::blocks::{load_mid_attention, load_resnet_block};

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
    /// Total ResBlocks per up block. Matches HF's `layers_per_block + 1`
    /// — diffusers' `UpDecoderBlock2D` always adds one extra resnet
    /// beyond the configured `layers_per_block`. SD 1.5 has
    /// `layers_per_block = 2`, so this field is 3.
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

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Stage of `decoder.up_blocks[i]`: a stack of ResNets followed by an
/// optional 2× nearest-upsample-and-conv.
struct VaeUpBlock<B: MathBackend> {
    resnets: Vec<VaeResnetBlock<B>>,
    upsampler: Option<Conv2d<B>>,
}

impl<B: MathBackend> VaeUpBlock<B> {
    fn to_device(&mut self) {
        for r in &mut self.resnets {
            r.to_device();
        }
        if let Some(u) = self.upsampler.as_mut() {
            u.to_device();
        }
    }

    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let mut x = clone_tensor::<B>(input);
        for r in &self.resnets {
            x = r.forward(&x);
        }
        if let Some(conv) = &self.upsampler {
            // Diffusers `Upsample2D`: nearest 2× then conv (3×3 padding 1).
            let dims = x.shape.dims();
            let (c, h, w) = (dims[0], dims[1], dims[2]);
            let upsampled_data = B::upsample_2d_nearest(&x.data, c, h, w, 2);
            let upsampled = Tensor::new(upsampled_data, Shape::new(&[c, h * 2, w * 2]));
            x = conv.forward(&upsampled);
        }
        x
    }
}

// -----------------------------------------------------------------------
// VaeDecoder
// -----------------------------------------------------------------------

/// VAE decoder.
pub struct VaeDecoder<B: MathBackend> {
    /// Architecture config.
    pub config: VaeDecoderConfig,
    /// Latent post-quantization 1×1 conv (`post_quant_conv` at the
    /// safetensors top level — *not* under `decoder.`).
    post_quant_conv: Conv2d<B>,
    /// First decoder conv: 4 → 512.
    conv_in: Conv2d<B>,
    /// Mid-block: ResNet → SelfAttn → ResNet, all at 512 ch.
    mid_resnet0: VaeResnetBlock<B>,
    mid_attn: VaeMidAttention<B>,
    mid_resnet1: VaeResnetBlock<B>,
    /// Four up-blocks (3 resnets each, 2× upsample on the first three).
    up_blocks: Vec<VaeUpBlock<B>>,
    /// Output GroupNorm + 3×3 conv (128 → 3).
    conv_norm_out: GroupNormParams<B>,
    conv_out: Conv2d<B>,
}

impl<B: MathBackend> VaeDecoder<B> {
    /// Pre-upload every parameter tensor in the VAE decoder to the backend's
    /// device-resident form. No-op on `CpuBackend`; idempotent on any backend.
    pub fn to_device(&mut self) {
        self.post_quant_conv.to_device();
        self.conv_in.to_device();
        self.mid_resnet0.to_device();
        self.mid_attn.to_device();
        self.mid_resnet1.to_device();
        for b in &mut self.up_blocks {
            b.to_device();
        }
        self.conv_norm_out.to_device();
        self.conv_out.to_device();
    }

    /// Decode a latent into pixels in `(-1, 1)`. Caller is responsible for
    /// clamping and rescaling to `[0, 1]` for image output.
    ///
    /// Input shape: `[in_channels, H/8, W/8]` (batch=1 squeezed). For SD
    /// 1.5 at 512×512 output, that's `[4, 64, 64]`.
    pub fn decode(&self, latent: &Tensor<B>) -> Result<Tensor<B>> {
        let dims = latent.shape.dims();
        if dims.len() != 3 || dims[0] != self.config.in_channels {
            return Err(Error::Llm(format!(
                "vae decode: expected [{}, H, W] latent, got {:?}",
                self.config.in_channels, dims
            )));
        }

        // Pre-scale by `1 / 0.18215`.
        let scaled = Tensor::new(
            B::scale(&latent.data, self.config.scaling_factor),
            latent.shape.clone(),
        );

        let mut x = self.post_quant_conv.forward(&scaled);
        x = self.conv_in.forward(&x);

        // Mid block.
        x = self.mid_resnet0.forward(&x);
        x = self.mid_attn.forward(&x);
        x = self.mid_resnet1.forward(&x);

        // Up blocks.
        for block in &self.up_blocks {
            x = block.forward(&x);
        }

        // Final norm + activation + conv.
        x = self.conv_norm_out.forward(&x);
        let x_silu_data = B::silu(&x.data);
        let x_silu = Tensor::new(x_silu_data, x.shape.clone());
        Ok(self.conv_out.forward(&x_silu))
    }
}

// -----------------------------------------------------------------------
// Safetensors loader
// -----------------------------------------------------------------------

#[cfg(feature = "safetensors")]
impl<B: MathBackend> VaeDecoder<B> {
    /// Load a VAE decoder from a HF `vae/diffusion_pytorch_model.safetensors`.
    ///
    /// Consumes 140 keys: 2 for `post_quant_conv`, 138 under `decoder.*`
    /// (conv_in/out, conv_norm_out, mid_block, 4 up_blocks). The encoder
    /// side of the VAE (~108 keys under `encoder.*` plus `quant_conv`)
    /// is intentionally ignored — txt2img doesn't need it.
    #[allow(clippy::too_many_lines)]
    pub fn from_safetensors(
        config: VaeDecoderConfig,
        ckpt: &crate::weights::SafetensorsCheckpoint,
    ) -> Result<Self> {
        use scry_vision::checkpoint::{load_conv2d_with_bias, load_tensor};

        let view = ckpt.tensors()?;
        let g = config.num_norm_groups;
        let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut consume = |key: &str| {
            consumed.insert(key.to_string());
        };

        // ---- post_quant_conv (top-level, no `decoder.` prefix) ----------
        let post_quant_conv = load_conv2d_with_bias::<B>(
            &view,
            "post_quant_conv",
            config.in_channels,
            config.in_channels,
            1,
            1,
            1,
            0,
        )
        .map_err(|e| Error::Llm(format!("load post_quant_conv: {e}")))?;
        consume("post_quant_conv.weight");
        consume("post_quant_conv.bias");

        // The decoder's first conv goes 4 -> deepest channel width
        // (block_out_channels[-1]).
        let mid_ch = *config
            .block_out_channels
            .last()
            .ok_or_else(|| Error::Llm("vae: empty block_out_channels".into()))?;
        let conv_in = load_conv2d_with_bias::<B>(
            &view,
            "decoder.conv_in",
            config.in_channels,
            mid_ch,
            3,
            3,
            1,
            1,
        )
        .map_err(|e| Error::Llm(format!("load decoder.conv_in: {e}")))?;
        consume("decoder.conv_in.weight");
        consume("decoder.conv_in.bias");

        // ---- Mid block: ResNet -> Attn -> ResNet (all at mid_ch) -------
        let mid_resnet0 = load_resnet_block::<B>(
            &view,
            "decoder.mid_block.resnets.0",
            mid_ch,
            mid_ch,
            g,
            &mut consume,
        )?;
        let mid_attn = load_mid_attention::<B>(
            &view,
            "decoder.mid_block.attentions.0",
            mid_ch,
            g,
            &mut consume,
        )?;
        let mid_resnet1 = load_resnet_block::<B>(
            &view,
            "decoder.mid_block.resnets.1",
            mid_ch,
            mid_ch,
            g,
            &mut consume,
        )?;

        // ---- Up blocks --------------------------------------------------
        // diffusers reverses block_out_channels for the decoder side: it
        // starts deepest and walks shallow. With block_out_channels =
        // [128, 256, 512, 512], the up-block widths are [512, 512, 512,
        // 128] reading in reverse — but the *transitions* are 512→512,
        // 512→512, 512→256, 256→128 (each up block ends at the
        // reverse-indexed width and feeds the next at that width).
        let n_blocks = config.block_out_channels.len();
        let mut up_blocks = Vec::with_capacity(n_blocks);
        let mut prev_ch = mid_ch;
        for block_i in 0..n_blocks {
            // Reverse-indexed: deepest block first.
            let out_ch = config.block_out_channels[n_blocks - 1 - block_i];
            let prefix = format!("decoder.up_blocks.{block_i}");
            let mut resnets = Vec::with_capacity(config.layers_per_block);
            for r in 0..config.layers_per_block {
                let in_ch = if r == 0 { prev_ch } else { out_ch };
                resnets.push(load_resnet_block::<B>(
                    &view,
                    &format!("{prefix}.resnets.{r}"),
                    in_ch,
                    out_ch,
                    g,
                    &mut consume,
                )?);
            }

            // The deepest 3 of 4 up-blocks have a 2× upsampler at the end.
            // Index 0..n_blocks-1 do, the last one doesn't.
            let upsampler = if block_i < n_blocks - 1 {
                let key = format!("{prefix}.upsamplers.0.conv");
                let conv = load_conv2d_with_bias::<B>(&view, &key, out_ch, out_ch, 3, 3, 1, 1)
                    .map_err(|e| Error::Llm(format!("load {key}: {e}")))?;
                consume(&format!("{key}.weight"));
                consume(&format!("{key}.bias"));
                Some(conv)
            } else {
                None
            };

            up_blocks.push(VaeUpBlock { resnets, upsampler });
            prev_ch = out_ch;
        }

        // ---- Output norm + conv ----------------------------------------
        let final_ch = *config
            .block_out_channels
            .first()
            .ok_or_else(|| Error::Llm("vae: empty block_out_channels".into()))?;
        let conv_norm_out_w = load_tensor::<B>(&view, "decoder.conv_norm_out.weight", &[final_ch])
            .map_err(|e| Error::Llm(format!("load decoder.conv_norm_out.weight: {e}")))?;
        let conv_norm_out_b = load_tensor::<B>(&view, "decoder.conv_norm_out.bias", &[final_ch])
            .map_err(|e| Error::Llm(format!("load decoder.conv_norm_out.bias: {e}")))?;
        consume("decoder.conv_norm_out.weight");
        consume("decoder.conv_norm_out.bias");
        let conv_norm_out = GroupNormParams {
            weight: conv_norm_out_w,
            bias: conv_norm_out_b,
            num_groups: g,
            channels: final_ch,
        };

        let conv_out = load_conv2d_with_bias::<B>(
            &view,
            "decoder.conv_out",
            final_ch,
            config.out_channels,
            3,
            3,
            1,
            1,
        )
        .map_err(|e| Error::Llm(format!("load decoder.conv_out: {e}")))?;
        consume("decoder.conv_out.weight");
        consume("decoder.conv_out.bias");

        // ---- 100% consumption check: only the decoder + post_quant
        // half. Encoder / quant_conv keys are intentionally unused.
        let relevant: std::collections::HashSet<String> = view
            .names()
            .into_iter()
            .filter(|n| n.starts_with("decoder.") || n.starts_with("post_quant_conv"))
            .cloned()
            .collect();
        let missing: Vec<String> = relevant.difference(&consumed).cloned().collect();
        if !missing.is_empty() {
            let mut sorted = missing;
            sorted.sort();
            return Err(Error::Llm(format!(
                "vae decoder loader: {} decoder/post_quant keys not consumed: {}{}",
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
            post_quant_conv,
            conv_in,
            mid_resnet0,
            mid_attn,
            mid_resnet1,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }
}
