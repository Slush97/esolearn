// SPDX-License-Identifier: MIT OR Apache-2.0
//! VAE encoder: image `[3, H, W]` → diagonal Gaussian latent
//! `(mean, logvar)` each `[4, H/8, W/8]`. The mirror of [`super::decoder`].
//!
//! Architecture (SD 1.5 / 2.x — `block_out_channels = [128, 256, 512, 512]`,
//! reading shallow → deep this time):
//!
//! ```text
//! image [3, 512, 512]
//!     │
//! conv_in (3×3, 3 → 128)
//!     │
//! down_blocks[0]: 2× ResNet (128→128) → asym-pad stride-2 conv
//!     │
//! down_blocks[1]: 2× ResNet (first 128→256, rest 256→256) → asym-pad stride-2
//!     │
//! down_blocks[2]: 2× ResNet (first 256→512, rest 512→512) → asym-pad stride-2
//!     │
//! down_blocks[3]: 2× ResNet (512→512) → no downsampler
//!     │
//! mid_block:  ResNet → SelfAttn (1-head, d=512) → ResNet
//!     │
//! conv_norm_out (GroupNorm 32-group on 512 ch) → SiLU → conv_out (3×3, 512 → 8)
//!     │
//! quant_conv (1×1, 8 → 8)        ← lives at top level, *not* under `encoder.`
//!     │
//! split channel-wise into (mean[4, H/8, W/8], logvar[4, H/8, W/8])
//! ```
//!
//! The encoder produces *parameters* of a diagonal Gaussian — sampling
//! `latent = mean + exp(0.5 * logvar) * noise` happens at the call site
//! (M10 v2 pipeline glue) so the caller controls the noise tensor for
//! determinism vs HF parity dumps.
//!
//! Note the **asymmetric padding** on the downsamplers: HF's
//! `Downsample2D` pads `(0, 1, 0, 1)` (right + bottom only) before a
//! stride-2 padding-0 conv. `scry_vision::nn::Conv2d` only supports
//! symmetric padding, so M10 v1 lands a small `pad_2d_zero` helper on
//! `MathBackend` to bridge the gap. v0 only ships the struct + loader;
//! [`VaeEncoder::encode`] errors until v1.
//!
//! Reference: `diffusers/src/diffusers/models/autoencoders/vae.py::Encoder`,
//! `models/downsampling.py::Downsample2D`, and
//! `models/autoencoders/autoencoder_kl.py::AutoencoderKL` (the `quant_conv`
//! lives on the outer module, not inside `Encoder`).

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;
use scry_vision::nn::conv2d::Conv2d;

use crate::error::{Error, Result};
#[cfg(feature = "safetensors")]
use crate::vae::blocks::{load_mid_attention, load_resnet_block};
use crate::vae::blocks::{GroupNormParams, VaeMidAttention, VaeResnetBlock};

/// VAE encoder configuration.
#[derive(Debug, Clone)]
pub struct VaeEncoderConfig {
    /// Image channel count (3 for RGB).
    pub in_channels: usize,
    /// Latent channel count (4 for SD VAE — note `conv_out` produces
    /// `2 * out_channels = 8` because the encoder predicts mean + logvar).
    pub out_channels: usize,
    /// Channel widths per down block, **shallow first** — opposite of
    /// the decoder's reading order. SD VAE = `[128, 256, 512, 512]`.
    pub block_out_channels: Vec<usize>,
    /// ResBlocks per down block. SD 1.5 encoder = 2 (the decoder uses 3
    /// because `UpDecoderBlock2D` adds an extra resnet beyond
    /// `layers_per_block`; `DownEncoderBlock2D` does not).
    pub layers_per_block: usize,
    /// Number of GroupNorm groups (32 in SD).
    pub num_norm_groups: usize,
    /// Latent scaling factor — multiply the sampled latent by this on
    /// the way out of the encoder. SD 1.5 = `0.18215`. SDXL = `0.13025`.
    /// (The decoder side multiplies by the *reciprocal* on the way in.)
    pub scaling_factor: f32,
}

impl VaeEncoderConfig {
    /// SD 1.5 / 2.x VAE.
    #[must_use]
    pub fn sd_1_5() -> Self {
        Self {
            in_channels: 3,
            out_channels: 4,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            num_norm_groups: 32,
            scaling_factor: 0.18215,
        }
    }

    /// SDXL VAE — same architecture, different scaling factor.
    #[must_use]
    pub fn sdxl() -> Self {
        Self {
            scaling_factor: 0.13025,
            ..Self::sd_1_5()
        }
    }
}

// -----------------------------------------------------------------------
// Down-block (mirror of decoder's `VaeUpBlock`)
// -----------------------------------------------------------------------

/// Stage of `encoder.down_blocks[i]`: a stack of ResNets followed by an
/// optional 2× asymmetric-pad stride-2 conv downsampler.
struct VaeDownBlock<B: MathBackend> {
    resnets: Vec<VaeResnetBlock<B>>,
    /// `None` on the deepest block (no spatial reduction at the bottom).
    /// The conv itself is constructed with `stride = 2, padding = 0`;
    /// the asymmetric `(0, 1, 0, 1)` pad happens in the forward pass.
    downsampler: Option<Conv2d<B>>,
}

impl<B: MathBackend> VaeDownBlock<B> {
    fn to_device(&mut self) {
        for r in &mut self.resnets {
            r.to_device();
        }
        if let Some(d) = self.downsampler.as_mut() {
            d.to_device();
        }
    }
}

// -----------------------------------------------------------------------
// VaeEncoder
// -----------------------------------------------------------------------

/// VAE encoder. Currently scaffolding only — [`Self::encode`] is wired up
/// in M10 v1 once the `pad_2d_zero` op lands on `MathBackend`.
pub struct VaeEncoder<B: MathBackend> {
    /// Architecture config.
    pub config: VaeEncoderConfig,
    /// First encoder conv: 3 → `block_out_channels[0]`.
    conv_in: Conv2d<B>,
    /// Four down-blocks (2 resnets each in SD; 2× downsampler on the first three).
    down_blocks: Vec<VaeDownBlock<B>>,
    /// Mid-block: ResNet → SelfAttn → ResNet, all at deepest width.
    mid_resnet0: VaeResnetBlock<B>,
    mid_attn: VaeMidAttention<B>,
    mid_resnet1: VaeResnetBlock<B>,
    /// Output GroupNorm + 3×3 conv (deepest → `2 * out_channels`).
    conv_norm_out: GroupNormParams<B>,
    conv_out: Conv2d<B>,
    /// Latent post-projection 1×1 conv (`quant_conv` at the safetensors
    /// top level — *not* under `encoder.`). Maps `2*out_channels →
    /// 2*out_channels`, refining the mean/logvar split before sampling.
    quant_conv: Conv2d<B>,
}

impl<B: MathBackend> VaeEncoder<B> {
    /// Pre-upload every parameter tensor in the VAE encoder to the backend's
    /// device-resident form. No-op on `CpuBackend`; idempotent on any backend.
    pub fn to_device(&mut self) {
        self.conv_in.to_device();
        for b in &mut self.down_blocks {
            b.to_device();
        }
        self.mid_resnet0.to_device();
        self.mid_attn.to_device();
        self.mid_resnet1.to_device();
        self.conv_norm_out.to_device();
        self.conv_out.to_device();
        self.quant_conv.to_device();
    }

    /// Encode an image into the parameters of a diagonal Gaussian latent
    /// distribution. Returns `(mean, logvar)`, each `[out_channels, H/8, W/8]`.
    ///
    /// Caller applies the reparameterization `latent = mean + exp(0.5 *
    /// logvar) * noise` (and the `scaling_factor` multiply) so the caller
    /// owns the noise tensor — matters for HF parity dumps where a fixed
    /// noise vector is injected, and for img2img where the pipeline picks
    /// a deterministic RNG.
    ///
    /// # Errors
    /// M10 v0 ships scaffolding only. The forward pass requires a
    /// `pad_2d_zero` op on `MathBackend` (HF's downsamplers use
    /// asymmetric padding) which lands in M10 v1. Until then this
    /// returns [`Error::Llm`].
    pub fn encode(&self, _image: &Tensor<B>) -> Result<(Tensor<B>, Tensor<B>)> {
        Err(Error::Llm(
            "vae encoder forward not yet implemented (M10 v1: needs asymmetric pad_2d op)".into(),
        ))
    }
}

// -----------------------------------------------------------------------
// Safetensors loader
// -----------------------------------------------------------------------

#[cfg(feature = "safetensors")]
impl<B: MathBackend> VaeEncoder<B> {
    /// Load a VAE encoder from a HF `vae/diffusion_pytorch_model.safetensors`.
    ///
    /// Consumes 108 keys: 2 for `quant_conv`, 106 under `encoder.*`
    /// (conv_in/out, conv_norm_out, mid_block, 4 down_blocks). The decoder
    /// side (`decoder.*` + `post_quant_conv`) is intentionally ignored —
    /// callers loading both should construct a `VaeDecoder` and a
    /// `VaeEncoder` from the same checkpoint.
    #[allow(clippy::too_many_lines)]
    pub fn from_safetensors(
        config: VaeEncoderConfig,
        ckpt: &crate::weights::SafetensorsCheckpoint,
    ) -> Result<Self> {
        use scry_vision::checkpoint::{load_conv2d_with_bias, load_tensor};

        let view = ckpt.tensors()?;
        let g = config.num_norm_groups;
        let mut consumed: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut consume = |key: &str| {
            consumed.insert(key.to_string());
        };

        // The conv_out / quant_conv output channel count is `2 *
        // out_channels` because the encoder produces (mean, logvar)
        // concatenated along channels.
        let latent_double = 2 * config.out_channels;

        // ---- encoder.conv_in (3 -> block_out_channels[0]) --------------
        let shallow_ch = *config
            .block_out_channels
            .first()
            .ok_or_else(|| Error::Llm("vae encoder: empty block_out_channels".into()))?;
        let conv_in = load_conv2d_with_bias::<B>(
            &view,
            "encoder.conv_in",
            config.in_channels,
            shallow_ch,
            3,
            3,
            1,
            1,
        )
        .map_err(|e| Error::Llm(format!("load encoder.conv_in: {e}")))?;
        consume("encoder.conv_in.weight");
        consume("encoder.conv_in.bias");

        // ---- Down blocks (shallow first) -------------------------------
        let n_blocks = config.block_out_channels.len();
        let mut down_blocks = Vec::with_capacity(n_blocks);
        let mut prev_ch = shallow_ch;
        for block_i in 0..n_blocks {
            let out_ch = config.block_out_channels[block_i];
            let prefix = format!("encoder.down_blocks.{block_i}");
            let mut resnets = Vec::with_capacity(config.layers_per_block);
            for r in 0..config.layers_per_block {
                // First resnet of each block takes the previous block's
                // out-channels as input; subsequent resnets stay at out_ch.
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

            // Downsampler on blocks 0..n_blocks-1; last block has none.
            // Conv is 3×3, stride 2, padding 0 (asymmetric pad applied in
            // forward — see module docstring).
            let downsampler = if block_i < n_blocks - 1 {
                let key = format!("{prefix}.downsamplers.0.conv");
                let conv = load_conv2d_with_bias::<B>(&view, &key, out_ch, out_ch, 3, 3, 2, 0)
                    .map_err(|e| Error::Llm(format!("load {key}: {e}")))?;
                consume(&format!("{key}.weight"));
                consume(&format!("{key}.bias"));
                Some(conv)
            } else {
                None
            };

            down_blocks.push(VaeDownBlock {
                resnets,
                downsampler,
            });
            prev_ch = out_ch;
        }

        // ---- Mid block: ResNet -> Attn -> ResNet (all at deep_ch) ------
        let deep_ch = *config
            .block_out_channels
            .last()
            .ok_or_else(|| Error::Llm("vae encoder: empty block_out_channels".into()))?;
        let mid_resnet0 = load_resnet_block::<B>(
            &view,
            "encoder.mid_block.resnets.0",
            deep_ch,
            deep_ch,
            g,
            &mut consume,
        )?;
        let mid_attn = load_mid_attention::<B>(
            &view,
            "encoder.mid_block.attentions.0",
            deep_ch,
            g,
            &mut consume,
        )?;
        let mid_resnet1 = load_resnet_block::<B>(
            &view,
            "encoder.mid_block.resnets.1",
            deep_ch,
            deep_ch,
            g,
            &mut consume,
        )?;

        // ---- Output norm + conv (deep_ch -> 2 * out_channels) ----------
        let conv_norm_out_w =
            load_tensor::<B>(&view, "encoder.conv_norm_out.weight", &[deep_ch])
                .map_err(|e| Error::Llm(format!("load encoder.conv_norm_out.weight: {e}")))?;
        let conv_norm_out_b = load_tensor::<B>(&view, "encoder.conv_norm_out.bias", &[deep_ch])
            .map_err(|e| Error::Llm(format!("load encoder.conv_norm_out.bias: {e}")))?;
        consume("encoder.conv_norm_out.weight");
        consume("encoder.conv_norm_out.bias");
        let conv_norm_out = GroupNormParams {
            weight: conv_norm_out_w,
            bias: conv_norm_out_b,
            num_groups: g,
            channels: deep_ch,
        };

        let conv_out = load_conv2d_with_bias::<B>(
            &view,
            "encoder.conv_out",
            deep_ch,
            latent_double,
            3,
            3,
            1,
            1,
        )
        .map_err(|e| Error::Llm(format!("load encoder.conv_out: {e}")))?;
        consume("encoder.conv_out.weight");
        consume("encoder.conv_out.bias");

        // ---- quant_conv (top-level, 1×1, 8 → 8) ------------------------
        let quant_conv = load_conv2d_with_bias::<B>(
            &view,
            "quant_conv",
            latent_double,
            latent_double,
            1,
            1,
            1,
            0,
        )
        .map_err(|e| Error::Llm(format!("load quant_conv: {e}")))?;
        consume("quant_conv.weight");
        consume("quant_conv.bias");

        // ---- 100% consumption check: encoder + quant_conv half. The
        // decoder side (decoder.* + post_quant_conv) is intentionally
        // ignored.
        let relevant: std::collections::HashSet<String> = view
            .names()
            .into_iter()
            .filter(|n| n.starts_with("encoder.") || n.starts_with("quant_conv"))
            .cloned()
            .collect();
        let missing: Vec<String> = relevant.difference(&consumed).cloned().collect();
        if !missing.is_empty() {
            let mut sorted = missing;
            sorted.sort();
            return Err(Error::Llm(format!(
                "vae encoder loader: {} encoder/quant_conv keys not consumed: {}{}",
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
            conv_in,
            down_blocks,
            mid_resnet0,
            mid_attn,
            mid_resnet1,
            conv_norm_out,
            conv_out,
            quant_conv,
        })
    }
}
