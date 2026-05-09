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

const NORM_EPS: f32 = 1e-6;

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

/// 1D GroupNorm parameters. The compute path goes through
/// `MathBackend::group_norm` directly; this struct just holds the affine.
struct GroupNormParams<B: MathBackend> {
    weight: Tensor<B>,
    bias: Tensor<B>,
    num_groups: usize,
    channels: usize,
}

impl<B: MathBackend> GroupNormParams<B> {
    /// Apply group norm to `[C, H, W]` (batch=1 implicit). Returns same shape.
    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        debug_assert_eq!(dims[0], self.channels);
        let spatial = dims[1] * dims[2];
        let out = B::group_norm(
            &input.data,
            &self.weight.data,
            &self.bias.data,
            self.num_groups,
            self.channels,
            spatial,
            NORM_EPS,
        );
        Tensor::new(out, input.shape.clone())
    }
}

/// VAE residual block (no timestep — that's the UNet flavor).
///
/// `silu → conv1 → silu → conv2`, sandwiched between two GroupNorms,
/// with a `conv_shortcut` 1×1 conv on the residual when channel count
/// changes. HF's `ResnetBlock2D` runs the activations *before* the
/// convs, not after — see the diffusers source.
struct VaeResnetBlock<B: MathBackend> {
    norm1: GroupNormParams<B>,
    conv1: Conv2d<B>,
    norm2: GroupNormParams<B>,
    conv2: Conv2d<B>,
    /// 1×1 channel-matching conv for the residual when `in != out`.
    /// Identity skip when `None`.
    conv_shortcut: Option<Conv2d<B>>,
}

impl<B: MathBackend> VaeResnetBlock<B> {
    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let h = self.norm1.forward(input);
        let h = silu_inplace::<B>(&h);
        let h = self.conv1.forward(&h);
        let h = self.norm2.forward(&h);
        let h = silu_inplace::<B>(&h);
        let h = self.conv2.forward(&h);

        let shortcut = match &self.conv_shortcut {
            Some(c) => c.forward(input),
            None => clone_tensor::<B>(input),
        };
        add_same_shape::<B>(&shortcut, &h)
    }
}

/// VAE mid-block self-attention. Single-head, channels-as-features —
/// HF's `AttnProcessor` with `heads=1`.
struct VaeMidAttention<B: MathBackend> {
    group_norm: GroupNormParams<B>,
    q_weight: Tensor<B>, // [C, C], scry-llm [in, out] convention
    q_bias: Tensor<B>,
    k_weight: Tensor<B>,
    k_bias: Tensor<B>,
    v_weight: Tensor<B>,
    v_bias: Tensor<B>,
    proj_weight: Tensor<B>,
    proj_bias: Tensor<B>,
    channels: usize,
}

impl<B: MathBackend> VaeMidAttention<B> {
    #[allow(clippy::cast_precision_loss)]
    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let (c, h, w) = (dims[0], dims[1], dims[2]);
        debug_assert_eq!(c, self.channels);
        let n = h * w;

        // Pre-norm.
        let normed = self.group_norm.forward(input);

        // Reshape [C, H, W] -> [H*W, C] (transpose).
        let flat = transpose_chw_to_hwc::<B>(&normed, c, n);

        // Q, K, V projections in [H*W, C] layout.
        let q = matmul_bias_2d::<B>(&flat, &self.q_weight, &self.q_bias, n, c, c);
        let k = matmul_bias_2d::<B>(&flat, &self.k_weight, &self.k_bias, n, c, c);
        let v = matmul_bias_2d::<B>(&flat, &self.v_weight, &self.v_bias, n, c, c);

        // Single-head scaled dot-product attention.
        // scores = Q @ K^T → [H*W, H*W], scaled.
        let scores_raw = B::matmul(&q.data, &k.data, n, c, n, false, true);
        let scale = 1.0 / (c as f32).sqrt();
        let attn = B::scaled_softmax(&scores_raw, scale, &Shape::new(&[n, n]));
        // out = attn @ V → [H*W, C]
        let out_data = B::matmul(&attn, &v.data, n, n, c, false, false);
        let out = Tensor::new(out_data, Shape::new(&[n, c]));

        // Output projection.
        let proj = matmul_bias_2d::<B>(&out, &self.proj_weight, &self.proj_bias, n, c, c);

        // Reshape [H*W, C] -> [C, H, W] and add residual.
        let proj_chw = transpose_hwc_to_chw::<B>(&proj, n, c, h, w);
        add_same_shape::<B>(input, &proj_chw)
    }
}

/// Stage of `decoder.up_blocks[i]`: a stack of ResNets followed by an
/// optional 2× nearest-upsample-and-conv.
struct VaeUpBlock<B: MathBackend> {
    resnets: Vec<VaeResnetBlock<B>>,
    upsampler: Option<Conv2d<B>>,
}

impl<B: MathBackend> VaeUpBlock<B> {
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

#[cfg(feature = "safetensors")]
fn load_resnet_block<B: MathBackend>(
    view: &safetensors::SafeTensors<'_>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    num_groups: usize,
    consume: &mut impl FnMut(&str),
) -> Result<VaeResnetBlock<B>> {
    use scry_vision::checkpoint::{load_conv2d_with_bias, load_tensor};

    let norm1 = GroupNormParams {
        weight: load_tensor::<B>(view, &format!("{prefix}.norm1.weight"), &[in_channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.norm1.weight: {e}")))?,
        bias: load_tensor::<B>(view, &format!("{prefix}.norm1.bias"), &[in_channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.norm1.bias: {e}")))?,
        num_groups,
        channels: in_channels,
    };
    consume(&format!("{prefix}.norm1.weight"));
    consume(&format!("{prefix}.norm1.bias"));

    let conv1 = load_conv2d_with_bias::<B>(
        view,
        &format!("{prefix}.conv1"),
        in_channels,
        out_channels,
        3,
        3,
        1,
        1,
    )
    .map_err(|e| Error::Llm(format!("load {prefix}.conv1: {e}")))?;
    consume(&format!("{prefix}.conv1.weight"));
    consume(&format!("{prefix}.conv1.bias"));

    let norm2 = GroupNormParams {
        weight: load_tensor::<B>(view, &format!("{prefix}.norm2.weight"), &[out_channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.norm2.weight: {e}")))?,
        bias: load_tensor::<B>(view, &format!("{prefix}.norm2.bias"), &[out_channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.norm2.bias: {e}")))?,
        num_groups,
        channels: out_channels,
    };
    consume(&format!("{prefix}.norm2.weight"));
    consume(&format!("{prefix}.norm2.bias"));

    let conv2 = load_conv2d_with_bias::<B>(
        view,
        &format!("{prefix}.conv2"),
        out_channels,
        out_channels,
        3,
        3,
        1,
        1,
    )
    .map_err(|e| Error::Llm(format!("load {prefix}.conv2: {e}")))?;
    consume(&format!("{prefix}.conv2.weight"));
    consume(&format!("{prefix}.conv2.bias"));

    let conv_shortcut = if in_channels == out_channels {
        None
    } else {
        let key = format!("{prefix}.conv_shortcut");
        let conv = load_conv2d_with_bias::<B>(view, &key, in_channels, out_channels, 1, 1, 1, 0)
            .map_err(|e| Error::Llm(format!("load {key}: {e}")))?;
        consume(&format!("{key}.weight"));
        consume(&format!("{key}.bias"));
        Some(conv)
    };

    Ok(VaeResnetBlock {
        norm1,
        conv1,
        norm2,
        conv2,
        conv_shortcut,
    })
}

#[cfg(feature = "safetensors")]
fn load_mid_attention<B: MathBackend>(
    view: &safetensors::SafeTensors<'_>,
    prefix: &str,
    channels: usize,
    num_groups: usize,
    consume: &mut impl FnMut(&str),
) -> Result<VaeMidAttention<B>> {
    use scry_vision::checkpoint::{load_f32, load_tensor};

    let group_norm = GroupNormParams {
        weight: load_tensor::<B>(view, &format!("{prefix}.group_norm.weight"), &[channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.group_norm.weight: {e}")))?,
        bias: load_tensor::<B>(view, &format!("{prefix}.group_norm.bias"), &[channels])
            .map_err(|e| Error::Llm(format!("load {prefix}.group_norm.bias: {e}")))?,
        num_groups,
        channels,
    };
    consume(&format!("{prefix}.group_norm.weight"));
    consume(&format!("{prefix}.group_norm.bias"));

    // q/k/v/proj_attn: HF stores [out=C, in=C] (PyTorch nn.Linear). We
    // need scry-llm's [in, out] convention, so transpose-on-load.
    let load_lin = |stem: &str, consume: &mut dyn FnMut(&str)| -> Result<(Tensor<B>, Tensor<B>)> {
        let w_key = format!("{prefix}.{stem}.weight");
        let b_key = format!("{prefix}.{stem}.bias");
        let raw = load_f32(view, &w_key).map_err(|e| Error::Llm(format!("load {w_key}: {e}")))?;
        if raw.len() != channels * channels {
            return Err(Error::Llm(format!(
                "{w_key}: expected {} elems, got {}",
                channels * channels,
                raw.len()
            )));
        }
        let mut t = vec![0.0f32; channels * channels];
        for in_i in 0..channels {
            for out_i in 0..channels {
                t[in_i * channels + out_i] = raw[out_i * channels + in_i];
            }
        }
        let weight = Tensor::from_vec(t, Shape::new(&[channels, channels]));
        let bias = load_tensor::<B>(view, &b_key, &[channels])
            .map_err(|e| Error::Llm(format!("load {b_key}: {e}")))?;
        consume(&w_key);
        consume(&b_key);
        Ok((weight, bias))
    };

    let (q_weight, q_bias) = load_lin("query", &mut |k| consume(k))?;
    let (k_weight, k_bias) = load_lin("key", &mut |k| consume(k))?;
    let (v_weight, v_bias) = load_lin("value", &mut |k| consume(k))?;
    let (proj_weight, proj_bias) = load_lin("proj_attn", &mut |k| consume(k))?;

    Ok(VaeMidAttention {
        group_norm,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        proj_weight,
        proj_bias,
        channels,
    })
}

// -----------------------------------------------------------------------
// Tiny tensor helpers — kept here so vae/decoder.rs is self-contained.
// -----------------------------------------------------------------------

fn silu_inplace<B: MathBackend>(t: &Tensor<B>) -> Tensor<B> {
    Tensor::new(B::silu(&t.data), t.shape.clone())
}

/// Materialize a fresh tensor without round-tripping through host. Mirrors
/// `unet/common.rs::clone_tensor` — `B::scale(_, 1.0)` is a no-op multiply
/// that produces a new device-resident storage on `ScryGpuBackend`.
fn clone_tensor<B: MathBackend>(t: &Tensor<B>) -> Tensor<B> {
    let storage = B::scale(&t.data, 1.0);
    Tensor::new(storage, t.shape.clone())
}

fn add_same_shape<B: MathBackend>(a: &Tensor<B>, b: &Tensor<B>) -> Tensor<B> {
    debug_assert_eq!(a.shape.dims(), b.shape.dims());
    let out = B::add(&a.data, &b.data, &a.shape, &b.shape, &a.shape);
    Tensor::new(out, a.shape.clone())
}

fn matmul_bias_2d<B: MathBackend>(
    a: &Tensor<B>,
    weight: &Tensor<B>,
    bias: &Tensor<B>,
    m: usize,
    k: usize,
    n: usize,
) -> Tensor<B> {
    let out = B::matmul_bias(&a.data, &weight.data, &bias.data, m, k, n, false, false);
    Tensor::new(out, Shape::new(&[m, n]))
}

/// `[C, H, W]` (NCHW flat) → `[H*W, C]`. Mirrors
/// `unet/common.rs::transpose_chw_to_hwc` — viewing the input as `[C, H*W]`
/// and routing through `B::transpose_2d` keeps the work on-device.
fn transpose_chw_to_hwc<B: MathBackend>(t: &Tensor<B>, c: usize, n: usize) -> Tensor<B> {
    debug_assert_eq!(t.shape.numel(), c * n);
    let storage = B::transpose_2d(&t.data, c, n);
    Tensor::new(storage, Shape::new(&[n, c]))
}

/// `[H*W, C]` → `[C, H, W]`. Same `B::transpose_2d` dispatch as
/// [`transpose_chw_to_hwc`]; the result-shape `[c, h, w]` is the
/// contiguous reinterpretation of `[c, h*w]`.
fn transpose_hwc_to_chw<B: MathBackend>(
    t: &Tensor<B>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Tensor<B> {
    debug_assert_eq!(n, h * w);
    debug_assert_eq!(t.shape.numel(), c * n);
    let storage = B::transpose_2d(&t.data, n, c);
    Tensor::new(storage, Shape::new(&[c, h, w]))
}
