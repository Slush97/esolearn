// SPDX-License-Identifier: MIT OR Apache-2.0
//! UNet down/mid/up block orchestration.
//!
//! Each stage of the UNet bundles `layers_per_block` ResBlocks plus an
//! optional [`SpatialTransformer`] per ResBlock and an optional
//! [`Downsample`] / [`Upsample`] at the end. Skip connections are recorded
//! per layer in [`DownBlock::forward`] and consumed in reverse order by
//! [`UpBlock::forward`].
//!
//! HF naming:
//! - Downsamplers live at `down_blocks.{i}.downsamplers.0.conv` (always
//!   index 0 in SD), with a 3×3 stride-2 padding-1 conv.
//! - Upsamplers live at `up_blocks.{i}.upsamplers.0.conv`, with a 3×3
//!   stride-1 padding-1 conv applied *after* a `nearest` 2× upsample.
//! - SD 1.5 down-side has `attention_per_block = [true, true, true, false]`
//!   (deepest stage skips attention). The up side mirrors with the index
//!   reversed: `up_blocks` carries attention at the *last* three stages
//!   (`[false, true, true, true]`).

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;
use scry_vision::nn::conv2d::Conv2d;

use super::attention::SpatialTransformer;
use super::resblock::ResBlock;
use crate::conditioning::Conditioning;
use crate::error::Result;

/// Strided 3×3 conv that halves spatial dims.
pub struct Downsample<B: MathBackend> {
    /// Input/output channel count (same — Conv2d 3×3 stride 2 keeps channels).
    pub channels: usize,
    pub(crate) conv: Conv2d<B>,
}

/// Nearest-neighbor 2× upsample followed by a 3×3 conv.
pub struct Upsample<B: MathBackend> {
    /// Input/output channel count.
    pub channels: usize,
    pub(crate) conv: Conv2d<B>,
}

/// Down stage: `[ResBlock + (optional) SpatialTransformer] × layers_per_block`,
/// optional Downsample at the end. Returns its output plus the per-layer
/// activations needed for skip connections.
pub struct DownBlock<B: MathBackend> {
    /// ResBlocks in this stage.
    pub resnets: Vec<ResBlock<B>>,
    /// Spatial transformers, one per ResBlock, when this stage has cross-attention.
    pub attentions: Option<Vec<SpatialTransformer<B>>>,
    /// Final downsample, present when this is not the deepest stage.
    pub downsampler: Option<Downsample<B>>,
}

impl<B: MathBackend> DownBlock<B> {
    /// Forward, returning the output plus per-layer skip activations
    /// (in encounter order) that the symmetric `UpBlock` consumes.
    pub fn forward(
        &mut self,
        feature_map: &Tensor<B>,
        time_embed: &Tensor<B>,
        conditioning: &Conditioning<B>,
    ) -> Result<(Tensor<B>, Vec<Tensor<B>>)> {
        let _ = (feature_map, time_embed, conditioning);
        todo!("M6: alternate ResBlock + (optional SpatialTransformer); record skips; downsample at end")
    }
}

/// Mid stage: `ResBlock + SpatialTransformer + ResBlock`. Single transformer
/// across SD 1.5 / SD 2.x / SDXL.
pub struct MidBlock<B: MathBackend> {
    /// First ResBlock.
    pub resnet_in: ResBlock<B>,
    /// Spatial transformer (always present in SD's mid block).
    pub attention: SpatialTransformer<B>,
    /// Second ResBlock.
    pub resnet_out: ResBlock<B>,
}

impl<B: MathBackend> MidBlock<B> {
    /// Forward.
    pub fn forward(
        &mut self,
        feature_map: &Tensor<B>,
        time_embed: &Tensor<B>,
        conditioning: &Conditioning<B>,
    ) -> Result<Tensor<B>> {
        let _ = (feature_map, time_embed, conditioning);
        todo!("M6: ResBlock → SpatialTransformer → ResBlock")
    }
}

/// Up stage: `[ResBlock + (optional) SpatialTransformer] × (layers_per_block + 1)`,
/// optional Upsample at the end. Concatenates the matching `DownBlock` skip
/// to the input of each ResBlock along the channel axis.
pub struct UpBlock<B: MathBackend> {
    /// ResBlocks in this stage.
    pub resnets: Vec<ResBlock<B>>,
    /// Spatial transformers, one per ResBlock, when this stage has cross-attention.
    pub attentions: Option<Vec<SpatialTransformer<B>>>,
    /// Final upsample, present when this is not the shallowest stage.
    pub upsampler: Option<Upsample<B>>,
}

impl<B: MathBackend> UpBlock<B> {
    /// Forward consuming skips from the matching DownBlock in reverse order.
    pub fn forward(
        &mut self,
        feature_map: &Tensor<B>,
        skips: &mut Vec<Tensor<B>>,
        time_embed: &Tensor<B>,
        conditioning: &Conditioning<B>,
    ) -> Result<Tensor<B>> {
        let _ = (feature_map, skips, time_embed, conditioning);
        todo!("M6: pop skip → concat along channels → ResBlock → (optional SpatialTransformer); upsample at end")
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> Downsample<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        channels: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use scry_vision::checkpoint::load_conv2d_with_bias;

        use crate::error::Error;

        let key = format!("{prefix}.conv");
        let conv = load_conv2d_with_bias::<B>(view, &key, channels, channels, 3, 3, 2, 1)
            .map_err(|e| Error::Llm(format!("load {key}: {e}")))?;
        consume(&format!("{key}.weight"));
        consume(&format!("{key}.bias"));
        Ok(Self { channels, conv })
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> Upsample<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        channels: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use scry_vision::checkpoint::load_conv2d_with_bias;

        use crate::error::Error;

        let key = format!("{prefix}.conv");
        let conv = load_conv2d_with_bias::<B>(view, &key, channels, channels, 3, 3, 1, 1)
            .map_err(|e| Error::Llm(format!("load {key}: {e}")))?;
        consume(&format!("{key}.weight"));
        consume(&format!("{key}.bias"));
        Ok(Self { channels, conv })
    }
}

/// Shared parameters for a generic down/mid/up stage loader. Avoids
/// fanning out a 7-arg signature five times.
#[cfg(feature = "safetensors")]
pub(crate) struct StageParams {
    pub(crate) num_resnets: usize,
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    /// Per-resnet input override (Vec of length `num_resnets`). When `None`,
    /// resnet 0 takes `in_channels` and the rest take `out_channels`. Used
    /// by the up-side, where each resnet's input is `(prev || skip)`.
    pub(crate) resnet_in_channels: Option<Vec<usize>>,
    pub(crate) time_embed_dim: usize,
    pub(crate) num_norm_groups: usize,
    pub(crate) has_attention: bool,
    pub(crate) num_attention_heads: usize,
    pub(crate) cross_attention_dim: usize,
    pub(crate) transformer_layers: usize,
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> DownBlock<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        params: &StageParams,
        has_downsampler: bool,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(params.num_resnets);
        for r in 0..params.num_resnets {
            let in_ch = if let Some(overrides) = &params.resnet_in_channels {
                overrides[r]
            } else if r == 0 {
                params.in_channels
            } else {
                params.out_channels
            };
            resnets.push(ResBlock::from_safetensors(
                view,
                &format!("{prefix}.resnets.{r}"),
                in_ch,
                params.out_channels,
                params.time_embed_dim,
                params.num_norm_groups,
                consume,
            )?);
        }

        let attentions = if params.has_attention {
            let mut atts = Vec::with_capacity(params.num_resnets);
            for r in 0..params.num_resnets {
                atts.push(SpatialTransformer::from_safetensors(
                    view,
                    &format!("{prefix}.attentions.{r}"),
                    params.out_channels,
                    params.num_attention_heads,
                    params.cross_attention_dim,
                    params.transformer_layers,
                    params.num_norm_groups,
                    consume,
                )?);
            }
            Some(atts)
        } else {
            None
        };

        let downsampler = if has_downsampler {
            Some(Downsample::from_safetensors(
                view,
                &format!("{prefix}.downsamplers.0"),
                params.out_channels,
                consume,
            )?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            attentions,
            downsampler,
        })
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> UpBlock<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        params: &StageParams,
        has_upsampler: bool,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        let mut resnets = Vec::with_capacity(params.num_resnets);
        for r in 0..params.num_resnets {
            let in_ch = if let Some(overrides) = &params.resnet_in_channels {
                overrides[r]
            } else if r == 0 {
                params.in_channels
            } else {
                params.out_channels
            };
            resnets.push(ResBlock::from_safetensors(
                view,
                &format!("{prefix}.resnets.{r}"),
                in_ch,
                params.out_channels,
                params.time_embed_dim,
                params.num_norm_groups,
                consume,
            )?);
        }

        let attentions = if params.has_attention {
            let mut atts = Vec::with_capacity(params.num_resnets);
            for r in 0..params.num_resnets {
                atts.push(SpatialTransformer::from_safetensors(
                    view,
                    &format!("{prefix}.attentions.{r}"),
                    params.out_channels,
                    params.num_attention_heads,
                    params.cross_attention_dim,
                    params.transformer_layers,
                    params.num_norm_groups,
                    consume,
                )?);
            }
            Some(atts)
        } else {
            None
        };

        let upsampler = if has_upsampler {
            Some(Upsample::from_safetensors(
                view,
                &format!("{prefix}.upsamplers.0"),
                params.out_channels,
                consume,
            )?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            attentions,
            upsampler,
        })
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> MidBlock<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        params: &StageParams,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        // Mid block is always 1280→1280→1280 with one transformer between.
        let resnet_in = ResBlock::from_safetensors(
            view,
            &format!("{prefix}.resnets.0"),
            params.in_channels,
            params.out_channels,
            params.time_embed_dim,
            params.num_norm_groups,
            consume,
        )?;
        let attention = SpatialTransformer::from_safetensors(
            view,
            &format!("{prefix}.attentions.0"),
            params.out_channels,
            params.num_attention_heads,
            params.cross_attention_dim,
            params.transformer_layers,
            params.num_norm_groups,
            consume,
        )?;
        let resnet_out = ResBlock::from_safetensors(
            view,
            &format!("{prefix}.resnets.1"),
            params.out_channels,
            params.out_channels,
            params.time_embed_dim,
            params.num_norm_groups,
            consume,
        )?;

        Ok(Self {
            resnet_in,
            attention,
            resnet_out,
        })
    }
}
