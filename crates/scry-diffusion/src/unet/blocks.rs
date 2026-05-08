// SPDX-License-Identifier: MIT OR Apache-2.0
//! UNet down/mid/up block orchestration.
//!
//! Each stage of the UNet bundles `layers_per_block` ResBlocks plus an
//! optional [`SpatialTransformer`] per ResBlock and an optional
//! [`Downsample`] / [`Upsample`] at the end. Skip connections are recorded
//! per layer in [`DownBlock::forward`] and consumed in reverse order by
//! [`UpBlock::forward`].

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use super::attention::SpatialTransformer;
use super::resblock::ResBlock;
use crate::conditioning::Conditioning;
use crate::error::Result;

/// Strided 3×3 conv that halves spatial dims.
pub struct Downsample<B: MathBackend> {
    /// Input/output channel count (same — Conv2d 3×3 stride 2 keeps channels).
    pub channels: usize,
    _backend: std::marker::PhantomData<B>,
}

/// Nearest-neighbor 2× upsample followed by a 3×3 conv.
pub struct Upsample<B: MathBackend> {
    /// Input/output channel count.
    pub channels: usize,
    _backend: std::marker::PhantomData<B>,
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
