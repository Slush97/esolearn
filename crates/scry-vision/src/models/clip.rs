// SPDX-License-Identifier: MIT OR Apache-2.0
//! CLIP visual encoder.
//!
//! Wraps a ViT backbone with a projection head to produce embeddings
//! in CLIP's joint vision-language space.
//!
//! Input: `[3, H, W]` → Output: `[proj_dim]` (L2-normalized embedding).

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::error::Result;
use crate::image::ImageBuffer;
use crate::pipeline::Embed;
use crate::postprocess::embedding::l2_normalize;
use crate::transform::resize::{InterpolationMode, Resize};
use crate::transform::to_tensor::ToTensor;
use crate::transform::ImageTransform;

use super::vit::{Vit, VitConfig};

/// CLIP visual encoder configuration.
#[derive(Clone, Debug)]
pub struct ClipConfig {
    pub vit: VitConfig,
    /// Projection dimension (e.g., 512 for ViT-B/32).
    pub proj_dim: usize,
}

impl ClipConfig {
    /// CLIP ViT-B/32 configuration.
    pub fn vit_b32() -> Self {
        Self {
            vit: VitConfig::vit_b32(),
            proj_dim: 512,
        }
    }

    /// CLIP ViT-B/16 configuration.
    pub fn vit_b16() -> Self {
        Self {
            vit: VitConfig::vit_b16(),
            proj_dim: 512,
        }
    }

    /// CLIP ViT-L/14 configuration.
    pub fn vit_l14() -> Self {
        Self {
            vit: VitConfig::vit_l14(),
            proj_dim: 768,
        }
    }
}

/// CLIP visual encoder: ViT + projection → L2-normalized embedding.
pub struct ClipVisual<B: MathBackend> {
    pub vit: Vit<B>,
    /// Projection weight: `[embed_dim, proj_dim]`.
    pub proj: Tensor<B>,
    pub config: ClipConfig,
}

impl<B: MathBackend> ClipVisual<B> {
    /// Create a zero-initialized CLIP visual encoder.
    pub fn new(config: ClipConfig) -> Self {
        let embed_dim = config.vit.embed_dim;
        Self {
            vit: Vit::new(config.vit.clone()),
            proj: Tensor::from_vec(
                vec![0.0; embed_dim * config.proj_dim],
                Shape::new(&[embed_dim, config.proj_dim]),
            ),
            config,
        }
    }

    /// Forward pass: `[3, H, W]` → `[proj_dim]` (L2-normalized).
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let d = self.config.vit.embed_dim;

        // ViT: [3, H, W] → [embed_dim]
        let cls = self.vit.forward(input);

        // Project: [1, embed_dim] @ [embed_dim, proj_dim] → [1, proj_dim]
        let cls_2d = B::from_vec(cls.to_vec(), &Shape::new(&[1, d]));
        let proj = B::matmul(
            &cls_2d,
            &self.proj.data,
            1,
            d,
            self.config.proj_dim,
            false,
            false,
        );

        // L2 normalize
        let mut embedding = B::to_vec(&proj);
        l2_normalize(&mut embedding);
        Tensor::from_vec(embedding, Shape::new(&[self.config.proj_dim]))
    }
}

/// CLIP visual encoder that implements the [`Embed`] pipeline trait.
///
/// Handles preprocessing (resize + normalize) internally.
pub struct ClipEmbedder<B: MathBackend> {
    pub model: ClipVisual<B>,
}

impl<B: MathBackend> ClipEmbedder<B> {
    pub fn new(config: ClipConfig) -> Self {
        Self {
            model: ClipVisual::new(config),
        }
    }
}

impl<B: MathBackend> Embed for ClipEmbedder<B> {
    fn embed(&self, image: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
        let img = ImageBuffer::from_raw(image.to_vec(), width, height, 3)?;

        let size = self.model.config.vit.image_size as u32;
        let resize = Resize::new(size, size, InterpolationMode::Bilinear);
        let resized = resize.apply(&img)?;

        let tensor = ToTensor::clip().apply::<B>(&resized);
        let output = self.model.forward(&tensor);
        Ok(output.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn clip_visual_output_shape() {
        let config = ClipConfig::vit_b32();
        let model = ClipVisual::<CpuBackend>::new(config);
        let input = Tensor::from_vec(vec![0.0; 3 * 224 * 224], Shape::new(&[3, 224, 224]));
        let output = model.forward(&input);
        assert_eq!(output.shape.dims(), &[512]);
    }

    #[test]
    fn clip_output_is_normalized() {
        let config = ClipConfig {
            vit: VitConfig {
                image_size: 32,
                patch_size: 16,
                embed_dim: 64,
                num_heads: 4,
                num_layers: 1,
                mlp_ratio: 4.0,
                in_channels: 3,
            },
            proj_dim: 32,
        };
        // Use non-zero weights to get a non-zero embedding
        let mut model = ClipVisual::<CpuBackend>::new(config);
        model.proj = Tensor::from_vec(vec![0.01; 64 * 32], Shape::new(&[64, 32]));
        model.vit.ln_post_gamma = Tensor::from_vec(vec![1.0; 64], Shape::new(&[64]));

        let input = Tensor::from_vec(vec![1.0; 3 * 32 * 32], Shape::new(&[3, 32, 32]));
        let output = model.forward(&input);
        let data = output.to_vec();

        // Check L2 norm ≈ 1.0
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            assert!((norm - 1.0).abs() < 1e-4, "norm = {norm}");
        }
    }
}
