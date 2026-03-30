// SPDX-License-Identifier: MIT OR Apache-2.0
//! Segment Anything Model (SAM) — interactive segmentation.
//!
//! SAM segments any object in an image given a point, box, or mask prompt.
//! The architecture has three components:
//! - **Image encoder** ([`image_encoder::SamVit`]) — windowed `ViT` that produces
//!   a 2D feature map
//! - **Prompt encoder** ([`prompt_encoder::PromptEncoder`]) — converts prompts
//!   to sparse and dense embeddings
//! - **Mask decoder** ([`mask_decoder::MaskDecoder`]) — two-way transformer that
//!   fuses image and prompt features into multi-mask output with `IoU` scores
//!
//! # Usage
//!
//! For the high-level `Segment` trait, use [`SamSegmenter`].
//! For the "encode once, prompt many" pattern, use [`Sam`] directly:
//!
//! ```ignore
//! let sam = Sam::from_safetensors(SamConfig::vit_b(), path)?;
//! let embedding = sam.encode_image(&image_tensor);
//! let output1 = sam.predict(&embedding, &prompt1);
//! let output2 = sam.predict(&embedding, &prompt2); // reuses encoder output
//! ```

pub mod image_encoder;
pub mod mask_decoder;
pub mod prompt_encoder;

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::Tensor;

use crate::error::Result;
use crate::image::ImageBuffer;
use crate::pipeline::{Mask, Segment, SegmentPrompt};
use crate::transform::resize::{InterpolationMode, Resize};
use crate::transform::to_tensor::ToTensor;
use crate::transform::ImageTransform;

use image_encoder::SamVit;
use mask_decoder::MaskDecoder;
use prompt_encoder::PromptEncoder;

// ── Config ──────────────────────────────────────────────────────────────────

/// SAM configuration for all three model components.
#[derive(Clone, Debug)]
pub struct SamConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: f32,
    pub global_attn_indices: Vec<usize>,
    pub window_size: usize,
    /// Output channel count of the neck / decoder embedding dimension.
    pub out_channels: usize,
    pub num_multimask_outputs: usize,
    pub iou_head_depth: usize,
    pub iou_head_hidden_dim: usize,
    pub decoder_depth: usize,
    pub decoder_num_heads: usize,
}

impl SamConfig {
    /// SAM `ViT`-B (base): 91M image encoder parameters.
    pub fn vit_b() -> Self {
        Self {
            image_size: 1024,
            patch_size: 16,
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            global_attn_indices: vec![2, 5, 8, 11],
            window_size: 14,
            out_channels: 256,
            num_multimask_outputs: 3,
            iou_head_depth: 3,
            iou_head_hidden_dim: 256,
            decoder_depth: 2,
            decoder_num_heads: 8,
        }
    }

    /// SAM `ViT`-L (large): 308M image encoder parameters.
    pub fn vit_l() -> Self {
        Self {
            embed_dim: 1024,
            depth: 24,
            num_heads: 16,
            global_attn_indices: vec![5, 11, 17, 23],
            ..Self::vit_b()
        }
    }

    /// SAM `ViT`-H (huge): 636M image encoder parameters.
    pub fn vit_h() -> Self {
        Self {
            embed_dim: 1280,
            depth: 32,
            num_heads: 16,
            global_attn_indices: vec![7, 15, 23, 31],
            ..Self::vit_b()
        }
    }

    /// Number of patches along one side: `image_size / patch_size`.
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Attention head dimension.
    pub fn d_head(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    /// Feed-forward hidden dimension.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    pub fn d_ff(&self) -> usize {
        (self.embed_dim as f32 * self.mlp_ratio) as usize
    }

    /// Number of input channels (always 3 for RGB).
    pub fn in_channels(&self) -> usize {
        3
    }
}

// ── Output types ────────────────────────────────────────────────────────────

/// Full SAM output: multiple masks with `IoU` scores.
pub struct SamOutput {
    /// Predicted mask logits (not thresholded). One `Vec<f32>` per mask.
    pub masks: Vec<Vec<f32>>,
    pub mask_width: u32,
    pub mask_height: u32,
    /// `IoU` confidence score for each mask.
    pub iou_scores: Vec<f32>,
}

/// Cached image encoder output for the "encode once, prompt many" pattern.
pub struct SamImageEmbedding<B: MathBackend> {
    /// Image features: `[out_channels, grid_h, grid_w]`.
    pub embeddings: Tensor<B>,
    pub original_width: u32,
    pub original_height: u32,
}

// ── Sam model ───────────────────────────────────────────────────────────────

/// Full Segment Anything Model.
pub struct Sam<B: MathBackend> {
    pub image_encoder: SamVit<B>,
    pub prompt_encoder: PromptEncoder<B>,
    pub mask_decoder: MaskDecoder<B>,
    pub config: SamConfig,
}

impl<B: MathBackend> Sam<B> {
    /// Create a zero-initialized SAM from config (for testing; real usage loads weights).
    pub fn new(config: SamConfig) -> Self {
        Self {
            image_encoder: SamVit::new(&config),
            prompt_encoder: PromptEncoder::new(&config),
            mask_decoder: MaskDecoder::new(&config),
            config,
        }
    }

    /// Encode an image (expensive). Cache the result for multiple prompts.
    ///
    /// Input: `[3, image_size, image_size]` (already preprocessed).
    pub fn encode_image(
        &self,
        image: &Tensor<B>,
        original_width: u32,
        original_height: u32,
    ) -> SamImageEmbedding<B> {
        let embeddings = self.image_encoder.forward(image);
        SamImageEmbedding {
            embeddings,
            original_width,
            original_height,
        }
    }

    /// Predict masks from cached image embeddings + prompt (cheap).
    pub fn predict(&self, embedding: &SamImageEmbedding<B>, prompt: &SegmentPrompt) -> SamOutput {
        let (sparse, dense) = self.prompt_encoder.forward(prompt);
        let image_pe = self.prompt_encoder.get_dense_pe();

        self.mask_decoder
            .forward(&embedding.embeddings, &image_pe, &sparse, &dense)
    }

    /// Convenience: encode + predict in one call.
    pub fn segment_image(
        &self,
        image: &Tensor<B>,
        width: u32,
        height: u32,
        prompt: &SegmentPrompt,
    ) -> SamOutput {
        let emb = self.encode_image(image, width, height);
        self.predict(&emb, prompt)
    }

    /// Select the best mask from a multi-mask output based on `IoU` scores.
    pub fn best_mask(output: &SamOutput) -> Mask {
        let best_idx = output
            .iou_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        let mask_data: Vec<u8> = output.masks[best_idx]
            .iter()
            .map(|&logit| if logit > 0.0 { 255 } else { 0 })
            .collect();

        Mask {
            data: mask_data,
            width: output.mask_width,
            height: output.mask_height,
        }
    }
}

// ── Safetensors loading ─────────────────────────────────────────────────────

#[cfg(feature = "safetensors")]
impl<B: MathBackend> Sam<B> {
    /// Load a full SAM model from a safetensors file.
    ///
    /// Expects HuggingFace SAM format with keys prefixed by `image_encoder.`,
    /// `prompt_encoder.`, and `mask_decoder.`.
    pub fn from_safetensors(config: SamConfig, path: &std::path::Path) -> Result<Self> {
        use crate::error::VisionError;

        let file = std::fs::File::open(path)
            .map_err(|e| VisionError::ModelLoad(format!("cannot open {}: {e}", path.display())))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| VisionError::ModelLoad(format!("mmap failed: {e}")))?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| VisionError::ModelLoad(format!("safetensors parse failed: {e}")))?;

        let image_encoder = SamVit::from_safetensors(&config, &tensors, "image_encoder.")?;
        let prompt_encoder = PromptEncoder::from_safetensors(&config, &tensors, "prompt_encoder.")?;
        let mask_decoder = MaskDecoder::from_safetensors(&config, &tensors, "mask_decoder.")?;

        Ok(Self {
            image_encoder,
            prompt_encoder,
            mask_decoder,
            config,
        })
    }
}

impl<B: MathBackend> Module<B> for Sam<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.image_encoder.parameters();
        params.extend(self.prompt_encoder.parameters());
        params.extend(self.mask_decoder.parameters());
        params
    }
}

// ── Pipeline wrapper ────────────────────────────────────────────────────────

/// SAM segmentation pipeline implementing the [`Segment`] trait.
///
/// Handles preprocessing (resize + pad to `image_size`, SAM normalization)
/// and postprocessing (select best mask, resize to original dimensions, threshold).
pub struct SamSegmenter<B: MathBackend> {
    pub model: Sam<B>,
}

/// SAM pixel normalization constants (`ImageNet` statistics).
const SAM_PIXEL_MEAN: [f32; 3] = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0];
const SAM_PIXEL_STD: [f32; 3] = [58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0];

impl<B: MathBackend> SamSegmenter<B> {
    /// Create a segmenter from a SAM model.
    pub fn new(model: Sam<B>) -> Self {
        Self { model }
    }
}

impl<B: MathBackend> Segment for SamSegmenter<B> {
    fn segment(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
        prompt: SegmentPrompt,
    ) -> Result<Mask> {
        let img = ImageBuffer::from_raw(image.to_vec(), width, height, 3)?;
        let target = self.model.config.image_size as u32;

        // Resize longest side to image_size, preserving aspect ratio
        let scale = target as f32 / width.max(height) as f32;
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let (new_w, new_h) = (
            (width as f32 * scale).round() as u32,
            (height as f32 * scale).round() as u32,
        );
        let resize = Resize::new(new_w, new_h, InterpolationMode::Bilinear);
        let resized = resize.apply(&img)?;

        // Pad to square (image_size × image_size)
        let mut padded_data = vec![0u8; (target * target * 3) as usize];
        for y in 0..new_h {
            let src_off = (y * new_w * 3) as usize;
            let dst_off = (y * target * 3) as usize;
            let row_len = (new_w * 3) as usize;
            padded_data[dst_off..dst_off + row_len]
                .copy_from_slice(&resized.data[src_off..src_off + row_len]);
        }
        let padded = ImageBuffer::from_raw(padded_data, target, target, 3)?;

        // SAM normalization
        let tensor = ToTensor::normalized(SAM_PIXEL_MEAN, SAM_PIXEL_STD).apply::<B>(&padded);

        // Run model
        let output = self.model.segment_image(&tensor, width, height, &prompt);

        // Select best mask and resize to original dimensions
        let mut best = Sam::<B>::best_mask(&output);

        // Resize mask back to original resolution
        if best.width != width || best.height != height {
            let mask_img = ImageBuffer::from_raw(best.data, best.width, best.height, 1)?;
            let resize_back = Resize::new(width, height, InterpolationMode::Nearest);
            let resized_mask = resize_back.apply(&mask_img)?;
            best = Mask {
                data: resized_mask.data,
                width,
                height,
            };
        }

        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;
    use scry_llm::tensor::shape::Shape;

    fn tiny_config() -> SamConfig {
        SamConfig {
            image_size: 64,
            patch_size: 16,
            embed_dim: 32,
            depth: 1,
            num_heads: 2,
            mlp_ratio: 2.0,
            global_attn_indices: vec![0],
            window_size: 2,
            out_channels: 16,
            num_multimask_outputs: 3,
            iou_head_depth: 2,
            iou_head_hidden_dim: 16,
            decoder_depth: 1,
            decoder_num_heads: 2,
        }
    }

    #[test]
    fn sam_configs_valid() {
        let b = SamConfig::vit_b();
        assert_eq!(b.num_patches_per_side(), 64);
        assert_eq!(b.d_head(), 64);
        assert_eq!(b.d_ff(), 3072);

        let l = SamConfig::vit_l();
        assert_eq!(l.num_patches_per_side(), 64);
        assert_eq!(l.d_head(), 64);

        let h = SamConfig::vit_h();
        assert_eq!(h.num_patches_per_side(), 64);
        assert_eq!(h.d_head(), 80);
    }

    #[test]
    fn sam_full_forward_tiny() {
        let config = tiny_config();
        let sam = Sam::<CpuBackend>::new(config.clone());
        let input = Tensor::from_vec(vec![0.0; 3 * 64 * 64], Shape::new(&[3, 64, 64]));
        let prompt = SegmentPrompt::Point { x: 32.0, y: 32.0 };
        let output = sam.segment_image(&input, 64, 64, &prompt);

        assert_eq!(output.masks.len(), 4);
        assert_eq!(output.mask_width, 16); // grid_h(4) * 4
        assert_eq!(output.mask_height, 16);
        assert_eq!(output.iou_scores.len(), 4);
    }

    #[test]
    fn best_mask_selects_highest_iou() {
        let output = SamOutput {
            masks: vec![vec![0.0; 4], vec![1.0; 4], vec![-1.0; 4], vec![0.5; 4]],
            mask_width: 2,
            mask_height: 2,
            iou_scores: vec![0.3, 0.9, 0.1, 0.7],
        };
        let best = Sam::<CpuBackend>::best_mask(&output);
        assert_eq!(best.width, 2);
        assert_eq!(best.height, 2);
        // Mask 1 has highest IoU (0.9), all logits = 1.0 > 0 → all foreground
        assert!(best.data.iter().all(|&v| v == 255));
    }

    #[test]
    fn best_mask_threshold() {
        let output = SamOutput {
            masks: vec![vec![-0.5, 0.5, -0.1, 0.1]],
            mask_width: 2,
            mask_height: 2,
            iou_scores: vec![0.8],
        };
        let best = Sam::<CpuBackend>::best_mask(&output);
        // logits: -0.5 → 0, 0.5 → 255, -0.1 → 0, 0.1 → 255
        assert_eq!(best.data, vec![0, 255, 0, 255]);
    }

    #[test]
    fn sam_segmenter_trait_produces_valid_mask() {
        let config = tiny_config();
        let sam = Sam::<CpuBackend>::new(config);
        let segmenter = SamSegmenter::new(sam);

        let image = vec![128u8; 32 * 32 * 3];
        let prompt = SegmentPrompt::Point { x: 16.0, y: 16.0 };
        let mask = segmenter.segment(&image, 32, 32, prompt).unwrap();

        assert_eq!(mask.width, 32);
        assert_eq!(mask.height, 32);
        assert_eq!(mask.data.len(), 32 * 32);
        // All values should be 0 or 255
        assert!(mask.data.iter().all(|&v| v == 0 || v == 255));
    }

    #[test]
    fn encode_once_predict_many() {
        let config = tiny_config();
        let sam = Sam::<CpuBackend>::new(config);

        let input = Tensor::from_vec(vec![0.0; 3 * 64 * 64], Shape::new(&[3, 64, 64]));
        let embedding = sam.encode_image(&input, 64, 64);

        let prompt1 = SegmentPrompt::Point { x: 10.0, y: 10.0 };
        let prompt2 = SegmentPrompt::Point { x: 50.0, y: 50.0 };

        let out1 = sam.predict(&embedding, &prompt1);
        let out2 = sam.predict(&embedding, &prompt2);

        assert_eq!(out1.masks.len(), 4);
        assert_eq!(out2.masks.len(), 4);
    }

    #[test]
    fn sam_parameter_count_tiny() {
        let config = tiny_config();
        let sam = Sam::<CpuBackend>::new(config);
        let total: usize = sam.parameters().iter().map(|p| p.numel()).sum();
        // Verify we have a reasonable number of parameters (not zero)
        assert!(total > 0);
    }
}
