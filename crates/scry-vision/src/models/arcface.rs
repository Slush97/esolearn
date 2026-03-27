// SPDX-License-Identifier: MIT OR Apache-2.0
//! ArcFace face embedding (ONNX-based).
//!
//! Wraps an ArcFace ONNX model and implements the [`Embed`] pipeline trait.
//! Preprocessing: AffineTransform (face alignment) → ToTensor (normalize).

use scry_llm::backend::cpu::CpuBackend;

use crate::error::Result;
use crate::image::ImageBuffer;
use crate::model::VisionModel;
use crate::pipeline::Embed;
use crate::postprocess::embedding::l2_normalize;
use crate::transform::resize::{InterpolationMode, Resize};
use crate::transform::to_tensor::ToTensor;
use crate::transform::ImageTransform;

/// ArcFace face embedder.
///
/// Expects a pre-cropped/aligned face image. For full pipeline usage,
/// first detect faces with SCRFD, align with [`AffineTransform`], then embed.
pub struct ArcFaceEmbedder {
    model: Box<dyn VisionModel>,
    input_size: u32,
    #[allow(dead_code)]
    embed_dim: usize,
}

impl ArcFaceEmbedder {
    pub fn new(model: Box<dyn VisionModel>, input_size: u32, embed_dim: usize) -> Self {
        Self {
            model,
            input_size,
            embed_dim,
        }
    }

    /// Standard ArcFace: 112×112 input, 512-dim embedding.
    #[cfg(feature = "onnx")]
    pub fn from_onnx(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let model = crate::model::OnnxModel::from_file(path)?;
        Ok(Self::new(Box::new(model), 112, 512))
    }
}

impl Embed for ArcFaceEmbedder {
    fn embed(&self, image: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
        let img = ImageBuffer::from_raw(image.to_vec(), width, height, 3)?;

        // Resize to model input size
        let resize = Resize::new(self.input_size, self.input_size, InterpolationMode::Bilinear);
        let resized = resize.apply(&img)?;

        // Normalize: scale to 0..1, then ImageNet-like normalization
        let to_tensor = ToTensor::normalized([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        let tensor = to_tensor.apply::<CpuBackend>(&resized);
        let input_data = tensor.to_vec();
        let input_shape = [1, 3, self.input_size as usize, self.input_size as usize];

        let output = self.model.forward(&input_data, &input_shape)?;

        // L2 normalize the embedding
        let mut embedding = output;
        l2_normalize(&mut embedding);

        Ok(embedding)
    }
}
