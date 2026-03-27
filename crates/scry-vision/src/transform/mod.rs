// SPDX-License-Identifier: MIT OR Apache-2.0
//! Image transforms for vision inference preprocessing.
//!
//! Transforms operate on [`ImageBuffer`] (u8 pixels) and are composable via
//! [`TransformPipeline`]. Use [`ToTensor`] as the final step to cross into
//! the `Tensor<B>` domain.
//!
//! Unlike scry-learn's [`Transformer`](scry_learn::preprocess::Transformer)
//! trait, these transforms have no `fit` step — all parameters are known at
//! construction time (deterministic inference preprocessing).

pub mod affine;
pub mod color;
pub mod crop;
pub mod normalize;
pub mod pad;
pub mod resize;
pub mod to_tensor;

pub use affine::AffineTransform;
pub use color::ColorConvert;
pub use crop::{CenterCrop, Crop};
pub use normalize::Normalize;
pub use pad::{Pad, PadMode};
pub use resize::{InterpolationMode, Letterbox, LetterboxInfo, Resize};
pub use to_tensor::ToTensor;

use crate::error::Result;
use crate::image::ImageBuffer;

/// A deterministic image transform.
///
/// All parameters are specified at construction time. There is no `fit` step
/// because inference preprocessing is deterministic (unlike training
/// augmentations or learned scalers).
pub trait ImageTransform {
    /// Apply this transform, returning a new image.
    fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer>;
}

/// A chain of image transforms applied in order.
///
/// ```ignore
/// use scry_vision::transform::{TransformPipeline, Resize, Normalize, InterpolationMode};
///
/// let pipeline = TransformPipeline::new()
///     .push(Resize::new(224, 224, InterpolationMode::Bilinear))
///     .push(Normalize::new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]));
///
/// let processed = pipeline.apply(&image)?;
/// ```
pub struct TransformPipeline {
    transforms: Vec<Box<dyn ImageTransform>>,
}

impl TransformPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Append a transform to the pipeline.
    #[must_use]
    pub fn push<T: ImageTransform + 'static>(mut self, t: T) -> Self {
        self.transforms.push(Box::new(t));
        self
    }

    /// Apply all transforms in order.
    pub fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer> {
        let mut current = image.clone();
        for t in &self.transforms {
            current = t.apply(&current)?;
        }
        Ok(current)
    }

    /// Number of transforms in the pipeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Whether the pipeline is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}
