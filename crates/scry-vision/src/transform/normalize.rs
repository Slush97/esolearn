// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-channel mean/std normalization.
//!
//! Normalizes pixel values from u8 [0, 255] to f32 using `(pixel/255 - mean) / std`,
//! then maps the result back to u8 [0, 255] range for storage in [`ImageBuffer`].
//!
//! For the actual float-domain normalization that models consume, see [`ToTensor`]
//! which handles the u8→f32 conversion. This transform is useful when you need
//! to chain multiple u8-domain transforms after normalization.
//!
//! Most model pipelines should use [`ToTensor`] with `normalize` parameters instead
//! of this transform, since it avoids the lossy u8 round-trip.

use crate::error::Result;
use crate::image::ImageBuffer;
use crate::transform::ImageTransform;

/// Per-channel mean/std normalization in the u8 domain.
///
/// Applies `out = clamp((pixel/255 - mean) / std * 255, 0, 255)` per channel.
/// This is a lossy operation due to u8 quantization. For lossless normalization,
/// use [`ToTensor`] with `mean`/`std` parameters directly.
#[derive(Clone, Debug)]
pub struct Normalize {
    /// Per-channel mean values (in 0..1 range).
    pub mean: [f32; 3],
    /// Per-channel std values (in 0..1 range).
    pub std: [f32; 3],
}

impl Normalize {
    /// Create a new normalizer with given mean and std (both in 0..1 range).
    #[must_use]
    pub fn new(mean: [f32; 3], std: [f32; 3]) -> Self {
        Self { mean, std }
    }

    /// ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    #[must_use]
    pub fn imagenet() -> Self {
        Self::new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    }

    /// CLIP normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711].
    #[must_use]
    pub fn clip() -> Self {
        Self::new(
            [0.481_454_66, 0.457_827_5, 0.408_210_73],
            [0.268_629_54, 0.261_302_58, 0.275_777_11],
        )
    }
}

impl ImageTransform for Normalize {
    fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer> {
        let ch = image.channels as usize;
        let mut data = image.data.clone();

        for pixel in data.chunks_exact_mut(ch) {
            for c in 0..ch.min(3) {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - self.mean[c]) / self.std[c];
                // Map back to 0..255 for u8 storage
                pixel[c] = (normalized * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(ImageBuffer {
            data,
            width: image.width,
            height: image.height,
            channels: image.channels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_imagenet_does_not_panic() {
        let img = ImageBuffer::zeros(4, 4, 3);
        let norm = Normalize::imagenet();
        let out = norm.apply(&img).unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
    }

    #[test]
    fn normalize_preserves_dimensions() {
        let img = ImageBuffer::from_raw(vec![128; 3 * 2 * 2], 2, 2, 3).unwrap();
        let norm = Normalize::new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        let out = norm.apply(&img).unwrap();
        assert_eq!(out.width, img.width);
        assert_eq!(out.height, img.height);
        assert_eq!(out.channels, img.channels);
    }
}
