// SPDX-License-Identifier: MIT OR Apache-2.0
//! Convert an [`ImageBuffer`] to a [`Tensor<B>`].
//!
//! This is the bridge from the pixel domain (u8, HWC) to the tensor domain
//! (f32, CHW) that models consume. It is intentionally *not* an
//! [`ImageTransform`] because it crosses a type boundary.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::image::ImageBuffer;

/// Convert an [`ImageBuffer`] (HWC u8) to a [`Tensor<B>`] (CHW f32).
///
/// Performs the HWC → CHW transpose and optional scaling / normalization:
///
/// - If `scale` is true, pixel values are divided by 255.0 to produce 0..1 range.
/// - If `mean`/`std` are set, normalization is applied *after* scaling:
///   `out[c] = (pixel[c] / 255.0 - mean[c]) / std[c]`
///
/// This is the preferred way to get normalized tensors for model input,
/// since it avoids the lossy u8 round-trip of the [`Normalize`] transform.
#[derive(Clone, Debug)]
pub struct ToTensor {
    /// Divide pixel values by 255.0.
    pub scale: bool,
    /// Per-channel mean for normalization (applied after scaling). `None` = no normalization.
    pub mean: Option<[f32; 3]>,
    /// Per-channel std for normalization (applied after scaling). `None` = no normalization.
    pub std: Option<[f32; 3]>,
}

impl ToTensor {
    /// Create a `ToTensor` that scales to 0..1 with no normalization.
    #[must_use]
    pub fn new(scale: bool) -> Self {
        Self {
            scale,
            mean: None,
            std: None,
        }
    }

    /// Create a `ToTensor` with per-channel normalization.
    ///
    /// Equivalent to `torchvision.transforms.Normalize(mean, std)` applied
    /// after `ToTensor()`.
    #[must_use]
    pub fn normalized(mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            scale: true,
            mean: Some(mean),
            std: Some(std),
        }
    }

    /// ImageNet normalization preset.
    #[must_use]
    pub fn imagenet() -> Self {
        Self::normalized([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    }

    /// CLIP normalization preset.
    #[must_use]
    pub fn clip() -> Self {
        Self::normalized(
            [0.481_454_66, 0.457_827_5, 0.408_210_73],
            [0.268_629_54, 0.261_302_58, 0.275_777_11],
        )
    }

    /// Convert the image to a tensor.
    ///
    /// Input: `ImageBuffer` with shape `[H, W, C]` (u8).
    /// Output: `Tensor<B>` with shape `[C, H, W]` (f32).
    pub fn apply<B: MathBackend>(&self, image: &ImageBuffer) -> Tensor<B> {
        let h = image.height as usize;
        let w = image.width as usize;
        let c = image.channels as usize;
        let num_pixels = h * w;

        // HWC → CHW transpose + scale/normalize in one pass
        let mut data = vec![0.0f32; c * num_pixels];

        let scale = if self.scale { 1.0 / 255.0 } else { 1.0 };

        for y in 0..h {
            for x in 0..w {
                let src_idx = (y * w + x) * c;
                let dst_pixel = y * w + x;
                for ch in 0..c {
                    let mut val = image.data[src_idx + ch] as f32 * scale;
                    if let (Some(mean), Some(std)) = (&self.mean, &self.std) {
                        if ch < 3 {
                            val = (val - mean[ch]) / std[ch];
                        }
                    }
                    data[ch * num_pixels + dst_pixel] = val;
                }
            }
        }

        Tensor::from_vec(data, Shape::new(&[c, h, w]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn to_tensor_shape() {
        let img = ImageBuffer::from_raw(vec![128u8; 3 * 4 * 2], 4, 2, 3).unwrap();
        let tensor: Tensor<CpuBackend> = ToTensor::new(true).apply(&img);
        assert_eq!(tensor.shape.dims(), &[3, 2, 4]); // CHW
        assert_eq!(tensor.numel(), 3 * 2 * 4);
    }

    #[test]
    fn to_tensor_scale() {
        // Single pixel, RGB = [255, 0, 128]
        let img = ImageBuffer::from_raw(vec![255, 0, 128], 1, 1, 3).unwrap();
        let tensor: Tensor<CpuBackend> = ToTensor::new(true).apply(&img);
        let data = tensor.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-6); // R channel: 255/255 = 1.0
        assert!((data[1]).abs() < 1e-6); // G channel: 0/255 = 0.0
        assert!((data[2] - 128.0 / 255.0).abs() < 1e-4); // B channel
    }

    #[test]
    fn to_tensor_no_scale() {
        let img = ImageBuffer::from_raw(vec![200, 100, 50], 1, 1, 3).unwrap();
        let tensor: Tensor<CpuBackend> = ToTensor::new(false).apply(&img);
        let data = tensor.to_vec();
        assert!((data[0] - 200.0).abs() < 1e-6);
        assert!((data[1] - 100.0).abs() < 1e-6);
        assert!((data[2] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn to_tensor_hwc_to_chw_layout() {
        // 2×1 image: pixel0=[10,20,30], pixel1=[40,50,60]
        let img = ImageBuffer::from_raw(vec![10, 20, 30, 40, 50, 60], 2, 1, 3).unwrap();
        let tensor: Tensor<CpuBackend> = ToTensor::new(false).apply(&img);
        let data = tensor.to_vec();
        // CHW: channel 0 = [10, 40], channel 1 = [20, 50], channel 2 = [30, 60]
        assert!((data[0] - 10.0).abs() < 1e-6);
        assert!((data[1] - 40.0).abs() < 1e-6);
        assert!((data[2] - 20.0).abs() < 1e-6);
        assert!((data[3] - 50.0).abs() < 1e-6);
        assert!((data[4] - 30.0).abs() < 1e-6);
        assert!((data[5] - 60.0).abs() < 1e-6);
    }

    #[test]
    fn to_tensor_normalized() {
        // Single white pixel
        let img = ImageBuffer::from_raw(vec![255, 255, 255], 1, 1, 3).unwrap();
        let tensor: Tensor<CpuBackend> = ToTensor::normalized(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        )
        .apply(&img);
        let data = tensor.to_vec();
        // (1.0 - 0.5) / 0.5 = 1.0
        for &v in &data {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }
}
