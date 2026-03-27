// SPDX-License-Identifier: MIT OR Apache-2.0
//! Color space conversions.

use crate::error::{Result, VisionError};
use crate::image::ImageBuffer;
use crate::transform::ImageTransform;

/// Color conversion mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    /// RGB → BGR (swap red and blue channels).
    RgbToBgr,
    /// BGR → RGB (swap red and blue channels).
    BgrToRgb,
    /// RGB → Grayscale (BT.601 luminance).
    RgbToGray,
}

/// Convert between color spaces.
#[derive(Clone, Debug)]
pub struct ColorConvert {
    pub mode: ColorMode,
}

impl ColorConvert {
    #[must_use]
    pub fn new(mode: ColorMode) -> Self {
        Self { mode }
    }

    #[must_use]
    pub fn rgb_to_bgr() -> Self {
        Self::new(ColorMode::RgbToBgr)
    }

    #[must_use]
    pub fn bgr_to_rgb() -> Self {
        Self::new(ColorMode::BgrToRgb)
    }

    #[must_use]
    pub fn rgb_to_gray() -> Self {
        Self::new(ColorMode::RgbToGray)
    }
}

impl ImageTransform for ColorConvert {
    fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer> {
        match self.mode {
            ColorMode::RgbToBgr | ColorMode::BgrToRgb => swap_rb(image),
            ColorMode::RgbToGray => rgb_to_gray(image),
        }
    }
}

/// Swap red and blue channels (works for both RGB→BGR and BGR→RGB).
fn swap_rb(image: &ImageBuffer) -> Result<ImageBuffer> {
    if image.channels < 3 {
        return Err(VisionError::Transform(
            "RGB↔BGR conversion requires at least 3 channels".into(),
        ));
    }
    let ch = image.channels as usize;
    let mut data = image.data.clone();
    for pixel in data.chunks_exact_mut(ch) {
        pixel.swap(0, 2);
    }
    Ok(ImageBuffer {
        data,
        width: image.width,
        height: image.height,
        channels: image.channels,
    })
}

/// Convert RGB to grayscale using BT.601 luminance: Y = 0.299R + 0.587G + 0.114B.
fn rgb_to_gray(image: &ImageBuffer) -> Result<ImageBuffer> {
    if image.channels < 3 {
        return Err(VisionError::Transform(
            "RGB→Gray conversion requires at least 3 channels".into(),
        ));
    }
    let ch = image.channels as usize;
    let num_pixels = image.num_pixels();
    let mut data = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        let src = i * ch;
        let r = image.data[src] as f32;
        let g = image.data[src + 1] as f32;
        let b = image.data[src + 2] as f32;
        data[i] = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
    }

    Ok(ImageBuffer {
        data,
        width: image.width,
        height: image.height,
        channels: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_rgb_bgr_roundtrip() {
        let data = vec![10, 20, 30, 40, 50, 60];
        let img = ImageBuffer::from_raw(data, 2, 1, 3).unwrap();
        let bgr = ColorConvert::rgb_to_bgr().apply(&img).unwrap();
        assert_eq!(bgr.pixel(0, 0, 0), Some(30)); // B
        assert_eq!(bgr.pixel(0, 0, 1), Some(20)); // G
        assert_eq!(bgr.pixel(0, 0, 2), Some(10)); // R
        let rgb = ColorConvert::bgr_to_rgb().apply(&bgr).unwrap();
        assert_eq!(rgb.data, img.data);
    }

    #[test]
    fn rgb_to_gray_white() {
        let data = vec![255, 255, 255]; // white pixel
        let img = ImageBuffer::from_raw(data, 1, 1, 3).unwrap();
        let gray = ColorConvert::rgb_to_gray().apply(&img).unwrap();
        assert_eq!(gray.channels, 1);
        assert_eq!(gray.pixel(0, 0, 0), Some(255));
    }

    #[test]
    fn rgb_to_gray_black() {
        let data = vec![0, 0, 0];
        let img = ImageBuffer::from_raw(data, 1, 1, 3).unwrap();
        let gray = ColorConvert::rgb_to_gray().apply(&img).unwrap();
        assert_eq!(gray.pixel(0, 0, 0), Some(0));
    }

    #[test]
    fn rgb_to_gray_pure_red() {
        let data = vec![255, 0, 0];
        let img = ImageBuffer::from_raw(data, 1, 1, 3).unwrap();
        let gray = ColorConvert::rgb_to_gray().apply(&img).unwrap();
        // 0.299 * 255 ≈ 76
        assert_eq!(gray.pixel(0, 0, 0), Some(76));
    }

    #[test]
    fn color_convert_too_few_channels() {
        let img = ImageBuffer::from_raw(vec![128; 4], 2, 2, 1).unwrap();
        assert!(ColorConvert::rgb_to_bgr().apply(&img).is_err());
        assert!(ColorConvert::rgb_to_gray().apply(&img).is_err());
    }
}
