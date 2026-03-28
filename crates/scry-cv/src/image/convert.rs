// SPDX-License-Identifier: MIT OR Apache-2.0
//! Color space and channel layout conversions.

use crate::error::Result;
use crate::image::buf::ImageBuf;
use crate::image::pixel::{Gray, Pixel, Rgb, Rgba};

/// ITU-R BT.709 luminance coefficients.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

// ── RGB → Gray ──

impl<T: Pixel> ImageBuf<T, Rgb> {
    /// Convert to grayscale using BT.709 luminance.
    pub fn to_gray(&self) -> ImageBuf<T, Gray> {
        let mut out = Vec::with_capacity(self.width() as usize * self.height() as usize);
        for y in 0..self.height() {
            for x in 0..self.width() {
                let px = self.pixel(x, y);
                let r = px[0].to_f32();
                let g = px[1].to_f32();
                let b = px[2].to_f32();
                out.push(T::from_f32(LUMA_R * r + LUMA_G * g + LUMA_B * b));
            }
        }
        ImageBuf::from_vec(out, self.width(), self.height())
            .expect("dimensions match by construction")
    }
}

// ── RGBA → Gray ──

impl<T: Pixel> ImageBuf<T, Rgba> {
    /// Convert to grayscale using BT.709 luminance (alpha is ignored).
    pub fn to_gray(&self) -> ImageBuf<T, Gray> {
        let mut out = Vec::with_capacity(self.width() as usize * self.height() as usize);
        for y in 0..self.height() {
            for x in 0..self.width() {
                let px = self.pixel(x, y);
                let r = px[0].to_f32();
                let g = px[1].to_f32();
                let b = px[2].to_f32();
                out.push(T::from_f32(LUMA_R * r + LUMA_G * g + LUMA_B * b));
            }
        }
        ImageBuf::from_vec(out, self.width(), self.height())
            .expect("dimensions match by construction")
    }

    /// Convert to RGB by dropping the alpha channel.
    pub fn to_rgb(&self) -> ImageBuf<T, Rgb> {
        let mut out = Vec::with_capacity(self.width() as usize * self.height() as usize * 3);
        for y in 0..self.height() {
            for x in 0..self.width() {
                let px = self.pixel(x, y);
                out.push(px[0]);
                out.push(px[1]);
                out.push(px[2]);
            }
        }
        ImageBuf::from_vec(out, self.width(), self.height())
            .expect("dimensions match by construction")
    }
}

// ── Gray → RGB ──

impl<T: Pixel> ImageBuf<T, Gray> {
    /// Convert grayscale to RGB by tripling each value.
    pub fn to_rgb(&self) -> ImageBuf<T, Rgb> {
        let mut out = Vec::with_capacity(self.width() as usize * self.height() as usize * 3);
        for &v in self.as_slice() {
            out.push(v);
            out.push(v);
            out.push(v);
        }
        ImageBuf::from_vec(out, self.width(), self.height())
            .expect("dimensions match by construction")
    }

    /// Convert grayscale to RGBA (opaque).
    pub fn to_rgba(&self, alpha: T) -> ImageBuf<T, Rgba> {
        let mut out = Vec::with_capacity(self.width() as usize * self.height() as usize * 4);
        for &v in self.as_slice() {
            out.push(v);
            out.push(v);
            out.push(v);
            out.push(alpha);
        }
        ImageBuf::from_vec(out, self.width(), self.height())
            .expect("dimensions match by construction")
    }
}

// ── RGB → RGBA ──

impl<T: Pixel> ImageBuf<T, Rgb> {
    /// Convert RGB to RGBA with the given alpha value.
    pub fn to_rgba(&self, alpha: T) -> ImageBuf<T, Rgba> {
        let mut out = Vec::with_capacity(self.width() as usize * self.height() as usize * 4);
        for y in 0..self.height() {
            for x in 0..self.width() {
                let px = self.pixel(x, y);
                out.push(px[0]);
                out.push(px[1]);
                out.push(px[2]);
                out.push(alpha);
            }
        }
        ImageBuf::from_vec(out, self.width(), self.height())
            .expect("dimensions match by construction")
    }
}

/// Create a grayscale `f32` image from a raw `u8` buffer (e.g. camera input).
///
/// Each byte is mapped to \[0.0, 1.0\].
pub fn gray_from_u8_slice(data: &[u8], width: u32, height: u32) -> Result<ImageBuf<f32, Gray>> {
    let expected = width as usize * height as usize;
    if data.len() != expected {
        return Err(crate::error::ScryVisionError::BufferSizeMismatch {
            expected,
            got: data.len(),
        });
    }
    let fdata: Vec<f32> = data.iter().map(|&v| v as f32 / 255.0).collect();
    ImageBuf::from_vec(fdata, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_gray_white() {
        let data = vec![255u8, 255, 255, 255, 255, 255];
        let img = ImageBuf::<u8, Rgb>::from_vec(data, 2, 1).unwrap();
        let gray = img.to_gray();
        assert_eq!(gray.pixel(0, 0), &[255]);
        assert_eq!(gray.pixel(1, 0), &[255]);
    }

    #[test]
    fn gray_to_rgb_round_trip() {
        let data = vec![100u8, 200];
        let gray = ImageBuf::<u8, Gray>::from_vec(data, 2, 1).unwrap();
        let rgb = gray.to_rgb();
        assert_eq!(rgb.pixel(0, 0), &[100, 100, 100]);
        assert_eq!(rgb.pixel(1, 0), &[200, 200, 200]);
    }

    #[test]
    fn rgba_to_gray() {
        // Pure red: luminance = 0.2126
        let data = vec![255u8, 0, 0, 255];
        let img = ImageBuf::<u8, Rgba>::from_vec(data, 1, 1).unwrap();
        let gray = img.to_gray();
        let v = gray.pixel(0, 0)[0];
        assert!((v as f32 - 54.0).abs() < 2.0, "expected ~54, got {v}");
    }

    #[test]
    fn gray_from_u8_slice_works() {
        let data = vec![0u8, 128, 255];
        let img = gray_from_u8_slice(&data, 3, 1).unwrap();
        assert!((img.pixel(2, 0)[0] - 1.0).abs() < 1e-5);
    }
}
