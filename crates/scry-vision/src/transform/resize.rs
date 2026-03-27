// SPDX-License-Identifier: MIT OR Apache-2.0
//! Image resize transforms.
//!
//! Provides bilinear and nearest-neighbor interpolation for resizing images
//! to a target width and height.

use crate::error::Result;
use crate::image::ImageBuffer;
use crate::transform::ImageTransform;

/// Interpolation mode for resizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Nearest-neighbor: fast, no blending.
    Nearest,
    /// Bilinear: smooth, blends 4 nearest pixels.
    Bilinear,
}

/// Resize an image to a fixed width and height.
#[derive(Clone, Debug)]
pub struct Resize {
    pub width: u32,
    pub height: u32,
    pub mode: InterpolationMode,
}

impl Resize {
    #[must_use]
    pub fn new(width: u32, height: u32, mode: InterpolationMode) -> Self {
        Self {
            width,
            height,
            mode,
        }
    }
}

impl ImageTransform for Resize {
    fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer> {
        let out = match self.mode {
            InterpolationMode::Nearest => resize_nearest(image, self.width, self.height),
            InterpolationMode::Bilinear => resize_bilinear(image, self.width, self.height),
        };
        Ok(out)
    }
}

fn resize_nearest(src: &ImageBuffer, dst_w: u32, dst_h: u32) -> ImageBuffer {
    let ch = src.channels as usize;
    let mut data = vec![0u8; dst_w as usize * dst_h as usize * ch];
    let x_ratio = src.width as f32 / dst_w as f32;
    let y_ratio = src.height as f32 / dst_h as f32;

    for dy in 0..dst_h {
        let sy = ((dy as f32 + 0.5) * y_ratio - 0.5)
            .round()
            .clamp(0.0, (src.height - 1) as f32) as u32;
        for dx in 0..dst_w {
            let sx = ((dx as f32 + 0.5) * x_ratio - 0.5)
                .round()
                .clamp(0.0, (src.width - 1) as f32) as u32;
            let src_idx = (sy as usize * src.width as usize + sx as usize) * ch;
            let dst_idx = (dy as usize * dst_w as usize + dx as usize) * ch;
            data[dst_idx..dst_idx + ch].copy_from_slice(&src.data[src_idx..src_idx + ch]);
        }
    }

    ImageBuffer {
        data,
        width: dst_w,
        height: dst_h,
        channels: src.channels,
    }
}

fn resize_bilinear(src: &ImageBuffer, dst_w: u32, dst_h: u32) -> ImageBuffer {
    let ch = src.channels as usize;
    let src_w = src.width as usize;
    let src_h = src.height as usize;
    let mut data = vec![0u8; dst_w as usize * dst_h as usize * ch];
    let x_ratio = src.width as f32 / dst_w as f32;
    let y_ratio = src.height as f32 / dst_h as f32;

    for dy in 0..dst_h as usize {
        let src_y = (dy as f32 + 0.5) * y_ratio - 0.5;
        let y0 = (src_y.floor() as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = src_y - y0 as f32;
        let fy = fy.clamp(0.0, 1.0);

        for dx in 0..dst_w as usize {
            let src_x = (dx as f32 + 0.5) * x_ratio - 0.5;
            let x0 = (src_x.floor() as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = src_x - x0 as f32;
            let fx = fx.clamp(0.0, 1.0);

            let idx00 = (y0 * src_w + x0) * ch;
            let idx10 = (y0 * src_w + x1) * ch;
            let idx01 = (y1 * src_w + x0) * ch;
            let idx11 = (y1 * src_w + x1) * ch;

            let dst_idx = (dy * dst_w as usize + dx) * ch;
            for c in 0..ch {
                let p00 = src.data[idx00 + c] as f32;
                let p10 = src.data[idx10 + c] as f32;
                let p01 = src.data[idx01 + c] as f32;
                let p11 = src.data[idx11 + c] as f32;
                let val = p00 * (1.0 - fx) * (1.0 - fy)
                    + p10 * fx * (1.0 - fy)
                    + p01 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                data[dst_idx + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    ImageBuffer {
        data,
        width: dst_w,
        height: dst_h,
        channels: src.channels,
    }
}

/// Aspect-preserving resize + padding (letterboxing).
///
/// The image is scaled to fit within `width × height` while preserving its
/// aspect ratio, then padded with `pad_value` on the shorter axis.
/// This is the standard YOLO input preprocessing step.
///
/// After applying `Letterbox`, use [`LetterboxInfo`] to undo the transform
/// on output coordinates (see [`crate::postprocess::boxes::rescale_detections`]).
#[derive(Clone, Debug)]
pub struct Letterbox {
    pub width: u32,
    pub height: u32,
    pub pad_value: u8,
    pub mode: InterpolationMode,
}

/// Information needed to undo a letterbox transform on coordinates.
#[derive(Clone, Debug)]
pub struct LetterboxInfo {
    /// Scale factor applied during resize (original → model input).
    pub scale: f32,
    /// Horizontal padding added (pixels in model input space).
    pub pad_x: f32,
    /// Vertical padding added (pixels in model input space).
    pub pad_y: f32,
}

impl LetterboxInfo {
    /// Convert from model input coordinates back to original image coordinates.
    pub fn unscale(&self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale)
    }
}

impl Letterbox {
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pad_value: 114,
            mode: InterpolationMode::Bilinear,
        }
    }

    /// Apply letterboxing, returning the padded image and transform metadata.
    pub fn apply_with_info(&self, image: &ImageBuffer) -> Result<(ImageBuffer, LetterboxInfo)> {
        let scale_x = self.width as f32 / image.width as f32;
        let scale_y = self.height as f32 / image.height as f32;
        let scale = scale_x.min(scale_y);

        let new_w = (image.width as f32 * scale).round() as u32;
        let new_h = (image.height as f32 * scale).round() as u32;

        // Resize to fit
        let resized = match self.mode {
            InterpolationMode::Nearest => resize_nearest(image, new_w, new_h),
            InterpolationMode::Bilinear => resize_bilinear(image, new_w, new_h),
        };

        let pad_x = (self.width - new_w) as f32 / 2.0;
        let pad_y = (self.height - new_h) as f32 / 2.0;
        let pad_left = pad_x.floor() as u32;
        let pad_top = pad_y.floor() as u32;
        let pad_right = self.width - new_w - pad_left;
        let pad_bottom = self.height - new_h - pad_top;

        // Pad to target size
        let ch = resized.channels as usize;
        let mut data = vec![self.pad_value; self.width as usize * self.height as usize * ch];

        for row in 0..new_h as usize {
            let src_start = row * new_w as usize * ch;
            let dst_start =
                ((pad_top as usize + row) * self.width as usize + pad_left as usize) * ch;
            let row_bytes = new_w as usize * ch;
            data[dst_start..dst_start + row_bytes]
                .copy_from_slice(&resized.data[src_start..src_start + row_bytes]);
        }

        let output = ImageBuffer::from_raw(data, self.width, self.height, resized.channels)?;
        let info = LetterboxInfo {
            scale,
            pad_x: pad_left as f32,
            pad_y: pad_top as f32,
        };

        let _ = pad_right;
        let _ = pad_bottom;

        Ok((output, info))
    }
}

impl ImageTransform for Letterbox {
    fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer> {
        let (output, _info) = self.apply_with_info(image)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2x2_rgb() -> ImageBuffer {
        // 2×2 RGB: red, green, blue, white
        #[rustfmt::skip]
        let data = vec![
            255, 0, 0,    0, 255, 0,
            0, 0, 255,    255, 255, 255,
        ];
        ImageBuffer::from_raw(data, 2, 2, 3).unwrap()
    }

    #[test]
    fn resize_nearest_identity() {
        let img = make_2x2_rgb();
        let resized = resize_nearest(&img, 2, 2);
        assert_eq!(resized.data, img.data);
    }

    #[test]
    fn resize_nearest_upscale() {
        let img = make_2x2_rgb();
        let resized = resize_nearest(&img, 4, 4);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);
        assert_eq!(resized.byte_len(), 4 * 4 * 3);
        // Top-left 2×2 block should be red
        assert_eq!(resized.pixel(0, 0, 0), Some(255));
        assert_eq!(resized.pixel(0, 0, 1), Some(0));
    }

    #[test]
    fn resize_bilinear_identity() {
        let img = make_2x2_rgb();
        let resized = resize_bilinear(&img, 2, 2);
        // Bilinear at same size should closely match the original
        for i in 0..img.data.len() {
            let diff = (resized.data[i] as i16 - img.data[i] as i16).unsigned_abs();
            assert!(diff <= 1, "pixel {i}: {} vs {}", resized.data[i], img.data[i]);
        }
    }

    #[test]
    fn resize_bilinear_output_dimensions() {
        let img = make_2x2_rgb();
        let resized = resize_bilinear(&img, 10, 8);
        assert_eq!(resized.width, 10);
        assert_eq!(resized.height, 8);
        assert_eq!(resized.byte_len(), 10 * 8 * 3);
    }

    #[test]
    fn resize_transform_trait() {
        let img = make_2x2_rgb();
        let transform = Resize::new(6, 6, InterpolationMode::Bilinear);
        let out = transform.apply(&img).unwrap();
        assert_eq!(out.width, 6);
        assert_eq!(out.height, 6);
    }

    #[test]
    fn letterbox_square_input() {
        let img = make_2x2_rgb();
        let lb = Letterbox::new(4, 4);
        let (out, info) = lb.apply_with_info(&img).unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
        // Square input → square target, scale = 2.0, no padding
        assert!((info.scale - 2.0).abs() < 1e-6);
        assert!((info.pad_x).abs() < 1e-6);
        assert!((info.pad_y).abs() < 1e-6);
    }

    #[test]
    fn letterbox_wide_input() {
        // 6×2 image → 4×4 target
        let img = ImageBuffer::from_raw(vec![128u8; 6 * 2 * 3], 6, 2, 3).unwrap();
        let lb = Letterbox::new(4, 4);
        let (out, info) = lb.apply_with_info(&img).unwrap();
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
        // scale = min(4/6, 4/2) = min(0.667, 2.0) = 0.667
        assert!((info.scale - 4.0 / 6.0).abs() < 0.01);
        // Resized: 4×1 (rounded), padded vertically
        assert!(info.pad_y > 0.0);
    }

    #[test]
    fn letterbox_trait_impl() {
        let img = make_2x2_rgb();
        let lb = Letterbox::new(8, 8);
        let out = lb.apply(&img).unwrap();
        assert_eq!(out.width, 8);
        assert_eq!(out.height, 8);
    }

    #[test]
    fn letterbox_unscale_roundtrip() {
        let info = LetterboxInfo {
            scale: 2.0,
            pad_x: 10.0,
            pad_y: 5.0,
        };
        let (x, y) = info.unscale(50.0, 25.0);
        assert!((x - 20.0).abs() < 1e-6); // (50-10)/2
        assert!((y - 10.0).abs() < 1e-6); // (25-5)/2
    }
}
