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
}
