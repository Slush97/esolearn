// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integral image (summed-area table) for O(1) rectangle sum queries.

use crate::image::{Gray, ImageBuf};

/// Integral image stored as `f64` to avoid precision loss on large images.
///
/// `sat[y][x]` = sum of all pixels in the rectangle `(0,0)..(x,y)` inclusive.
/// Stored with 1-pixel border of zeros for simpler boundary handling,
/// so dimensions are `(width+1) x (height+1)`.
pub struct IntegralImage {
    data: Vec<f64>,
    /// Width of the *padded* table (= source width + 1).
    w: usize,
}

impl IntegralImage {
    /// Build an integral image from a grayscale f32 buffer.
    pub fn from_gray_f32(img: &ImageBuf<f32, Gray>) -> Self {
        let iw = img.width() as usize;
        let ih = img.height() as usize;
        let w = iw + 1;
        let h = ih + 1;
        let mut data = vec![0.0f64; w * h];

        for y in 0..ih {
            let mut row_sum = 0.0f64;
            for x in 0..iw {
                row_sum += img.pixel(x as u32, y as u32)[0] as f64;
                data[(y + 1) * w + (x + 1)] = row_sum + data[y * w + (x + 1)];
            }
        }

        Self { data, w }
    }

    /// Build an integral image from a grayscale u8 buffer.
    pub fn from_gray_u8(img: &ImageBuf<u8, Gray>) -> Self {
        let iw = img.width() as usize;
        let ih = img.height() as usize;
        let w = iw + 1;
        let h = ih + 1;
        let mut data = vec![0.0f64; w * h];

        for y in 0..ih {
            let mut row_sum = 0.0f64;
            for x in 0..iw {
                row_sum += img.pixel(x as u32, y as u32)[0] as f64;
                data[(y + 1) * w + (x + 1)] = row_sum + data[y * w + (x + 1)];
            }
        }

        Self { data, w }
    }

    /// Sum of pixel values in the rectangle from `(x0, y0)` to `(x1, y1)` inclusive.
    ///
    /// Coordinates are in the *original* image space (not the padded table).
    #[inline]
    pub fn rect_sum(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
        let (x0, y0, x1, y1) = (x0 as usize, y0 as usize, x1 as usize + 1, y1 as usize + 1);
        self.data[y1 * self.w + x1] - self.data[y0 * self.w + x1]
            - self.data[y1 * self.w + x0]
            + self.data[y0 * self.w + x0]
    }

    /// Mean pixel value in the rectangle from `(x0, y0)` to `(x1, y1)` inclusive.
    #[inline]
    pub fn rect_mean(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
        let area = (x1 - x0 + 1) as f64 * (y1 - y0 + 1) as f64;
        self.rect_sum(x0, y0, x1, y1) / area
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image() -> ImageBuf<f32, Gray> {
        // 3x3 image:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        ImageBuf::from_vec(data, 3, 3).unwrap()
    }

    #[test]
    fn full_image_sum() {
        let img = make_test_image();
        let sat = IntegralImage::from_gray_f32(&img);
        let sum = sat.rect_sum(0, 0, 2, 2);
        assert!((sum - 45.0).abs() < 1e-10, "expected 45, got {sum}");
    }

    #[test]
    fn single_pixel() {
        let img = make_test_image();
        let sat = IntegralImage::from_gray_f32(&img);
        assert!((sat.rect_sum(1, 1, 1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn sub_rect() {
        let img = make_test_image();
        let sat = IntegralImage::from_gray_f32(&img);
        // bottom-right 2x2: [5,6,8,9] = 28
        let sum = sat.rect_sum(1, 1, 2, 2);
        assert!((sum - 28.0).abs() < 1e-10, "expected 28, got {sum}");
    }

    #[test]
    fn rect_mean() {
        let img = make_test_image();
        let sat = IntegralImage::from_gray_f32(&img);
        let mean = sat.rect_mean(0, 0, 2, 2);
        assert!((mean - 5.0).abs() < 1e-10, "expected 5.0, got {mean}");
    }
}
