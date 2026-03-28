// SPDX-License-Identifier: MIT OR Apache-2.0
//! Box blur via integral image for O(1)-per-pixel averaging.

use crate::error::{Result, ScryVisionError};
use crate::image::{Gray, ImageBuf};
use crate::integral::IntegralImage;

/// Apply a box blur (mean filter) of the given radius.
///
/// Window size is `(2*radius + 1) x (2*radius + 1)`.
/// Uses an integral image for O(1) per-pixel cost.
pub fn box_blur(img: &ImageBuf<f32, Gray>, radius: u32) -> Result<ImageBuf<f32, Gray>> {
    if radius == 0 {
        return Err(ScryVisionError::InvalidParameter(
            "box blur radius must be >= 1".into(),
        ));
    }
    let w = img.width();
    let h = img.height();
    let sat = IntegralImage::from_gray_f32(img);
    let r = radius as i32;
    let mut out = vec![0.0f32; w as usize * h as usize];

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let x0 = (x - r).max(0) as u32;
            let y0 = (y - r).max(0) as u32;
            let x1 = (x + r).min(w as i32 - 1) as u32;
            let y1 = (y + r).min(h as i32 - 1) as u32;
            out[y as usize * w as usize + x as usize] =
                sat.rect_mean(x0, y0, x1, y1) as f32;
        }
    }

    ImageBuf::from_vec(out, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_image_unchanged() {
        let data = vec![0.5f32; 16];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 4, 4).unwrap();
        let blurred = box_blur(&img, 1).unwrap();
        for &v in blurred.as_slice() {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn box_blur_smooths() {
        let mut data = vec![0.0f32; 8 * 8];
        data[4 * 8 + 4] = 1.0; // single bright pixel
        let img = ImageBuf::<f32, Gray>::from_vec(data, 8, 8).unwrap();
        let blurred = box_blur(&img, 1).unwrap();

        // The bright pixel should be spread out
        let center = blurred.pixel(4, 4)[0];
        assert!(center < 1.0, "center should be reduced from 1.0: {center}");
        assert!(center > 0.0, "center should still be positive: {center}");
    }
}
