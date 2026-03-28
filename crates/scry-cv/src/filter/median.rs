// SPDX-License-Identifier: MIT OR Apache-2.0
//! Median filter for salt-and-pepper noise removal.

use crate::error::{Result, ScryVisionError};
use crate::image::{Gray, ImageBuf};

/// Apply a median filter with the given radius.
///
/// Window size is `(2*radius + 1) x (2*radius + 1)`.
pub fn median_filter(img: &ImageBuf<f32, Gray>, radius: u32) -> Result<ImageBuf<f32, Gray>> {
    if radius == 0 {
        return Err(ScryVisionError::InvalidParameter(
            "median radius must be >= 1".into(),
        ));
    }
    let w = img.width();
    let h = img.height();
    let r = radius as i32;
    let mut out = vec![0.0f32; w as usize * h as usize];
    let cap = ((2 * radius + 1) * (2 * radius + 1)) as usize;
    let mut window = Vec::with_capacity(cap);

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            window.clear();
            for wy in -r..=r {
                for wx in -r..=r {
                    let sx = (x + wx).clamp(0, w as i32 - 1) as u32;
                    let sy = (y + wy).clamp(0, h as i32 - 1) as u32;
                    window.push(img.pixel(sx, sy)[0]);
                }
            }
            // Partial sort to find median
            let mid = window.len() / 2;
            window.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            out[y as usize * w as usize + x as usize] = window[mid];
        }
    }

    ImageBuf::from_vec(out, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_impulse_noise() {
        let mut data = vec![0.5f32; 7 * 7];
        // Add salt-and-pepper
        data[3 * 7 + 3] = 1.0;
        data[3 * 7 + 4] = 0.0;
        let img = ImageBuf::<f32, Gray>::from_vec(data, 7, 7).unwrap();
        let filtered = median_filter(&img, 1).unwrap();

        // Center pixels should be close to 0.5 (median of mostly 0.5 values)
        let c = filtered.pixel(3, 3)[0];
        assert!((c - 0.5).abs() < 0.01, "median should restore: got {c}");
    }

    #[test]
    fn uniform_image_unchanged() {
        let data = vec![0.3f32; 25];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 5, 5).unwrap();
        let filtered = median_filter(&img, 1).unwrap();
        for &v in filtered.as_slice() {
            assert!((v - 0.3).abs() < 1e-5);
        }
    }
}
