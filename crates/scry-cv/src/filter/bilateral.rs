// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bilateral filter — edge-preserving smoothing.

use crate::error::{Result, ScryVisionError};
use crate::image::{Gray, ImageBuf};

/// Apply a bilateral filter to a grayscale f32 image.
///
/// - `sigma_space`: spatial Gaussian sigma (controls kernel size)
/// - `sigma_color`: color Gaussian sigma (controls edge preservation)
///
/// Smaller `sigma_color` preserves edges more aggressively.
pub fn bilateral_filter(
    img: &ImageBuf<f32, Gray>,
    sigma_space: f32,
    sigma_color: f32,
) -> Result<ImageBuf<f32, Gray>> {
    if sigma_space <= 0.0 || sigma_color <= 0.0 {
        return Err(ScryVisionError::InvalidParameter(
            "bilateral sigma values must be positive".into(),
        ));
    }
    let w = img.width();
    let h = img.height();
    let r = (3.0 * sigma_space).ceil() as i32;
    let s2_space = 2.0 * sigma_space * sigma_space;
    let s2_color = 2.0 * sigma_color * sigma_color;
    let mut out = vec![0.0f32; w as usize * h as usize];

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let center = img.pixel(x as u32, y as u32)[0];
            let mut weight_sum = 0.0f32;
            let mut val_sum = 0.0f32;

            for wy in -r..=r {
                for wx in -r..=r {
                    let sx = (x + wx).clamp(0, w as i32 - 1) as u32;
                    let sy = (y + wy).clamp(0, h as i32 - 1) as u32;
                    let neighbor = img.pixel(sx, sy)[0];

                    let dist_sq = (wx * wx + wy * wy) as f32;
                    let color_diff = center - neighbor;
                    let w_space = (-dist_sq / s2_space).exp();
                    let w_color = (-color_diff * color_diff / s2_color).exp();
                    let w = w_space * w_color;

                    weight_sum += w;
                    val_sum += w * neighbor;
                }
            }

            out[y as usize * w as usize + x as usize] = if weight_sum > 1e-7 {
                val_sum / weight_sum
            } else {
                center
            };
        }
    }

    ImageBuf::from_vec(out, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_step_edge() {
        // Left half = 0, right half = 1
        let mut data = vec![0.0f32; 16 * 16];
        for y in 0..16u32 {
            for x in 8..16u32 {
                data[(y * 16 + x) as usize] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 16, 16).unwrap();
        let filtered = bilateral_filter(&img, 3.0, 0.1).unwrap();

        // Interior of each half should remain close to original
        let left = filtered.pixel(2, 8)[0];
        let right = filtered.pixel(13, 8)[0];
        assert!(left < 0.1, "left side should stay dark: {left}");
        assert!(right > 0.9, "right side should stay bright: {right}");
    }
}
