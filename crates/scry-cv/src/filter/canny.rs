// SPDX-License-Identifier: MIT OR Apache-2.0
//! Canny edge detection.
//!
//! Classic multi-stage edge detector: Gaussian smoothing, Sobel gradients,
//! non-maximum suppression, and hysteresis thresholding.
//!
//! Reference: J. Canny, "A Computational Approach to Edge Detection" (1986).

use crate::error::{Result, ScryVisionError};
use crate::filter::gaussian::gaussian_blur;
use crate::filter::sobel::{sobel_x, sobel_y};
use crate::image::{Gray, ImageBuf};

/// Detect edges using the Canny algorithm with default Gaussian sigma (1.4).
///
/// Returns a binary f32 image: 1.0 for edge pixels, 0.0 for non-edge.
///
/// # Arguments
/// * `img` — Input grayscale image.
/// * `low` — Low hysteresis threshold (weak edges below this are discarded).
/// * `high` — High hysteresis threshold (strong edges above this seed the trace).
///
/// # Example
///
/// ```
/// use scry_cv::prelude::*;
/// use scry_cv::filter::canny::canny;
///
/// let img = GrayImageF::new(32, 32).unwrap();
/// let edges = canny(&img, 0.1, 0.3).unwrap();
/// assert_eq!(edges.dimensions(), (32, 32));
/// ```
pub fn canny(
    img: &ImageBuf<f32, Gray>,
    low: f32,
    high: f32,
) -> Result<ImageBuf<f32, Gray>> {
    canny_with_sigma(img, low, high, 1.4)
}

/// Detect edges using the Canny algorithm with a custom Gaussian sigma.
///
/// See [`canny`] for details.
pub fn canny_with_sigma(
    img: &ImageBuf<f32, Gray>,
    low: f32,
    high: f32,
    sigma: f32,
) -> Result<ImageBuf<f32, Gray>> {
    if low < 0.0 {
        return Err(ScryVisionError::InvalidParameter(
            "low threshold must be >= 0".into(),
        ));
    }
    if high < low {
        return Err(ScryVisionError::InvalidParameter(
            "high threshold must be >= low threshold".into(),
        ));
    }
    if sigma <= 0.0 {
        return Err(ScryVisionError::InvalidParameter(
            "sigma must be > 0".into(),
        ));
    }

    let w = img.width() as usize;
    let h = img.height() as usize;
    if w < 3 || h < 3 {
        return ImageBuf::new(img.width(), img.height());
    }

    // 1. Gaussian blur
    let blurred = gaussian_blur(img, sigma)?;

    // 2. Sobel gradients
    let gx = sobel_x(&blurred)?;
    let gy = sobel_y(&blurred)?;
    let gx_data = gx.as_slice();
    let gy_data = gy.as_slice();

    // 3. Gradient magnitude and direction
    let mut magnitude = vec![0.0f32; w * h];
    let mut direction = vec![0.0f32; w * h];
    for i in 0..w * h {
        magnitude[i] = gx_data[i].hypot(gy_data[i]);
        direction[i] = gy_data[i].atan2(gx_data[i]);
    }

    // 4. Non-maximum suppression
    let nms = non_maximum_suppression(&magnitude, &direction, w, h);

    // 5. Hysteresis thresholding
    let edges = hysteresis(&nms, low, high, w, h);

    ImageBuf::from_vec(edges, img.width(), img.height())
}

/// Thin edges by suppressing non-maximum gradient magnitudes.
///
/// For each pixel, quantize the gradient direction to one of 4 axes
/// (0, 45, 90, 135 degrees) and keep the pixel only if it is a local
/// maximum along that axis.
fn non_maximum_suppression(
    mag: &[f32],
    dir: &[f32],
    w: usize,
    h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let m = mag[idx];
            if m == 0.0 {
                continue;
            }

            // Quantize angle to nearest 0, 45, 90, or 135 degrees.
            // atan2 returns [-pi, pi]; map to [0, pi) for direction.
            let angle = dir[idx].rem_euclid(std::f32::consts::PI);

            // Neighbor offsets along the gradient direction:
            //   0°   → compare left/right  (dx=1, dy=0)
            //   45°  → compare NE/SW       (dx=1, dy=-1)
            //   90°  → compare up/down     (dx=0, dy=1)
            //   135° → compare NW/SE       (dx=-1, dy=-1)
            let (n1, n2) = if angle < std::f32::consts::FRAC_PI_8
                || angle >= 7.0 * std::f32::consts::FRAC_PI_8
            {
                // ~0° horizontal
                (mag[y * w + x + 1], mag[y * w + x - 1])
            } else if angle < 3.0 * std::f32::consts::FRAC_PI_8 {
                // ~45° diagonal
                (mag[(y - 1) * w + x + 1], mag[(y + 1) * w + x - 1])
            } else if angle < 5.0 * std::f32::consts::FRAC_PI_8 {
                // ~90° vertical
                (mag[(y - 1) * w + x], mag[(y + 1) * w + x])
            } else {
                // ~135° diagonal
                (mag[(y - 1) * w + x - 1], mag[(y + 1) * w + x + 1])
            };

            if m >= n1 && m >= n2 {
                out[idx] = m;
            }
        }
    }

    out
}

/// Two-threshold hysteresis: keep strong edges and weak edges connected to them.
///
/// Uses a stack-based flood fill from strong pixels through weak pixels.
fn hysteresis(
    nms: &[f32],
    low: f32,
    high: f32,
    w: usize,
    h: usize,
) -> Vec<f32> {
    const STRONG: u8 = 2;
    const WEAK: u8 = 1;

    let mut label = vec![0u8; w * h];
    let mut stack: Vec<usize> = Vec::new();

    // Classify pixels
    for i in 0..w * h {
        if nms[i] >= high {
            label[i] = STRONG;
            stack.push(i);
        } else if nms[i] >= low {
            label[i] = WEAK;
        }
    }

    // Flood-fill: promote weak pixels connected to strong pixels
    while let Some(idx) = stack.pop() {
        let x = idx % w;
        let y = idx / w;

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                    continue;
                }
                let nidx = ny as usize * w + nx as usize;
                if label[nidx] == WEAK {
                    label[nidx] = STRONG;
                    stack.push(nidx);
                }
            }
        }
    }

    // Output binary image
    label
        .iter()
        .map(|&l| if l == STRONG { 1.0 } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_edges_on_uniform_image() {
        let img = ImageBuf::<f32, Gray>::from_vec(vec![0.5; 32 * 32], 32, 32).unwrap();
        let edges = canny(&img, 0.1, 0.3).unwrap();
        assert!(
            edges.as_slice().iter().all(|&v| v == 0.0),
            "uniform image should have no edges"
        );
    }

    #[test]
    fn detects_rectangle_edges() {
        // Bright rectangle on dark background
        let (w, h) = (48u32, 48u32);
        let mut data = vec![0.0f32; (w * h) as usize];
        for y in 12..36 {
            for x in 12..36 {
                data[y * w as usize + x] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, w, h).unwrap();
        let edges = canny(&img, 0.05, 0.15).unwrap();

        let edge_count: usize = edges.as_slice().iter().filter(|&&v| v > 0.5).count();
        // A 24x24 rectangle should produce roughly 4*24 = 96 edge pixels (give or take)
        assert!(
            edge_count > 40,
            "expected edge pixels around rectangle, got {edge_count}"
        );
    }

    #[test]
    fn output_is_binary() {
        let data: Vec<f32> = (0..64)
            .flat_map(|y| (0..64).map(move |x| ((x + y) as f32 / 128.0).sin()))
            .collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 64, 64).unwrap();
        let edges = canny(&img, 0.05, 0.15).unwrap();
        assert!(
            edges.as_slice().iter().all(|&v| v == 0.0 || v == 1.0),
            "output must be binary (0.0 or 1.0)"
        );
    }

    #[test]
    fn rejects_invalid_thresholds() {
        let img = ImageBuf::<f32, Gray>::new(16, 16).unwrap();
        assert!(canny(&img, 0.5, 0.1).is_err(), "high < low should error");
        assert!(
            canny(&img, -0.1, 0.3).is_err(),
            "negative low should error"
        );
        assert!(
            canny_with_sigma(&img, 0.1, 0.3, 0.0).is_err(),
            "zero sigma should error"
        );
    }
}
