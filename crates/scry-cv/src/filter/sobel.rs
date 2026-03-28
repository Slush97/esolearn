// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sobel and Scharr gradient operators.

use crate::error::Result;
use crate::filter::convolve_separable;
use crate::image::{Gray, ImageBuf};

/// Compute horizontal Sobel gradient (dI/dx).
pub fn sobel_x(img: &ImageBuf<f32, Gray>) -> Result<ImageBuf<f32, Gray>> {
    // Sobel X = [-1, 0, 1] outer [1, 2, 1]
    let h = vec![-1.0, 0.0, 1.0];
    let v = vec![1.0, 2.0, 1.0];
    convolve_separable(img, &h, &v)
}

/// Compute vertical Sobel gradient (dI/dy).
pub fn sobel_y(img: &ImageBuf<f32, Gray>) -> Result<ImageBuf<f32, Gray>> {
    // Sobel Y = [1, 2, 1] outer [-1, 0, 1]
    let h = vec![1.0, 2.0, 1.0];
    let v = vec![-1.0, 0.0, 1.0];
    convolve_separable(img, &h, &v)
}

/// Compute horizontal Scharr gradient (more rotation-invariant than Sobel).
pub fn scharr_x(img: &ImageBuf<f32, Gray>) -> Result<ImageBuf<f32, Gray>> {
    let h = vec![-1.0, 0.0, 1.0];
    let v = vec![3.0, 10.0, 3.0];
    convolve_separable(img, &h, &v)
}

/// Compute vertical Scharr gradient.
pub fn scharr_y(img: &ImageBuf<f32, Gray>) -> Result<ImageBuf<f32, Gray>> {
    let h = vec![3.0, 10.0, 3.0];
    let v = vec![-1.0, 0.0, 1.0];
    convolve_separable(img, &h, &v)
}

/// Compute gradient magnitude from an image: `sqrt(dx^2 + dy^2)`.
pub fn gradient_magnitude(img: &ImageBuf<f32, Gray>) -> Result<ImageBuf<f32, Gray>> {
    let dx = sobel_x(img)?;
    let dy = sobel_y(img)?;
    let data: Vec<f32> = dx
        .as_slice()
        .iter()
        .zip(dy.as_slice())
        .map(|(&gx, &gy)| gx.hypot(gy))
        .collect();
    ImageBuf::from_vec(data, img.width(), img.height())
}

/// Compute gradient direction (angle in radians) from an image.
pub fn gradient_direction(img: &ImageBuf<f32, Gray>) -> Result<ImageBuf<f32, Gray>> {
    let dx = sobel_x(img)?;
    let dy = sobel_y(img)?;
    let data: Vec<f32> = dx
        .as_slice()
        .iter()
        .zip(dy.as_slice())
        .map(|(&gx, &gy)| gy.atan2(gx))
        .collect();
    ImageBuf::from_vec(data, img.width(), img.height())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sobel_detects_horizontal_edge() {
        // Top half = 0, bottom half = 1
        let mut data = vec![0.0f32; 8 * 8];
        for y in 4..8u32 {
            for x in 0..8u32 {
                data[(y * 8 + x) as usize] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 8, 8).unwrap();
        let gy = sobel_y(&img).unwrap();

        // Row 3 (just above edge) and row 4 (just below) should have strong response
        let edge_response: f32 = gy
            .as_slice()
            .iter()
            .map(|v| v.abs())
            .sum::<f32>();
        assert!(edge_response > 0.0, "should detect horizontal edge");
    }

    #[test]
    fn sobel_detects_vertical_edge() {
        // Left half = 0, right half = 1
        let mut data = vec![0.0f32; 8 * 8];
        for y in 0..8u32 {
            for x in 4..8u32 {
                data[(y * 8 + x) as usize] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 8, 8).unwrap();
        let gx = sobel_x(&img).unwrap();

        let edge_response: f32 = gx.as_slice().iter().map(|v| v.abs()).sum();
        assert!(edge_response > 0.0, "should detect vertical edge");
    }

    #[test]
    fn gradient_magnitude_nonnegative() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 8, 8).unwrap();
        let mag = gradient_magnitude(&img).unwrap();
        assert!(mag.as_slice().iter().all(|&v| v >= 0.0));
    }
}
