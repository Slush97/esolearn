// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gaussian blur via separable convolution.

use crate::error::Result;
use crate::filter::convolve_separable;
use crate::image::{Gray, ImageBuf};
use crate::math::gaussian_kernel_1d;

/// Apply Gaussian blur to a grayscale f32 image.
///
/// `sigma` controls the blur radius; the kernel size is `ceil(3*sigma)*2 + 1`.
pub fn gaussian_blur(img: &ImageBuf<f32, Gray>, sigma: f32) -> Result<ImageBuf<f32, Gray>> {
    let sigma = sigma.max(0.5);
    let kernel = gaussian_kernel_1d(sigma);
    convolve_separable(img, &kernel, &kernel)
}

/// Apply Gaussian blur with separate horizontal and vertical sigma values.
pub fn gaussian_blur_xy(
    img: &ImageBuf<f32, Gray>,
    sigma_x: f32,
    sigma_y: f32,
) -> Result<ImageBuf<f32, Gray>> {
    let kx = gaussian_kernel_1d(sigma_x.max(0.5));
    let ky = gaussian_kernel_1d(sigma_y.max(0.5));
    convolve_separable(img, &kx, &ky)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_reduces_variance() {
        // Checkerboard pattern
        let mut data = vec![0.0f32; 16 * 16];
        for y in 0..16u32 {
            for x in 0..16u32 {
                data[(y * 16 + x) as usize] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data.clone(), 16, 16).unwrap();
        let blurred = gaussian_blur(&img, 2.0).unwrap();

        // Variance of blurred should be less than original
        let mean_orig: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var_orig: f32 =
            data.iter().map(|v| (v - mean_orig).powi(2)).sum::<f32>() / data.len() as f32;

        let bdata = blurred.as_slice();
        let mean_blur: f32 = bdata.iter().sum::<f32>() / bdata.len() as f32;
        let var_blur: f32 =
            bdata.iter().map(|v| (v - mean_blur).powi(2)).sum::<f32>() / bdata.len() as f32;

        assert!(
            var_blur < var_orig,
            "blur should reduce variance: orig={var_orig}, blur={var_blur}"
        );
    }

    #[test]
    fn blur_preserves_mean() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data.clone(), 8, 8).unwrap();
        let blurred = gaussian_blur(&img, 1.5).unwrap();

        // Interior pixels should have similar mean (border effects at edges)
        let sum_orig: f32 = data.iter().sum();
        let sum_blur: f32 = blurred.as_slice().iter().sum();
        let mean_orig = sum_orig / data.len() as f32;
        let mean_blur = sum_blur / blurred.len() as f32;

        assert!(
            (mean_orig - mean_blur).abs() < 0.05,
            "means differ: orig={mean_orig}, blur={mean_blur}"
        );
    }
}
