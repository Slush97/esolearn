// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gaussian image pyramid for multi-scale feature detection.

use crate::error::Result;
use crate::filter::gaussian::gaussian_blur;
use crate::image::{Gray, ImageBuf};

/// Gaussian pyramid: a sequence of progressively downsampled and blurred images.
pub struct GaussianPyramid {
    /// Pyramid levels, from finest (index 0 = original) to coarsest.
    pub levels: Vec<ImageBuf<f32, Gray>>,
    /// Scale factor at each level relative to the original.
    pub scale_factors: Vec<f32>,
}

impl GaussianPyramid {
    /// Build a Gaussian pyramid with the given number of levels and scale factor.
    ///
    /// - `n_levels`: number of pyramid levels (including the original)
    /// - `scale_factor`: downsampling ratio between adjacent levels (typical: 2.0 or 1.2)
    /// - `sigma`: Gaussian blur sigma applied before each downsample
    pub fn build(
        img: &ImageBuf<f32, Gray>,
        n_levels: usize,
        scale_factor: f32,
        sigma: f32,
    ) -> Result<Self> {
        let mut levels = Vec::with_capacity(n_levels);
        let mut scale_factors = Vec::with_capacity(n_levels);

        levels.push(img.clone());
        scale_factors.push(1.0);

        let mut current = img.clone();
        let mut cumulative_scale = 1.0f32;

        for _ in 1..n_levels {
            // Blur before downsampling (anti-aliasing)
            let blurred = gaussian_blur(&current, sigma)?;

            // Compute new dimensions
            cumulative_scale *= scale_factor;
            let new_w = ((img.width() as f32 / cumulative_scale).round() as u32).max(1);
            let new_h = ((img.height() as f32 / cumulative_scale).round() as u32).max(1);

            if new_w < 2 || new_h < 2 {
                break;
            }

            let downsampled = downsample_bilinear(&blurred, new_w, new_h)?;
            current = downsampled.clone();
            levels.push(downsampled);
            scale_factors.push(cumulative_scale);
        }

        Ok(Self {
            levels,
            scale_factors,
        })
    }

    /// Number of levels in the pyramid.
    #[inline]
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }
}

/// Downsample a grayscale f32 image using bilinear interpolation.
pub fn downsample_bilinear(
    img: &ImageBuf<f32, Gray>,
    new_w: u32,
    new_h: u32,
) -> Result<ImageBuf<f32, Gray>> {
    let mut out = vec![0.0f32; new_w as usize * new_h as usize];
    let sx = img.width() as f32 / new_w as f32;
    let sy = img.height() as f32 / new_h as f32;

    for y in 0..new_h {
        for x in 0..new_w {
            let src_x = (x as f32 + 0.5) * sx - 0.5;
            let src_y = (y as f32 + 0.5) * sy - 0.5;
            out[y as usize * new_w as usize + x as usize] =
                crate::math::bilinear_at(img.as_slice(), img.width(), img.height(), src_x, src_y);
        }
    }

    ImageBuf::from_vec(out, new_w, new_h)
}

/// Upsample a grayscale f32 image using bilinear interpolation.
pub fn upsample_bilinear(
    img: &ImageBuf<f32, Gray>,
    new_w: u32,
    new_h: u32,
) -> Result<ImageBuf<f32, Gray>> {
    downsample_bilinear(img, new_w, new_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pyramid_levels_decrease_in_size() {
        let data = vec![0.5f32; 64 * 64];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 64, 64).unwrap();
        let pyr = GaussianPyramid::build(&img, 5, 2.0, 1.0).unwrap();

        assert!(pyr.n_levels() >= 4);
        for i in 1..pyr.n_levels() {
            let (pw, ph) = pyr.levels[i].dimensions();
            let (pw0, ph0) = pyr.levels[i - 1].dimensions();
            assert!(pw <= pw0 && ph <= ph0, "level {i} should be smaller");
        }
    }

    #[test]
    fn downsample_preserves_mean() {
        let data: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data.clone(), 16, 16).unwrap();
        let small = downsample_bilinear(&img, 8, 8).unwrap();

        let mean_orig: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let mean_small: f32 = small.as_slice().iter().sum::<f32>() / small.len() as f32;
        assert!(
            (mean_orig - mean_small).abs() < 0.05,
            "means: orig={mean_orig}, small={mean_small}"
        );
    }
}
