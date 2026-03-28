// SPDX-License-Identifier: MIT OR Apache-2.0
//! Laplacian pyramid for multi-scale detail decomposition.

use crate::error::Result;
use crate::image::{Gray, ImageBuf};
use crate::pyramid::gaussian::{upsample_bilinear, GaussianPyramid};

/// Laplacian pyramid: stores band-pass detail at each scale.
///
/// Each level is the difference between consecutive Gaussian pyramid levels
/// (upsampled to match dimensions). The last level stores the low-frequency
/// residual.
pub struct LaplacianPyramid {
    /// Detail images at each scale, plus the coarsest Gaussian level.
    pub levels: Vec<ImageBuf<f32, Gray>>,
}

impl LaplacianPyramid {
    /// Build a Laplacian pyramid from a Gaussian pyramid.
    pub fn from_gaussian(gpyr: &GaussianPyramid) -> Result<Self> {
        let n = gpyr.n_levels();
        let mut levels = Vec::with_capacity(n);

        for i in 0..n - 1 {
            let fine = &gpyr.levels[i];
            let coarse = &gpyr.levels[i + 1];
            let (fw, fh) = fine.dimensions();

            // Upsample coarse to fine's size
            let upsampled = upsample_bilinear(coarse, fw, fh)?;

            // Laplacian = fine - upsampled(coarse)
            let detail: Vec<f32> = fine
                .as_slice()
                .iter()
                .zip(upsampled.as_slice())
                .map(|(&a, &b)| a - b)
                .collect();
            levels.push(ImageBuf::from_vec(detail, fw, fh)?);
        }

        // Last level is the coarsest Gaussian level
        levels.push(gpyr.levels[n - 1].clone());

        Ok(Self { levels })
    }

    /// Reconstruct the original image by collapsing the Laplacian pyramid.
    pub fn reconstruct(&self) -> Result<ImageBuf<f32, Gray>> {
        let n = self.levels.len();
        let mut current = self.levels[n - 1].clone();

        for i in (0..n - 1).rev() {
            let detail = &self.levels[i];
            let (dw, dh) = detail.dimensions();

            let upsampled = upsample_bilinear(&current, dw, dh)?;
            let reconstructed: Vec<f32> = detail
                .as_slice()
                .iter()
                .zip(upsampled.as_slice())
                .map(|(&d, &u)| d + u)
                .collect();
            current = ImageBuf::from_vec(reconstructed, dw, dh)?;
        }

        Ok(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruct_matches_original() {
        let data: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data.clone(), 16, 16).unwrap();

        let gpyr = GaussianPyramid::build(&img, 4, 2.0, 1.0).unwrap();
        let lpyr = LaplacianPyramid::from_gaussian(&gpyr).unwrap();
        let reconstructed = lpyr.reconstruct().unwrap();

        let max_err: f32 = data
            .iter()
            .zip(reconstructed.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_err < 0.05, "reconstruction error too large: {max_err}");
    }
}
