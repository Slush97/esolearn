// SPDX-License-Identifier: MIT OR Apache-2.0
//! ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor.
//!
//! Reference: Rublee et al., "ORB: An Efficient Alternative to SIFT or SURF" (ICCV 2011).

use crate::error::Result;
use crate::features::brief::compute_rbrief;
use crate::features::fast::{detect_fast9, nms};
use crate::features::keypoint::{BinaryDescriptor, KeyPoint};
use crate::image::{Gray, ImageBuf};
use crate::pyramid::gaussian::GaussianPyramid;

/// ORB feature detector and descriptor extractor.
///
/// # Example
///
/// ```
/// use scry_cv::features::orb::Orb;
/// use scry_cv::prelude::*;
///
/// let img = GrayImageF::new(64, 64).unwrap();
/// let orb = Orb::new();
/// let (keypoints, descriptors) = orb.detect_and_compute(&img).unwrap();
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Orb {
    /// Maximum number of features to detect.
    pub n_features: usize,
    /// Scale factor between pyramid levels.
    pub scale_factor: f32,
    /// Number of pyramid levels.
    pub n_levels: usize,
    /// FAST threshold.
    pub fast_threshold: f32,
    /// Patch size for BRIEF descriptor.
    pub patch_size: u32,
}

impl Orb {
    /// Create an ORB detector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_features: 500,
            scale_factor: 1.2,
            n_levels: 8,
            fast_threshold: 0.05,
            patch_size: 31,
        }
    }

    /// Set the maximum number of features.
    #[must_use]
    pub fn n_features(mut self, n: usize) -> Self {
        self.n_features = n;
        self
    }

    /// Set the pyramid scale factor.
    #[must_use]
    pub fn scale_factor(mut self, s: f32) -> Self {
        self.scale_factor = s;
        self
    }

    /// Set the number of pyramid levels.
    #[must_use]
    pub fn n_levels(mut self, n: usize) -> Self {
        self.n_levels = n;
        self
    }

    /// Set the FAST corner threshold.
    #[must_use]
    pub fn fast_threshold(mut self, t: f32) -> Self {
        self.fast_threshold = t;
        self
    }

    /// Detect keypoints and compute descriptors.
    pub fn detect_and_compute(
        &self,
        img: &ImageBuf<f32, Gray>,
    ) -> Result<(Vec<KeyPoint>, Vec<BinaryDescriptor>)> {
        // Build Gaussian pyramid
        let sigma = 1.2; // pre-smoothing sigma
        let pyr = GaussianPyramid::build(img, self.n_levels, self.scale_factor, sigma)?;

        // Distribute features across levels (more at finer scales)
        let features_per_level =
            distribute_features(self.n_features, pyr.n_levels(), self.scale_factor);

        let mut all_kps = Vec::new();
        let mut all_desc = Vec::new();

        for (level, level_img) in pyr.levels.iter().enumerate() {
            let scale = pyr.scale_factors[level];
            let max_kps = features_per_level[level];

            // Detect FAST corners
            let mut corners = detect_fast9(level_img, self.fast_threshold);

            // Non-maximum suppression
            corners = nms(&corners, 5.0);

            // Sort by response and keep top N
            corners.sort_unstable_by(|a, b| {
                b.response
                    .partial_cmp(&a.response)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            corners.truncate(max_kps);

            // Compute orientation via intensity centroid
            compute_orientation(level_img, &mut corners);

            // Compute rotated BRIEF descriptors
            let (kps, descs) = compute_rbrief(level_img, &corners, self.patch_size);

            // Scale keypoint coordinates back to original image space
            for mut kp in kps {
                kp.x *= scale;
                kp.y *= scale;
                kp.scale = scale;
                kp.octave = level as i32;
                all_kps.push(kp);
            }
            all_desc.extend(descs);
        }

        // Final truncation to n_features
        if all_kps.len() > self.n_features {
            // Sort by response, keep strongest
            let mut indices: Vec<usize> = (0..all_kps.len()).collect();
            indices.sort_unstable_by(|&a, &b| {
                all_kps[b]
                    .response
                    .partial_cmp(&all_kps[a].response)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            indices.truncate(self.n_features);
            indices.sort_unstable(); // restore order for stable extraction

            all_kps = indices.iter().map(|&i| all_kps[i].clone()).collect();
            all_desc = indices.iter().map(|&i| all_desc[i].clone()).collect();
        }

        Ok((all_kps, all_desc))
    }
}

impl Default for Orb {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute keypoint orientation via intensity centroid method.
fn compute_orientation(img: &ImageBuf<f32, Gray>, keypoints: &mut [KeyPoint]) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let data = img.as_slice();
    let radius = 15i32;

    for kp in keypoints.iter_mut() {
        let cx = kp.x.round() as i32;
        let cy = kp.y.round() as i32;

        let mut m01 = 0.0f32;
        let mut m10 = 0.0f32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                let px = (cx + dx).clamp(0, w - 1) as usize;
                let py = (cy + dy).clamp(0, h - 1) as usize;
                let val = data[py * w as usize + px];
                m10 += dx as f32 * val;
                m01 += dy as f32 * val;
            }
        }

        kp.angle = m01.atan2(m10);
    }
}

/// Distribute `n_features` across `n_levels` with more features at finer scales.
fn distribute_features(n_features: usize, n_levels: usize, scale_factor: f32) -> Vec<usize> {
    if n_levels == 0 {
        return vec![];
    }
    if n_levels == 1 {
        return vec![n_features];
    }

    let factor = 1.0 / scale_factor;
    let mut n_per_level = Vec::with_capacity(n_levels);
    let mut n_desired =
        n_features as f64 * (1.0 - factor as f64) / (1.0 - (factor as f64).powi(n_levels as i32));

    for _ in 0..n_levels - 1 {
        n_per_level.push(n_desired.round() as usize);
        n_desired *= factor as f64;
    }
    // Last level gets the remainder
    let assigned: usize = n_per_level.iter().sum();
    n_per_level.push(n_features.saturating_sub(assigned));

    n_per_level
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orb_on_checkerboard() {
        // Create a checkerboard pattern (should have corners)
        let mut data = vec![0.0f32; 64 * 64];
        for y in 0..64u32 {
            for x in 0..64u32 {
                data[(y * 64 + x) as usize] = if (x / 8 + y / 8) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 64, 64).unwrap();
        let orb = Orb::new().n_features(50).n_levels(3);
        let (kps, descs) = orb.detect_and_compute(&img).unwrap();

        assert!(
            !kps.is_empty(),
            "checkerboard should have detectable features"
        );
        assert_eq!(kps.len(), descs.len());
        assert_eq!(descs[0].n_bits(), 256);
    }

    #[test]
    fn orb_on_uniform_returns_few() {
        let data = vec![0.5f32; 64 * 64];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 64, 64).unwrap();
        let orb = Orb::new().n_features(100);
        let (kps, _) = orb.detect_and_compute(&img).unwrap();
        assert!(kps.len() < 10, "uniform image should have few features");
    }

    #[test]
    fn distribute_features_sums_correctly() {
        let dist = distribute_features(500, 8, 1.2);
        let total: usize = dist.iter().sum();
        assert_eq!(total, 500);
        assert_eq!(dist.len(), 8);
    }

    #[test]
    fn descriptor_is_256_bits() {
        let mut data = vec![0.0f32; 64 * 64];
        for y in 0..64u32 {
            for x in 0..64u32 {
                data[(y * 64 + x) as usize] = if (x / 8 + y / 8) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 64, 64).unwrap();
        let orb = Orb::new().n_features(10).n_levels(2);
        let (_, descs) = orb.detect_and_compute(&img).unwrap();
        for d in &descs {
            assert_eq!(d.data.len(), 32, "256 bits = 32 bytes");
        }
    }
}
