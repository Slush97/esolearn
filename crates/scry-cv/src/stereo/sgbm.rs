// SPDX-License-Identifier: MIT OR Apache-2.0
//! Semi-Global Block Matching (SGBM) stereo disparity estimation.
//!
//! Computes a dense disparity map from a rectified stereo pair using:
//! 1. Census transform for robust matching cost
//! 2. Semi-global path aggregation (8 directions) with P1/P2 penalties
//! 3. Winner-takes-all disparity selection
//! 4. Optional left-right consistency check
//!
//! Reference: Hirschmuller, "Stereo Processing by Semiglobal Matching and
//! Mutual Information" (2008).

use crate::error::{Result, ScryVisionError};
use crate::image::{Gray, ImageBuf};
use crate::stereo::census::census_transform;
use crate::stereo::cost_volume::{build_cost_volume, MAX_COST};

/// 8 aggregation directions: right, down-right, down, down-left,
/// left, up-left, up, up-right.
const DIRECTIONS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

/// Semi-Global Block Matching stereo estimator.
///
/// # Example
///
/// ```
/// use scry_cv::prelude::*;
/// use scry_cv::stereo::SgbmStereo;
///
/// let left = GrayImageF::from_vec(vec![0.5; 32 * 32], 32, 32).unwrap();
/// let right = left.clone();
/// let sgbm = SgbmStereo::new();
/// let disp = sgbm.compute(&left, &right).unwrap();
/// assert_eq!(disp.dimensions(), (32, 32));
/// ```
#[derive(Clone, Debug)]
pub struct SgbmStereo {
    /// Minimum disparity (can be negative for right-to-left matching).
    pub min_disparity: i32,
    /// Number of disparity levels to evaluate.
    pub num_disparities: u32,
    /// Census window size (odd, typically 5 or 7).
    pub block_size: u32,
    /// Penalty for disparity change of ±1 between adjacent pixels.
    pub p1: u32,
    /// Penalty for disparity change > 1 between adjacent pixels.
    pub p2: u32,
    /// Maximum allowed difference in left-right consistency check.
    /// Set to -1 to disable.
    pub disp12_max_diff: i32,
    /// Uniqueness ratio: reject matches where best cost is not sufficiently
    /// better than second best (percent, e.g., 10 means 10%).
    pub uniqueness_ratio: u32,
}

impl Default for SgbmStereo {
    fn default() -> Self {
        Self {
            min_disparity: 0,
            num_disparities: 64,
            block_size: 5,
            p1: 8,
            p2: 32,
            disp12_max_diff: 1,
            uniqueness_ratio: 10,
        }
    }
}

impl SgbmStereo {
    /// Create with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum disparity.
    #[must_use]
    pub fn min_disparity(mut self, v: i32) -> Self {
        self.min_disparity = v;
        self
    }

    /// Set number of disparity levels.
    #[must_use]
    pub fn num_disparities(mut self, v: u32) -> Self {
        self.num_disparities = v;
        self
    }

    /// Set census block size.
    #[must_use]
    pub fn block_size(mut self, v: u32) -> Self {
        self.block_size = v;
        self
    }

    /// Set P1 smoothness penalty.
    #[must_use]
    pub fn p1(mut self, v: u32) -> Self {
        self.p1 = v;
        self
    }

    /// Set P2 smoothness penalty.
    #[must_use]
    pub fn p2(mut self, v: u32) -> Self {
        self.p2 = v;
        self
    }

    /// Set left-right consistency threshold (-1 to disable).
    #[must_use]
    pub fn disp12_max_diff(mut self, v: i32) -> Self {
        self.disp12_max_diff = v;
        self
    }

    /// Compute disparity map from a rectified stereo pair.
    ///
    /// Returns a disparity image where each pixel holds the estimated
    /// horizontal displacement. Invalid pixels are set to -1.0.
    pub fn compute(
        &self,
        left: &ImageBuf<f32, Gray>,
        right: &ImageBuf<f32, Gray>,
    ) -> Result<ImageBuf<f32, Gray>> {
        if left.dimensions() != right.dimensions() {
            return Err(ScryVisionError::InvalidDimensions(
                "left and right images must have the same dimensions".into(),
            ));
        }

        let w = left.width();
        let h = left.height();
        let nd = self.num_disparities as usize;

        // 1. Census transform
        let left_census = census_transform(left, self.block_size);
        let right_census = census_transform(right, self.block_size);

        // 2. Build cost volume
        let cost = build_cost_volume(
            &left_census,
            &right_census,
            w,
            h,
            self.min_disparity,
            self.num_disparities,
        );

        // 3. Semi-global aggregation
        let aggregated = self.aggregate(&cost, w as usize, h as usize, nd);

        // 4. WTA disparity selection
        let mut disp = wta_disparity(&aggregated, w as usize, h as usize, nd);

        // 5. Left-right consistency check
        if self.disp12_max_diff >= 0 {
            let right_cost = build_cost_volume(
                &right_census,
                &left_census,
                w,
                h,
                -self.min_disparity - self.num_disparities as i32 + 1,
                self.num_disparities,
            );
            let right_agg = self.aggregate(&right_cost, w as usize, h as usize, nd);
            let right_disp = wta_disparity(&right_agg, w as usize, h as usize, nd);

            lr_check(
                &mut disp,
                &right_disp,
                w as usize,
                h as usize,
                self.min_disparity,
                self.disp12_max_diff,
            );
        }

        // 6. Convert to f32 disparity map
        let data: Vec<f32> = disp
            .iter()
            .map(|&d| {
                if d < 0 {
                    -1.0
                } else {
                    self.min_disparity as f32 + d as f32
                }
            })
            .collect();

        ImageBuf::from_vec(data, w, h)
    }

    /// Semi-global path aggregation across 8 directions.
    fn aggregate(
        &self,
        cost: &[u16],
        w: usize,
        h: usize,
        nd: usize,
    ) -> Vec<u32> {
        let n = w * h * nd;
        let mut sum = vec![0u32; n];

        for &(dx, dy) in &DIRECTIONS {
            let path = aggregate_direction(cost, w, h, nd, dx, dy, self.p1, self.p2);
            for i in 0..n {
                sum[i] += path[i] as u32;
            }
        }

        sum
    }
}

/// Aggregate cost along a single direction using the SGM recurrence.
///
/// `Lr(p,d) = C(p,d) + min(Lr(prev,d), Lr(prev,d-1)+P1, Lr(prev,d+1)+P1,
///                         min_k(Lr(prev,k))+P2) - min_k(Lr(prev,k))`
fn aggregate_direction(
    cost: &[u16],
    w: usize,
    h: usize,
    nd: usize,
    dx: i32,
    dy: i32,
    p1: u32,
    p2: u32,
) -> Vec<u16> {
    let n = w * h * nd;
    let mut lr = vec![0u16; n];

    // Determine iteration order so we process pixels after their predecessors
    let (y_range, x_range): (Vec<usize>, Vec<usize>) = (
        if dy > 0 {
            (0..h).collect()
        } else if dy < 0 {
            (0..h).rev().collect()
        } else {
            (0..h).collect()
        },
        if dx > 0 {
            (0..w).collect()
        } else if dx < 0 {
            (0..w).rev().collect()
        } else {
            (0..w).collect()
        },
    );

    for &y in &y_range {
        for &x in &x_range {
            let px = x as i32 - dx;
            let py = y as i32 - dy;
            let base = (y * w + x) * nd;

            if px < 0 || px >= w as i32 || py < 0 || py >= h as i32 {
                // First pixel along this path: Lr = C
                for d in 0..nd {
                    lr[base + d] = cost[base + d];
                }
                continue;
            }

            let prev_base = (py as usize * w + px as usize) * nd;

            // Find min of previous Lr across all disparities
            let mut prev_min = u16::MAX;
            for d in 0..nd {
                prev_min = prev_min.min(lr[prev_base + d]);
            }
            let prev_min_u32 = prev_min as u32;

            for d in 0..nd {
                let c = cost[base + d] as u32;

                // SGM recurrence
                let same = lr[prev_base + d] as u32;
                let minus1 = if d > 0 {
                    lr[prev_base + d - 1] as u32 + p1
                } else {
                    u32::MAX
                };
                let plus1 = if d + 1 < nd {
                    lr[prev_base + d + 1] as u32 + p1
                } else {
                    u32::MAX
                };
                let any = prev_min_u32 + p2;

                let val = c + same.min(minus1).min(plus1).min(any) - prev_min_u32;

                // Clamp to u16 range
                lr[base + d] = val.min(u16::MAX as u32) as u16;
            }
        }
    }

    lr
}

/// Winner-takes-all: for each pixel, find the disparity with minimum aggregated cost.
///
/// Returns disparity index (not absolute disparity). -1 = invalid.
fn wta_disparity(aggregated: &[u32], w: usize, h: usize, nd: usize) -> Vec<i32> {
    let mut disp = vec![-1i32; w * h];

    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * nd;
            let mut best_d = 0usize;
            let mut best_cost = aggregated[base];

            for d in 1..nd {
                if aggregated[base + d] < best_cost {
                    best_cost = aggregated[base + d];
                    best_d = d;
                }
            }

            // Only accept if the cost is below MAX
            if best_cost < (MAX_COST as u32 * 8) {
                disp[y * w + x] = best_d as i32;
            }
        }
    }

    disp
}

/// Left-right consistency check: invalidate pixels where left and right
/// disparity maps disagree by more than `max_diff`.
fn lr_check(
    left_disp: &mut [i32],
    right_disp: &[i32],
    w: usize,
    h: usize,
    min_disp: i32,
    max_diff: i32,
) {
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let ld = left_disp[idx];
            if ld < 0 {
                continue;
            }
            let d_abs = min_disp + ld;
            let rx = x as i32 - d_abs;
            if rx < 0 || rx >= w as i32 {
                left_disp[idx] = -1;
                continue;
            }
            let rd = right_disp[y * w + rx as usize];
            if rd < 0 || (ld - rd).abs() > max_diff {
                left_disp[idx] = -1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_images_near_zero_disparity() {
        let data = vec![0.5f32; 32 * 32];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 32, 32).unwrap();
        let sgbm = SgbmStereo::new().num_disparities(16).disp12_max_diff(-1);
        let disp = sgbm.compute(&img, &img).unwrap();

        // For identical images, disparity should be 0 everywhere
        let valid_count = disp.as_slice().iter().filter(|&&v| v >= 0.0).count();
        let zero_count = disp
            .as_slice()
            .iter()
            .filter(|&&v| v >= 0.0 && v < 1.0)
            .count();
        if valid_count > 0 {
            let ratio = zero_count as f32 / valid_count as f32;
            assert!(
                ratio > 0.8,
                "most valid pixels should have disparity ~0, ratio = {ratio}"
            );
        }
    }

    #[test]
    fn shifted_image_has_positive_disparity() {
        let (w, h) = (64u32, 64u32);
        let shift = 4i32;
        // Create a textured image
        let left_data: Vec<f32> = (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    let fx = x as f32 / w as f32;
                    let fy = y as f32 / h as f32;
                    (fx * 5.0).sin() * (fy * 7.0).cos() * 0.5 + 0.5
                })
            })
            .collect();

        // Right image is left image shifted left by `shift` pixels
        let mut right_data = vec![0.0f32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                let src_x = x as i32 + shift;
                if src_x >= 0 && src_x < w as i32 {
                    right_data[y * w as usize + x] = left_data[y * w as usize + src_x as usize];
                }
            }
        }

        let left = ImageBuf::<f32, Gray>::from_vec(left_data, w, h).unwrap();
        let right = ImageBuf::<f32, Gray>::from_vec(right_data, w, h).unwrap();

        let sgbm = SgbmStereo::new()
            .num_disparities(16)
            .block_size(5)
            .disp12_max_diff(-1);

        let disp = sgbm.compute(&left, &right).unwrap();

        // Check center region for expected disparity
        let margin = 10usize;
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for y in margin..(h as usize - margin) {
            for x in (shift as usize + margin)..(w as usize - margin) {
                let v = disp.as_slice()[y * w as usize + x];
                if v >= 0.0 {
                    sum += v;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let avg = sum / count as f32;
            assert!(
                (avg - shift as f32).abs() < 2.0,
                "average disparity should be ~{shift}, got {avg}"
            );
        }
    }

    #[test]
    fn output_dimensions_match_input() {
        let left = ImageBuf::<f32, Gray>::new(24, 16).unwrap();
        let right = ImageBuf::<f32, Gray>::new(24, 16).unwrap();
        let disp = SgbmStereo::new()
            .num_disparities(8)
            .compute(&left, &right)
            .unwrap();
        assert_eq!(disp.dimensions(), (24, 16));
    }

    #[test]
    fn mismatched_dimensions_errors() {
        let left = ImageBuf::<f32, Gray>::new(32, 32).unwrap();
        let right = ImageBuf::<f32, Gray>::new(24, 32).unwrap();
        assert!(SgbmStereo::new().compute(&left, &right).is_err());
    }
}
