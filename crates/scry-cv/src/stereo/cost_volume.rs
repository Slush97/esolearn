// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cost volume construction for stereo matching.
//!
//! Builds a 3D cost volume `[height × width × num_disparities]` where each
//! entry is the matching cost between a left-image pixel and the
//! corresponding right-image pixel at a given disparity.

use crate::stereo::census::census_hamming;

/// Maximum cost value (used when the right pixel is out of bounds).
pub const MAX_COST: u16 = 64;

/// Build a cost volume from census-transformed left and right images.
///
/// The cost at `(y, x, d)` is the Hamming distance between
/// `left_census[y * w + x]` and `right_census[y * w + (x - d)]`,
/// where `d = min_disp + d_index`.
///
/// Returns a flat `Vec<u16>` of size `height * width * num_disp`.
pub fn build_cost_volume(
    left_census: &[u64],
    right_census: &[u64],
    width: u32,
    height: u32,
    min_disp: i32,
    num_disp: u32,
) -> Vec<u16> {
    let w = width as usize;
    let h = height as usize;
    let nd = num_disp as usize;
    let mut volume = vec![MAX_COST; h * w * nd];

    for y in 0..h {
        for x in 0..w {
            let left_val = left_census[y * w + x];
            let base = (y * w + x) * nd;

            for di in 0..nd {
                let d = min_disp + di as i32;
                let rx = x as i32 - d;
                if rx >= 0 && rx < w as i32 {
                    let right_val = right_census[y * w + rx as usize];
                    volume[base + di] = census_hamming(left_val, right_val) as u16;
                }
            }
        }
    }

    volume
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Gray, ImageBuf};
    use crate::stereo::census::census_transform;

    #[test]
    fn identical_images_zero_cost_at_d0() {
        let data = vec![0.5f32; 16 * 16];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 16, 16).unwrap();
        let census = census_transform(&img, 5);
        let vol = build_cost_volume(&census, &census, 16, 16, 0, 16);

        // At disparity 0, cost should be 0 everywhere
        for y in 0..16usize {
            for x in 0..16usize {
                let cost = vol[(y * 16 + x) * 16 + 0]; // d=0
                assert_eq!(cost, 0, "identical images at d=0 should have zero cost");
            }
        }
    }

    #[test]
    fn volume_dimensions() {
        let census = vec![0u64; 8 * 6];
        let vol = build_cost_volume(&census, &census, 8, 6, 0, 16);
        assert_eq!(vol.len(), 8 * 6 * 16);
    }
}
