// SPDX-License-Identifier: MIT OR Apache-2.0
//! Census transform for robust stereo matching.
//!
//! Encodes per-pixel neighborhood comparisons as bit strings. The Hamming
//! distance between two census values measures matching cost, which is
//! robust to illumination changes.

use crate::image::{Gray, ImageBuf};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Compute the census transform of a grayscale image.
///
/// For each pixel, compares it against all neighbors in a `window × window`
/// region and encodes the result as a bit string (u64). The window must be
/// odd and at most 9 (giving 80 neighbors, which fits in u64 with room).
///
/// Returns a flat `Vec<u64>` in row-major order.
pub fn census_transform(img: &ImageBuf<f32, Gray>, window: u32) -> Vec<u64> {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let half = window as i32 / 2;
    let data = img.as_slice();
    let rows: Vec<i32> = (0..h).collect();
    #[cfg(feature = "rayon")]
    let iter = rows.par_iter();
    #[cfg(not(feature = "rayon"))]
    let iter = rows.iter();

    iter.flat_map(|&y| {
        (0..w)
            .map(move |x| {
                let center = data[(y * w + x) as usize];
                let mut bits = 0u64;
                let mut bit_pos = 0u32;

                for dy in -half..=half {
                    for dx in -half..=half {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = (x + dx).clamp(0, w - 1);
                        let ny = (y + dy).clamp(0, h - 1);
                        let neighbor = data[(ny * w + nx) as usize];
                        if neighbor < center {
                            bits |= 1u64 << bit_pos;
                        }
                        bit_pos += 1;
                        if bit_pos >= 64 {
                            break;
                        }
                    }
                    if bit_pos >= 64 {
                        break;
                    }
                }

                bits
            })
            .collect::<Vec<u64>>()
    })
    .collect()
}

/// Hamming distance between two census values.
#[inline]
pub fn census_hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_image_has_zero_census() {
        let data = vec![0.5f32; 16 * 16];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 16, 16).unwrap();
        let census = census_transform(&img, 5);
        // All neighbors equal center → no bits set (we use strict <)
        for &c in &census {
            assert_eq!(c, 0, "uniform image should have zero census values");
        }
    }

    #[test]
    fn hamming_distance_basic() {
        assert_eq!(census_hamming(0b1010, 0b0110), 2);
        assert_eq!(census_hamming(0, 0), 0);
        assert_eq!(census_hamming(u64::MAX, 0), 64);
    }

    #[test]
    fn gradient_image_has_nonzero_census() {
        let data: Vec<f32> = (0..16 * 16).map(|i| i as f32 / 255.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 16, 16).unwrap();
        let census = census_transform(&img, 5);
        let nonzero = census.iter().filter(|&&c| c != 0).count();
        assert!(nonzero > 0, "gradient image should have nonzero census");
    }
}
