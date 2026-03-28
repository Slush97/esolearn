// SPDX-License-Identifier: MIT OR Apache-2.0
//! BRIEF binary descriptor computation.

use crate::features::keypoint::{BinaryDescriptor, KeyPoint};
use crate::image::{Gray, ImageBuf};
use crate::rng::FastRng;

/// Number of test pairs in a standard BRIEF descriptor.
const N_PAIRS: usize = 256;

/// Compute BRIEF descriptors for the given keypoints.
///
/// Uses a fixed set of 256 random test pairs within a patch of the given size.
/// Keypoints too close to the border are skipped.
///
/// Returns `(surviving_keypoints, descriptors)`.
pub fn compute_brief(
    img: &ImageBuf<f32, Gray>,
    keypoints: &[KeyPoint],
    patch_size: u32,
) -> (Vec<KeyPoint>, Vec<BinaryDescriptor>) {
    let pairs = generate_pairs(patch_size, 42);
    let half = patch_size as i32 / 2;
    let w = img.width() as i32;
    let h = img.height() as i32;
    let data = img.as_slice();

    let mut out_kps = Vec::with_capacity(keypoints.len());
    let mut out_desc = Vec::with_capacity(keypoints.len());

    for kp in keypoints {
        let cx = kp.x.round() as i32;
        let cy = kp.y.round() as i32;

        // Skip border keypoints
        if cx - half < 0 || cx + half >= w || cy - half < 0 || cy + half >= h {
            continue;
        }

        let mut desc = vec![0u8; N_PAIRS / 8];
        for (i, &(x1, y1, x2, y2)) in pairs.iter().enumerate() {
            let p1 = data[(cy + y1 as i32) as usize * w as usize + (cx + x1 as i32) as usize];
            let p2 = data[(cy + y2 as i32) as usize * w as usize + (cx + x2 as i32) as usize];
            if p1 < p2 {
                desc[i / 8] |= 1 << (i % 8);
            }
        }

        out_kps.push(kp.clone());
        out_desc.push(BinaryDescriptor { data: desc });
    }

    (out_kps, out_desc)
}

/// Compute rotation-compensated BRIEF (rBRIEF) for ORB.
///
/// Each test pair is rotated by the keypoint's orientation angle.
pub fn compute_rbrief(
    img: &ImageBuf<f32, Gray>,
    keypoints: &[KeyPoint],
    patch_size: u32,
) -> (Vec<KeyPoint>, Vec<BinaryDescriptor>) {
    let base_pairs = generate_pairs(patch_size, 42);
    let half = patch_size as i32 / 2;
    let w = img.width() as i32;
    let h = img.height() as i32;
    let data = img.as_slice();

    let mut out_kps = Vec::with_capacity(keypoints.len());
    let mut out_desc = Vec::with_capacity(keypoints.len());

    for kp in keypoints {
        let cx = kp.x.round() as i32;
        let cy = kp.y.round() as i32;

        if cx - half < 0 || cx + half >= w || cy - half < 0 || cy + half >= h {
            continue;
        }

        let cos_a = kp.angle.cos();
        let sin_a = kp.angle.sin();

        let mut desc = vec![0u8; N_PAIRS / 8];
        for (i, &(x1, y1, x2, y2)) in base_pairs.iter().enumerate() {
            // Rotate test pair by keypoint angle
            let rx1 = (x1 as f32 * cos_a - y1 as f32 * sin_a).round() as i32;
            let ry1 = (x1 as f32 * sin_a + y1 as f32 * cos_a).round() as i32;
            let rx2 = (x2 as f32 * cos_a - y2 as f32 * sin_a).round() as i32;
            let ry2 = (x2 as f32 * sin_a + y2 as f32 * cos_a).round() as i32;

            let px1 = (cx + rx1).clamp(0, w - 1);
            let py1 = (cy + ry1).clamp(0, h - 1);
            let px2 = (cx + rx2).clamp(0, w - 1);
            let py2 = (cy + ry2).clamp(0, h - 1);

            let p1 = data[py1 as usize * w as usize + px1 as usize];
            let p2 = data[py2 as usize * w as usize + px2 as usize];
            if p1 < p2 {
                desc[i / 8] |= 1 << (i % 8);
            }
        }

        out_kps.push(kp.clone());
        out_desc.push(BinaryDescriptor { data: desc });
    }

    (out_kps, out_desc)
}

/// Generate 256 deterministic random test pairs within a patch.
fn generate_pairs(patch_size: u32, seed: u64) -> Vec<(i8, i8, i8, i8)> {
    let mut rng = FastRng::new(seed);
    let half = (patch_size / 2) as i8;
    let range = 2 * half as usize + 1;

    (0..N_PAIRS)
        .map(|_| {
            let x1 = rng.usize(0..range) as i8 - half;
            let y1 = rng.usize(0..range) as i8 - half;
            let x2 = rng.usize(0..range) as i8 - half;
            let y2 = rng.usize(0..range) as i8 - half;
            (x1, y1, x2, y2)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brief_descriptor_length() {
        let data: Vec<f32> = (0..64 * 64).map(|i| (i % 256) as f32 / 255.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 64, 64).unwrap();
        let kps = vec![KeyPoint::new(32.0, 32.0)];
        let (_, descs) = compute_brief(&img, &kps, 31);
        assert_eq!(descs.len(), 1);
        assert_eq!(descs[0].n_bits(), 256);
    }

    #[test]
    fn brief_deterministic() {
        let data: Vec<f32> = (0..64 * 64).map(|i| (i % 256) as f32 / 255.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 64, 64).unwrap();
        let kps = vec![KeyPoint::new(32.0, 32.0)];
        let (_, d1) = compute_brief(&img, &kps, 31);
        let (_, d2) = compute_brief(&img, &kps, 31);
        assert_eq!(d1[0].data, d2[0].data);
    }

    #[test]
    fn border_keypoints_skipped() {
        let data = vec![0.5f32; 32 * 32];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 32, 32).unwrap();
        let kps = vec![KeyPoint::new(2.0, 2.0)]; // too close to border for patch_size=31
        let (surviving, _) = compute_brief(&img, &kps, 31);
        assert!(surviving.is_empty());
    }
}
