// SPDX-License-Identifier: MIT OR Apache-2.0
//! FAST-9 corner detector (Features from Accelerated Segment Test).

use crate::features::keypoint::KeyPoint;
use crate::image::{Gray, ImageBuf};

/// The 16 pixel offsets forming the Bresenham circle of radius 3.
const CIRCLE: [(i32, i32); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

/// Detect FAST-9 corners in a grayscale f32 image.
///
/// A pixel is a corner if at least 9 contiguous pixels on the 16-pixel
/// Bresenham circle are all brighter or all darker than the center by
/// at least `threshold`.
///
/// Returns keypoints with `response` set to the corner score.
pub fn detect_fast9(img: &ImageBuf<f32, Gray>, threshold: f32) -> Vec<KeyPoint> {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let data = img.as_slice();
    let mut corners = Vec::new();

    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let center = data[y as usize * w as usize + x as usize];
            let hi = center + threshold;
            let lo = center - threshold;

            // Quick reject: test pixels 0, 4, 8, 12 (the cardinal directions)
            let p0 = data[(y + CIRCLE[0].1) as usize * w as usize + (x + CIRCLE[0].0) as usize];
            let p4 = data[(y + CIRCLE[4].1) as usize * w as usize + (x + CIRCLE[4].0) as usize];
            let p8 = data[(y + CIRCLE[8].1) as usize * w as usize + (x + CIRCLE[8].0) as usize];
            let p12 =
                data[(y + CIRCLE[12].1) as usize * w as usize + (x + CIRCLE[12].0) as usize];

            // At least 3 of {0,4,8,12} must be above hi or below lo
            let n_above = (p0 > hi) as u8 + (p4 > hi) as u8 + (p8 > hi) as u8 + (p12 > hi) as u8;
            let n_below = (p0 < lo) as u8 + (p4 < lo) as u8 + (p8 < lo) as u8 + (p12 < lo) as u8;

            if n_above < 3 && n_below < 3 {
                continue;
            }

            // Full 16-pixel test: check for 9 contiguous pixels
            if let Some(score) = fast9_score(data, w, x, y, hi, lo) {
                corners.push(
                    KeyPoint::new(x as f32, y as f32)
                        .with_response(score),
                );
            }
        }
    }

    corners
}

/// Check for 9 contiguous brighter or darker pixels and return the score.
fn fast9_score(data: &[f32], w: i32, cx: i32, cy: i32, hi: f32, lo: f32) -> Option<f32> {
    // Read all 16 circle pixels
    let mut bright = [false; 16];
    let mut dark = [false; 16];
    let center = data[cy as usize * w as usize + cx as usize];

    for (i, &(dx, dy)) in CIRCLE.iter().enumerate() {
        let v = data[(cy + dy) as usize * w as usize + (cx + dx) as usize];
        bright[i] = v > hi;
        dark[i] = v < lo;
    }

    // Check for 9 contiguous in the circular buffer
    let has_bright = has_contiguous(&bright, 9);
    let has_dark = has_contiguous(&dark, 9);

    if !has_bright && !has_dark {
        return None;
    }

    // Score = sum of |p_i - center| - threshold for qualifying pixels
    let mut score = 0.0f32;
    for &(dx, dy) in &CIRCLE {
        let v = data[(cy + dy) as usize * w as usize + (cx + dx) as usize];
        let diff = (v - center).abs();
        score += diff;
    }

    Some(score)
}

/// Check if a 16-element circular buffer has `n` contiguous `true` values.
fn has_contiguous(arr: &[bool; 16], n: usize) -> bool {
    let mut run = 0usize;
    // Check 16 + (n-1) positions to handle wrap-around
    for i in 0..16 + n - 1 {
        if arr[i % 16] {
            run += 1;
            if run >= n {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// Non-maximum suppression: keep only local maxima in a `radius` neighborhood.
pub fn nms(keypoints: &[KeyPoint], radius: f32) -> Vec<KeyPoint> {
    let r2 = radius * radius;
    let mut keep = vec![true; keypoints.len()];

    for i in 0..keypoints.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..keypoints.len() {
            if !keep[j] {
                continue;
            }
            let dx = keypoints[i].x - keypoints[j].x;
            let dy = keypoints[i].y - keypoints[j].y;
            if dx * dx + dy * dy < r2 {
                if keypoints[i].response >= keypoints[j].response {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break;
                }
            }
        }
    }

    keypoints
        .iter()
        .zip(&keep)
        .filter(|(_, &k)| k)
        .map(|(kp, _)| kp.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_corner_on_bright_cross() {
        // Create an image with a bright corner pattern
        let mut data = vec![0.5f32; 32 * 32];
        // Make a bright cross at (16, 16)
        for &(dx, dy) in &CIRCLE {
            let x = (16 + dx) as usize;
            let y = (16 + dy) as usize;
            data[y * 32 + x] = 1.0;
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 32, 32).unwrap();
        let corners = detect_fast9(&img, 0.2);
        // Should detect the center as a corner
        let near_center = corners
            .iter()
            .any(|kp| (kp.x - 16.0).abs() < 2.0 && (kp.y - 16.0).abs() < 2.0);
        assert!(near_center, "should detect corner near (16,16)");
    }

    #[test]
    fn no_corners_on_uniform() {
        let data = vec![0.5f32; 32 * 32];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 32, 32).unwrap();
        let corners = detect_fast9(&img, 0.1);
        assert!(corners.is_empty(), "uniform image should have no corners");
    }

    #[test]
    fn has_contiguous_works() {
        let mut arr = [false; 16];
        for i in 0..9 {
            arr[i] = true;
        }
        assert!(has_contiguous(&arr, 9));
        assert!(!has_contiguous(&arr, 10));
    }

    #[test]
    fn has_contiguous_wraps() {
        let mut arr = [false; 16];
        for i in 12..16 {
            arr[i] = true;
        }
        for i in 0..5 {
            arr[i] = true;
        }
        assert!(has_contiguous(&arr, 9));
    }

    #[test]
    fn nms_suppresses_neighbors() {
        let kps = vec![
            KeyPoint::new(10.0, 10.0).with_response(5.0),
            KeyPoint::new(11.0, 10.0).with_response(3.0),
            KeyPoint::new(50.0, 50.0).with_response(4.0),
        ];
        let result = nms(&kps, 5.0);
        assert_eq!(result.len(), 2); // first suppresses second
        assert!((result[0].x - 10.0).abs() < 0.1);
        assert!((result[1].x - 50.0).abs() < 0.1);
    }
}
