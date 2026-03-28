// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hough Circle Transform using gradient-direction voting.
//!
//! Detects circles in a binary edge image by voting along gradient directions
//! to find candidate centers, then verifying radii by counting supporting
//! edge pixels on each candidate circle.

use crate::error::{Result, ScryVisionError};
use crate::filter::sobel::{sobel_x, sobel_y};
use crate::image::{Gray, ImageBuf};

/// A detected circle.
#[derive(Clone, Debug)]
pub struct HoughCircle {
    /// Center x coordinate.
    pub cx: f32,
    /// Center y coordinate.
    pub cy: f32,
    /// Radius in pixels.
    pub radius: f32,
    /// Number of votes supporting this circle.
    pub votes: u32,
}

/// Detect circles in a grayscale image.
///
/// Uses a two-stage approach:
/// 1. Compute edges and gradient directions, vote along gradients for centers.
/// 2. For each candidate center above `center_threshold`, count edge pixels at
///    each radius in `[min_radius, max_radius]` to find the best radius.
///
/// # Arguments
/// * `img` — Grayscale f32 image (will compute edges and gradients internally).
/// * `center_threshold` — Minimum votes for a candidate center.
/// * `radius_threshold` — Minimum edge support for a circle to be accepted.
/// * `min_radius` — Minimum circle radius.
/// * `max_radius` — Maximum circle radius.
/// * `min_dist` — Minimum distance between detected circle centers.
///
/// # Example
///
/// ```
/// use scry_cv::prelude::*;
/// use scry_cv::hough::circles::hough_circles;
///
/// let img = GrayImageF::new(64, 64).unwrap();
/// let circles = hough_circles(&img, 10, 8, 5, 30, 10.0).unwrap();
/// assert!(circles.is_empty());
/// ```
pub fn hough_circles(
    img: &ImageBuf<f32, Gray>,
    center_threshold: u32,
    radius_threshold: u32,
    min_radius: u32,
    max_radius: u32,
    min_dist: f32,
) -> Result<Vec<HoughCircle>> {
    if min_radius > max_radius {
        return Err(ScryVisionError::InvalidParameter(
            "min_radius must be <= max_radius".into(),
        ));
    }

    let w = img.width() as usize;
    let h = img.height() as usize;

    // Compute gradients
    let gx = sobel_x(img)?;
    let gy = sobel_y(img)?;
    let gx_data = gx.as_slice();
    let gy_data = gy.as_slice();

    // Compute gradient magnitude for edge detection
    let mut edge_pixels: Vec<(usize, usize)> = Vec::new();
    let mag_threshold = 0.1f32;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mag = gx_data[idx].hypot(gy_data[idx]);
            if mag > mag_threshold {
                edge_pixels.push((x, y));
            }
        }
    }

    // Stage 1: Vote for centers along gradient direction
    let mut center_accum = vec![0u32; w * h];

    for &(x, y) in &edge_pixels {
        let idx = y * w + x;
        let mag = gx_data[idx].hypot(gy_data[idx]);
        if mag < 1e-6 {
            continue;
        }
        let dx = gx_data[idx] / mag;
        let dy = gy_data[idx] / mag;

        // Vote in both directions along the gradient
        for sign in [-1.0f32, 1.0] {
            for r in min_radius..=max_radius {
                let cx = x as f32 + sign * dx * r as f32;
                let cy = y as f32 + sign * dy * r as f32;
                let cxi = cx.round() as i32;
                let cyi = cy.round() as i32;
                if cxi >= 0 && cxi < w as i32 && cyi >= 0 && cyi < h as i32 {
                    center_accum[cyi as usize * w + cxi as usize] += 1;
                }
            }
        }
    }

    // Extract candidate centers above threshold
    let mut candidates: Vec<(usize, usize, u32)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let votes = center_accum[y * w + x];
            if votes >= center_threshold {
                candidates.push((x, y, votes));
            }
        }
    }
    candidates.sort_by(|a, b| b.2.cmp(&a.2));

    // Stage 2: For each candidate center, find the best radius
    let mut circles: Vec<HoughCircle> = Vec::new();

    for &(cx, cy, _) in &candidates {
        // Check minimum distance to already-accepted circles
        let too_close = circles.iter().any(|c| {
            let dx = cx as f32 - c.cx;
            let dy = cy as f32 - c.cy;
            dx.hypot(dy) < min_dist
        });
        if too_close {
            continue;
        }

        // Count edge pixels at each radius
        let mut best_r = 0u32;
        let mut best_votes = 0u32;

        for r in min_radius..=max_radius {
            let mut votes = 0u32;
            // Sample points on the circle
            let circumference = (2.0 * std::f32::consts::PI * r as f32).ceil() as usize;
            let n_samples = circumference.max(16);
            for i in 0..n_samples {
                let angle = 2.0 * std::f32::consts::PI * i as f32 / n_samples as f32;
                let px = (cx as f32 + r as f32 * angle.cos()).round() as i32;
                let py = (cy as f32 + r as f32 * angle.sin()).round() as i32;
                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    let pidx = py as usize * w + px as usize;
                    let mag = gx_data[pidx].hypot(gy_data[pidx]);
                    if mag > mag_threshold {
                        votes += 1;
                    }
                }
            }
            if votes > best_votes {
                best_votes = votes;
                best_r = r;
            }
        }

        if best_votes >= radius_threshold {
            circles.push(HoughCircle {
                cx: cx as f32,
                cy: cy as f32,
                radius: best_r as f32,
                votes: best_votes,
            });
        }
    }

    circles.sort_by(|a, b| b.votes.cmp(&a.votes));
    Ok(circles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw_circle(data: &mut [f32], w: usize, cx: f32, cy: f32, r: f32) {
        let n = (2.0 * std::f32::consts::PI * r).ceil() as usize * 2;
        for i in 0..n {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / n as f32;
            let px = (cx + r * angle.cos()).round() as i32;
            let py = (cy + r * angle.sin()).round() as i32;
            if px >= 0 && (px as usize) < w && py >= 0 {
                let idx = py as usize * w + px as usize;
                if idx < data.len() {
                    data[idx] = 1.0;
                }
            }
        }
    }

    #[test]
    fn detects_circle() {
        let (w, h) = (80u32, 80u32);
        let mut data = vec![0.0f32; (w * h) as usize];
        draw_circle(&mut data, w as usize, 40.0, 40.0, 20.0);
        let img = ImageBuf::<f32, Gray>::from_vec(data, w, h).unwrap();

        let circles = hough_circles(&img, 8, 10, 15, 25, 10.0).unwrap();
        assert!(!circles.is_empty(), "should detect circle");
        let best = &circles[0];
        assert!(
            (best.cx - 40.0).abs() < 5.0,
            "cx should be ~40, got {}",
            best.cx
        );
        assert!(
            (best.cy - 40.0).abs() < 5.0,
            "cy should be ~40, got {}",
            best.cy
        );
        assert!(
            (best.radius - 20.0).abs() < 3.0,
            "radius should be ~20, got {}",
            best.radius
        );
    }

    #[test]
    fn no_circles_on_empty() {
        let img = ImageBuf::<f32, Gray>::new(32, 32).unwrap();
        let circles = hough_circles(&img, 5, 5, 3, 15, 5.0).unwrap();
        assert!(circles.is_empty());
    }

    #[test]
    fn rejects_invalid_radius_range() {
        let img = ImageBuf::<f32, Gray>::new(32, 32).unwrap();
        assert!(hough_circles(&img, 5, 5, 20, 10, 5.0).is_err());
    }
}
