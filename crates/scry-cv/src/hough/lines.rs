// SPDX-License-Identifier: MIT OR Apache-2.0
//! Standard Hough Transform for line detection in (rho, theta) parameter space.
//!
//! Given a binary edge image, accumulates votes for all possible lines passing
//! through each edge pixel and returns lines that exceed a vote threshold.

use crate::error::{Result, ScryVisionError};
use crate::image::{Gray, ImageBuf};

/// A detected line in polar coordinates.
#[derive(Clone, Debug)]
pub struct HoughLine {
    /// Perpendicular distance from the origin to the line.
    pub rho: f32,
    /// Angle of the line normal in radians `[0, pi)`.
    pub theta: f32,
    /// Number of votes (edge pixels) supporting this line.
    pub votes: u32,
}

/// Detect lines in a binary edge image using the Standard Hough Transform.
///
/// # Arguments
/// * `edges` — Binary edge image (values > 0.5 are edge pixels).
/// * `rho_res` — Distance resolution of the accumulator in pixels (typically 1.0).
/// * `theta_res` — Angle resolution of the accumulator in radians (typically `PI/180`).
/// * `threshold` — Minimum number of votes for a line to be returned.
///
/// Returns lines sorted by vote count (descending).
///
/// # Example
///
/// ```
/// use scry_cv::prelude::*;
/// use scry_cv::hough::lines::hough_lines;
/// use std::f32::consts::PI;
///
/// let img = GrayImageF::new(64, 64).unwrap();
/// let lines = hough_lines(&img, 1.0, PI / 180.0, 10).unwrap();
/// assert!(lines.is_empty()); // no edges → no lines
/// ```
pub fn hough_lines(
    edges: &ImageBuf<f32, Gray>,
    rho_res: f32,
    theta_res: f32,
    threshold: u32,
) -> Result<Vec<HoughLine>> {
    if rho_res <= 0.0 {
        return Err(ScryVisionError::InvalidParameter(
            "rho_res must be > 0".into(),
        ));
    }
    if theta_res <= 0.0 {
        return Err(ScryVisionError::InvalidParameter(
            "theta_res must be > 0".into(),
        ));
    }

    let w = edges.width() as usize;
    let h = edges.height() as usize;
    let diagonal = ((w * w + h * h) as f32).sqrt();

    // Accumulator dimensions
    let n_theta = (std::f32::consts::PI / theta_res).ceil() as usize;
    let max_rho = diagonal;
    let n_rho = (2.0 * max_rho / rho_res).ceil() as usize + 1;
    let rho_offset = (n_rho / 2) as f32 * rho_res; // rho can be negative

    // Precompute sin/cos tables
    let mut cos_table = vec![0.0f32; n_theta];
    let mut sin_table = vec![0.0f32; n_theta];
    for i in 0..n_theta {
        let theta = i as f32 * theta_res;
        cos_table[i] = theta.cos();
        sin_table[i] = theta.sin();
    }

    // Vote
    let mut accum = vec![0u32; n_rho * n_theta];
    let data = edges.as_slice();

    for y in 0..h {
        for x in 0..w {
            if data[y * w + x] <= 0.5 {
                continue;
            }
            let fx = x as f32;
            let fy = y as f32;
            for ti in 0..n_theta {
                let rho = fx * cos_table[ti] + fy * sin_table[ti];
                let ri = ((rho + rho_offset) / rho_res).round() as usize;
                if ri < n_rho {
                    accum[ri * n_theta + ti] += 1;
                }
            }
        }
    }

    // Extract peaks above threshold
    let mut lines: Vec<HoughLine> = Vec::new();
    for ri in 0..n_rho {
        for ti in 0..n_theta {
            let votes = accum[ri * n_theta + ti];
            if votes >= threshold {
                let rho = ri as f32 * rho_res - rho_offset;
                let theta = ti as f32 * theta_res;
                lines.push(HoughLine { rho, theta, votes });
            }
        }
    }

    // Sort by votes descending
    lines.sort_by(|a, b| b.votes.cmp(&a.votes));
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn detects_horizontal_line() {
        let (w, h) = (64u32, 64u32);
        let mut data = vec![0.0f32; (w * h) as usize];
        // Draw horizontal line at y=30
        for x in 0..w {
            data[30 * w as usize + x as usize] = 1.0;
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, w, h).unwrap();
        let lines = hough_lines(&img, 1.0, PI / 180.0, 30).unwrap();

        assert!(!lines.is_empty(), "should detect horizontal line");
        // Horizontal line: theta ≈ pi/2, rho ≈ 30
        let best = &lines[0];
        assert!(
            (best.theta - PI / 2.0).abs() < 0.05,
            "theta should be ~pi/2, got {}",
            best.theta
        );
        assert!(
            (best.rho - 30.0).abs() < 2.0,
            "rho should be ~30, got {}",
            best.rho
        );
    }

    #[test]
    fn detects_vertical_line() {
        let (w, h) = (64u32, 64u32);
        let mut data = vec![0.0f32; (w * h) as usize];
        // Draw vertical line at x=20
        for y in 0..h {
            data[y as usize * w as usize + 20] = 1.0;
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, w, h).unwrap();
        let lines = hough_lines(&img, 1.0, PI / 180.0, 30).unwrap();

        assert!(!lines.is_empty(), "should detect vertical line");
        let best = &lines[0];
        // Vertical line: theta ≈ 0, rho ≈ 20
        assert!(
            best.theta < 0.05 || best.theta > PI - 0.05,
            "theta should be ~0 or ~pi, got {}",
            best.theta
        );
        assert!(
            best.rho.abs() - 20.0 < 2.0,
            "rho should be ~20, got {}",
            best.rho
        );
    }

    #[test]
    fn no_lines_on_empty_image() {
        let img = ImageBuf::<f32, Gray>::new(32, 32).unwrap();
        let lines = hough_lines(&img, 1.0, PI / 180.0, 5).unwrap();
        assert!(lines.is_empty());
    }

    #[test]
    fn rejects_invalid_params() {
        let img = ImageBuf::<f32, Gray>::new(16, 16).unwrap();
        assert!(hough_lines(&img, 0.0, PI / 180.0, 5).is_err());
        assert!(hough_lines(&img, 1.0, 0.0, 5).is_err());
    }
}
