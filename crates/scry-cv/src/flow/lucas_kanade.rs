// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pyramidal Lucas-Kanade sparse optical flow.
//!
//! Bouguet, "Pyramidal Implementation of the Lucas Kanade Feature Tracker" (2001).
//! Tracks a sparse set of points from one frame to the next using image gradients
//! and a coarse-to-fine pyramid scheme.

use crate::error::Result;
use crate::flow::flow_field::SparseFlowResult;
use crate::flow::SparseOpticalFlow;
use crate::image::{Gray, ImageBuf};
use crate::math::bilinear_at;
use crate::pyramid::gaussian::downsample_bilinear;

/// Pyramidal Lucas-Kanade sparse optical flow tracker.
///
/// # Example
///
/// ```
/// use scry_cv::flow::{LucasKanade, SparseOpticalFlow};
/// use scry_cv::prelude::*;
///
/// let prev = GrayImageF::new(64, 64).unwrap();
/// let next = GrayImageF::new(64, 64).unwrap();
/// let mut lk = LucasKanade::new();
/// let result = lk.calc(&prev, &next, &[(32.0, 32.0)]).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct LucasKanade {
    /// Window half-size (full window is `2*win_size+1` square).
    pub win_size: u32,
    /// Number of pyramid levels.
    pub max_level: u32,
    /// Maximum iterations per level for the iterative solver.
    pub max_iters: u32,
    /// Convergence threshold (displacement delta).
    pub epsilon: f32,
    /// Minimum eigenvalue threshold for the structure tensor.
    /// Points with smaller eigenvalue are marked as lost.
    pub min_eig_threshold: f32,
}

impl Default for LucasKanade {
    fn default() -> Self {
        Self {
            win_size: 7,
            max_level: 3,
            max_iters: 30,
            epsilon: 0.01,
            min_eig_threshold: 1e-4,
        }
    }
}

impl LucasKanade {
    /// Create a new tracker with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set window half-size.
    #[must_use]
    pub fn win_size(mut self, v: u32) -> Self {
        self.win_size = v;
        self
    }

    /// Set maximum pyramid levels.
    #[must_use]
    pub fn max_level(mut self, v: u32) -> Self {
        self.max_level = v;
        self
    }

    /// Set maximum iterations.
    #[must_use]
    pub fn max_iters(mut self, v: u32) -> Self {
        self.max_iters = v;
        self
    }
}

impl SparseOpticalFlow for LucasKanade {
    fn calc(
        &mut self,
        prev: &ImageBuf<f32, Gray>,
        next: &ImageBuf<f32, Gray>,
        prev_pts: &[(f32, f32)],
    ) -> Result<SparseFlowResult> {
        // Build pyramids
        let prev_pyr = build_pyramid(prev, self.max_level)?;
        let next_pyr = build_pyramid(next, self.max_level)?;

        let n = prev_pts.len();
        let mut next_pts = vec![(0.0f32, 0.0f32); n];
        let mut status = vec![true; n];
        let mut errors = vec![0.0f32; n];

        let top_level = prev_pyr.len() - 1;

        for (i, &(px, py)) in prev_pts.iter().enumerate() {
            // Initial guess at coarsest level
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;

            for level in (0..=top_level).rev() {
                let scale = (1u32 << level) as f32;
                let prev_img = &prev_pyr[level];
                let next_img = &next_pyr[level];

                let pw = prev_img.width();
                let ph = prev_img.height();
                let prev_data = prev_img.as_slice();
                let next_data = next_img.as_slice();

                // Point location at this level
                let cx = px / scale;
                let cy = py / scale;

                let ws = self.win_size as i32;

                // Compute structure tensor G and mismatch vector b
                let mut g_xx = 0.0f32;
                let mut g_xy = 0.0f32;
                let mut g_yy = 0.0f32;

                // Precompute gradients and template differences
                for wy in -ws..=ws {
                    for wx in -ws..=ws {
                        let sx = cx + wx as f32;
                        let sy = cy + wy as f32;

                        if sx < 0.0
                            || sy < 0.0
                            || sx >= (pw - 1) as f32
                            || sy >= (ph - 1) as f32
                        {
                            continue;
                        }

                        let ix = gradient_x(prev_data, pw, ph, sx, sy);
                        let iy = gradient_y(prev_data, pw, ph, sx, sy);

                        g_xx += ix * ix;
                        g_xy += ix * iy;
                        g_yy += iy * iy;
                    }
                }

                // Check min eigenvalue of G
                let trace = g_xx + g_yy;
                let det = g_xx * g_yy - g_xy * g_xy;
                let disc = (trace * trace - 4.0 * det).max(0.0);
                let min_eig = (trace - disc.sqrt()) * 0.5;

                let win_area = ((2 * ws + 1) * (2 * ws + 1)) as f32;
                if min_eig < self.min_eig_threshold * win_area {
                    status[i] = false;
                    break;
                }

                let inv_det = 1.0 / det;

                // Iterative refinement
                let mut vx = gx / scale;
                let mut vy = gy / scale;

                for _ in 0..self.max_iters {
                    let mut bx = 0.0f32;
                    let mut by = 0.0f32;

                    for wy in -ws..=ws {
                        for wx in -ws..=ws {
                            let sx = cx + wx as f32;
                            let sy = cy + wy as f32;
                            let tx = sx + vx;
                            let ty = sy + vy;

                            if sx < 0.0
                                || sy < 0.0
                                || sx >= (pw - 1) as f32
                                || sy >= (ph - 1) as f32
                                || tx < 0.0
                                || ty < 0.0
                                || tx >= (pw - 1) as f32
                                || ty >= (ph - 1) as f32
                            {
                                continue;
                            }

                            let ix = gradient_x(prev_data, pw, ph, sx, sy);
                            let iy = gradient_y(prev_data, pw, ph, sx, sy);
                            let it = bilinear_at(next_data, pw, ph, tx, ty)
                                - bilinear_at(prev_data, pw, ph, sx, sy);

                            bx += ix * it;
                            by += iy * it;
                        }
                    }

                    // Solve G * delta = -b
                    let dvx = -inv_det * (g_yy * bx - g_xy * by);
                    let dvy = -inv_det * (-g_xy * bx + g_xx * by);

                    vx += dvx;
                    vy += dvy;

                    if dvx * dvx + dvy * dvy < self.epsilon * self.epsilon {
                        break;
                    }
                }

                // Propagate to next (finer) level
                gx = vx * scale;
                gy = vy * scale;
            }

            if status[i] {
                next_pts[i] = (px + gx, py + gy);

                // Compute tracking error (SSD in the window at finest level)
                let prev_data = prev_pyr[0].as_slice();
                let next_data = next_pyr[0].as_slice();
                let pw = prev_pyr[0].width();
                let ph = prev_pyr[0].height();
                let ws = self.win_size as i32;
                let mut err = 0.0f32;
                let mut count = 0u32;

                for wy in -ws..=ws {
                    for wx in -ws..=ws {
                        let sx = px + wx as f32;
                        let sy = py + wy as f32;
                        let tx = next_pts[i].0 + wx as f32;
                        let ty = next_pts[i].1 + wy as f32;

                        if sx >= 0.0
                            && sy >= 0.0
                            && sx < (pw - 1) as f32
                            && sy < (ph - 1) as f32
                            && tx >= 0.0
                            && ty >= 0.0
                            && tx < (pw - 1) as f32
                            && ty < (ph - 1) as f32
                        {
                            let d = bilinear_at(prev_data, pw, ph, sx, sy)
                                - bilinear_at(next_data, pw, ph, tx, ty);
                            err += d * d;
                            count += 1;
                        }
                    }
                }

                errors[i] = if count > 0 {
                    (err / count as f32).sqrt()
                } else {
                    f32::MAX
                };
            }
        }

        Ok(SparseFlowResult {
            next_pts,
            status,
            errors,
        })
    }
}

/// Build a factor-of-2 Gaussian pyramid.
fn build_pyramid(img: &ImageBuf<f32, Gray>, max_level: u32) -> Result<Vec<ImageBuf<f32, Gray>>> {
    let mut pyr = Vec::with_capacity(max_level as usize + 1);
    pyr.push(img.clone());

    for _ in 0..max_level {
        let prev = pyr.last().unwrap();
        let (w, h) = prev.dimensions();
        let nw = (w / 2).max(1);
        let nh = (h / 2).max(1);
        if nw < 2 || nh < 2 {
            break;
        }
        let blurred = crate::filter::gaussian::gaussian_blur(prev, 1.0)?;
        pyr.push(downsample_bilinear(&blurred, nw, nh)?);
    }

    Ok(pyr)
}

/// Central-difference gradient in x at sub-pixel location.
#[inline]
fn gradient_x(data: &[f32], w: u32, h: u32, x: f32, y: f32) -> f32 {
    let x0 = (x - 1.0).max(0.0);
    let x1 = (x + 1.0).min((w - 1) as f32);
    (bilinear_at(data, w, h, x1, y) - bilinear_at(data, w, h, x0, y)) * 0.5
}

/// Central-difference gradient in y at sub-pixel location.
#[inline]
fn gradient_y(data: &[f32], w: u32, h: u32, x: f32, y: f32) -> f32 {
    let y0 = (y - 1.0).max(0.0);
    let y1 = (y + 1.0).min((h - 1) as f32);
    (bilinear_at(data, w, h, x, y1) - bilinear_at(data, w, h, x, y0)) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Gray, ImageBuf};

    fn make_shifted_pair(w: u32, h: u32, dx: f32, dy: f32) -> (ImageBuf<f32, Gray>, ImageBuf<f32, Gray>) {
        // Gaussian blob centered at (w/2, h/2)
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let sigma = 8.0f32;

        let make_frame = |ox: f32, oy: f32| -> ImageBuf<f32, Gray> {
            let data: Vec<f32> = (0..h)
                .flat_map(|y| {
                    (0..w).map(move |x| {
                        let fx = x as f32 - cx - ox;
                        let fy = y as f32 - cy - oy;
                        (-(fx * fx + fy * fy) / (2.0 * sigma * sigma)).exp()
                    })
                })
                .collect();
            ImageBuf::from_vec(data, w, h).unwrap()
        };

        (make_frame(0.0, 0.0), make_frame(dx, dy))
    }

    #[test]
    fn tracks_horizontal_shift() {
        let (prev, next) = make_shifted_pair(64, 64, 3.0, 0.0);
        let mut lk = LucasKanade::new().win_size(11).max_level(3);
        let pts = vec![(32.0, 32.0)];

        let result = lk.calc(&prev, &next, &pts).unwrap();
        assert!(result.status[0], "tracking should succeed");

        let (nx, ny) = result.next_pts[0];
        let err_x = (nx - 35.0).abs();
        let err_y = (ny - 32.0).abs();
        assert!(
            err_x < 1.5 && err_y < 1.5,
            "expected ~(35, 32), got ({nx}, {ny})"
        );
    }

    #[test]
    fn tracks_diagonal_shift() {
        let (prev, next) = make_shifted_pair(64, 64, 2.0, 2.0);
        let mut lk = LucasKanade::new().win_size(11).max_level(3);
        let pts = vec![(32.0, 32.0)];

        let result = lk.calc(&prev, &next, &pts).unwrap();
        assert!(result.status[0], "tracking should succeed");

        let (nx, ny) = result.next_pts[0];
        let err_x = (nx - 34.0).abs();
        let err_y = (ny - 34.0).abs();
        assert!(
            err_x < 1.5 && err_y < 1.5,
            "expected ~(34, 34), got ({nx}, {ny})"
        );
    }

    #[test]
    fn zero_flow_on_identical_frames() {
        let (prev, _) = make_shifted_pair(64, 64, 0.0, 0.0);
        let next = prev.clone();
        let mut lk = LucasKanade::new();
        // Only test center where there's enough texture
        let pts = vec![(32.0, 32.0)];

        let result = lk.calc(&prev, &next, &pts).unwrap();
        for (i, &(nx, ny)) in result.next_pts.iter().enumerate() {
            if !result.status[i] {
                continue; // skip lost points
            }
            let (px, py) = pts[i];
            assert!(
                (nx - px).abs() < 0.5 && (ny - py).abs() < 0.5,
                "point {i}: expected ~({px}, {py}), got ({nx}, {ny})"
            );
        }
    }

    #[test]
    fn lost_on_uniform_region() {
        let data = vec![0.5f32; 64 * 64];
        let prev = ImageBuf::<f32, Gray>::from_vec(data.clone(), 64, 64).unwrap();
        let next = ImageBuf::<f32, Gray>::from_vec(data, 64, 64).unwrap();
        let mut lk = LucasKanade::new();

        let result = lk.calc(&prev, &next, &[(32.0, 32.0)]).unwrap();
        assert!(!result.status[0], "should lose tracking on uniform region");
    }
}
