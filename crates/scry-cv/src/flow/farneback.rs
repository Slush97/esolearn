// SPDX-License-Identifier: MIT OR Apache-2.0
//! Farneback dense optical flow.
//!
//! Reference: Farneback, "Two-Frame Motion Estimation Based on Polynomial
//! Expansion" (2003). Multi-resolution, iterative dense flow using local
//! polynomial expansion of image neighborhoods.

use crate::error::Result;
use crate::flow::flow_field::FlowField;
use crate::flow::poly_expansion::{poly_expand, PolyExpansion};
use crate::flow::DenseOpticalFlow;
use crate::image::{Gray, ImageBuf};
use crate::pyramid::gaussian::{downsample_bilinear, upsample_bilinear};

/// Farneback dense optical flow estimator.
///
/// # Example
///
/// ```
/// use scry_cv::flow::{Farneback, DenseOpticalFlow};
/// use scry_cv::prelude::*;
///
/// let prev = GrayImageF::new(32, 32).unwrap();
/// let next = GrayImageF::new(32, 32).unwrap();
/// let mut fb = Farneback::new();
/// let flow = fb.calc(&prev, &next).unwrap();
/// assert_eq!(flow.width, 32);
/// ```
#[derive(Clone, Debug)]
pub struct Farneback {
    /// Number of pyramid levels.
    pub levels: u32,
    /// Pyramid scale factor (< 1.0, typically 0.5).
    pub pyr_scale: f32,
    /// Window size for weighted averaging of polynomial coefficients.
    pub win_size: u32,
    /// Number of iterations at each pyramid level.
    pub iterations: u32,
    /// Size of the polynomial expansion neighborhood (5 or 7).
    pub poly_n: u32,
    /// Standard deviation of the Gaussian used in polynomial expansion.
    pub poly_sigma: f32,
}

impl Default for Farneback {
    fn default() -> Self {
        Self {
            levels: 3,
            pyr_scale: 0.5,
            win_size: 15,
            iterations: 3,
            poly_n: 5,
            poly_sigma: 1.1,
        }
    }
}

impl Farneback {
    /// Create with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of pyramid levels.
    #[must_use]
    pub fn levels(mut self, v: u32) -> Self {
        self.levels = v;
        self
    }

    /// Set number of iterations per level.
    #[must_use]
    pub fn iterations(mut self, v: u32) -> Self {
        self.iterations = v;
        self
    }

    /// Set polynomial neighborhood size (5 or 7).
    #[must_use]
    pub fn poly_n(mut self, v: u32) -> Self {
        self.poly_n = v;
        self
    }

    /// Set polynomial sigma.
    #[must_use]
    pub fn poly_sigma(mut self, v: f32) -> Self {
        self.poly_sigma = v;
        self
    }

    /// Set window size for averaging.
    #[must_use]
    pub fn win_size(mut self, v: u32) -> Self {
        self.win_size = v;
        self
    }
}

impl DenseOpticalFlow for Farneback {
    fn calc(
        &mut self,
        prev: &ImageBuf<f32, Gray>,
        next: &ImageBuf<f32, Gray>,
    ) -> Result<FlowField> {
        // Build image pyramids
        let prev_pyr = build_pyr(prev, self.levels, self.pyr_scale)?;
        let next_pyr = build_pyr(next, self.levels, self.pyr_scale)?;

        let top = prev_pyr.len() - 1;

        // Initialize flow at coarsest level
        let top_w = prev_pyr[top].width();
        let top_h = prev_pyr[top].height();
        let mut flow = FlowField::zeros(top_w, top_h);

        // Coarse to fine
        for level in (0..=top).rev() {
            let prev_img = &prev_pyr[level];
            let next_img = &next_pyr[level];
            let lw = prev_img.width();
            let lh = prev_img.height();

            // Upscale flow if not at the top level
            if level < top {
                let scale_x = lw as f32 / flow.width as f32;
                let scale_y = lh as f32 / flow.height as f32;
                flow = upscale_flow(&flow, lw, lh, scale_x, scale_y)?;
            }

            // Polynomial expansion for both frames
            let pe_prev = poly_expand(prev_img, self.poly_n, self.poly_sigma);
            let pe_next = poly_expand(next_img, self.poly_n, self.poly_sigma);

            // Iterative refinement
            for _ in 0..self.iterations {
                update_flow(&mut flow, &pe_prev, &pe_next, self.win_size);
            }
        }

        Ok(flow)
    }
}

/// Build a Gaussian pyramid with the given scale factor.
fn build_pyr(
    img: &ImageBuf<f32, Gray>,
    levels: u32,
    pyr_scale: f32,
) -> Result<Vec<ImageBuf<f32, Gray>>> {
    let mut pyr = Vec::with_capacity(levels as usize);
    pyr.push(img.clone());

    for _ in 1..levels {
        let prev = pyr.last().unwrap();
        let (pw, ph) = prev.dimensions();
        let nw = ((pw as f32 * pyr_scale).round() as u32).max(2);
        let nh = ((ph as f32 * pyr_scale).round() as u32).max(2);
        if nw < 4 || nh < 4 {
            break;
        }
        let blurred = crate::filter::gaussian::gaussian_blur(prev, 0.8)?;
        pyr.push(downsample_bilinear(&blurred, nw, nh)?);
    }

    Ok(pyr)
}

/// Upscale a flow field to new dimensions, scaling the vectors by the
/// ratio between old and new resolution.
fn upscale_flow(
    flow: &FlowField,
    new_w: u32,
    new_h: u32,
    scale_x: f32,
    scale_y: f32,
) -> Result<FlowField> {
    let vx_img = ImageBuf::<f32, Gray>::from_vec(flow.vx.clone(), flow.width, flow.height)?;
    let vy_img = ImageBuf::<f32, Gray>::from_vec(flow.vy.clone(), flow.width, flow.height)?;

    let vx_up = upsample_bilinear(&vx_img, new_w, new_h)?;
    let vy_up = upsample_bilinear(&vy_img, new_w, new_h)?;

    // Scale flow vectors proportionally to the resolution change.
    // At the coarser level, 1 pixel of flow = scale_x pixels at finer level.
    let vx: Vec<f32> = vx_up.as_slice().iter().map(|&v| v * scale_x).collect();
    let vy: Vec<f32> = vy_up.as_slice().iter().map(|&v| v * scale_y).collect();

    Ok(FlowField {
        vx,
        vy,
        width: new_w,
        height: new_h,
    })
}

/// Single iteration of Farneback flow update.
///
/// For each pixel, warps the next-frame polynomial coefficients by the current
/// flow estimate, then solves the 2×2 system to compute the flow update.
fn update_flow(
    flow: &mut FlowField,
    pe_prev: &PolyExpansion,
    pe_next: &PolyExpansion,
    win_size: u32,
) {
    let w = flow.width as i32;
    let h = flow.height as i32;
    let half = win_size as i32 / 2;

    let mut new_vx = vec![0.0f32; flow.vx.len()];
    let mut new_vy = vec![0.0f32; flow.vy.len()];

    for y in 0..h {
        for x in 0..w {
            let idx = y as usize * w as usize + x as usize;

            // Current flow estimate
            let fx = flow.vx[idx];
            let fy = flow.vy[idx];

            // Weighted sum over window of A and b terms
            let mut sum_a11 = 0.0f32;
            let mut sum_a12 = 0.0f32;
            let mut sum_a22 = 0.0f32;
            let mut sum_b1 = 0.0f32;
            let mut sum_b2 = 0.0f32;

            for wy in -half..=half {
                for wx in -half..=half {
                    let nx = x + wx;
                    let ny = y + wy;
                    if nx < 0 || nx >= w || ny < 0 || ny >= h {
                        continue;
                    }

                    let nidx = ny as usize * w as usize + nx as usize;

                    // Warped position in next frame
                    let wx2 = nx as f32 + fx;
                    let wy2 = ny as f32 + fy;
                    let wx2i = (wx2.round() as i32).clamp(0, w - 1);
                    let wy2i = (wy2.round() as i32).clamp(0, h - 1);
                    let nidx2 = wy2i as usize * w as usize + wx2i as usize;

                    // Average the A matrices from both frames
                    let a11 = (pe_prev.a11[nidx] + pe_next.a11[nidx2]) * 0.5;
                    let a12 = (pe_prev.a12[nidx] + pe_next.a12[nidx2]) * 0.5;
                    let a22 = (pe_prev.a22[nidx] + pe_next.a22[nidx2]) * 0.5;

                    // Δb = b_next - b_prev (Farneback convention)
                    let db1 = pe_next.b1[nidx2] - pe_prev.b1[nidx];
                    let db2 = pe_next.b2[nidx2] - pe_prev.b2[nidx];

                    // Accumulate: G += A^T A, h += A^T Δb
                    sum_a11 += a11 * a11 + a12 * a12;
                    sum_a12 += a11 * a12 + a12 * a22;
                    sum_a22 += a12 * a12 + a22 * a22;
                    sum_b1 += a11 * db1 + a12 * db2;
                    sum_b2 += a12 * db1 + a22 * db2;
                }
            }

            // Solve G * d = -h/2 (Farneback displacement equation)
            #[allow(clippy::suspicious_operation_groupings)]
            let det = sum_a11 * sum_a22 - sum_a12 * sum_a12;
            if det.abs() > 1e-10 {
                let inv = -0.5 / det;
                new_vx[idx] = fx + inv * (sum_a22 * sum_b1 - sum_a12 * sum_b2);
                new_vy[idx] = fy + inv * (-sum_a12 * sum_b1 + sum_a11 * sum_b2);
            } else {
                new_vx[idx] = fx;
                new_vy[idx] = fy;
            }
        }
    }

    flow.vx = new_vx;
    flow.vy = new_vy;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_flow_on_identical_frames() {
        let data: Vec<f32> = (0..32)
            .flat_map(|y| {
                (0..32).map(move |x| {
                    let fx = x as f32 - 16.0;
                    let fy = y as f32 - 16.0;
                    (-(fx * fx + fy * fy) / 50.0).exp()
                })
            })
            .collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 32, 32).unwrap();
        let mut fb = Farneback::new().levels(2).iterations(3);
        let flow = fb.calc(&img, &img).unwrap();

        // Average flow magnitude should be near zero
        let avg_mag: f32 = flow
            .vx
            .iter()
            .zip(&flow.vy)
            .map(|(&vx, &vy)| vx.hypot(vy))
            .sum::<f32>()
            / flow.vx.len() as f32;

        assert!(
            avg_mag < 0.5,
            "identical frames should have ~zero flow, avg mag = {avg_mag}"
        );
    }

    #[test]
    fn detects_global_translation_direction() {
        // Verify Farneback detects the correct direction of motion.
        // Exact magnitude is hard to get perfect at low resolution, so we
        // just check the sign and that the magnitude is non-trivial.
        let w = 48u32;
        let h = 48u32;
        let sigma = 10.0f32;

        let make_frame = |offset: f32| -> ImageBuf<f32, Gray> {
            let data: Vec<f32> = (0..h)
                .flat_map(|y| {
                    (0..w).map(move |x| {
                        let fx = x as f32 - w as f32 / 2.0 - offset;
                        let fy = y as f32 - h as f32 / 2.0;
                        (-(fx * fx + fy * fy) / (2.0 * sigma * sigma)).exp()
                    })
                })
                .collect();
            ImageBuf::from_vec(data, w, h).unwrap()
        };

        let prev = make_frame(0.0);
        let next = make_frame(2.0); // shift right by 2

        let mut fb = Farneback::new()
            .levels(1)
            .iterations(10)
            .poly_n(7)
            .poly_sigma(1.5)
            .win_size(15);

        let flow = fb.calc(&prev, &next).unwrap();

        // Check flow at center region
        let margin = 15;
        let mut sum_vx = 0.0f32;
        let mut count = 0u32;
        for y in margin..(h - margin) {
            for x in margin..(w - margin) {
                let (vx, _) = flow.at(x, y);
                sum_vx += vx;
                count += 1;
            }
        }
        let avg_vx = sum_vx / count as f32;

        // Flow should be positive (rightward motion)
        assert!(
            avg_vx > 0.5,
            "expected positive horizontal flow for rightward shift, got {avg_vx}"
        );
    }

    #[test]
    fn flow_field_dimensions_match_input() {
        let prev = ImageBuf::<f32, Gray>::new(40, 30).unwrap();
        let next = ImageBuf::<f32, Gray>::new(40, 30).unwrap();
        let mut fb = Farneback::new();
        let flow = fb.calc(&prev, &next).unwrap();
        assert_eq!(flow.width, 40);
        assert_eq!(flow.height, 30);
    }
}
