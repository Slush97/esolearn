// SPDX-License-Identifier: MIT OR Apache-2.0
//! Polynomial expansion for Farneback optical flow.
//!
//! Approximates a local neighborhood of each pixel with a quadratic polynomial:
//! `f(x) ≈ x^T A x + b^T x + c`, where A is 2×2 symmetric, b is 2×1, c is scalar.
//!
//! Reference: Farneback, "Two-Frame Motion Estimation Based on Polynomial Expansion" (2003).

use crate::image::{Gray, ImageBuf};

/// Result of polynomial expansion at each pixel.
///
/// For an image of size W×H, each output has W×H elements stored row-major.
pub struct PolyExpansion {
    /// `a11` coefficient of A matrix (W×H).
    pub a11: Vec<f32>,
    /// `a12` coefficient of A matrix (= a21, symmetric) (W×H).
    pub a12: Vec<f32>,
    /// `a22` coefficient of A matrix (W×H).
    pub a22: Vec<f32>,
    /// `b1` coefficient (W×H).
    pub b1: Vec<f32>,
    /// `b2` coefficient (W×H).
    pub b2: Vec<f32>,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
}

/// Compute polynomial expansion coefficients for a grayscale image.
///
/// `poly_n` is the polynomial expansion neighborhood size (typically 5 or 7).
/// `poly_sigma` is the standard deviation of the Gaussian weight (typically 1.1 or 1.5).
pub fn poly_expand(
    img: &ImageBuf<f32, Gray>,
    poly_n: u32,
    poly_sigma: f32,
) -> PolyExpansion {
    let w = img.width();
    let h = img.height();
    let data = img.as_slice();
    let n = w as usize * h as usize;
    let half = poly_n as i32 / 2;

    // Precompute Gaussian weights
    let sigma2 = 2.0 * poly_sigma * poly_sigma;
    let mut weights = Vec::new();
    let mut coords = Vec::new();
    for dy in -half..=half {
        for dx in -half..=half {
            let r2 = (dx * dx + dy * dy) as f32;
            let w_val = (-r2 / sigma2).exp();
            weights.push(w_val);
            coords.push((dx, dy));
        }
    }

    let mut a11 = vec![0.0f32; n];
    let mut a12 = vec![0.0f32; n];
    let mut a22 = vec![0.0f32; n];
    let mut b1 = vec![0.0f32; n];
    let mut b2 = vec![0.0f32; n];

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let idx = y as usize * w as usize + x as usize;

            // Weighted least-squares fit of f(dx, dy) ≈ a11*dx² + 2*a12*dx*dy + a22*dy² + b1*dx + b2*dy + c
            // Using normal equations with the polynomial basis [dx², dx*dy, dy², dx, dy, 1]
            //
            // For efficiency, use the closed-form solution from Farneback:
            // G = Σ w_i * r_i * r_i^T,  where r_i = [dx², dx*dy, dy², dx, dy, 1]
            // h = Σ w_i * r_i * f(x+dx, y+dy)
            // coefficients = G^{-1} h
            //
            // Simplified: since we only need A and b (not c), we use the structure:
            let mut s_xx = 0.0f32;
            let mut s_xy = 0.0f32;
            let mut s_yy = 0.0f32;

            // Gram matrix components for the [dx, dy] part
            let mut g_dxdx = 0.0f32;
            let mut g_dxdy = 0.0f32;
            let mut g_dydy = 0.0f32;
            let mut g_dx_f = 0.0f32;
            let mut g_dy_f = 0.0f32;

            // Components for quadratic terms
            let mut g_xx_xx = 0.0f32;
            let mut g_xx_xy = 0.0f32;
            let mut g_xx_yy = 0.0f32;
            let mut g_xy_xy = 0.0f32;
            let mut g_xy_yy = 0.0f32;
            let mut g_yy_yy = 0.0f32;
            let mut g_xx_f = 0.0f32;
            let mut g_xy_f = 0.0f32;
            let mut g_yy_f = 0.0f32;

            let f_center = data[idx];

            for (k, &(dx, dy)) in coords.iter().enumerate() {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
                    continue;
                }

                let f_val = data[ny as usize * w as usize + nx as usize];
                let wt = weights[k];
                let fdx = dx as f32;
                let fdy = dy as f32;
                let xx = fdx * fdx;
                let xy = fdx * fdy;
                let yy = fdy * fdy;

                // Accumulate for linear part (gradient)
                g_dxdx += wt * fdx * fdx;
                g_dxdy += wt * fdx * fdy;
                g_dydy += wt * fdy * fdy;
                g_dx_f += wt * fdx * f_val;
                g_dy_f += wt * fdy * f_val;

                // Accumulate for quadratic part (Hessian)
                g_xx_xx += wt * xx * xx;
                g_xx_xy += wt * xx * xy;
                g_xx_yy += wt * xx * yy;
                g_xy_xy += wt * xy * xy;
                g_xy_yy += wt * xy * yy;
                g_yy_yy += wt * yy * yy;
                g_xx_f += wt * xx * f_val;
                g_xy_f += wt * xy * f_val;
                g_yy_f += wt * yy * f_val;

                s_xx += wt * xx;
                s_xy += wt * xy;
                s_yy += wt * yy;
            }

            // Solve for b (gradient) using 2×2 system: [g_dxdx, g_dxdy; g_dxdy, g_dydy] * [b1, b2] = [g_dx_f, g_dy_f]
            let det_b = g_dxdx * g_dydy - g_dxdy * g_dxdy;
            if det_b.abs() > 1e-10 {
                let inv = 1.0 / det_b;
                b1[idx] = inv * (g_dydy * g_dx_f - g_dxdy * g_dy_f);
                b2[idx] = inv * (-g_dxdy * g_dx_f + g_dxdx * g_dy_f);
            }

            // Solve for A (Hessian/2) using 3×3 system for [a11, a12, a22]
            // [g_xx_xx, g_xx_xy, g_xx_yy] [a11]   [g_xx_f - s_xx * c]
            // [g_xx_xy, g_xy_xy, g_xy_yy] [a12] = [g_xy_f - s_xy * c]
            // [g_xx_yy, g_xy_yy, g_yy_yy] [a22]   [g_yy_f - s_yy * c]
            //
            // Approximate c ≈ f_center for simplicity
            let rhs0 = g_xx_f - s_xx * f_center;
            let rhs1 = g_xy_f - s_xy * f_center;
            let rhs2 = g_yy_f - s_yy * f_center;

            let det_a = g_xx_xx * (g_xy_xy * g_yy_yy - g_xy_yy * g_xy_yy)
                - g_xx_xy * (g_xx_xy * g_yy_yy - g_xy_yy * g_xx_yy)
                + g_xx_yy * (g_xx_xy * g_xy_yy - g_xy_xy * g_xx_yy);

            if det_a.abs() > 1e-10 {
                let inv = 1.0 / det_a;
                a11[idx] = inv
                    * (rhs0 * (g_xy_xy * g_yy_yy - g_xy_yy * g_xy_yy)
                        - g_xx_xy * (rhs1 * g_yy_yy - g_xy_yy * rhs2)
                        + g_xx_yy * (rhs1 * g_xy_yy - g_xy_xy * rhs2));
                // The basis [dx², dx·dy, dy²] gives the coefficient of dx·dy,
                // which is 2·A₁₂ for the symmetric matrix A. Halve to get A₁₂.
                a12[idx] = 0.5
                    * inv
                    * (g_xx_xx * (rhs1 * g_yy_yy - g_xy_yy * rhs2)
                        - rhs0 * (g_xx_xy * g_yy_yy - g_xy_yy * g_xx_yy)
                        + g_xx_yy * (g_xx_xy * rhs2 - rhs1 * g_xx_yy));
                a22[idx] = inv
                    * (g_xx_xx * (g_xy_xy * rhs2 - rhs1 * g_xy_yy)
                        - g_xx_xy * (g_xx_xy * rhs2 - rhs1 * g_xx_yy)
                        + rhs0 * (g_xx_xy * g_xy_yy - g_xy_xy * g_xx_yy));
            }
        }
    }

    PolyExpansion {
        a11,
        a12,
        a22,
        b1,
        b2,
        width: w,
        height: h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_on_uniform_has_zero_gradient() {
        let data = vec![0.5f32; 32 * 32];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 32, 32).unwrap();
        let pe = poly_expand(&img, 5, 1.1);

        // Interior pixels should have near-zero gradients
        let mut interior_b1 = 0.0f32;
        let mut interior_b2 = 0.0f32;
        for y in 5..27 {
            for x in 5..27 {
                interior_b1 += pe.b1[y * 32 + x].abs();
                interior_b2 += pe.b2[y * 32 + x].abs();
            }
        }

        let count = 22.0 * 22.0;
        assert!(
            interior_b1 / count < 1e-5,
            "b1 should be ~0 on uniform image, avg = {}",
            interior_b1 / count
        );
        assert!(
            interior_b2 / count < 1e-5,
            "b2 should be ~0 on uniform image, avg = {}",
            interior_b2 / count
        );
    }

    #[test]
    fn expansion_detects_horizontal_gradient() {
        // Linear ramp in x: f(x,y) = x/32
        let data: Vec<f32> = (0..32)
            .flat_map(|_y| (0..32).map(|x| x as f32 / 32.0))
            .collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 32, 32).unwrap();
        let pe = poly_expand(&img, 5, 1.1);

        // b1 (df/dx) should be positive in interior
        let mut sum_b1 = 0.0f32;
        for y in 5..27 {
            for x in 5..27 {
                sum_b1 += pe.b1[y * 32 + x];
            }
        }
        let avg_b1 = sum_b1 / (22.0 * 22.0);

        assert!(
            avg_b1 > 0.01,
            "b1 should be positive for x-ramp, avg = {avg_b1}"
        );
    }
}
