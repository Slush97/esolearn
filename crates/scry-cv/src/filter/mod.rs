// SPDX-License-Identifier: MIT OR Apache-2.0
//! Image filtering: convolution, blur, gradients, and edge operators.

pub mod bilateral;
pub mod box_filter;
pub mod gaussian;
pub mod median;
pub mod sobel;

use crate::error::{Result, ScryVisionError};
use crate::image::{Gray, ImageBuf};

/// Apply a generic 2D convolution with a separable kernel.
///
/// `h_kernel` is the horizontal (row) kernel, `v_kernel` is the vertical (column)
/// kernel. The output is a new grayscale f32 image.
pub fn convolve_separable(
    img: &ImageBuf<f32, Gray>,
    h_kernel: &[f32],
    v_kernel: &[f32],
) -> Result<ImageBuf<f32, Gray>> {
    if h_kernel.is_empty() || v_kernel.is_empty() {
        return Err(ScryVisionError::InvalidParameter(
            "kernel must not be empty".into(),
        ));
    }
    let w = img.width();
    let h = img.height();
    let hr = (h_kernel.len() / 2) as i32;
    let vr = (v_kernel.len() / 2) as i32;

    // Horizontal pass → tmp
    let mut tmp = vec![0.0f32; w as usize * h as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = 0.0f32;
            for (ki, &kv) in h_kernel.iter().enumerate() {
                let sx = (x + ki as i32 - hr).clamp(0, w as i32 - 1) as u32;
                sum += img.pixel(sx, y as u32)[0] * kv;
            }
            tmp[y as usize * w as usize + x as usize] = sum;
        }
    }

    // Vertical pass → out
    let mut out = vec![0.0f32; w as usize * h as usize];
    for y in 0..h as i32 {
        for x in 0..w as usize {
            let mut sum = 0.0f32;
            for (ki, &kv) in v_kernel.iter().enumerate() {
                let sy = (y + ki as i32 - vr).clamp(0, h as i32 - 1) as usize;
                sum += tmp[sy * w as usize + x] * kv;
            }
            out[y as usize * w as usize + x] = sum;
        }
    }

    ImageBuf::from_vec(out, w, h)
}

/// Apply a generic 2D convolution with a non-separable kernel.
///
/// `kernel` is row-major, `kw` x `kh` (both must be odd).
pub fn convolve_2d(
    img: &ImageBuf<f32, Gray>,
    kernel: &[f32],
    kw: u32,
    kh: u32,
) -> Result<ImageBuf<f32, Gray>> {
    if kernel.len() != (kw * kh) as usize || kw % 2 == 0 || kh % 2 == 0 {
        return Err(ScryVisionError::InvalidParameter(
            "kernel dimensions must be odd and match data length".into(),
        ));
    }
    let w = img.width();
    let h = img.height();
    let rx = (kw / 2) as i32;
    let ry = (kh / 2) as i32;
    let mut out = vec![0.0f32; w as usize * h as usize];

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = 0.0f32;
            for ky in 0..kh as i32 {
                for kx in 0..kw as i32 {
                    let sx = (x + kx - rx).clamp(0, w as i32 - 1) as u32;
                    let sy = (y + ky - ry).clamp(0, h as i32 - 1) as u32;
                    sum += img.pixel(sx, sy)[0] * kernel[(ky * kw as i32 + kx) as usize];
                }
            }
            out[y as usize * w as usize + x as usize] = sum;
        }
    }

    ImageBuf::from_vec(out, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_convolution() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let img = ImageBuf::<f32, Gray>::from_vec(data.clone(), 3, 3).unwrap();
        // Identity kernel
        let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let out = convolve_2d(&img, &kernel, 3, 3).unwrap();
        for i in 0..9 {
            assert!(
                (out.as_slice()[i] - data[i]).abs() < 1e-5,
                "index {i}: expected {}, got {}",
                data[i],
                out.as_slice()[i]
            );
        }
    }

    #[test]
    fn separable_matches_2d_gaussian() {
        let data: Vec<f32> = (0..25).map(|i| i as f32 / 24.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 5, 5).unwrap();

        let k1d = crate::math::gaussian_kernel_1d(1.0);
        let sep = convolve_separable(&img, &k1d, &k1d).unwrap();

        // Build 2D kernel from outer product
        let n = k1d.len();
        let mut k2d = vec![0.0f32; n * n];
        for (iy, &ky) in k1d.iter().enumerate() {
            for (ix, &kx) in k1d.iter().enumerate() {
                k2d[iy * n + ix] = ky * kx;
            }
        }
        let full = convolve_2d(&img, &k2d, n as u32, n as u32).unwrap();

        for i in 0..25 {
            assert!(
                (sep.as_slice()[i] - full.as_slice()[i]).abs() < 1e-4,
                "index {i}: sep={}, full={}",
                sep.as_slice()[i],
                full.as_slice()[i]
            );
        }
    }
}
