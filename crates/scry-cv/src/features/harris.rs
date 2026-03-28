// SPDX-License-Identifier: MIT OR Apache-2.0
//! Harris corner detector.

use crate::error::Result;
use crate::filter::gaussian::gaussian_blur;
use crate::filter::sobel::{sobel_x, sobel_y};
use crate::image::{Gray, ImageBuf};

/// Compute the Harris corner response map.
///
/// `k` is the Harris sensitivity parameter (typically 0.04-0.06).
/// `sigma` is the Gaussian window sigma for structure tensor smoothing.
///
/// Response = det(M) - k * trace(M)^2, where M is the structure tensor.
pub fn harris_response(
    img: &ImageBuf<f32, Gray>,
    k: f32,
    sigma: f32,
) -> Result<ImageBuf<f32, Gray>> {
    let dx = sobel_x(img)?;
    let dy = sobel_y(img)?;

    let w = img.width();
    let h = img.height();
    let n = (w * h) as usize;

    // Compute products Ixx, Iyy, Ixy
    let mut ixx = vec![0.0f32; n];
    let mut iyy = vec![0.0f32; n];
    let mut ixy = vec![0.0f32; n];

    for i in 0..n {
        let gx = dx.as_slice()[i];
        let gy = dy.as_slice()[i];
        ixx[i] = gx * gx;
        iyy[i] = gy * gy;
        ixy[i] = gx * gy;
    }

    // Smooth the structure tensor components
    let ixx_img = ImageBuf::<f32, Gray>::from_vec(ixx, w, h)?;
    let iyy_img = ImageBuf::<f32, Gray>::from_vec(iyy, w, h)?;
    let ixy_img = ImageBuf::<f32, Gray>::from_vec(ixy, w, h)?;

    let sxx = gaussian_blur(&ixx_img, sigma)?;
    let syy = gaussian_blur(&iyy_img, sigma)?;
    let sxy = gaussian_blur(&ixy_img, sigma)?;

    // Harris response
    let mut response = vec![0.0f32; n];
    for i in 0..n {
        let xx = sxx.as_slice()[i];
        let yy = syy.as_slice()[i];
        let xy = sxy.as_slice()[i];
        let det = xx * yy - xy * xy;
        let trace = xx + yy;
        response[i] = det - k * trace * trace;
    }

    ImageBuf::from_vec(response, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harris_detects_checkerboard_corners() {
        // 16x16 checkerboard with 4x4 squares
        let mut data = vec![0.0f32; 16 * 16];
        for y in 0..16u32 {
            for x in 0..16u32 {
                data[(y * 16 + x) as usize] = if (x / 4 + y / 4) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 16, 16).unwrap();
        let resp = harris_response(&img, 0.04, 1.0).unwrap();

        // The corners of the checkerboard squares should have high response
        let max_resp = resp
            .as_slice()
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        assert!(max_resp > 0.0, "should have positive corners: {max_resp}");
    }

    #[test]
    fn harris_low_on_flat() {
        let data = vec![0.5f32; 16 * 16];
        let img = ImageBuf::<f32, Gray>::from_vec(data, 16, 16).unwrap();
        let resp = harris_response(&img, 0.04, 1.0).unwrap();

        let max_abs = resp
            .as_slice()
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 1e-5, "flat image should have ~zero response: {max_abs}");
    }
}
