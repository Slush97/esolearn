// SPDX-License-Identifier: MIT OR Apache-2.0
//! Numeric helper functions used across the crate.

/// Bilinear interpolation at sub-pixel coordinates.
///
/// `data` is a row-major f32 buffer of size `(width * height)`.
/// Returns the interpolated value at `(x, y)`, clamping to image bounds.
#[inline]
pub fn bilinear_at(data: &[f32], width: u32, height: u32, x: f32, y: f32) -> f32 {
    let x0 = (x.floor() as i32).clamp(0, width as i32 - 1) as u32;
    let y0 = (y.floor() as i32).clamp(0, height as i32 - 1) as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let w = width as usize;
    let p00 = data[y0 as usize * w + x0 as usize];
    let p10 = data[y0 as usize * w + x1 as usize];
    let p01 = data[y1 as usize * w + x0 as usize];
    let p11 = data[y1 as usize * w + x1 as usize];

    let top = p00 + fx * (p10 - p00);
    let bot = p01 + fx * (p11 - p01);
    top + fy * (bot - top)
}

/// Fast 2-argument arctangent approximation.
///
/// Maximum error ~0.01 radians. Use when `f32::atan2` precision is
/// acceptable but speed matters (e.g. orientation computation per keypoint).
#[inline]
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    // Fall back to std for now — profile before replacing with approximation.
    y.atan2(x)
}

/// Clamp a value to `[lo, hi]`.
#[inline]
pub fn clamp<T: PartialOrd>(v: T, lo: T, hi: T) -> T {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// Compute Gaussian kernel weights (1D, symmetric, normalized).
///
/// Returns a vector of `2 * radius + 1` weights.
pub fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;

    for i in 0..size {
        let x = i as f32 - radius as f32;
        let w = (-x * x / s2).exp();
        kernel.push(w);
        sum += w;
    }

    // Normalize
    let inv = 1.0 / sum;
    for w in &mut kernel {
        *w *= inv;
    }
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_center() {
        // 2x2 image: [0, 1, 2, 3]
        let data = [0.0f32, 1.0, 2.0, 3.0];
        let v = bilinear_at(&data, 2, 2, 0.5, 0.5);
        assert!((v - 1.5).abs() < 1e-5, "expected 1.5, got {v}");
    }

    #[test]
    fn bilinear_corner() {
        let data = [10.0f32, 20.0, 30.0, 40.0];
        let v = bilinear_at(&data, 2, 2, 0.0, 0.0);
        assert!((v - 10.0).abs() < 1e-5);
    }

    #[test]
    fn gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(1.0);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn gaussian_kernel_symmetric() {
        let k = gaussian_kernel_1d(2.0);
        let n = k.len();
        for i in 0..n / 2 {
            assert!((k[i] - k[n - 1 - i]).abs() < 1e-7);
        }
    }
}
