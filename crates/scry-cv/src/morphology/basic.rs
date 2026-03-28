// SPDX-License-Identifier: MIT OR Apache-2.0
//! Basic morphological operations: erode, dilate, open, close.

use crate::error::{Result, ScryVisionError};
use crate::image::{Gray, ImageBuf};

/// Erode a grayscale image: minimum over the structuring element.
///
/// `kernel` is a flat boolean mask of size `ksize x ksize` (must be odd).
pub fn erode(
    img: &ImageBuf<f32, Gray>,
    kernel: &[bool],
    ksize: u32,
) -> Result<ImageBuf<f32, Gray>> {
    validate_kernel(kernel, ksize)?;
    morphological_op(img, kernel, ksize, f32::min)
}

/// Dilate a grayscale image: maximum over the structuring element.
pub fn dilate(
    img: &ImageBuf<f32, Gray>,
    kernel: &[bool],
    ksize: u32,
) -> Result<ImageBuf<f32, Gray>> {
    validate_kernel(kernel, ksize)?;
    morphological_op(img, kernel, ksize, f32::max)
}

/// Morphological opening: erode then dilate. Removes small bright spots.
pub fn open(
    img: &ImageBuf<f32, Gray>,
    kernel: &[bool],
    ksize: u32,
) -> Result<ImageBuf<f32, Gray>> {
    let eroded = erode(img, kernel, ksize)?;
    dilate(&eroded, kernel, ksize)
}

/// Morphological closing: dilate then erode. Fills small dark gaps.
pub fn close(
    img: &ImageBuf<f32, Gray>,
    kernel: &[bool],
    ksize: u32,
) -> Result<ImageBuf<f32, Gray>> {
    let dilated = dilate(img, kernel, ksize)?;
    erode(&dilated, kernel, ksize)
}

/// Morphological gradient: dilate - erode.
pub fn gradient(
    img: &ImageBuf<f32, Gray>,
    kernel: &[bool],
    ksize: u32,
) -> Result<ImageBuf<f32, Gray>> {
    let d = dilate(img, kernel, ksize)?;
    let e = erode(img, kernel, ksize)?;
    let data: Vec<f32> = d
        .as_slice()
        .iter()
        .zip(e.as_slice())
        .map(|(&a, &b)| a - b)
        .collect();
    ImageBuf::from_vec(data, img.width(), img.height())
}

/// Top-hat: original - opening. Extracts bright features smaller than the kernel.
pub fn top_hat(
    img: &ImageBuf<f32, Gray>,
    kernel: &[bool],
    ksize: u32,
) -> Result<ImageBuf<f32, Gray>> {
    let opened = open(img, kernel, ksize)?;
    let data: Vec<f32> = img
        .as_slice()
        .iter()
        .zip(opened.as_slice())
        .map(|(&a, &b)| a - b)
        .collect();
    ImageBuf::from_vec(data, img.width(), img.height())
}

/// Black-hat: closing - original. Extracts dark features smaller than the kernel.
pub fn black_hat(
    img: &ImageBuf<f32, Gray>,
    kernel: &[bool],
    ksize: u32,
) -> Result<ImageBuf<f32, Gray>> {
    let closed = close(img, kernel, ksize)?;
    let data: Vec<f32> = closed
        .as_slice()
        .iter()
        .zip(img.as_slice())
        .map(|(&a, &b)| a - b)
        .collect();
    ImageBuf::from_vec(data, img.width(), img.height())
}

// ── Internal ──

fn validate_kernel(kernel: &[bool], ksize: u32) -> Result<()> {
    if ksize % 2 == 0 || ksize == 0 {
        return Err(ScryVisionError::InvalidParameter(
            "kernel size must be odd and > 0".into(),
        ));
    }
    if kernel.len() != (ksize * ksize) as usize {
        return Err(ScryVisionError::InvalidParameter(format!(
            "kernel length {} doesn't match ksize {}",
            kernel.len(),
            ksize
        )));
    }
    Ok(())
}

fn morphological_op(
    img: &ImageBuf<f32, Gray>,
    kernel: &[bool],
    ksize: u32,
    op: fn(f32, f32) -> f32,
) -> Result<ImageBuf<f32, Gray>> {
    let w = img.width();
    let h = img.height();
    let r = (ksize / 2) as i32;
    let mut out = vec![0.0f32; w as usize * h as usize];

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut val = if op(0.0, 1.0) > 0.5 {
                f32::NEG_INFINITY // dilate: start with -inf
            } else {
                f32::INFINITY // erode: start with +inf
            };

            for ky in 0..ksize as i32 {
                for kx in 0..ksize as i32 {
                    if !kernel[(ky * ksize as i32 + kx) as usize] {
                        continue;
                    }
                    let sx = (x + kx - r).clamp(0, w as i32 - 1) as u32;
                    let sy = (y + ky - r).clamp(0, h as i32 - 1) as u32;
                    val = op(val, img.pixel(sx, sy)[0]);
                }
            }

            out[y as usize * w as usize + x as usize] = val;
        }
    }

    ImageBuf::from_vec(out, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::{make_kernel, StructuringElement};

    fn rect_kernel_3() -> (Vec<bool>, u32) {
        (make_kernel(StructuringElement::Rect, 3), 3)
    }

    #[test]
    fn erode_shrinks_bright_region() {
        let mut data = vec![0.0f32; 7 * 7];
        for y in 2..5u32 {
            for x in 2..5u32 {
                data[(y * 7 + x) as usize] = 1.0;
            }
        }
        let img = ImageBuf::<f32, Gray>::from_vec(data, 7, 7).unwrap();
        let (k, ks) = rect_kernel_3();
        let eroded = erode(&img, &k, ks).unwrap();

        // Center should still be 1.0, but border of bright region should be 0.0
        assert!((eroded.pixel(3, 3)[0] - 1.0).abs() < 1e-5);
        assert!(eroded.pixel(2, 2)[0] < 0.5, "border should be eroded");
    }

    #[test]
    fn dilate_expands_bright_region() {
        let mut data = vec![0.0f32; 7 * 7];
        data[3 * 7 + 3] = 1.0; // single bright pixel
        let img = ImageBuf::<f32, Gray>::from_vec(data, 7, 7).unwrap();
        let (k, ks) = rect_kernel_3();
        let dilated = dilate(&img, &k, ks).unwrap();

        // Neighbors should also be 1.0
        assert!((dilated.pixel(2, 3)[0] - 1.0).abs() < 1e-5);
        assert!((dilated.pixel(4, 3)[0] - 1.0).abs() < 1e-5);
        assert!((dilated.pixel(3, 2)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn opening_idempotent() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 8, 8).unwrap();
        let (k, ks) = rect_kernel_3();
        let o1 = open(&img, &k, ks).unwrap();
        let o2 = open(&o1, &k, ks).unwrap();

        let max_diff: f32 = o1
            .as_slice()
            .iter()
            .zip(o2.as_slice())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-5,
            "opening should be idempotent, max_diff={max_diff}"
        );
    }

    #[test]
    fn gradient_nonnegative() {
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let img = ImageBuf::<f32, Gray>::from_vec(data, 8, 8).unwrap();
        let (k, ks) = rect_kernel_3();
        let grad = gradient(&img, &k, ks).unwrap();
        assert!(
            grad.as_slice().iter().all(|&v| v >= -1e-7),
            "gradient should be non-negative"
        );
    }
}
