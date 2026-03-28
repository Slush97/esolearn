// SPDX-License-Identifier: MIT OR Apache-2.0
//! Template matching: slide a template over an image and compute a similarity
//! score at each position.
//!
//! Supports Sum of Squared Differences (SSD) and Normalized Cross-Correlation
//! (NCC), both in raw and normalized (zero-mean) variants.

use crate::error::{Result, ScryVisionError};
use crate::image::{Gray, ImageBuf};
use crate::integral::IntegralImage;

/// Template matching method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemplateMatchMethod {
    /// Sum of Squared Differences. Lower = better match.
    Ssd,
    /// Normalized SSD (divided by local energy). Lower = better match.
    SsdNormed,
    /// Cross-Correlation Coefficient (zero-mean NCC). Higher = better match.
    CCorr,
    /// Normalized Cross-Correlation Coefficient. Higher = better match, range [-1, 1].
    CCorrNormed,
}

/// Slide `template` over `img` and compute a score map.
///
/// The output image has dimensions `(img_w - tmpl_w + 1, img_h - tmpl_h + 1)`.
/// Each pixel in the output is the matching score at that position.
///
/// # Example
///
/// ```
/// use scry_cv::prelude::*;
/// use scry_cv::matching::template::{match_template, TemplateMatchMethod};
///
/// let img = GrayImageF::from_vec(vec![0.0; 32 * 32], 32, 32).unwrap();
/// let tmpl = GrayImageF::from_vec(vec![0.0; 8 * 8], 8, 8).unwrap();
/// let score = match_template(&img, &tmpl, TemplateMatchMethod::Ssd).unwrap();
/// assert_eq!(score.dimensions(), (25, 25));
/// ```
pub fn match_template(
    img: &ImageBuf<f32, Gray>,
    template: &ImageBuf<f32, Gray>,
    method: TemplateMatchMethod,
) -> Result<ImageBuf<f32, Gray>> {
    let iw = img.width() as usize;
    let ih = img.height() as usize;
    let tw = template.width() as usize;
    let th = template.height() as usize;

    if tw > iw || th > ih {
        return Err(ScryVisionError::InvalidParameter(
            "template must be smaller than the image".into(),
        ));
    }
    if tw == 0 || th == 0 {
        return Err(ScryVisionError::InvalidParameter(
            "template must not be empty".into(),
        ));
    }

    let ow = iw - tw + 1;
    let oh = ih - th + 1;

    // Precompute template statistics
    let t_data = template.as_slice();
    let t_sum: f64 = t_data.iter().map(|&v| v as f64).sum();
    let t_sq_sum: f64 = t_data.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let t_n = (tw * th) as f64;
    let t_mean = t_sum / t_n;
    let t_var = t_sq_sum / t_n - t_mean * t_mean;
    let t_std = t_var.max(0.0).sqrt();

    // Build integral images for the source image
    let sat = IntegralImage::from_gray_f32(img);
    // Integral of squared values
    let sq_data: Vec<f32> = img.as_slice().iter().map(|&v| v * v).collect();
    let sq_img = ImageBuf::<f32, Gray>::from_vec(sq_data, img.width(), img.height())?;
    let sat_sq = IntegralImage::from_gray_f32(&sq_img);

    let img_data = img.as_slice();
    let mut out = vec![0.0f32; ow * oh];

    for oy in 0..oh {
        for ox in 0..ow {
            let x0 = ox as u32;
            let y0 = oy as u32;
            let x1 = (ox + tw - 1) as u32;
            let y1 = (oy + th - 1) as u32;

            let patch_sum = sat.rect_sum(x0, y0, x1, y1);
            let patch_sq_sum = sat_sq.rect_sum(x0, y0, x1, y1);
            let patch_mean = patch_sum / t_n;
            let patch_var = patch_sq_sum / t_n - patch_mean * patch_mean;
            let patch_std = patch_var.max(0.0).sqrt();

            let score = match method {
                TemplateMatchMethod::Ssd => {
                    // SSD = Σ(I - T)² = Σ I² - 2Σ IT + Σ T²
                    let cross = cross_sum(img_data, t_data, iw, tw, th, ox, oy);
                    (patch_sq_sum - 2.0 * cross + t_sq_sum) as f32
                }
                TemplateMatchMethod::SsdNormed => {
                    let cross = cross_sum(img_data, t_data, iw, tw, th, ox, oy);
                    let ssd = patch_sq_sum - 2.0 * cross + t_sq_sum;
                    let denom = patch_sq_sum.sqrt() * t_sq_sum.sqrt();
                    if denom > 1e-12 {
                        (ssd / denom) as f32
                    } else {
                        0.0
                    }
                }
                TemplateMatchMethod::CCorr => {
                    // Zero-mean cross-correlation: Σ(I - mean_I)(T - mean_T)
                    let cross = cross_sum(img_data, t_data, iw, tw, th, ox, oy);
                    (cross - t_n * patch_mean * t_mean) as f32
                }
                TemplateMatchMethod::CCorrNormed => {
                    // Normalized: CCorr / (std_I * std_T * N)
                    let cross = cross_sum(img_data, t_data, iw, tw, th, ox, oy);
                    let ccorr = cross - t_n * patch_mean * t_mean;
                    let denom = patch_std * t_std * t_n;
                    if denom > 1e-12 {
                        (ccorr / denom) as f32
                    } else {
                        0.0
                    }
                }
            };

            out[oy * ow + ox] = score;
        }
    }

    ImageBuf::from_vec(out, ow as u32, oh as u32)
}

/// Compute cross-sum Σ I(x+dx, y+dy) * T(dx, dy) for a patch at (ox, oy).
fn cross_sum(
    img: &[f32],
    tmpl: &[f32],
    iw: usize,
    tw: usize,
    th: usize,
    ox: usize,
    oy: usize,
) -> f64 {
    let mut sum = 0.0f64;
    for ty in 0..th {
        let img_row = (oy + ty) * iw + ox;
        let tmpl_row = ty * tw;
        for tx in 0..tw {
            sum += img[img_row + tx] as f64 * tmpl[tmpl_row + tx] as f64;
        }
    }
    sum
}

/// Find the location with the best match in a score map.
///
/// For SSD methods (lower = better), returns the minimum location.
/// For correlation methods (higher = better), returns the maximum location.
///
/// Returns `(x, y, score)`.
pub fn find_best_match(
    score_map: &ImageBuf<f32, Gray>,
    method: TemplateMatchMethod,
) -> (u32, u32, f32) {
    let data = score_map.as_slice();
    let w = score_map.width();
    let minimize = matches!(
        method,
        TemplateMatchMethod::Ssd | TemplateMatchMethod::SsdNormed
    );

    let mut best_idx = 0usize;
    let mut best_val = data[0];
    for (i, &v) in data.iter().enumerate().skip(1) {
        if (minimize && v < best_val) || (!minimize && v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    let x = (best_idx % w as usize) as u32;
    let y = (best_idx / w as usize) as u32;
    (x, y, best_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_match_ssd() {
        // Place a known pattern at (5, 5) in a larger image
        let (iw, ih) = (32u32, 32u32);
        let mut img_data = vec![0.1f32; (iw * ih) as usize];
        let (tw, th) = (6u32, 6u32);
        let tmpl_data: Vec<f32> = (0..tw * th).map(|i| (i as f32 + 1.0) / 36.0).collect();
        // Stamp template at (5, 5)
        for ty in 0..th as usize {
            for tx in 0..tw as usize {
                img_data[(5 + ty) * iw as usize + (5 + tx)] = tmpl_data[ty * tw as usize + tx];
            }
        }

        let img = ImageBuf::<f32, Gray>::from_vec(img_data, iw, ih).unwrap();
        let tmpl = ImageBuf::<f32, Gray>::from_vec(tmpl_data, tw, th).unwrap();

        let score = match_template(&img, &tmpl, TemplateMatchMethod::Ssd).unwrap();
        let (bx, by, bv) = find_best_match(&score, TemplateMatchMethod::Ssd);
        assert_eq!((bx, by), (5, 5), "best match should be at stamp location");
        assert!(bv < 1e-6, "SSD at exact match should be ~0, got {bv}");
    }

    #[test]
    fn self_match_ncc() {
        let (iw, ih) = (32u32, 32u32);
        let mut img_data = vec![0.0f32; (iw * ih) as usize];
        let (tw, th) = (6u32, 6u32);
        let tmpl_data: Vec<f32> = (0..tw * th).map(|i| (i as f32 + 1.0) / 36.0).collect();
        for ty in 0..th as usize {
            for tx in 0..tw as usize {
                img_data[(5 + ty) * iw as usize + (5 + tx)] = tmpl_data[ty * tw as usize + tx];
            }
        }

        let img = ImageBuf::<f32, Gray>::from_vec(img_data, iw, ih).unwrap();
        let tmpl = ImageBuf::<f32, Gray>::from_vec(tmpl_data, tw, th).unwrap();

        let score = match_template(&img, &tmpl, TemplateMatchMethod::CCorrNormed).unwrap();
        let (bx, by, bv) = find_best_match(&score, TemplateMatchMethod::CCorrNormed);
        assert_eq!((bx, by), (5, 5));
        assert!(
            bv > 0.99,
            "NCC at exact match should be ~1.0, got {bv}"
        );
    }

    #[test]
    fn template_larger_than_image_errors() {
        let img = ImageBuf::<f32, Gray>::new(8, 8).unwrap();
        let tmpl = ImageBuf::<f32, Gray>::new(16, 16).unwrap();
        assert!(match_template(&img, &tmpl, TemplateMatchMethod::Ssd).is_err());
    }

    #[test]
    fn output_dimensions() {
        let img = ImageBuf::<f32, Gray>::new(20, 15).unwrap();
        let tmpl = ImageBuf::<f32, Gray>::new(5, 3).unwrap();
        let score = match_template(&img, &tmpl, TemplateMatchMethod::Ssd).unwrap();
        assert_eq!(score.width(), 16); // 20 - 5 + 1
        assert_eq!(score.height(), 13); // 15 - 3 + 1
    }
}
