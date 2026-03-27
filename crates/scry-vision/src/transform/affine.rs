// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2D affine transform for image warping.
//!
//! Primary use case: face alignment via landmark-based similarity transforms
//! (ArcFace, CosFace). Given source landmarks and a canonical target template,
//! computes the 2×3 affine matrix and warps the image.

use crate::error::Result;
use crate::image::ImageBuffer;
use crate::transform::ImageTransform;

/// A 2D affine transform represented as a 2×3 matrix.
///
/// The transformation maps `(x, y)` to:
/// ```text
/// x' = m[0][0]*x + m[0][1]*y + m[0][2]
/// y' = m[1][0]*x + m[1][1]*y + m[1][2]
/// ```
#[derive(Clone, Debug)]
pub struct AffineTransform {
    /// 2×3 affine matrix (row-major): `[[a, b, tx], [c, d, ty]]`.
    pub matrix: [[f32; 3]; 2],
    /// Output width.
    pub out_width: u32,
    /// Output height.
    pub out_height: u32,
    /// Fill value for pixels outside the source image.
    pub fill: u8,
}

impl AffineTransform {
    /// Create an affine transform from a 2×3 forward matrix.
    ///
    /// The matrix maps *source* coordinates to *destination* coordinates.
    /// Internally, the inverse is used for backward warping.
    #[must_use]
    pub fn new(matrix: [[f32; 3]; 2], out_width: u32, out_height: u32) -> Self {
        Self {
            matrix,
            out_width,
            out_height,
            fill: 0,
        }
    }

    /// Estimate a similarity transform (rotation + uniform scale + translation)
    /// from corresponding 2D point pairs using least-squares.
    ///
    /// `src` and `dst` are slices of `[x, y]` pairs. At least 2 pairs are needed.
    ///
    /// The resulting matrix maps src → dst. When used for face alignment,
    /// `src` = detected landmarks, `dst` = canonical template landmarks (in
    /// output image coordinates).
    #[must_use]
    pub fn estimate_similarity(src: &[[f32; 2]], dst: &[[f32; 2]], out_width: u32, out_height: u32) -> Self {
        assert!(src.len() >= 2 && src.len() == dst.len(), "need >= 2 matching point pairs");
        let n = src.len() as f32;

        // Least-squares for similarity: [a, b, tx, ty]
        // dst_x = a * src_x - b * src_y + tx
        // dst_y = b * src_x + a * src_y + ty
        let mut sx = 0.0f32;
        let mut sy = 0.0f32;
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        for i in 0..src.len() {
            sx += src[i][0];
            sy += src[i][1];
            dx += dst[i][0];
            dy += dst[i][1];
        }
        let mean_sx = sx / n;
        let mean_sy = sy / n;
        let mean_dx = dx / n;
        let mean_dy = dy / n;

        let mut num_a = 0.0f32;
        let mut num_b = 0.0f32;
        let mut denom = 0.0f32;

        for i in 0..src.len() {
            let sxc = src[i][0] - mean_sx;
            let syc = src[i][1] - mean_sy;
            let dxc = dst[i][0] - mean_dx;
            let dyc = dst[i][1] - mean_dy;

            num_a += sxc * dxc + syc * dyc;
            num_b += sxc * dyc - syc * dxc;
            denom += sxc * sxc + syc * syc;
        }

        let a = num_a / denom;
        let b = num_b / denom;
        let tx = mean_dx - a * mean_sx + b * mean_sy;
        let ty = mean_dy - b * mean_sx - a * mean_sy;

        Self::new([[a, -b, tx], [b, a, ty]], out_width, out_height)
    }

    /// Invert the 2×3 affine matrix.
    ///
    /// For the forward matrix `[[a, b, tx], [c, d, ty]]`, the inverse maps
    /// destination coordinates back to source coordinates.
    fn inverse(&self) -> [[f32; 3]; 2] {
        let [[a, b, tx], [c, d, ty]] = self.matrix;
        let det = a * d - b * c;
        let inv_det = 1.0 / det;
        let ia = d * inv_det;
        let ib = -b * inv_det;
        let ic = -c * inv_det;
        let id = a * inv_det;
        let itx = -(ia * tx + ib * ty);
        let ity = -(ic * tx + id * ty);
        [[ia, ib, itx], [ic, id, ity]]
    }
}

impl ImageTransform for AffineTransform {
    fn apply(&self, image: &ImageBuffer) -> Result<ImageBuffer> {
        let ch = image.channels as usize;
        let src_w = image.width as i64;
        let src_h = image.height as i64;
        let mut data = vec![self.fill; self.out_width as usize * self.out_height as usize * ch];

        let inv = self.inverse();

        for dy in 0..self.out_height {
            for dx in 0..self.out_width {
                let fxd = dx as f32;
                let fyd = dy as f32;
                // Map destination → source
                let sx = inv[0][0] * fxd + inv[0][1] * fyd + inv[0][2];
                let sy = inv[1][0] * fxd + inv[1][1] * fyd + inv[1][2];

                // Bilinear interpolation
                let x0 = sx.floor() as i64;
                let y0 = sy.floor() as i64;
                let x1 = x0 + 1;
                let y1 = y0 + 1;

                if x0 < 0 || y0 < 0 || x1 >= src_w || y1 >= src_h {
                    continue; // leave as fill
                }

                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let idx00 = (y0 as usize * image.width as usize + x0 as usize) * ch;
                let idx10 = (y0 as usize * image.width as usize + x1 as usize) * ch;
                let idx01 = (y1 as usize * image.width as usize + x0 as usize) * ch;
                let idx11 = (y1 as usize * image.width as usize + x1 as usize) * ch;
                let dst_idx = (dy as usize * self.out_width as usize + dx as usize) * ch;

                for c in 0..ch {
                    let p00 = image.data[idx00 + c] as f32;
                    let p10 = image.data[idx10 + c] as f32;
                    let p01 = image.data[idx01 + c] as f32;
                    let p11 = image.data[idx11 + c] as f32;
                    let val = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;
                    data[dst_idx + c] = val.round().clamp(0.0, 255.0) as u8;
                }
            }
        }

        ImageBuffer::from_raw(data, self.out_width, self.out_height, image.channels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_transform() {
        let data = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let img = ImageBuffer::from_raw(data, 2, 2, 3).unwrap();
        let t = AffineTransform::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 2, 2);
        let out = t.apply(&img).unwrap();
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 2);
    }

    #[test]
    fn translation_transform() {
        // 4×4 gray image with pixel value = x + y*4
        let mut data = vec![0u8; 16];
        for y in 0..4u32 {
            for x in 0..4u32 {
                data[(y * 4 + x) as usize] = (x + y * 4) as u8;
            }
        }
        let img = ImageBuffer::from_raw(data, 4, 4, 1).unwrap();
        // Translate by (1, 1)
        let t = AffineTransform::new([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], 4, 4);
        let out = t.apply(&img).unwrap();
        // Pixel at (1,1) in output should come from (0,0) in source = 0
        assert_eq!(out.pixel(1, 1, 0), Some(0));
        // Pixel at (2,2) in output should come from (1,1) in source = 5
        assert_eq!(out.pixel(2, 2, 0), Some(5));
    }

    #[test]
    fn estimate_similarity_identity() {
        let src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let dst = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let t = AffineTransform::estimate_similarity(&src, &dst, 2, 2);
        // Should be approximately identity
        assert!((t.matrix[0][0] - 1.0).abs() < 1e-4);
        assert!((t.matrix[0][1]).abs() < 1e-4);
        assert!((t.matrix[0][2]).abs() < 1e-4);
        assert!((t.matrix[1][0]).abs() < 1e-4);
        assert!((t.matrix[1][1] - 1.0).abs() < 1e-4);
        assert!((t.matrix[1][2]).abs() < 1e-4);
    }

    #[test]
    fn estimate_similarity_scale() {
        // Scale by 2x
        let src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let dst = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let t = AffineTransform::estimate_similarity(&src, &dst, 4, 4);
        assert!((t.matrix[0][0] - 2.0).abs() < 1e-4);
        assert!((t.matrix[1][1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn out_of_bounds_fills() {
        let data = vec![128u8; 4]; // 2×2 gray
        let img = ImageBuffer::from_raw(data, 2, 2, 1).unwrap();
        // Large translation — everything should be fill (0)
        let mut t = AffineTransform::new([[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]], 2, 2);
        t.fill = 42;
        let out = t.apply(&img).unwrap();
        assert!(out.data.iter().all(|&v| v == 42));
    }
}
