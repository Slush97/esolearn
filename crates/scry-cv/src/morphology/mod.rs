// SPDX-License-Identifier: MIT OR Apache-2.0
//! Morphological operations: erode, dilate, open, close, gradient, top-hat.

pub mod basic;

pub use basic::{close, dilate, erode, open};

/// Structuring element shape.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum StructuringElement {
    /// Rectangular (filled square/rectangle).
    Rect,
    /// Cross-shaped (diamond).
    Cross,
    /// Elliptical (approximated).
    Ellipse,
}

/// Generate a structuring element as a flat boolean mask.
///
/// Returns a `size x size` mask (must be odd) where `true` means "include".
pub fn make_kernel(shape: StructuringElement, size: u32) -> Vec<bool> {
    let n = size as usize;
    let r = n / 2;
    let mut mask = vec![false; n * n];

    for y in 0..n {
        for x in 0..n {
            let include = match shape {
                StructuringElement::Rect => true,
                StructuringElement::Cross => x == r || y == r,
                StructuringElement::Ellipse => {
                    let dx = x as f32 - r as f32;
                    let dy = y as f32 - r as f32;
                    (dx * dx + dy * dy) <= (r as f32 + 0.5) * (r as f32 + 0.5)
                }
            };
            mask[y * n + x] = include;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_kernel_all_true() {
        let k = make_kernel(StructuringElement::Rect, 3);
        assert_eq!(k.len(), 9);
        assert!(k.iter().all(|&v| v));
    }

    #[test]
    fn cross_kernel_shape() {
        let k = make_kernel(StructuringElement::Cross, 3);
        // Cross pattern:
        // .#.
        // ###
        // .#.
        let expected = [false, true, false, true, true, true, false, true, false];
        assert_eq!(k, expected);
    }

    #[test]
    fn ellipse_kernel_center_included() {
        let k = make_kernel(StructuringElement::Ellipse, 5);
        assert!(k[12]); // center of 5x5
    }
}
