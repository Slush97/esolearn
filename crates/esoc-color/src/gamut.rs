// SPDX-License-Identifier: MIT OR Apache-2.0
//! OKLCH → sRGB gamut clipping via chroma bisection.

use crate::oklab::OkLch;
use crate::Color;

/// Check if a linear RGB color is within the sRGB gamut.
pub fn in_gamut(c: Color) -> bool {
    let eps = -1e-4;
    let max = 1.0 + 1e-4;
    c.r >= eps && c.r <= max && c.g >= eps && c.g <= max && c.b >= eps && c.b <= max
}

/// Clamp a linear RGB color to `[0, 1]`.
pub fn clamp(c: Color) -> Color {
    Color::new(
        c.r.clamp(0.0, 1.0),
        c.g.clamp(0.0, 1.0),
        c.b.clamp(0.0, 1.0),
        c.a.clamp(0.0, 1.0),
    )
}

/// Map an OKLCH color to the nearest in-gamut sRGB color by bisecting chroma.
///
/// Preserves lightness and hue, reduces chroma until the color fits in sRGB.
pub fn gamut_clip_oklch(lch: OkLch) -> Color {
    let rgb = lch.to_linear_rgb();
    if in_gamut(rgb) {
        return clamp(rgb);
    }

    // Bisect chroma between 0 and original
    let mut lo = 0.0_f32;
    let mut hi = lch.c;

    for _ in 0..24 {
        let mid = (lo + hi) * 0.5;
        let test = OkLch::new(lch.l, mid, lch.h).to_linear_rgb();
        if in_gamut(test) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    clamp(OkLch::new(lch.l, lo, lch.h).to_linear_rgb())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_gamut_colors() {
        assert!(in_gamut(Color::BLACK));
        assert!(in_gamut(Color::WHITE));
        assert!(in_gamut(Color::RED));
    }

    #[test]
    fn out_of_gamut_clipped() {
        // A very saturated OKLCH that's out of sRGB
        let lch = OkLch::new(0.7, 0.4, 150.0);
        let clipped = gamut_clip_oklch(lch);
        assert!(in_gamut(clipped));
        // Lightness preserved approximately
        let clipped_lch = clipped.to_oklch();
        assert!((clipped_lch.l - lch.l).abs() < 0.02);
    }

    #[test]
    fn already_in_gamut() {
        let lch = Color::from_hex("#808080").unwrap().to_oklch();
        let clipped = gamut_clip_oklch(lch);
        let original = lch.to_linear_rgb();
        assert!((clipped.r - original.r).abs() < 1e-3);
    }
}
