// SPDX-License-Identifier: MIT OR Apache-2.0
//! WCAG contrast ratio checking.

use crate::srgb;
use crate::Color;

/// Compute the relative luminance of a linear RGB color (ITU-R BT.709).
pub fn relative_luminance(c: Color) -> f32 {
    0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
}

/// Compute the WCAG 2.x contrast ratio between two colors.
///
/// Returns a value in `[1, 21]`. Higher is better.
pub fn contrast_ratio(a: Color, b: Color) -> f32 {
    let la = relative_luminance(a) + 0.05;
    let lb = relative_luminance(b) + 0.05;
    if la > lb {
        la / lb
    } else {
        lb / la
    }
}

/// Check if two colors meet WCAG AA for normal text (ratio >= 4.5).
pub fn meets_aa(a: Color, b: Color) -> bool {
    contrast_ratio(a, b) >= 4.5
}

/// Check if two colors meet WCAG AAA for normal text (ratio >= 7.0).
pub fn meets_aaa(a: Color, b: Color) -> bool {
    contrast_ratio(a, b) >= 7.0
}

/// Suggest black or white text color for maximum contrast on a background.
pub fn text_color_on(bg: Color) -> Color {
    let lum = relative_luminance(bg);
    // sRGB-encode to get perceptual midpoint
    if srgb::encode(lum) > 0.5 {
        Color::BLACK
    } else {
        Color::WHITE
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn black_white_ratio() {
        let ratio = contrast_ratio(Color::BLACK, Color::WHITE);
        assert!((ratio - 21.0).abs() < 0.1);
    }

    #[test]
    fn same_color_ratio() {
        let ratio = contrast_ratio(Color::RED, Color::RED);
        assert!((ratio - 1.0).abs() < 1e-3);
    }

    #[test]
    fn text_on_dark() {
        let text = text_color_on(Color::BLACK);
        assert_eq!(text.r, Color::WHITE.r);
    }

    #[test]
    fn text_on_light() {
        let text = text_color_on(Color::WHITE);
        assert_eq!(text.r, Color::BLACK.r);
    }

    #[test]
    fn aa_check() {
        assert!(meets_aa(Color::BLACK, Color::WHITE));
        assert!(!meets_aa(Color::GRAY, Color::WHITE));
    }
}
