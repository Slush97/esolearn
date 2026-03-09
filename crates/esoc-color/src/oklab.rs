// SPDX-License-Identifier: MIT OR Apache-2.0
//! `OKLab` and `OKLCH` color spaces — perceptual color math.
//!
//! Reference: Björn Ottosson, "A perceptual color space for image processing"
//! <https://bottosson.github.io/posts/oklab/>

use crate::Color;

/// A color in `OKLab` perceptual color space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OkLab {
    /// Lightness `[0, 1]`.
    pub l: f32,
    /// Green-red axis.
    pub a: f32,
    /// Blue-yellow axis.
    pub b: f32,
}

/// A color in `OKLCH` (cylindrical `OKLab`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OkLch {
    /// Lightness `[0, 1]`.
    pub l: f32,
    /// Chroma (saturation).
    pub c: f32,
    /// Hue in degrees `[0, 360)`.
    pub h: f32,
}

// Linear RGB → LMS (M1 matrix from Ottosson)
#[inline]
fn linear_rgb_to_lms(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let l = 0.412_221_5 * r + 0.536_332_55 * g + 0.051_445_94 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
    (l, m, s)
}

// LMS^(1/3) → OKLab (M2 matrix)
#[inline]
fn lms_to_oklab(l: f32, m: f32, s: f32) -> (f32, f32, f32) {
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    let lab_l = 0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_;
    let lab_a = 1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_;
    let lab_b = 0.025_904_037 * l_ + 0.782_771_8 * m_ - 0.808_675_77 * s_;
    (lab_l, lab_a, lab_b)
}

// OKLab → LMS^(1/3) (inverse M2)
#[inline]
fn oklab_to_lms_cubed(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let l_ = l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = l - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;

    (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_)
}

// LMS → linear RGB (inverse M1)
#[inline]
fn lms_to_linear_rgb(l: f32, m: f32, s: f32) -> (f32, f32, f32) {
    let r = 4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s;
    let g = -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s;
    let b = -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s;
    (r, g, b)
}

impl OkLab {
    /// Create a new `OKLab` color.
    pub fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    /// Convert from linear RGB.
    pub fn from_linear_rgb(c: Color) -> Self {
        let (l, m, s) = linear_rgb_to_lms(c.r, c.g, c.b);
        let (lab_l, lab_a, lab_b) = lms_to_oklab(l, m, s);
        Self {
            l: lab_l,
            a: lab_a,
            b: lab_b,
        }
    }

    /// Convert to linear RGB.
    pub fn to_linear_rgb(self) -> Color {
        let (l, m, s) = oklab_to_lms_cubed(self.l, self.a, self.b);
        let (r, g, b) = lms_to_linear_rgb(l, m, s);
        Color::new(r, g, b, 1.0)
    }

    /// Convert to OKLCH.
    pub fn to_oklch(self) -> OkLch {
        let c = self.a.hypot(self.b);
        let h = if c < 1e-8 {
            0.0
        } else {
            self.b.atan2(self.a).to_degrees().rem_euclid(360.0)
        };
        OkLch {
            l: self.l,
            c,
            h,
        }
    }

    /// Linearly interpolate in `OKLab` space.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            l: self.l + (other.l - self.l) * t,
            a: self.a + (other.a - self.a) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

impl OkLch {
    /// Create a new OKLCH color.
    pub fn new(l: f32, c: f32, h: f32) -> Self {
        Self { l, c, h }
    }

    /// Convert to `OKLab`.
    pub fn to_oklab(self) -> OkLab {
        let h_rad = self.h.to_radians();
        OkLab {
            l: self.l,
            a: self.c * h_rad.cos(),
            b: self.c * h_rad.sin(),
        }
    }

    /// Convert to linear RGB.
    pub fn to_linear_rgb(self) -> Color {
        self.to_oklab().to_linear_rgb()
    }

    /// Interpolate in OKLCH with shortest-arc hue interpolation.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);

        // Shortest-arc hue interpolation
        let mut dh = other.h - self.h;
        if dh > 180.0 {
            dh -= 360.0;
        } else if dh < -180.0 {
            dh += 360.0;
        }

        Self {
            l: self.l + (other.l - self.l) * t,
            c: self.c + (other.c - self.c) * t,
            h: (self.h + dh * t).rem_euclid(360.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_roundtrip() {
        let lab = OkLab::from_linear_rgb(Color::BLACK);
        assert!(lab.l.abs() < 1e-4);
        let back = lab.to_linear_rgb();
        assert!(back.r.abs() < 1e-4);
    }

    #[test]
    fn white_roundtrip() {
        let lab = OkLab::from_linear_rgb(Color::WHITE);
        assert!((lab.l - 1.0).abs() < 1e-3);
        let back = lab.to_linear_rgb();
        assert!((back.r - 1.0).abs() < 1e-3);
        assert!((back.g - 1.0).abs() < 1e-3);
        assert!((back.b - 1.0).abs() < 1e-3);
    }

    #[test]
    fn oklch_hue_wrapping() {
        let a = OkLch::new(0.7, 0.15, 350.0);
        let b = OkLch::new(0.7, 0.15, 10.0);
        let mid = a.lerp(b, 0.5);
        // Should go through 0° not 180°
        assert!(mid.h < 10.0 || mid.h > 350.0);
    }

    #[test]
    fn color_rgb_roundtrip() {
        let colors = [
            Color::from_hex("#ff0000").unwrap(),
            Color::from_hex("#00ff00").unwrap(),
            Color::from_hex("#0000ff").unwrap(),
            Color::from_hex("#1f77b4").unwrap(),
        ];
        for c in colors {
            let lab = OkLab::from_linear_rgb(c);
            let back = lab.to_linear_rgb();
            assert!(
                (c.r - back.r).abs() < 1e-3,
                "r: {} vs {} for {:?}",
                c.r,
                back.r,
                c
            );
            assert!(
                (c.g - back.g).abs() < 1e-3,
                "g: {} vs {} for {:?}",
                c.g,
                back.g,
                c
            );
            assert!(
                (c.b - back.b).abs() < 1e-3,
                "b: {} vs {} for {:?}",
                c.b,
                back.b,
                c
            );
        }
    }

    #[test]
    fn oklch_roundtrip() {
        let c = Color::from_hex("#1f77b4").unwrap();
        let lch = c.to_oklch();
        let lab = lch.to_oklab();
        let original_lab = c.to_oklab();
        assert!((lab.l - original_lab.l).abs() < 1e-5);
        assert!((lab.a - original_lab.a).abs() < 1e-5);
        assert!((lab.b - original_lab.b).abs() < 1e-5);
    }
}
