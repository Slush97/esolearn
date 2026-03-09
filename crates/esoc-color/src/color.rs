// SPDX-License-Identifier: MIT OR Apache-2.0
//! Core color type: f32 linear RGBA.

use crate::oklab::{OkLab, OkLch};
use crate::srgb;

/// An RGBA color with f32 channels in linear space.
///
/// This is the core color type, designed for GPU upload (`#[repr(C)]`).
/// Channels are in `[0.0, 1.0]` linear light (not sRGB gamma-encoded).
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Color {
    /// Red channel (linear).
    pub r: f32,
    /// Green channel (linear).
    pub g: f32,
    /// Blue channel (linear).
    pub b: f32,
    /// Alpha channel (1.0 = fully opaque).
    pub a: f32,
}

impl Color {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    /// Black.
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    /// White.
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    /// 50% gray (linear).
    pub const GRAY: Self = Self {
        r: 0.5,
        g: 0.5,
        b: 0.5,
        a: 1.0,
    };
    /// Red.
    pub const RED: Self = Self {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    /// Green.
    pub const GREEN: Self = Self {
        r: 0.0,
        g: 0.5,
        b: 0.0,
        a: 1.0,
    };
    /// Blue.
    pub const BLUE: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };

    /// Create from linear RGBA channels.
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque color from linear RGB.
    pub fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Create from sRGB 8-bit values (gamma-decoded to linear).
    pub fn from_srgb8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: srgb::decode(f32::from(r) / 255.0),
            g: srgb::decode(f32::from(g) / 255.0),
            b: srgb::decode(f32::from(b) / 255.0),
            a: 1.0,
        }
    }

    /// Create from an sRGB hex string (`#RRGGBB` or `#RRGGBBAA`).
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix('#').unwrap_or(hex);
        let parse_byte = |s: &str| u8::from_str_radix(s, 16).ok();

        match hex.len() {
            6 => {
                let r = parse_byte(&hex[0..2])?;
                let g = parse_byte(&hex[2..4])?;
                let b = parse_byte(&hex[4..6])?;
                Some(Self::from_srgb8(r, g, b))
            }
            8 => {
                let r = parse_byte(&hex[0..2])?;
                let g = parse_byte(&hex[2..4])?;
                let b = parse_byte(&hex[4..6])?;
                let a = parse_byte(&hex[6..8])?;
                let mut c = Self::from_srgb8(r, g, b);
                c.a = f32::from(a) / 255.0;
                Some(c)
            }
            _ => None,
        }
    }

    /// Return this color with a new alpha value.
    pub fn with_alpha(mut self, a: f32) -> Self {
        self.a = a;
        self
    }

    /// Linearly interpolate in linear RGB space.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Interpolate in `OKLab` perceptual space (better for gradients).
    pub fn lerp_oklab(self, other: Self, t: f32) -> Self {
        let a = OkLab::from_linear_rgb(self);
        let b = OkLab::from_linear_rgb(other);
        let mixed = a.lerp(b, t);
        let mut c = mixed.to_linear_rgb();
        c.a = self.a + (other.a - self.a) * t.clamp(0.0, 1.0);
        c
    }

    /// Convert to sRGB 8-bit (gamma-encoded).
    pub fn to_srgb8(self) -> [u8; 4] {
        [
            (srgb::encode(self.r) * 255.0 + 0.5) as u8,
            (srgb::encode(self.g) * 255.0 + 0.5) as u8,
            (srgb::encode(self.b) * 255.0 + 0.5) as u8,
            (self.a * 255.0 + 0.5) as u8,
        ]
    }

    /// Format as sRGB hex string (`#RRGGBB`).
    pub fn to_hex(self) -> String {
        let [r, g, b, _] = self.to_srgb8();
        format!("#{r:02x}{g:02x}{b:02x}")
    }

    /// Format as an SVG color string.
    pub fn to_svg_string(self) -> String {
        let [r, g, b, _] = self.to_srgb8();
        if (self.a - 1.0).abs() < 1e-6 {
            format!("rgb({r},{g},{b})")
        } else {
            format!("rgba({r},{g},{b},{:.3})", self.a)
        }
    }

    /// Convert to `OKLab`.
    pub fn to_oklab(self) -> OkLab {
        OkLab::from_linear_rgb(self)
    }

    /// Convert to OKLCH.
    pub fn to_oklch(self) -> OkLch {
        self.to_oklab().to_oklch()
    }

    /// Create from `OKLab`.
    pub fn from_oklab(lab: OkLab) -> Self {
        lab.to_linear_rgb()
    }

    /// Create from OKLCH.
    pub fn from_oklch(lch: OkLch) -> Self {
        lch.to_oklab().to_linear_rgb()
    }

    /// Raw f32x4 array for GPU upload.
    pub fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::BLACK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        let c = Color::from_hex("#1f77b4").unwrap();
        let hex = c.to_hex();
        assert_eq!(hex, "#1f77b4");
    }

    #[test]
    fn hex_with_alpha() {
        let c = Color::from_hex("#ff000080").unwrap();
        assert!((c.a - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn hex_invalid() {
        assert!(Color::from_hex("#gg0000").is_none());
        assert!(Color::from_hex("#123").is_none());
    }

    #[test]
    fn lerp_midpoint() {
        let mid = Color::BLACK.lerp(Color::WHITE, 0.5);
        assert!((mid.r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn svg_string() {
        let c = Color::from_hex("#ff0000").unwrap();
        assert_eq!(c.to_svg_string(), "rgb(255,0,0)");
    }

    #[test]
    fn oklab_roundtrip() {
        let c = Color::from_hex("#1f77b4").unwrap();
        let lab = c.to_oklab();
        let back = Color::from_oklab(lab);
        assert!((c.r - back.r).abs() < 1e-4);
        assert!((c.g - back.g).abs() < 1e-4);
        assert!((c.b - back.b).abs() < 1e-4);
    }
}
