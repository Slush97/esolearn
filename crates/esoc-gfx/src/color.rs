// SPDX-License-Identifier: MIT OR Apache-2.0
//! Color representation with RGBA channels and hex parsing.

use crate::error::{GfxError, Result};

/// An RGBA color with f64 channels in `[0.0, 1.0]`.
///
/// **Deprecated:** This type uses sRGB f64 values. Prefer [`esoc_color::Color`]
/// which uses linear f32 RGBA (GPU-native). Use `esoc_color::Color::from(legacy_color)`
/// to convert.
#[deprecated(
    note = "Use esoc_color::Color instead — this type uses sRGB f64, esoc_color uses linear f32"
)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red channel.
    pub r: f64,
    /// Green channel.
    pub g: f64,
    /// Blue channel.
    pub b: f64,
    /// Alpha channel (1.0 = fully opaque).
    pub a: f64,
}

#[allow(deprecated)]
impl Color {
    /// Fully transparent.
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
    /// Gray (50%).
    pub const GRAY: Self = Self {
        r: 0.5,
        g: 0.5,
        b: 0.5,
        a: 1.0,
    };
    /// Light gray.
    pub const LIGHT_GRAY: Self = Self {
        r: 0.83,
        g: 0.83,
        b: 0.83,
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

    /// Create a new color from RGBA channels in `[0.0, 1.0]`.
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque color from RGB channels in `[0.0, 1.0]`.
    pub fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Create a color from 8-bit RGB values.
    pub fn from_rgb8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: f64::from(r) / 255.0,
            g: f64::from(g) / 255.0,
            b: f64::from(b) / 255.0,
            a: 1.0,
        }
    }

    /// Parse a hex color string (`#RRGGBB` or `#RRGGBBAA`).
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.strip_prefix('#').unwrap_or(hex);
        let parse_byte = |s: &str| {
            u8::from_str_radix(s, 16)
                .map_err(|_| GfxError::InvalidColor(format!("invalid hex byte: {s}")))
        };

        match hex.len() {
            6 => {
                let r = parse_byte(&hex[0..2])?;
                let g = parse_byte(&hex[2..4])?;
                let b = parse_byte(&hex[4..6])?;
                Ok(Self::from_rgb8(r, g, b))
            }
            8 => {
                let r = parse_byte(&hex[0..2])?;
                let g = parse_byte(&hex[2..4])?;
                let b = parse_byte(&hex[4..6])?;
                let a = parse_byte(&hex[6..8])?;
                Ok(Self::new(
                    f64::from(r) / 255.0,
                    f64::from(g) / 255.0,
                    f64::from(b) / 255.0,
                    f64::from(a) / 255.0,
                ))
            }
            _ => Err(GfxError::InvalidColor(format!(
                "expected 6 or 8 hex digits, got {}",
                hex.len()
            ))),
        }
    }

    /// Linearly interpolate between two colors.
    pub fn lerp(self, other: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Return this color with a new alpha value.
    pub fn with_alpha(mut self, a: f64) -> Self {
        self.a = a;
        self
    }

    /// Format as an SVG color string (`rgb(R,G,B)` or `rgba(R,G,B,A)`).
    pub fn to_svg_string(self) -> String {
        let r = (self.r * 255.0).round() as u8;
        let g = (self.g * 255.0).round() as u8;
        let b = (self.b * 255.0).round() as u8;
        if (self.a - 1.0).abs() < 1e-6 {
            format!("rgb({r},{g},{b})")
        } else {
            format!("rgba({r},{g},{b},{:.3})", self.a)
        }
    }

    /// Format as a hex string (`#RRGGBB`).
    pub fn to_hex(self) -> String {
        let r = (self.r * 255.0).round() as u8;
        let g = (self.g * 255.0).round() as u8;
        let b = (self.b * 255.0).round() as u8;
        format!("#{r:02x}{g:02x}{b:02x}")
    }
}

#[allow(deprecated)]
impl From<Color> for esoc_color::Color {
    /// Convert from legacy sRGB f64 Color to linear f32 Color.
    ///
    /// Applies sRGB→linear conversion on each channel.
    fn from(c: Color) -> Self {
        fn srgb_to_linear(s: f64) -> f32 {
            let v = if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
            v as f32
        }
        Self::new(
            srgb_to_linear(c.r),
            srgb_to_linear(c.g),
            srgb_to_linear(c.b),
            c.a as f32,
        )
    }
}

#[allow(deprecated)]
impl From<esoc_color::Color> for Color {
    /// Convert from new linear f32 Color back to legacy sRGB f64 Color.
    ///
    /// Applies linear→sRGB conversion on each channel.
    fn from(c: esoc_color::Color) -> Self {
        fn linear_to_srgb(l: f32) -> f64 {
            let v = if l <= 0.003_130_8 {
                l * 12.92
            } else {
                1.055 * l.powf(1.0 / 2.4) - 0.055
            };
            f64::from(v)
        }
        Self {
            r: linear_to_srgb(c.r),
            g: linear_to_srgb(c.g),
            b: linear_to_srgb(c.b),
            a: f64::from(c.a),
        }
    }
}

#[allow(deprecated)]
impl Default for Color {
    fn default() -> Self {
        Self::BLACK
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_parsing() {
        let c = Color::from_hex("#1f77b4").unwrap();
        assert_eq!((c.r * 255.0).round() as u8, 0x1f);
        assert_eq!((c.g * 255.0).round() as u8, 0x77);
        assert_eq!((c.b * 255.0).round() as u8, 0xb4);
        assert!((c.a - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hex_parsing_with_alpha() {
        let c = Color::from_hex("#ff000080").unwrap();
        assert!((c.r - 1.0).abs() < 0.01);
        assert!((c.a - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn test_hex_invalid() {
        assert!(Color::from_hex("#gg0000").is_err());
        assert!(Color::from_hex("#123").is_err());
    }

    #[test]
    fn test_lerp() {
        let a = Color::BLACK;
        let b = Color::WHITE;
        let mid = a.lerp(b, 0.5);
        assert!((mid.r - 0.5).abs() < 1e-6);
        assert!((mid.g - 0.5).abs() < 1e-6);
        assert!((mid.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_to_svg_string() {
        assert_eq!(Color::RED.to_svg_string(), "rgb(255,0,0)");
        let semi = Color::RED.with_alpha(0.5);
        assert_eq!(semi.to_svg_string(), "rgba(255,0,0,0.500)");
    }

    #[test]
    fn test_to_hex() {
        assert_eq!(Color::RED.to_hex(), "#ff0000");
    }
}
