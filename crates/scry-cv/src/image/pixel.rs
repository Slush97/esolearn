// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pixel scalar traits and channel layout types.

use std::fmt;

// ── Sealed trait ──

mod sealed {
    pub trait Sealed {}
}

// ── Channel layout ──

/// Describes the number and semantics of channels in an image.
pub trait ChannelLayout: sealed::Sealed + Copy + Clone + fmt::Debug + Send + Sync + 'static {
    /// Number of channels (1, 3, or 4).
    const CHANNELS: usize;
    /// Human-readable name for error messages.
    const NAME: &'static str;
}

/// Single-channel grayscale.
#[derive(Clone, Copy, Debug)]
pub struct Gray;

impl sealed::Sealed for Gray {}
impl ChannelLayout for Gray {
    const CHANNELS: usize = 1;
    const NAME: &'static str = "Gray";
}

/// Three-channel RGB.
#[derive(Clone, Copy, Debug)]
pub struct Rgb;

impl sealed::Sealed for Rgb {}
impl ChannelLayout for Rgb {
    const CHANNELS: usize = 3;
    const NAME: &'static str = "Rgb";
}

/// Four-channel RGBA.
#[derive(Clone, Copy, Debug)]
pub struct Rgba;

impl sealed::Sealed for Rgba {}
impl ChannelLayout for Rgba {
    const CHANNELS: usize = 4;
    const NAME: &'static str = "Rgba";
}

// ── Pixel scalar trait ──

/// Scalar type for pixel values.
///
/// Implemented for `u8`, `u16`, `f32`, and `f64`.
pub trait Pixel: Copy + Default + Send + Sync + fmt::Debug + 'static {
    /// The zero value.
    const ZERO: Self;
    /// The maximum representable value (255 for u8, 1.0 for floats).
    const MAX: Self;
    /// Convert to f32 normalized to \[0, 1\].
    fn to_f32(self) -> f32;
    /// Convert from f32 in \[0, 1\] (clamped).
    fn from_f32(v: f32) -> Self;
}

impl Pixel for u8 {
    const ZERO: Self = 0;
    const MAX: Self = 255;

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32 / 255.0
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as Self
    }
}

impl Pixel for u16 {
    const ZERO: Self = 0;
    const MAX: Self = 65535;

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32 / 65535.0
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as Self
    }
}

impl Pixel for f32 {
    const ZERO: Self = 0.0;
    const MAX: Self = 1.0;

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl Pixel for f64 {
    const ZERO: Self = 0.0;
    const MAX: Self = 1.0;

    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_round_trip() {
        for v in 0..=255u8 {
            let f = v.to_f32();
            assert!((0.0..=1.0).contains(&f));
            let back = u8::from_f32(f);
            assert_eq!(v, back, "round-trip failed for {v}");
        }
    }

    #[test]
    fn f32_identity() {
        let vals = [0.0f32, 0.25, 0.5, 0.75, 1.0];
        for &v in &vals {
            assert!((v.to_f32() - v).abs() < f32::EPSILON);
            assert!((f32::from_f32(v) - v).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn channel_counts() {
        assert_eq!(Gray::CHANNELS, 1);
        assert_eq!(Rgb::CHANNELS, 3);
        assert_eq!(Rgba::CHANNELS, 4);
    }
}
