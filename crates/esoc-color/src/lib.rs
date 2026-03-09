// SPDX-License-Identifier: MIT OR Apache-2.0
//! OKLab/OKLCH color system for data visualization.
//!
//! Core type is [`Color`] — f32 linear RGBA, GPU-native. All color math
//! operates in `OKLab` perceptual space. Zero external dependencies.

pub mod contrast;
pub mod cvd;
pub mod gamut;
pub mod oklab;
pub mod palette;
pub mod scale;
pub mod srgb;

mod color;
pub use color::Color;
pub use oklab::{OkLab, OkLch};
pub use palette::Palette;
pub use scale::ColorScale;
