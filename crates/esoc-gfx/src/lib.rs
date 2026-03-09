// SPDX-License-Identifier: MIT OR Apache-2.0
//! # esoc-gfx
//!
//! Low-level 2D vector graphics engine for esoc-chart.
//! SVG output with zero dependencies; optional PNG via resvg.

#![warn(missing_docs)]
#![deny(unsafe_code)]
#![allow(clippy::format_push_string)]

#[allow(deprecated)]
pub mod backend;
pub mod canvas;
pub mod color;
#[allow(deprecated)]
pub mod element;
pub mod error;
pub mod geom;
pub mod layer;
#[allow(deprecated)]
pub mod palette;
pub mod path;
pub mod scene_svg;
#[allow(deprecated)]
pub mod style;
pub mod text;
pub mod transform;

/// Convenience re-exports.
pub mod prelude {
    pub use crate::backend::svg::{render_svg, save_svg, SvgBackend};
    pub use crate::backend::RenderBackend;
    pub use crate::canvas::Canvas;
    #[allow(deprecated)]
    pub use crate::color::Color;
    pub use crate::element::{DrawElement, Element};
    pub use crate::error::{GfxError, Result};
    pub use crate::geom::{Point, Rect, Size};
    pub use crate::layer::Layer;
    pub use crate::palette::Palette;
    pub use crate::path::PathBuilder;
    pub use crate::style::{DashPattern, Fill, FontStyle, Stroke, TextAnchor};
    pub use crate::text::{HeuristicTextMeasurer, TextMeasurer};
    pub use crate::transform::{AxisTransform, CoordinateTransform, ViewportTransform};

    #[cfg(feature = "png")]
    pub use crate::backend::png::{save_png, PngBackend};
}
