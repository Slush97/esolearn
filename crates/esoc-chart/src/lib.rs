// SPDX-License-Identifier: MIT OR Apache-2.0
//! # esoc-chart
//!
//! High-level charting API for data visualization, built on esoc-gfx.
//! Provides matplotlib-equivalent plotting for Rust ML workflows.
//!
//! ## New Grammar-of-Graphics API
//!
//! The new API provides 3 levels:
//! - **Express**: one-liner functions (`scatter`, `line`, `bar`)
//! - **Grammar**: composable `Chart`/`Layer`/`Encoding` types
//! - **Scene Graph**: direct access via `esoc-scene`

#![warn(missing_docs)]
#![deny(unsafe_code)]
#![allow(clippy::doc_markdown)]

// ── Legacy API (preserved for backward compatibility) ────────────
#[allow(deprecated)]
pub mod axes;
pub mod axis;
#[allow(deprecated)]
pub mod chart;
pub mod error;
pub mod figure;
#[allow(deprecated)]
pub mod legend;
pub mod render;
pub mod series;
#[allow(deprecated)]
pub mod theme;

#[cfg(feature = "scry-learn")]
#[allow(deprecated)]
pub mod interop;

// ── New Grammar-of-Graphics API ──────────────────────────────────
pub mod compile;
pub mod express;
pub mod grammar;
pub mod new_theme;

/// Legacy re-exports.
#[allow(deprecated)]
pub mod prelude {
    pub use crate::axes::Axes;
    pub use crate::axis::{AxisConfig, AxisPosition, Scale};
    pub use crate::chart::{
        BarSeries, BoxPlotSeries, ErrorBarSeries, HeatmapSeries, HistogramSeries, LineSeries,
        ScatterSeries,
    };
    pub use crate::error::{ChartError, Result};
    pub use crate::figure::Figure;
    pub use crate::legend::LegendPosition;
    pub use crate::series::SeriesRenderer;
    pub use crate::theme::Theme;

    // Re-export key gfx types for convenience
    pub use esoc_gfx::color::Color;
    pub use esoc_gfx::style::{DashPattern, Fill, Stroke};

    #[cfg(feature = "scry-learn")]
    pub use crate::interop::*;
}

/// New API re-exports.
pub mod v2 {
    pub use crate::express::{area, bar, boxplot, grouped_bar, histogram, line, pie, scatter, stacked_bar};
    pub use crate::grammar::annotation::Annotation;
    pub use crate::grammar::chart::Chart;
    pub use crate::grammar::coord::CoordSystem;
    pub use crate::grammar::encoding::{nominal, quantitative};
    pub use crate::grammar::facet::{Facet, FacetScales};
    pub use crate::grammar::layer::{Layer, MarkType};
    pub use crate::grammar::position::Position;
    pub use crate::grammar::stat::Stat;
    pub use crate::new_theme::NewTheme;

    // Re-export scene/color types
    pub use esoc_color::{Color, OkLab, OkLch, Palette};
    pub use esoc_scene::SceneGraph;
}
