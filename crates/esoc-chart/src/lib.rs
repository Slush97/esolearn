// SPDX-License-Identifier: MIT OR Apache-2.0
//! # esoc-chart
//!
//! High-level charting API for data visualization, built on `esoc-gfx`.
//! Provides matplotlib-equivalent plotting for Rust ML workflows.
//!
//! ## API Levels
//!
//! - **Express** ([`express`]): one-liner functions (`scatter`, `line`, `bar`, …)
//! - **Grammar** ([`grammar`]): composable [`grammar::chart::Chart`] / [`grammar::layer::Layer`] /
//!   [`grammar::encoding::Encoding`] types
//! - **Scene Graph**: direct access via the `esoc-scene` crate

#![warn(missing_docs)]
#![deny(unsafe_code)]
#![allow(clippy::doc_markdown)]

pub mod axis;
pub mod compile;
pub mod error;
pub mod express;
pub mod grammar;
pub mod new_theme;

#[cfg(feature = "scry-learn")]
pub mod interop;

/// Convenience re-exports for the grammar-of-graphics API.
pub mod prelude {
    pub use crate::error::{ChartError, Result};
    pub use crate::express::{
        area, bar, boxplot, grouped_bar, heatmap, heatmap_ref, histogram, line, pie_labeled,
        scatter, stacked_bar, treemap,
    };
    pub use crate::grammar::annotation::Annotation;
    pub use crate::grammar::chart::Chart;
    pub use crate::grammar::coord::CoordSystem;
    pub use crate::grammar::facet::{Facet, FacetScales};
    pub use crate::grammar::layer::{Layer, MarkType};
    pub use crate::grammar::position::Position;
    pub use crate::grammar::stat::Stat;
    pub use crate::new_theme::{NewTheme, Theme};

    pub use esoc_color::{Color, OkLab, OkLch, Palette};
    pub use esoc_scene::SceneGraph;
}

