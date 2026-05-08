// SPDX-License-Identifier: MIT OR Apache-2.0
//! # esoc-gfx
//!
//! Renders an [`esoc_scene::SceneGraph`] to SVG (and optionally PNG via the
//! `png` feature). Designed as the rendering backend for `esoc-chart`.
//!
//! ## Quick start
//!
//! ```no_run
//! use esoc_gfx::scene_svg::{render_scene_svg, save_scene_svg};
//! # let scene = esoc_scene::SceneGraph::new();
//! let svg: String = render_scene_svg(&scene, 800.0, 600.0)?;
//! save_scene_svg(&scene, 800.0, 600.0, "out.svg")?;
//! # Ok::<_, esoc_gfx::error::GfxError>(())
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]
#![allow(clippy::format_push_string)]

pub mod error;
pub mod scene_svg;

#[cfg(feature = "png")]
pub(crate) fn usvg_options_with_fonts() -> resvg::usvg::Options<'static> {
    let mut opt = resvg::usvg::Options::default();
    let fontdb = opt.fontdb_mut();
    fontdb.load_system_fonts();
    opt
}

pub use error::{GfxError, Result};
pub use scene_svg::{render_scene_svg, render_scene_svg_with_metadata, save_scene_svg};

#[cfg(feature = "png")]
pub use scene_svg::{render_scene_png, save_scene_png};
