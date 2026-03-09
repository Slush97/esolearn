// SPDX-License-Identifier: MIT OR Apache-2.0
//! Render backends for converting a Canvas to output formats.

pub mod svg;

#[cfg(feature = "png")]
pub mod png;

use crate::canvas::Canvas;
use crate::error::Result;

/// A render backend converts a [`Canvas`] into an output format.
pub trait RenderBackend {
    /// The output type (e.g., `String` for SVG, `Vec<u8>` for PNG).
    type Output;

    /// Render the canvas to the output format.
    fn render(&self, canvas: &Canvas) -> Result<Self::Output>;
}
