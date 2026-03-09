// SPDX-License-Identifier: MIT OR Apache-2.0
//! Figure: top-level chart container.

use crate::axes::Axes;
use crate::error::Result;
use crate::render::render_figure;
use crate::theme::Theme;

/// A figure is the top-level container for one or more plot areas.
pub struct Figure {
    /// Figure width in pixels.
    pub width: f64,
    /// Figure height in pixels.
    pub height: f64,
    /// Figure title.
    pub title: Option<String>,
    /// Theme.
    pub theme: Theme,
    /// Plot areas.
    pub(crate) axes: Vec<Axes>,
}

impl Figure {
    /// Create a new figure with default dimensions.
    pub fn new() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            title: None,
            theme: Theme::default(),
            axes: Vec::new(),
        }
    }

    /// Set the figure size.
    pub fn size(mut self, width: f64, height: f64) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the figure title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the theme.
    pub fn theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }

    /// Add a new axes (plot area) and return a mutable reference to it.
    pub fn add_axes(&mut self) -> &mut Axes {
        self.axes.push(Axes::new());
        self.axes.last_mut().unwrap()
    }

    /// Render to an SVG string.
    pub fn to_svg(&self) -> Result<String> {
        let canvas = render_figure(self)?;
        let svg = esoc_gfx::backend::svg::render_svg(&canvas)?;
        Ok(svg)
    }

    /// Save as an SVG file.
    pub fn save_svg(&self, path: &str) -> Result<()> {
        let svg = self.to_svg()?;
        std::fs::write(path, svg)?;
        Ok(())
    }

    /// Render to PNG bytes (requires `png` feature).
    #[cfg(feature = "png")]
    pub fn to_png(&self) -> Result<Vec<u8>> {
        let canvas = render_figure(self)?;
        let bytes = esoc_gfx::backend::png::PngBackend
            .render(&canvas)
            .map_err(crate::error::ChartError::Gfx)?;
        Ok(bytes)
    }

    /// Save as a PNG file (requires `png` feature).
    #[cfg(feature = "png")]
    pub fn save_png(&self, path: &str) -> Result<()> {
        let bytes = self.to_png()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

impl Default for Figure {
    fn default() -> Self {
        Self::new()
    }
}
