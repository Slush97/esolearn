// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chart: the top-level grammar container.

use crate::error::Result;
use crate::grammar::annotation::Annotation;
use crate::grammar::coord::CoordSystem;
use crate::grammar::facet::{Facet, FacetScales};
use crate::grammar::layer::Layer;
use crate::new_theme::NewTheme;
use esoc_scene::SceneGraph;

/// A chart definition (grammar of graphics).
#[derive(Clone, Debug)]
pub struct Chart {
    /// Layers of marks.
    pub layers: Vec<Layer>,
    /// Chart title.
    pub title: Option<String>,
    /// X-axis label.
    pub x_label: Option<String>,
    /// Y-axis label.
    pub y_label: Option<String>,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
    /// Theme.
    pub theme: NewTheme,
    /// Annotations (reference lines, bands, text).
    pub annotations: Vec<Annotation>,
    /// Subtitle (below title).
    pub subtitle: Option<String>,
    /// Caption (below plot).
    pub caption: Option<String>,
    /// Coordinate system.
    pub coord: CoordSystem,
    /// Faceting mode.
    pub facet: Facet,
    /// Facet scale sharing.
    pub facet_scales: FacetScales,
}

impl Chart {
    /// Create a new empty chart.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            title: None,
            x_label: None,
            y_label: None,
            width: 800.0,
            height: 600.0,
            theme: NewTheme::default(),
            annotations: Vec::new(),
            subtitle: None,
            caption: None,
            coord: CoordSystem::default(),
            facet: Facet::default(),
            facet_scales: FacetScales::default(),
        }
    }

    /// Add a layer.
    pub fn layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Set the title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the X-axis label.
    pub fn x_label(mut self, label: impl Into<String>) -> Self {
        self.x_label = Some(label.into());
        self
    }

    /// Set the Y-axis label.
    pub fn y_label(mut self, label: impl Into<String>) -> Self {
        self.y_label = Some(label.into());
        self
    }

    /// Set dimensions.
    pub fn size(mut self, width: f32, height: f32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set theme.
    pub fn theme(mut self, theme: NewTheme) -> Self {
        self.theme = theme;
        self
    }

    /// Add an annotation.
    pub fn annotate(mut self, annotation: Annotation) -> Self {
        self.annotations.push(annotation);
        self
    }

    /// Set subtitle.
    pub fn subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.subtitle = Some(subtitle.into());
        self
    }

    /// Set caption.
    pub fn caption(mut self, caption: impl Into<String>) -> Self {
        self.caption = Some(caption.into());
        self
    }

    /// Set coordinate system.
    pub fn coord(mut self, coord: CoordSystem) -> Self {
        self.coord = coord;
        self
    }

    /// Set faceting mode.
    pub fn facet(mut self, facet: Facet) -> Self {
        self.facet = facet;
        self
    }

    /// Set facet scale sharing.
    pub fn facet_scales(mut self, scales: FacetScales) -> Self {
        self.facet_scales = scales;
        self
    }

    /// Compile to a scene graph.
    pub fn build(&self) -> Result<SceneGraph> {
        crate::compile::compile_chart(self)
    }

    /// Compile and render to SVG.
    pub fn to_svg(&self) -> Result<String> {
        let scene = self.build()?;
        let svg = esoc_gfx::scene_svg::render_scene_svg(&scene, self.width, self.height)?;
        Ok(svg)
    }

    /// Save as SVG.
    pub fn save_svg(&self, path: &str) -> Result<()> {
        let svg = self.to_svg()?;
        std::fs::write(path, svg)?;
        Ok(())
    }

    /// Compile and render to PNG (requires `png` feature).
    #[cfg(feature = "png")]
    pub fn to_png(&self) -> Result<Vec<u8>> {
        let scene = self.build()?;
        let bytes =
            esoc_gfx::scene_svg::render_scene_png(&scene, self.width, self.height)?;
        Ok(bytes)
    }

    /// Save as PNG (requires `png` feature).
    #[cfg(feature = "png")]
    pub fn save_png(&self, path: &str) -> Result<()> {
        let bytes = self.to_png()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

impl Default for Chart {
    fn default() -> Self {
        Self::new()
    }
}
