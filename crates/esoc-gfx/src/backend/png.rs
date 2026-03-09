// SPDX-License-Identifier: MIT OR Apache-2.0
//! PNG render backend — SVG → resvg → PNG (requires `png` feature).

use crate::backend::svg::render_svg;
use crate::backend::RenderBackend;
use crate::canvas::Canvas;
use crate::error::{GfxError, Result};

/// PNG render backend using resvg.
#[derive(Clone, Debug, Default)]
pub struct PngBackend;

impl RenderBackend for PngBackend {
    type Output = Vec<u8>;

    fn render(&self, canvas: &Canvas) -> Result<Self::Output> {
        let svg_str = render_svg(canvas)?;
        svg_to_png(&svg_str, canvas.width as u32, canvas.height as u32)
    }
}

/// Convert an SVG string to PNG bytes.
fn svg_to_png(svg: &str, width: u32, height: u32) -> Result<Vec<u8>> {
    let tree = resvg::usvg::Tree::from_str(svg, &resvg::usvg::Options::default())
        .map_err(|e| GfxError::Render(format!("SVG parse error: {e}")))?;

    let mut pixmap = tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| GfxError::Render("failed to create pixmap".to_string()))?;

    resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());

    pixmap
        .encode_png()
        .map_err(|e| GfxError::Render(format!("PNG encode error: {e}")))
}

/// Save a canvas as a PNG file.
pub fn save_png(canvas: &Canvas, path: &str) -> Result<()> {
    let bytes = PngBackend.render(canvas)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
