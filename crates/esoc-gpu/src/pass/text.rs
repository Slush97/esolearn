// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text render pass (bitmap glyph atlas, Phase 2 placeholder).
//!
//! In Phase 2, text is rendered as simple colored rectangles (no atlas yet).
//! The MSDF glyph atlas will be added in Phase 4.

use crate::context::GpuContext;
use crate::pass::rect::RectInstance;

/// Placeholder text pass that renders text bounding boxes as colored rects.
///
/// Real MSDF text rendering is Phase 4 work.
pub struct TextPass {
    // For now, text rendering delegates to the rect pipeline
    // as a visible placeholder (colored background rectangles).
}

impl TextPass {
    /// Create a new text pass (placeholder).
    pub fn new(_ctx: &GpuContext) -> Self {
        Self {}
    }

    /// Estimate text bounds for a simple approximation.
    pub fn estimate_text_bounds(text: &str, font_size: f32) -> (f32, f32) {
        // Rough monospace approximation: 0.6 × font_size per char
        let w = text.len() as f32 * font_size * 0.6;
        let h = font_size;
        (w, h)
    }

    /// Convert a TextMark to a rect instance for placeholder rendering.
    pub fn text_to_rect(
        position: [f32; 2],
        text: &str,
        font_size: f32,
        color: [f32; 4],
    ) -> RectInstance {
        let (w, h) = Self::estimate_text_bounds(text, font_size);
        RectInstance {
            rect: [position[0], position[1] - h, w, h],
            fill_color: [0.0; 4], // transparent fill
            stroke_color: color,
            params: [0.0, 0.0, 0.0, 0.0],
        }
    }
}
