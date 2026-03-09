// SPDX-License-Identifier: MIT OR Apache-2.0
//! Render passes — one per mark type.

pub mod line;
pub mod point;
pub mod rect;
pub mod rule;
pub mod tess;
pub mod text;

/// Viewport uniform data.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ViewportUniforms {
    /// x, y, width, height.
    pub viewport: [f32; 4],
}

unsafe impl bytemuck::Pod for ViewportUniforms {}
unsafe impl bytemuck::Zeroable for ViewportUniforms {}
