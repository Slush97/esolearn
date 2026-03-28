// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tessellated geometry render pass (areas, paths).
//!
//! Uses Lyon-tessellated triangle meshes uploaded as vertex/index buffers.

use crate::buffer::DynamicBuffer;
use crate::context::GpuContext;
use crate::tessellate::TessVertex;

/// Uniform data for tessellated draw calls.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TessUniforms {
    /// Viewport.
    pub viewport: [f32; 4],
    /// Fill color.
    pub color: [f32; 4],
    /// Transform (3x3, padded to std140).
    pub transform: [f32; 12],
}

unsafe impl bytemuck::Pod for TessUniforms {}
unsafe impl bytemuck::Zeroable for TessUniforms {}

/// Tessellated geometry pass.
pub struct TessPass {
    vertex_buffer: DynamicBuffer,
    index_buffer: DynamicBuffer,
}

impl TessPass {
    /// Create a new tessellation pass.
    pub fn new(ctx: &GpuContext) -> Self {
        Self {
            vertex_buffer: DynamicBuffer::new(
                &ctx.device,
                "tess_vertices",
                4096,
                wgpu::BufferUsages::VERTEX,
            ),
            index_buffer: DynamicBuffer::new(
                &ctx.device,
                "tess_indices",
                4096,
                wgpu::BufferUsages::INDEX,
            ),
        }
    }

    /// Upload tessellated mesh data.
    pub fn upload(
        &mut self,
        ctx: &GpuContext,
        vertices: &[TessVertex],
        indices: &[u32],
    ) {
        self.vertex_buffer
            .write(&ctx.device, &ctx.queue, bytemuck::cast_slice(vertices));
        self.index_buffer
            .write(&ctx.device, &ctx.queue, bytemuck::cast_slice(indices));
    }

    /// Get vertex buffer reference.
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        self.vertex_buffer.buffer()
    }

    /// Get index buffer reference.
    pub fn index_buffer(&self) -> &wgpu::Buffer {
        self.index_buffer.buffer()
    }
}
