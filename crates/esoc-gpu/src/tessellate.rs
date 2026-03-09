// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lyon integration for tessellating areas and paths into triangle meshes.

use esoc_scene::mark::{AreaMark, PathCommand, PathMark};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers,
};

/// A simple vertex for tessellated geometry.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TessVertex {
    /// Position.
    pub position: [f32; 2],
}

// Safety: TessVertex is repr(C) with only f32 fields.
unsafe impl bytemuck::Pod for TessVertex {}
unsafe impl bytemuck::Zeroable for TessVertex {}

/// Tessellated mesh.
pub struct TessMesh {
    /// Vertex positions.
    pub vertices: Vec<TessVertex>,
    /// Triangle indices.
    pub indices: Vec<u32>,
}

/// Tessellate an area mark (filled region between upper and lower boundaries).
pub fn tessellate_area(area: &AreaMark) -> TessMesh {
    let mut builder = Path::builder();

    if let Some(&first) = area.upper.first() {
        builder.begin(point(first[0], first[1]));
        for &p in &area.upper[1..] {
            builder.line_to(point(p[0], p[1]));
        }
        // Walk lower boundary in reverse
        for &p in area.lower.iter().rev() {
            builder.line_to(point(p[0], p[1]));
        }
        builder.close();
    }

    let path = builder.build();
    tessellate_fill(&path)
}

/// Tessellate a path mark.
pub fn tessellate_path(path_mark: &PathMark, fill: bool) -> TessMesh {
    let mut builder = Path::builder();

    for cmd in &path_mark.commands {
        match *cmd {
            PathCommand::MoveTo(p) => {
                builder.begin(point(p[0], p[1]));
            }
            PathCommand::LineTo(p) => {
                builder.line_to(point(p[0], p[1]));
            }
            PathCommand::CubicTo(c1, c2, end) => {
                builder.cubic_bezier_to(
                    point(c1[0], c1[1]),
                    point(c2[0], c2[1]),
                    point(end[0], end[1]),
                );
            }
            PathCommand::QuadTo(c, end) => {
                builder.quadratic_bezier_to(point(c[0], c[1]), point(end[0], end[1]));
            }
            PathCommand::Close => {
                builder.close();
            }
        }
    }

    let path = builder.build();
    if fill {
        tessellate_fill(&path)
    } else {
        tessellate_stroke(&path, 1.0)
    }
}

fn tessellate_fill(path: &Path) -> TessMesh {
    let mut buffers: VertexBuffers<TessVertex, u32> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();

    let _ = tessellator.tessellate_path(
        path,
        &FillOptions::default(),
        &mut BuffersBuilder::new(&mut buffers, |vertex: lyon::tessellation::FillVertex| {
            TessVertex {
                position: [vertex.position().x, vertex.position().y],
            }
        }),
    );

    TessMesh {
        vertices: buffers.vertices,
        indices: buffers.indices,
    }
}

fn tessellate_stroke(path: &Path, width: f32) -> TessMesh {
    let mut buffers: VertexBuffers<TessVertex, u32> = VertexBuffers::new();
    let mut tessellator = StrokeTessellator::new();

    let _ = tessellator.tessellate_path(
        path,
        &StrokeOptions::default().with_line_width(width),
        &mut BuffersBuilder::new(&mut buffers, |vertex: lyon::tessellation::StrokeVertex| {
            TessVertex {
                position: [vertex.position().x, vertex.position().y],
            }
        }),
    );

    TessMesh {
        vertices: buffers.vertices,
        indices: buffers.indices,
    }
}
