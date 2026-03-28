// SPDX-License-Identifier: MIT OR Apache-2.0
//! Top-level renderer orchestrating all mark passes.

use esoc_color::Color;
use esoc_scene::mark::{Mark, MarkBatch};
use esoc_scene::node::NodeContent;
use esoc_scene::style::FillStyle;
use esoc_scene::transform::Affine2D;
use esoc_scene::SceneGraph;

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pass::line::{LineInstance, LinePass};
use crate::pass::point::{PointInstance, PointPass};
use crate::pass::rect::{RectInstance, RectPass};
use crate::pass::rule::{RuleInstance, RulePass};
use crate::pass::text::TextPass;
use crate::pass::tess::TessPass;

/// Viewport dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    /// X offset.
    pub x: f32,
    /// Y offset.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
}

/// The main GPU renderer.
pub struct Renderer {
    ctx: GpuContext,
    rect_pass: RectPass,
    point_pass: PointPass,
    line_pass: LinePass,
    rule_pass: RulePass,
    #[allow(dead_code)]
    text_pass: TextPass,
    #[allow(dead_code)]
    tess_pass: TessPass,
}

impl Renderer {
    /// Create a new headless renderer.
    pub fn new_headless() -> Option<Self> {
        let ctx = GpuContext::new_headless()?;
        let rect_pass = RectPass::new(&ctx);
        let point_pass = PointPass::new(&ctx);
        let line_pass = LinePass::new(&ctx);
        let rule_pass = RulePass::new(&ctx);
        let text_pass = TextPass::new(&ctx);
        let tess_pass = TessPass::new(&ctx);

        Some(Self {
            ctx,
            rect_pass,
            point_pass,
            line_pass,
            rule_pass,
            text_pass,
            tess_pass,
        })
    }

    /// Create a renderer from an existing GPU context.
    pub fn from_context(ctx: GpuContext) -> Self {
        let rect_pass = RectPass::new(&ctx);
        let point_pass = PointPass::new(&ctx);
        let line_pass = LinePass::new(&ctx);
        let rule_pass = RulePass::new(&ctx);
        let text_pass = TextPass::new(&ctx);
        let tess_pass = TessPass::new(&ctx);

        Self {
            ctx,
            rect_pass,
            point_pass,
            line_pass,
            rule_pass,
            text_pass,
            tess_pass,
        }
    }

    /// Access the GPU context.
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Render a scene graph to a texture view.
    pub fn render(
        &mut self,
        scene: &SceneGraph,
        target: &wgpu::TextureView,
        viewport: Viewport,
    ) {
        let vp = [viewport.x, viewport.y, viewport.width, viewport.height];

        // Collect instances from the scene
        let (rects, points, lines, rules) = self.collect_instances(scene);

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("esoc_render"),
            });

        // Clear pass
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }

        // Draw each pass
        self.rule_pass
            .draw(&self.ctx, &mut encoder, target, vp, &rules);
        self.rect_pass
            .draw(&self.ctx, &mut encoder, target, vp, &rects);
        self.line_pass
            .draw(&self.ctx, &mut encoder, target, vp, &lines);
        self.point_pass
            .draw(&self.ctx, &mut encoder, target, vp, &points);

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Render to an RGBA image buffer.
    pub fn render_to_image(
        &mut self,
        scene: &SceneGraph,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, GpuError> {
        if width == 0 || height == 0 {
            return Err(GpuError::InvalidViewport { width, height });
        }

        let texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render_target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.ctx.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        self.render(
            scene,
            &view,
            Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
            },
        );

        // Read back pixels
        let bytes_per_row = (width * 4 + 255) & !255; // align to 256
        let output_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| GpuError::ReadbackFailed("channel closed".into()))?
            .map_err(|e| GpuError::ReadbackFailed(format!("{e:?}")))?;

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height {
            let start = (row * bytes_per_row) as usize;
            let end = start + (width * 4) as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        Ok(pixels)
    }

    /// Collect all instances from the scene graph, sorted by z-order.
    fn collect_instances(
        &self,
        scene: &SceneGraph,
    ) -> (
        Vec<RectInstance>,
        Vec<PointInstance>,
        Vec<LineInstance>,
        Vec<RuleInstance>,
    ) {
        let mut rects = Vec::new();
        let mut points = Vec::new();
        let mut lines = Vec::new();
        let mut rules = Vec::new();

        let marks = scene.collect_marks();

        for (node_id, world_transform) in &marks {
            let Some(node) = scene.get(*node_id) else {
                continue;
            };

            match &node.content {
                NodeContent::Mark(mark) => {
                    self.collect_mark(mark, *world_transform, &mut rects, &mut points, &mut lines, &mut rules);
                }
                NodeContent::Batch(batch) => {
                    self.collect_batch(batch, *world_transform, &mut rects, &mut points, &mut lines, &mut rules);
                }
                NodeContent::Container => {}
            }
        }

        (rects, points, lines, rules)
    }

    #[allow(clippy::unused_self)]
    fn collect_mark(
        &self,
        mark: &Mark,
        transform: Affine2D,
        rects: &mut Vec<RectInstance>,
        points: &mut Vec<PointInstance>,
        lines: &mut Vec<LineInstance>,
        rules: &mut Vec<RuleInstance>,
    ) {
        match mark {
            Mark::Rect(r) => {
                let p = transform.apply([r.bounds.x, r.bounds.y]);
                let fill = fill_to_rgba(&r.fill);
                let stroke = color_to_rgba(r.stroke.color);
                rects.push(RectInstance {
                    rect: [p[0], p[1], r.bounds.w * transform.a, r.bounds.h * transform.d],
                    fill_color: fill,
                    stroke_color: stroke,
                    params: [r.corner_radius, r.stroke.width, 0.0, 0.0],
                });
            }
            Mark::Point(p) => {
                let pos = transform.apply(p.center);
                let fill = fill_to_rgba(&p.fill);
                let stroke = color_to_rgba(p.stroke.color);
                points.push(PointInstance {
                    center_size: [pos[0], pos[1], p.size, f32::from(p.shape.type_index() as u16)],
                    fill_color: fill,
                    stroke_color: stroke,
                    params: [p.stroke.width, 0.0, 0.0, 0.0],
                });
            }
            Mark::Line(l) => {
                if l.points.len() >= 2 {
                    let color = color_to_rgba(l.stroke.color);
                    for w in l.points.windows(2) {
                        let p0 = transform.apply(w[0]);
                        let p1 = transform.apply(w[1]);
                        lines.push(LineInstance {
                            endpoints: [p0[0], p0[1], p1[0], p1[1]],
                            params: [l.stroke.width, 0.0, 0.0, 0.0],
                            color,
                        });
                    }
                }
            }
            Mark::Rule(r) => {
                let color = color_to_rgba(r.stroke.color);
                for (start, end) in &r.segments {
                    let p0 = transform.apply(*start);
                    let p1 = transform.apply(*end);
                    rules.push(RuleInstance {
                        endpoints: [p0[0], p0[1], p1[0], p1[1]],
                        color,
                        params: [r.stroke.width, 0.0, 0.0, 0.0],
                    });
                }
            }
            Mark::Text(_) | Mark::Area(_) | Mark::Arc(_) | Mark::Path(_) | Mark::Image(_) => {
                // Phase 4 marks — not yet rendered
            }
        }
    }

    #[allow(clippy::unused_self)]
    fn collect_batch(
        &self,
        batch: &MarkBatch,
        transform: Affine2D,
        rects: &mut Vec<RectInstance>,
        points: &mut Vec<PointInstance>,
        _lines: &mut Vec<LineInstance>,
        rules: &mut Vec<RuleInstance>,
    ) {
        match batch {
            MarkBatch::Points {
                positions,
                sizes,
                fills,
                shape,
                strokes,
            } => {
                let shape_idx = f32::from(shape.type_index() as u16);
                for (i, pos) in positions.iter().enumerate() {
                    let p = transform.apply(*pos);
                    let fill = fill_to_rgba(fills.get(i));
                    let stroke = color_to_rgba(strokes.get(i).color);
                    points.push(PointInstance {
                        center_size: [p[0], p[1], *sizes.get(i), shape_idx],
                        fill_color: fill,
                        stroke_color: stroke,
                        params: [strokes.get(i).width, 0.0, 0.0, 0.0],
                    });
                }
            }
            MarkBatch::Rules { segments, stroke } => {
                let color = color_to_rgba(stroke.color);
                for (start, end) in segments {
                    let p0 = transform.apply(*start);
                    let p1 = transform.apply(*end);
                    rules.push(RuleInstance {
                        endpoints: [p0[0], p0[1], p1[0], p1[1]],
                        color,
                        params: [stroke.width, 0.0, 0.0, 0.0],
                    });
                }
            }
            MarkBatch::Rects {
                rects: batch_rects,
                fills,
                strokes,
                corner_radius,
            } => {
                for (i, r) in batch_rects.iter().enumerate() {
                    let p = transform.apply([r.x, r.y]);
                    let fill = fill_to_rgba(fills.get(i));
                    let stroke = color_to_rgba(strokes.get(i).color);
                    rects.push(RectInstance {
                        rect: [p[0], p[1], r.w * transform.a, r.h * transform.d],
                        fill_color: fill,
                        stroke_color: stroke,
                        params: [*corner_radius, strokes.get(i).width, 0.0, 0.0],
                    });
                }
            }
        }
    }
}

fn fill_to_rgba(fill: &FillStyle) -> [f32; 4] {
    match fill {
        FillStyle::None => [0.0; 4],
        FillStyle::Solid(c) => c.to_array(),
    }
}

fn color_to_rgba(c: Color) -> [f32; 4] {
    c.to_array()
}
