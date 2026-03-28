// SPDX-License-Identifier: MIT OR Apache-2.0
//! SVG backend that consumes a `SceneGraph` from esoc-scene.

use std::fmt::Write;

use esoc_scene::mark::{
    ArcMark, AreaMark, LineMark, Mark, MarkBatch, PathCommand, PathMark, PointMark, RectMark,
    RuleMark, TextAnchor, TextMark,
};
use esoc_scene::node::NodeContent;
use esoc_scene::style::{FillStyle, LineCap, LineJoin, MarkerShape, StrokeStyle};
use esoc_scene::transform::Affine2D;
use esoc_scene::SceneGraph;

use crate::error::Result;

/// Render a scene graph to an SVG string.
pub fn render_scene_svg(scene: &SceneGraph, width: f32, height: f32) -> Result<String> {
    render_scene_svg_with_metadata(scene, width, height, None, None)
}

/// Render a scene graph to an SVG string with optional accessibility metadata.
pub fn render_scene_svg_with_metadata(
    scene: &SceneGraph,
    width: f32,
    height: f32,
    title: Option<&str>,
    description: Option<&str>,
) -> Result<String> {
    let mut svg = String::with_capacity(4096);
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">"#,
    )?;

    if let Some(t) = title {
        writeln!(svg, "  <title>{}</title>", escape_xml(t))?;
    }
    if let Some(d) = description {
        writeln!(svg, "  <desc>{}</desc>", escape_xml(d))?;
    }

    // White background
    writeln!(
        svg,
        r#"  <rect width="{width}" height="{height}" fill="white"/>"#,
    )?;

    // Collect marks in z-order
    let marks = scene.collect_marks();

    for (node_id, world_transform) in &marks {
        let Some(node) = scene.get(*node_id) else {
            continue;
        };

        match &node.content {
            NodeContent::Mark(mark) => {
                write_mark(&mut svg, mark, *world_transform, 1)?;
            }
            NodeContent::Batch(batch) => {
                write_batch(&mut svg, batch, *world_transform, 1)?;
            }
            NodeContent::Container => {}
        }
    }

    writeln!(svg, "</svg>")?;
    Ok(svg)
}

/// Save a scene graph as an SVG file.
pub fn save_scene_svg(scene: &SceneGraph, width: f32, height: f32, path: &str) -> Result<()> {
    let svg = render_scene_svg(scene, width, height)?;
    std::fs::write(path, svg)?;
    Ok(())
}

/// Render a scene graph to PNG bytes (requires `png` feature).
#[cfg(feature = "png")]
pub fn render_scene_png(scene: &SceneGraph, width: f32, height: f32) -> Result<Vec<u8>> {
    let svg_str = render_scene_svg(scene, width, height)?;
    let opt = crate::usvg_options_with_fonts();
    let tree = resvg::usvg::Tree::from_str(&svg_str, &opt)
        .map_err(|e| crate::error::GfxError::Render(format!("SVG parse error: {e}")))?;

    let w = width as u32;
    let h = height as u32;
    let mut pixmap = tiny_skia::Pixmap::new(w, h)
        .ok_or_else(|| crate::error::GfxError::Render("failed to create pixmap".to_string()))?;

    resvg::render(&tree, tiny_skia::Transform::default(), &mut pixmap.as_mut());

    pixmap
        .encode_png()
        .map_err(|e| crate::error::GfxError::Render(format!("PNG encode error: {e}")))
}

/// Save a scene graph as a PNG file (requires `png` feature).
#[cfg(feature = "png")]
pub fn save_scene_png(scene: &SceneGraph, width: f32, height: f32, path: &str) -> Result<()> {
    let bytes = render_scene_png(scene, width, height)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

fn write_mark(w: &mut String, mark: &Mark, transform: Affine2D, indent: usize) -> Result<()> {
    let pad = "  ".repeat(indent);
    match mark {
        Mark::Line(l) => write_line(w, l, transform, &pad)?,
        Mark::Rect(r) => write_rect(w, r, transform, &pad)?,
        Mark::Point(p) => write_point(w, p, transform, &pad)?,
        Mark::Area(a) => write_area(w, a, transform, &pad)?,
        Mark::Text(t) => write_text(w, t, transform, &pad)?,
        Mark::Arc(a) => write_arc(w, a, transform, &pad)?,
        Mark::Rule(r) => write_rule(w, r, transform, &pad)?,
        Mark::Path(p) => write_path_mark(w, p, transform, &pad)?,
        Mark::Image(_) => {} // Not supported in SVG backend
    }
    Ok(())
}

fn write_batch(
    w: &mut String,
    batch: &MarkBatch,
    transform: Affine2D,
    indent: usize,
) -> Result<()> {
    let pad = "  ".repeat(indent);
    match batch {
        MarkBatch::Points {
            positions,
            sizes,
            fills,
            shape,
            strokes,
        } => {
            for (i, pos) in positions.iter().enumerate() {
                let p = transform.apply(*pos);
                let size = *sizes.get(i);
                let fill = fills.get(i);
                let stroke = strokes.get(i);
                write_point_at(w, p, size, *shape, fill, stroke, &pad)?;
            }
        }
        MarkBatch::Rules { segments, stroke } => {
            for (start, end) in segments {
                let p0 = transform.apply(*start);
                let p1 = transform.apply(*end);
                write_line_segment(w, p0, p1, stroke, &pad)?;
            }
        }
        MarkBatch::Rects {
            rects,
            fills,
            strokes,
            corner_radius,
        } => {
            for (i, r) in rects.iter().enumerate() {
                let p = transform.apply([r.x, r.y]);
                let fill = fills.get(i);
                let stroke = strokes.get(i);
                write_rect_at(
                    w,
                    &RectParams {
                        p,
                        width: r.w * transform.a,
                        height: r.h * transform.d,
                        fill,
                        stroke,
                        corner_radius: *corner_radius,
                    },
                    &pad,
                )?;
            }
        }
    }
    Ok(())
}

fn write_line(w: &mut String, l: &LineMark, transform: Affine2D, pad: &str) -> Result<()> {
    if l.points.len() < 2 {
        return Ok(());
    }
    let pts: String = l
        .points
        .iter()
        .map(|p| {
            let tp = transform.apply(*p);
            format!("{},{}", tp[0], tp[1])
        })
        .collect::<Vec<_>>()
        .join(" ");
    let stroke_attrs = stroke_attrs(&l.stroke);
    writeln!(
        w,
        r#"{pad}<polyline points="{pts}" fill="none"{stroke_attrs}/>"#
    )?;
    Ok(())
}

fn write_rect(w: &mut String, r: &RectMark, transform: Affine2D, pad: &str) -> Result<()> {
    let p = transform.apply([r.bounds.x, r.bounds.y]);
    write_rect_at(
        w,
        &RectParams {
            p,
            width: r.bounds.w * transform.a,
            height: r.bounds.h * transform.d,
            fill: &r.fill,
            stroke: &r.stroke,
            corner_radius: r.corner_radius,
        },
        pad,
    )
}

/// Parameters for writing a rect SVG element.
struct RectParams<'a> {
    p: [f32; 2],
    width: f32,
    height: f32,
    fill: &'a FillStyle,
    stroke: &'a StrokeStyle,
    corner_radius: f32,
}

fn write_rect_at(w: &mut String, params: &RectParams<'_>, pad: &str) -> Result<()> {
    let RectParams {
        p,
        width,
        height,
        fill,
        stroke,
        corner_radius,
    } = params;
    let fill_str = fill_svg(fill);
    let stroke_str = stroke_attrs(stroke);
    let rx = if *corner_radius > 0.0 {
        format!(r#" rx="{corner_radius}""#)
    } else {
        String::new()
    };
    writeln!(
        w,
        r#"{pad}<rect x="{}" y="{}" width="{width}" height="{height}" fill="{fill_str}"{stroke_str}{rx}/>"#,
        p[0], p[1],
    )?;
    Ok(())
}

fn write_point(w: &mut String, p: &PointMark, transform: Affine2D, pad: &str) -> Result<()> {
    let pos = transform.apply(p.center);
    write_point_at(w, pos, p.size, p.shape, &p.fill, &p.stroke, pad)
}

fn write_point_at(
    w: &mut String,
    pos: [f32; 2],
    size: f32,
    shape: MarkerShape,
    fill: &FillStyle,
    stroke: &StrokeStyle,
    pad: &str,
) -> Result<()> {
    let r = size * 0.5;
    let fill_str = fill_svg(fill);
    let stroke_str = stroke_attrs(stroke);

    match shape {
        MarkerShape::Circle => {
            writeln!(
                w,
                r#"{pad}<circle cx="{}" cy="{}" r="{r}" fill="{fill_str}"{stroke_str}/>"#,
                pos[0], pos[1],
            )?;
        }
        MarkerShape::Square => {
            writeln!(
                w,
                r#"{pad}<rect x="{}" y="{}" width="{size}" height="{size}" fill="{fill_str}"{stroke_str}/>"#,
                pos[0] - r,
                pos[1] - r,
            )?;
        }
        MarkerShape::Diamond => {
            let pts = format!(
                "{},{} {},{} {},{} {},{}",
                pos[0],
                pos[1] - r,
                pos[0] + r,
                pos[1],
                pos[0],
                pos[1] + r,
                pos[0] - r,
                pos[1],
            );
            writeln!(
                w,
                r#"{pad}<polygon points="{pts}" fill="{fill_str}"{stroke_str}/>"#,
            )?;
        }
        MarkerShape::TriangleUp => {
            let pts = format!(
                "{},{} {},{} {},{}",
                pos[0],
                pos[1] - r,
                pos[0] + r,
                pos[1] + r,
                pos[0] - r,
                pos[1] + r,
            );
            writeln!(
                w,
                r#"{pad}<polygon points="{pts}" fill="{fill_str}"{stroke_str}/>"#,
            )?;
        }
        _ => {
            // Fallback: circle
            writeln!(
                w,
                r#"{pad}<circle cx="{}" cy="{}" r="{r}" fill="{fill_str}"{stroke_str}/>"#,
                pos[0], pos[1],
            )?;
        }
    }
    Ok(())
}

fn write_area(w: &mut String, a: &AreaMark, transform: Affine2D, pad: &str) -> Result<()> {
    if a.upper.is_empty() {
        return Ok(());
    }
    let mut d = String::new();
    let first = transform.apply(a.upper[0]);
    write!(d, "M{},{}", first[0], first[1])?;
    for p in &a.upper[1..] {
        let tp = transform.apply(*p);
        write!(d, " L{},{}", tp[0], tp[1])?;
    }
    for p in a.lower.iter().rev() {
        let tp = transform.apply(*p);
        write!(d, " L{},{}", tp[0], tp[1])?;
    }
    d.push('Z');

    let fill_str = fill_svg(&a.fill);
    let stroke_str = stroke_attrs(&a.stroke);
    writeln!(w, r#"{pad}<path d="{d}" fill="{fill_str}"{stroke_str}/>"#)?;
    Ok(())
}

fn write_text(w: &mut String, t: &TextMark, transform: Affine2D, pad: &str) -> Result<()> {
    let pos = transform.apply(t.position);
    let fill_str = fill_svg(&t.fill);
    let rotation = if t.angle.abs() > 0.01 {
        format!(r#" transform="rotate({},{},{})""#, t.angle, pos[0], pos[1])
    } else {
        String::new()
    };
    let weight = if t.font.weight >= 700 {
        r#" font-weight="bold""#
    } else {
        ""
    };
    let anchor_attr = match t.anchor {
        TextAnchor::Start => "",
        TextAnchor::Middle => r#" text-anchor="middle""#,
        TextAnchor::End => r#" text-anchor="end""#,
    };
    writeln!(
        w,
        r#"{pad}<text x="{}" y="{}" font-family="{}" font-size="{}" fill="{fill_str}"{weight}{anchor_attr}{rotation}>{}</text>"#,
        pos[0],
        pos[1],
        escape_xml(&t.font.family),
        t.font.size,
        escape_xml(&t.text),
    )?;
    Ok(())
}

fn write_arc(w: &mut String, a: &ArcMark, transform: Affine2D, pad: &str) -> Result<()> {
    let center = transform.apply(a.center);
    let fill_str = fill_svg(&a.fill);
    let stroke_str = stroke_attrs(&a.stroke);

    // SVG arc path
    let start_angle = a.start_angle;
    let end_angle = a.end_angle;
    let outer = a.outer_radius;
    let inner = a.inner_radius;

    let x1 = center[0] + outer * start_angle.cos();
    let y1 = center[1] + outer * start_angle.sin();
    let x2 = center[0] + outer * end_angle.cos();
    let y2 = center[1] + outer * end_angle.sin();

    let large_arc = u8::from((end_angle - start_angle).abs() > std::f32::consts::PI);

    let mut d = format!("M{x1},{y1} A{outer},{outer} 0 {large_arc} 1 {x2},{y2}");

    if inner > 0.0 {
        let x3 = center[0] + inner * end_angle.cos();
        let y3 = center[1] + inner * end_angle.sin();
        let x4 = center[0] + inner * start_angle.cos();
        let y4 = center[1] + inner * start_angle.sin();
        write!(
            d,
            " L{x3},{y3} A{inner},{inner} 0 {large_arc} 0 {x4},{y4} Z"
        )?;
    } else {
        write!(d, " L{},{} Z", center[0], center[1])?;
    }

    writeln!(w, r#"{pad}<path d="{d}" fill="{fill_str}"{stroke_str}/>"#)?;
    Ok(())
}

fn write_rule(w: &mut String, r: &RuleMark, transform: Affine2D, pad: &str) -> Result<()> {
    for (start, end) in &r.segments {
        let p0 = transform.apply(*start);
        let p1 = transform.apply(*end);
        write_line_segment(w, p0, p1, &r.stroke, pad)?;
    }
    Ok(())
}

fn write_line_segment(
    w: &mut String,
    p0: [f32; 2],
    p1: [f32; 2],
    stroke: &StrokeStyle,
    pad: &str,
) -> Result<()> {
    let stroke_str = stroke_attrs(stroke);
    writeln!(
        w,
        r#"{pad}<line x1="{}" y1="{}" x2="{}" y2="{}"{stroke_str}/>"#,
        p0[0], p0[1], p1[0], p1[1],
    )?;
    Ok(())
}

fn write_path_mark(w: &mut String, p: &PathMark, transform: Affine2D, pad: &str) -> Result<()> {
    let mut d = String::new();
    for cmd in &p.commands {
        match *cmd {
            PathCommand::MoveTo(pt) => {
                let tp = transform.apply(pt);
                write!(d, "M{},{} ", tp[0], tp[1])?;
            }
            PathCommand::LineTo(pt) => {
                let tp = transform.apply(pt);
                write!(d, "L{},{} ", tp[0], tp[1])?;
            }
            PathCommand::CubicTo(c1, c2, end) => {
                let tc1 = transform.apply(c1);
                let tc2 = transform.apply(c2);
                let te = transform.apply(end);
                write!(
                    d,
                    "C{},{} {},{} {},{} ",
                    tc1[0], tc1[1], tc2[0], tc2[1], te[0], te[1]
                )?;
            }
            PathCommand::QuadTo(c, end) => {
                let tc = transform.apply(c);
                let te = transform.apply(end);
                write!(d, "Q{},{} {},{} ", tc[0], tc[1], te[0], te[1])?;
            }
            PathCommand::Close => {
                d.push_str("Z ");
            }
        }
    }

    let fill_str = fill_svg(&p.fill);
    let stroke_str = stroke_attrs(&p.stroke);
    writeln!(w, r#"{pad}<path d="{d}" fill="{fill_str}"{stroke_str}/>"#)?;
    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────

fn fill_svg(fill: &FillStyle) -> String {
    match fill {
        FillStyle::None => "none".to_string(),
        FillStyle::Solid(c) => c.to_svg_string(),
    }
}

fn stroke_attrs(stroke: &StrokeStyle) -> String {
    if stroke.is_none() {
        return String::new();
    }
    let mut s = format!(
        r#" stroke="{}" stroke-width="{}""#,
        stroke.color.to_svg_string(),
        stroke.width,
    );
    if !stroke.dash.is_empty() {
        let dash: String = stroke
            .dash
            .iter()
            .map(|d| format!("{d}"))
            .collect::<Vec<_>>()
            .join(",");
        write!(s, r#" stroke-dasharray="{dash}""#).unwrap();
    }
    match stroke.line_cap {
        LineCap::Round => s.push_str(r#" stroke-linecap="round""#),
        LineCap::Square => s.push_str(r#" stroke-linecap="square""#),
        LineCap::Butt => {}
    }
    match stroke.line_join {
        LineJoin::Round => s.push_str(r#" stroke-linejoin="round""#),
        LineJoin::Bevel => s.push_str(r#" stroke-linejoin="bevel""#),
        LineJoin::Miter => {}
    }
    s
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use esoc_color::Color;
    use esoc_scene::bounds::BoundingBox;
    use esoc_scene::node::Node;
    use esoc_scene::style::FontStyle;

    #[test]
    fn basic_scene_svg() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();

        // Add a rect
        scene.insert_child(
            root,
            Node::with_mark(Mark::Rect(RectMark {
                bounds: BoundingBox::new(10.0, 10.0, 80.0, 60.0),
                fill: FillStyle::Solid(Color::from_hex("#1f77b4").unwrap()),
                stroke: StrokeStyle::default(),
                corner_radius: 0.0,
            })),
        );

        // Add a point
        scene.insert_child(
            root,
            Node::with_mark(Mark::Point(PointMark {
                center: [50.0, 50.0],
                size: 10.0,
                shape: MarkerShape::Circle,
                fill: FillStyle::Solid(Color::RED),
                stroke: StrokeStyle::default(),
            })),
        );

        let svg = render_scene_svg(&scene, 100.0, 100.0).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn line_mark_svg() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();

        scene.insert_child(
            root,
            Node::with_mark(Mark::Line(LineMark {
                points: vec![[0.0, 0.0], [50.0, 50.0], [100.0, 0.0]],
                stroke: StrokeStyle::solid(Color::BLUE, 2.0),
                interpolation: esoc_scene::mark::Interpolation::Linear,
            })),
        );

        let svg = render_scene_svg(&scene, 100.0, 100.0).unwrap();
        assert!(svg.contains("<polyline"));
    }

    #[test]
    fn text_escaping() {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();

        scene.insert_child(
            root,
            Node::with_mark(Mark::Text(TextMark {
                position: [10.0, 20.0],
                text: "x < y & z > w".to_string(),
                font: FontStyle::default(),
                fill: FillStyle::Solid(Color::BLACK),
                angle: 0.0,
                anchor: TextAnchor::Start,
            })),
        );

        let svg = render_scene_svg(&scene, 100.0, 100.0).unwrap();
        assert!(svg.contains("x &lt; y &amp; z &gt; w"));
    }
}
