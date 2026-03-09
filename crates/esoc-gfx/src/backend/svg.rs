// SPDX-License-Identifier: MIT OR Apache-2.0
//! SVG render backend — zero external dependencies.

use std::fmt::Write;

use crate::backend::RenderBackend;
use crate::canvas::Canvas;
use crate::element::{escape_xml, stroke_dash_attrs, DrawElement, Element};
use crate::error::Result;

/// SVG render backend that produces an SVG XML string.
#[derive(Clone, Debug, Default)]
pub struct SvgBackend;

impl RenderBackend for SvgBackend {
    type Output = String;

    fn render(&self, canvas: &Canvas) -> Result<Self::Output> {
        let mut svg = String::with_capacity(4096);
        write_svg(&mut svg, canvas)?;
        Ok(svg)
    }
}

fn write_svg(w: &mut String, canvas: &Canvas) -> Result<()> {
    writeln!(
        w,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        canvas.width, canvas.height, canvas.width, canvas.height
    )?;

    // Write defs (gradients + clips)
    let has_defs = !canvas.gradients().is_empty() || !canvas.clips().is_empty();
    if has_defs {
        writeln!(w, "  <defs>")?;
        for grad in canvas.gradients() {
            writeln!(
                w,
                r#"    <linearGradient id="{}" x1="{}" y1="{}" x2="{}" y2="{}">"#,
                escape_xml(&grad.id),
                grad.x1,
                grad.y1,
                grad.x2,
                grad.y2
            )?;
            for (offset, color) in &grad.stops {
                writeln!(
                    w,
                    r#"      <stop offset="{}" stop-color="{}" stop-opacity="{}"/>"#,
                    offset,
                    color.to_hex(),
                    color.a
                )?;
            }
            writeln!(w, "    </linearGradient>")?;
        }
        for clip in canvas.clips() {
            writeln!(w, r#"    <clipPath id="{}">"#, escape_xml(&clip.id))?;
            writeln!(
                w,
                r#"      <rect x="{}" y="{}" width="{}" height="{}"/>"#,
                clip.rect.x, clip.rect.y, clip.rect.width, clip.rect.height
            )?;
            writeln!(w, "    </clipPath>")?;
        }
        writeln!(w, "  </defs>")?;
    }

    // Write elements sorted by layer
    let mut clip_counter = ClipCounter::new();
    for elem in canvas.elements_sorted() {
        write_element_tree(w, elem, 1, &mut clip_counter)?;
    }

    writeln!(w, "</svg>")?;
    Ok(())
}

/// Counter for generating unique clip-path IDs within a single render.
struct ClipCounter(usize);

impl ClipCounter {
    fn new() -> Self {
        Self(0)
    }
    fn next(&mut self) -> usize {
        let id = self.0;
        self.0 += 1;
        id
    }
}

fn write_element_tree(w: &mut String, elem: &DrawElement, indent: usize, clip_counter: &mut ClipCounter) -> Result<()> {
    let pad = "  ".repeat(indent);
    match &elem.kind {
        Element::Line {
            start,
            end,
            stroke,
        } => {
            let dash = stroke_dash_attrs(&stroke.dash);
            writeln!(
                w,
                r#"{pad}<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}"{dash}/>"#,
                start.x,
                start.y,
                end.x,
                end.y,
                stroke.color.to_svg_string(),
                stroke.width,
            )?;
        }
        Element::Polyline {
            points,
            stroke,
            fill,
        } => {
            let pts: String = points
                .iter()
                .map(|p| format!("{},{}", p.x, p.y))
                .collect::<Vec<_>>()
                .join(" ");
            let dash = stroke_dash_attrs(&stroke.dash);
            writeln!(
                w,
                r#"{pad}<polyline points="{pts}" fill="{}" stroke="{}" stroke-width="{}"{dash} stroke-linejoin="{}" stroke-linecap="{}"/>"#,
                fill.to_svg_string(),
                stroke.color.to_svg_string(),
                stroke.width,
                stroke.line_join.as_svg_str(),
                stroke.line_cap.as_svg_str(),
            )?;
        }
        Element::Rect {
            rect,
            fill,
            stroke,
            rx,
        } => {
            let stroke_attrs = match stroke {
                Some(s) => format!(
                    r#" stroke="{}" stroke-width="{}""#,
                    s.color.to_svg_string(),
                    s.width
                ),
                None => String::new(),
            };
            let rx_attr = if *rx > 0.0 {
                format!(r#" rx="{rx}""#)
            } else {
                String::new()
            };
            writeln!(
                w,
                r#"{pad}<rect x="{}" y="{}" width="{}" height="{}" fill="{}"{stroke_attrs}{rx_attr}/>"#,
                rect.x, rect.y, rect.width, rect.height, fill.to_svg_string(),
            )?;
        }
        Element::Circle {
            center,
            radius,
            fill,
            stroke,
        } => {
            let stroke_attrs = match stroke {
                Some(s) => format!(
                    r#" stroke="{}" stroke-width="{}""#,
                    s.color.to_svg_string(),
                    s.width
                ),
                None => String::new(),
            };
            writeln!(
                w,
                r#"{pad}<circle cx="{}" cy="{}" r="{}" fill="{}"{stroke_attrs}/>"#,
                center.x, center.y, radius, fill.to_svg_string(),
            )?;
        }
        Element::Path {
            data, fill, stroke, ..
        } => {
            let stroke_attrs = match stroke {
                Some(s) => {
                    let dash = stroke_dash_attrs(&s.dash);
                    format!(
                        r#" stroke="{}" stroke-width="{}"{dash}"#,
                        s.color.to_svg_string(),
                        s.width
                    )
                }
                None => String::new(),
            };
            writeln!(
                w,
                r#"{pad}<path d="{}" fill="{}"{stroke_attrs}/>"#,
                data.d,
                fill.to_svg_string(),
            )?;
        }
        Element::Text {
            position,
            content,
            font,
            rotation,
        } => {
            let transform = match rotation {
                Some(deg) => format!(r#" transform="rotate({deg},{},{})""#, position.x, position.y),
                None => String::new(),
            };
            writeln!(
                w,
                r#"{pad}<text x="{}" y="{}" font-family="{}" font-size="{}" font-weight="{}" fill="{}" text-anchor="{}"{transform}>{}</text>"#,
                position.x,
                position.y,
                escape_xml(&font.family),
                font.size,
                font.weight,
                font.color.to_svg_string(),
                font.anchor.as_svg_str(),
                escape_xml(content),
            )?;
        }
        Element::Group { children, clip } => {
            let clip_attr = match clip {
                Some(rect) => {
                    let clip_id = clip_counter.next();
                    writeln!(w, "{pad}<defs>")?;
                    writeln!(
                        w,
                        r#"{pad}  <clipPath id="clip{clip_id}"><rect x="{}" y="{}" width="{}" height="{}"/></clipPath>"#,
                        rect.x, rect.y, rect.width, rect.height
                    )?;
                    writeln!(w, "{pad}</defs>")?;
                    format!(r#" clip-path="url(#clip{clip_id})""#)
                }
                None => String::new(),
            };
            writeln!(w, "{pad}<g{clip_attr}>")?;
            for child in children {
                write_element_tree(w, child, indent + 1, clip_counter)?;
            }
            writeln!(w, "{pad}</g>")?;
        }
    }
    Ok(())
}

/// Render a canvas to an SVG string.
pub fn render_svg(canvas: &Canvas) -> Result<String> {
    SvgBackend.render(canvas)
}

/// Save a canvas as an SVG file.
pub fn save_svg(canvas: &Canvas, path: &str) -> Result<()> {
    let svg = render_svg(canvas)?;
    std::fs::write(path, svg)?;
    Ok(())
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::canvas::Canvas;
    use crate::color::Color;
    use crate::element::{DrawElement, Element};
    use crate::geom::Rect;
    use crate::layer::Layer;
    use crate::style::{Fill, FontStyle, Stroke};

    #[test]
    fn test_svg_output_basic() {
        let mut canvas = Canvas::new(200.0, 100.0);
        canvas.add(DrawElement::new(
            Element::filled_rect(Rect::new(0.0, 0.0, 200.0, 100.0), Fill::Solid(Color::WHITE)),
            Layer::Background,
        ));
        canvas.add(DrawElement::new(
            Element::circle(100.0, 50.0, 20.0, Fill::Solid(Color::RED)),
            Layer::Data,
        ));
        let svg = render_svg(&canvas).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn test_svg_text_escaping() {
        let mut canvas = Canvas::new(100.0, 100.0);
        canvas.add(DrawElement::new(
            Element::text(10.0, 20.0, "x < y & z > w", FontStyle::new(12.0)),
            Layer::Annotations,
        ));
        let svg = render_svg(&canvas).unwrap();
        assert!(svg.contains("x &lt; y &amp; z &gt; w"));
    }

    #[test]
    fn test_svg_polyline() {
        let mut canvas = Canvas::new(100.0, 100.0);
        canvas.add(DrawElement::new(
            Element::polyline(
                vec![
                    crate::geom::Point::new(0.0, 0.0),
                    crate::geom::Point::new(50.0, 50.0),
                    crate::geom::Point::new(100.0, 0.0),
                ],
                Stroke::solid(Color::BLUE, 2.0),
            ),
            Layer::Data,
        ));
        let svg = render_svg(&canvas).unwrap();
        assert!(svg.contains("<polyline"));
    }

    #[test]
    fn test_svg_wellformed() {
        let mut canvas = Canvas::new(400.0, 300.0);
        canvas.add(DrawElement::filled_rect(
            Rect::new(0.0, 0.0, 400.0, 300.0),
            Fill::Solid(Color::WHITE),
            Layer::Background,
        ));
        canvas.add(DrawElement::circle(
            200.0, 150.0, 50.0,
            Fill::Solid(Color::BLUE),
            Layer::Data,
        ));
        canvas.add(DrawElement::text(
            200.0, 30.0, "Test Chart",
            FontStyle::new(16.0),
            Layer::Annotations,
        ));
        let svg = render_svg(&canvas).unwrap();
        // Basic well-formedness: starts with <svg, ends with </svg>
        assert!(svg.trim().starts_with("<svg"));
        assert!(svg.trim().ends_with("</svg>"));
    }
}
