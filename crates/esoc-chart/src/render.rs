// SPDX-License-Identifier: MIT OR Apache-2.0
//! Rendering pipeline: Figure → Canvas.

use esoc_gfx::canvas::Canvas;
use esoc_gfx::element::{DrawElement, Element};
use esoc_gfx::geom::Rect;
use esoc_gfx::layer::Layer;
use esoc_gfx::style::{Fill, FontStyle, Stroke, TextAnchor};
use esoc_gfx::transform::{AxisTransform, CoordinateTransform, ViewportTransform};

use crate::axes::Axes;
use crate::axis::Scale;
use crate::error::Result;
use crate::figure::Figure;
use crate::legend::{render_legend, LegendEntry};
use crate::series::DataBounds;

/// Margins for plot area (in pixels).
struct Margins {
    top: f64,
    right: f64,
    bottom: f64,
    left: f64,
}

/// Render a figure to a canvas.
pub fn render_figure(fig: &Figure) -> Result<Canvas> {
    let mut canvas = Canvas::new(fig.width, fig.height);

    // Background
    canvas.draw(
        Element::filled_rect(
            Rect::new(0.0, 0.0, fig.width, fig.height),
            Fill::Solid(fig.theme.background),
        ),
        Layer::Background,
    );

    // Figure title
    let title_offset = if let Some(title) = &fig.title {
        let font = FontStyle {
            family: fig.theme.font_family.clone(),
            size: fig.theme.title_font_size,
            weight: 700,
            color: fig.theme.foreground,
            anchor: TextAnchor::Middle,
        };
        canvas.add(DrawElement::text(
            fig.width / 2.0,
            fig.theme.title_font_size + 8.0,
            title,
            font,
            Layer::Annotations,
        ));
        fig.theme.title_font_size + 16.0
    } else {
        8.0
    };

    // For now, single-axes layout (fill available space)
    let n_axes = fig.axes.len().max(1);
    let avail_height = fig.height - title_offset;
    let axes_height = avail_height / n_axes as f64;

    for (i, axes) in fig.axes.iter().enumerate() {
        let axes_y = title_offset + i as f64 * axes_height;
        let axes_rect = Rect::new(0.0, axes_y, fig.width, axes_height);
        render_axes(&mut canvas, axes, axes_rect, &fig.theme)?;
    }

    Ok(canvas)
}

fn render_axes(canvas: &mut Canvas, axes: &Axes, bounds: Rect, theme: &crate::theme::Theme) -> Result<()> {
    // Compute margins
    let margins = compute_margins(axes, theme);
    let plot_area = Rect::new(
        bounds.x + margins.left,
        bounds.y + margins.top,
        (bounds.width - margins.left - margins.right).max(1.0),
        (bounds.height - margins.top - margins.bottom).max(1.0),
    );

    // Merge data bounds from all series
    let data_bounds = axes
        .series
        .iter()
        .map(|s| s.data_bounds())
        .reduce(DataBounds::union)
        .unwrap_or(DataBounds::new(0.0, 1.0, 0.0, 1.0));

    // Apply manual range overrides or pad
    let (x_min, x_max) = axes.x_config.range.unwrap_or_else(|| {
        let b = data_bounds.pad(0.05);
        (b.x_min, b.x_max)
    });
    let (y_min, y_max) = axes.y_config.range.unwrap_or_else(|| {
        let b = data_bounds.pad(0.05);
        (b.y_min, b.y_max)
    });

    // Build transforms
    let x_transform = match &axes.x_config.scale {
        Scale::Linear => AxisTransform::Linear { min: x_min, max: x_max },
        Scale::Log => AxisTransform::Log { min: x_min, max: x_max },
        Scale::Categorical(labels) => AxisTransform::Categorical { count: labels.len() },
    };
    let y_transform = match &axes.y_config.scale {
        Scale::Linear => AxisTransform::Linear { min: y_min, max: y_max },
        Scale::Log => AxisTransform::Log { min: y_min, max: y_max },
        Scale::Categorical(labels) => AxisTransform::Categorical { count: labels.len() },
    };
    let viewport = ViewportTransform::new(plot_area);
    let coord_transform = CoordinateTransform::new(x_transform, y_transform, viewport);

    // Grid lines
    if theme.show_grid {
        render_grid(canvas, &plot_area, x_min, x_max, y_min, y_max, axes, theme);
    }

    // Axis frame
    render_axis_frame(canvas, &plot_area, x_min, x_max, y_min, y_max, axes, theme);

    // Data series
    for (i, series) in axes.series.iter().enumerate() {
        series.render(canvas, &coord_transform, theme, i);
    }

    // Axes title
    if let Some(title) = &axes.title {
        let font = FontStyle {
            family: theme.font_family.clone(),
            size: theme.title_font_size * 0.9,
            weight: 700,
            color: theme.foreground,
            anchor: TextAnchor::Middle,
        };
        canvas.add(DrawElement::text(
            plot_area.x + plot_area.width / 2.0,
            plot_area.y - 8.0,
            title,
            font,
            Layer::Annotations,
        ));
    }

    // X-axis label
    if let Some(label) = &axes.x_config.label {
        let font = FontStyle {
            family: theme.font_family.clone(),
            size: theme.label_font_size,
            weight: 400,
            color: theme.foreground,
            anchor: TextAnchor::Middle,
        };
        canvas.add(DrawElement::text(
            plot_area.x + plot_area.width / 2.0,
            plot_area.bottom() + margins.bottom - 8.0,
            label,
            font,
            Layer::Annotations,
        ));
    }

    // Y-axis label (rotated)
    if let Some(label) = &axes.y_config.label {
        let font = FontStyle {
            family: theme.font_family.clone(),
            size: theme.label_font_size,
            weight: 400,
            color: theme.foreground,
            anchor: TextAnchor::Middle,
        };
        let lx = plot_area.x - margins.left + theme.label_font_size + 4.0;
        let ly = plot_area.y + plot_area.height / 2.0;
        canvas.add(DrawElement::new(
            Element::Text {
                position: esoc_gfx::geom::Point::new(lx, ly),
                content: label.clone(),
                font,
                rotation: Some(-90.0),
            },
            Layer::Annotations,
        ));
    }

    // Legend
    if axes.show_legend {
        let entries: Vec<LegendEntry> = axes
            .series
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                s.label().map(|l| LegendEntry {
                    label: l.to_string(),
                    color: theme.palette.get(i),
                })
            })
            .collect();
        if !entries.is_empty() {
            render_legend(canvas, plot_area, &entries, axes.legend_position, theme);
        }
    }

    Ok(())
}

fn compute_margins(axes: &Axes, theme: &crate::theme::Theme) -> Margins {
    let top = if axes.title.is_some() {
        theme.title_font_size * 1.5 + 10.0
    } else {
        20.0
    };
    let bottom = if axes.x_config.label.is_some() {
        theme.tick_font_size * 1.5 + theme.label_font_size + 20.0
    } else {
        theme.tick_font_size * 1.5 + 20.0
    };
    let left = if axes.y_config.label.is_some() {
        theme.tick_font_size * 4.0 + theme.label_font_size + 15.0
    } else {
        theme.tick_font_size * 4.0 + 15.0
    };
    let right = 20.0;

    Margins {
        top,
        right,
        bottom,
        left,
    }
}

fn render_grid(
    canvas: &mut Canvas,
    plot_area: &Rect,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    axes: &Axes,
    theme: &crate::theme::Theme,
) {
    let grid_stroke = Stroke::solid(theme.grid_color, theme.grid_width);

    // X grid
    let x_ticks = crate::axis::nice_ticks(x_min, x_max, axes.x_config.tick_count);
    for &pos in &x_ticks.positions {
        if pos < x_min || pos > x_max {
            continue;
        }
        let t = if (x_max - x_min).abs() < 1e-15 {
            0.5
        } else {
            (pos - x_min) / (x_max - x_min)
        };
        let px = plot_area.x + t * plot_area.width;
        canvas.add(DrawElement::line(
            px, plot_area.y, px, plot_area.bottom(),
            grid_stroke.clone(),
            Layer::Grid,
        ));
    }

    // Y grid
    let y_ticks = crate::axis::nice_ticks(y_min, y_max, axes.y_config.tick_count);
    for &pos in &y_ticks.positions {
        if pos < y_min || pos > y_max {
            continue;
        }
        let t = if (y_max - y_min).abs() < 1e-15 {
            0.5
        } else {
            (pos - y_min) / (y_max - y_min)
        };
        let py = plot_area.bottom() - t * plot_area.height;
        canvas.add(DrawElement::line(
            plot_area.x, py, plot_area.right(), py,
            grid_stroke.clone(),
            Layer::Grid,
        ));
    }
}

fn render_axis_frame(
    canvas: &mut Canvas,
    plot_area: &Rect,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    axes: &Axes,
    theme: &crate::theme::Theme,
) {
    let axis_stroke = Stroke::solid(theme.foreground, theme.axis_width);

    // Bottom axis line
    canvas.add(DrawElement::line(
        plot_area.x, plot_area.bottom(), plot_area.right(), plot_area.bottom(),
        axis_stroke.clone(),
        Layer::Grid,
    ));
    // Left axis line
    canvas.add(DrawElement::line(
        plot_area.x, plot_area.y, plot_area.x, plot_area.bottom(),
        axis_stroke,
        Layer::Grid,
    ));

    // X tick marks and labels
    let x_ticks = crate::axis::nice_ticks(x_min, x_max, axes.x_config.tick_count);
    let tick_font = FontStyle {
        family: theme.font_family.clone(),
        size: theme.tick_font_size,
        weight: 400,
        color: theme.foreground,
        anchor: TextAnchor::Middle,
    };

    for (i, &pos) in x_ticks.positions.iter().enumerate() {
        if pos < x_min || pos > x_max {
            continue;
        }
        let t = if (x_max - x_min).abs() < 1e-15 {
            0.5
        } else {
            (pos - x_min) / (x_max - x_min)
        };
        let px = plot_area.x + t * plot_area.width;

        // Tick mark
        canvas.add(DrawElement::line(
            px, plot_area.bottom(), px, plot_area.bottom() + 5.0,
            Stroke::solid(theme.foreground, 0.5),
            Layer::Grid,
        ));

        // Tick label
        let label = axes.x_config.tick_labels.as_ref()
            .and_then(|tl| tl.get(i))
            .unwrap_or(&x_ticks.labels[i]);
        canvas.add(DrawElement::text(
            px,
            plot_area.bottom() + 5.0 + theme.tick_font_size,
            label,
            tick_font.clone(),
            Layer::Grid,
        ));
    }

    // Y tick marks and labels
    let y_ticks = crate::axis::nice_ticks(y_min, y_max, axes.y_config.tick_count);
    let y_tick_font = FontStyle {
        family: theme.font_family.clone(),
        size: theme.tick_font_size,
        weight: 400,
        color: theme.foreground,
        anchor: TextAnchor::End,
    };

    for (i, &pos) in y_ticks.positions.iter().enumerate() {
        if pos < y_min || pos > y_max {
            continue;
        }
        let t = if (y_max - y_min).abs() < 1e-15 {
            0.5
        } else {
            (pos - y_min) / (y_max - y_min)
        };
        let py = plot_area.bottom() - t * plot_area.height;

        // Tick mark
        canvas.add(DrawElement::line(
            plot_area.x - 5.0, py, plot_area.x, py,
            Stroke::solid(theme.foreground, 0.5),
            Layer::Grid,
        ));

        // Tick label
        let label = axes.y_config.tick_labels.as_ref()
            .and_then(|tl| tl.get(i))
            .unwrap_or(&y_ticks.labels[i]);
        canvas.add(DrawElement::text(
            plot_area.x - 8.0,
            py + theme.tick_font_size * 0.35,
            label,
            y_tick_font.clone(),
            Layer::Grid,
        ));
    }
}

#[cfg(test)]
mod tests {
    use crate::figure::Figure;

    #[test]
    fn test_render_empty_figure() {
        let mut fig = Figure::new().title("Empty");
        fig.add_axes();
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_render_scatter_svg() {
        let mut fig = Figure::new().size(400.0, 300.0).title("Scatter Test");
        let ax = fig.add_axes();
        ax.x_label("X").y_label("Y");
        ax.scatter(&[1.0, 2.0, 3.0, 4.0], &[1.0, 4.0, 2.0, 3.0])
            .label("data")
            .done();
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<circle"));
        assert!(svg.contains("Scatter Test"));
    }

    #[test]
    fn test_render_line_svg() {
        let mut fig = Figure::new();
        let ax = fig.add_axes();
        ax.line(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.5])
            .label("trend")
            .done();
        let svg = fig.to_svg().unwrap();
        assert!(svg.contains("<polyline"));
    }
}
