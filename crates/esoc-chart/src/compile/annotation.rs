// SPDX-License-Identifier: MIT OR Apache-2.0
//! Annotation compilation: reference lines, bands, text → scene marks.

use crate::grammar::annotation::Annotation;
use crate::new_theme::NewTheme;
use esoc_scene::bounds::DataBounds;
use crate::compile::layout;
use esoc_scene::bounds::BoundingBox;
use esoc_scene::mark::{AreaMark, Mark, RectMark, RuleMark, TextAnchor, TextMark};
use esoc_scene::node::{Node, NodeId};
use esoc_scene::scale::Scale;
use esoc_scene::style::{FillStyle, FontStyle, StrokeStyle};
use esoc_scene::SceneGraph;

/// Generate annotation marks in the scene graph.
#[allow(clippy::too_many_arguments)]
pub fn generate_annotations(
    scene: &mut SceneGraph,
    plot_id: NodeId,
    root_id: NodeId,
    annotations: &[Annotation],
    data_bounds: &DataBounds,
    plot_w: f32,
    plot_h: f32,
    plot_x: f32,
    plot_y: f32,
    theme: &NewTheme,
) {
    let x_scale = Scale::Linear {
        domain: (data_bounds.x_min, data_bounds.x_max),
        range: (0.0, plot_w),
    };
    let y_scale = Scale::Linear {
        domain: (data_bounds.y_min, data_bounds.y_max),
        range: (plot_h, 0.0),
    };

    for ann in annotations {
        match ann {
            Annotation::HLine {
                y,
                color,
                width,
                dash,
                label,
            } => {
                let y_px = y_scale.map(*y);
                let stroke = StrokeStyle {
                    color: *color,
                    width: *width,
                    dash: dash.clone().unwrap_or_default(),
                    ..Default::default()
                };
                let rule = Node::with_mark(Mark::Rule(RuleMark {
                    segments: vec![([0.0, y_px], [plot_w, y_px])],
                    stroke,
                }))
                .z_order(3);
                scene.insert_child(plot_id, rule);

                if let Some(label_text) = label {
                    let text = Node::with_mark(Mark::Text(TextMark {
                        position: [plot_x + plot_w + 3.0, plot_y + y_px],
                        text: label_text.clone(),
                        font: FontStyle {
                            family: theme.font_family.clone(),
                            size: theme.tick_font_size,
                            weight: 400,
                            italic: false,
                        },
                        fill: FillStyle::Solid(*color),
                        angle: 0.0,
                        anchor: TextAnchor::Start,
                    }))
                    .z_order(4);
                    scene.insert_child(root_id, text);
                }
            }
            Annotation::VLine {
                x,
                color,
                width,
                dash,
                label,
            } => {
                let x_px = x_scale.map(*x);
                let stroke = StrokeStyle {
                    color: *color,
                    width: *width,
                    dash: dash.clone().unwrap_or_default(),
                    ..Default::default()
                };
                let rule = Node::with_mark(Mark::Rule(RuleMark {
                    segments: vec![([x_px, 0.0], [x_px, plot_h])],
                    stroke,
                }))
                .z_order(3);
                scene.insert_child(plot_id, rule);

                if let Some(label_text) = label {
                    let text = Node::with_mark(Mark::Text(TextMark {
                        position: [plot_x + x_px, plot_y - 3.0],
                        text: label_text.clone(),
                        font: FontStyle {
                            family: theme.font_family.clone(),
                            size: theme.tick_font_size,
                            weight: 400,
                            italic: false,
                        },
                        fill: FillStyle::Solid(*color),
                        angle: 0.0,
                        anchor: TextAnchor::Middle,
                    }))
                    .z_order(4);
                    scene.insert_child(root_id, text);
                }
            }
            Annotation::Band {
                y_min,
                y_max,
                color,
                label,
            } => {
                let y_top = y_scale.map(*y_max);
                let y_bot = y_scale.map(*y_min);
                let upper = vec![[0.0, y_top], [plot_w, y_top]];
                let lower = vec![[0.0, y_bot], [plot_w, y_bot]];
                let area = Node::with_mark(Mark::Area(AreaMark {
                    upper,
                    lower,
                    fill: FillStyle::Solid(*color),
                    stroke: StrokeStyle {
                        width: 0.0,
                        ..Default::default()
                    },
                }))
                .z_order(0);
                scene.insert_child(plot_id, area);

                if let Some(label_text) = label {
                    let mid_y = (y_top + y_bot) * 0.5;
                    let text = Node::with_mark(Mark::Text(TextMark {
                        position: [plot_x + plot_w - 5.0, plot_y + mid_y],
                        text: label_text.clone(),
                        font: FontStyle {
                            family: theme.font_family.clone(),
                            size: theme.tick_font_size,
                            weight: 400,
                            italic: true,
                        },
                        fill: FillStyle::Solid(theme.foreground),
                        angle: 0.0,
                        anchor: TextAnchor::End,
                    }))
                    .z_order(4);
                    scene.insert_child(root_id, text);
                }
            }
            Annotation::Text {
                x,
                y,
                text,
                color,
                font_size,
            } => {
                let x_px = x_scale.map(*x);
                let y_px = y_scale.map(*y);

                // Semi-transparent background for readability over data
                let text_w = layout::estimate_text_width(text, *font_size);
                let bg = Node::with_mark(Mark::Rect(RectMark {
                    bounds: BoundingBox::new(
                        x_px - 2.0,
                        y_px - font_size * 0.8,
                        text_w + 4.0,
                        font_size * 1.2,
                    ),
                    fill: FillStyle::Solid(theme.background.with_alpha(0.8)),
                    stroke: StrokeStyle { width: 0.0, ..Default::default() },
                    corner_radius: 2.0,
                }))
                .z_order(3);
                scene.insert_child(plot_id, bg);

                let text_node = Node::with_mark(Mark::Text(TextMark {
                    position: [x_px, y_px],
                    text: text.clone(),
                    font: FontStyle {
                        family: theme.font_family.clone(),
                        size: *font_size,
                        weight: 400,
                        italic: false,
                    },
                    fill: FillStyle::Solid(*color),
                    angle: 0.0,
                    anchor: TextAnchor::Start,
                }))
                .z_order(4);
                scene.insert_child(plot_id, text_node);
            }
        }
    }
}

/// Generate subtitle below the title.
pub fn generate_subtitle(
    scene: &mut SceneGraph,
    root_id: NodeId,
    subtitle: &str,
    chart_width: f32,
    title_font_size: f32,
    theme: &NewTheme,
) {
    // Position subtitle below title with proper spacing.
    // Title is rendered at margins.top * 0.6 ≈ title_font_size * 0.6 + 12.
    // Place subtitle one full line below that.
    let title_y = title_font_size + 10.0;
    let y = title_y + title_font_size + 6.0;
    let text = Node::with_mark(Mark::Text(TextMark {
        position: [chart_width * 0.5, y],
        text: subtitle.to_string(),
        font: FontStyle {
            family: theme.font_family.clone(),
            size: theme.subtitle_font_size,
            weight: 400,
            italic: false,
        },
        fill: FillStyle::Solid(theme.muted_foreground),
        angle: 0.0,
        anchor: TextAnchor::Middle,
    }))
    .z_order(10);
    scene.insert_child(root_id, text);
}

/// Generate caption below the plot area.
pub fn generate_caption(
    scene: &mut SceneGraph,
    root_id: NodeId,
    caption: &str,
    chart_width: f32,
    chart_height: f32,
    theme: &NewTheme,
) {
    let text = Node::with_mark(Mark::Text(TextMark {
        position: [chart_width - 10.0, chart_height - 5.0],
        text: caption.to_string(),
        font: FontStyle {
            family: theme.font_family.clone(),
            size: theme.tick_font_size,
            weight: 400,
            italic: true,
        },
        fill: FillStyle::Solid(theme.foreground),
        angle: 0.0,
        anchor: TextAnchor::End,
    }))
    .z_order(10);
    scene.insert_child(root_id, text);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::annotation::Annotation;

    fn test_scene() -> (SceneGraph, NodeId, NodeId) {
        let mut scene = SceneGraph::with_root();
        let root = scene.root().unwrap();
        let plot = Node::container();
        let plot_id = scene.insert_child(root, plot);
        (scene, root, plot_id)
    }

    fn test_bounds() -> DataBounds {
        DataBounds::new(0.0, 100.0, 0.0, 100.0)
    }

    /// Count all non-container nodes in the scene.
    fn count_marks(scene: &SceneGraph) -> usize {
        scene
            .iter()
            .filter(|(_, node)| !matches!(node.content, esoc_scene::node::NodeContent::Container))
            .count()
    }

    #[test]
    fn hline_generates_rule() {
        let (mut scene, root, plot_id) = test_scene();
        let bounds = test_bounds();
        let theme = NewTheme::default();
        let annotations = vec![Annotation::hline(50.0)];
        generate_annotations(
            &mut scene, plot_id, root, &annotations, &bounds,
            400.0, 300.0, 50.0, 50.0, &theme,
        );
        assert!(count_marks(&scene) >= 1);
    }

    #[test]
    fn vline_generates_rule() {
        let (mut scene, root, plot_id) = test_scene();
        let bounds = test_bounds();
        let theme = NewTheme::default();
        let annotations = vec![Annotation::vline(25.0)];
        generate_annotations(
            &mut scene, plot_id, root, &annotations, &bounds,
            400.0, 300.0, 50.0, 50.0, &theme,
        );
        assert!(count_marks(&scene) >= 1);
    }

    #[test]
    fn band_generates_area() {
        let (mut scene, root, plot_id) = test_scene();
        let bounds = test_bounds();
        let theme = NewTheme::default();
        let annotations = vec![Annotation::band(20.0, 80.0)];
        generate_annotations(
            &mut scene, plot_id, root, &annotations, &bounds,
            400.0, 300.0, 50.0, 50.0, &theme,
        );
        assert!(count_marks(&scene) >= 1);
    }

    #[test]
    fn text_generates_text_mark() {
        let (mut scene, root, plot_id) = test_scene();
        let bounds = test_bounds();
        let theme = NewTheme::default();
        let annotations = vec![Annotation::text(50.0, 50.0, "hello")];
        generate_annotations(
            &mut scene, plot_id, root, &annotations, &bounds,
            400.0, 300.0, 50.0, 50.0, &theme,
        );
        assert!(count_marks(&scene) >= 1);
    }
}
