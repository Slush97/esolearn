// SPDX-License-Identifier: MIT OR Apache-2.0
//! DOM-level integration tests for the SVG backend.
//!
//! These tests render small known scenes through `render_scene_svg`, parse the
//! output with `roxmltree`, and assert on element kinds, attributes, ordering,
//! and accessibility metadata. They complement the inline unit tests, which
//! only check that target substrings appear in the raw output.

use esoc_color::Color;
use esoc_gfx::scene_svg::{render_scene_svg, render_scene_svg_with_metadata};
use esoc_scene::SceneGraph;
use esoc_scene::bounds::BoundingBox;
use esoc_scene::mark::{
    Interpolation, LineMark, Mark, PointMark, RectMark, RuleMark, TextAnchor, TextMark,
};
use esoc_scene::node::Node;
use esoc_scene::style::{FillStyle, FontStyle, LineCap, MarkerShape, StrokeStyle};
use esoc_scene::transform::Affine2D;
use roxmltree::{Document, Node as XmlNode};

const W: f32 = 200.0;
const H: f32 = 100.0;

// ── Helpers ─────────────────────────────────────────────────────────

fn scene_with_root() -> (SceneGraph, esoc_scene::node::NodeId) {
    let s = SceneGraph::with_root();
    let root = s.root().expect("with_root() should produce a root");
    (s, root)
}

fn render(scene: &SceneGraph) -> String {
    render_scene_svg(scene, W, H).expect("render must succeed")
}

fn parse(svg: &str) -> Document<'_> {
    Document::parse(svg).expect("rendered SVG must be well-formed XML")
}

/// Walk every element in the document (excluding the root <svg>).
fn elements<'a, 'input>(doc: &'a Document<'input>) -> impl Iterator<Item = XmlNode<'a, 'input>> {
    doc.root_element()
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() != "svg")
}

fn count_tag(doc: &Document<'_>, tag: &str) -> usize {
    elements(doc).filter(|n| n.tag_name().name() == tag).count()
}

fn first_with_tag<'a, 'input>(
    doc: &'a Document<'input>,
    tag: &str,
) -> Option<XmlNode<'a, 'input>> {
    elements(doc).find(|n| n.tag_name().name() == tag)
}

fn attr_f32(node: XmlNode<'_, '_>, name: &str) -> f32 {
    node.attribute(name)
        .unwrap_or_else(|| panic!("attribute `{name}` missing on <{}>", node.tag_name().name()))
        .parse()
        .unwrap_or_else(|e| panic!("attribute `{name}` not a float: {e}"))
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3
}

// ── Skeleton ────────────────────────────────────────────────────────

#[test]
fn empty_scene_emits_valid_svg_skeleton() {
    let scene = SceneGraph::with_root();
    let svg = render(&scene);
    let doc = parse(&svg);

    let root = doc.root_element();
    assert_eq!(root.tag_name().name(), "svg");
    assert_eq!(root.attribute("width"), Some("200"));
    assert_eq!(root.attribute("height"), Some("100"));
    assert_eq!(root.attribute("viewBox"), Some("0 0 200 100"));
    assert_eq!(root.attribute("role"), Some("img"));

    // The renderer always emits a white background rect; with no marks that
    // should be the *only* child element.
    let children: Vec<_> = root.children().filter(|n| n.is_element()).collect();
    assert_eq!(
        children.len(),
        1,
        "empty scene should emit only the background rect, got {children:?}"
    );
    assert_eq!(children[0].tag_name().name(), "rect");
    assert_eq!(children[0].attribute("fill"), Some("white"));
}

// ── Mark geometry ───────────────────────────────────────────────────

#[test]
fn rect_mark_emits_rect_with_correct_geometry() {
    let (mut scene, root) = scene_with_root();
    scene.insert_child(
        root,
        Node::with_mark(Mark::Rect(RectMark {
            bounds: BoundingBox::new(10.0, 20.0, 80.0, 40.0),
            fill: FillStyle::Solid(Color::from_hex("#1f77b4").unwrap()),
            stroke: StrokeStyle::default(),
            corner_radius: 0.0,
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);

    // Two rects total: background + ours. Find the non-white one.
    let rects: Vec<_> = elements(&doc)
        .filter(|n| n.tag_name().name() == "rect" && n.attribute("fill") != Some("white"))
        .collect();
    assert_eq!(rects.len(), 1, "expected exactly one foreground rect");
    let r = rects[0];

    assert!(approx(attr_f32(r, "x"), 10.0));
    assert!(approx(attr_f32(r, "y"), 20.0));
    assert!(approx(attr_f32(r, "width"), 80.0));
    assert!(approx(attr_f32(r, "height"), 40.0));
    assert!(r.attribute("fill").unwrap().starts_with("rgb"));
    // Default stroke is black width=1, so stroke attrs must appear.
    assert!(r.attribute("stroke").is_some());
    assert_eq!(r.attribute("stroke-width"), Some("1"));
    // No corner radius -> no rx attr.
    assert_eq!(r.attribute("rx"), None);
}

#[test]
fn rect_with_corner_radius_emits_rx() {
    let (mut scene, root) = scene_with_root();
    scene.insert_child(
        root,
        Node::with_mark(Mark::Rect(RectMark {
            bounds: BoundingBox::new(0.0, 0.0, 10.0, 10.0),
            fill: FillStyle::None,
            stroke: StrokeStyle::default(),
            corner_radius: 4.0,
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);

    let r = elements(&doc)
        .find(|n| n.tag_name().name() == "rect" && n.attribute("fill") == Some("none"))
        .expect("foreground rect with corner radius should exist");
    assert!(approx(attr_f32(r, "rx"), 4.0));
}

#[test]
fn marker_shapes_map_to_distinct_elements() {
    // Each shape should produce its own kind of element (or polygon).
    let cases = [
        (MarkerShape::Circle, "circle"),
        (MarkerShape::Square, "rect"),
        (MarkerShape::Diamond, "polygon"),
        (MarkerShape::TriangleUp, "polygon"),
    ];

    for (shape, expected_tag) in cases {
        let (mut scene, root) = scene_with_root();
        scene.insert_child(
            root,
            Node::with_mark(Mark::Point(PointMark {
                center: [50.0, 25.0],
                size: 8.0,
                shape,
                fill: FillStyle::Solid(Color::RED),
                stroke: StrokeStyle::default(),
            })),
        );
        let svg = render(&scene);
        let doc = parse(&svg);

        // Foreground is the one matching the expected tag (the bg is "rect" with fill="white";
        // for Square the count check differentiates instead).
        let foreground = match shape {
            MarkerShape::Square => elements(&doc)
                .filter(|n| n.tag_name().name() == "rect" && n.attribute("fill") != Some("white"))
                .count(),
            _ => count_tag(&doc, expected_tag),
        };
        assert!(
            foreground >= 1,
            "shape {shape:?} should emit at least one <{expected_tag}>, got svg:\n{svg}"
        );
    }
}

#[test]
fn circle_marker_uses_correct_radius_and_center() {
    let (mut scene, root) = scene_with_root();
    scene.insert_child(
        root,
        Node::with_mark(Mark::Point(PointMark {
            center: [120.0, 30.0],
            size: 10.0, // -> radius 5
            shape: MarkerShape::Circle,
            fill: FillStyle::Solid(Color::RED),
            stroke: StrokeStyle::default(),
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);

    let c = first_with_tag(&doc, "circle").expect("circle must be emitted");
    assert!(approx(attr_f32(c, "cx"), 120.0));
    assert!(approx(attr_f32(c, "cy"), 30.0));
    assert!(approx(attr_f32(c, "r"), 5.0));
}

#[test]
fn line_mark_emits_polyline_with_all_points() {
    let (mut scene, root) = scene_with_root();
    let pts = vec![[0.0, 0.0], [50.0, 25.0], [100.0, 50.0], [150.0, 25.0]];
    scene.insert_child(
        root,
        Node::with_mark(Mark::Line(LineMark {
            points: pts.clone(),
            stroke: StrokeStyle::solid(Color::BLUE, 2.0),
            interpolation: Interpolation::Linear,
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);

    let pl = first_with_tag(&doc, "polyline").expect("polyline must be emitted");
    let points_attr = pl.attribute("points").expect("polyline missing `points`");
    let pair_count = points_attr.split_whitespace().count();
    assert_eq!(pair_count, pts.len(), "polyline should encode every input point");
    assert_eq!(pl.attribute("fill"), Some("none"));
    assert_eq!(pl.attribute("stroke-width"), Some("2"));
}

#[test]
fn line_with_fewer_than_two_points_is_skipped() {
    let (mut scene, root) = scene_with_root();
    scene.insert_child(
        root,
        Node::with_mark(Mark::Line(LineMark {
            points: vec![[5.0, 5.0]],
            stroke: StrokeStyle::default(),
            interpolation: Interpolation::Linear,
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);
    assert_eq!(count_tag(&doc, "polyline"), 0);
}

#[test]
fn rule_mark_emits_one_line_per_segment() {
    let (mut scene, root) = scene_with_root();
    scene.insert_child(
        root,
        Node::with_mark(Mark::Rule(RuleMark {
            segments: vec![
                ([0.0, 10.0], [200.0, 10.0]),
                ([0.0, 50.0], [200.0, 50.0]),
                ([0.0, 90.0], [200.0, 90.0]),
            ],
            stroke: StrokeStyle::solid(Color::BLACK, 1.0),
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);
    assert_eq!(count_tag(&doc, "line"), 3);
}

// ── Text ────────────────────────────────────────────────────────────

#[test]
fn text_anchor_attribute_reflects_anchor_kind() {
    let (mut scene, root) = scene_with_root();
    for (i, anchor) in [TextAnchor::Start, TextAnchor::Middle, TextAnchor::End]
        .into_iter()
        .enumerate()
    {
        scene.insert_child(
            root,
            Node::with_mark(Mark::Text(TextMark {
                position: [10.0, (i as f32 + 1.0) * 20.0],
                text: format!("{anchor:?}"),
                font: FontStyle::default(),
                fill: FillStyle::Solid(Color::BLACK),
                angle: 0.0,
                anchor,
            })),
        );
    }
    let svg = render(&scene);
    let doc = parse(&svg);

    let texts: Vec<_> = elements(&doc)
        .filter(|n| n.tag_name().name() == "text")
        .collect();
    assert_eq!(texts.len(), 3);
    assert_eq!(texts[0].attribute("text-anchor"), None); // Start = default
    assert_eq!(texts[1].attribute("text-anchor"), Some("middle"));
    assert_eq!(texts[2].attribute("text-anchor"), Some("end"));
}

#[test]
fn text_xml_special_chars_round_trip_through_parser() {
    let (mut scene, root) = scene_with_root();
    let raw = "x < y & z > \"w\"";
    scene.insert_child(
        root,
        Node::with_mark(Mark::Text(TextMark {
            position: [10.0, 20.0],
            text: raw.into(),
            font: FontStyle::default(),
            fill: FillStyle::Solid(Color::BLACK),
            angle: 0.0,
            anchor: TextAnchor::Start,
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);

    let t = first_with_tag(&doc, "text").expect("text element must exist");
    // The XML parser should give us back the original string after decoding entities.
    assert_eq!(t.text(), Some(raw));
}

// ── Transforms ──────────────────────────────────────────────────────

#[test]
fn parent_transform_propagates_to_child_marks() {
    // A container with translate(100, 25) should shift a point at [0, 0]
    // to [100, 25] in the rendered output.
    let (mut scene, root) = scene_with_root();
    let group = scene.insert_child(
        root,
        Node::container().transform(Affine2D::translate(100.0, 25.0)),
    );
    scene.insert_child(
        group,
        Node::with_mark(Mark::Point(PointMark {
            center: [0.0, 0.0],
            size: 4.0,
            shape: MarkerShape::Circle,
            fill: FillStyle::Solid(Color::RED),
            stroke: StrokeStyle::default(),
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);

    let c = first_with_tag(&doc, "circle").expect("circle must be emitted");
    assert!(approx(attr_f32(c, "cx"), 100.0), "got cx={}", attr_f32(c, "cx"));
    assert!(approx(attr_f32(c, "cy"), 25.0), "got cy={}", attr_f32(c, "cy"));
}

// ── Stroke styling ──────────────────────────────────────────────────

#[test]
fn stroke_dash_pattern_emits_dasharray() {
    let (mut scene, root) = scene_with_root();
    let stroke = StrokeStyle {
        color: Color::BLACK,
        width: 1.5,
        dash: vec![4.0, 2.0],
        dash_offset: 0.0,
        line_cap: LineCap::Round,
        line_join: esoc_scene::style::LineJoin::Miter,
    };
    scene.insert_child(
        root,
        Node::with_mark(Mark::Rule(RuleMark {
            segments: vec![([0.0, 50.0], [200.0, 50.0])],
            stroke,
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);

    let line = first_with_tag(&doc, "line").expect("line element must exist");
    assert_eq!(line.attribute("stroke-dasharray"), Some("4,2"));
    assert_eq!(line.attribute("stroke-linecap"), Some("round"));
}

// ── Accessibility metadata ──────────────────────────────────────────

#[test]
fn title_and_description_render_as_first_children() {
    let scene = SceneGraph::with_root();
    let svg = render_scene_svg_with_metadata(&scene, W, H, Some("Chart title"), Some("Long desc"))
        .expect("render");
    let doc = parse(&svg);

    let root = doc.root_element();
    let element_children: Vec<_> = root.children().filter(|n| n.is_element()).collect();
    // <title>, <desc>, <rect bg> in that order.
    assert!(element_children.len() >= 3);
    assert_eq!(element_children[0].tag_name().name(), "title");
    assert_eq!(element_children[0].text(), Some("Chart title"));
    assert_eq!(element_children[1].tag_name().name(), "desc");
    assert_eq!(element_children[1].text(), Some("Long desc"));
}

#[test]
fn metadata_text_is_xml_escaped() {
    let scene = SceneGraph::with_root();
    let svg =
        render_scene_svg_with_metadata(&scene, W, H, Some("a & b < c"), None).expect("render");
    let doc = parse(&svg);

    let title = first_with_tag(&doc, "title").expect("title element");
    assert_eq!(title.text(), Some("a & b < c"));
}

// ── Rendering order ─────────────────────────────────────────────────

#[test]
fn marks_render_in_insertion_order() {
    let (mut scene, root) = scene_with_root();
    // Insert rect first, then circle. SVG paints in document order, so the
    // circle should appear *after* the rect in the document, drawn on top.
    scene.insert_child(
        root,
        Node::with_mark(Mark::Rect(RectMark {
            bounds: BoundingBox::new(0.0, 0.0, 200.0, 100.0),
            fill: FillStyle::Solid(Color::BLUE),
            stroke: StrokeStyle::default(),
            corner_radius: 0.0,
        })),
    );
    scene.insert_child(
        root,
        Node::with_mark(Mark::Point(PointMark {
            center: [100.0, 50.0],
            size: 20.0,
            shape: MarkerShape::Circle,
            fill: FillStyle::Solid(Color::RED),
            stroke: StrokeStyle::default(),
        })),
    );
    let svg = render(&scene);
    let doc = parse(&svg);

    // Sequence of element tags after the background rect.
    let tags: Vec<_> = doc
        .root_element()
        .children()
        .filter(|n| n.is_element())
        .map(|n| n.tag_name().name().to_string())
        .collect();
    // tags = ["rect" (bg), "rect" (foreground), "circle"]
    assert_eq!(tags.len(), 3, "got tags {tags:?}");
    assert_eq!(tags[1], "rect");
    assert_eq!(tags[2], "circle");
}
