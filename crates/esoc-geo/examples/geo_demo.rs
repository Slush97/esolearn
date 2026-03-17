// SPDX-License-Identifier: MIT OR Apache-2.0
//! Demo: render world map + US states map to SVG using esoc-geo projections.
//!
//! Run with:
//!   cargo run -p esoc-geo --features bundled --example geo_demo

use esoc_color::Color;
use esoc_geo::bundled;
use esoc_geo::geometry::{GeoCollection, GeoGeometry, GeoPolygon};
use esoc_geo::projection::{AlbersUsa, NaturalEarth1};
use esoc_geo::Projection;
use esoc_gfx::scene_svg::save_scene_svg;
use esoc_scene::mark::{Mark, PathCommand, PathMark};
use esoc_scene::node::Node;
use esoc_scene::style::{FillStyle, StrokeStyle};
use esoc_scene::SceneGraph;

/// A palette of 10 distinguishable colors (Tableau 10).
const PALETTE: [&str; 10] = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
];

fn color_at(i: usize) -> Color {
    Color::from_hex(PALETTE[i % PALETTE.len()]).unwrap()
}

/// Project a polygon ring into pixel-space path commands.
fn project_ring(
    ring: &[esoc_geo::GeoPoint],
    proj: &dyn Projection,
    scale: f64,
    ox: f64,
    oy: f64,
) -> Vec<PathCommand> {
    let mut cmds = Vec::with_capacity(ring.len() + 1);
    for (i, pt) in ring.iter().enumerate() {
        let (px, py) = proj.project(pt.lon, pt.lat);
        let x = (px * scale + ox) as f32;
        let y = (-py * scale + oy) as f32; // flip Y: lat increases upward
        if i == 0 {
            cmds.push(PathCommand::MoveTo([x, y]));
        } else {
            cmds.push(PathCommand::LineTo([x, y]));
        }
    }
    cmds.push(PathCommand::Close);
    cmds
}

/// Convert a GeoPolygon into PathCommands (exterior + holes).
fn polygon_to_commands(
    poly: &GeoPolygon,
    proj: &dyn Projection,
    scale: f64,
    ox: f64,
    oy: f64,
) -> Vec<PathCommand> {
    let mut cmds = project_ring(&poly.exterior, proj, scale, ox, oy);
    for hole in &poly.holes {
        cmds.extend(project_ring(hole, proj, scale, ox, oy));
    }
    cmds
}

/// Compute projected bounding box of a GeoCollection.
fn projected_bounds(collection: &GeoCollection, proj: &dyn Projection) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut include = |pt: &esoc_geo::GeoPoint| {
        let (px, py) = proj.project(pt.lon, pt.lat);
        min_x = min_x.min(px);
        max_x = max_x.max(px);
        min_y = min_y.min(py);
        max_y = max_y.max(py);
    };

    for feature in &collection.features {
        let polys: &[GeoPolygon] = match &feature.geometry {
            GeoGeometry::Polygon(p) => std::slice::from_ref(p),
            GeoGeometry::MultiPolygon(mp) => &mp.polygons,
            _ => continue,
        };
        for poly in polys {
            for pt in &poly.exterior {
                include(pt);
            }
        }
    }
    (min_x, max_x, min_y, max_y)
}

/// Compute scale and offsets to fit projected bounds into a canvas with padding.
fn fit_to_canvas(
    canvas_w: f64,
    canvas_h: f64,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    padding: f64,
) -> (f64, f64, f64) {
    let data_w = max_x - min_x;
    let data_h = max_y - min_y;
    let avail_w = canvas_w - 2.0 * padding;
    let avail_h = canvas_h - 2.0 * padding;
    let scale = (avail_w / data_w).min(avail_h / data_h);
    let cx = (min_x + max_x) / 2.0;
    let cy = (min_y + max_y) / 2.0;
    // ox, oy such that: pixel_x = px * scale + ox, pixel_y = -py * scale + oy
    let ox = canvas_w / 2.0 - cx * scale;
    let oy = canvas_h / 2.0 + cy * scale;
    (scale, ox, oy)
}

fn main() {
    // ── World map (Natural Earth I) ─────────────────────────────────────
    let width = 960.0_f32;
    let height = 500.0_f32;
    let proj = NaturalEarth1;

    let countries = bundled::world_countries();
    let (min_x, max_x, min_y, max_y) = projected_bounds(countries, &proj);
    let (scale, ox, oy) = fit_to_canvas(
        f64::from(width), f64::from(height), min_x, max_x, min_y, max_y, 10.0,
    );

    let mut scene = SceneGraph::with_root();
    let root = scene.root().unwrap();
    for (i, feature) in countries.features.iter().enumerate() {
        let cmds = match &feature.geometry {
            GeoGeometry::Polygon(poly) => polygon_to_commands(poly, &proj, scale, ox, oy),
            GeoGeometry::MultiPolygon(mp) => {
                let mut all = Vec::new();
                for poly in &mp.polygons {
                    all.extend(polygon_to_commands(poly, &proj, scale, ox, oy));
                }
                all
            }
            _ => continue,
        };

        scene.insert_child(
            root,
            Node::with_mark(Mark::Path(PathMark {
                commands: cmds,
                fill: FillStyle::Solid(color_at(i)),
                stroke: StrokeStyle::solid(Color::WHITE, 0.5),
            })),
        );
    }

    save_scene_svg(&scene, width, height, "world_map.svg").expect("failed to write world_map.svg");
    println!("wrote world_map.svg ({width}×{height}) — {} countries", countries.features.len());

    // ── US states (Albers USA composite) ────────────────────────────────
    let us_w = 960.0_f32;
    let us_h = 600.0_f32;
    let albers = AlbersUsa::new();

    let states = bundled::us_states();

    // Filter to 50 states + DC (exclude territories the composite projection doesn't cover)
    let us_features: Vec<_> = states.features.iter().filter(|f| {
        let name = f.properties.get("NAME").and_then(|v| v.as_str()).unwrap_or("");
        !matches!(name, "Puerto Rico" | "Guam" | "American Samoa"
            | "Commonwealth of the Northern Mariana Islands" | "United States Virgin Islands")
    }).collect();
    let filtered = esoc_geo::GeoCollection {
        features: us_features.iter().map(|f| (*f).clone()).collect(),
    };
    let (us_min_x, us_max_x, us_min_y, us_max_y) = projected_bounds(&filtered, &albers);
    let (us_scale, us_ox, us_oy) = fit_to_canvas(
        f64::from(us_w), f64::from(us_h), us_min_x, us_max_x, us_min_y, us_max_y, 20.0,
    );

    let mut us_scene = SceneGraph::with_root();
    let us_root = us_scene.root().unwrap();
    for (i, feature) in us_features.iter().enumerate() {
        let cmds = match &feature.geometry {
            GeoGeometry::Polygon(poly) => polygon_to_commands(poly, &albers, us_scale, us_ox, us_oy),
            GeoGeometry::MultiPolygon(mp) => {
                let mut all = Vec::new();
                for poly in &mp.polygons {
                    all.extend(polygon_to_commands(poly, &albers, us_scale, us_ox, us_oy));
                }
                all
            }
            _ => continue,
        };

        // Color by index for variety
        let fill = color_at(i);

        us_scene.insert_child(
            us_root,
            Node::with_mark(Mark::Path(PathMark {
                commands: cmds,
                fill: FillStyle::Solid(fill),
                stroke: StrokeStyle::solid(Color::WHITE, 0.8),
            })),
        );
    }

    save_scene_svg(&us_scene, us_w, us_h, "us_states.svg").expect("failed to write us_states.svg");
    println!("wrote us_states.svg ({us_w}×{us_h}) — {} states", us_features.len());
}
