// SPDX-License-Identifier: MIT OR Apache-2.0
//! Comprehensive chart review: every chart type with variations to verify the audit fixes.

use esoc_chart::express::*;
use esoc_chart::grammar::annotation::Annotation;
use esoc_chart::grammar::chart::Chart;
use esoc_chart::grammar::coord::CoordSystem;
use esoc_chart::grammar::facet::{Facet, FacetScales};
use esoc_chart::grammar::layer::{Layer, MarkType};
use esoc_chart::grammar::position::Position;
use esoc_chart::grammar::stat::Stat;
use esoc_chart::new_theme::NewTheme;

fn main() -> esoc_chart::error::Result<()> {
    let mut sections: Vec<(&str, String)> = Vec::new();

    // ── Simple RNG for reproducible data ─────────────────────────────
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-15);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }
    let mut rng = Rng(42);

    // ═══════════════════════════════════════════════════════════════════
    // SCATTER PLOTS
    // ═══════════════════════════════════════════════════════════════════

    // Basic scatter
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.3, 4.1, 3.0, 5.8, 4.9, 7.2, 6.5, 8.1];
        let svg = scatter(&x, &y)
            .title("Basic Scatter")
            .x_label("X")
            .y_label("Y")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Scatter – Basic", svg));
    }

    // Scatter with categories + legend
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let y = vec![2.3, 4.1, 3.0, 5.8, 4.9, 7.2, 6.5, 8.1, 3.5, 5.2, 6.8, 7.9];
        let cats = vec!["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"];
        let svg = scatter(&x, &y)
            .color_by(&cats)
            .title("Scatter – 3 Categories")
            .x_label("Feature 1")
            .y_label("Feature 2")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Scatter – Categories", svg));
    }

    // Dense scatter (auto opacity)
    {
        let n = 400;
        let x: Vec<f64> = (0..n).map(|_| rng.normal() * 3.0 + 5.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 0.8 + rng.normal() * 2.0).collect();
        let svg = scatter(&x, &y)
            .title("Dense Scatter (n=400, auto-opacity)")
            .x_label("Feature A")
            .y_label("Feature B")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Scatter – Dense", svg));
    }

    // Single point scatter (edge case)
    {
        let svg = scatter(&[5.0], &[10.0])
            .title("Single Point")
            .size(400.0, 300.0)
            .to_svg()?;
        sections.push(("Scatter – Single Point", svg));
    }

    // Scatter with description (accessibility)
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 3.0, 5.0, 4.5];
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Point).with_x(x).with_y(y))
            .title("Accessible Chart")
            .description("A scatter plot showing 5 data points with an upward trend")
            .size(500.0, 350.0);
        let svg = chart.to_svg()?;
        // Verify SVG has role="img", <title>, <desc>
        assert!(svg.contains(r#"role="img""#));
        assert!(svg.contains("<title>"));
        assert!(svg.contains("<desc>"));
        sections.push(("Scatter – Accessibility", svg));
    }

    // ═══════════════════════════════════════════════════════════════════
    // LINE CHARTS
    // ═══════════════════════════════════════════════════════════════════

    // Basic line
    {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.8).sin() * 3.0 + v).collect();
        let svg = line(&x, &y)
            .title("Line Chart")
            .x_label("Time")
            .y_label("Value")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Line – Basic", svg));
    }

    // Multi-line with legend
    {
        let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.5).collect();
        let y1: Vec<f64> = x.iter().map(|&v| (v * 0.4).sin() * 5.0 + 10.0).collect();
        let y2: Vec<f64> = x.iter().map(|&v| (v * 0.4).cos() * 4.0 + 12.0).collect();
        let y3: Vec<f64> = x.iter().map(|&v| v * 0.5 + 5.0).collect();

        let chart = Chart::new()
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(y1).with_label("sin"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(y2).with_label("cos"))
            .layer(Layer::new(MarkType::Line).with_x(x).with_y(y3).with_label("linear"))
            .title("Multi-Line with Legend")
            .x_label("Time")
            .y_label("Signal")
            .size(500.0, 350.0);
        sections.push(("Line – Multi-series", chart.to_svg()?));
    }

    // LOESS smooth overlay
    {
        let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.25).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.5).sin() * 3.0 + rng.normal() * 0.8).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Point).with_x(x.clone()).with_y(y.clone()).with_label("Raw"))
            .layer(
                Layer::new(MarkType::Line)
                    .with_x(x)
                    .with_y(y)
                    .stat(Stat::Smooth { bandwidth: 0.3 })
                    .with_label("LOESS"),
            )
            .title("LOESS Smoothing")
            .x_label("X")
            .y_label("Y")
            .size(500.0, 350.0);
        sections.push(("Line – LOESS Overlay", chart.to_svg()?));
    }

    // ═══════════════════════════════════════════════════════════════════
    // BAR CHARTS
    // ═══════════════════════════════════════════════════════════════════

    // Basic bar (no legend expected)
    {
        let cats = vec!["Rust", "Python", "Go", "Java", "C++"];
        let vals = vec![42.0, 35.0, 28.0, 22.0, 18.0];
        let svg = bar(&cats, &vals)
            .title("Bar Chart (no legend)")
            .x_label("Language")
            .y_label("Score")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Bar – Basic", svg));
    }

    // Bar with many categories (label rotation)
    {
        let cats: Vec<String> = (0..15).map(|i| format!("Category {}", i + 1)).collect();
        let vals: Vec<f64> = (0..15).map(|i| (i as f64 * 3.7 + 5.0) % 30.0 + 5.0).collect();
        let cat_refs: Vec<&str> = cats.iter().map(|s| s.as_str()).collect();
        let svg = bar(&cat_refs, &vals)
            .title("Bar – Label Rotation")
            .x_label("Category")
            .y_label("Value")
            .size(600.0, 350.0)
            .to_svg()?;
        sections.push(("Bar – Rotated Labels", svg));
    }

    // Horizontal bar (flipped)
    {
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0, 4.0])
                    .with_y(vec![42.0, 35.0, 28.0, 22.0, 18.0])
                    .with_categories(vec![
                        "Rust".into(), "Python".into(), "Go".into(),
                        "Java".into(), "C++".into(),
                    ]),
            )
            .coord(CoordSystem::Flipped)
            .title("Horizontal Bars")
            .x_label("Score")
            .y_label("Language")
            .size(500.0, 350.0);
        sections.push(("Bar – Horizontal", chart.to_svg()?));
    }

    // Grouped bar
    {
        let cats = vec!["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"];
        let groups = vec!["2023", "2023", "2023", "2023", "2024", "2024", "2024", "2024", "2025", "2025", "2025", "2025"];
        let vals = vec![10.0, 14.0, 18.0, 12.0, 12.0, 18.0, 22.0, 15.0, 14.0, 20.0, 28.0, 19.0];
        let svg = grouped_bar(&cats, &groups, &vals)
            .title("Grouped Bar – 3 Series")
            .x_label("Quarter")
            .y_label("Revenue ($M)")
            .size(550.0, 350.0)
            .to_svg()?;
        sections.push(("Bar – Grouped", svg));
    }

    // Stacked bar
    {
        let cats = vec!["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"];
        let groups = vec!["Product", "Product", "Product", "Product", "Service", "Service", "Service", "Service"];
        let vals = vec![10.0, 15.0, 20.0, 18.0, 5.0, 8.0, 12.0, 10.0];
        let svg = stacked_bar(&cats, &groups, &vals)
            .title("Stacked Bar")
            .x_label("Quarter")
            .y_label("Revenue ($M)")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Bar – Stacked", svg));
    }

    // Stacked bar with sparse groups (tests key-based stacking fix)
    {
        // Group A only has Q1,Q2; Group B has Q2,Q3,Q4 — sparse overlap
        let cats = vec!["Q1", "Q2", "Q2", "Q3", "Q4"];
        let groups = vec!["Alpha", "Alpha", "Beta", "Beta", "Beta"];
        let vals = vec![10.0, 20.0, 15.0, 25.0, 12.0];
        let svg = stacked_bar(&cats, &groups, &vals)
            .title("Stacked – Sparse Groups")
            .x_label("Quarter")
            .y_label("Value")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Bar – Sparse Stacked", svg));
    }

    // Stacked bar with mixed positive/negative (diverging stack)
    {
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0])
                    .with_y(vec![10.0, 15.0, 12.0, 18.0])
                    .with_label("Revenue")
                    .position(Position::Stack),
            )
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0])
                    .with_y(vec![-4.0, -8.0, -5.0, -6.0])
                    .with_label("Costs")
                    .position(Position::Stack),
            )
            .title("Diverging Stack (+/-)")
            .x_label("Period")
            .y_label("Net Change")
            .size(500.0, 350.0);
        sections.push(("Bar – Diverging Stack", chart.to_svg()?));
    }

    // ═══════════════════════════════════════════════════════════════════
    // HISTOGRAM
    // ═══════════════════════════════════════════════════════════════════

    {
        let data: Vec<f64> = (0..500).map(|_| rng.normal() * 2.0 + 10.0).collect();
        let svg = histogram(&data)
            .bins(25)
            .title("Histogram (n=500, 25 bins)")
            .x_label("Value")
            .y_label("Count")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Histogram", svg));
    }

    // ═══════════════════════════════════════════════════════════════════
    // AREA CHARTS
    // ═══════════════════════════════════════════════════════════════════

    {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.3).sin().abs() * 20.0 + 5.0).collect();
        let svg = area(&x, &y)
            .title("Area Chart")
            .x_label("Day")
            .y_label("Traffic")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Area – Basic", svg));
    }

    // Stacked area
    {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y1: Vec<f64> = x.iter().map(|&v| (v * 0.3).sin().abs() * 10.0 + 5.0).collect();
        let y2: Vec<f64> = x.iter().map(|&v| (v * 0.2).cos().abs() * 8.0 + 3.0).collect();
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Area)
                    .with_x(x.clone()).with_y(y1).with_label("Direct")
                    .position(Position::Stack),
            )
            .layer(
                Layer::new(MarkType::Area)
                    .with_x(x).with_y(y2).with_label("Referral")
                    .position(Position::Stack),
            )
            .title("Stacked Area")
            .x_label("Week")
            .y_label("Visits")
            .size(500.0, 350.0);
        sections.push(("Area – Stacked", chart.to_svg()?));
    }

    // ═══════════════════════════════════════════════════════════════════
    // PIE / DONUT
    // ═══════════════════════════════════════════════════════════════════

    {
        let vals = vec![35.0, 25.0, 20.0, 15.0, 5.0];
        let labels = vec!["Chrome", "Firefox", "Safari", "Edge", "Other"];
        let svg = pie(&vals, &labels)
            .title("Pie Chart")
            .size(400.0, 400.0)
            .to_svg()?;
        sections.push(("Pie", svg));
    }

    {
        let vals = vec![60.0, 25.0, 15.0];
        let labels = vec!["Pass", "Warn", "Fail"];
        let svg = pie(&vals, &labels)
            .donut(0.55)
            .title("Donut Chart")
            .size(400.0, 400.0)
            .to_svg()?;
        sections.push(("Donut", svg));
    }

    // ═══════════════════════════════════════════════════════════════════
    // BOX PLOT
    // ═══════════════════════════════════════════════════════════════════

    {
        let mut cats = Vec::new();
        let mut vals = Vec::new();
        for (label, base, spread) in &[("Control", 50.0, 15.0), ("Drug A", 65.0, 10.0), ("Drug B", 70.0, 20.0)] {
            for _ in 0..40 {
                vals.push(base + (rng.uniform() - 0.5) * spread * 2.0);
                cats.push(*label);
            }
            // Add outlier
            vals.push(base + spread * 4.0);
            cats.push(*label);
        }
        let svg = boxplot(&cats, &vals)
            .title("Box Plot with Outliers")
            .x_label("Treatment")
            .y_label("Response")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Box Plot", svg));
    }

    // ═══════════════════════════════════════════════════════════════════
    // HEATMAPS
    // ═══════════════════════════════════════════════════════════════════

    // Basic heatmap with annotations + gradient legend
    {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![2.0, 8.0, 6.0, 4.0, 2.0],
            vec![3.0, 3.0, 9.0, 3.0, 3.0],
        ];
        let svg = heatmap(data)
            .annotate()
            .row_labels(vec!["A".into(), "B".into(), "C".into(), "D".into()])
            .col_labels(vec!["v1".into(), "v2".into(), "v3".into(), "v4".into(), "v5".into()])
            .title("Heatmap (annotated + gradient legend)")
            .x_label("Variable")
            .y_label("Group")
            .size(500.0, 400.0)
            .to_svg()?;
        sections.push(("Heatmap – Annotated", svg));
    }

    // Confusion matrix
    {
        let data = vec![
            vec![45.0, 3.0, 2.0],
            vec![1.0, 40.0, 5.0],
            vec![0.0, 4.0, 50.0],
        ];
        let svg = heatmap(data)
            .annotate()
            .row_labels(vec!["Cat".into(), "Dog".into(), "Bird".into()])
            .col_labels(vec!["Cat".into(), "Dog".into(), "Bird".into()])
            .title("Confusion Matrix")
            .x_label("Predicted")
            .y_label("Actual")
            .size(400.0, 400.0)
            .to_svg()?;
        sections.push(("Heatmap – Confusion Matrix", svg));
    }

    // Heatmap with custom color scale
    {
        let data = vec![
            vec![0.0, 0.3, 0.7, 1.0],
            vec![0.2, 0.5, 0.8, 0.9],
            vec![0.1, 0.4, 0.6, 0.95],
        ];
        let mut theme = NewTheme::light();
        theme.color_scale = Some(esoc_color::ColorScale::rdbu());
        let svg = heatmap(data)
            .annotate()
            .title("Heatmap – RdBu Color Scale")
            .theme(theme)
            .size(400.0, 350.0)
            .to_svg()?;
        sections.push(("Heatmap – Custom Color Scale", svg));
    }

    // ═══════════════════════════════════════════════════════════════════
    // FACETED CHARTS (small multiples)
    // ═══════════════════════════════════════════════════════════════════

    // Faceted scatter
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut facets = Vec::new();
        for panel in &["East", "West", "North", "South"] {
            for _ in 0..25 {
                x.push(rng.uniform() * 10.0);
                y.push(rng.uniform() * 10.0);
                facets.push(*panel);
            }
        }
        let svg = scatter(&x, &y)
            .facet_wrap(&facets, 2)
            .title("Faceted Scatter (2 cols)")
            .x_label("X")
            .y_label("Y")
            .size(550.0, 450.0)
            .to_svg()?;
        sections.push(("Facet – Scatter", svg));
    }

    // Faceted scatter with categories + legend (tests faceted legend fix)
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut cats = Vec::new();
        let mut facets = Vec::new();
        for panel in &["Male", "Female"] {
            for cat in &["Young", "Old"] {
                for _ in 0..12 {
                    x.push(rng.uniform() * 10.0);
                    y.push(rng.uniform() * 10.0);
                    cats.push(*cat);
                    facets.push(*panel);
                }
            }
        }
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Point)
                    .with_x(x)
                    .with_y(y)
                    .with_categories(cats.iter().map(|s| s.to_string()).collect())
                    .with_facet_values(facets.iter().map(|s| s.to_string()).collect()),
            )
            .facet(Facet::Wrap { ncol: 2 })
            .title("Faceted + Categories + Legend")
            .x_label("X")
            .y_label("Y")
            .size(550.0, 350.0);
        sections.push(("Facet – With Legend", chart.to_svg()?));
    }

    // Faceted with FreeY scales (tests FreeY fix: shared X, free Y)
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut facets = Vec::new();
        // Panel A: small values; Panel B: large values
        for _ in 0..20 {
            x.push(rng.uniform() * 10.0);
            y.push(rng.uniform() * 5.0);
            facets.push("Small Range");
        }
        for _ in 0..20 {
            x.push(rng.uniform() * 10.0);
            y.push(rng.uniform() * 500.0);
            facets.push("Large Range");
        }
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Point)
                    .with_x(x)
                    .with_y(y)
                    .with_facet_values(facets.iter().map(|s| s.to_string()).collect()),
            )
            .facet(Facet::Wrap { ncol: 2 })
            .facet_scales(FacetScales::FreeY)
            .title("FreeY Scales (shared X, free Y)")
            .x_label("X")
            .y_label("Y")
            .size(550.0, 350.0);
        sections.push(("Facet – FreeY", chart.to_svg()?));
    }

    // ═══════════════════════════════════════════════════════════════════
    // ANNOTATIONS
    // ═══════════════════════════════════════════════════════════════════

    {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v * 1.5 + rng.normal() * 3.0).collect();
        let chart = scatter(&x, &y)
            .title("Annotations Demo")
            .x_label("X")
            .y_label("Y")
            .size(500.0, 350.0)
            .build()
            .annotate(Annotation::hline(15.0).with_label("Target"))
            .annotate(Annotation::vline(10.0).with_label("Midpoint"))
            .annotate(Annotation::band(10.0, 20.0).with_label("Peak zone"));
        sections.push(("Annotations", chart.to_svg()?));
    }

    // ═══════════════════════════════════════════════════════════════════
    // SUBTITLE, CAPTION, LINE+SCATTER OVERLAY
    // ═══════════════════════════════════════════════════════════════════

    {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_data: Vec<f64> = vec![2.1, 3.8, 3.2, 5.5, 4.8, 7.1, 6.3, 8.0, 7.5, 9.2];
        let y_trend: Vec<f64> = x.iter().map(|&v| v * 0.8 + 2.0).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Point).with_x(x.clone()).with_y(y_data).with_label("Data"))
            .layer(Layer::new(MarkType::Line).with_x(x).with_y(y_trend).with_label("Trend"))
            .title("Revenue Trend")
            .subtitle("H1 2026 with linear fit")
            .caption("Source: internal CRM data")
            .x_label("Month")
            .y_label("Revenue ($K)")
            .size(500.0, 380.0);
        sections.push(("Line + Scatter + Subtitle/Caption", chart.to_svg()?));
    }

    // ═══════════════════════════════════════════════════════════════════
    // DARK THEME
    // ═══════════════════════════════════════════════════════════════════

    {
        let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.3).sin() * 5.0 + 8.0).collect();
        let svg = line(&x, &y)
            .title("Dark Theme")
            .x_label("Time")
            .y_label("Value")
            .theme(NewTheme::dark())
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Theme – Dark", svg));
    }

    // ═══════════════════════════════════════════════════════════════════
    // PUBLICATION THEME
    // ═══════════════════════════════════════════════════════════════════

    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.3, 4.1, 3.0, 5.8, 4.9, 7.2, 6.5, 8.1];
        let svg = scatter(&x, &y)
            .title("Publication Theme (no grid, serif)")
            .x_label("X")
            .y_label("Y")
            .theme(NewTheme::publication())
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Theme – Publication", svg));
    }

    // ═══════════════════════════════════════════════════════════════════
    // BUILD HTML
    // ═══════════════════════════════════════════════════════════════════

    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>esoc-chart Review — All Chart Types</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #f0f0f4; color: #333; }
  header { background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 2.5rem 2rem; text-align: center; }
  header h1 { font-size: 2rem; font-weight: 300; letter-spacing: 0.02em; }
  header p { margin-top: 0.5rem; opacity: 0.7; font-size: 0.95rem; }
  .stats { display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; }
  .stats span { background: rgba(255,255,255,0.15); padding: 0.3rem 0.8rem; border-radius: 4px; font-size: 0.85rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; padding: 2rem; max-width: 1600px; margin: 0 auto; }
  .card { background: white; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); overflow: hidden; transition: box-shadow 0.2s; }
  .card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.12); }
  .card h2 { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #555; padding: 1rem 1.5rem 0; }
  .card .chart-wrap { padding: 0.5rem 1rem 0.75rem; }
  .card svg { display: block; width: 100%; height: auto; }
  .card.dark-bg .chart-wrap { background: #1e1e2e; border-radius: 0 0 8px 8px; }
  .feedback { padding: 0 1rem 1rem; }
  .feedback textarea { width: 100%; min-height: 50px; border: 1px solid #e0e0e0; border-radius: 4px; padding: 0.5rem; font-family: inherit; font-size: 0.82rem; resize: vertical; }
  .feedback textarea:focus { outline: none; border-color: #1a1a2e; }
  .feedback .status { font-size: 0.72rem; color: #aaa; margin-top: 0.2rem; }
  .actions { padding: 1.5rem 2rem; text-align: center; }
  .actions button { background: #1a1a2e; color: white; border: none; border-radius: 4px; padding: 0.6rem 1.5rem; font-size: 0.9rem; cursor: pointer; margin: 0 0.5rem; }
  .actions button:hover { background: #2a2a4e; }
</style>
<script>
  const feedback = {};
  function loadFeedback() {
    try { Object.assign(feedback, JSON.parse(localStorage.getItem('chart_review_feedback') || '{}')); } catch {}
    document.querySelectorAll('.feedback textarea').forEach(ta => {
      const key = ta.dataset.chart;
      if (feedback[key]) ta.value = feedback[key];
    });
  }
  function saveFeedback(key, value) {
    feedback[key] = value;
    localStorage.setItem('chart_review_feedback', JSON.stringify(feedback));
  }
  function exportFeedback() {
    const blob = new Blob([JSON.stringify(feedback, null, 2)], {type: 'application/json'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = 'chart_review_feedback.json'; a.click();
  }
  window.addEventListener('DOMContentLoaded', loadFeedback);
</script>
</head>
<body>
<header>
  <h1>esoc-chart Review</h1>
  <p>Comprehensive sample of all chart types &amp; variations after audit fixes</p>
  <div class="stats">
"#,
    );

    html.push_str(&format!(
        "    <span>{} charts</span>\n",
        sections.len()
    ));
    html.push_str("    <span>6 phases of fixes</span>\n");
    html.push_str("    <span>23 new tests</span>\n");
    html.push_str("  </div>\n</header>\n<div class=\"grid\">\n");

    for (title, svg) in &sections {
        let key = title.to_lowercase().replace([' ', '–', '+', '/', '(', ')'], "_").replace("__", "_");
        let dark_class = if title.contains("Dark") { " dark-bg" } else { "" };
        html.push_str(&format!(
            concat!(
                "<div class=\"card{dark_class}\">\n",
                "  <h2>{title}</h2>\n",
                "  <div class=\"chart-wrap\">{svg}</div>\n",
                "  <div class=\"feedback\">\n",
                "    <textarea data-chart=\"{key}\" placeholder=\"Notes on {title}…\" ",
                "oninput=\"saveFeedback('{key}', this.value)\"></textarea>\n",
                "    <div class=\"status\">Auto-saved</div>\n",
                "  </div>\n",
                "</div>\n",
            ),
            title = title,
            svg = svg,
            key = key,
            dark_class = dark_class,
        ));
    }

    html.push_str(concat!(
        "</div>\n",
        "<div class=\"actions\">\n",
        "  <button onclick=\"exportFeedback()\">Export Feedback JSON</button>\n",
        "</div>\n",
        "</body>\n</html>\n",
    ));

    let out_path = "chart_review.html";
    std::fs::write(out_path, &html).expect("failed to write HTML");
    println!("Saved {} ({} charts)", out_path, sections.len());

    Ok(())
}
