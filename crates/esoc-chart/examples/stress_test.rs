// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stress test: exercises every chart type, option, theme, annotation, facet mode,
//! position, stat, and edge case to surface visual issues.

use esoc_chart::express::*;
use esoc_chart::grammar::annotation::Annotation;
use esoc_chart::grammar::chart::Chart;
use esoc_chart::grammar::coord::CoordSystem;
use esoc_chart::grammar::facet::{Facet, FacetScales};
use esoc_chart::grammar::layer::{Layer, MarkType};
use esoc_chart::grammar::position::Position;
use esoc_chart::grammar::stat::{AggregateFunc, Stat};
use esoc_chart::new_theme::NewTheme;
use esoc_color::{Color, ColorScale, Palette};

fn main() -> esoc_chart::error::Result<()> {
    let mut sections: Vec<(&str, String)> = Vec::new();

    // ── Simple RNG ───────────────────────────────────────────────────
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
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + self.uniform() * (hi - lo)
        }
    }
    let mut rng = Rng(12345);

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 1: SCATTER VARIATIONS
    // ═════════════════════════════════════════════════════════════════

    // 1a. Tiny scatter (2 points)
    {
        let svg = scatter(&[1.0, 2.0], &[3.0, 4.0])
            .title("2-Point Scatter")
            .x_label("X")
            .y_label("Y")
            .size(400.0, 300.0)
            .to_svg()?;
        sections.push(("Scatter — 2 Points", svg));
    }

    // 1b. Large scatter with auto-opacity
    {
        let n = 1000;
        let x: Vec<f64> = (0..n).map(|_| rng.normal() * 5.0 + 50.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 0.6 + rng.normal() * 8.0 + 10.0).collect();
        let svg = scatter(&x, &y)
            .title("Dense Scatter (n=1000)")
            .x_label("Income ($K)")
            .y_label("Spending ($K)")
            .size(500.0, 400.0)
            .to_svg()?;
        sections.push(("Scatter — Dense 1K", svg));
    }

    // 1c. Scatter with 6 categories
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut cats = Vec::new();
        let names = ["Setosa", "Versicolor", "Virginica", "Hybrid-A", "Hybrid-B", "Unknown"];
        for (i, name) in names.iter().enumerate() {
            let cx = 2.0 + i as f64 * 1.5;
            let cy = 3.0 + (i as f64 * 0.7).sin() * 2.0;
            for _ in 0..20 {
                x.push(cx + rng.normal() * 0.5);
                y.push(cy + rng.normal() * 0.5);
                cats.push(*name);
            }
        }
        let svg = scatter(&x, &y)
            .color_by(&cats)
            .title("Scatter — 6 Categories (Iris-like)")
            .x_label("Sepal Length")
            .y_label("Petal Width")
            .size(550.0, 400.0)
            .to_svg()?;
        sections.push(("Scatter — 6 Categories", svg));
    }

    // 1d. Scatter with jitter position
    {
        let x: Vec<f64> = (0..60).map(|i| (i % 3) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 2.0 + rng.normal() * 0.5 + 5.0).collect();
        let cats: Vec<String> = (0..60).map(|i| ["Low", "Med", "High"][i % 3].into()).collect();
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Point)
                    .with_x(x)
                    .with_y(y)
                    .with_categories(cats)
                    .position(Position::Jitter { x_amount: 0.2, y_amount: 0.0 }),
            )
            .title("Scatter — Jittered (strip plot)")
            .x_label("Group")
            .y_label("Value")
            .size(450.0, 350.0);
        sections.push(("Scatter — Jitter", chart.to_svg()?));
    }

    // 1e. Scatter with all annotations
    {
        let x: Vec<f64> = (0..30).map(|_| rng.range(0.0, 100.0)).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 0.8 + rng.normal() * 10.0).collect();
        let chart = scatter(&x, &y)
            .title("Scatter — All Annotation Types")
            .x_label("X")
            .y_label("Y")
            .size(550.0, 400.0)
            .build()
            .annotate(Annotation::hline(40.0).with_label("Mean").with_color(Color::from_hex("#e74c3c").unwrap()))
            .annotate(Annotation::vline(50.0).with_label("Midpoint").with_color(Color::from_hex("#3498db").unwrap()))
            .annotate(Annotation::band(20.0, 60.0).with_label("Normal range").with_color(Color::from_hex("#2ecc71").unwrap().with_alpha(0.12)))
            .annotate(Annotation::text(70.0, 70.0, "Outlier zone"));
        sections.push(("Scatter — All Annotations", chart.to_svg()?));
    }

    // 1f. Scatter with LOESS smooth overlay
    {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.2).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.5).sin() * 4.0 + v * 0.3 + rng.normal() * 1.0).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Point).with_x(x.clone()).with_y(y.clone()).with_label("Raw"))
            .layer(
                Layer::new(MarkType::Line)
                    .with_x(x)
                    .with_y(y)
                    .stat(Stat::Smooth { bandwidth: 0.25 })
                    .with_label("LOESS (bw=0.25)"),
            )
            .title("Scatter + LOESS Smooth")
            .x_label("Time")
            .y_label("Signal")
            .size(500.0, 380.0);
        sections.push(("Scatter — LOESS Overlay", chart.to_svg()?));
    }

    // 1g. Scatter with subtitle, caption, description
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.1, 3.8, 3.2, 5.5, 4.8, 7.1, 6.3, 8.0, 7.5, 9.2];
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Point).with_x(x).with_y(y))
            .title("Monthly Revenue")
            .subtitle("Jan–Oct 2026, all regions")
            .caption("Source: internal CRM")
            .description("Scatter plot of monthly revenue showing upward trend")
            .x_label("Month")
            .y_label("Revenue ($M)")
            .size(500.0, 400.0);
        sections.push(("Scatter — Subtitle+Caption", chart.to_svg()?));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 2: LINE VARIATIONS
    // ═════════════════════════════════════════════════════════════════

    // 2a. Single line
    {
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.5).sin() * 3.0 + v * 0.2).collect();
        let svg = line(&x, &y)
            .title("Smooth Sine Wave")
            .x_label("Time (s)")
            .y_label("Amplitude")
            .size(600.0, 300.0)
            .to_svg()?;
        sections.push(("Line — Single", svg));
    }

    // 2b. Multi-line (5 series)
    {
        let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(x.iter().map(|&v| (v * 0.2).sin() * 10.0 + 20.0).collect()).with_label("Server A"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(x.iter().map(|&v| (v * 0.15).cos() * 8.0 + 25.0).collect()).with_label("Server B"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(x.iter().map(|&v| v * 0.5 + 10.0).collect()).with_label("Server C"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(x.iter().map(|&v| 30.0 - v * 0.3).collect()).with_label("Server D"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(x.iter().map(|&v| ((v * 0.3).sin() * 5.0 + 18.0).max(5.0)).collect()).with_label("Server E"))
            .title("5-Server Latency Dashboard")
            .x_label("Minute")
            .y_label("Latency (ms)")
            .size(650.0, 400.0);
        sections.push(("Line — 5 Series", chart.to_svg()?));
    }

    // 2c. Line + scatter overlay (actual vs predicted)
    {
        let x: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let actual: Vec<f64> = vec![3.0, 5.2, 4.1, 7.8, 6.5, 9.1, 8.3, 10.2, 9.8, 12.1, 11.5, 13.7, 12.9, 15.0, 14.2];
        let predicted: Vec<f64> = x.iter().map(|&v| v * 0.9 + 2.5).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Point).with_x(x.clone()).with_y(actual).with_label("Actual"))
            .layer(Layer::new(MarkType::Line).with_x(x).with_y(predicted).with_label("Predicted"))
            .title("Actual vs Predicted")
            .x_label("Week")
            .y_label("Sales ($K)")
            .size(500.0, 380.0);
        sections.push(("Line — Actual vs Predicted", chart.to_svg()?));
    }

    // 2d. Noisy line with wide LOESS
    {
        let x: Vec<f64> = (0..80).map(|i| i as f64 * 0.125).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.8).sin() * 5.0 + rng.normal() * 2.5).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(y.clone()).with_label("Raw"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(y.clone()).stat(Stat::Smooth { bandwidth: 0.15 }).with_label("bw=0.15"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(y).stat(Stat::Smooth { bandwidth: 0.5 }).with_label("bw=0.5"))
            .title("LOESS Bandwidth Comparison")
            .x_label("X")
            .y_label("Y")
            .size(550.0, 380.0);
        sections.push(("Line — LOESS Bandwidths", chart.to_svg()?));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 3: BAR VARIATIONS
    // ═════════════════════════════════════════════════════════════════

    // 3a. 3 bars — minimal
    {
        let svg = bar(&["A", "B", "C"], &[10.0, 20.0, 15.0])
            .title("3-Bar Minimal")
            .size(350.0, 280.0)
            .to_svg()?;
        sections.push(("Bar — 3 Bars", svg));
    }

    // 3b. Single bar
    {
        let svg = bar(&["Total"], &[42.0])
            .title("Single Bar")
            .y_label("Count")
            .size(300.0, 280.0)
            .to_svg()?;
        sections.push(("Bar — Single", svg));
    }

    // 3c. Many bars (20 categories, label rotation stress)
    {
        let cats: Vec<String> = (0..20).map(|i| format!("Department {}", i + 1)).collect();
        let vals: Vec<f64> = (0..20).map(|i| 10.0 + (i as f64 * 2.7).sin().abs() * 40.0).collect();
        let cat_refs: Vec<&str> = cats.iter().map(|s| s.as_str()).collect();
        let svg = bar(&cat_refs, &vals)
            .title("20 Departments — Label Rotation Stress")
            .x_label("Department")
            .y_label("Budget ($K)")
            .size(700.0, 400.0)
            .to_svg()?;
        sections.push(("Bar — 20 Categories", svg));
    }

    // 3d. Very long category names
    {
        let cats = [
            "Engineering & Product Development",
            "Marketing & Communications",
            "Human Resources Management",
            "Finance & Accounting Dept.",
        ];
        let vals = vec![85.0, 62.0, 45.0, 71.0];
        let svg = bar(&cats, &vals)
            .title("Long Category Names")
            .x_label("Department")
            .y_label("Headcount")
            .size(550.0, 400.0)
            .to_svg()?;
        sections.push(("Bar — Long Names", svg));
    }

    // 3e. Horizontal bar
    {
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                    .with_y(vec![95.0, 87.0, 76.0, 68.0, 55.0, 42.0])
                    .with_categories(vec![
                        "Rust".into(), "Go".into(), "Python".into(),
                        "Java".into(), "C++".into(), "JavaScript".into(),
                    ]),
            )
            .coord(CoordSystem::Flipped)
            .title("Horizontal Bars — Performance Score")
            .x_label("Score")
            .y_label("Language")
            .size(500.0, 380.0);
        sections.push(("Bar — Horizontal", chart.to_svg()?));
    }

    // 3f. Horizontal bar with very long labels
    {
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0])
                    .with_y(vec![88.0, 72.0, 65.0, 91.0])
                    .with_categories(vec![
                        "Customer Satisfaction Index".into(),
                        "Net Promoter Score".into(),
                        "Employee Engagement Rate".into(),
                        "Year-over-Year Revenue Growth".into(),
                    ]),
            )
            .coord(CoordSystem::Flipped)
            .title("Horizontal — Long Y Labels")
            .x_label("Percentage")
            .y_label("KPI")
            .size(550.0, 350.0);
        sections.push(("Bar — Horiz Long Labels", chart.to_svg()?));
    }

    // 3g. Grouped bar (3 groups × 4 quarters)
    {
        let cats = vec!["Q1","Q2","Q3","Q4","Q1","Q2","Q3","Q4","Q1","Q2","Q3","Q4"];
        let groups = vec!["2023","2023","2023","2023","2024","2024","2024","2024","2025","2025","2025","2025"];
        let vals = vec![10.0,14.0,18.0,12.0, 12.0,18.0,22.0,15.0, 14.0,20.0,28.0,19.0];
        let svg = grouped_bar(&cats, &groups, &vals)
            .title("Grouped Bar — 3 Years × 4 Quarters")
            .x_label("Quarter")
            .y_label("Revenue ($M)")
            .size(550.0, 380.0)
            .to_svg()?;
        sections.push(("Bar — Grouped 3×4", svg));
    }

    // 3h. Grouped bar — many groups (5 groups)
    {
        let mut cats = Vec::new();
        let mut groups = Vec::new();
        let mut vals = Vec::new();
        let regions = ["North", "South", "East", "West", "Central"];
        let quarters = ["Q1", "Q2", "Q3"];
        for q in quarters {
            for (i, r) in regions.iter().enumerate() {
                cats.push(q);
                groups.push(*r);
                vals.push(10.0 + i as f64 * 5.0 + rng.range(0.0, 15.0));
            }
        }
        let svg = grouped_bar(&cats, &groups, &vals)
            .title("Grouped Bar — 5 Groups (crowded)")
            .x_label("Quarter")
            .y_label("Sales ($K)")
            .size(600.0, 380.0)
            .to_svg()?;
        sections.push(("Bar — Grouped 5 Groups", svg));
    }

    // 3i. Stacked bar
    {
        let cats = vec!["Jan","Feb","Mar","Apr","May","Jun","Jan","Feb","Mar","Apr","May","Jun"];
        let groups = vec!["Online","Online","Online","Online","Online","Online","Retail","Retail","Retail","Retail","Retail","Retail"];
        let vals = vec![20.0,25.0,30.0,28.0,35.0,40.0, 15.0,12.0,18.0,20.0,16.0,14.0];
        let svg = stacked_bar(&cats, &groups, &vals)
            .title("Stacked Bar — Online vs Retail")
            .x_label("Month")
            .y_label("Revenue ($K)")
            .size(550.0, 380.0)
            .to_svg()?;
        sections.push(("Bar — Stacked", svg));
    }

    // 3j. Stacked bar — 4 groups (tall stack)
    {
        let mut cats = Vec::new();
        let mut groups = Vec::new();
        let mut vals = Vec::new();
        let products = ["Widget", "Gadget", "Doohickey", "Thingamajig"];
        let months = ["Jan", "Feb", "Mar", "Apr"];
        for m in months {
            for p in products {
                cats.push(m);
                groups.push(p);
                vals.push(rng.range(5.0, 30.0));
            }
        }
        let svg = stacked_bar(&cats, &groups, &vals)
            .title("Stacked — 4 Product Lines")
            .x_label("Month")
            .y_label("Units Sold")
            .size(500.0, 380.0)
            .to_svg()?;
        sections.push(("Bar — Stacked 4 Groups", svg));
    }

    // 3k. Diverging stack (positive + negative)
    {
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                    .with_y(vec![15.0, 22.0, 18.0, 25.0, 20.0, 28.0])
                    .with_label("Revenue")
                    .position(Position::Stack),
            )
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                    .with_y(vec![-8.0, -12.0, -6.0, -10.0, -14.0, -9.0])
                    .with_label("Costs")
                    .position(Position::Stack),
            )
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                    .with_y(vec![-3.0, -4.0, -5.0, -3.0, -2.0, -4.0])
                    .with_label("Tax")
                    .position(Position::Stack),
            )
            .title("Diverging Stack — Revenue/Costs/Tax")
            .x_label("Month")
            .y_label("$M")
            .size(550.0, 380.0);
        sections.push(("Bar — Diverging 3 Layers", chart.to_svg()?));
    }

    // 3l. Bars with values near zero
    {
        let svg = bar(&["A", "B", "C", "D", "E"], &[0.1, 0.05, 0.2, 0.01, 0.15])
            .title("Bars — Very Small Values")
            .y_label("Rate")
            .size(450.0, 320.0)
            .to_svg()?;
        sections.push(("Bar — Small Values", svg));
    }

    // 3m. Bars with huge variance
    {
        let svg = bar(&["Tiny", "Medium", "Huge"], &[2.0, 50.0, 980.0])
            .title("Bars — Huge Variance")
            .y_label("Count")
            .size(400.0, 320.0)
            .to_svg()?;
        sections.push(("Bar — Huge Variance", svg));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 4: HISTOGRAM VARIATIONS
    // ═════════════════════════════════════════════════════════════════

    // 4a. Normal distribution
    {
        let data: Vec<f64> = (0..800).map(|_| rng.normal() * 3.0 + 50.0).collect();
        let svg = histogram(&data)
            .bins(30)
            .title("Histogram — Normal (n=800, 30 bins)")
            .x_label("Value")
            .y_label("Frequency")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Histogram — Normal", svg));
    }

    // 4b. Bimodal distribution
    {
        let mut data: Vec<f64> = (0..400).map(|_| rng.normal() * 2.0 + 30.0).collect();
        data.extend((0..400).map(|_| rng.normal() * 2.0 + 45.0));
        let svg = histogram(&data)
            .bins(35)
            .title("Histogram — Bimodal")
            .x_label("Measurement")
            .y_label("Count")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Histogram — Bimodal", svg));
    }

    // 4c. Skewed (exponential-like)
    {
        let data: Vec<f64> = (0..600).map(|_| (-rng.uniform().max(1e-10).ln()) * 5.0).collect();
        let svg = histogram(&data)
            .bins(40)
            .title("Histogram — Skewed (Exponential)")
            .x_label("Wait Time (s)")
            .y_label("Count")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Histogram — Skewed", svg));
    }

    // 4d. Few bins
    {
        let data: Vec<f64> = (0..200).map(|_| rng.normal() * 5.0 + 100.0).collect();
        let svg = histogram(&data)
            .bins(5)
            .title("Histogram — 5 Bins Only")
            .x_label("Score")
            .y_label("Count")
            .size(450.0, 320.0)
            .to_svg()?;
        sections.push(("Histogram — 5 Bins", svg));
    }

    // 4e. Many bins (tiny bars)
    {
        let data: Vec<f64> = (0..500).map(|_| rng.normal() * 2.0).collect();
        let svg = histogram(&data)
            .bins(80)
            .title("Histogram — 80 Bins (sparse)")
            .x_label("X")
            .y_label("Count")
            .size(600.0, 320.0)
            .to_svg()?;
        sections.push(("Histogram — 80 Bins", svg));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 5: AREA VARIATIONS
    // ═════════════════════════════════════════════════════════════════

    // 5a. Basic area
    {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.2).sin().abs() * 30.0 + 5.0 + rng.normal() * 2.0).collect();
        let svg = area(&x, &y)
            .title("Area — Website Traffic")
            .x_label("Day")
            .y_label("Visitors (K)")
            .size(550.0, 350.0)
            .to_svg()?;
        sections.push(("Area — Basic", svg));
    }

    // 5b. Stacked area (2 series)
    {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y1: Vec<f64> = x.iter().map(|&v| (v * 0.15).sin().abs() * 10.0 + 8.0).collect();
        let y2: Vec<f64> = x.iter().map(|&v| (v * 0.2).cos().abs() * 7.0 + 4.0).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Area).with_x(x.clone()).with_y(y1).with_label("Organic").position(Position::Stack))
            .layer(Layer::new(MarkType::Area).with_x(x).with_y(y2).with_label("Paid").position(Position::Stack))
            .title("Stacked Area — Traffic Sources")
            .x_label("Day")
            .y_label("Sessions")
            .size(550.0, 380.0);
        sections.push(("Area — Stacked 2", chart.to_svg()?));
    }

    // 5c. Stacked area (4 series)
    {
        let x: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Area).with_x(x.clone()).with_y(x.iter().map(|&v| (v * 0.3).sin().abs() * 8.0 + 3.0).collect()).with_label("Direct").position(Position::Stack))
            .layer(Layer::new(MarkType::Area).with_x(x.clone()).with_y(x.iter().map(|&v| (v * 0.2).cos().abs() * 6.0 + 2.0).collect()).with_label("Social").position(Position::Stack))
            .layer(Layer::new(MarkType::Area).with_x(x.clone()).with_y(x.iter().map(|&v| (v * 0.15).sin().abs() * 4.0 + 1.5).collect()).with_label("Email").position(Position::Stack))
            .layer(Layer::new(MarkType::Area).with_x(x.clone()).with_y(x.iter().map(|&v| (v * 0.25).cos().abs() * 5.0 + 2.0).collect()).with_label("Referral").position(Position::Stack))
            .title("Stacked Area — 4 Channels")
            .x_label("Week")
            .y_label("Visits (K)")
            .size(600.0, 400.0);
        sections.push(("Area — Stacked 4", chart.to_svg()?));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 6: PIE / DONUT VARIATIONS
    // ═════════════════════════════════════════════════════════════════

    // 6a. Basic pie (5 slices)
    {
        let svg = pie(&[35.0, 25.0, 20.0, 12.0, 8.0], &["Chrome", "Firefox", "Safari", "Edge", "Other"])
            .title("Browser Market Share")
            .size(400.0, 400.0)
            .to_svg()?;
        sections.push(("Pie — 5 Slices", svg));
    }

    // 6b. Pie with many slices (10)
    {
        let vals: Vec<f64> = (0..10).map(|i| 20.0 - i as f64 * 1.5).collect();
        let labels: Vec<String> = (0..10).map(|i| format!("Slice {}", i + 1)).collect();
        let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        let svg = pie(&vals, &label_refs)
            .title("Pie — 10 Slices")
            .size(450.0, 450.0)
            .to_svg()?;
        sections.push(("Pie — 10 Slices", svg));
    }

    // 6c. Pie with one dominant slice
    {
        let svg = pie(&[90.0, 5.0, 3.0, 2.0], &["Dominant", "Small-A", "Small-B", "Tiny"])
            .title("Pie — One Dominant (90%)")
            .size(400.0, 400.0)
            .to_svg()?;
        sections.push(("Pie — Dominant Slice", svg));
    }

    // 6d. Equal slices
    {
        let svg = pie(&[25.0, 25.0, 25.0, 25.0], &["North", "South", "East", "West"])
            .title("Pie — Equal Slices")
            .size(400.0, 400.0)
            .to_svg()?;
        sections.push(("Pie — Equal", svg));
    }

    // 6e. Donut variants
    {
        let svg = pie(&[60.0, 25.0, 15.0], &["Pass", "Warn", "Fail"])
            .donut(0.55)
            .title("Donut — 55% hole")
            .size(380.0, 380.0)
            .to_svg()?;
        sections.push(("Donut — 55%", svg));
    }
    {
        let svg = pie(&[40.0, 30.0, 20.0, 10.0], &["A", "B", "C", "D"])
            .donut(0.8)
            .title("Donut — 80% hole (thin ring)")
            .size(380.0, 380.0)
            .to_svg()?;
        sections.push(("Donut — 80% (thin)", svg));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 7: BOX PLOT VARIATIONS
    // ═════════════════════════════════════════════════════════════════

    // 7a. 3 groups
    {
        let mut cats = Vec::new();
        let mut vals = Vec::new();
        for (label, center, spread) in &[("Control", 50.0, 10.0), ("Drug A", 65.0, 8.0), ("Drug B", 70.0, 15.0)] {
            for _ in 0..50 {
                vals.push(center + rng.normal() * spread);
                cats.push(*label);
            }
            vals.push(center + spread * 4.0); // outlier
            cats.push(*label);
        }
        let svg = boxplot(&cats, &vals)
            .title("Box Plot — Clinical Trial")
            .x_label("Treatment")
            .y_label("Response")
            .size(500.0, 380.0)
            .to_svg()?;
        sections.push(("Box Plot — 3 Groups", svg));
    }

    // 7b. Many groups (7)
    {
        let mut cats = Vec::new();
        let mut vals = Vec::new();
        let days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        for (i, day) in days.iter().enumerate() {
            let center = 40.0 + (i as f64 * 0.9).sin() * 15.0;
            let spread = 5.0 + i as f64 * 1.5;
            for _ in 0..30 {
                vals.push(center + rng.normal() * spread);
                cats.push(*day);
            }
        }
        let svg = boxplot(&cats, &vals)
            .title("Box Plot — Daily Response Times")
            .x_label("Day of Week")
            .y_label("Time (ms)")
            .size(600.0, 380.0)
            .to_svg()?;
        sections.push(("Box Plot — 7 Groups", svg));
    }

    // 7c. Box plot with tight distributions (minimal spread)
    {
        let mut cats = Vec::new();
        let mut vals = Vec::new();
        for (label, center) in &[("Batch 1", 100.0), ("Batch 2", 100.5), ("Batch 3", 99.8)] {
            for _ in 0..40 {
                vals.push(center + rng.normal() * 0.3);
                cats.push(*label);
            }
        }
        let svg = boxplot(&cats, &vals)
            .title("Box Plot — Tight Distributions")
            .x_label("Batch")
            .y_label("Weight (g)")
            .size(450.0, 350.0)
            .to_svg()?;
        sections.push(("Box Plot — Tight", svg));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 8: HEATMAP VARIATIONS
    // ═════════════════════════════════════════════════════════════════

    // 8a. Small heatmap with annotations
    {
        let data = vec![
            vec![1.0, 5.0, 9.0],
            vec![2.0, 6.0, 7.0],
            vec![3.0, 4.0, 8.0],
        ];
        let svg = heatmap(data)
            .annotate()
            .row_labels(vec!["X".into(), "Y".into(), "Z".into()])
            .col_labels(vec!["A".into(), "B".into(), "C".into()])
            .title("Heatmap — 3×3 Annotated")
            .x_label("Feature")
            .y_label("Sample")
            .size(350.0, 350.0)
            .to_svg()?;
        sections.push(("Heatmap — 3×3", svg));
    }

    // 8b. Large heatmap (8×8)
    {
        let data: Vec<Vec<f64>> = (0..8).map(|r| {
            (0..8).map(|c| {
                ((r as f64 * 0.5 + c as f64 * 0.3).sin() * 50.0 + 50.0).round()
            }).collect()
        }).collect();
        let rows: Vec<String> = (0..8).map(|i| format!("Gene {}", i + 1)).collect();
        let cols: Vec<String> = (0..8).map(|i| format!("Sample {}", i + 1)).collect();
        let svg = heatmap(data)
            .annotate()
            .row_labels(rows)
            .col_labels(cols)
            .title("Heatmap — 8×8 Gene Expression")
            .x_label("Sample")
            .y_label("Gene")
            .size(600.0, 500.0)
            .to_svg()?;
        sections.push(("Heatmap — 8×8", svg));
    }

    // 8c. Confusion matrix (4×4)
    {
        let data = vec![
            vec![85.0, 3.0, 1.0, 2.0],
            vec![2.0, 78.0, 5.0, 1.0],
            vec![0.0, 4.0, 90.0, 3.0],
            vec![1.0, 2.0, 3.0, 82.0],
        ];
        let labels = vec!["Cat".into(), "Dog".into(), "Bird".into(), "Fish".into()];
        let svg = heatmap(data)
            .annotate()
            .row_labels(labels.clone())
            .col_labels(labels)
            .title("Confusion Matrix — 4 Classes")
            .x_label("Predicted")
            .y_label("Actual")
            .size(450.0, 450.0)
            .to_svg()?;
        sections.push(("Heatmap — Confusion 4×4", svg));
    }

    // 8d. Heatmap with RdBu color scale
    {
        let data = vec![
            vec![-1.0, -0.5, 0.0, 0.5, 1.0],
            vec![0.5, -0.3, 0.8, -0.7, 0.2],
            vec![0.1, 0.9, -0.8, 0.3, -0.4],
        ];
        let mut theme = NewTheme::light();
        theme.color_scale = Some(ColorScale::rdbu());
        let svg = heatmap(data)
            .annotate()
            .title("Heatmap — RdBu Diverging Scale")
            .theme(theme)
            .size(450.0, 350.0)
            .to_svg()?;
        sections.push(("Heatmap — RdBu", svg));
    }

    // 8e. Heatmap no annotations (just gradient)
    {
        let data: Vec<Vec<f64>> = (0..6).map(|r| {
            (0..10).map(|c| (r * 10 + c) as f64).collect()
        }).collect();
        let svg = heatmap(data)
            .title("Heatmap — 6×10 No Annotations")
            .x_label("Time")
            .y_label("Sensor")
            .size(600.0, 400.0)
            .to_svg()?;
        sections.push(("Heatmap — Plain 6×10", svg));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 9: FACETED CHARTS
    // ═════════════════════════════════════════════════════════════════

    // 9a. Faceted scatter — 4 panels
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut facets = Vec::new();
        for panel in &["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"] {
            for _ in 0..30 {
                x.push(rng.range(0.0, 100.0));
                y.push(rng.range(0.0, 100.0));
                facets.push(*panel);
            }
        }
        let svg = scatter(&x, &y)
            .facet_wrap(&facets, 2)
            .title("Faceted Scatter — Quarterly")
            .x_label("Impressions")
            .y_label("Clicks")
            .size(600.0, 500.0)
            .to_svg()?;
        sections.push(("Facet — Scatter 2×2", svg));
    }

    // 9b. Faceted scatter with categories + legend
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut cats = Vec::new();
        let mut facets = Vec::new();
        for region in &["US", "EU", "APAC"] {
            for segment in &["Enterprise", "SMB", "Consumer"] {
                for _ in 0..10 {
                    x.push(rng.range(10.0, 90.0));
                    y.push(rng.range(10.0, 90.0));
                    cats.push(*segment);
                    facets.push(*region);
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
            .facet(Facet::Wrap { ncol: 3 })
            .title("Faceted — Region × Segment")
            .x_label("Deal Size ($K)")
            .y_label("Win Rate (%)")
            .size(700.0, 350.0);
        sections.push(("Facet — Categories+Legend", chart.to_svg()?));
    }

    // 9c. Faceted scatter — FreeY scales
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut facets = Vec::new();
        // Panel A: small y range
        for _ in 0..25 {
            x.push(rng.range(0.0, 10.0));
            y.push(rng.range(0.0, 5.0));
            facets.push("Small Range");
        }
        // Panel B: large y range
        for _ in 0..25 {
            x.push(rng.range(0.0, 10.0));
            y.push(rng.range(0.0, 500.0));
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
            .title("Faceted — FreeY (different Y scales)")
            .x_label("X")
            .y_label("Y")
            .size(600.0, 350.0);
        sections.push(("Facet — FreeY", chart.to_svg()?));
    }

    // 9d. Faceted with FreeX
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut facets = Vec::new();
        for _ in 0..25 {
            x.push(rng.range(0.0, 10.0));
            y.push(rng.range(0.0, 50.0));
            facets.push("Narrow X");
        }
        for _ in 0..25 {
            x.push(rng.range(0.0, 1000.0));
            y.push(rng.range(0.0, 50.0));
            facets.push("Wide X");
        }
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Point)
                    .with_x(x)
                    .with_y(y)
                    .with_facet_values(facets.iter().map(|s| s.to_string()).collect()),
            )
            .facet(Facet::Wrap { ncol: 2 })
            .facet_scales(FacetScales::FreeX)
            .title("Faceted — FreeX (different X scales)")
            .x_label("X")
            .y_label("Y")
            .size(600.0, 350.0);
        sections.push(("Facet — FreeX", chart.to_svg()?));
    }

    // 9e. Faceted with Free (both axes)
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut facets = Vec::new();
        for _ in 0..20 {
            x.push(rng.range(0.0, 5.0));
            y.push(rng.range(0.0, 5.0));
            facets.push("Small");
        }
        for _ in 0..20 {
            x.push(rng.range(100.0, 200.0));
            y.push(rng.range(1000.0, 2000.0));
            facets.push("Large");
        }
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Point)
                    .with_x(x)
                    .with_y(y)
                    .with_facet_values(facets.iter().map(|s| s.to_string()).collect()),
            )
            .facet(Facet::Wrap { ncol: 2 })
            .facet_scales(FacetScales::Free)
            .title("Faceted — Free (both axes independent)")
            .x_label("X")
            .y_label("Y")
            .size(600.0, 350.0);
        sections.push(("Facet — Free Both", chart.to_svg()?));
    }

    // 9f. 6 panels (3 cols)
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut facets = Vec::new();
        let panels = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"];
        for panel in panels {
            for _ in 0..15 {
                x.push(rng.range(0.0, 10.0));
                y.push(rng.range(0.0, 10.0));
                facets.push(panel);
            }
        }
        let svg = scatter(&x, &y)
            .facet_wrap(&facets, 3)
            .title("6 Panels (3 cols)")
            .x_label("X")
            .y_label("Y")
            .size(700.0, 500.0)
            .to_svg()?;
        sections.push(("Facet — 6 Panels", svg));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 10: THEME VARIATIONS
    // ═════════════════════════════════════════════════════════════════

    // 10a. Dark theme — scatter
    {
        let x: Vec<f64> = (0..40).map(|_| rng.range(0.0, 100.0)).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 0.7 + rng.normal() * 10.0).collect();
        let svg = scatter(&x, &y)
            .title("Dark Theme — Scatter")
            .x_label("X")
            .y_label("Y")
            .theme(NewTheme::dark())
            .size(500.0, 380.0)
            .to_svg()?;
        sections.push(("Theme — Dark Scatter", svg));
    }

    // 10b. Dark theme — multi-line
    {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(x.iter().map(|&v| (v * 0.3).sin() * 5.0 + 10.0).collect()).with_label("CPU"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(x.iter().map(|&v| (v * 0.2).cos() * 4.0 + 8.0).collect()).with_label("Memory"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(x.iter().map(|&v| v * 0.3 + 2.0).collect()).with_label("Disk"))
            .title("Dark Theme — System Monitor")
            .x_label("Time (s)")
            .y_label("Usage (%)")
            .theme(NewTheme::dark())
            .size(550.0, 380.0);
        sections.push(("Theme — Dark Lines", chart.to_svg()?));
    }

    // 10c. Dark theme — bar
    {
        let svg = bar(&["Mon", "Tue", "Wed", "Thu", "Fri"], &[120.0, 95.0, 150.0, 88.0, 110.0])
            .title("Dark Theme — Bars")
            .x_label("Day")
            .y_label("Tickets Closed")
            .theme(NewTheme::dark())
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Theme — Dark Bar", svg));
    }

    // 10d. Dark theme — area
    {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.2).sin().abs() * 20.0 + 5.0).collect();
        let svg = area(&x, &y)
            .title("Dark Theme — Area")
            .x_label("Day")
            .y_label("Events")
            .theme(NewTheme::dark())
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Theme — Dark Area", svg));
    }

    // 10e. Publication theme — scatter
    {
        let x: Vec<f64> = (0..30).map(|_| rng.range(0.0, 10.0)).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 1.2 + rng.normal() * 1.5).collect();
        let svg = scatter(&x, &y)
            .title("Publication Theme")
            .x_label("Independent Variable")
            .y_label("Dependent Variable")
            .theme(NewTheme::publication())
            .size(500.0, 380.0)
            .to_svg()?;
        sections.push(("Theme — Publication", svg));
    }

    // 10f. Custom theme — big fonts
    {
        let mut theme = NewTheme::light();
        theme.title_font_size = 22.0;
        theme.label_font_size = 16.0;
        theme.tick_font_size = 14.0;
        theme.line_width = 3.0;
        theme.point_size = 8.0;
        theme.grid_width = 1.5;

        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = vec![3.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0, 9.5, 12.0];
        let svg = scatter(&x, &y)
            .title("Custom Theme — Big Fonts")
            .x_label("Week")
            .y_label("Score")
            .theme(theme)
            .size(500.0, 400.0)
            .to_svg()?;
        sections.push(("Theme — Big Fonts", svg));
    }

    // 10g. Custom theme — no grid
    {
        let mut theme = NewTheme::light();
        theme.show_grid = false;

        let svg = bar(&["A", "B", "C", "D"], &[30.0, 45.0, 25.0, 55.0])
            .title("Custom — No Grid")
            .y_label("Value")
            .theme(theme)
            .size(400.0, 320.0)
            .to_svg()?;
        sections.push(("Theme — No Grid", svg));
    }

    // 10h. Custom palette
    {
        let mut theme = NewTheme::light();
        theme.palette = Palette::diverging(
            Color::from_hex("#e74c3c").unwrap(),
            Color::from_hex("#f1c40f").unwrap(),
            Color::from_hex("#2ecc71").unwrap(),
            6,
        );

        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut cats = Vec::new();
        for (i, name) in ["Bad", "Poor", "Ok", "Good", "Great", "Excellent"].iter().enumerate() {
            for _ in 0..10 {
                x.push(i as f64 + rng.normal() * 0.3);
                y.push(rng.range(0.0, 10.0));
                cats.push(*name);
            }
        }
        let svg = scatter(&x, &y)
            .color_by(&cats)
            .title("Custom Diverging Palette")
            .x_label("Rating")
            .y_label("Score")
            .theme(theme)
            .size(550.0, 380.0)
            .to_svg()?;
        sections.push(("Theme — Custom Palette", svg));
    }

    // ═════════════════════════════════════════════════════════════════
    //  SECTION 11: EDGE CASES & REALISTIC SCENARIOS
    // ═════════════════════════════════════════════════════════════════

    // 11a. All negative values
    {
        let svg = bar(&["Loss A", "Loss B", "Loss C"], &[-15.0, -30.0, -22.0])
            .title("All Negative Values")
            .y_label("P&L ($K)")
            .size(400.0, 320.0)
            .to_svg()?;
        sections.push(("Edge — All Negative", svg));
    }

    // 11b. Mixed positive/negative bar (non-stacked)
    {
        let svg = bar(&["Jan", "Feb", "Mar", "Apr", "May"], &[10.0, -5.0, 15.0, -3.0, 8.0])
            .title("Mixed +/- Bar (no stack)")
            .y_label("Net Change")
            .size(450.0, 320.0)
            .to_svg()?;
        sections.push(("Edge — Mixed +/-", svg));
    }

    // 11c. Very wide chart
    {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.1).sin() * 10.0 + 50.0).collect();
        let svg = line(&x, &y)
            .title("Very Wide Chart (900×200)")
            .x_label("Sample")
            .y_label("Value")
            .size(900.0, 200.0)
            .to_svg()?;
        sections.push(("Edge — Very Wide", svg));
    }

    // 11d. Very tall chart
    {
        let svg = bar(&["A", "B", "C"], &[100.0, 200.0, 150.0])
            .title("Very Tall (300×600)")
            .y_label("Val")
            .size(300.0, 600.0)
            .to_svg()?;
        sections.push(("Edge — Very Tall", svg));
    }

    // 11e. Tiny chart
    {
        let svg = scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 2.0])
            .title("Tiny (200×150)")
            .size(200.0, 150.0)
            .to_svg()?;
        sections.push(("Edge — Tiny", svg));
    }

    // 11f. Very long title
    {
        let svg = scatter(&[1.0, 2.0, 3.0], &[1.0, 4.0, 2.0])
            .title("This Is an Extremely Long Title That Might Overflow the Chart Boundary — How Does It Look?")
            .x_label("X Axis With A Somewhat Long Label Too")
            .y_label("Y Label")
            .size(500.0, 380.0)
            .to_svg()?;
        sections.push(("Edge — Long Title", svg));
    }

    // 11g. Scatter with identical Y values (flat line)
    {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![5.0; 10];
        let svg = scatter(&x, &y)
            .title("Flat — All Y=5")
            .x_label("X")
            .y_label("Y")
            .size(400.0, 300.0)
            .to_svg()?;
        sections.push(("Edge — Flat Y", svg));
    }

    // 11h. Aggregate stat — mean
    {
        let x = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let y = vec![10.0, 12.0, 11.0, 20.0, 22.0, 18.0, 15.0, 17.0, 16.0];
        let cats: Vec<String> = vec!["A","A","A","B","B","B","C","C","C"].into_iter().map(String::from).collect();
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(x)
                    .with_y(y)
                    .with_categories(cats)
                    .stat(Stat::Aggregate { func: AggregateFunc::Mean }),
            )
            .title("Aggregate — Mean per Category")
            .x_label("Category")
            .y_label("Mean Value")
            .size(450.0, 350.0);
        sections.push(("Stat — Aggregate Mean", chart.to_svg()?));
    }

    // 11i. Fill position (100% stacked)
    {
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0])
                    .with_y(vec![30.0, 40.0, 25.0])
                    .with_label("TypeA")
                    .position(Position::Fill),
            )
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0])
                    .with_y(vec![70.0, 60.0, 75.0])
                    .with_label("TypeB")
                    .position(Position::Fill),
            )
            .title("100% Stacked (Fill Position)")
            .x_label("Group")
            .y_label("Proportion")
            .size(450.0, 350.0);
        sections.push(("Position — Fill", chart.to_svg()?));
    }

    // 11j. Line + Area combo (confidence band look)
    {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y_main: Vec<f64> = x.iter().map(|&v| v * 0.5 + 10.0 + (v * 0.3).sin() * 3.0).collect();
        let y_area: Vec<f64> = x.iter().map(|&v| v * 0.5 + 10.0 + (v * 0.3).sin() * 3.0 + 5.0).collect();
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Area).with_x(x.clone()).with_y(y_area).with_label("Upper Bound"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(y_main.clone()).with_label("Forecast"))
            .layer(Layer::new(MarkType::Point).with_x(x.clone()).with_y(y_main).with_label("Data"))
            .title("Forecast with Confidence Band")
            .x_label("Day")
            .y_label("Metric")
            .size(550.0, 380.0);
        sections.push(("Combo — Area+Line+Point", chart.to_svg()?));
    }

    // ═════════════════════════════════════════════════════════════════
    //  BUILD HTML
    // ═════════════════════════════════════════════════════════════════

    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>esoc-chart Stress Test</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #f0f0f4; color: #333; }
  header { background: linear-gradient(135deg, #2d1b69, #11998e); color: white; padding: 2.5rem 2rem; text-align: center; }
  header h1 { font-size: 2rem; font-weight: 300; letter-spacing: 0.02em; }
  header p { margin-top: 0.5rem; opacity: 0.7; font-size: 0.95rem; }
  .stats { display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; flex-wrap: wrap; }
  .stats span { background: rgba(255,255,255,0.15); padding: 0.3rem 0.8rem; border-radius: 4px; font-size: 0.85rem; }
  .section-title { font-size: 1.2rem; font-weight: 600; color: #555; padding: 1.5rem 2rem 0.5rem; max-width: 1600px; margin: 0 auto; border-top: 2px solid #ddd; margin-top: 1rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; padding: 1rem 2rem; max-width: 1600px; margin: 0 auto; }
  .card { background: white; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); overflow: hidden; transition: box-shadow 0.2s; }
  .card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.12); }
  .card h2 { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #555; padding: 1rem 1.5rem 0; }
  .card .chart-wrap { padding: 0.5rem 1rem 0.75rem; }
  .card svg { display: block; width: 100%; height: auto; }
  .card.dark-bg .chart-wrap { background: #1e1e2e; border-radius: 0 0 8px 8px; }
  .feedback { padding: 0 1rem 1rem; }
  .feedback textarea { width: 100%; min-height: 50px; border: 1px solid #e0e0e0; border-radius: 4px; padding: 0.5rem; font-family: inherit; font-size: 0.82rem; resize: vertical; }
  .feedback textarea:focus { outline: none; border-color: #2d1b69; }
  .feedback .status { font-size: 0.72rem; color: #aaa; margin-top: 0.2rem; }
  .actions { padding: 1.5rem 2rem; text-align: center; }
  .actions button { background: #2d1b69; color: white; border: none; border-radius: 4px; padding: 0.6rem 1.5rem; font-size: 0.9rem; cursor: pointer; margin: 0 0.5rem; }
  .actions button:hover { background: #3d2b79; }
</style>
<script>
  const feedback = {};
  function loadFeedback() {
    try { Object.assign(feedback, JSON.parse(localStorage.getItem('stress_test_feedback') || '{}')); } catch {}
    document.querySelectorAll('.feedback textarea').forEach(ta => {
      const key = ta.dataset.chart;
      if (feedback[key]) ta.value = feedback[key];
    });
  }
  function saveFeedback(key, value) {
    feedback[key] = value;
    localStorage.setItem('stress_test_feedback', JSON.stringify(feedback));
  }
  function exportFeedback() {
    const blob = new Blob([JSON.stringify(feedback, null, 2)], {type: 'application/json'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = 'stress_test_feedback.json'; a.click();
  }
  window.addEventListener('DOMContentLoaded', loadFeedback);
</script>
</head>
<body>
<header>
  <h1>esoc-chart Stress Test</h1>
  <p>Comprehensive exercise of every chart type, option, theme, position, stat, and edge case</p>
  <div class="stats">
"#,
    );

    html.push_str(&format!(
        "    <span>{} charts</span>\n",
        sections.len()
    ));
    html.push_str("    <span>11 categories</span>\n");
    html.push_str("    <span>3 themes</span>\n");
    html.push_str("    <span>5 positions</span>\n");
    html.push_str("    <span>4 facet scale modes</span>\n");
    html.push_str("  </div>\n</header>\n");

    for (title, svg) in &sections {
        let key = title.to_lowercase().replace([' ', '–', '—', '+', '/', '(', ')', '%'], "_").replace("__", "_");
        let dark_class = if title.contains("Dark") { " dark-bg" } else { "" };
        html.push_str("<div class=\"grid\">\n");
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
        html.push_str("</div>\n");
    }

    html.push_str(concat!(
        "<div class=\"actions\">\n",
        "  <button onclick=\"exportFeedback()\">Export Feedback JSON</button>\n",
        "</div>\n",
        "</body>\n</html>\n",
    ));

    let out_path = "stress_test.html";
    std::fs::write(out_path, &html).expect("failed to write HTML");
    println!("Saved {} ({} charts)", out_path, sections.len());

    Ok(())
}
