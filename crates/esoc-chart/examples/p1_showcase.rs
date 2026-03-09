// SPDX-License-Identifier: MIT OR Apache-2.0
//! P1 autoformatting showcase — demonstrates all new polish features.
//!
//! Generates charts specifically designed to show:
//! - Ratio-based font hierarchy & muted subtitle
//! - Histogram bar spacing (bins nearly touching)
//! - Area-based point sizing
//! - Density-adaptive scatter opacity
//! - Per-chart-type gridlines (bar = horizontal only)
//! - Improved number formatting (SI/commas)
//! - Area chart, pie chart, donut chart, stacked bar, grouped bar

use esoc_chart::v2::*;

fn main() -> esoc_chart::error::Result<()> {
    // ── Simple LCG for reproducibility ────────────────────────────────
    let mut seed: u64 = 42;
    let mut rng = || -> f64 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut normal = || -> f64 {
        let u1 = rng().max(1e-15);
        let u2 = rng();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // ── 1. Dense scatter (opacity demo) ───────────────────────────────
    let n = 500;
    let x: Vec<f64> = (0..n).map(|_| normal() * 3.0 + 5.0).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * 0.8 + normal() * 2.0).collect();

    let svg = scatter(&x, &y)
        .title("Dense Scatter — Opacity & Point Sizing")
        .x_label("feature A")
        .y_label("feature B")
        .size(700.0, 500.0)
        .to_svg()?;
    std::fs::write("dense_scatter.svg", &svg)?;
    println!("Saved dense_scatter.svg");

    // ── 2. Histogram (bins touching) ──────────────────────────────────
    let hist_data: Vec<f64> = (0..400).map(|_| normal() * 1.5 + 10.0).collect();

    let svg = histogram(&hist_data)
        .bins(30)
        .title("Normal Distribution — Tight Bins")
        .x_label("value")
        .y_label("count")
        .size(700.0, 450.0)
        .to_svg()?;
    std::fs::write("hist_tight_bins.svg", &svg)?;
    println!("Saved hist_tight_bins.svg");

    // ── 3. Bar chart (horizontal-only gridlines) ──────────────────────
    let langs = ["Rust", "Python", "TypeScript", "Go", "Java", "C++", "Ruby", "Swift"];
    let users: Vec<f64> = vec![85000.0, 1200000.0, 950000.0, 420000.0, 780000.0, 650000.0, 180000.0, 310000.0];

    let svg = bar(&langs, &users)
        .title("Language Users (thousands)")
        .size(700.0, 450.0)
        .to_svg()?;
    std::fs::write("bar_large_values.svg", &svg)?;
    println!("Saved bar_large_values.svg");

    // ── 4. Area chart ─────────────────────────────────────────────────
    let x_area: Vec<f64> = (0..60).map(|i| i as f64 * 0.5).collect();
    let y_area: Vec<f64> = x_area.iter().map(|&xi| (xi * 0.3).sin() * 20.0 + 25.0 + (xi * 0.1).cos() * 5.0).collect();

    let svg = area(&x_area, &y_area)
        .title("Server Load Over Time")
        .x_label("minutes")
        .y_label("requests / sec")
        .size(700.0, 400.0)
        .to_svg()?;
    std::fs::write("area_chart.svg", &svg)?;
    println!("Saved area_chart.svg");

    // ── 5. Pie chart ──────────────────────────────────────────────────
    let pie_vals = [35.0, 25.0, 20.0, 12.0, 8.0];
    let pie_labels = ["Chrome", "Safari", "Firefox", "Edge", "Other"];

    let svg = pie(&pie_vals, &pie_labels)
        .title("Browser Market Share")
        .size(500.0, 500.0)
        .to_svg()?;
    std::fs::write("pie_chart.svg", &svg)?;
    println!("Saved pie_chart.svg");

    // ── 6. Donut chart ────────────────────────────────────────────────
    let svg = pie(&pie_vals, &pie_labels)
        .donut(0.5)
        .title("Browser Share (Donut)")
        .size(500.0, 500.0)
        .to_svg()?;
    std::fs::write("donut_chart.svg", &svg)?;
    println!("Saved donut_chart.svg");

    // ── 7. Stacked bar ───────────────────────────────────────────────
    let stack_cats = ["Q1", "Q2", "Q3", "Q4"];
    let stack_groups = [
        "Product A", "Product A", "Product A", "Product A",
        "Product B", "Product B", "Product B", "Product B",
        "Product C", "Product C", "Product C", "Product C",
    ];
    let stack_vals = [
        30.0, 45.0, 55.0, 40.0,    // Product A
        20.0, 25.0, 30.0, 35.0,    // Product B
        15.0, 10.0, 20.0, 25.0,    // Product C
    ];
    // stacked_bar expects (categories, groups, values) where each row is (cat, group, value)
    let cats_expanded: Vec<&str> = stack_cats.iter().copied().cycle().take(12).collect();
    // Groups need to match the value ordering
    let svg = stacked_bar(&cats_expanded, &stack_groups, &stack_vals)
        .title("Quarterly Revenue by Product")
        .x_label("Quarter")
        .y_label("Revenue ($M)")
        .size(700.0, 450.0)
        .to_svg()?;
    std::fs::write("stacked_bar.svg", &svg)?;
    println!("Saved stacked_bar.svg");

    // ── 8. Grouped bar ───────────────────────────────────────────────
    let svg = grouped_bar(&cats_expanded, &stack_groups, &stack_vals)
        .title("Quarterly Revenue — Grouped")
        .x_label("Quarter")
        .y_label("Revenue ($M)")
        .size(700.0, 450.0)
        .to_svg()?;
    std::fs::write("grouped_bar.svg", &svg)?;
    println!("Saved grouped_bar.svg");

    // ── 9. Boxplot via v2 API ────────────────────────────────────────
    let mut box_cats = Vec::new();
    let mut box_vals = Vec::new();
    for label in ["Setosa", "Versicolor", "Virginica"] {
        let center = match label {
            "Setosa" => 1.5,
            "Versicolor" => 4.3,
            _ => 5.8,
        };
        for _ in 0..60 {
            box_cats.push(label);
            box_vals.push(center + normal() * 0.5);
        }
    }

    let svg = boxplot(&box_cats, &box_vals)
        .title("Petal Length by Species")
        .x_label("Species")
        .y_label("Petal Length (cm)")
        .size(600.0, 450.0)
        .to_svg()?;
    std::fs::write("boxplot_v2.svg", &svg)?;
    println!("Saved boxplot_v2.svg");

    // ── 10. Scatter with subtitle & caption (font hierarchy demo) ────
    let x_sm: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y_sm: Vec<f64> = x_sm.iter().map(|&xi| xi.sqrt() * 3.0 + normal()).collect();

    let chart = Chart::new()
        .layer(Layer::new(MarkType::Point).with_x(x_sm).with_y(y_sm))
        .title("Growth Trend Analysis")
        .subtitle("Subtitle uses muted color and smaller font")
        .caption("Source: synthetic data")
        .x_label("Day")
        .y_label("Value")
        .size(700.0, 500.0);

    let svg = chart.to_svg()?;
    std::fs::write("font_hierarchy.svg", &svg)?;
    println!("Saved font_hierarchy.svg");

    // ── 11. Categorical scatter with many points (opacity per category) ─
    let mut cx = Vec::new();
    let mut cy = Vec::new();
    let mut cc = Vec::new();
    for (label, cx_off, cy_off) in [("Group A", 0.0, 0.0), ("Group B", 5.0, 3.0), ("Group C", 2.5, 6.0)] {
        for _ in 0..150 {
            cx.push(cx_off + normal() * 1.2);
            cy.push(cy_off + normal() * 1.2);
            cc.push(label);
        }
    }

    let svg = scatter(&cx, &cy)
        .color_by(&cc)
        .title("Dense Categorical Scatter")
        .x_label("x")
        .y_label("y")
        .size(700.0, 500.0)
        .to_svg()?;
    std::fs::write("dense_categorical.svg", &svg)?;
    println!("Saved dense_categorical.svg");

    // ── 12. Multi-line with grammar API (dark theme) ──────────────────
    let epochs: Vec<f64> = (1..=40).map(|i| i as f64).collect();
    let loss1: Vec<f64> = epochs.iter().map(|&e| 2.5 * (-e / 10.0).exp() + 0.1 + normal() * 0.02).collect();
    let loss2: Vec<f64> = epochs.iter().map(|&e| 2.0 * (-e / 15.0).exp() + 0.15 + normal() * 0.03).collect();

    let chart = Chart::new()
        .layer(Layer::new(MarkType::Line).with_x(epochs.clone()).with_y(loss1))
        .layer(Layer::new(MarkType::Line).with_x(epochs).with_y(loss2))
        .title("Model Comparison — Dark Theme")
        .subtitle("Lower is better")
        .x_label("Epoch")
        .y_label("Loss")
        .theme(NewTheme::dark())
        .size(700.0, 450.0);

    let svg = chart.to_svg()?;
    std::fs::write("dark_theme.svg", &svg)?;
    println!("Saved dark_theme.svg");

    println!("\nAll P1 showcase charts generated!");
    Ok(())
}
