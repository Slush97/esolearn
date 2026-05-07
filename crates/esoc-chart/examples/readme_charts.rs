// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generate SVG images for the README.

use esoc_chart::express::{
    area, bar, boxplot, grouped_bar, heatmap, histogram, line, pie_labeled, scatter, stacked_bar,
    treemap,
};
use esoc_chart::grammar::annotation::Annotation;
use esoc_chart::grammar::chart::Chart;
use esoc_chart::grammar::coord::CoordSystem;
use esoc_chart::grammar::layer::{Layer, MarkType};
use esoc_chart::grammar::stat::Stat;
use esoc_chart::new_theme::NewTheme;

fn main() -> esoc_chart::error::Result<()> {
    // Simple deterministic RNG
    struct Rng(u64);
    impl Rng {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-15);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }
    let mut rng = Rng(42);

    let dir = "crates/esoc-chart/images";
    std::fs::create_dir_all(dir).unwrap();

    // ── 1. Scatter ──────────────────────────────────────────────────
    {
        let n = 80;
        let x: Vec<f64> = (0..n).map(|_| rng.uniform() * 10.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 0.4 * xi * xi - 2.0 * xi + 3.0 + rng.normal() * 2.0)
            .collect();
        scatter(&x, &y)
            .title("Quadratic Trend")
            .x_label("x")
            .y_label("y")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/scatter.svg"))?;
    }

    // ── 2. Scatter with categories ──────────────────────────────────
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut cats = Vec::new();
        for (label, cx, cy) in [
            ("Setosa", 5.0, 3.4),
            ("Versicolor", 5.9, 2.8),
            ("Virginica", 6.6, 3.0),
        ] {
            for _ in 0..40 {
                x.push(cx + rng.normal() * 0.4);
                y.push(cy + rng.normal() * 0.3);
                cats.push(label);
            }
        }
        scatter(&x, &y)
            .color_by(&cats)
            .title("Iris Clusters")
            .x_label("Sepal Length")
            .y_label("Sepal Width")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/scatter_categories.svg"))?;
    }

    // ── 3. Line chart ───────────────────────────────────────────────
    {
        let x: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.2).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.8).sin() * 3.0 + v * 0.5).collect();
        line(&x, &y)
            .title("Signal + Trend")
            .x_label("Time (s)")
            .y_label("Amplitude")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/line.svg"))?;
    }

    // ── 4. Multi-line (grammar API) ─────────────────────────────────
    {
        let epochs: Vec<f64> = (1..=30).map(f64::from).collect();
        let train_loss: Vec<f64> = epochs
            .iter()
            .map(|&e| 2.5 * (-e / 8.0).exp() + 0.1)
            .collect();
        let val_loss: Vec<f64> = epochs
            .iter()
            .map(|&e| 2.5 * (-e / 10.0).exp() + 0.25 + rng.normal() * 0.05)
            .collect();

        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Line)
                    .with_x(epochs.clone())
                    .with_y(train_loss)
                    .with_label("Train"),
            )
            .layer(
                Layer::new(MarkType::Line)
                    .with_x(epochs)
                    .with_y(val_loss)
                    .with_label("Validation"),
            )
            .title("Training Curves")
            .x_label("Epoch")
            .y_label("Loss")
            .size(560.0, 380.0);
        chart.save_svg_to(format!("{dir}/multi_line.svg"))?;
    }

    // ── 5. Bar chart ────────────────────────────────────────────────
    {
        let cats = ["Rust", "Python", "Go", "TypeScript", "Java"];
        let vals = [92.0, 87.0, 79.0, 73.0, 68.0];
        bar(&cats, &vals)
            .title("Developer Satisfaction")
            .x_label("Language")
            .y_label("Score (%)")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/bar.svg"))?;
    }

    // ── 6. Grouped bar ──────────────────────────────────────────────
    {
        let cats = ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"];
        let groups = [
            "2024", "2024", "2024", "2024", "2025", "2025", "2025", "2025",
        ];
        let vals = [12.0, 18.0, 22.0, 15.0, 14.0, 20.0, 28.0, 19.0];
        grouped_bar(&cats, &groups, &vals)
            .title("Quarterly Revenue")
            .x_label("Quarter")
            .y_label("Revenue ($M)")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/grouped_bar.svg"))?;
    }

    // ── 7. Stacked bar ──────────────────────────────────────────────
    {
        let cats = ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"];
        let groups = [
            "Product", "Product", "Product", "Product", "Service", "Service", "Service", "Service",
        ];
        let vals = [10.0, 15.0, 20.0, 18.0, 5.0, 8.0, 12.0, 10.0];
        stacked_bar(&cats, &groups, &vals)
            .title("Revenue by Segment")
            .x_label("Quarter")
            .y_label("Revenue ($M)")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/stacked_bar.svg"))?;
    }

    // ── 8. Histogram ────────────────────────────────────────────────
    {
        let data: Vec<f64> = (0..500).map(|_| rng.normal() * 1.5 + 10.0).collect();
        histogram(&data)
            .bins(25)
            .title("Feature Distribution")
            .x_label("Value")
            .y_label("Count")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/histogram.svg"))?;
    }

    // ── 9. Area chart ───────────────────────────────────────────────
    {
        let x: Vec<f64> = (0..40).map(f64::from).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| (v * 0.2).sin().abs() * 25.0 + 8.0 + rng.normal() * 1.5)
            .collect();
        area(&x, &y)
            .title("Daily Active Users")
            .x_label("Day")
            .y_label("Users (k)")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/area.svg"))?;
    }

    // ── 10. Pie chart ───────────────────────────────────────────────
    {
        let labels = ["Chrome", "Firefox", "Safari", "Edge", "Other"];
        let vals = [64.0, 12.0, 10.0, 8.0, 6.0];
        pie_labeled(&labels, &vals)
            .title("Browser Market Share")
            .size(420.0, 420.0)
            .save_svg(format!("{dir}/pie.svg"))?;
    }

    // ── 11. Donut chart ─────────────────────────────────────────────
    {
        let labels = ["Pass", "Warn", "Fail"];
        let vals = [72.0, 18.0, 10.0];
        pie_labeled(&labels, &vals)
            .donut(0.5)
            .title("Test Suite Results")
            .size(420.0, 420.0)
            .save_svg(format!("{dir}/donut.svg"))?;
    }

    // ── 12. Box plot ────────────────────────────────────────────────
    {
        let mut cats = Vec::new();
        let mut vals = Vec::new();
        for (label, center, spread) in [
            ("Control", 50.0, 12.0),
            ("Treatment A", 62.0, 10.0),
            ("Treatment B", 71.0, 8.0),
        ] {
            for _ in 0..40 {
                vals.push(center + rng.normal() * spread);
                cats.push(label);
            }
        }
        boxplot(&cats, &vals)
            .title("Treatment Comparison")
            .x_label("Group")
            .y_label("Response")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/boxplot.svg"))?;
    }

    // ── 13. Heatmap ─────────────────────────────────────────────────
    {
        let data = vec![
            vec![0.92, 0.05, 0.03],
            vec![0.04, 0.88, 0.08],
            vec![0.02, 0.06, 0.92],
        ];
        heatmap(data)
            .annotate()
            .with_row_labels(&["Cat", "Dog", "Bird"])
            .with_col_labels(&["Cat", "Dog", "Bird"])
            .title("Confusion Matrix")
            .x_label("Predicted")
            .y_label("Actual")
            .size(420.0, 420.0)
            .save_svg(format!("{dir}/heatmap.svg"))?;
    }

    // ── 14. Treemap ─────────────────────────────────────────────────
    {
        let labels = ["AWS", "Azure", "GCP", "Alibaba", "Oracle", "IBM"];
        let vals = [32.0, 23.0, 11.0, 5.0, 3.0, 2.0];
        treemap(&labels, &vals)
            .title("Cloud Market Share (%)")
            .size(560.0, 380.0)
            .save_svg(format!("{dir}/treemap.svg"))?;
    }

    // ── 15. LOESS smooth ────────────────────────────────────────────
    {
        let x: Vec<f64> = (0..60).map(|i| f64::from(i) * 0.15).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| (v * 0.5).sin() * 3.0 + rng.normal() * 0.8)
            .collect();
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Point)
                    .with_x(x.clone())
                    .with_y(y.clone())
                    .with_label("Raw"),
            )
            .layer(
                Layer::new(MarkType::Line)
                    .with_x(x)
                    .with_y(y)
                    .stat(Stat::Smooth { bandwidth: 0.3 })
                    .with_label("LOESS"),
            )
            .title("LOESS Smoothing")
            .x_label("x")
            .y_label("y")
            .size(560.0, 380.0);
        chart.save_svg_to(format!("{dir}/loess.svg"))?;
    }

    // ── 16. Annotations ─────────────────────────────────────────────
    {
        let x: Vec<f64> = (0..30).map(f64::from).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| v * 1.2 + rng.normal() * 3.0 + 5.0)
            .collect();
        let chart = scatter(&x, &y)
            .title("Annotated Scatter")
            .x_label("Day")
            .y_label("Metric")
            .size(560.0, 380.0)
            .build()
            .annotate(Annotation::hline(25.0).with_label("Target"))
            .annotate(Annotation::band(15.0, 25.0));
        chart.save_svg_to(format!("{dir}/annotations.svg"))?;
    }

    // ── 17. Dark theme ──────────────────────────────────────────────
    {
        let epochs: Vec<f64> = (1..=25).map(f64::from).collect();
        let loss: Vec<f64> = epochs
            .iter()
            .map(|&e| 3.0 * (-e / 6.0).exp() + 0.15)
            .collect();
        let acc: Vec<f64> = epochs
            .iter()
            .map(|&e| 0.95 * (1.0 - (-e / 5.0).exp()))
            .collect();

        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Line)
                    .with_x(epochs.clone())
                    .with_y(loss)
                    .with_label("Loss"),
            )
            .layer(
                Layer::new(MarkType::Line)
                    .with_x(epochs)
                    .with_y(acc)
                    .with_label("Accuracy"),
            )
            .title("Model Training")
            .x_label("Epoch")
            .y_label("Value")
            .theme(NewTheme::dark())
            .size(560.0, 380.0);
        chart.save_svg_to(format!("{dir}/dark_theme.svg"))?;
    }

    // ── 18. Horizontal bar ──────────────────────────────────────────
    {
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0, 4.0])
                    .with_y(vec![92.0, 87.0, 79.0, 73.0, 68.0])
                    .with_categories(vec![
                        "Rust".into(),
                        "Python".into(),
                        "Go".into(),
                        "TypeScript".into(),
                        "Java".into(),
                    ]),
            )
            .coord(CoordSystem::Flipped)
            .title("Satisfaction Scores")
            .x_label("Score (%)")
            .y_label("Language")
            .size(560.0, 380.0);
        chart.save_svg_to(format!("{dir}/horizontal_bar.svg"))?;
    }

    println!("Generated 18 SVGs in {dir}/");
    Ok(())
}
