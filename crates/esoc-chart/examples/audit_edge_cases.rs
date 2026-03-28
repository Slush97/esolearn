// SPDX-License-Identifier: MIT OR Apache-2.0
//! Edge-case charts for audit: tests formatting issues at boundary conditions.

use std::fmt::Write;

use esoc_chart::express::{bar, heatmap, histogram, line, pie_labeled, scatter};
use esoc_chart::new_theme::NewTheme;

fn try_chart(
    name: &str,
    f: impl FnOnce() -> esoc_chart::error::Result<String>,
) -> Option<(String, String)> {
    match f() {
        Ok(svg) => Some((name.to_string(), svg)),
        Err(e) => {
            eprintln!("FAIL [{name}]: {e}");
            None
        }
    }
}

fn main() {
    let mut sections: Vec<(String, String)> = Vec::new();

    // 1. Single data point
    if let Some(s) = try_chart("Single Point", || {
        scatter(&[5.0], &[10.0])
            .title("Single Point")
            .size(400.0, 300.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 2. Two points
    if let Some(s) = try_chart("Two Points", || {
        scatter(&[0.0, 100.0], &[0.0, 100.0])
            .title("Two Points")
            .size(400.0, 300.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 3. Very long category labels
    if let Some(s) = try_chart("Long Labels", || {
        let cats = vec![
            "This is an extremely long category label",
            "Another very very long label here",
            "Short",
            "Medium length label",
            "Yet another verbose category name",
        ];
        bar(&cats, &[10.0, 25.0, 15.0, 30.0, 20.0])
            .title("Long Category Labels")
            .x_label("Categories")
            .y_label("Value")
            .size(600.0, 400.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 4. Large numeric values (millions)
    if let Some(s) = try_chart("Large Numbers", || {
        let x: Vec<f64> = (0..10).map(|i| f64::from(i) * 1_000_000.0).collect();
        let y: Vec<f64> = (0..10).map(|i| f64::from(i).powi(2) * 500_000.0).collect();
        scatter(&x, &y)
            .title("Large Numbers (Millions)")
            .x_label("Revenue ($)")
            .y_label("Profit ($)")
            .size(600.0, 350.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 5. Very small numbers
    if let Some(s) = try_chart("Small Numbers", || {
        let x: Vec<f64> = (0..10).map(|i| f64::from(i) * 0.0001).collect();
        let y: Vec<f64> = (0..10).map(|i| f64::from(i) * 0.00005).collect();
        scatter(&x, &y)
            .title("Small Numbers")
            .size(600.0, 350.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 6. All-negative bars
    if let Some(s) = try_chart("Negative Bars", || {
        bar(&["A", "B", "C", "D"], &[-10.0, -25.0, -5.0, -30.0])
            .title("All-Negative Bars")
            .y_label("Loss")
            .size(500.0, 350.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 7. Mixed +/- bars
    if let Some(s) = try_chart("Mixed Bars", || {
        bar(&["Q1", "Q2", "Q3", "Q4"], &[15.0, -10.0, 25.0, -5.0])
            .title("Mixed +/- Bars")
            .y_label("P&L")
            .size(500.0, 350.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 8. Dense scatter (1000 points)
    if let Some(s) = try_chart("Dense Scatter", || {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> f64 {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                (self.0 >> 11) as f64 / (1u64 << 53) as f64
            }
        }
        let mut rng = Rng(123);
        let x: Vec<f64> = (0..1000).map(|_| rng.next() * 100.0).collect();
        let y: Vec<f64> = (0..1000).map(|_| rng.next() * 100.0).collect();
        scatter(&x, &y)
            .title("Dense Scatter (n=1000)")
            .size(500.0, 400.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 9. Single bar
    if let Some(s) = try_chart("Single Bar", || {
        bar(&["Only One"], &[42.0])
            .title("Single Bar")
            .size(500.0, 350.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 10. Many categories (20 bars)
    if let Some(s) = try_chart("Many Categories", || {
        let cats: Vec<String> = (0..20).map(|i| format!("Cat_{i}")).collect();
        let cats_ref: Vec<&str> = cats.iter().map(|s| s.as_str()).collect();
        let vals: Vec<f64> = (0..20).map(|i| (f64::from(i) * 7.0) % 50.0 + 5.0).collect();
        bar(&cats_ref, &vals)
            .title("Many Categories (20)")
            .size(700.0, 400.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 11. Narrow chart
    if let Some(s) = try_chart("Narrow Chart", || {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 15.0, 25.0, 30.0];
        line(&x, &y)
            .title("Narrow Chart")
            .x_label("X")
            .y_label("Y")
            .size(300.0, 400.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 12. Wide chart
    if let Some(s) = try_chart("Wide Chart", || {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 15.0, 25.0, 30.0];
        line(&x, &y)
            .title("Wide Chart")
            .x_label("X")
            .y_label("Y")
            .size(1000.0, 250.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 13. Histogram (few bins)
    if let Some(s) = try_chart("Few Bins", || {
        histogram(&[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
            .bins(3)
            .title("Histogram (3 bins)")
            .size(400.0, 300.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 14. Dark theme scatter
    if let Some(s) = try_chart("Dark Scatter", || {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.3, 4.1, 3.0, 5.8, 4.9, 7.2, 6.5, 8.1];
        let cats = vec!["A", "B", "A", "B", "A", "B", "A", "B"];
        scatter(&x, &y)
            .color_by(&cats)
            .title("Dark Theme Scatter")
            .x_label("X")
            .y_label("Y")
            .theme(NewTheme::dark())
            .size(500.0, 350.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 15. Dark theme bars
    if let Some(s) = try_chart("Dark Bars", || {
        bar(&["A", "B", "C", "D"], &[10.0, 25.0, 15.0, 30.0])
            .title("Dark Theme Bars")
            .theme(NewTheme::dark())
            .size(500.0, 350.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 16. Flat line (constant y)
    if let Some(s) = try_chart("Flat Line", || {
        let x: Vec<f64> = (0..10).map(f64::from).collect();
        line(&x, &[5.0; 10])
            .title("Constant Y")
            .size(400.0, 300.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 17. Pie with many small slices
    if let Some(s) = try_chart("Many Pie Slices", || {
        let mut vals = vec![50.0, 30.0, 15.0];
        vals.extend(std::iter::repeat_n(0.5, 10));
        let labels: Vec<String> = (0..vals.len()).map(|i| format!("Slice {i}")).collect();
        let labels_ref: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
        pie_labeled(&labels_ref, &vals)
            .title("Pie: Many Slices")
            .size(500.0, 400.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // 18. Heatmap
    if let Some(s) = try_chart("Heatmap", || {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        heatmap(data)
            .title("Basic Heatmap")
            .with_row_labels(&["R1", "R2", "R3"])
            .with_col_labels(&["C1", "C2", "C3", "C4"])
            .size(500.0, 350.0)
            .to_svg()
    }) {
        sections.push(s);
    }

    // ── Write HTML ─────────────────────────────────────────────────────
    let mut html = String::from(
        r#"<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Audit Edge Cases</title>
<style>
body { font-family: system-ui; background: #f5f5f5; padding: 20px; }
.chart-card { background: white; border-radius: 8px; padding: 16px; margin: 16px 0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12); display: inline-block; vertical-align: top; }
h2 { color: #333; font-size: 14px; margin: 0 0 8px 0; }
.fail { background: #fff0f0; border: 1px solid #fcc; }
</style></head><body>
<h1>Audit Edge Cases</h1>"#,
    );
    for (title, svg) in &sections {
        writeln!(
            html,
            "<div class=\"chart-card\"><h2>{title}</h2>{svg}</div>"
        )
        .unwrap();
    }
    html.push_str("</body></html>");
    std::fs::write("audit_edge_cases.html", &html).unwrap();
    println!("Saved audit_edge_cases.html ({} charts)", sections.len());
}
