// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generate an HTML gallery showcasing all esoc-chart capabilities.

use esoc_chart::express::*;
use esoc_chart::grammar::annotation::Annotation;
use esoc_chart::grammar::chart::Chart;
use esoc_chart::grammar::coord::CoordSystem;
use esoc_chart::grammar::layer::{Layer, MarkType};
use esoc_chart::grammar::stat::Stat;

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

    // ── Scatter ──────────────────────────────────────────────────────
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.3, 4.1, 3.0, 5.8, 4.9, 7.2, 6.5, 8.1];
        let svg = scatter(&x, &y)
            .title("Scatter Plot")
            .x_label("X")
            .y_label("Y")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Scatter", svg));
    }

    // ── Scatter with categories ──────────────────────────────────────
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.3, 4.1, 3.0, 5.8, 4.9, 7.2, 6.5, 8.1];
        let cats = vec!["A", "B", "A", "B", "A", "B", "A", "B"];
        let svg = scatter(&x, &y)
            .color_by(&cats)
            .title("Colored Scatter")
            .x_label("X")
            .y_label("Y")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Scatter (colored)", svg));
    }

    // ── Dense scatter (opacity demo) ─────────────────────────────────
    {
        let n = 300;
        let x: Vec<f64> = (0..n).map(|_| rng.normal() * 3.0 + 5.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 0.8 + rng.normal() * 2.0).collect();
        let svg = scatter(&x, &y)
            .title("Dense Scatter (auto opacity)")
            .x_label("Feature A")
            .y_label("Feature B")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Dense Scatter", svg));
    }

    // ── Line ─────────────────────────────────────────────────────────
    {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.8).sin() * 3.0 + v).collect();
        let svg = line(&x, &y)
            .title("Line Chart")
            .x_label("Time")
            .y_label("Value")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Line", svg));
    }

    // ── Multi-line (grammar API) ─────────────────────────────────────
    {
        let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.5).collect();
        let y1: Vec<f64> = x.iter().map(|&v| (v * 0.4).sin() * 5.0 + 10.0).collect();
        let y2: Vec<f64> = x.iter().map(|&v| (v * 0.4).cos() * 4.0 + 12.0).collect();
        let y3: Vec<f64> = x.iter().map(|&v| v * 0.5 + 5.0).collect();

        let chart = Chart::new()
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(y1).with_label("sin"))
            .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(y2).with_label("cos"))
            .layer(Layer::new(MarkType::Line).with_x(x).with_y(y3).with_label("linear"))
            .title("Multi-Line Chart")
            .x_label("Time")
            .y_label("Signal")
            .size(500.0, 350.0);
        sections.push(("Multi-Line", chart.to_svg()?));
    }

    // ── Line + Scatter overlay (grammar API) ─────────────────────────
    {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y_data: Vec<f64> = vec![2.1, 3.8, 3.2, 5.5, 4.8, 7.1, 6.3, 8.0, 7.5, 9.2];
        let y_trend: Vec<f64> = x.iter().map(|&v| v * 0.8 + 2.0).collect();

        let chart = Chart::new()
            .layer(Layer::new(MarkType::Point).with_x(x.clone()).with_y(y_data).with_label("Data"))
            .layer(Layer::new(MarkType::Line).with_x(x).with_y(y_trend).with_label("Trend"))
            .title("Scatter + Trend Line")
            .x_label("X")
            .y_label("Y")
            .size(500.0, 350.0);
        sections.push(("Scatter + Line Overlay", chart.to_svg()?));
    }

    // ── LOESS Smooth ─────────────────────────────────────────────────
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
        sections.push(("LOESS Smooth", chart.to_svg()?));
    }

    // ── Bar ──────────────────────────────────────────────────────────
    {
        let cats = vec!["Rust", "Python", "Go", "Java", "C++"];
        let vals = vec![42.0, 35.0, 28.0, 22.0, 18.0];
        let svg = bar(&cats, &vals)
            .title("Language Popularity")
            .x_label("Language")
            .y_label("Score")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Bar", svg));
    }

    // ── Horizontal Bar (flipped coords) ──────────────────────────────
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
        sections.push(("Horizontal Bar", chart.to_svg()?));
    }

    // ── Histogram ────────────────────────────────────────────────────
    {
        let data: Vec<f64> = (0..300).map(|_| rng.normal() * 1.5 + 10.0).collect();
        let svg = histogram(&data)
            .bins(20)
            .title("Histogram")
            .x_label("Value")
            .y_label("Count")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Histogram", svg));
    }

    // ── Area ─────────────────────────────────────────────────────────
    {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| (v * 0.3).sin().abs() * 20.0 + 5.0).collect();
        let svg = area(&x, &y)
            .title("Area Chart")
            .x_label("Day")
            .y_label("Traffic")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Area", svg));
    }

    // ── Pie ──────────────────────────────────────────────────────────
    {
        let vals = vec![35.0, 25.0, 20.0, 15.0, 5.0];
        let labels = vec!["Chrome", "Firefox", "Safari", "Edge", "Other"];
        let svg = pie(&vals, &labels)
            .title("Browser Share")
            .size(400.0, 400.0)
            .to_svg()?;
        sections.push(("Pie", svg));
    }

    // ── Donut ────────────────────────────────────────────────────────
    {
        let vals = vec![60.0, 25.0, 15.0];
        let labels = vec!["Pass", "Warn", "Fail"];
        let svg = pie(&vals, &labels)
            .donut(0.5)
            .title("Test Results")
            .size(400.0, 400.0)
            .to_svg()?;
        sections.push(("Donut", svg));
    }

    // ── Grouped Bar ──────────────────────────────────────────────────
    {
        let cats = vec!["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"];
        let groups = vec!["2024", "2024", "2024", "2024", "2025", "2025", "2025", "2025"];
        let vals = vec![12.0, 18.0, 22.0, 15.0, 14.0, 20.0, 28.0, 19.0];
        let svg = grouped_bar(&cats, &groups, &vals)
            .title("Quarterly Revenue")
            .x_label("Quarter")
            .y_label("Revenue ($M)")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Grouped Bar", svg));
    }

    // ── Stacked Bar ──────────────────────────────────────────────────
    {
        let cats = vec!["Q1", "Q2", "Q3", "Q1", "Q2", "Q3"];
        let groups = vec!["Product", "Product", "Product", "Service", "Service", "Service"];
        let vals = vec![10.0, 15.0, 20.0, 5.0, 8.0, 12.0];
        let svg = stacked_bar(&cats, &groups, &vals)
            .title("Revenue by Segment")
            .x_label("Quarter")
            .y_label("Revenue ($M)")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Stacked Bar", svg));
    }

    // ── Box Plot ─────────────────────────────────────────────────────
    {
        let mut cats = Vec::new();
        let mut vals = Vec::new();
        for label in &["Control", "Treatment A", "Treatment B"] {
            let base = match *label {
                "Control" => 50.0,
                "Treatment A" => 65.0,
                _ => 70.0,
            };
            for _ in 0..30 {
                vals.push(base + (rng.uniform() - 0.5) * 30.0);
                cats.push(*label);
            }
        }
        let svg = boxplot(&cats, &vals)
            .title("Treatment Comparison")
            .x_label("Group")
            .y_label("Response")
            .size(500.0, 350.0)
            .to_svg()?;
        sections.push(("Box Plot", svg));
    }

    // ── Annotations (hline, vline, band, text) ───────────────────────
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
            .annotate(Annotation::band(10.0, 20.0))
            .annotate(Annotation::text(15.0, 25.0, "Peak zone"));
        sections.push(("Annotations", chart.to_svg()?));
    }

    // ── Subtitle + Caption ───────────────────────────────────────────
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 25.0, 18.0, 32.0, 28.0];
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Line).with_x(x).with_y(y))
            .title("Monthly Sales")
            .subtitle("Jan–May 2026")
            .caption("Source: internal data")
            .x_label("Month")
            .y_label("Revenue ($K)")
            .size(500.0, 350.0);
        sections.push(("Subtitle + Caption", chart.to_svg()?));
    }

    // ── Faceted Scatter (small multiples) ────────────────────────────
    {
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut facets = Vec::new();
        for panel in &["East", "West", "North", "South"] {
            for _ in 0..20 {
                x.push(rng.uniform() * 10.0);
                y.push(rng.uniform() * 10.0);
                facets.push(*panel);
            }
        }
        let svg = scatter(&x, &y)
            .facet_wrap(&facets, 2)
            .title("Regional Data")
            .x_label("X")
            .y_label("Y")
            .size(500.0, 400.0)
            .to_svg()?;
        sections.push(("Faceted Scatter", svg));
    }

    // ── Heatmap ──────────────────────────────────────────────────────
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
            .title("Heatmap")
            .x_label("Variable")
            .y_label("Group")
            .size(450.0, 380.0)
            .to_svg()?;
        sections.push(("Heatmap", svg));
    }

    // ── Confusion Matrix ─────────────────────────────────────────────
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
        sections.push(("Confusion Matrix", svg));
    }

    // ── Build HTML ───────────────────────────────────────────────────
    let mut html = String::from(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>esoc-chart Gallery</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #f5f5f5; color: #333; }
  header { background: #1a1a2e; color: white; padding: 2rem; text-align: center; }
  header h1 { font-size: 2rem; font-weight: 300; }
  header p { margin-top: 0.5rem; opacity: 0.7; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 1.5rem; padding: 2rem; max-width: 1400px; margin: 0 auto; }
  .card { background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }
  .card h2 { font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #666; padding: 1rem 1.5rem 0; }
  .card svg { display: block; width: 100%; height: auto; padding: 0.5rem 1rem 1rem; }
  .feedback { padding: 0 1rem 1rem; }
  .feedback textarea { width: 100%; min-height: 60px; border: 1px solid #ddd; border-radius: 4px; padding: 0.5rem; font-family: inherit; font-size: 0.85rem; resize: vertical; }
  .feedback textarea:focus { outline: none; border-color: #1a1a2e; }
  .feedback .status { font-size: 0.75rem; color: #999; margin-top: 0.25rem; }
  .actions { padding: 1.5rem 2rem; text-align: center; }
  .actions button { background: #1a1a2e; color: white; border: none; border-radius: 4px; padding: 0.6rem 1.5rem; font-size: 0.9rem; cursor: pointer; }
  .actions button:hover { background: #2a2a4e; }
</style>
<script>
  const feedback = {};
  function loadFeedback() {
    try { Object.assign(feedback, JSON.parse(localStorage.getItem('chart_feedback') || '{}')); } catch {}
    document.querySelectorAll('.feedback textarea').forEach(ta => {
      const key = ta.dataset.chart;
      if (feedback[key]) ta.value = feedback[key];
    });
  }
  function saveFeedback(key, value) {
    feedback[key] = value;
    localStorage.setItem('chart_feedback', JSON.stringify(feedback));
  }
  function exportFeedback() {
    const blob = new Blob([JSON.stringify(feedback, null, 2)], {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'chart_feedback.json';
    a.click();
  }
  window.addEventListener('DOMContentLoaded', loadFeedback);
</script>
</head>
<body>
<header>
  <h1>esoc-chart Gallery</h1>
  <p>All charts generated with the express &amp; grammar APIs</p>
</header>
<div class="grid">
"#,
    );

    for (title, svg) in &sections {
        let key = title.to_lowercase().replace(' ', "_");
        html.push_str(&format!(
            concat!(
                "<div class=\"card\">\n",
                "  <h2>{title}</h2>\n",
                "  {svg}\n",
                "  <div class=\"feedback\">\n",
                "    <textarea data-chart=\"{key}\" placeholder=\"Feedback on {title}…\" ",
                "oninput=\"saveFeedback('{key}', this.value)\"></textarea>\n",
                "    <div class=\"status\">Auto-saved to browser</div>\n",
                "  </div>\n",
                "</div>\n",
            ),
            title = title,
            svg = svg,
            key = key,
        ));
    }

    html.push_str(concat!(
        "</div>\n",
        "<div class=\"actions\">\n",
        "  <button onclick=\"exportFeedback()\">Export Feedback as JSON</button>\n",
        "</div>\n",
        "</body>\n</html>\n",
    ));

    std::fs::write("chart_gallery.html", &html).expect("failed to write HTML");
    println!("Saved chart_gallery.html ({} charts)", sections.len());

    Ok(())
}
