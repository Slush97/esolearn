use std::collections::HashMap;
use std::fs;
use std::path::Path;

use base64::Engine;
use esoc_chart::express::*;
use esoc_chart::grammar::chart::Chart;
use esoc_chart::grammar::layer::{Layer, MarkType};
use esoc_chart::new_theme::NewTheme;
use serde::{Deserialize, Serialize};

// ── Gemini structured feedback ──────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
struct ChartFeedback {
    chart_type: String,
    overall_score: u8,
    strengths: Vec<String>,
    issues: Vec<String>,
    suggestions: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AuditReport {
    charts: Vec<ChartFeedback>,
}

// ── Chart generation ────────────────────────────────────────────────

fn generate_charts(out: &Path) -> Vec<(String, std::path::PathBuf)> {
    let theme = NewTheme::light();
    let mut manifest = Vec::new();

    // Scatter
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.3, 4.1, 3.0, 5.8, 4.5, 7.2, 6.1, 8.4];
        let cats = vec!["A", "A", "A", "A", "B", "B", "B", "B"];
        let chart = scatter(&x, &y)
            .color_by(&cats)
            .title("Scatter Plot")
            .x_label("X Axis")
            .y_label("Y Axis")
            .theme(theme.clone())
            .build();
        save(&chart, out, "scatter", &mut manifest);
    }

    // Line
    {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|v| (v * 0.8).sin() * 3.0 + 5.0).collect();
        let chart = line(&x, &y)
            .title("Line Chart")
            .x_label("Time")
            .y_label("Value")
            .theme(theme.clone())
            .build();
        save(&chart, out, "line", &mut manifest);
    }

    // Bar
    {
        let cats = vec!["Mon", "Tue", "Wed", "Thu", "Fri"];
        let vals = vec![12.0, 19.0, 8.0, 15.0, 22.0];
        let chart = bar(&cats, &vals)
            .title("Bar Chart")
            .x_label("Day")
            .y_label("Sales")
            .theme(theme.clone())
            .build();
        save(&chart, out, "bar", &mut manifest);
    }

    // Histogram
    {
        let data: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64 / 200.0;
                // pseudo-normal via simple transform
                (t * std::f64::consts::TAU).sin() * 10.0 + 50.0 + (i % 7) as f64
            })
            .collect();
        let chart = histogram(&data)
            .bins(15)
            .title("Histogram")
            .x_label("Value")
            .y_label("Frequency")
            .theme(theme.clone())
            .build();
        save(&chart, out, "histogram", &mut manifest);
    }

    // Boxplot
    {
        let mut cats = Vec::new();
        let mut vals = Vec::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.5, 4.5] {
            cats.push("Group A");
            vals.push(v);
        }
        for v in [3.0, 5.0, 7.0, 9.0, 11.0, 8.0, 6.0, 7.5] {
            cats.push("Group B");
            vals.push(v);
        }
        let chart = boxplot(&cats, &vals)
            .title("Box Plot")
            .x_label("Group")
            .y_label("Measurement")
            .theme(theme.clone())
            .build();
        save(&chart, out, "boxplot", &mut manifest);
    }

    // Area
    {
        let x: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v * v * 0.1 + 2.0).collect();
        let chart = area(&x, &y)
            .title("Area Chart")
            .x_label("X")
            .y_label("Y")
            .theme(theme.clone())
            .build();
        save(&chart, out, "area", &mut manifest);
    }

    // Pie
    {
        let labels = vec!["Desktop", "Mobile", "Tablet", "Other"];
        let values = vec![45.0, 30.0, 15.0, 10.0];
        let chart = pie_labeled(&labels, &values)
            .title("Pie Chart")
            .theme(theme.clone())
            .build();
        save(&chart, out, "pie", &mut manifest);
    }

    // Donut
    {
        let labels = vec!["Pass", "Fail", "Skip"];
        let values = vec![78.0, 15.0, 7.0];
        let chart = pie_labeled(&labels, &values)
            .donut(0.5)
            .title("Donut Chart")
            .theme(theme.clone())
            .build();
        save(&chart, out, "donut", &mut manifest);
    }

    // Treemap
    {
        let labels = vec!["Rust", "Python", "JS", "Go", "C++", "Java"];
        let values = vec![35.0, 25.0, 20.0, 10.0, 6.0, 4.0];
        let chart = treemap(&labels, &values)
            .title("Treemap")
            .theme(theme.clone())
            .build();
        save(&chart, out, "treemap", &mut manifest);
    }

    // Stacked Bar
    {
        let cats = vec!["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"];
        let groups = vec![
            "Revenue", "Revenue", "Revenue", "Revenue", "Costs", "Costs", "Costs", "Costs",
        ];
        let vals = vec![50.0, 60.0, 55.0, 70.0, 30.0, 35.0, 40.0, 38.0];
        let chart = stacked_bar(&cats, &groups, &vals)
            .title("Stacked Bar")
            .x_label("Quarter")
            .y_label("Amount ($K)")
            .theme(theme.clone())
            .try_build()
            .expect("stacked bar build");
        save(&chart, out, "stacked_bar", &mut manifest);
    }

    // Grouped Bar
    {
        let cats = vec!["Q1", "Q2", "Q3", "Q1", "Q2", "Q3"];
        let groups = vec!["2024", "2024", "2024", "2025", "2025", "2025"];
        let vals = vec![40.0, 55.0, 48.0, 52.0, 63.0, 58.0];
        let chart = grouped_bar(&cats, &groups, &vals)
            .title("Grouped Bar")
            .x_label("Quarter")
            .y_label("Revenue ($K)")
            .theme(theme.clone())
            .try_build()
            .expect("grouped bar build");
        save(&chart, out, "grouped_bar", &mut manifest);
    }

    // Heatmap
    {
        let data = vec![
            vec![1.0, 0.8, 0.3, 0.1],
            vec![0.8, 1.0, 0.5, 0.2],
            vec![0.3, 0.5, 1.0, 0.7],
            vec![0.1, 0.2, 0.7, 1.0],
        ];
        let labels = vec!["A", "B", "C", "D"];
        let chart = heatmap(data)
            .with_row_labels(&labels)
            .with_col_labels(&labels)
            .annotate()
            .title("Correlation Heatmap")
            .theme(theme.clone())
            .build();
        save(&chart, out, "heatmap", &mut manifest);
    }

    // Multi-layer (scatter + trend line)
    {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.1, 18.0, 20.2];
        let chart = scatter(&x, &y)
            .trend_line()
            .title("Scatter with Trend Line")
            .x_label("Input")
            .y_label("Output")
            .theme(theme.clone())
            .build();
        save(&chart, out, "scatter_trend", &mut manifest);
    }

    // Horizontal bar (flipped coords)
    {
        let chart = Chart::new()
            .layer(
                Layer::new(MarkType::Bar)
                    .with_x(vec![0.0, 1.0, 2.0, 3.0, 4.0])
                    .with_y(vec![25.0, 40.0, 30.0, 55.0, 35.0])
                    .with_categories(vec![
                        "Rust".into(),
                        "Python".into(),
                        "Go".into(),
                        "TypeScript".into(),
                        "Java".into(),
                    ]),
            )
            .coord(esoc_chart::grammar::coord::CoordSystem::Flipped)
            .title("Horizontal Bar")
            .x_label("Language")
            .y_label("Popularity")
            .theme(theme.clone());
        save(&chart, out, "horizontal_bar", &mut manifest);
    }

    // ── Additional charts ─────────────────────────────────────────────

    // Dark theme scatter
    {
        let x = vec![1.0, 2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.0, 9.0, 10.0];
        let y = vec![3.2, 5.1, 4.8, 7.3, 6.9, 9.1, 8.5, 11.0, 10.2, 12.8];
        let cats = vec![
            "Train", "Train", "Train", "Train", "Train", "Test", "Test", "Test", "Test", "Test",
        ];
        let chart = scatter(&x, &y)
            .color_by(&cats)
            .title("Dark Theme Scatter")
            .x_label("Epoch")
            .y_label("Loss")
            .theme(NewTheme::dark())
            .build();
        save(&chart, out, "scatter_dark", &mut manifest);
    }

    // Multi-series line
    {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| (v * 0.3).sin() * 20.0 + 50.0).collect();
        let cats: Vec<String> = x
            .iter()
            .map(|v| {
                if *v < 15.0 {
                    "Sensor A".into()
                } else {
                    "Sensor B".into()
                }
            })
            .collect();
        let chart = line(&x, &y)
            .color_by(&cats)
            .title("Multi-Series Line")
            .x_label("Time (s)")
            .y_label("Temperature (°C)")
            .theme(theme.clone())
            .build();
        save(&chart, out, "line_multi", &mut manifest);
    }

    // Bar with error bars
    {
        let cats = vec!["Control", "Drug A", "Drug B", "Drug C"];
        let vals = vec![5.2, 8.1, 7.4, 9.6];
        let errs = vec![0.8, 1.2, 0.9, 1.5];
        let chart = bar(&cats, &vals)
            .error_bars(&errs)
            .title("Treatment Response")
            .x_label("Group")
            .y_label("Response (AU)")
            .theme(theme.clone())
            .build();
        save(&chart, out, "bar_error", &mut manifest);
    }

    // Scatter with annotations
    {
        use esoc_chart::grammar::annotation::Annotation;
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![10.0, 12.0, 15.0, 14.0, 18.0, 22.0, 25.0, 28.0];
        let chart = scatter(&x, &y)
            .title("Annotated Scatter")
            .x_label("Week")
            .y_label("Revenue ($K)")
            .hline(20.0)
            .vline(4.0)
            .theme(theme.clone())
            .build()
            .annotate(Annotation::band(15.0, 25.0));
        save(&chart, out, "scatter_annotated", &mut manifest);
    }

    // Subtitle + caption
    {
        let x: Vec<f64> = (0..12).map(|i| i as f64 + 1.0).collect();
        let y = vec![
            42.0, 45.0, 48.0, 52.0, 55.0, 60.0, 58.0, 62.0, 65.0, 68.0, 72.0, 75.0,
        ];
        let chart = Chart::new()
            .layer(Layer::new(MarkType::Line).with_x(x).with_y(y))
            .title("Monthly Active Users")
            .subtitle("Jan–Dec 2025")
            .caption("Source: internal analytics")
            .x_label("Month")
            .y_label("Users (K)")
            .theme(theme.clone());
        save(&chart, out, "line_subtitle", &mut manifest);
    }

    // Large scatter (many points)
    {
        let n = 200;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                t * 100.0 + (i as f64 * 0.7).sin() * 15.0
            })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                t * 80.0 + (i as f64 * 1.3).cos() * 20.0 + 10.0
            })
            .collect();
        let cats: Vec<String> = (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    "Cluster A".into()
                } else if i % 3 == 1 {
                    "Cluster B".into()
                } else {
                    "Cluster C".into()
                }
            })
            .collect();
        let chart = scatter(&x, &y)
            .color_by(&cats)
            .title("Dense Scatter (200 pts)")
            .x_label("Feature 1")
            .y_label("Feature 2")
            .theme(theme.clone())
            .build();
        save(&chart, out, "scatter_dense", &mut manifest);
    }

    // Faceted scatter
    {
        let x = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let y = vec![2.0, 4.0, 3.0, 5.0, 7.0, 6.0, 1.0, 3.0, 5.0, 4.0, 6.0, 8.0];
        let facets = vec![
            "2024", "2024", "2024", "2024", "2024", "2024", "2025", "2025", "2025", "2025", "2025",
            "2025",
        ];
        let chart = scatter(&x, &y)
            .facet_wrap(&facets, 2)
            .title("Faceted Scatter")
            .x_label("Month")
            .y_label("Sales")
            .theme(theme.clone())
            .build();
        save(&chart, out, "scatter_faceted", &mut manifest);
    }

    // Publication theme line
    {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.2).collect();
        let y: Vec<f64> = x.iter().map(|v| (-v * 0.3).exp() * 100.0).collect();
        let chart = line(&x, &y)
            .title("Exponential Decay")
            .x_label("Time (s)")
            .y_label("Concentration (mg/L)")
            .theme(NewTheme::publication())
            .build();
        save(&chart, out, "line_publication", &mut manifest);
    }

    // Donut with many slices
    {
        let labels = vec![
            "Chrome", "Safari", "Firefox", "Edge", "Opera", "Samsung", "Other",
        ];
        let values = vec![64.0, 19.0, 3.5, 5.0, 2.5, 2.0, 4.0];
        let chart = pie_labeled(&labels, &values)
            .donut(0.4)
            .title("Browser Market Share")
            .theme(theme.clone())
            .build();
        save(&chart, out, "donut_browsers", &mut manifest);
    }

    // Area with categories
    {
        let x: Vec<f64> = (0..20)
            .map(|i| i as f64)
            .collect::<Vec<_>>()
            .iter()
            .cloned()
            .cycle()
            .take(40)
            .collect();
        let y: Vec<f64> = (0..40)
            .map(|i| {
                if i < 20 {
                    (i as f64 * 0.3).sin() * 5.0 + 15.0
                } else {
                    ((i - 20) as f64 * 0.3).cos() * 4.0 + 10.0
                }
            })
            .collect();
        let cats: Vec<String> = (0..40)
            .map(|i| {
                if i < 20 {
                    "Upload".into()
                } else {
                    "Download".into()
                }
            })
            .collect();
        let chart = area(&x, &y)
            .color_by(&cats)
            .title("Network Traffic")
            .x_label("Hour")
            .y_label("Throughput (Mbps)")
            .theme(theme.clone())
            .build();
        save(&chart, out, "area_multi", &mut manifest);
    }

    // Heatmap (larger, no annotations)
    {
        let data: Vec<Vec<f64>> = (0..8)
            .map(|r| {
                (0..8)
                    .map(|c| {
                        let dist = ((r as f64 - 3.5).powi(2) + (c as f64 - 3.5).powi(2)).sqrt();
                        (4.0 - dist).max(0.0)
                    })
                    .collect()
            })
            .collect();
        let rows: Vec<String> = (0..8).map(|i| format!("R{}", i + 1)).collect();
        let cols: Vec<String> = (0..8).map(|i| format!("C{}", i + 1)).collect();
        let chart = heatmap(data)
            .with_row_labels(&rows)
            .with_col_labels(&cols)
            .title("8x8 Radial Heatmap")
            .theme(theme.clone())
            .build();
        save(&chart, out, "heatmap_large", &mut manifest);
    }

    manifest
}

fn save(chart: &Chart, out: &Path, name: &str, manifest: &mut Vec<(String, std::path::PathBuf)>) {
    let path = out.join(format!("{name}.png"));
    chart
        .save_png_to(&path)
        .unwrap_or_else(|e| panic!("failed to save {name}.png: {e}"));
    println!("  saved {}", path.display());
    manifest.push((name.to_string(), path));
}

// ── Gemini API ──────────────────────────────────────────────────────

fn ask_gemini(
    client: &reqwest::blocking::Client,
    api_key: &str,
    chart_name: &str,
    png_bytes: &[u8],
) -> Result<ChartFeedback, String> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);

    let prompt = format!(
        r#"You are a data-visualization expert reviewing a chart rendered by a Rust charting library.

The chart type is: **{chart_name}**

Analyze this chart image and return ONLY valid JSON (no markdown fences) with this schema:
{{
  "chart_type": "{chart_name}",
  "overall_score": <1-10>,
  "strengths": ["..."],
  "issues": ["..."],
  "suggestions": ["..."]
}}

Evaluate: layout, readability, color choices, axis labels, title, legend, data-ink ratio, and anything that looks off or could be improved. Be specific and actionable."#
    );

    let body = serde_json::json!({
        "contents": [{
            "parts": [
                { "text": prompt },
                { "inline_data": { "mime_type": "image/png", "data": b64 } }
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    });

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    );

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        return Err(format!("Gemini API {status}: {text}"));
    }

    let json: serde_json::Value = resp.json().map_err(|e| format!("parse response: {e}"))?;

    // Extract text from Gemini response
    let text = json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or("missing text in Gemini response")?;

    serde_json::from_str::<ChartFeedback>(text)
        .map_err(|e| format!("parse feedback JSON: {e}\nraw: {text}"))
}

// ── Main ────────────────────────────────────────────────────────────

fn main() {
    let out = Path::new("examples/chart_audit/audit_output");
    fs::create_dir_all(out).expect("create output dir");

    println!("Generating charts...");
    let manifest = generate_charts(out);
    println!("\nGenerated {} chart PNGs.\n", manifest.len());

    // Check for Gemini API key
    let api_key = match std::env::var("GEMINI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            println!("Set GEMINI_API_KEY to enable AI feedback.");
            println!("PNGs are in: {}", out.display());
            println!("\nExample: GEMINI_API_KEY=your-key cargo run -p chart-audit");
            return;
        }
    };

    println!("Sending charts to Gemini for review...\n");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .expect("build HTTP client");

    let mut report = AuditReport { charts: Vec::new() };
    let mut errors: HashMap<String, String> = HashMap::new();

    for (name, path) in &manifest {
        print!("  reviewing {name}...");
        let png_bytes = fs::read(path).expect("read PNG");
        match ask_gemini(&client, &api_key, name, &png_bytes) {
            Ok(feedback) => {
                println!(" score: {}/10", feedback.overall_score);
                report.charts.push(feedback);
            }
            Err(e) => {
                println!(" ERROR: {e}");
                errors.insert(name.clone(), e);
            }
        }
    }

    // Save structured report
    let report_path = out.join("audit_report.json");
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    fs::write(&report_path, &json).expect("write report");
    println!("\nReport saved to: {}", report_path.display());

    // Print summary
    if !report.charts.is_empty() {
        let avg: f64 = report
            .charts
            .iter()
            .map(|c| c.overall_score as f64)
            .sum::<f64>()
            / report.charts.len() as f64;
        println!("\n── Summary ──");
        println!("  Charts reviewed: {}", report.charts.len());
        println!("  Average score:   {avg:.1}/10");
        if !errors.is_empty() {
            println!("  Errors:          {}", errors.len());
        }
        println!();

        // Print per-chart highlights
        for fb in &report.charts {
            println!("  {} ({}/10)", fb.chart_type, fb.overall_score);
            if let Some(top_issue) = fb.issues.first() {
                println!("    top issue: {top_issue}");
            }
            if let Some(top_suggestion) = fb.suggestions.first() {
                println!("    suggestion: {top_suggestion}");
            }
        }
    }
}
