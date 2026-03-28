// SPDX-License-Identifier: MIT OR Apache-2.0
//! Standalone line chart — no feature flags needed.

use esoc_chart::v2::{line, Chart, Layer, MarkType};

fn main() -> esoc_chart::error::Result<()> {
    // Sine and cosine waves
    let x: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let y_sin: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
    let y_cos: Vec<f64> = x.iter().map(|&xi| xi.cos()).collect();

    // Single line chart: sin(x)
    let svg = line(&x, &y_sin)
        .title("sin(x)")
        .x_label("x")
        .y_label("amplitude")
        .to_svg()?;

    std::fs::write("basic_line.svg", &svg)?;
    println!("Saved basic_line.svg ({} bytes)", svg.len());

    // Multi-layer chart: sin + cos via Grammar API
    let chart = Chart::new()
        .layer(Layer::new(MarkType::Line).with_x(x.clone()).with_y(y_sin))
        .layer(Layer::new(MarkType::Line).with_x(x).with_y(y_cos))
        .title("sin(x) and cos(x)")
        .x_label("x")
        .y_label("amplitude")
        .size(900.0, 500.0);

    let svg2 = chart.to_svg()?;
    std::fs::write("multi_line.svg", &svg2)?;
    println!("Saved multi_line.svg ({} bytes)", svg2.len());

    Ok(())
}
