// SPDX-License-Identifier: MIT OR Apache-2.0
//! Standalone scatter plot — no feature flags needed.

use esoc_chart::v2::scatter;

fn main() -> esoc_chart::error::Result<()> {
    // Some synthetic data: y ≈ x² + noise
    let x: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.2).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi * xi + ((xi * 7.0).sin()) * 1.5)
        .collect();

    let svg = scatter(&x, &y)
        .title("y = x² + noise")
        .x_label("x")
        .y_label("y")
        .to_svg()?;

    std::fs::write("basic_scatter.svg", &svg)?;
    println!("Saved basic_scatter.svg ({} bytes)", svg.len());
    Ok(())
}
