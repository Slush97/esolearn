// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scatter plot with color-coded categories — no feature flags needed.

use esoc_chart::v2::*;

fn main() -> esoc_chart::error::Result<()> {
    // Three clusters of points
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut cats = Vec::new();

    // Simple LCG for reproducibility (no deps)
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

    for (cx, cy, label) in [(2.0, 2.0, "Alpha"), (6.0, 6.0, "Beta"), (6.0, 2.0, "Gamma")] {
        for _ in 0..40 {
            x.push(cx + normal() * 0.7);
            y.push(cy + normal() * 0.7);
            cats.push(label);
        }
    }

    let svg = scatter(&x, &y)
        .color_by(&cats)
        .title("Three Clusters")
        .x_label("feature 1")
        .y_label("feature 2")
        .to_svg()?;

    std::fs::write("scatter_categories.svg", &svg)?;
    println!("Saved scatter_categories.svg ({} bytes)", svg.len());
    Ok(())
}
