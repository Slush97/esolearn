// SPDX-License-Identifier: MIT OR Apache-2.0
//! Visualize feature distributions with histograms and box plots.
//!
//! Generates a synthetic 3-feature dataset and creates:
//! - Per-feature histograms
//! - Side-by-side box plots

use esoc_chart::prelude::*;
use scry_learn::prelude::*;

fn main() -> Result<()> {
    // ── Synthetic dataset: 3 features with different distributions ───
    let n = 300;
    let mut rng = SimpleRng::new(99);

    // Feature 0: standard normal
    let f0: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    // Feature 1: right-skewed (exponential-ish)
    let f1: Vec<f64> = (0..n).map(|_| -rng.uniform().max(1e-15).ln()).collect();
    // Feature 2: bimodal (mixture of two normals)
    let f2: Vec<f64> = (0..n)
        .map(|_| {
            if rng.uniform() < 0.4 {
                -2.0 + rng.normal() * 0.5
            } else {
                2.0 + rng.normal() * 0.7
            }
        })
        .collect();

    let target: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
    let dataset = Dataset::new(
        vec![f0.clone(), f1.clone(), f2.clone()],
        target,
        vec!["Normal".into(), "Exponential".into(), "Bimodal".into()],
        "class",
    );

    // ── 1. Histogram per feature ─────────────────────────────────────
    let feature_names = &dataset.feature_names;
    let features = [&f0, &f1, &f2];

    for (i, feat) in features.iter().enumerate() {
        let mut fig = Figure::new()
            .size(600.0, 400.0)
            .title(format!("Distribution of \"{}\"", feature_names[i]));

        let ax = fig.add_axes();
        ax.x_label(&feature_names[i]).y_label("Count");
        ax.histogram(feat).label(&feature_names[i]).bins(25).done();

        let path = format!("hist_{}.svg", feature_names[i].to_lowercase());
        fig.save_svg(&path)?;
        println!("Saved {path}");
    }

    // ── 2. Box plots of all features ─────────────────────────────────
    let mut fig = Figure::new().size(600.0, 450.0).title("Feature Box Plots");

    let ax = fig.add_axes();
    ax.x_label("Feature").y_label("Value");
    ax.boxplot(vec![f0, f1, f2])
        .labels(feature_names.clone())
        .done();

    fig.save_svg("feature_boxplots.svg")?;
    println!("Saved feature_boxplots.svg");

    Ok(())
}

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
