// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scatter plot of K-Means clustering results on synthetic data.

use esoc_chart::prelude::*;
use scry_learn::prelude::*;

fn main() -> Result<()> {
    // ── Generate 3 Gaussian clusters ─────────────────────────────────
    let centers = [(1.0, 1.0), (5.0, 1.0), (3.0, 5.0)];
    let n_per_cluster = 80;

    let mut f0 = Vec::new();
    let mut f1 = Vec::new();
    let mut target = Vec::new();

    let mut rng = SimpleRng::new(123);
    for (ci, &(cx, cy)) in centers.iter().enumerate() {
        for _ in 0..n_per_cluster {
            f0.push(cx + rng.normal() * 0.8);
            f1.push(cy + rng.normal() * 0.8);
            target.push(ci as f64);
        }
    }

    let dataset = Dataset::new(
        vec![f0.clone(), f1.clone()],
        target.clone(),
        vec!["x".into(), "y".into()],
        "cluster",
    );

    // ── Run K-Means ──────────────────────────────────────────────────
    let mut kmeans = KMeans::new(3).seed(42).max_iter(100);
    kmeans.fit(&dataset).expect("kmeans fit failed");
    let rows = to_row_major(&dataset.features);
    let labels = kmeans.predict(&rows).expect("kmeans predict failed");

    // ── Plot: ground truth vs predicted clusters side by side ────────
    // Ground truth
    let mut fig = Figure::new()
        .size(800.0, 500.0)
        .title("K-Means Clustering — Ground Truth Coloring");

    let palette = esoc_gfx::palette::Palette::tab10();
    let ax = fig.add_axes();
    ax.x_label("x").y_label("y");

    for ci in 0..3 {
        let (sx, sy): (Vec<f64>, Vec<f64>) = f0
            .iter()
            .zip(f1.iter())
            .zip(target.iter())
            .filter(|(_, &t)| t as usize == ci)
            .map(|((x, y), _)| (*x, *y))
            .unzip();
        ax.scatter(&sx, &sy)
            .label(format!("True cluster {ci}"))
            .color(palette.get(ci))
            .radius(4.0)
            .done();
    }
    fig.save_svg("clusters_truth.svg")?;
    println!("Saved clusters_truth.svg");

    // Predicted labels
    let mut fig2 = Figure::new()
        .size(800.0, 500.0)
        .title("K-Means Clustering — Predicted Labels");

    let ax2 = fig2.add_axes();
    ax2.x_label("x").y_label("y");

    for ci in 0..3_usize {
        let (sx, sy): (Vec<f64>, Vec<f64>) = f0
            .iter()
            .zip(f1.iter())
            .zip(labels.iter())
            .filter(|&(_, &l)| l == ci)
            .map(|((x, y), _)| (*x, *y))
            .unzip();
        ax2.scatter(&sx, &sy)
            .label(format!("Cluster {ci}"))
            .color(palette.get(ci))
            .radius(4.0)
            .done();
    }
    fig2.save_svg("clusters_predicted.svg")?;
    println!("Saved clusters_predicted.svg");

    Ok(())
}

fn to_row_major(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if cols.is_empty() {
        return vec![];
    }
    let n_samples = cols[0].len();
    (0..n_samples)
        .map(|i| cols.iter().map(|col| col[i]).collect())
        .collect()
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
