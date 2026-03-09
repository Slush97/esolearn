// SPDX-License-Identifier: MIT OR Apache-2.0
//! Train logistic regression on a synthetic binary dataset, then visualize:
//! - ROC curve with AUC
//! - Confusion matrix heatmap
//! - Classification report bar chart

use esoc_chart::prelude::*;
use scry_learn::prelude::*;

fn main() -> Result<()> {
    // ── Synthetic binary classification data ──────────────────────────
    // Two Gaussian blobs: class 0 centred at (1,1), class 1 at (3,3).
    let n_per_class = 100;
    let mut f0 = Vec::new(); // feature 0
    let mut f1 = Vec::new(); // feature 1
    let mut target = Vec::new();

    let mut rng = SimpleRng::new(42);
    for _ in 0..n_per_class {
        f0.push(1.0 + rng.normal());
        f1.push(1.0 + rng.normal());
        target.push(0.0);
    }
    for _ in 0..n_per_class {
        f0.push(3.0 + rng.normal());
        f1.push(3.0 + rng.normal());
        target.push(1.0);
    }

    let dataset = Dataset::new(
        vec![f0, f1],
        target,
        vec!["x0".into(), "x1".into()],
        "class",
    );

    // ── Train / test split ───────────────────────────────────────────
    let (train, test) = train_test_split(&dataset, 0.3, 42);

    // ── Fit logistic regression ──────────────────────────────────────
    let mut model = LogisticRegression::new();
    model.fit(&train).expect("fit failed");

    let test_rows = to_row_major(&test.features);
    let y_pred = model.predict(&test_rows).expect("predict failed");
    let y_proba = model.predict_proba(&test_rows).expect("predict_proba failed");
    // For binary classification, scores = P(class=1)
    let y_scores: Vec<f64> = y_proba.iter().map(|p: &Vec<f64>| {
        if p.len() == 2 { p[1] } else { p[0] }
    }).collect();

    // ── 1. ROC Curve ─────────────────────────────────────────────────
    let roc = roc_curve(&test.target, &y_scores);
    let roc_fig = roc.roc_figure();
    roc_fig.save_svg("logistic_roc.svg")?;
    println!("Saved logistic_roc.svg (AUC = {:.3})", roc.auc);

    // ── 2. Confusion Matrix ──────────────────────────────────────────
    let cm = confusion_matrix(&test.target, &y_pred);
    let cm_fig = cm.figure();
    cm_fig.save_svg("logistic_confusion.svg")?;
    println!("Saved logistic_confusion.svg");

    // ── 3. Classification Report ─────────────────────────────────────
    let report = classification_report(&test.target, &y_pred);
    println!("{report}");
    let report_fig = report.figure();
    report_fig.save_svg("logistic_report.svg")?;
    println!("Saved logistic_report.svg");

    Ok(())
}

/// Transpose column-major features to row-major for `predict`/`predict_proba`.
fn to_row_major(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if cols.is_empty() { return vec![]; }
    let n_samples = cols[0].len();
    (0..n_samples).map(|i| cols.iter().map(|col| col[i]).collect()).collect()
}

// ── Minimal RNG for the example (no external deps) ──────────────────
struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.0
    }

    /// Uniform [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
