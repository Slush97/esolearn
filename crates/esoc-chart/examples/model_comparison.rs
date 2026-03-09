// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare multiple classifiers' ROC curves on the same plot.
//!
//! Trains Logistic Regression, Random Forest, and KNN on a synthetic dataset,
//! then overlays their ROC curves for visual comparison.

use esoc_chart::prelude::*;
use scry_learn::prelude::*;

fn main() -> Result<()> {
    // ── Synthetic binary dataset with some overlap ───────────────────
    let n = 150;
    let mut f0 = Vec::with_capacity(2 * n);
    let mut f1 = Vec::with_capacity(2 * n);
    let mut target = Vec::with_capacity(2 * n);

    let mut rng = SimpleRng::new(7);
    for _ in 0..n {
        f0.push(rng.normal() * 1.5);
        f1.push(rng.normal() * 1.5);
        target.push(0.0);
    }
    for _ in 0..n {
        f0.push(2.0 + rng.normal() * 1.5);
        f1.push(2.0 + rng.normal() * 1.5);
        target.push(1.0);
    }

    let dataset = Dataset::new(
        vec![f0, f1],
        target,
        vec!["f0".into(), "f1".into()],
        "class",
    );
    let (train, test) = train_test_split(&dataset, 0.3, 42);

    // ── Fit models ───────────────────────────────────────────────────
    let test_rows = to_row_major(&test.features);

    let mut lr = LogisticRegression::new();
    lr.fit(&train).expect("LR fit");
    let lr_proba = lr.predict_proba(&test_rows).expect("LR proba");
    let lr_scores: Vec<f64> = lr_proba.iter().map(|p: &Vec<f64>| if p.len() == 2 { p[1] } else { p[0] }).collect();

    let mut rf = RandomForestClassifier::new().n_estimators(50).seed(42);
    rf.fit(&train).expect("RF fit");
    let rf_proba = rf.predict_proba(&test_rows).expect("RF proba");
    let rf_scores: Vec<f64> = rf_proba.iter().map(|p: &Vec<f64>| if p.len() == 2 { p[1] } else { p[0] }).collect();

    let mut knn = KnnClassifier::new().k(5);
    knn.fit(&train).expect("KNN fit");
    let knn_proba = knn.predict_proba(&test_rows).expect("KNN proba");
    let knn_scores: Vec<f64> = knn_proba.iter().map(|p: &Vec<f64>| if p.len() == 2 { p[1] } else { p[0] }).collect();

    // ── Compute ROC curves ───────────────────────────────────────────
    let roc_lr = roc_curve(&test.target, &lr_scores);
    let roc_rf = roc_curve(&test.target, &rf_scores);
    let roc_knn = roc_curve(&test.target, &knn_scores);

    // ── Plot all on one figure ───────────────────────────────────────
    let mut fig = Figure::new()
        .size(700.0, 650.0)
        .title("Model Comparison — ROC Curves");

    let ax = fig.add_axes();
    ax.x_label("False Positive Rate")
        .y_label("True Positive Rate")
        .x_range(0.0, 1.0)
        .y_range(0.0, 1.0);

    ax.line(&roc_lr.fpr, &roc_lr.tpr)
        .label(format!("Logistic Regression (AUC={:.3})", roc_lr.auc))
        .color(Color::from_hex("#1f77b4").unwrap())
        .width(2.0)
        .done();

    ax.line(&roc_rf.fpr, &roc_rf.tpr)
        .label(format!("Random Forest (AUC={:.3})", roc_rf.auc))
        .color(Color::from_hex("#ff7f0e").unwrap())
        .width(2.0)
        .done();

    ax.line(&roc_knn.fpr, &roc_knn.tpr)
        .label(format!("KNN k=5 (AUC={:.3})", roc_knn.auc))
        .color(Color::from_hex("#2ca02c").unwrap())
        .width(2.0)
        .done();

    // Diagonal reference
    ax.line(&[0.0, 1.0], &[0.0, 1.0])
        .color(Color::GRAY)
        .dash(&[5.0, 5.0])
        .width(1.0)
        .done();

    fig.save_svg("model_comparison_roc.svg")?;
    println!("Saved model_comparison_roc.svg");
    println!("  LR  AUC = {:.3}", roc_lr.auc);
    println!("  RF  AUC = {:.3}", roc_rf.auc);
    println!("  KNN AUC = {:.3}", roc_knn.auc);

    Ok(())
}

fn to_row_major(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if cols.is_empty() { return vec![]; }
    let n_samples = cols[0].len();
    (0..n_samples).map(|i| cols.iter().map(|col| col[i]).collect()).collect()
}

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
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
