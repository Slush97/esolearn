// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare multiple classifiers' ROC curves on the same plot.
//!
//! Trains Logistic Regression, Random Forest, and KNN on a synthetic dataset,
//! then overlays their ROC curves for visual comparison.

use esoc_chart::express::line;
use scry_learn::prelude::*;

fn main() -> esoc_chart::error::Result<()> {
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
    let test_rows = to_row_major(&test.features);

    let mut lr = LogisticRegression::new();
    lr.fit(&train).expect("LR fit");
    let lr_scores = positive_class_scores(&lr.predict_proba(&test_rows).expect("LR proba"));

    let mut rf = RandomForestClassifier::new().n_estimators(50).seed(42);
    rf.fit(&train).expect("RF fit");
    let rf_scores = positive_class_scores(&rf.predict_proba(&test_rows).expect("RF proba"));

    let mut knn = KnnClassifier::new().k(5);
    knn.fit(&train).expect("KNN fit");
    let knn_scores = positive_class_scores(&knn.predict_proba(&test_rows).expect("KNN proba"));

    let roc_lr = roc_curve(&test.target, &lr_scores);
    let roc_rf = roc_curve(&test.target, &rf_scores);
    let roc_knn = roc_curve(&test.target, &knn_scores);

    let lr_label = format!("Logistic Regression (AUC={:.3})", roc_lr.auc);
    let rf_label = format!("Random Forest (AUC={:.3})", roc_rf.auc);
    let knn_label = format!("KNN k=5 (AUC={:.3})", roc_knn.auc);
    let diag_label = "Random classifier".to_string();

    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut series = Vec::new();
    for (label, roc) in [
        (&lr_label, &roc_lr),
        (&rf_label, &roc_rf),
        (&knn_label, &roc_knn),
    ] {
        for (&fpr, &tpr) in roc.fpr.iter().zip(roc.tpr.iter()) {
            x.push(fpr);
            y.push(tpr);
            series.push(label.clone());
        }
    }
    for (&fx, &fy) in [0.0, 1.0].iter().zip([0.0, 1.0].iter()) {
        x.push(fx);
        y.push(fy);
        series.push(diag_label.clone());
    }

    line(&x, &y)
        .color_by(&series)
        .title("Model Comparison — ROC Curves")
        .x_label("False Positive Rate")
        .y_label("True Positive Rate")
        .x_domain(0.0, 1.0)
        .y_domain(0.0, 1.0)
        .size(700.0, 650.0)
        .save_svg("model_comparison_roc.svg")?;

    println!("Saved model_comparison_roc.svg");
    println!("  LR  AUC = {:.3}", roc_lr.auc);
    println!("  RF  AUC = {:.3}", roc_rf.auc);
    println!("  KNN AUC = {:.3}", roc_knn.auc);

    Ok(())
}

fn positive_class_scores(proba: &[Vec<f64>]) -> Vec<f64> {
    proba
        .iter()
        .map(|p| if p.len() == 2 { p[1] } else { p[0] })
        .collect()
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
