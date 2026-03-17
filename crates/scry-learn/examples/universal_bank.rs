//! Universal Bank personal loan analysis with scry-learn.
//!
//! Loads the UniversalBank CSV, cleans the data, normalizes features,
//! and runs classification models to predict personal loan acceptance.
//!
//! Run: `cargo run --example universal_bank -p scry-learn --features csv --release`

use std::path::PathBuf;
use std::time::Instant;

use scry_learn::dataset::Dataset;
use scry_learn::linear::LogisticRegression;
use scry_learn::metrics::{accuracy, classification_report};
use scry_learn::naive_bayes::GaussianNb;
use scry_learn::neighbors::KnnClassifier;
use scry_learn::preprocess::{MinMaxScaler, SimpleImputer, StandardScaler, Transformer};
use scry_learn::split::{cross_val_score_stratified, train_test_split, ScoringFn};
use scry_learn::svm::LinearSVC;
use scry_learn::tree::{
    DecisionTreeClassifier, GradientBoostingClassifier, RandomForestClassifier,
};

fn datasets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("datasets")
}

fn main() -> scry_learn::error::Result<()> {
    let csv_path = datasets_dir().join("universal_bank.csv");
    let csv_str = csv_path.to_str().expect("invalid path");

    // ── 1. Load ─────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("  Universal Bank — Personal Loan Prediction");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut data = Dataset::from_csv(csv_str, "Personal Loan")?;
    println!(
        "Loaded: {} samples, {} features, {} classes\n",
        data.n_samples(),
        data.n_features(),
        data.n_classes()
    );

    // ── 2. Explore — raw summary ────────────────────────────────────
    println!("── Raw Data Summary ──");
    data.describe();
    println!();

    // ── 3. Clean — drop non-predictive columns (ID, ZIP Code) ──────
    let drop_cols: Vec<&str> = vec!["ID", "ZIP Code"];
    let keep: Vec<usize> = data
        .feature_names
        .iter()
        .enumerate()
        .filter(|(_, name)| !drop_cols.contains(&name.as_str()))
        .map(|(i, _)| i)
        .collect();

    let kept_names: Vec<String> = keep.iter().map(|&i| data.feature_names[i].clone()).collect();
    let kept_features: Vec<Vec<f64>> = keep.iter().map(|&i| data.features[i].clone()).collect();

    data = Dataset::new(
        kept_features,
        data.target.clone(),
        kept_names,
        &data.target_name,
    );

    println!("Dropped: {:?}", drop_cols);
    println!(
        "Remaining features ({}): {:?}\n",
        data.n_features(),
        data.feature_names
    );

    // ── 4. Clean — impute any NaN values ────────────────────────────
    let nan_counts: Vec<(String, usize)> = data
        .feature_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let nans = data.features[i].iter().filter(|v| v.is_nan()).count();
            (name.clone(), nans)
        })
        .filter(|(_, c)| *c > 0)
        .collect();

    if nan_counts.is_empty() {
        println!("No missing values detected — skipping imputation.\n");
    } else {
        println!("Missing values found:");
        for (name, count) in &nan_counts {
            println!("  {name}: {count} NaN");
        }
        let mut imputer = SimpleImputer::new();
        imputer.fit(&data)?;
        imputer.transform(&mut data)?;
        println!("Imputed with column means.\n");
    }

    // ── 5. Check for negative Experience values ─────────────────────
    if let Some(exp_idx) = data.feature_names.iter().position(|n| n == "Experience") {
        let neg_count = data.features[exp_idx].iter().filter(|&&v| v < 0.0).count();
        if neg_count > 0 {
            println!("Found {neg_count} negative Experience values — clipping to 0.");
            for v in &mut data.features[exp_idx] {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
            println!();
        }
    }

    // ── 6. Cleaned summary ──────────────────────────────────────────
    println!("── Cleaned Data Summary ──");
    data.describe();
    println!();

    // ── 7. Class distribution ───────────────────────────────────────
    let positive = data.target.iter().filter(|&&v| v == 1.0).count();
    let negative = data.target.iter().filter(|&&v| v == 0.0).count();
    println!("── Class Distribution ──");
    println!(
        "  Declined (0): {} ({:.1}%)",
        negative,
        100.0 * negative as f64 / data.n_samples() as f64
    );
    println!(
        "  Accepted (1): {} ({:.1}%)",
        positive,
        100.0 * positive as f64 / data.n_samples() as f64
    );
    println!("  → Imbalanced dataset — accuracy alone may be misleading.\n");

    // ── 8. Normalize — StandardScaler for distance-based models ─────
    let mut scaled = data.clone();
    let mut scaler = StandardScaler::new();
    scaler.fit(&scaled)?;
    scaler.transform(&mut scaled)?;

    // Also prepare a MinMax-scaled version
    let mut minmax_scaled = data.clone();
    let mut mm_scaler = MinMaxScaler::new();
    mm_scaler.fit(&minmax_scaled)?;
    mm_scaler.transform(&mut minmax_scaled)?;

    // ── 9. Train/test split ─────────────────────────────────────────
    let (train, test) = train_test_split(&data, 0.2, 42);
    let (train_s, test_s) = train_test_split(&scaled, 0.2, 42);
    println!(
        "Train/Test split: {} / {} (80/20, seed=42)\n",
        train.n_samples(),
        test.n_samples()
    );

    // ── 10. Model comparison — 5-fold stratified CV ─────────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("  5-Fold Stratified Cross-Validation");
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  {:25} {:>10} {:>8} {:>12}",
        "Model", "Mean Acc", "Std", "Time"
    );
    println!("  {}", "─".repeat(55));

    let scorer: ScoringFn = accuracy;

    let models: Vec<(&str, f64, f64, f64)> = vec![
        cv_run("Decision Tree", &DecisionTreeClassifier::new().max_depth(8), &data, scorer),
        cv_run(
            "Random Forest",
            &RandomForestClassifier::new().n_estimators(50).max_depth(10).seed(42),
            &data,
            scorer,
        ),
        cv_run(
            "Gradient Boosting",
            &GradientBoostingClassifier::new()
                .n_estimators(100)
                .max_depth(5)
                .learning_rate(0.1),
            &data,
            scorer,
        ),
        cv_run("Gaussian NB", &GaussianNb::new(), &data, scorer),
        cv_run(
            "Logistic Regression",
            &LogisticRegression::new().max_iter(500).learning_rate(0.01),
            &scaled,
            scorer,
        ),
        cv_run("KNN (k=5)", &KnnClassifier::new().k(5), &scaled, scorer),
        cv_run(
            "LinearSVC",
            &LinearSVC::new().c(1.0).max_iter(1000),
            &scaled,
            scorer,
        ),
    ];

    for (name, mean, std, ms) in &models {
        let time_str = if *ms < 1000.0 {
            format!("{ms:.1} ms")
        } else {
            format!("{:.2} s", ms / 1000.0)
        };
        println!("  {name:25} {mean:>10.4} {std:>8.4} {time_str:>12}");
    }
    println!();

    // ── 11. Best model — full evaluation on test set ────────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("  Test-Set Evaluation — Random Forest");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut rf = RandomForestClassifier::new()
        .n_estimators(100)
        .max_depth(10)
        .seed(42);
    rf.fit(&train)?;
    let test_rows = to_row_major(&test);
    let preds = rf.predict(&test_rows)?;

    let report = classification_report(&test.target, &preds);
    println!("{report}\n");

    // ── 12. Also show Logistic Regression on scaled data ────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("  Test-Set Evaluation — Logistic Regression (scaled)");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut lr = LogisticRegression::new().max_iter(500).learning_rate(0.01);
    lr.fit(&train_s)?;
    let test_s_rows = to_row_major(&test_s);
    let preds_lr = lr.predict(&test_s_rows)?;

    let report_lr = classification_report(&test_s.target, &preds_lr);
    println!("{report_lr}\n");

    println!("═══════════════════════════════════════════════════════════");
    println!("  Done.");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}

/// Convert column-major Dataset features to row-major Vec<Vec<f64>> for predict().
fn to_row_major(data: &Dataset) -> Vec<Vec<f64>> {
    let n = data.n_samples();
    let m = data.n_features();
    (0..n)
        .map(|i| (0..m).map(|j| data.features[j][i]).collect())
        .collect()
}

fn cv_run<M: scry_learn::pipeline::PipelineModel + Clone + Send + Sync>(
    name: &str,
    model: &M,
    data: &Dataset,
    scorer: ScoringFn,
) -> (&'static str, f64, f64, f64) {
    let name_static: &'static str = Box::leak(name.to_string().into_boxed_str());
    let start = Instant::now();
    let scores = cross_val_score_stratified(model, data, 5, scorer, 42).unwrap_or_else(|e| {
        eprintln!("  WARN: {name} failed: {e}");
        vec![0.0; 5]
    });
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    (name_static, mean, variance.sqrt(), elapsed)
}
