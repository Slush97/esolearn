//! Industry benchmark report: scry-learn 5-fold CV accuracy on UCI datasets.
//!
//! Runs cross-validated accuracy for every model on every UCI dataset
//! and prints a formatted comparison table to stdout.
//!
//! Run: `cargo run --example industry_report -p scry-learn --release`

use std::path::PathBuf;
use std::time::Instant;

use scry_learn::dataset::Dataset;
use scry_learn::linear::LogisticRegression;
use scry_learn::metrics::accuracy;
use scry_learn::naive_bayes::GaussianNb;
use scry_learn::neighbors::KnnClassifier;
use scry_learn::preprocess::{StandardScaler, Transformer};
use scry_learn::split::{cross_val_score_stratified, ScoringFn};
use scry_learn::svm::LinearSVC;
use scry_learn::tree::{
    DecisionTreeClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
};

// ─── Fixture loading ─────────────────────────────────────────────

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_features_csv(name: &str) -> (Vec<Vec<f64>>, Vec<String>) {
    let path = fixtures_dir().join(name);
    let mut rdr = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let headers: Vec<String> = rdr.headers().unwrap().iter().map(String::from).collect();
    let n_cols = headers.len();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        let row: Vec<f64> = record.iter().map(|s| s.parse::<f64>().unwrap()).collect();
        rows.push(row);
    }
    let mut cols = vec![vec![0.0; rows.len()]; n_cols];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            cols[j][i] = val;
        }
    }
    (cols, headers)
}

fn load_target_csv(name: &str) -> Vec<f64> {
    let path = fixtures_dir().join(name);
    let mut rdr = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let mut target = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        target.push(record[0].parse::<f64>().unwrap());
    }
    target
}

fn load_dataset(base: &str) -> Dataset {
    let (features, feat_names) = load_features_csv(&format!("{base}_features.csv"));
    let target = load_target_csv(&format!("{base}_target.csv"));
    Dataset::new(features, target, feat_names, "target")
}

// ─── Model result ────────────────────────────────────────────────

struct ModelResult {
    name: String,
    mean_accuracy: f64,
    std_accuracy: f64,
    fold_scores: Vec<f64>,
    cv_time_ms: f64,
}

fn cv_result(name: &str, scores: Vec<f64>, elapsed_ms: f64) -> ModelResult {
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    ModelResult {
        name: name.to_string(),
        mean_accuracy: mean,
        std_accuracy: variance.sqrt(),
        fold_scores: scores,
        cv_time_ms: elapsed_ms,
    }
}

/// Run 5-fold stratified CV for a concrete model type.
fn run_cv<M: scry_learn::pipeline::PipelineModel + Clone + Send + Sync>(
    name: &str,
    model: &M,
    data: &Dataset,
) -> ModelResult {
    let scorer: ScoringFn = accuracy;
    let start = Instant::now();
    let scores = cross_val_score_stratified(model, data, 5, scorer, 42).unwrap_or_else(|e| {
        eprintln!("  WARN: {name} failed: {e}");
        vec![0.0; 5]
    });
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    cv_result(name, scores, elapsed)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  scry-learn Industry Benchmark Report");
    println!("  5-Fold Stratified Cross-Validation on UCI Datasets");
    println!("═══════════════════════════════════════════════════════════════════");

    let datasets = ["iris", "wine", "breast_cancer", "digits"];

    for &ds_name in &datasets {
        let data = load_dataset(ds_name);
        let n_classes = data.n_classes();

        println!("\n┌─────────────────────────────────────────────────────────────────┐");
        println!(
            "│  Dataset: {:<15} ({} samples, {} features, {} classes)  │",
            ds_name,
            data.n_samples(),
            data.n_features(),
            n_classes,
        );
        println!("├─────────────────────────┬──────────┬────────┬─────────────────┤");
        println!(
            "│ {:23} │ {:>8} │ {:>6} │ {:>15} │",
            "Model", "Mean Acc", "Std", "CV Time"
        );
        println!("├─────────────────────────┼──────────┼────────┼─────────────────┤");

        // Scale data for models that need it
        let mut scaled = data.clone();
        let mut scaler = StandardScaler::new();
        scaler.fit(&scaled).unwrap();
        scaler.transform(&mut scaled).unwrap();

        let models: Vec<ModelResult> = vec![
            run_cv(
                "Decision Tree",
                &DecisionTreeClassifier::new().max_depth(10),
                &data,
            ),
            run_cv(
                "Random Forest",
                &RandomForestClassifier::new()
                    .n_estimators(20)
                    .max_depth(10)
                    .seed(42),
                &data,
            ),
            run_cv(
                "Gradient Boosting",
                &GradientBoostingClassifier::new()
                    .n_estimators(100)
                    .max_depth(5)
                    .learning_rate(0.1),
                &data,
            ),
            run_cv(
                "HistGBT",
                &HistGradientBoostingClassifier::new()
                    .n_estimators(100)
                    .max_depth(6)
                    .learning_rate(0.1),
                &data,
            ),
            run_cv(
                "Logistic Regression",
                &LogisticRegression::new().max_iter(500).learning_rate(0.01),
                &scaled,
            ),
            run_cv("KNN (k=5)", &KnnClassifier::new().k(5), &scaled),
            run_cv("Gaussian NB", &GaussianNb::new(), &data),
            run_cv(
                "LinearSVC",
                &LinearSVC::new().c(1.0).max_iter(1000),
                &scaled,
            ),
        ];

        for r in &models {
            let time_str = if r.cv_time_ms < 1000.0 {
                format!("{:.1} ms", r.cv_time_ms)
            } else {
                format!("{:.2} s", r.cv_time_ms / 1000.0)
            };
            println!(
                "│ {:23} │ {:>7.4} │ {:>5.4} │ {:>15} │",
                r.name, r.mean_accuracy, r.std_accuracy, time_str
            );
        }

        println!("└─────────────────────────┴──────────┴────────┴─────────────────┘");

        // Print fold details
        println!("  Fold details:");
        for r in &models {
            let folds: Vec<String> = r.fold_scores.iter().map(|s| format!("{s:.3}")).collect();
            println!("    {:23} [{}]", r.name, folds.join(", "));
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Done. Compare these results with Python baselines:");
    println!("    python3 benches/python/bench_sklearn.py");
    println!("    python3 benches/python/bench_xgboost.py");
    println!("    python3 benches/python/bench_lightgbm.py");
    println!("═══════════════════════════════════════════════════════════════════");
}
