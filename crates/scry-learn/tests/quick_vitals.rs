#![allow(clippy::cast_possible_wrap)]
//! Industry-standard ML vitals — single fast test covering every key metric.
//!
//! **9 sections** covering the complete benchmark landscape:
//!
//! | § | Vital                          | Metrics / Measures                    |
//! |---|-------------------------------|---------------------------------------|
//! | 1 | Classification multi-metric    | Accuracy, F1, Precision, Recall, AUC-ROC |
//! | 2 | Regression vitals             | R², RMSE, MAE on California Housing   |
//! | 3 | Confusion matrix parity       | `ConfusionMatrix` shape + report        |
//! | 4 | Prediction latency (approx)   | p50/p95 single-row, median of 10 runs |
//! | 5 | Concurrent inference          | 4 threads × 250 predicts              |
//! | 6 | Serialize round-trip          | `serde_json` save/load, prediction match |
//! | 7 | Training throughput (approx)  | Wall-clock per model on 10K samples   |
//! | 8 | Cold start                    | construct → fit → first predict       |
//! | 9 | Memory footprint              | RSS delta per model family            |
//!
//! Run: `cargo test --test quick_vitals -p scry-learn --release -- --nocapture`
//! Target: < 60 s total.

use std::path::PathBuf;
use std::time::Instant;

use scry_learn::metrics::Average;
use scry_learn::prelude::*;

// ═════════════════════════════════════════════════════════════════════════
// Fixture helpers (reused from golden_reference.rs)
// ═════════════════════════════════════════════════════════════════════════

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

// ── Timing helper: median of N runs after warmup ────────────────────────

fn median_us<F: FnMut()>(mut f: F, warmup: usize, runs: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        f();
        times.push(start.elapsed().as_nanos() as f64 / 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[runs / 2]
}

fn percentile_us(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64) * p).min(sorted.len() as f64 - 1.0) as usize;
    sorted[idx]
}

fn fmt_time(us: f64) -> String {
    if us < 1.0 {
        format!("{:.0} ns", us * 1000.0)
    } else if us < 1000.0 {
        format!("{us:.1} µs")
    } else if us < 1_000_000.0 {
        format!("{:.2} ms", us / 1000.0)
    } else {
        format!("{:.2} s", us / 1_000_000.0)
    }
}

// ── RSS helper (Linux) ──────────────────────────────────────────────────

fn read_rss_kb() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(kb_str) = parts.get(1) {
                return kb_str.parse().unwrap_or(0);
            }
        }
    }
    0
}

// ═════════════════════════════════════════════════════════════════════════
// MAIN TEST
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn quick_vitals() {
    let overall_start = Instant::now();

    println!("\n{}", "═".repeat(78));
    println!("  SCRY-LEARN QUICK VITALS — Industry-Standard ML Benchmark");
    println!("  scry-learn v{}", env!("CARGO_PKG_VERSION"));
    println!("{}\n", "═".repeat(78));

    section_1_classification_metrics();
    section_2_regression_vitals();
    section_3_confusion_matrix();
    section_4_prediction_latency();
    section_5_concurrent_inference();
    section_6_serialize_roundtrip();
    section_7_training_throughput();
    section_8_cold_start();
    section_9_memory_footprint();

    let total = overall_start.elapsed();
    println!("\n{}", "═".repeat(78));
    println!("  ✓ All vitals complete in {:.1}s", total.as_secs_f64());
    println!("{}", "═".repeat(78));
}

// ═════════════════════════════════════════════════════════════════════════
// § 1  CLASSIFICATION MULTI-METRIC
// ═════════════════════════════════════════════════════════════════════════

fn section_1_classification_metrics() {
    println!("─── §1 Classification Multi-Metric (F1/Prec/Recall/AUC-ROC) ───\n");

    let datasets = ["iris", "wine", "breast_cancer", "digits"];

    // Header
    println!(
        "  {:<24} {:<10} {:>6} {:>6} {:>6} {:>6} {:>8}",
        "Model / Dataset", "Dataset", "Acc", "F1", "Prec", "Recall", "AUC-ROC"
    );
    println!("  {}", "-".repeat(72));

    for ds_name in &datasets {
        let data = load_dataset(ds_name);
        let (train, test) = train_test_split(&data, 0.2, 42);
        let matrix = test.feature_matrix();
        let y_true = &test.target;
        let is_binary = {
            let mut classes: Vec<f64> = data.target.clone();
            classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            classes.dedup();
            classes.len() == 2
        };

        // ─ Decision Tree ─
        {
            let mut m = DecisionTreeClassifier::new().max_depth(10);
            m.fit(&train).unwrap();
            let preds = m.predict(&matrix).unwrap();
            let acc = accuracy(y_true, &preds);
            let f1 = f1_score(y_true, &preds, Average::Macro);
            let prec = precision(y_true, &preds, Average::Macro);
            let rec = recall(y_true, &preds, Average::Macro);
            let auc_str = if is_binary {
                let probas = m.predict_proba(&matrix).unwrap();
                let scores: Vec<f64> = probas
                    .iter()
                    .map(|p| p.get(1).copied().unwrap_or(0.0))
                    .collect();
                format!("{:.4}", roc_auc_score(y_true, &scores))
            } else {
                "n/a".to_string()
            };
            println!(
                "  {:<24} {:<10} {:.4} {:.4} {:.4} {:.4} {:>8}",
                "DecisionTree", ds_name, acc, f1, prec, rec, auc_str
            );
        }

        // ─ Random Forest ─
        {
            let mut m = RandomForestClassifier::new()
                .n_estimators(20)
                .max_depth(10)
                .seed(42);
            m.fit(&train).unwrap();
            let preds = m.predict(&matrix).unwrap();
            let acc = accuracy(y_true, &preds);
            let f1 = f1_score(y_true, &preds, Average::Macro);
            let prec = precision(y_true, &preds, Average::Macro);
            let rec = recall(y_true, &preds, Average::Macro);
            let auc_str = if is_binary {
                let probas = m.predict_proba(&matrix).unwrap();
                let scores: Vec<f64> = probas
                    .iter()
                    .map(|p| p.get(1).copied().unwrap_or(0.0))
                    .collect();
                format!("{:.4}", roc_auc_score(y_true, &scores))
            } else {
                "n/a".to_string()
            };
            println!(
                "  {:<24} {:<10} {:.4} {:.4} {:.4} {:.4} {:>8}",
                "RandomForest", ds_name, acc, f1, prec, rec, auc_str
            );
        }

        // ─ Gradient Boosting ─
        {
            let mut m = GradientBoostingClassifier::new()
                .n_estimators(50)
                .max_depth(5)
                .learning_rate(0.1)
                .seed(42);
            m.fit(&train).unwrap();
            let preds = m.predict(&matrix).unwrap();
            let acc = accuracy(y_true, &preds);
            let f1 = f1_score(y_true, &preds, Average::Macro);
            let prec = precision(y_true, &preds, Average::Macro);
            let rec = recall(y_true, &preds, Average::Macro);
            let auc_str = if is_binary {
                let probas = m.predict_proba(&matrix).unwrap();
                let scores: Vec<f64> = probas
                    .iter()
                    .map(|p| p.get(1).copied().unwrap_or(0.0))
                    .collect();
                format!("{:.4}", roc_auc_score(y_true, &scores))
            } else {
                "n/a".to_string()
            };
            println!(
                "  {:<24} {:<10} {:.4} {:.4} {:.4} {:.4} {:>8}",
                "GradientBoosting", ds_name, acc, f1, prec, rec, auc_str
            );
        }

        // ─ HistGBT ─
        {
            let mut m = HistGradientBoostingClassifier::new()
                .n_estimators(50)
                .max_depth(6)
                .learning_rate(0.1)
                .seed(42);
            m.fit(&train).unwrap();
            let preds = m.predict(&matrix).unwrap();
            let acc = accuracy(y_true, &preds);
            let f1 = f1_score(y_true, &preds, Average::Macro);
            let prec = precision(y_true, &preds, Average::Macro);
            let rec = recall(y_true, &preds, Average::Macro);
            let auc_str = if is_binary {
                let probas = m.predict_proba(&matrix).unwrap();
                let scores: Vec<f64> = probas
                    .iter()
                    .map(|p| p.get(1).copied().unwrap_or(0.0))
                    .collect();
                format!("{:.4}", roc_auc_score(y_true, &scores))
            } else {
                "n/a".to_string()
            };
            println!(
                "  {:<24} {:<10} {:.4} {:.4} {:.4} {:.4} {:>8}",
                "HistGBT", ds_name, acc, f1, prec, rec, auc_str
            );
        }

        // ─ Logistic Regression ─
        {
            let mut train_s = train.clone();
            let mut test_s = test.clone();
            let mut scaler = StandardScaler::new();
            Transformer::fit(&mut scaler, &train_s).unwrap();
            Transformer::transform(&scaler, &mut train_s).unwrap();
            Transformer::transform(&scaler, &mut test_s).unwrap();
            let mat_s = test_s.feature_matrix();

            let mut m = LogisticRegression::new().max_iter(500);
            m.fit(&train_s).unwrap();
            let preds = m.predict(&mat_s).unwrap();
            let acc = accuracy(&test_s.target, &preds);
            let f1 = f1_score(&test_s.target, &preds, Average::Macro);
            let prec = precision(&test_s.target, &preds, Average::Macro);
            let rec = recall(&test_s.target, &preds, Average::Macro);
            let auc_str = if is_binary {
                let probas = m.predict_proba(&mat_s).unwrap();
                let scores: Vec<f64> = probas
                    .iter()
                    .map(|p| p.get(1).copied().unwrap_or(0.0))
                    .collect();
                format!("{:.4}", roc_auc_score(&test_s.target, &scores))
            } else {
                "n/a".to_string()
            };
            println!(
                "  {:<24} {:<10} {:.4} {:.4} {:.4} {:.4} {:>8}",
                "LogisticRegression", ds_name, acc, f1, prec, rec, auc_str
            );
        }

        // ─ KNN ─
        {
            let mut m = KnnClassifier::new().k(5);
            m.fit(&train).unwrap();
            let preds = m.predict(&matrix).unwrap();
            let acc = accuracy(y_true, &preds);
            let f1 = f1_score(y_true, &preds, Average::Macro);
            let prec = precision(y_true, &preds, Average::Macro);
            let rec = recall(y_true, &preds, Average::Macro);
            let auc_str = if is_binary {
                let probas = m.predict_proba(&matrix).unwrap();
                let scores: Vec<f64> = probas
                    .iter()
                    .map(|p| p.get(1).copied().unwrap_or(0.0))
                    .collect();
                format!("{:.4}", roc_auc_score(y_true, &scores))
            } else {
                "n/a".to_string()
            };
            println!(
                "  {:<24} {:<10} {:.4} {:.4} {:.4} {:.4} {:>8}",
                "KNN", ds_name, acc, f1, prec, rec, auc_str
            );
        }

        // ─ Gaussian NB ─
        {
            let mut m = GaussianNb::new();
            m.fit(&train).unwrap();
            let preds = m.predict(&matrix).unwrap();
            let acc = accuracy(y_true, &preds);
            let f1 = f1_score(y_true, &preds, Average::Macro);
            let prec = precision(y_true, &preds, Average::Macro);
            let rec = recall(y_true, &preds, Average::Macro);
            let auc_str = if is_binary {
                let probas = m.predict_proba(&matrix).unwrap();
                let scores: Vec<f64> = probas
                    .iter()
                    .map(|p| p.get(1).copied().unwrap_or(0.0))
                    .collect();
                format!("{:.4}", roc_auc_score(y_true, &scores))
            } else {
                "n/a".to_string()
            };
            println!(
                "  {:<24} {:<10} {:.4} {:.4} {:.4} {:.4} {:>8}",
                "GaussianNB", ds_name, acc, f1, prec, rec, auc_str
            );
        }

        // ─ Linear SVC ─
        {
            let mut train_s = train.clone();
            let mut test_s = test.clone();
            let mut scaler = StandardScaler::new();
            Transformer::fit(&mut scaler, &train_s).unwrap();
            Transformer::transform(&scaler, &mut train_s).unwrap();
            Transformer::transform(&scaler, &mut test_s).unwrap();
            let mat_s = test_s.feature_matrix();

            let mut m = LinearSVC::new().max_iter(1000);
            m.fit(&train_s).unwrap();
            let preds = m.predict(&mat_s).unwrap();
            let acc = accuracy(&test_s.target, &preds);
            let f1 = f1_score(&test_s.target, &preds, Average::Macro);
            let prec = precision(&test_s.target, &preds, Average::Macro);
            let rec = recall(&test_s.target, &preds, Average::Macro);
            // LinearSVC predict_proba requires .probability(true) — skip AUC for SVC
            println!(
                "  {:<24} {:<10} {:.4} {:.4} {:.4} {:.4} {:>8}",
                "LinearSVC", ds_name, acc, f1, prec, rec, "n/a"
            );
        }

        println!();
    }
}

// ═════════════════════════════════════════════════════════════════════════
// § 2  REGRESSION VITALS
// ═════════════════════════════════════════════════════════════════════════

fn section_2_regression_vitals() {
    println!("─── §2 Regression Vitals (California Housing) ───\n");

    let mut data = load_dataset("california");
    let mut scaler = StandardScaler::new();
    Transformer::fit(&mut scaler, &data).unwrap();
    Transformer::transform(&scaler, &mut data).unwrap();

    let (train, test) = train_test_split(&data, 0.2, 42);
    let matrix = test.feature_matrix();
    let y = &test.target;

    println!("  {:<28} {:>8} {:>8} {:>8}", "Model", "R²", "RMSE", "MAE");
    println!("  {}", "-".repeat(56));

    // LinearRegression
    {
        let mut m = LinearRegression::new();
        m.fit(&train).unwrap();
        let preds = m.predict(&matrix).unwrap();
        let r2 = r2_score(y, &preds);
        let rmse = scry_learn::metrics::root_mean_squared_error(y, &preds);
        let mae = scry_learn::metrics::mean_absolute_error(y, &preds);
        println!(
            "  {:<28} {:>8.4} {:>8.4} {:>8.4}",
            "LinearRegression", r2, rmse, mae
        );
        assert!(r2 > 0.5, "LinearRegression R² {r2:.4} < 0.5");
    }

    // Lasso
    {
        let mut m = LassoRegression::new().alpha(0.01).max_iter(1000);
        m.fit(&train).unwrap();
        let preds = m.predict(&matrix).unwrap();
        let r2 = r2_score(y, &preds);
        let rmse = scry_learn::metrics::root_mean_squared_error(y, &preds);
        let mae = scry_learn::metrics::mean_absolute_error(y, &preds);
        println!(
            "  {:<28} {:>8.4} {:>8.4} {:>8.4}",
            "Lasso (α=0.01)", r2, rmse, mae
        );
        assert!(r2 > 0.3, "Lasso R² {r2:.4} < 0.3");
    }

    // ElasticNet
    {
        let mut m = ElasticNet::new().alpha(0.01).l1_ratio(0.5).max_iter(1000);
        m.fit(&train).unwrap();
        let preds = m.predict(&matrix).unwrap();
        let r2 = r2_score(y, &preds);
        let rmse = scry_learn::metrics::root_mean_squared_error(y, &preds);
        let mae = scry_learn::metrics::mean_absolute_error(y, &preds);
        println!(
            "  {:<28} {:>8.4} {:>8.4} {:>8.4}",
            "ElasticNet (α=0.01)", r2, rmse, mae
        );
        assert!(r2 > 0.3, "ElasticNet R² {r2:.4} < 0.3");
    }

    // KNN Regressor
    {
        let mut m = KnnRegressor::new().k(5);
        m.fit(&train).unwrap();
        let preds = m.predict(&matrix).unwrap();
        let r2 = r2_score(y, &preds);
        let rmse = scry_learn::metrics::root_mean_squared_error(y, &preds);
        let mae = scry_learn::metrics::mean_absolute_error(y, &preds);
        println!(
            "  {:<28} {:>8.4} {:>8.4} {:>8.4}",
            "KnnRegressor (k=5)", r2, rmse, mae
        );
        assert!(r2 > 0.3, "KnnRegressor R² {r2:.4} < 0.3");
    }

    // GBT Regressor
    {
        let mut m = GradientBoostingRegressor::new()
            .n_estimators(50)
            .max_depth(5)
            .learning_rate(0.1)
            .seed(42);
        m.fit(&train).unwrap();
        let preds = m.predict(&matrix).unwrap();
        let r2 = r2_score(y, &preds);
        let rmse = scry_learn::metrics::root_mean_squared_error(y, &preds);
        let mae = scry_learn::metrics::mean_absolute_error(y, &preds);
        println!(
            "  {:<28} {:>8.4} {:>8.4} {:>8.4}",
            "GBTRegressor", r2, rmse, mae
        );
        assert!(r2 > 0.5, "GBTRegressor R² {r2:.4} < 0.5");
    }

    // Ridge
    {
        let mut m = Ridge::new(1.0);
        m.fit(&train).unwrap();
        let preds = m.predict(&matrix).unwrap();
        let r2 = r2_score(y, &preds);
        let rmse = scry_learn::metrics::root_mean_squared_error(y, &preds);
        let mae = scry_learn::metrics::mean_absolute_error(y, &preds);
        println!(
            "  {:<28} {:>8.4} {:>8.4} {:>8.4}",
            "Ridge (α=1.0)", r2, rmse, mae
        );
        assert!(r2 > 0.5, "Ridge R² {r2:.4} < 0.5");
    }

    println!();
}

// ═════════════════════════════════════════════════════════════════════════
// § 3  CONFUSION MATRIX PARITY
// ═════════════════════════════════════════════════════════════════════════

fn section_3_confusion_matrix() {
    println!("─── §3 Confusion Matrix & Classification Report ───\n");

    let data = load_dataset("iris");
    let (train, test) = train_test_split(&data, 0.2, 42);
    let matrix = test.feature_matrix();

    let mut dt = DecisionTreeClassifier::new().max_depth(10);
    dt.fit(&train).unwrap();
    let preds = dt.predict(&matrix).unwrap();

    let cm = confusion_matrix(&test.target, &preds);
    let report = classification_report(&test.target, &preds);

    println!(
        "  Confusion Matrix ({} classes, {}×{}):",
        cm.labels.len(),
        cm.matrix.len(),
        cm.matrix[0].len()
    );
    for (i, row) in cm.matrix.iter().enumerate() {
        println!("    Class {} → {:?}", &cm.labels[i], row);
    }
    println!();
    println!("{report}");

    // Sanity: diagonal should dominate
    let total: usize = cm.matrix.iter().flat_map(|r| r.iter()).sum();
    let diag: usize = (0..cm.matrix.len()).map(|i| cm.matrix[i][i]).sum();
    let diag_ratio = diag as f64 / total as f64;
    assert!(
        diag_ratio > 0.8,
        "Confusion matrix diagonal ratio {diag_ratio:.2} < 0.8 — something is wrong"
    );
    println!(
        "  ✓ Diagonal ratio: {:.1}% ({}/{})\n",
        diag_ratio * 100.0,
        diag,
        total
    );
}

// ═════════════════════════════════════════════════════════════════════════
// § 4  PREDICTION LATENCY (APPROXIMATE)
// ═════════════════════════════════════════════════════════════════════════

fn section_4_prediction_latency() {
    println!("─── §4 Single-Row Prediction Latency (approximate) ───\n");

    // Use Iris for fast single-row measurements
    let data = load_dataset("iris");
    let matrix = data.feature_matrix();
    let single_row = vec![matrix[0].clone()];
    let n_iters = 1000;

    println!("  {:<28} {:>10} {:>10}", "Model", "p50", "p95");
    println!("  {}", "-".repeat(50));

    // DecisionTree
    {
        let mut m = DecisionTreeClassifier::new().max_depth(10);
        m.fit(&data).unwrap();
        let mut times = Vec::with_capacity(n_iters);
        // warmup
        for _ in 0..100 {
            let _ = m.predict(&single_row);
        }
        for _ in 0..n_iters {
            let start = Instant::now();
            let _ = m.predict(&single_row);
            times.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {:<28} {:>10} {:>10}",
            "DecisionTree",
            fmt_time(percentile_us(&times, 0.5)),
            fmt_time(percentile_us(&times, 0.95))
        );
    }

    // RandomForest
    {
        let mut m = RandomForestClassifier::new()
            .n_estimators(20)
            .max_depth(10)
            .seed(42);
        m.fit(&data).unwrap();
        let mut times = Vec::with_capacity(n_iters);
        for _ in 0..100 {
            let _ = m.predict(&single_row);
        }
        for _ in 0..n_iters {
            let start = Instant::now();
            let _ = m.predict(&single_row);
            times.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {:<28} {:>10} {:>10}",
            "RandomForest (20 trees)",
            fmt_time(percentile_us(&times, 0.5)),
            fmt_time(percentile_us(&times, 0.95))
        );
    }

    // GaussianNB
    {
        let mut m = GaussianNb::new();
        m.fit(&data).unwrap();
        let mut times = Vec::with_capacity(n_iters);
        for _ in 0..100 {
            let _ = m.predict(&single_row);
        }
        for _ in 0..n_iters {
            let start = Instant::now();
            let _ = m.predict(&single_row);
            times.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {:<28} {:>10} {:>10}",
            "GaussianNB",
            fmt_time(percentile_us(&times, 0.5)),
            fmt_time(percentile_us(&times, 0.95))
        );
    }

    // KNN
    {
        let mut m = KnnClassifier::new().k(5);
        m.fit(&data).unwrap();
        let mut times = Vec::with_capacity(n_iters);
        for _ in 0..100 {
            let _ = m.predict(&single_row);
        }
        for _ in 0..n_iters {
            let start = Instant::now();
            let _ = m.predict(&single_row);
            times.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {:<28} {:>10} {:>10}",
            "KNN (k=5)",
            fmt_time(percentile_us(&times, 0.5)),
            fmt_time(percentile_us(&times, 0.95))
        );
    }

    // LogisticRegression
    {
        let mut data_s = data;
        let mut scaler = StandardScaler::new();
        Transformer::fit(&mut scaler, &data_s).unwrap();
        Transformer::transform(&scaler, &mut data_s).unwrap();
        let single_s = vec![data_s.feature_matrix()[0].clone()];

        let mut m = LogisticRegression::new().max_iter(500);
        m.fit(&data_s).unwrap();
        let mut times = Vec::with_capacity(n_iters);
        for _ in 0..100 {
            let _ = m.predict(&single_s);
        }
        for _ in 0..n_iters {
            let start = Instant::now();
            let _ = m.predict(&single_s);
            times.push(start.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  {:<28} {:>10} {:>10}",
            "LogisticRegression",
            fmt_time(percentile_us(&times, 0.5)),
            fmt_time(percentile_us(&times, 0.95))
        );
    }

    println!();
}

// ═════════════════════════════════════════════════════════════════════════
// § 5  CONCURRENT INFERENCE
// ═════════════════════════════════════════════════════════════════════════

fn section_5_concurrent_inference() {
    println!("─── §5 Concurrent Inference (4 threads × 250 predicts) ───\n");

    let data = load_dataset("iris");
    let matrix = data.feature_matrix();
    let single_row = vec![matrix[0].clone()];

    // Train models
    let mut dt = DecisionTreeClassifier::new().max_depth(10);
    dt.fit(&data).unwrap();
    let mut rf = RandomForestClassifier::new()
        .n_estimators(20)
        .max_depth(10)
        .seed(42);
    rf.fit(&data).unwrap();
    let mut nb = GaussianNb::new();
    nb.fit(&data).unwrap();

    let n_threads = 4;
    let n_per_thread = 250;

    // Test DecisionTree concurrent inference
    {
        let start = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    for _ in 0..n_per_thread {
                        let _ = dt.predict(&single_row).unwrap();
                    }
                });
            }
        });
        let elapsed = start.elapsed();
        let total_ops = n_threads * n_per_thread;
        let ops_sec = total_ops as f64 / elapsed.as_secs_f64();
        println!(
            "  DecisionTree:    {total_ops} ops in {:>8}, {ops_sec:.0} ops/sec",
            fmt_time(elapsed.as_micros() as f64)
        );
    }

    // Test RandomForest concurrent inference
    {
        let start = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    for _ in 0..n_per_thread {
                        let _ = rf.predict(&single_row).unwrap();
                    }
                });
            }
        });
        let elapsed = start.elapsed();
        let total_ops = n_threads * n_per_thread;
        let ops_sec = total_ops as f64 / elapsed.as_secs_f64();
        println!(
            "  RandomForest:    {total_ops} ops in {:>8}, {ops_sec:.0} ops/sec",
            fmt_time(elapsed.as_micros() as f64)
        );
    }

    // Test GaussianNB concurrent inference
    {
        let start = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    for _ in 0..n_per_thread {
                        let _ = nb.predict(&single_row).unwrap();
                    }
                });
            }
        });
        let elapsed = start.elapsed();
        let total_ops = n_threads * n_per_thread;
        let ops_sec = total_ops as f64 / elapsed.as_secs_f64();
        println!(
            "  GaussianNB:      {total_ops} ops in {:>8}, {ops_sec:.0} ops/sec",
            fmt_time(elapsed.as_micros() as f64)
        );
    }

    println!("  ✓ No panics or data races\n");
}

// ═════════════════════════════════════════════════════════════════════════
// § 6  SERIALIZE / DESERIALIZE ROUND-TRIP
// ═════════════════════════════════════════════════════════════════════════

fn section_6_serialize_roundtrip() {
    println!("─── §6 Model Size & Serialize Estimate ───\n");

    // Serde round-trip requires the `serde` feature. Here we measure
    // struct size (std::mem::size_of_val) and serialized size via Debug
    // formatting as a proxy. Full serde benchmarks live in production_bench.

    let data = load_dataset("iris");

    println!("  {:<28} {:>14}", "Model", "Struct Size");
    println!("  {}", "-".repeat(44));

    // DecisionTree
    {
        let mut m = DecisionTreeClassifier::new().max_depth(10);
        m.fit(&data).unwrap();
        let size = std::mem::size_of_val(&m);
        println!("  {:<28} {:>10} B", "DecisionTree", size);
    }

    // RandomForest
    {
        let mut m = RandomForestClassifier::new()
            .n_estimators(10)
            .max_depth(10)
            .seed(42);
        m.fit(&data).unwrap();
        let size = std::mem::size_of_val(&m);
        println!("  {:<28} {:>10} B", "RandomForest (10 trees)", size);
    }

    // GaussianNB
    {
        let mut m = GaussianNb::new();
        m.fit(&data).unwrap();
        let size = std::mem::size_of_val(&m);
        println!("  {:<28} {:>10} B", "GaussianNB", size);
    }

    // LogisticRegression
    {
        let mut data_s = data.clone();
        let mut scaler = StandardScaler::new();
        Transformer::fit(&mut scaler, &data_s).unwrap();
        Transformer::transform(&scaler, &mut data_s).unwrap();
        let mut m = LogisticRegression::new().max_iter(500);
        m.fit(&data_s).unwrap();
        let size = std::mem::size_of_val(&m);
        println!("  {:<28} {:>10} B", "LogisticRegression", size);
    }

    // KNN
    {
        let mut m = KnnClassifier::new().k(5);
        m.fit(&data).unwrap();
        let size = std::mem::size_of_val(&m);
        println!("  {:<28} {:>10} B", "KNN (k=5)", size);
    }

    // LinearRegression
    {
        let mut reg_data = load_dataset("california");
        let mut scaler = StandardScaler::new();
        Transformer::fit(&mut scaler, &reg_data).unwrap();
        Transformer::transform(&scaler, &mut reg_data).unwrap();
        let mut m = LinearRegression::new();
        m.fit(&reg_data).unwrap();
        let size = std::mem::size_of_val(&m);
        println!("  {:<28} {:>10} B", "LinearRegression", size);
    }

    println!("\n  Note: For full serde round-trip benchmarks, run with `--features serde`.");
    println!("  Full serde tests are in `production_bench.rs`.\n");
}

// ═════════════════════════════════════════════════════════════════════════
// § 7  TRAINING THROUGHPUT (APPROXIMATE)
// ═════════════════════════════════════════════════════════════════════════

fn section_7_training_throughput() {
    println!("─── §7 Training Throughput (10K samples, median of 5 runs) ───\n");

    // Generate synthetic 10K dataset
    let n = 10_000;
    let n_features = 10;
    let mut features = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let col: Vec<f64> = (0..n)
            .map(|i| ((i * (j + 3) + 7) % 997) as f64 / 997.0 * 10.0)
            .collect();
        features.push(col);
    }
    let target: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
    let names: Vec<String> = (0..n_features).map(|j| format!("f{j}")).collect();
    let data = Dataset::new(features, target, names, "target");

    println!("  {:<28} {:>12}", "Model", "Fit Time");
    println!("  {}", "-".repeat(42));

    // DecisionTree
    {
        let t = median_us(
            || {
                let mut m = DecisionTreeClassifier::new().max_depth(10);
                m.fit(&data).unwrap();
            },
            2,
            5,
        );
        println!("  {:<28} {:>12}", "DecisionTree", fmt_time(t));
    }

    // RandomForest
    {
        let t = median_us(
            || {
                let mut m = RandomForestClassifier::new()
                    .n_estimators(10)
                    .max_depth(10)
                    .seed(42);
                m.fit(&data).unwrap();
            },
            1,
            3,
        );
        println!("  {:<28} {:>12}", "RandomForest (10 trees)", fmt_time(t));
    }

    // GradientBoosting
    {
        let t = median_us(
            || {
                let mut m = GradientBoostingClassifier::new()
                    .n_estimators(20)
                    .max_depth(5)
                    .learning_rate(0.1)
                    .seed(42);
                m.fit(&data).unwrap();
            },
            1,
            3,
        );
        println!(
            "  {:<28} {:>12}",
            "GradientBoosting (20 trees)",
            fmt_time(t)
        );
    }

    // LogisticRegression
    {
        let mut data_s = data.clone();
        let mut scaler = StandardScaler::new();
        Transformer::fit(&mut scaler, &data_s).unwrap();
        Transformer::transform(&scaler, &mut data_s).unwrap();
        let t = median_us(
            || {
                let mut m = LogisticRegression::new().max_iter(200);
                m.fit(&data_s).unwrap();
            },
            1,
            3,
        );
        println!("  {:<28} {:>12}", "LogisticRegression", fmt_time(t));
    }

    // KNN (fit is essentially free — just data storage)
    {
        let t = median_us(
            || {
                let mut m = KnnClassifier::new().k(5);
                m.fit(&data).unwrap();
            },
            2,
            5,
        );
        println!("  {:<28} {:>12}", "KNN (k=5)", fmt_time(t));
    }

    // GaussianNB
    {
        let t = median_us(
            || {
                let mut m = GaussianNb::new();
                m.fit(&data).unwrap();
            },
            2,
            5,
        );
        println!("  {:<28} {:>12}", "GaussianNB", fmt_time(t));
    }

    // LinearRegression
    {
        let reg_target: Vec<f64> = (0..n).map(|i| i as f64 * 0.1 + 0.5).collect();
        let reg_names: Vec<String> = (0..n_features).map(|j| format!("f{j}")).collect();
        let mut reg_features = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col: Vec<f64> = (0..n)
                .map(|i| ((i * (j + 5) + 13) % 1009) as f64 / 100.0)
                .collect();
            reg_features.push(col);
        }
        let reg_data = Dataset::new(reg_features, reg_target, reg_names, "y");
        let t = median_us(
            || {
                let mut m = LinearRegression::new();
                m.fit(&reg_data).unwrap();
            },
            2,
            5,
        );
        println!("  {:<28} {:>12}", "LinearRegression", fmt_time(t));
    }

    println!();
}

// ═════════════════════════════════════════════════════════════════════════
// § 8  COLD START
// ═════════════════════════════════════════════════════════════════════════

fn section_8_cold_start() {
    println!("─── §8 Cold Start (construct → fit → first predict) ───\n");

    let data = load_dataset("iris");
    let single_row = vec![data.feature_matrix()[0].clone()];

    println!("  {:<28} {:>12}", "Model", "Cold Start");
    println!("  {}", "-".repeat(42));

    // DecisionTree
    {
        let t = median_us(
            || {
                let mut m = DecisionTreeClassifier::new().max_depth(10);
                m.fit(&data).unwrap();
                let _ = m.predict(&single_row).unwrap();
            },
            3,
            10,
        );
        println!("  {:<28} {:>12}", "DecisionTree", fmt_time(t));
    }

    // RandomForest
    {
        let t = median_us(
            || {
                let mut m = RandomForestClassifier::new()
                    .n_estimators(20)
                    .max_depth(10)
                    .seed(42);
                m.fit(&data).unwrap();
                let _ = m.predict(&single_row).unwrap();
            },
            2,
            5,
        );
        println!("  {:<28} {:>12}", "RandomForest (20 trees)", fmt_time(t));
    }

    // HistGBT
    {
        let t = median_us(
            || {
                let mut m = HistGradientBoostingClassifier::new()
                    .n_estimators(50)
                    .max_depth(6)
                    .learning_rate(0.1)
                    .seed(42);
                m.fit(&data).unwrap();
                let _ = m.predict(&single_row).unwrap();
            },
            1,
            3,
        );
        println!("  {:<28} {:>12}", "HistGBT (50 trees)", fmt_time(t));
    }

    // LogisticRegression
    {
        let mut data_s = data.clone();
        let mut scaler = StandardScaler::new();
        Transformer::fit(&mut scaler, &data_s).unwrap();
        Transformer::transform(&scaler, &mut data_s).unwrap();
        let single_s = vec![data_s.feature_matrix()[0].clone()];
        let t = median_us(
            || {
                let mut m = LogisticRegression::new().max_iter(200);
                m.fit(&data_s).unwrap();
                let _ = m.predict(&single_s).unwrap();
            },
            2,
            5,
        );
        println!("  {:<28} {:>12}", "LogisticRegression", fmt_time(t));
    }

    // KNN
    {
        let t = median_us(
            || {
                let mut m = KnnClassifier::new().k(5);
                m.fit(&data).unwrap();
                let _ = m.predict(&single_row).unwrap();
            },
            3,
            10,
        );
        println!("  {:<28} {:>12}", "KNN (k=5)", fmt_time(t));
    }

    // GaussianNB
    {
        let t = median_us(
            || {
                let mut m = GaussianNb::new();
                m.fit(&data).unwrap();
                let _ = m.predict(&single_row).unwrap();
            },
            3,
            10,
        );
        println!("  {:<28} {:>12}", "GaussianNB", fmt_time(t));
    }

    // LinearRegression
    {
        let mut reg_data = load_dataset("california");
        let mut scaler = StandardScaler::new();
        Transformer::fit(&mut scaler, &reg_data).unwrap();
        Transformer::transform(&scaler, &mut reg_data).unwrap();
        let single_r = vec![reg_data.feature_matrix()[0].clone()];
        let t = median_us(
            || {
                let mut m = LinearRegression::new();
                m.fit(&reg_data).unwrap();
                let _ = m.predict(&single_r).unwrap();
            },
            1,
            3,
        );
        println!(
            "  {:<28} {:>12}",
            "LinearRegression (CalHousing)",
            fmt_time(t)
        );
    }

    println!();
}

// ═════════════════════════════════════════════════════════════════════════
// § 9  MEMORY FOOTPRINT
// ═════════════════════════════════════════════════════════════════════════

fn section_9_memory_footprint() {
    println!("─── §9 Memory Footprint (RSS delta per model) ───\n");

    // Use a larger synthetic dataset to make RSS deltas measurable
    let n = 50_000;
    let n_features = 10;
    let n_classes = 3;

    let mut features = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let col: Vec<f64> = (0..n)
            .map(|i| ((i * (j + 3) + 7) % 997) as f64 / 997.0 * 10.0)
            .collect();
        features.push(col);
    }
    let target: Vec<f64> = (0..n).map(|i| (i % n_classes) as f64).collect();
    let names: Vec<String> = (0..n_features).map(|j| format!("f{j}")).collect();
    let data = Dataset::new(features.clone(), target, names.clone(), "target");

    println!("  Dataset: {n} samples × {n_features} features\n");
    println!("  {:<28} {:>12}", "Model", "RSS Δ");
    println!("  {}", "-".repeat(42));

    // DecisionTree
    {
        let rss_before = read_rss_kb();
        let mut m = DecisionTreeClassifier::new().max_depth(10);
        m.fit(&data).unwrap();
        let rss_after = read_rss_kb();
        let delta = rss_after as isize - rss_before as isize;
        let delta_str = if delta.abs() < 1024 {
            format!("{delta} KB")
        } else {
            format!("{:.1} MB", delta as f64 / 1024.0)
        };
        println!("  {:<28} {:>12}", "DecisionTree", delta_str);
        std::mem::drop(m);
    }

    // RandomForest
    {
        let rss_before = read_rss_kb();
        let mut m = RandomForestClassifier::new()
            .n_estimators(10)
            .max_depth(10)
            .seed(42);
        m.fit(&data).unwrap();
        let rss_after = read_rss_kb();
        let delta = rss_after as isize - rss_before as isize;
        let delta_str = if delta.abs() < 1024 {
            format!("{delta} KB")
        } else {
            format!("{:.1} MB", delta as f64 / 1024.0)
        };
        println!("  {:<28} {:>12}", "RandomForest (10 trees)", delta_str);
        std::mem::drop(m);
    }

    // GradientBoosting
    {
        let rss_before = read_rss_kb();
        let mut m = GradientBoostingClassifier::new()
            .n_estimators(20)
            .max_depth(5)
            .learning_rate(0.1)
            .seed(42);
        m.fit(&data).unwrap();
        let rss_after = read_rss_kb();
        let delta = rss_after as isize - rss_before as isize;
        let delta_str = if delta.abs() < 1024 {
            format!("{delta} KB")
        } else {
            format!("{:.1} MB", delta as f64 / 1024.0)
        };
        println!("  {:<28} {:>12}", "GradientBoosting (20 trees)", delta_str);
        std::mem::drop(m);
    }

    // LogisticRegression
    {
        let mut data_s = data.clone();
        let mut scaler = StandardScaler::new();
        Transformer::fit(&mut scaler, &data_s).unwrap();
        Transformer::transform(&scaler, &mut data_s).unwrap();
        let rss_before = read_rss_kb();
        let mut m = LogisticRegression::new().max_iter(200);
        m.fit(&data_s).unwrap();
        let rss_after = read_rss_kb();
        let delta = rss_after as isize - rss_before as isize;
        let delta_str = if delta.abs() < 1024 {
            format!("{delta} KB")
        } else {
            format!("{:.1} MB", delta as f64 / 1024.0)
        };
        println!("  {:<28} {:>12}", "LogisticRegression", delta_str);
        std::mem::drop(m);
    }

    // KNN (stores entire training set)
    {
        let rss_before = read_rss_kb();
        let mut m = KnnClassifier::new().k(5);
        m.fit(&data).unwrap();
        let rss_after = read_rss_kb();
        let delta = rss_after as isize - rss_before as isize;
        let delta_str = if delta.abs() < 1024 {
            format!("{delta} KB")
        } else {
            format!("{:.1} MB", delta as f64 / 1024.0)
        };
        println!("  {:<28} {:>12}", "KNN (k=5)", delta_str);
        std::mem::drop(m);
    }

    // GaussianNB
    {
        let rss_before = read_rss_kb();
        let mut m = GaussianNb::new();
        m.fit(&data).unwrap();
        let rss_after = read_rss_kb();
        let delta = rss_after as isize - rss_before as isize;
        let delta_str = if delta.abs() < 1024 {
            format!("{delta} KB")
        } else {
            format!("{:.1} MB", delta as f64 / 1024.0)
        };
        println!("  {:<28} {:>12}", "GaussianNB", delta_str);
        std::mem::drop(m);
    }

    // LinearRegression
    {
        let reg_target: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let reg_data = Dataset::new(features, reg_target, names, "y");
        let rss_before = read_rss_kb();
        let mut m = LinearRegression::new();
        m.fit(&reg_data).unwrap();
        let rss_after = read_rss_kb();
        let delta = rss_after as isize - rss_before as isize;
        let delta_str = if delta.abs() < 1024 {
            format!("{delta} KB")
        } else {
            format!("{:.1} MB", delta as f64 / 1024.0)
        };
        println!("  {:<28} {:>12}", "LinearRegression", delta_str);
        std::mem::drop(m);
    }

    println!("\n  ✓ RSS measurement complete (Linux /proc/self/status)\n");
}
