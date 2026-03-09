#![allow(
    clippy::struct_field_names,
    clippy::type_complexity,
    clippy::or_fun_call,
    clippy::option_if_let_else
)]
//! Consolidated benchmark comparison: scry-learn vs scikit-learn vs `XGBoost` vs `LightGBM`.
//!
//! Loads Python baseline JSON results and runs scry-learn models inline,
//! then prints a unified comparison table with accuracy, latency percentiles,
//! and memory footprint.
//!
//! Run: `cargo run --example benchmark_comparison -p scry-learn --release`

use std::collections::HashMap;
use std::fs;
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

// ─── Latency measurement ────────────────────────────────────────

struct LatencyStats {
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
}

fn measure_latency<F: FnMut()>(mut f: F, n_iters: usize) -> LatencyStats {
    let mut times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        times.push(elapsed.as_nanos() as f64 / 1000.0); // microseconds
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = times.len();
    LatencyStats {
        p50_us: times[n / 2],
        p95_us: times[(n as f64 * 0.95) as usize],
        p99_us: times[(n as f64 * 0.99) as usize],
    }
}

// ─── Memory measurement (Linux) ─────────────────────────────────

/// Read peak RSS (`VmHWM`) from /proc/self/status in KB.
#[cfg(target_os = "linux")]
fn peak_rss_kb() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmHWM:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts.get(1)?.parse().ok();
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_kb() -> Option<u64> {
    None
}

fn fmt_time(us: f64) -> String {
    if us < 1.0 {
        format!("{:.1} ns", us * 1000.0)
    } else if us < 1_000.0 {
        format!("{us:.1} µs")
    } else if us < 1_000_000.0 {
        format!("{:.2} ms", us / 1_000.0)
    } else {
        format!("{:.2} s", us / 1_000_000.0)
    }
}

// ─── Python result loading ──────────────────────────────────────

#[derive(Default)]
struct PythonAccuracy {
    mean: f64,
}

fn load_python_accuracy(filename: &str) -> HashMap<String, HashMap<String, PythonAccuracy>> {
    let benches_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("python");
    let path = benches_dir.join(filename);
    let mut results: HashMap<String, HashMap<String, PythonAccuracy>> = HashMap::new();

    let Ok(contents) = fs::read_to_string(&path) else {
        eprintln!("  WARN: Could not read {}", path.display());
        return results;
    };

    let json: serde_json::Value = serde_json::from_str(&contents).unwrap();

    if let Some(cv) = json.get("accuracy_cv").and_then(|v| v.as_object()) {
        for (key, val) in cv {
            let mean = val
                .get("mean_accuracy")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);

            // sklearn uses "model/dataset" keys, xgboost/lightgbm use "dataset" keys
            if key.contains('/') {
                // sklearn format: "model_name/dataset_name"
                let parts: Vec<&str> = key.splitn(2, '/').collect();
                let model = parts[0].to_string();
                let dataset = parts[1].to_string();
                results
                    .entry(dataset)
                    .or_default()
                    .insert(model, PythonAccuracy { mean });
            } else {
                // xgboost/lightgbm format: just dataset name (single model)
                results
                    .entry(key.clone())
                    .or_default()
                    .insert("default".to_string(), PythonAccuracy { mean });
            }
        }
    }

    results
}

// ─── Model result ────────────────────────────────────────────────

#[allow(dead_code)]
struct ModelResult {
    name: String,
    mean_accuracy: f64,
    std_accuracy: f64,
    cv_time_ms: f64,
}

fn run_cv<M: scry_learn::pipeline::PipelineModel + Clone>(
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
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    ModelResult {
        name: name.to_string(),
        mean_accuracy: mean,
        std_accuracy: variance.sqrt(),
        cv_time_ms: elapsed,
    }
}

fn delta_str(scry: f64, other: f64) -> String {
    let diff = (scry - other) * 100.0;
    if diff.abs() < 0.05 {
        "  tie".to_string()
    } else if diff > 0.0 {
        format!("+{diff:.1}%")
    } else {
        format!("{diff:.1}%")
    }
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  scry-learn Consolidated Benchmark Comparison");
    println!("  scry-learn vs scikit-learn vs XGBoost vs LightGBM");
    println!("═══════════════════════════════════════════════════════════════════════════");

    // Load Python baselines
    let sklearn = load_python_accuracy("sklearn_cv_results.json");
    let xgboost = load_python_accuracy("xgboost_results.json");
    let lightgbm = load_python_accuracy("lightgbm_results.json");

    let datasets = ["iris", "wine", "breast_cancer", "digits"];

    // ─── SECTION 1: Accuracy Comparison ─────────────────────────
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SECTION 1: 5-Fold Stratified CV Accuracy                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    for &ds_name in &datasets {
        let data = load_dataset(ds_name);

        // Scale data for models that need it
        let mut scaled = data.clone();
        let mut scaler = StandardScaler::new();
        scaler.fit(&scaled).unwrap();
        scaler.transform(&mut scaled).unwrap();

        let results: Vec<ModelResult> = vec![
            run_cv(
                "decision_tree",
                &DecisionTreeClassifier::new().max_depth(10),
                &data,
            ),
            run_cv(
                "random_forest",
                &RandomForestClassifier::new()
                    .n_estimators(20)
                    .max_depth(10)
                    .seed(42),
                &data,
            ),
            run_cv(
                "gradient_boosting",
                &GradientBoostingClassifier::new()
                    .n_estimators(100)
                    .max_depth(5)
                    .learning_rate(0.1),
                &data,
            ),
            run_cv(
                "hist_gbt",
                &HistGradientBoostingClassifier::new()
                    .n_estimators(100)
                    .max_depth(6)
                    .learning_rate(0.1),
                &data,
            ),
            run_cv(
                "logistic_regression",
                &LogisticRegression::new().max_iter(500).learning_rate(0.01),
                &scaled,
            ),
            run_cv("knn", &KnnClassifier::new().k(5), &scaled),
            run_cv("gaussian_nb", &GaussianNb::new(), &data),
            run_cv(
                "linear_svc",
                &LinearSVC::new().c(1.0).max_iter(1000),
                &scaled,
            ),
        ];

        println!(
            "\n  Dataset: {} ({} × {}, {} classes)",
            ds_name,
            data.n_samples(),
            data.n_features(),
            data.n_classes()
        );
        println!(
            "  {:23} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Model", "scry", "sklearn", "Δ skl", "xgb/lgb", "Δ"
        );
        println!("  {}", "-".repeat(73));

        for r in &results {
            let sk_acc = sklearn
                .get(ds_name)
                .and_then(|m| m.get(&r.name))
                .map(|a| a.mean);

            // For XGBoost/LightGBM, only HistGBT comparison is meaningful
            let xgb_acc = if r.name == "hist_gbt" {
                xgboost
                    .get(ds_name)
                    .and_then(|m| m.get("default"))
                    .map(|a| a.mean)
            } else {
                None
            };
            let lgb_acc = if r.name == "hist_gbt" {
                lightgbm
                    .get(ds_name)
                    .and_then(|m| m.get("default"))
                    .map(|a| a.mean)
            } else {
                None
            };

            let sk_str = sk_acc.map_or("  n/a".to_string(), |v| format!("{v:.4}"));
            let sk_delta = sk_acc.map_or(String::new(), |v| delta_str(r.mean_accuracy, v));

            // Show XGBoost or LightGBM (whichever is available, prefer XGBoost)
            let (xgb_str, xgb_delta) = if let Some(xgb) = xgb_acc {
                (format!("{xgb:.4}"), delta_str(r.mean_accuracy, xgb))
            } else if let Some(lgb) = lgb_acc {
                (format!("{lgb:.4}"), delta_str(r.mean_accuracy, lgb))
            } else {
                (String::new(), String::new())
            };

            println!(
                "  {:23} {:>7.4} {:>8} {:>8} {:>8} {:>8}",
                r.name, r.mean_accuracy, sk_str, sk_delta, xgb_str, xgb_delta
            );
        }
    }

    // ─── SECTION 2: HistGBT Head-to-Head ────────────────────────
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SECTION 2: HistGBT Head-to-Head — scry vs XGBoost vs LightGBM          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    println!(
        "\n  {:18} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Dataset", "scry", "XGBoost", "Δ xgb", "LightGBM", "Δ lgb"
    );
    println!("  {}", "-".repeat(68));

    let mut scry_wins = 0;
    let mut xgb_wins = 0;
    let mut lgb_wins = 0;
    let mut ties = 0;

    for &ds_name in &datasets {
        let data = load_dataset(ds_name);
        let hgbt_result = run_cv(
            "hist_gbt",
            &HistGradientBoostingClassifier::new()
                .n_estimators(100)
                .max_depth(6)
                .learning_rate(0.1),
            &data,
        );

        let xgb_acc = xgboost
            .get(ds_name)
            .and_then(|m| m.get("default"))
            .map_or(0.0, |a| a.mean);
        let lgb_acc = lightgbm
            .get(ds_name)
            .and_then(|m| m.get("default"))
            .map_or(0.0, |a| a.mean);

        let dx = delta_str(hgbt_result.mean_accuracy, xgb_acc);
        let dl = delta_str(hgbt_result.mean_accuracy, lgb_acc);

        // Count wins (within 0.5% is a tie)
        let diff_xgb = (hgbt_result.mean_accuracy - xgb_acc).abs();
        let diff_lgb = (hgbt_result.mean_accuracy - lgb_acc).abs();

        if diff_xgb < 0.005 && diff_lgb < 0.005 {
            ties += 1;
        } else if hgbt_result.mean_accuracy >= xgb_acc && hgbt_result.mean_accuracy >= lgb_acc {
            scry_wins += 1;
        } else if xgb_acc >= lgb_acc {
            xgb_wins += 1;
        } else {
            lgb_wins += 1;
        }

        println!(
            "  {:18} {:>7.4} {:>8.4} {:>8} {:>8.4} {:>8}",
            ds_name, hgbt_result.mean_accuracy, xgb_acc, dx, lgb_acc, dl
        );
    }

    println!("  {}", "-".repeat(68));
    println!(
        "  Scoreboard: scry wins {scry_wins}, XGBoost wins {xgb_wins}, LightGBM wins {lgb_wins}, ties {ties}"
    );

    // ─── SECTION 3: Prediction Latency ──────────────────────────
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SECTION 3: Single-Row Prediction Latency (p50 / p95 / p99)             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    let data = load_dataset("iris");

    // Pre-train models
    let mut dt = DecisionTreeClassifier::new().max_depth(10);
    dt.fit(&data).unwrap();

    let mut rf = RandomForestClassifier::new()
        .n_estimators(20)
        .max_depth(10)
        .seed(42);
    rf.fit(&data).unwrap();

    let mut hgbt = HistGradientBoostingClassifier::new()
        .n_estimators(100)
        .max_depth(6)
        .learning_rate(0.1);
    hgbt.fit(&data).unwrap();

    let mut nb = GaussianNb::new();
    nb.fit(&data).unwrap();

    let mut knn = KnnClassifier::new().k(5);
    let mut scaled = data.clone();
    let mut scaler = StandardScaler::new();
    scaler.fit(&scaled).unwrap();
    scaler.transform(&mut scaled).unwrap();
    knn.fit(&scaled).unwrap();

    let single_row = vec![data.sample(0)];
    let scaled_row = vec![scaled.sample(0)];

    println!(
        "\n  {:23} {:>10} {:>10} {:>10}",
        "Model", "p50", "p95", "p99"
    );
    println!("  {}", "-".repeat(55));

    let models_latency: Vec<(&str, Box<dyn Fn()>)> = vec![
        (
            "Decision Tree",
            Box::new(|| {
                dt.predict(&single_row).unwrap();
            }),
        ),
        (
            "Random Forest",
            Box::new(|| {
                rf.predict(&single_row).unwrap();
            }),
        ),
        (
            "HistGBT",
            Box::new(|| {
                hgbt.predict(&single_row).unwrap();
            }),
        ),
        (
            "Gaussian NB",
            Box::new(|| {
                nb.predict(&single_row).unwrap();
            }),
        ),
        (
            "KNN (k=5)",
            Box::new(|| {
                knn.predict(&scaled_row).unwrap();
            }),
        ),
    ];

    for (name, predict_fn) in &models_latency {
        let stats = measure_latency(predict_fn, 10_000);
        println!(
            "  {:23} {:>10} {:>10} {:>10}",
            name,
            fmt_time(stats.p50_us),
            fmt_time(stats.p95_us),
            fmt_time(stats.p99_us)
        );
    }

    // ─── SECTION 4: Memory Footprint ────────────────────────────
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║  SECTION 4: Memory Footprint                                            ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    if let Some(rss) = peak_rss_kb() {
        println!(
            "\n  Process peak RSS (VmHWM): {} KB ({:.1} MB)",
            rss,
            rss as f64 / 1024.0
        );
    } else {
        println!("\n  Peak RSS: not available (Linux /proc/self/status required)");
    }

    // Model serialization size (using serde_json if serde feature enabled)
    println!("\n  Note: Model serialization sizes require the `serde` feature flag.");
    println!("  Run with: cargo run --example benchmark_comparison -p scry-learn --release --features serde");

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("  ⚠️  All timing numbers approximate — system was under load during collection.");
    println!("  Re-run on an idle system for production-grade latency measurements.");
    println!("═══════════════════════════════════════════════════════════════════════════");
}
