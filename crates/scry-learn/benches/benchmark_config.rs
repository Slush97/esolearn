#![allow(clippy::needless_range_loop)]
//! Centralized benchmark configuration: canonical model hyperparameters,
//! dataset loaders, data generators, and result schema.
//!
//! **Goal**: single source of truth so every benchmark file uses identical
//! configurations, identical data, and identical hyperparameters.
//!
//! Import this module from any `[[bench]]` or `#[test]` harness with:
//! ```ignore
//! #[path = "../benches/benchmark_config.rs"]
//! mod benchmark_config;
//! ```

#![allow(dead_code, missing_docs)]

use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

/// Global seed for all benchmarks. Changing this invalidates all golden baselines.
pub const SEED: u64 = 42;

/// Standard benchmark sizes (samples).
pub const SIZES_SMALL: &[usize] = &[1_000, 5_000, 10_000];
pub const SIZES_MEDIUM: &[usize] = &[10_000, 50_000, 100_000];

/// Path to test fixture CSVs.
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

// ═══════════════════════════════════════════════════════════════════
// Canonical Model Configurations
// ═══════════════════════════════════════════════════════════════════

/// Every model's benchmark hyperparameters, matching sklearn defaults exactly.
/// These are the ONLY configs that should appear in benchmark code.
pub mod configs {
    /// Decision Tree: `max_depth=10` (sklearn: `max_depth=10`)
    pub const DT_MAX_DEPTH: usize = 10;

    /// Random Forest: `n_estimators=20, max_depth=10, seed=42`
    /// sklearn: `n_estimators=20, max_depth=10, random_state=42`
    pub const RF_N_ESTIMATORS: usize = 20;
    pub const RF_MAX_DEPTH: usize = 10;

    /// Gradient Boosting: `n_estimators=100, max_depth=5, lr=0.1`
    /// sklearn: identical
    pub const GBT_N_ESTIMATORS: usize = 100;
    pub const GBT_MAX_DEPTH: usize = 5;
    pub const GBT_LR: f64 = 0.1;

    /// `HistGBT`: `n_estimators=100, max_depth=6, lr=0.1`
    /// sklearn: `max_iter=100, max_depth=6, learning_rate=0.1`
    pub const HGBT_N_ESTIMATORS: usize = 100;
    pub const HGBT_MAX_DEPTH: usize = 6;
    pub const HGBT_LR: f64 = 0.1;

    /// Logistic Regression: `max_iter=200, lr=0.01`
    /// sklearn: `max_iter=200, solver='lbfgs', C=1.0`
    pub const LOGREG_MAX_ITER: usize = 200;
    pub const LOGREG_LR: f64 = 0.01;

    /// KNN: `k=5, uniform weights`
    /// sklearn: `n_neighbors=5, weights='uniform'`
    pub const KNN_K: usize = 5;

    /// `LinearSVC`: `C=1.0, max_iter=2000`
    /// sklearn: `max_iter=2000, dual='auto'`
    pub const SVC_C: f64 = 1.0;
    pub const SVC_MAX_ITER: usize = 2000;

    /// `KernelSVC`: `C=1.0, RBF gamma=0.1`
    pub const KSVC_C: f64 = 1.0;
    pub const KSVC_GAMMA: f64 = 0.1;

    /// Lasso: `alpha=0.01`
    /// sklearn: `alpha=0.01, max_iter=1000`
    pub const LASSO_ALPHA: f64 = 0.01;

    /// `ElasticNet`: `alpha=0.01, l1_ratio=0.5`
    /// sklearn: identical
    pub const ENET_ALPHA: f64 = 0.01;
    pub const ENET_L1_RATIO: f64 = 0.5;

    /// Ridge: `alpha=1.0`
    pub const RIDGE_ALPHA: f64 = 1.0;

    /// `KMeans`: `k=3, max_iter=100, n_init=3, seed=42`
    pub const KMEANS_K: usize = 3;
    pub const KMEANS_MAX_ITER: usize = 100;

    /// DBSCAN: `eps=1.0, min_samples=5`
    pub const DBSCAN_EPS: f64 = 1.0;
    pub const DBSCAN_MIN_SAMPLES: usize = 5;

    /// Isolation Forest: `n_estimators=100, seed=42`
    pub const IFOREST_N_ESTIMATORS: usize = 100;

    /// `MultinomialNB`: `alpha=1.0` (Laplace smoothing)
    /// sklearn: `alpha=1.0`
    pub const MNB_ALPHA: f64 = 1.0;

    /// `BernoulliNB`: `alpha=1.0, binarize=0.0`
    /// sklearn: `alpha=1.0, binarize=0.0`
    pub const BNB_ALPHA: f64 = 1.0;
    pub const BNB_BINARIZE: f64 = 0.0;

    /// MLPClassifier/Regressor: `hidden=[64,32], max_iter=300, lr=0.001, seed=42`
    /// sklearn: `hidden_layer_sizes=(64,32), max_iter=300, learning_rate_init=0.001, random_state=42`
    pub const MLP_HIDDEN: &[usize] = &[64, 32];
    pub const MLP_MAX_ITER: usize = 300;
    pub const MLP_LR: f64 = 0.001;
    pub const MLP_SEED: u64 = 42;

    /// `DecisionTreeRegressor`: `max_depth=10`
    pub const DTR_MAX_DEPTH: usize = 10;

    /// `RandomForestRegressor`: `n_estimators=20, max_depth=10`
    pub const RFR_N_ESTIMATORS: usize = 20;
    pub const RFR_MAX_DEPTH: usize = 10;

    /// `HistGBTRegressor`: `n_estimators=100, max_depth=6, lr=0.1`
    pub const HGBTR_N_ESTIMATORS: usize = 100;
    pub const HGBTR_MAX_DEPTH: usize = 6;
    pub const HGBTR_LR: f64 = 0.1;

    /// `LinearSVR`: `C=1.0, epsilon=0.1, max_iter=2000`
    pub const SVR_C: f64 = 1.0;
    pub const SVR_EPSILON: f64 = 0.1;
    pub const SVR_MAX_ITER: usize = 2000;

    /// `KernelSVR`: `C=1.0, epsilon=0.1, gamma=0.1`
    pub const KSVR_C: f64 = 1.0;
    pub const KSVR_EPSILON: f64 = 0.1;
    pub const KSVR_GAMMA: f64 = 0.1;

    /// `MiniBatchKMeans`: `k=3, batch_size=100, seed=42`
    pub const MBKM_K: usize = 3;
    pub const MBKM_BATCH_SIZE: usize = 100;

    /// HDBSCAN: `min_cluster_size=5`
    pub const HDBSCAN_MIN_CLUSTER_SIZE: usize = 5;

    /// `AgglomerativeClustering`: `n_clusters=3`
    pub const AGGLO_N_CLUSTERS: usize = 3;
}

// ═══════════════════════════════════════════════════════════════════
// Data Generators (canonical versions — use ONLY these)
// ═══════════════════════════════════════════════════════════════════

use scry_learn::dataset::Dataset;

/// Generate a binary classification dataset with well-separated classes.
/// Class 0: features ~ U(0, 2), Class 1: features ~ U(offset, offset+2).
pub fn gen_classification(n: usize, n_features: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut features: Vec<Vec<f64>> = (0..n_features).map(|_| Vec::with_capacity(n)).collect();
    let mut target = Vec::with_capacity(n);

    for i in 0..n {
        let class = if i < n / 2 { 0.0 } else { 1.0 };
        for (j, col) in features.iter_mut().enumerate() {
            let offset = if class == 0.0 {
                0.0
            } else {
                3.0 + j as f64 * 0.5
            };
            col.push(rng.f64() * 2.0 + offset);
        }
        target.push(class);
    }

    let names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "class")
}

/// Generate a multiclass classification dataset.
pub fn gen_multiclass(n: usize, n_features: usize, n_classes: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];
    let per_class = n / n_classes;

    for i in 0..n {
        let class = (i / per_class).min(n_classes - 1);
        target[i] = class as f64;
        for j in 0..n_features {
            col_major[j][i] = rng.f64() * 2.0 + (class as f64) * 2.0 + (j as f64) * 0.3;
        }
    }

    let names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    Dataset::new(col_major, target, names, "class")
}

/// Generate a regression dataset: y = Σ `x_j·(j+1)` + noise.
pub fn gen_regression(n: usize, n_features: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n_features {
            let v = rng.f64() * 10.0;
            col_major[j][i] = v;
            sum += v * (j as f64 + 1.0);
        }
        target[i] = sum + rng.f64() * 0.1;
    }

    let names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    Dataset::new(col_major, target, names, "y")
}

/// Generate a dataset with outliers for anomaly detection.
/// 95% normal, 5% outlier (shifted by 10σ).
pub fn gen_anomaly(n: usize, n_features: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let n_outlier = n / 20;
    let n_normal = n - n_outlier;
    let mut col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n]; // 0=normal, 1=outlier

    for i in 0..n {
        let is_outlier = i >= n_normal;
        for j in 0..n_features {
            let offset = if is_outlier { 10.0 } else { 0.0 };
            col_major[j][i] = rng.f64() * 2.0 - 1.0 + offset;
        }
        target[i] = if is_outlier { 1.0 } else { 0.0 };
    }

    let names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    Dataset::new(col_major, target, names, "outlier")
}

// ═══════════════════════════════════════════════════════════════════
// UCI Dataset Loaders (from CSV fixtures)
// ═══════════════════════════════════════════════════════════════════

/// Load a feature CSV into column-major format for Dataset.
pub fn load_features_csv(name: &str) -> (Vec<Vec<f64>>, Vec<String>) {
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

    // Transpose to column-major.
    let mut cols = vec![vec![0.0; rows.len()]; n_cols];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            cols[j][i] = val;
        }
    }
    (cols, headers)
}

/// Load a target CSV.
pub fn load_target_csv(name: &str) -> Vec<f64> {
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

/// Load a UCI dataset from CSV fixtures.
pub fn load_dataset(base: &str) -> Dataset {
    let (features, feat_names) = load_features_csv(&format!("{base}_features.csv"));
    let target = load_target_csv(&format!("{base}_target.csv"));
    Dataset::new(features, target, feat_names, "target")
}

/// Build row-major feature matrix from a Dataset.
pub fn to_row_major(ds: &Dataset) -> Vec<Vec<f64>> {
    let n = ds.n_samples();
    let d = ds.features.len();
    (0..n)
        .map(|i| (0..d).map(|j| ds.features[j][i]).collect())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Golden Baseline Types
// ═══════════════════════════════════════════════════════════════════

/// A golden baseline entry: expected metric for a specific model × dataset.
#[derive(Debug, Clone)]
pub struct GoldenBaseline {
    pub model: &'static str,
    pub dataset: &'static str,
    pub metric: &'static str,
    pub expected: f64,
    pub tolerance: f64,
}

/// All golden baselines, derived from verified BENCHMARKS.md results.
/// These use 5-fold stratified CV with seed=42.
pub fn golden_baselines_classification() -> Vec<GoldenBaseline> {
    vec![
        // ── Iris ──
        GoldenBaseline { model: "DecisionTree", dataset: "iris", metric: "accuracy_5fold_cv", expected: 0.9467, tolerance: 0.025 },
        GoldenBaseline { model: "RandomForest", dataset: "iris", metric: "accuracy_5fold_cv", expected: 0.9533, tolerance: 0.025 },
        GoldenBaseline { model: "GradientBoosting", dataset: "iris", metric: "accuracy_5fold_cv", expected: 0.9533, tolerance: 0.025 },
        GoldenBaseline { model: "HistGBT", dataset: "iris", metric: "accuracy_5fold_cv", expected: 0.9333, tolerance: 0.025 },
        GoldenBaseline { model: "GaussianNB", dataset: "iris", metric: "accuracy_5fold_cv", expected: 0.9600, tolerance: 0.025 },
        GoldenBaseline { model: "KNN", dataset: "iris", metric: "accuracy_5fold_cv", expected: 0.9467, tolerance: 0.035 },
        // ── Wine ──
        GoldenBaseline { model: "DecisionTree", dataset: "wine", metric: "accuracy_5fold_cv", expected: 0.9034, tolerance: 0.030 },
        GoldenBaseline { model: "RandomForest", dataset: "wine", metric: "accuracy_5fold_cv", expected: 0.9495, tolerance: 0.030 },
        GoldenBaseline { model: "GradientBoosting", dataset: "wine", metric: "accuracy_5fold_cv", expected: 0.9552, tolerance: 0.030 },
        GoldenBaseline { model: "HistGBT", dataset: "wine", metric: "accuracy_5fold_cv", expected: 0.9493, tolerance: 0.030 },
        GoldenBaseline { model: "GaussianNB", dataset: "wine", metric: "accuracy_5fold_cv", expected: 0.9663, tolerance: 0.025 },
        // ── Breast Cancer ──
        GoldenBaseline { model: "DecisionTree", dataset: "breast_cancer", metric: "accuracy_5fold_cv", expected: 0.9350, tolerance: 0.025 },
        GoldenBaseline { model: "RandomForest", dataset: "breast_cancer", metric: "accuracy_5fold_cv", expected: 0.9526, tolerance: 0.025 },
        GoldenBaseline { model: "GradientBoosting", dataset: "breast_cancer", metric: "accuracy_5fold_cv", expected: 0.9312, tolerance: 0.030 },
        GoldenBaseline { model: "HistGBT", dataset: "breast_cancer", metric: "accuracy_5fold_cv", expected: 0.9649, tolerance: 0.025 },
        GoldenBaseline { model: "GaussianNB", dataset: "breast_cancer", metric: "accuracy_5fold_cv", expected: 0.9421, tolerance: 0.025 },
        // ── Digits ──
        GoldenBaseline { model: "DecisionTree", dataset: "digits", metric: "accuracy_5fold_cv", expected: 0.8431, tolerance: 0.030 },
        GoldenBaseline { model: "RandomForest", dataset: "digits", metric: "accuracy_5fold_cv", expected: 0.9610, tolerance: 0.020 },
        GoldenBaseline { model: "GradientBoosting", dataset: "digits", metric: "accuracy_5fold_cv", expected: 0.9571, tolerance: 0.020 },
        GoldenBaseline { model: "HistGBT", dataset: "digits", metric: "accuracy_5fold_cv", expected: 0.9711, tolerance: 0.015 },
        GoldenBaseline { model: "GaussianNB", dataset: "digits", metric: "accuracy_5fold_cv", expected: 0.8237, tolerance: 0.035 },
        GoldenBaseline { model: "KNN", dataset: "digits", metric: "accuracy_5fold_cv", expected: 0.9760, tolerance: 0.015 },
        // ── MultinomialNB ──
        GoldenBaseline { model: "MultinomialNB", dataset: "iris", metric: "accuracy_5fold_cv", expected: 0.9533, tolerance: 0.035 },
        GoldenBaseline { model: "MultinomialNB", dataset: "digits", metric: "accuracy_5fold_cv", expected: 0.8700, tolerance: 0.040 },
        // ── BernoulliNB ──
        GoldenBaseline { model: "BernoulliNB", dataset: "breast_cancer", metric: "accuracy_5fold_cv", expected: 0.6274, tolerance: 0.040 },
        GoldenBaseline { model: "BernoulliNB", dataset: "digits", metric: "accuracy_5fold_cv", expected: 0.8200, tolerance: 0.060 },
    ]
}

/// Golden baselines for regression (California Housing, 80/20 split).
pub fn golden_baselines_regression() -> Vec<GoldenBaseline> {
    vec![
        GoldenBaseline { model: "LinearRegression", dataset: "california", metric: "r2", expected: 0.5588, tolerance: 0.030 },
        GoldenBaseline { model: "Lasso", dataset: "california", metric: "r2", expected: 0.5717, tolerance: 0.030 },
        GoldenBaseline { model: "ElasticNet", dataset: "california", metric: "r2", expected: 0.5686, tolerance: 0.030 },
        GoldenBaseline { model: "Ridge", dataset: "california", metric: "r2", expected: 0.5588, tolerance: 0.030 },
        GoldenBaseline { model: "GBTRegressor", dataset: "california", metric: "r2", expected: 0.7879, tolerance: 0.025 },
        GoldenBaseline { model: "DTRegressor", dataset: "california", metric: "r2", expected: 0.6570, tolerance: 0.050 },
        GoldenBaseline { model: "RFRegressor", dataset: "california", metric: "r2", expected: 0.7602, tolerance: 0.040 },
        GoldenBaseline { model: "HistGBTRegressor", dataset: "california", metric: "r2", expected: 0.8200, tolerance: 0.040 },
        GoldenBaseline { model: "LinearSVR", dataset: "california", metric: "r2", expected: 0.5617, tolerance: 0.050 },

        GoldenBaseline { model: "MLPRegressor", dataset: "california", metric: "r2", expected: 0.7089, tolerance: 0.050 },
    ]
}

// ═══════════════════════════════════════════════════════════════════
// Hardware Metadata
// ═══════════════════════════════════════════════════════════════════

/// Capture hardware and environment details for result JSON.
pub fn hardware_metadata() -> Vec<(&'static str, String)> {
    let mut meta = Vec::new();

    // CPU info
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in cpuinfo.lines() {
            if line.starts_with("model name") {
                if let Some(name) = line.split(':').nth(1) {
                    meta.push(("cpu_model", name.trim().to_string()));
                    break;
                }
            }
        }
    }

    // Memory
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        meta.push(("ram_gb", format!("{:.1}", kb as f64 / 1_048_576.0)));
                    }
                }
                break;
            }
        }
    }

    meta.push(("os", std::env::consts::OS.to_string()));
    meta.push(("arch", std::env::consts::ARCH.to_string()));
    meta.push(("rust_version", env!("CARGO_PKG_RUST_VERSION").to_string()));

    meta
}

// ═══════════════════════════════════════════════════════════════════
// Result Schema (for JSON output)
// ═══════════════════════════════════════════════════════════════════

/// A single benchmark result.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub model: String,
    pub dataset: String,
    pub metric_name: String,
    pub metric_value: f64,
    pub fit_time_us: Option<f64>,
    pub predict_time_us: Option<f64>,
}

impl BenchResult {
    /// Format as a fixed-width table row.
    pub fn print_row(&self) {
        print!("  {:<28} {:<16}", self.model, self.dataset);
        print!(" {:>10}: {:.4}", self.metric_name, self.metric_value);
        if let Some(fit) = self.fit_time_us {
            print!("  fit: {fit:.0}µs");
        }
        if let Some(pred) = self.predict_time_us {
            print!("  pred: {pred:.1}µs");
        }
        println!();
    }
}
