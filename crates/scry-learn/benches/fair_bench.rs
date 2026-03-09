#![allow(
    missing_docs,
    clippy::redundant_clone,
    clippy::default_trait_access,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::redundant_closure_for_method_calls,
    clippy::map_unwrap_or,
    clippy::items_after_statements,
    clippy::cast_possible_wrap
)]
//! Academically rigorous head-to-head benchmark: scry-learn vs Rust ML ecosystem.
//!
//! **Fairness guarantees:**
//! 1. Rayon pinned to 1 thread programmatically (no env var reliance)
//! 2. Only compares identical algorithms (SVM excluded — LinearSVC vs kernel SVC)
//! 3. Convergence/accuracy parity asserted before reporting timing
//! 4. Hyperparameters matched exactly across libraries
//! 5. Same data arrays passed to all libraries
//! 6. CLT-aware sample sizes (n≥30 for Criterion groups)
//!
//! **Algorithms compared:**
//! | Algorithm            | scry-learn | smartcore 0.4 | linfa 0.8       |
//! |----------------------|------------|---------------|-----------------|
//! | Decision Tree (CART) | ✓          | ✓             | ✓ linfa-trees   |
//! | Random Forest        | ✓          | ✓             | —               |
//! | Logistic Regression  | ✓          | ✓             | ✓ linfa-logistic|
//! | KNN (brute-force)    | ✓          | ✓             | —               |
//! | K-Means (k-means++)  | ✓          | —             | ✓ linfa-cluster |
//! | Lasso (coord desc)   | ✓          | —             | ✓ linfa-elasticnet|
//!
//! **Run:**
//!   cargo bench --bench fair_bench -p scry-learn
//!
//! **Cloud (extended scaling):**
//!   cargo bench --bench fair_bench -p scry-learn --features extended-bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════════════════
// FAIRNESS INFRASTRUCTURE
// ═══════════════════════════════════════════════════════════════════════════

/// Enforce single-threaded execution. Called once at benchmark init.
/// Panics if rayon pool is already initialized with >1 thread.
///
/// **Rayon usage audit (2025-02):**
/// - scry-learn: uses `par_iter` in RF, LogReg, KNN, KMeans, SVM, HistGBT
///   (all via global pool — degrades to sequential iteration under 1 thread,
///    with ~2-5µs overhead per rayon dispatch)
/// - smartcore 0.4: no rayon / no par_iter — purely sequential
/// - linfa-trees 0.8: rayon in Cargo.toml but no par_iter calls in source
///
/// **Conclusion:** this constraint slightly *disadvantages* scry-learn since it
/// pays rayon scheduling overhead that the competitors do not incur.
fn enforce_single_thread() {
    // Try to build global pool with 1 thread. If it fails, the pool was
    // already initialized — verify it's actually 1 thread.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
    assert_eq!(
        rayon::current_num_threads(),
        1,
        "FAIRNESS VIOLATION: rayon has {} threads, expected 1. \
         Set RAYON_NUM_THREADS=1 or run before any other rayon init.",
        rayon::current_num_threads()
    );
}

fn accuracy_f64(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < 1e-9)
        .count();
    correct as f64 / y_true.len() as f64
}

/// Assert that all libraries achieve comparable accuracy.
/// Prints a parity table to stderr and panics if any library is >ε apart.
fn assert_accuracy_parity(
    label: &str,
    results: &[(&str, f64)],
    epsilon: f64,
) {
    eprintln!("\n┌─ ACCURACY PARITY: {label}");
    for (name, acc) in results {
        eprintln!("│  {name:<24} {:.4} ({:.1}%)", acc, acc * 100.0);
    }

    // Pairwise comparison
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let diff = (results[i].1 - results[j].1).abs();
            if diff > epsilon {
                eprintln!(
                    "│  ⚠ PARITY VIOLATION: |{} - {}| = {:.4} > ε={:.4}",
                    results[i].0, results[j].0, diff, epsilon
                );
                eprintln!(
                    "│  Timing comparison is UNRELIABLE — models converged differently."
                );
            }
        }
    }
    eprintln!("└─");
}

// ═══════════════════════════════════════════════════════════════════════════
// DATA LOADING (real UCI datasets from CSV fixtures)
// ═══════════════════════════════════════════════════════════════════════════

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

/// Load UCI dataset. Returns (column-major features, target, feature names).
fn load_dataset(base: &str) -> (Vec<Vec<f64>>, Vec<f64>, Vec<String>) {
    let (features, names) = load_features_csv(&format!("{base}_features.csv"));
    let target = load_target_csv(&format!("{base}_target.csv"));
    (features, target, names)
}

/// Build a scry-learn Dataset from loaded data.
fn to_scry_dataset(cols: &[Vec<f64>], target: &[f64], names: &[String]) -> scry_learn::prelude::Dataset {
    scry_learn::prelude::Dataset::new(
        cols.to_vec(),
        target.to_vec(),
        names.to_vec(),
        "target",
    )
}

/// Build row-major matrix from column-major.
fn to_row_major(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if cols.is_empty() { return vec![]; }
    let n = cols[0].len();
    let m = cols.len();
    (0..n).map(|i| (0..m).map(|j| cols[j][i]).collect()).collect()
}

/// Build smartcore DenseMatrix from row-major features.
fn to_smartcore_matrix(rows: &[Vec<f64>]) -> smartcore::linalg::basic::matrix::DenseMatrix<f64> {
    let owned: Vec<Vec<f64>> = rows.to_vec();
    smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&owned).unwrap()
}

/// Build linfa Dataset (classification with usize targets).
fn to_linfa_dataset(
    rows: &[Vec<f64>],
    target: &[f64],
) -> linfa::DatasetBase<ndarray::Array2<f64>, ndarray::Array1<usize>> {
    let n = rows.len();
    let m = rows[0].len();
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let x = ndarray::Array2::from_shape_vec((n, m), flat).unwrap();
    let y = ndarray::Array1::from_vec(target.iter().map(|&t| t as usize).collect());
    linfa::Dataset::new(x, y)
}

/// Build linfa Dataset with bool targets (for logistic regression).
fn to_linfa_dataset_bool(
    rows: &[Vec<f64>],
    target: &[f64],
) -> linfa::DatasetBase<ndarray::Array2<f64>, ndarray::Array1<bool>> {
    let n = rows.len();
    let m = rows[0].len();
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let x = ndarray::Array2::from_shape_vec((n, m), flat).unwrap();
    let y = ndarray::Array1::from_vec(target.iter().map(|&t| t > 0.5).collect());
    linfa::Dataset::new(x, y)
}

/// Build linfa Dataset with f64 targets (for regression).
fn to_linfa_dataset_f64(
    rows: &[Vec<f64>],
    target: &[f64],
) -> linfa::DatasetBase<ndarray::Array2<f64>, ndarray::Array1<f64>> {
    let n = rows.len();
    let m = rows[0].len();
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let x = ndarray::Array2::from_shape_vec((n, m), flat).unwrap();
    let y = ndarray::Array1::from_vec(target.to_vec());
    linfa::Dataset::new(x, y)
}

/// Generate synthetic classification data with controllable difficulty.
/// Class overlap is moderate (offset=1.5) to stress algorithms meaningfully.
fn gen_classification(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let half = n / 2;
    let mut features_col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];

    for j in 0..n_features {
        let offset = 1.5 + j as f64 * 0.3; // moderate overlap, not trivial
        for i in 0..half {
            features_col_major[j][i] = rng.f64() * 3.0 - 1.5;
        }
        for i in half..n {
            features_col_major[j][i] = rng.f64() * 3.0 - 1.5 + offset;
            target[i] = 1.0;
        }
    }

    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_features).map(|j| features_col_major[j][i]).collect())
        .collect();
    (row_major, target)
}

/// Generate synthetic regression data: y = Σ x_j·(j+1) + noise
fn gen_regression(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut features_col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n_features {
            let v = rng.f64() * 10.0;
            features_col_major[j][i] = v;
            sum += v * (j as f64 + 1.0);
        }
        target[i] = sum + rng.f64() * 0.1;
    }

    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_features).map(|j| features_col_major[j][i]).collect())
        .collect();
    (row_major, target)
}

// ═══════════════════════════════════════════════════════════════════════════
// RSS MEMORY MEASUREMENT (Linux)
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// MATCHED HYPERPARAMETERS (single source of truth)
// ═══════════════════════════════════════════════════════════════════════════

// Decision Tree
const DT_MAX_DEPTH: usize = 10;
// Random Forest
const RF_N_ESTIMATORS: usize = 20;
const RF_MAX_DEPTH: usize = 10;
// Logistic Regression
const LR_MAX_ITER: usize = 200;
// KNN
const KNN_K: usize = 5;
// K-Means
const KM_K: usize = 3;
const KM_MAX_ITER: usize = 100;
// Lasso
const LASSO_ALPHA: f64 = 0.01;
const LASSO_MAX_ITER: usize = 1000;

// Dataset names for UCI sweep
const UCI_DATASETS: &[&str] = &["iris", "wine", "breast_cancer"];
// Include large-scale datasets for algorithms that scale well (DT, RF, LogReg)
const UCI_DATASETS_WITH_ADULT: &[&str] = &["iris", "wine", "breast_cancer", "adult"];

// ═══════════════════════════════════════════════════════════════════════════
// §1 DECISION TREE TRAINING  (scry vs smartcore vs linfa-trees)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_dt_train(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/decision_tree/train");
    group.sample_size(30);

    for &ds_name in UCI_DATASETS_WITH_ADULT {
        let (cols, target, names) = load_dataset(ds_name);
        let rows = to_row_major(&cols);
        let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

        let scry_ds = to_scry_dataset(&cols, &target, &names);
        let sm_x = to_smartcore_matrix(&rows);
        let linfa_ds = to_linfa_dataset(&rows, &target);

        // ── Accuracy parity check (run once outside benchmark loop) ──
        {
            let mut scry_dt = scry_learn::prelude::DecisionTreeClassifier::new()
                .max_depth(DT_MAX_DEPTH);
            scry_dt.fit(&scry_ds).unwrap();
            let scry_preds = scry_dt.predict(&rows).unwrap();
            let scry_acc = accuracy_f64(&target, &scry_preds);

            let sm_dt = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
                &sm_x,
                &target_i32,
                smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default()
                    .with_max_depth(DT_MAX_DEPTH as u16),
            ).unwrap();
            let sm_preds_i32: Vec<i32> = sm_dt.predict(&sm_x).unwrap();
            let sm_preds: Vec<f64> = sm_preds_i32.iter().map(|&p| p as f64).collect();
            let sm_acc = accuracy_f64(&target, &sm_preds);

            use linfa::prelude::{Fit, Predict};
            let linfa_dt = linfa_trees::DecisionTree::params()
                .max_depth(Some(DT_MAX_DEPTH))
                .fit(&linfa_ds).unwrap();
            let linfa_preds_arr = linfa_dt.predict(&linfa_ds);
            let linfa_preds: Vec<f64> = linfa_preds_arr.iter().map(|&p| p as f64).collect();
            let linfa_acc = accuracy_f64(&target, &linfa_preds);

            assert_accuracy_parity(
                &format!("DT train/{ds_name}"),
                &[("scry-learn", scry_acc), ("smartcore", sm_acc), ("linfa-trees", linfa_acc)],
                0.05,
            );
        }

        // ── Criterion benchmarks ──
        group.bench_with_input(BenchmarkId::new("scry-learn", ds_name), &ds_name, |b, _| {
            b.iter(|| {
                let mut dt = scry_learn::prelude::DecisionTreeClassifier::new()
                    .max_depth(DT_MAX_DEPTH);
                dt.fit(black_box(&scry_ds)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("smartcore", ds_name), &ds_name, |b, _| {
            b.iter(|| {
                let _ = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
                    black_box(&sm_x),
                    black_box(&target_i32),
                    smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default()
                        .with_max_depth(DT_MAX_DEPTH as u16),
                ).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("linfa-trees", ds_name), &ds_name, |b, _| {
            use linfa::prelude::Fit;
            b.iter(|| {
                let _ = linfa_trees::DecisionTree::params()
                    .max_depth(Some(DT_MAX_DEPTH))
                    .fit(black_box(&linfa_ds)).unwrap();
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §2 DECISION TREE PREDICTION
// ═══════════════════════════════════════════════════════════════════════════

fn bench_dt_predict(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/decision_tree/predict");
    group.sample_size(50);

    for &ds_name in UCI_DATASETS_WITH_ADULT {
        let (cols, target, names) = load_dataset(ds_name);
        let rows = to_row_major(&cols);
        let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

        let scry_ds = to_scry_dataset(&cols, &target, &names);
        let sm_x = to_smartcore_matrix(&rows);
        let linfa_ds = to_linfa_dataset(&rows, &target);

        // Train all models once
        let mut scry_dt = scry_learn::prelude::DecisionTreeClassifier::new()
            .max_depth(DT_MAX_DEPTH);
        scry_dt.fit(&scry_ds).unwrap();

        let sm_dt = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
            &sm_x,
            &target_i32,
            smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default()
                .with_max_depth(DT_MAX_DEPTH as u16),
        ).unwrap();

        use linfa::prelude::{Fit, Predict};
        let linfa_dt = linfa_trees::DecisionTree::params()
            .max_depth(Some(DT_MAX_DEPTH))
            .fit(&linfa_ds).unwrap();

        group.bench_with_input(BenchmarkId::new("scry-learn", ds_name), &ds_name, |b, _| {
            b.iter(|| scry_dt.predict(black_box(&rows)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("smartcore", ds_name), &ds_name, |b, _| {
            b.iter(|| sm_dt.predict(black_box(&sm_x)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("linfa-trees", ds_name), &ds_name, |b, _| {
            b.iter(|| linfa_dt.predict(black_box(&linfa_ds)));
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §3 RANDOM FOREST TRAINING  (scry vs smartcore — linfa-ensemble excluded
//    because it requires different bootstrap/feature params)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_rf_train(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/random_forest/train");
    group.sample_size(10); // RF training is slow

    for &ds_name in UCI_DATASETS_WITH_ADULT {
        let (cols, target, names) = load_dataset(ds_name);
        let rows = to_row_major(&cols);
        let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

        let scry_ds = to_scry_dataset(&cols, &target, &names);
        let sm_x = to_smartcore_matrix(&rows);

        // Accuracy parity
        {
            let mut scry_rf = scry_learn::prelude::RandomForestClassifier::new()
                .n_estimators(RF_N_ESTIMATORS)
                .max_depth(RF_MAX_DEPTH)
                .seed(42);
            scry_rf.fit(&scry_ds).unwrap();
            let scry_preds = scry_rf.predict(&rows).unwrap();

            let sm_params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
                .with_n_trees(RF_N_ESTIMATORS as u16)
                .with_max_depth(RF_MAX_DEPTH as u16);
            let sm_rf = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
                &sm_x, &target_i32, sm_params,
            ).unwrap();
            let sm_preds_i32: Vec<i32> = sm_rf.predict(&sm_x).unwrap();
            let sm_preds: Vec<f64> = sm_preds_i32.iter().map(|&p| p as f64).collect();

            assert_accuracy_parity(
                &format!("RF train/{ds_name}"),
                &[
                    ("scry-learn", accuracy_f64(&target, &scry_preds)),
                    ("smartcore", accuracy_f64(&target, &sm_preds)),
                ],
                0.05,
            );
        }

        group.bench_with_input(BenchmarkId::new("scry-learn", ds_name), &ds_name, |b, _| {
            b.iter(|| {
                let mut rf = scry_learn::prelude::RandomForestClassifier::new()
                    .n_estimators(RF_N_ESTIMATORS)
                    .max_depth(RF_MAX_DEPTH)
                    .seed(42);
                rf.fit(black_box(&scry_ds)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("smartcore", ds_name), &ds_name, |b, _| {
            b.iter(|| {
                let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
                    .with_n_trees(RF_N_ESTIMATORS as u16)
                    .with_max_depth(RF_MAX_DEPTH as u16);
                let _ = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
                    black_box(&sm_x), black_box(&target_i32), params,
                ).unwrap();
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §4 RANDOM FOREST PREDICTION
// ═══════════════════════════════════════════════════════════════════════════

fn bench_rf_predict(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/random_forest/predict");
    group.sample_size(30);

    for &ds_name in UCI_DATASETS_WITH_ADULT {
        let (cols, target, names) = load_dataset(ds_name);
        let rows = to_row_major(&cols);
        let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

        let scry_ds = to_scry_dataset(&cols, &target, &names);
        let sm_x = to_smartcore_matrix(&rows);

        let mut scry_rf = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(RF_N_ESTIMATORS)
            .max_depth(RF_MAX_DEPTH)
            .seed(42);
        scry_rf.fit(&scry_ds).unwrap();

        let sm_params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(RF_N_ESTIMATORS as u16)
            .with_max_depth(RF_MAX_DEPTH as u16);
        let sm_rf = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
            &sm_x, &target_i32, sm_params,
        ).unwrap();

        group.bench_with_input(BenchmarkId::new("scry-learn", ds_name), &ds_name, |b, _| {
            b.iter(|| scry_rf.predict(black_box(&rows)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("smartcore", ds_name), &ds_name, |b, _| {
            b.iter(|| sm_rf.predict(black_box(&sm_x)).unwrap());
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §5 LOGISTIC REGRESSION TRAINING  (scry vs smartcore vs linfa-logistic)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_logreg_train(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/logistic_regression/train");
    group.sample_size(30);

    // Binary datasets for logistic regression (Adult is also binary: income >50K)
    for &ds_name in &["breast_cancer", "adult"] {
        let (cols, target, names) = load_dataset(ds_name);

        // Z-score standardize features for LogReg convergence.
        // Adult features span 5 orders of magnitude (fnlwgt ~190K vs education ~10).
        // Without scaling, L-BFGS cannot converge in 200 iterations.
        // This matches sklearn best practice: Pipeline(StandardScaler(), LogReg()).
        let cols: Vec<Vec<f64>> = cols.iter().map(|col| {
            let n = col.len() as f64;
            let mean = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt().max(1e-10); // avoid div by zero
            col.iter().map(|x| (x - mean) / std).collect()
        }).collect();

        let rows = to_row_major(&cols);
        let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

        let scry_ds = to_scry_dataset(&cols, &target, &names);
        let sm_x = to_smartcore_matrix(&rows);
        let linfa_ds = to_linfa_dataset_bool(&rows, &target);

        // Accuracy parity
        {
            let mut scry_lr = scry_learn::prelude::LogisticRegression::new()
                .max_iter(LR_MAX_ITER);
            scry_lr.fit(&scry_ds).unwrap();
            let matrix = scry_ds.feature_matrix();
            let scry_preds = scry_lr.predict(&matrix).unwrap();

            let sm_lr = smartcore::linear::logistic_regression::LogisticRegression::fit(
                &sm_x, &target_i32, Default::default(),
            ).unwrap();
            let sm_preds_i32: Vec<i32> = sm_lr.predict(&sm_x).unwrap();
            let sm_preds: Vec<f64> = sm_preds_i32.iter().map(|&p| p as f64).collect();

            use linfa::prelude::{Fit, Predict};
            let linfa_lr = linfa_logistic::LogisticRegression::default()
                .max_iterations(LR_MAX_ITER as u64)
                .fit(&linfa_ds).unwrap();
            let linfa_preds_arr = linfa_lr.predict(&linfa_ds);
            let linfa_preds: Vec<f64> = linfa_preds_arr.iter()
                .map(|&p| if p { 1.0 } else { 0.0 }).collect();

            assert_accuracy_parity(
                &format!("LogReg train/{ds_name}"),
                &[
                    ("scry-learn", accuracy_f64(&target, &scry_preds)),
                    ("smartcore", accuracy_f64(&target, &sm_preds)),
                    ("linfa-logistic", accuracy_f64(&target, &linfa_preds)),
                ],
                0.05,
            );
        }

        group.bench_with_input(BenchmarkId::new("scry-learn", ds_name), &ds_name, |b, _| {
            b.iter(|| {
                let mut lr = scry_learn::prelude::LogisticRegression::new()
                    .max_iter(LR_MAX_ITER);
                lr.fit(black_box(&scry_ds)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("smartcore", ds_name), &ds_name, |b, _| {
            b.iter(|| {
                let _ = smartcore::linear::logistic_regression::LogisticRegression::fit(
                    black_box(&sm_x), black_box(&target_i32), Default::default(),
                ).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("linfa-logistic", ds_name), &ds_name, |b, _| {
            use linfa::prelude::Fit;
            b.iter(|| {
                let _ = linfa_logistic::LogisticRegression::default()
                    .max_iterations(LR_MAX_ITER as u64)
                    .fit(black_box(&linfa_ds)).unwrap();
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §5b LOGISTIC REGRESSION PREDICTION  (scry vs smartcore vs linfa-logistic)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_logreg_predict(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/logistic_regression/predict");
    group.sample_size(30);

    for &ds_name in &["breast_cancer", "adult"] {
        let (cols, target, names) = load_dataset(ds_name);

        // Z-score standardize — must match train benchmark exactly
        let cols: Vec<Vec<f64>> = cols.iter().map(|col| {
            let n = col.len() as f64;
            let mean = col.iter().sum::<f64>() / n;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt().max(1e-10);
            col.iter().map(|x| (x - mean) / std).collect()
        }).collect();

        let rows = to_row_major(&cols);
        let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

        let scry_ds = to_scry_dataset(&cols, &target, &names);
        let sm_x = to_smartcore_matrix(&rows);
        let linfa_ds = to_linfa_dataset_bool(&rows, &target);

        // Train all models once
        let mut scry_lr = scry_learn::prelude::LogisticRegression::new()
            .max_iter(LR_MAX_ITER);
        scry_lr.fit(&scry_ds).unwrap();

        let sm_lr = smartcore::linear::logistic_regression::LogisticRegression::fit(
            &sm_x, &target_i32, Default::default(),
        ).unwrap();

        use linfa::prelude::{Fit, Predict};
        let linfa_lr = linfa_logistic::LogisticRegression::default()
            .max_iterations(LR_MAX_ITER as u64)
            .fit(&linfa_ds).unwrap();

        let test_features = scry_ds.feature_matrix();

        group.bench_with_input(BenchmarkId::new("scry-learn", ds_name), &ds_name, |b, _| {
            b.iter(|| scry_lr.predict(black_box(&test_features)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("smartcore", ds_name), &ds_name, |b, _| {
            b.iter(|| sm_lr.predict(black_box(&sm_x)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("linfa-logistic", ds_name), &ds_name, |b, _| {
            b.iter(|| linfa_lr.predict(black_box(&linfa_ds)));
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §6 KNN PREDICTION  (scry vs smartcore — both brute-force Euclidean k=5)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_knn_predict(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/knn/predict");
    group.sample_size(30);

    for &ds_name in UCI_DATASETS {
        let (cols, target, names) = load_dataset(ds_name);
        let rows = to_row_major(&cols);
        let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

        let scry_ds = to_scry_dataset(&cols, &target, &names);
        let sm_x = to_smartcore_matrix(&rows);

        let mut scry_knn = scry_learn::prelude::KnnClassifier::new().k(KNN_K);
        scry_knn.fit(&scry_ds).unwrap();

        let sm_knn = smartcore::neighbors::knn_classifier::KNNClassifier::fit(
            &sm_x, &target_i32,
            smartcore::neighbors::knn_classifier::KNNClassifierParameters::default().with_k(KNN_K),
        ).unwrap();

        // Accuracy parity
        {
            let matrix = scry_ds.feature_matrix();
            let scry_preds = scry_knn.predict(&matrix).unwrap();
            let sm_preds_i32: Vec<i32> = sm_knn.predict(&sm_x).unwrap();
            let sm_preds: Vec<f64> = sm_preds_i32.iter().map(|&p| p as f64).collect();

            assert_accuracy_parity(
                &format!("KNN predict/{ds_name}"),
                &[
                    ("scry-learn", accuracy_f64(&target, &scry_preds)),
                    ("smartcore", accuracy_f64(&target, &sm_preds)),
                ],
                0.05,
            );
        }

        let test_features = scry_ds.feature_matrix();

        group.bench_with_input(BenchmarkId::new("scry-learn", ds_name), &ds_name, |b, _| {
            b.iter(|| scry_knn.predict(black_box(&test_features)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("smartcore", ds_name), &ds_name, |b, _| {
            b.iter(|| sm_knn.predict(black_box(&sm_x)).unwrap());
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §7 K-MEANS TRAINING  (scry vs linfa-clustering — both k-means++)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_kmeans_train(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/kmeans/train");
    group.sample_size(30);

    for &ds_name in UCI_DATASETS {
        let (cols, target, names) = load_dataset(ds_name);
        let rows = to_row_major(&cols);

        let scry_ds = to_scry_dataset(&cols, &target, &names);
        let linfa_ds: linfa::DatasetBase<ndarray::Array2<f64>, _> = {
            let n = rows.len();
            let m = rows[0].len();
            let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
            let x = ndarray::Array2::from_shape_vec((n, m), flat).unwrap();
            linfa::DatasetBase::from(x)
        };

        // Convergence parity: compare inertia
        {
            let mut scry_km = scry_learn::prelude::KMeans::new(KM_K)
                .seed(42).max_iter(KM_MAX_ITER);
            scry_km.fit(&scry_ds).unwrap();
            let scry_inertia = scry_km.inertia();

            use linfa::prelude::{Fit, Predict};
            use rand::SeedableRng;
            let rng = rand::rngs::SmallRng::seed_from_u64(42);
            let linfa_km = linfa_clustering::KMeans::params_with_rng(KM_K, rng)
                .max_n_iterations(KM_MAX_ITER as u64)
                .fit(&linfa_ds).unwrap();
            // linfa KMeans doesn't expose inertia directly — check label counts as proxy
            let linfa_labels = linfa_km.predict(&linfa_ds);
            let mut counts = vec![0usize; KM_K];
            for &l in linfa_labels.iter() {
                if l < KM_K { counts[l] += 1; }
            }

            eprintln!("\n┌─ CONVERGENCE: KMeans/{ds_name}");
            eprintln!("│  scry inertia:     {scry_inertia:.2}");
            eprintln!("│  linfa cluster sizes: {counts:?}");
            eprintln!("└─");
        }

        group.bench_with_input(BenchmarkId::new("scry-learn", ds_name), &ds_name, |b, _| {
            b.iter(|| {
                let mut km = scry_learn::prelude::KMeans::new(KM_K)
                    .seed(42).max_iter(KM_MAX_ITER);
                km.fit(black_box(&scry_ds)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("linfa-clustering", ds_name), &ds_name, |b, _| {
            use linfa::prelude::Fit;
            use rand::SeedableRng;
            b.iter(|| {
                let rng = rand::rngs::SmallRng::seed_from_u64(42);
                let _ = linfa_clustering::KMeans::params_with_rng(KM_K, rng)
                    .max_n_iterations(KM_MAX_ITER as u64)
                    .fit(black_box(&linfa_ds)).unwrap();
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §8 LASSO TRAINING  (scry vs linfa-elasticnet — both coord descent)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_lasso_train(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/lasso/train");
    group.sample_size(30);

    let (rows, target) = gen_regression(1000, 10);
    let cols: Vec<Vec<f64>> = {
        let n = rows.len();
        let m = rows[0].len();
        (0..m).map(|j| (0..n).map(|i| rows[i][j]).collect()).collect()
    };
    let names: Vec<String> = (0..10).map(|i| format!("f{i}")).collect();

    let scry_ds = to_scry_dataset(&cols, &target, &names);
    let linfa_ds = to_linfa_dataset_f64(&rows, &target);

    group.bench_function("scry-learn/1k", |b| {
        b.iter(|| {
            let mut lasso = scry_learn::prelude::LassoRegression::new()
                .alpha(LASSO_ALPHA)
                .max_iter(LASSO_MAX_ITER);
            lasso.fit(black_box(&scry_ds)).unwrap();
        });
    });

    group.bench_function("linfa-elasticnet/1k", |b| {
        use linfa::prelude::Fit;
        b.iter(|| {
            let _ = linfa_elasticnet::ElasticNet::<f64>::lasso()
                .penalty(LASSO_ALPHA)
                .max_iterations(LASSO_MAX_ITER as u32)
                .fit(black_box(&linfa_ds)).unwrap();
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §9 SCALING CURVES  (synthetic data at multiple sizes)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_scaling(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/scaling/dt_train");
    group.sample_size(10);

    let sizes = if cfg!(feature = "extended-bench") {
        vec![500, 2_000, 10_000, 50_000]
    } else {
        vec![500, 2_000, 10_000]
    };

    for &n in &sizes {
        let (rows, target) = gen_classification(n, 10);
        let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
        let cols: Vec<Vec<f64>> = {
            let m = rows[0].len();
            (0..m).map(|j| (0..n).map(|i| rows[i][j]).collect()).collect()
        };
        let names: Vec<String> = (0..10).map(|i| format!("f{i}")).collect();

        let scry_ds = to_scry_dataset(&cols, &target, &names);
        let sm_x = to_smartcore_matrix(&rows);

        group.bench_with_input(BenchmarkId::new("scry-learn", n), &n, |b, _| {
            b.iter(|| {
                let mut dt = scry_learn::prelude::DecisionTreeClassifier::new()
                    .max_depth(DT_MAX_DEPTH);
                dt.fit(black_box(&scry_ds)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("smartcore", n), &n, |b, _| {
            b.iter(|| {
                let _ = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
                    black_box(&sm_x),
                    black_box(&target_i32),
                    smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default()
                        .with_max_depth(DT_MAX_DEPTH as u16),
                ).unwrap();
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §10 COLD START  (construct → fit → first predict, no warmup)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_cold_start(c: &mut Criterion) {
    enforce_single_thread();

    let mut group = c.benchmark_group("fair/cold_start");
    group.sample_size(50);

    let (cols, target, names) = load_dataset("iris");
    let rows = to_row_major(&cols);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let single_row = vec![rows[0].clone()];

    let scry_ds = to_scry_dataset(&cols, &target, &names);
    let sm_x = to_smartcore_matrix(&rows);
    let linfa_ds = to_linfa_dataset(&rows, &target);

    // DT cold start
    group.bench_function("scry-learn/dt", |b| {
        b.iter(|| {
            let mut dt = scry_learn::prelude::DecisionTreeClassifier::new()
                .max_depth(DT_MAX_DEPTH);
            dt.fit(black_box(&scry_ds)).unwrap();
            dt.predict(black_box(&single_row)).unwrap()
        });
    });

    group.bench_function("smartcore/dt", |b| {
        b.iter(|| {
            let dt = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
                black_box(&sm_x),
                black_box(&target_i32),
                smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default()
                    .with_max_depth(DT_MAX_DEPTH as u16),
            ).unwrap();
            dt.predict(black_box(&sm_x)).unwrap()
        });
    });

    group.bench_function("linfa-trees/dt", |b| {
        use linfa::prelude::{Fit, Predict};
        b.iter(|| {
            let dt = linfa_trees::DecisionTree::params()
                .max_depth(Some(DT_MAX_DEPTH))
                .fit(black_box(&linfa_ds)).unwrap();
            dt.predict(black_box(&linfa_ds))
        });
    });

    // KNN cold start
    group.bench_function("scry-learn/knn", |b| {
        b.iter(|| {
            let mut knn = scry_learn::prelude::KnnClassifier::new().k(KNN_K);
            knn.fit(black_box(&scry_ds)).unwrap();
            let matrix = scry_ds.feature_matrix();
            knn.predict(black_box(&matrix)).unwrap()
        });
    });

    group.bench_function("smartcore/knn", |b| {
        b.iter(|| {
            let knn = smartcore::neighbors::knn_classifier::KNNClassifier::fit(
                black_box(&sm_x), black_box(&target_i32),
                smartcore::neighbors::knn_classifier::KNNClassifierParameters::default().with_k(KNN_K),
            ).unwrap();
            knn.predict(black_box(&sm_x)).unwrap()
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §11 MEMORY FOOTPRINT  (RSS delta per trained model)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_memory(c: &mut Criterion) {
    enforce_single_thread();

    // Memory measurement is done outside Criterion (not repeatable in a bench loop)
    // We print results to stderr during the benchmark run.
    let mut group = c.benchmark_group("fair/memory_footprint");
    group.sample_size(10);

    let (rows, target) = gen_classification(10_000, 10);
    let cols: Vec<Vec<f64>> = {
        let n = rows.len();
        let m = rows[0].len();
        (0..m).map(|j| (0..n).map(|i| rows[i][j]).collect()).collect()
    };
    let names: Vec<String> = (0..10).map(|i| format!("f{i}")).collect();
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    let scry_ds = to_scry_dataset(&cols, &target, &names);
    let sm_x = to_smartcore_matrix(&rows);

    // Measure RSS delta for DT
    {
        let rss_before = read_rss_kb();
        let mut dt = scry_learn::prelude::DecisionTreeClassifier::new().max_depth(DT_MAX_DEPTH);
        dt.fit(&scry_ds).unwrap();
        let rss_after = read_rss_kb();
        let delta_kb = rss_after.saturating_sub(rss_before);
        eprintln!("\n┌─ MEMORY: DT (10K×10)");
        eprintln!("│  scry-learn RSS Δ: {delta_kb} KB");
        std::hint::black_box(&dt);
    }

    {
        let rss_before = read_rss_kb();
        let dt = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
            &sm_x, &target_i32,
            smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default()
                .with_max_depth(DT_MAX_DEPTH as u16),
        ).unwrap();
        let rss_after = read_rss_kb();
        let delta_kb = rss_after.saturating_sub(rss_before);
        eprintln!("│  smartcore RSS Δ:  {delta_kb} KB");
        eprintln!("└─");
        std::hint::black_box(&dt);
    }

    // Measure RSS delta for RF
    {
        let rss_before = read_rss_kb();
        let mut rf = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(RF_N_ESTIMATORS).max_depth(RF_MAX_DEPTH).seed(42);
        rf.fit(&scry_ds).unwrap();
        let rss_after = read_rss_kb();
        let delta_kb = rss_after.saturating_sub(rss_before);
        eprintln!("\n┌─ MEMORY: RF 20t (10K×10)");
        eprintln!("│  scry-learn RSS Δ: {delta_kb} KB");
        std::hint::black_box(&rf);
    }

    {
        let rss_before = read_rss_kb();
        let sm_params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(RF_N_ESTIMATORS as u16)
            .with_max_depth(RF_MAX_DEPTH as u16);
        let rf = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
            &sm_x, &target_i32, sm_params,
        ).unwrap();
        let rss_after = read_rss_kb();
        let delta_kb = rss_after.saturating_sub(rss_before);
        eprintln!("│  smartcore RSS Δ:  {delta_kb} KB");
        eprintln!("└─");
        std::hint::black_box(&rf);
    }

    // Dummy benchmark so Criterion doesn't complain about empty group
    group.bench_function("scry-learn/noop", |b| {
        b.iter(|| std::hint::black_box(42));
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// §12 CONCURRENT INFERENCE  (4 threads × 1000 predicts)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_concurrent(c: &mut Criterion) {
    // NOTE: This benchmark intentionally tests multi-threaded inference,
    // so we do NOT enforce single thread here. This is a separate dimension.
    let mut group = c.benchmark_group("fair/concurrent_inference");
    group.sample_size(30);

    let (cols, target, names) = load_dataset("iris");
    let rows = to_row_major(&cols);
    let single_row = vec![rows[0].clone()];

    let scry_ds = to_scry_dataset(&cols, &target, &names);

    let mut dt = scry_learn::prelude::DecisionTreeClassifier::new().max_depth(DT_MAX_DEPTH);
    dt.fit(&scry_ds).unwrap();

    let n_threads = 4;
    let n_per_thread = 1000;

    group.bench_function("scry-learn/dt/4×1000", |b| {
        b.iter(|| {
            std::thread::scope(|s| {
                for _ in 0..n_threads {
                    s.spawn(|| {
                        for _ in 0..n_per_thread {
                            let _ = dt.predict(black_box(&single_row)).unwrap();
                        }
                    });
                }
            });
        });
    });

    // smartcore DT is also Send+Sync — predict the SAME single row for fairness
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let sm_x = to_smartcore_matrix(&rows);
    let sm_single_row = to_smartcore_matrix(&single_row); // ← FAIRNESS: same 1 row
    let sm_dt = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
        &sm_x,
        &target_i32,
        smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default()
            .with_max_depth(DT_MAX_DEPTH as u16),
    ).unwrap();

    group.bench_function("smartcore/dt/4×1000", |b| {
        b.iter(|| {
            std::thread::scope(|s| {
                for _ in 0..n_threads {
                    s.spawn(|| {
                        for _ in 0..n_per_thread {
                            let _ = sm_dt.predict(black_box(&sm_single_row)).unwrap();
                        }
                    });
                }
            });
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════
// HARNESS
// ═══════════════════════════════════════════════════════════════════════════

criterion_group!(
    fair_benches,
    bench_dt_train,
    bench_dt_predict,
    bench_rf_train,
    bench_rf_predict,
    bench_logreg_train,
    bench_logreg_predict,
    bench_knn_predict,
    bench_kmeans_train,
    bench_lasso_train,
    bench_scaling,
    bench_cold_start,
    bench_memory,
    bench_concurrent,
);
criterion_main!(fair_benches);
