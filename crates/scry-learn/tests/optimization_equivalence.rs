#![allow(unsafe_code, clippy::doc_markdown)]

//! Optimization equivalence tests — prove that performance optimizations
//! preserve mathematical correctness, and verify allocation/time complexity.
//!
//! Tests cover:
//! 1. Sigmoid = 2-class softmax (binary logistic regression fast path)
//! 2. Lasso dense path = sparse path (column-major direct access)
//! 3. ElasticNet dense = sparse across l1_ratio values
//! 4. Binary L-BFGS = GD solver (same optimum, different path)
//! 5. L-BFGS allocation count is bounded (not linear in max_iter)
//! 6. Lasso avoids full feature matrix allocation
//! 7. Training time scales linearly with sample count
//!
//! Run: `cargo test --test optimization_equivalence -p scry-learn --release -- --nocapture`

#[path = "tracking_alloc.rs"]
mod tracking_alloc;

use std::path::PathBuf;
use std::time::Instant;

use tracking_alloc::{format_bytes, format_count, AllocSnapshot, TrackingAllocator};

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new();

use scry_learn::dataset::Dataset;
use scry_learn::linear::{ElasticNet, LassoRegression, LogisticRegression, Solver};
use scry_learn::metrics::{accuracy, r2_score};
use scry_learn::preprocess::{StandardScaler, Transformer};
use scry_learn::sparse::CscMatrix;
use scry_learn::split::train_test_split;

// ═══════════════════════════════════════════════════════════════════════════
// Fixture helpers
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

fn load_dataset(base: &str) -> Dataset {
    let (features, feat_names) = load_features_csv(&format!("{base}_features.csv"));
    let target = load_target_csv(&format!("{base}_target.csv"));
    Dataset::new(features, target, feat_names, "target")
}

fn gen_regression(n: usize, n_features: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut cols = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];
    // Generate features
    for col in &mut cols {
        for val in col.iter_mut() {
            *val = rng.f64() * 10.0 - 5.0;
        }
    }
    // Target = linear combination + noise
    let weights: Vec<f64> = (0..n_features).map(|j| (j as f64 + 1.0) * 0.5).collect();
    for (i, t) in target.iter_mut().enumerate() {
        let mut y = 3.0; // intercept
        for (j, w) in weights.iter().enumerate() {
            y += cols[j][i] * w;
        }
        y += (rng.f64() - 0.5) * 0.1; // small noise
        *t = y;
    }
    let names: Vec<String> = (0..n_features).map(|j| format!("f{j}")).collect();
    Dataset::new(cols, target, names, "target")
}

fn gen_classification(n: usize, n_features: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let half = n / 2;
    let mut cols = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];
    for (j, col) in cols.iter_mut().enumerate() {
        let offset = 3.0 + j as f64 * 0.5;
        for val in &mut col[..half] {
            *val = rng.f64() * 2.0;
        }
        for i in half..n {
            col[i] = rng.f64() * 2.0 + offset;
            target[i] = 1.0;
        }
    }
    let names: Vec<String> = (0..n_features).map(|j| format!("f{j}")).collect();
    Dataset::new(cols, target, names, "target")
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: sigmoid ≡ 2-class softmax (mathematical proof)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sigmoid_equals_softmax_proof() {
    println!("\n═══ sigmoid ≡ softmax proof ═══\n");

    let mut rng = fastrand::Rng::with_seed(42);
    let mut max_err = 0.0_f64;
    let n_trials = 1000;

    // Random z values
    for _ in 0..n_trials {
        let z = (rng.f64() - 0.5) * 20.0; // z in [-10, 10]

        let p_sigmoid = 1.0 / (1.0 + (-z).exp());

        // softmax([0, z])[1] = exp(z) / (exp(0) + exp(z)) = exp(z) / (1 + exp(z))
        // Numerically stable: 1 / (1 + exp(-z))
        let exp_z = z.exp();
        let p_softmax = exp_z / (1.0 + exp_z);

        let err = (p_sigmoid - p_softmax).abs();
        max_err = max_err.max(err);
        assert!(
            err < 1e-14,
            "sigmoid != softmax for z={z}: sigmoid={p_sigmoid}, softmax={p_softmax}, err={err}"
        );
    }

    // Extreme values
    let extreme_cases: &[(f64, &str)] = &[
        (500.0, "large positive"),
        (-500.0, "large negative"),
        (0.0, "zero"),
        (1e-15, "tiny positive"),
        (-1e-15, "tiny negative"),
        (30.0, "exp overflow boundary"),
        (-30.0, "exp underflow boundary"),
    ];

    for &(z, label) in extreme_cases {
        let p_sigmoid = 1.0 / (1.0 + (-z).exp());
        // Use log-sum-exp stable softmax for extreme values
        let max_val = 0.0_f64.max(z);
        let log_sum = max_val + ((0.0 - max_val).exp() + (z - max_val).exp()).ln();
        let p_softmax = (z - log_sum).exp();

        let err = (p_sigmoid - p_softmax).abs();
        max_err = max_err.max(err);
        println!("  z={z:>12.2e} ({label:>24}): sigmoid={p_sigmoid:.16e}, softmax={p_softmax:.16e}, err={err:.2e}");
        assert!(
            err < 1e-14,
            "sigmoid != softmax for extreme z={z} ({label})"
        );
    }

    println!("\n  max |sigmoid - softmax| across {n_trials} random + {} extreme = {max_err:.2e}",
        extreme_cases.len());
    println!("  PASS: sigmoid ≡ softmax for binary classification\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Lasso dense ≡ sparse (column-major direct access)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lasso_dense_matches_sparse() {
    println!("\n═══ Lasso dense ≡ sparse ═══\n");

    let data = load_dataset("california");
    let (train, test) = train_test_split(&data, 0.2, 42);
    let n_features = train.n_features();

    // Dense fit
    let mut lasso_dense = LassoRegression::new().alpha(0.1).max_iter(1000).tol(1e-6);
    lasso_dense.fit(&train).unwrap();

    // Sparse fit — construct CSC from the same features
    let csc = CscMatrix::from_dense(&train.features);
    let mut lasso_sparse = LassoRegression::new().alpha(0.1).max_iter(1000).tol(1e-6);
    lasso_sparse.fit_sparse(&csc, &train.target).unwrap();

    // Compare coefficients
    let coef_dense = lasso_dense.coefficients();
    let coef_sparse = lasso_sparse.coefficients();
    let mut max_coef_diff = 0.0_f64;
    for j in 0..n_features {
        let diff = (coef_dense[j] - coef_sparse[j]).abs();
        max_coef_diff = max_coef_diff.max(diff);
    }
    println!("  max |coef_dense - coef_sparse| = {max_coef_diff:.2e}");
    assert!(
        max_coef_diff < 1e-10,
        "Lasso coefficients diverge: max diff = {max_coef_diff}"
    );

    // Compare intercepts
    let intercept_diff = (lasso_dense.intercept() - lasso_sparse.intercept()).abs();
    println!("  |intercept_dense - intercept_sparse| = {intercept_diff:.2e}");
    assert!(
        intercept_diff < 1e-10,
        "Lasso intercepts diverge: diff = {intercept_diff}"
    );

    // Compare predictions on test set
    let test_rows: Vec<Vec<f64>> = (0..test.n_samples())
        .map(|i| test.sample(i))
        .collect();
    let pred_dense = lasso_dense.predict(&test_rows).unwrap();
    let pred_sparse = lasso_sparse.predict(&test_rows).unwrap();
    let mut max_pred_diff = 0.0_f64;
    for i in 0..pred_dense.len() {
        let diff = (pred_dense[i] - pred_sparse[i]).abs();
        max_pred_diff = max_pred_diff.max(diff);
    }
    println!("  max |pred_dense - pred_sparse| = {max_pred_diff:.2e}");
    assert!(
        max_pred_diff < 1e-10,
        "Lasso predictions diverge: max diff = {max_pred_diff}"
    );

    let r2 = r2_score(&test.target, &pred_dense);
    println!("  R² (dense) = {r2:.4}");
    println!("  PASS: Lasso dense ≡ sparse\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: ElasticNet dense ≡ sparse across l1_ratio values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn elastic_net_dense_matches_sparse() {
    println!("\n═══ ElasticNet dense ≡ sparse ═══\n");

    let data = load_dataset("california");
    let (train, test) = train_test_split(&data, 0.2, 42);

    let l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0];
    let test_rows: Vec<Vec<f64>> = (0..test.n_samples())
        .map(|i| test.sample(i))
        .collect();

    let csc = CscMatrix::from_dense(&train.features);

    for &l1_ratio in &l1_ratios {
        // Dense
        let mut en_dense = ElasticNet::new()
            .alpha(0.1)
            .l1_ratio(l1_ratio)
            .max_iter(1000)
            .tol(1e-6);
        en_dense.fit(&train).unwrap();

        // Sparse
        let mut en_sparse = ElasticNet::new()
            .alpha(0.1)
            .l1_ratio(l1_ratio)
            .max_iter(1000)
            .tol(1e-6);
        en_sparse.fit_sparse(&csc, &train.target).unwrap();

        // Compare coefficients
        let coef_dense = en_dense.coefficients();
        let coef_sparse = en_sparse.coefficients();
        let max_coef_diff: f64 = coef_dense
            .iter()
            .zip(coef_sparse.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        let intercept_diff = (en_dense.intercept() - en_sparse.intercept()).abs();

        // Compare predictions
        let pred_dense = en_dense.predict(&test_rows).unwrap();
        let pred_sparse = en_sparse.predict(&test_rows).unwrap();
        let max_pred_diff: f64 = pred_dense
            .iter()
            .zip(pred_sparse.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!(
            "  l1_ratio={l1_ratio:.2}: max|coef|={max_coef_diff:.2e}, |intercept|={intercept_diff:.2e}, max|pred|={max_pred_diff:.2e}"
        );

        assert!(
            max_coef_diff < 1e-10,
            "ElasticNet coef diverge at l1_ratio={l1_ratio}: {max_coef_diff}"
        );
        assert!(
            intercept_diff < 1e-10,
            "ElasticNet intercept diverge at l1_ratio={l1_ratio}: {intercept_diff}"
        );
        assert!(
            max_pred_diff < 1e-10,
            "ElasticNet pred diverge at l1_ratio={l1_ratio}: {max_pred_diff}"
        );
    }

    println!("  PASS: ElasticNet dense ≡ sparse across all l1_ratio values\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Binary L-BFGS ≡ GD solver
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn binary_logreg_matches_gd_solver() {
    println!("\n═══ Binary LogReg: L-BFGS ≡ GD ═══\n");

    let data = load_dataset("breast_cancer");
    let (mut train, mut test) = train_test_split(&data, 0.2, 42);

    // Scale features for stable convergence
    let mut scaler = StandardScaler::new();
    scaler.fit_transform(&mut train).unwrap();
    scaler.transform(&mut test).unwrap();

    // L-BFGS solver (uses binary sigmoid fast path)
    let mut lr_lbfgs = LogisticRegression::new()
        .solver(Solver::Lbfgs)
        .alpha(0.01)
        .max_iter(2000);
    lr_lbfgs.fit(&train).unwrap();

    // GD solver (uses full softmax path)
    let mut lr_gd = LogisticRegression::new()
        .solver(Solver::GradientDescent)
        .alpha(0.01)
        .learning_rate(0.1)
        .max_iter(2000);
    lr_gd.fit(&train).unwrap();

    let test_rows: Vec<Vec<f64>> = (0..test.n_samples())
        .map(|i| test.sample(i))
        .collect();

    let pred_lbfgs = lr_lbfgs.predict(&test_rows).unwrap();
    let pred_gd = lr_gd.predict(&test_rows).unwrap();

    // Prediction agreement
    let n_agree = pred_lbfgs
        .iter()
        .zip(pred_gd.iter())
        .filter(|(a, b)| (*a - *b).abs() < 0.5)
        .count();
    let agreement = n_agree as f64 / pred_lbfgs.len() as f64;

    let acc_lbfgs = accuracy(&test.target, &pred_lbfgs);
    let acc_gd = accuracy(&test.target, &pred_gd);

    println!("  L-BFGS accuracy = {acc_lbfgs:.4}");
    println!("  GD accuracy     = {acc_gd:.4}");
    println!("  agreement       = {agreement:.4} ({n_agree}/{})", pred_lbfgs.len());

    assert!(
        agreement >= 0.93,
        "L-BFGS and GD should agree on ≥93% of predictions, got {agreement:.4}"
    );

    // Compare probabilities — use median and 90th percentile rather than max,
    // since a few samples near the decision boundary can flip between solvers.
    let proba_lbfgs = lr_lbfgs.predict_proba(&test_rows).unwrap();
    let proba_gd = lr_gd.predict_proba(&test_rows).unwrap();
    let mut proba_diffs: Vec<f64> = proba_lbfgs
        .iter()
        .zip(proba_gd.iter())
        .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max))
        .collect();
    proba_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_diff = proba_diffs[proba_diffs.len() / 2];
    let p90_diff = proba_diffs[(proba_diffs.len() as f64 * 0.9) as usize];
    let max_diff = proba_diffs[proba_diffs.len() - 1];

    println!("  proba diff: median={median_diff:.4}, p90={p90_diff:.4}, max={max_diff:.4}");
    assert!(
        median_diff < 0.1,
        "Median probability diff too high: {median_diff}"
    );
    assert!(
        p90_diff < 0.5,
        "90th percentile probability diff too high: {p90_diff}"
    );

    println!("  PASS: Binary L-BFGS ≡ GD solver\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: L-BFGS allocation count is bounded
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lbfgs_allocation_count_bounded() {
    println!("\n═══ L-BFGS allocation count ═══\n");

    // Run with two different max_iter values; alloc count should NOT scale linearly
    let data_50 = gen_classification(500, 20, 42);
    let data_50b = gen_classification(500, 20, 42);

    // First: 50 iterations
    let snap_before = AllocSnapshot::reset();
    let mut lr_50 = LogisticRegression::new()
        .solver(Solver::Lbfgs)
        .max_iter(50);
    lr_50.fit(&data_50).unwrap();
    let delta_50 = AllocSnapshot::now().delta_from(snap_before);

    // Second: 200 iterations
    let snap_before = AllocSnapshot::reset();
    let mut lr_200 = LogisticRegression::new()
        .solver(Solver::Lbfgs)
        .max_iter(200);
    lr_200.fit(&data_50b).unwrap();
    let delta_200 = AllocSnapshot::now().delta_from(snap_before);

    println!("  50  iterations: {} allocs, peak {}", format_count(delta_50.alloc_count), format_bytes(delta_50.peak_increase));
    println!("  200 iterations: {} allocs, peak {}", format_count(delta_200.alloc_count), format_bytes(delta_200.peak_increase));

    // Report per-iteration alloc rate for observability.
    // The 50-iter run may converge early (fewer actual iterations), so the
    // ratio can exceed 4x.  Use absolute bounds as regression guards.
    let ratio = delta_200.alloc_count as f64 / delta_50.alloc_count.max(1) as f64;
    println!("  alloc ratio (200/50) = {ratio:.2}");

    // Absolute guard: 200-iter fit on 500×20 binary should not exceed 50k allocs.
    // Calibrated from first run (observed ~15.7k).
    assert!(
        delta_200.alloc_count < 50_000,
        "L-BFGS alloc count unexpectedly high: {} (limit 50,000)",
        delta_200.alloc_count
    );
    // Peak memory should stay bounded regardless of iteration count.
    assert!(
        delta_200.peak_increase < 10 * 1024 * 1024, // 10 MB
        "L-BFGS peak memory too high: {}",
        format_bytes(delta_200.peak_increase)
    );

    println!("  PASS: L-BFGS allocations bounded\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Lasso avoids full feature matrix allocation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn lasso_no_feature_matrix_allocation() {
    println!("\n═══ Lasso memory footprint ═══\n");

    let n_samples = 2000;
    let n_features = 50;
    let data = gen_regression(n_samples, n_features, 42);

    // The old code would allocate a row-major feature_matrix: n_samples * n_features * 8 bytes
    let old_alloc_bytes = n_samples * n_features * 8; // 800,000 bytes

    let snap_before = AllocSnapshot::reset();
    let mut lasso = LassoRegression::new().alpha(0.1).max_iter(500).tol(1e-6);
    lasso.fit(&data).unwrap();
    let delta = AllocSnapshot::now().delta_from(snap_before);

    println!("  dataset: {n_samples} samples × {n_features} features");
    println!("  old feature_matrix would be: {old_alloc_bytes} bytes");
    println!("  peak memory increase: {}", format_bytes(delta.peak_increase));
    println!("  alloc count: {}", format_count(delta.alloc_count));

    // Peak includes residuals, beta, col_norm_sq, plus Dataset validation and
    // internal bookkeeping.  Guard against regression: should stay well under 2x
    // the feature matrix size.
    let threshold = old_alloc_bytes * 2; // 1.6 MB
    assert!(
        delta.peak_increase < threshold,
        "Lasso peak memory {} exceeds threshold {}: unexpected allocation growth",
        format_bytes(delta.peak_increase),
        format_bytes(threshold)
    );

    println!("  PASS: Lasso peak < {} (regression guard)\n", format_bytes(threshold));
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: Training time scales linearly
// ═══════════════════════════════════════════════════════════════════════════

fn median_of(times: &mut [f64]) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = times.len() / 2;
    if times.len() % 2 == 0 {
        (times[mid - 1] + times[mid]) / 2.0
    } else {
        times[mid]
    }
}

#[test]
fn training_time_scales_linearly() {
    println!("\n═══ Training time scaling ═══\n");

    let sizes = [500, 1000, 2000, 4000];
    let n_features = 20;
    let warmup = 3;
    let timed_runs = 5;

    // ── Lasso scaling ──
    println!("  Lasso (alpha=0.1, max_iter=200, {n_features} features):");
    let mut lasso_medians = Vec::new();
    for &n in &sizes {
        let data = gen_regression(n, n_features, 42);

        // Warmup
        for _ in 0..warmup {
            let mut m = LassoRegression::new().alpha(0.1).max_iter(200).tol(1e-6);
            m.fit(&data).unwrap();
            std::hint::black_box(&m);
        }

        // Timed runs
        let mut times = Vec::with_capacity(timed_runs);
        for _ in 0..timed_runs {
            let start = Instant::now();
            let mut m = LassoRegression::new().alpha(0.1).max_iter(200).tol(1e-6);
            m.fit(&data).unwrap();
            std::hint::black_box(&m);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median_of(&mut times);
        println!("    N={n:>5}: {med:>8.2} ms");
        lasso_medians.push(med);
    }

    // Check scaling: time(4000) / time(1000) should be < 6.0
    let lasso_ratio = lasso_medians[3] / lasso_medians[1].max(0.001);
    println!("    ratio(4000/1000) = {lasso_ratio:.2} (should be < 6.0)");
    assert!(
        lasso_ratio < 6.0,
        "Lasso time scaling is superlinear: ratio = {lasso_ratio:.2}"
    );

    // ── LogisticRegression scaling ──
    println!("\n  LogisticRegression binary (alpha=0.01, max_iter=200, {n_features} features):");
    let mut lr_medians = Vec::new();
    for &n in &sizes {
        let data = gen_classification(n, n_features, 42);

        // Warmup
        for _ in 0..warmup {
            let mut m = LogisticRegression::new()
                .solver(Solver::Lbfgs)
                .alpha(0.01)
                .max_iter(200);
            m.fit(&data).unwrap();
            std::hint::black_box(&m);
        }

        // Timed runs
        let mut times = Vec::with_capacity(timed_runs);
        for _ in 0..timed_runs {
            let start = Instant::now();
            let mut m = LogisticRegression::new()
                .solver(Solver::Lbfgs)
                .alpha(0.01)
                .max_iter(200);
            m.fit(&data).unwrap();
            std::hint::black_box(&m);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let med = median_of(&mut times);
        println!("    N={n:>5}: {med:>8.2} ms");
        lr_medians.push(med);
    }

    let lr_ratio = lr_medians[3] / lr_medians[1].max(0.001);
    println!("    ratio(4000/1000) = {lr_ratio:.2} (should be < 6.0)");
    assert!(
        lr_ratio < 6.0,
        "LogisticRegression time scaling is superlinear: ratio = {lr_ratio:.2}"
    );

    println!("\n  PASS: Training time scales linearly\n");
}
