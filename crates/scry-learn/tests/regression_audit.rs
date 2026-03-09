#![allow(
    clippy::items_after_statements,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::default_trait_access
)]
//! Regression model cross-library accuracy audit.
//!
//! Compares R2/MSE for regression models across scry-learn, smartcore, and linfa
//! using proper 80/20 train/test splits. Outputs only measured numbers — no
//! feature comparison tables, no marketing language.
//!
//! Run: cargo test --test `regression_audit` -p scry-learn --release -- --nocapture

use std::time::Instant;

fn gen_regression(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n_features {
            let v = rng.f64() * 10.0;
            col_major[j][i] = v;
            sum += v * (j as f64 + 1.0);
        }
        target[i] = sum + rng.f64() * 0.5;
    }
    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_features).map(|j| col_major[j][i]).collect())
        .collect();
    (col_major, target, row_major)
}

fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (t - p).powi(2))
        .sum::<f64>()
        / y_true.len() as f64
}

fn r2(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|&t| (t - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();
    if ss_tot < 1e-15 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// FNV-1a hash of prediction vector for cross-machine reproducibility.
fn prediction_checksum(preds: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &p in preds {
        h ^= p.to_bits();
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Simple 80/20 split with deterministic shuffle (seed=42).
fn split_data(
    col_major: &[Vec<f64>],
    target: &[f64],
    row_major: &[Vec<f64>],
) -> (
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<Vec<f64>>,
) {
    let n = target.len();
    let mut indices: Vec<usize> = (0..n).collect();
    // Fisher-Yates with seed
    let mut rng = fastrand::Rng::with_seed(42);
    for i in (1..n).rev() {
        let j = rng.usize(0..=i);
        indices.swap(i, j);
    }
    let split = (n as f64 * 0.8) as usize;
    let train_idx = &indices[..split];
    let test_idx = &indices[split..];

    let n_features = col_major.len();
    let train_col: Vec<Vec<f64>> = (0..n_features)
        .map(|j| train_idx.iter().map(|&i| col_major[j][i]).collect())
        .collect();
    let train_target: Vec<f64> = train_idx.iter().map(|&i| target[i]).collect();
    let train_row: Vec<Vec<f64>> = train_idx.iter().map(|&i| row_major[i].clone()).collect();

    let test_col: Vec<Vec<f64>> = (0..n_features)
        .map(|j| test_idx.iter().map(|&i| col_major[j][i]).collect())
        .collect();
    let test_target: Vec<f64> = test_idx.iter().map(|&i| target[i]).collect();
    let test_row: Vec<Vec<f64>> = test_idx.iter().map(|&i| row_major[i].clone()).collect();

    (
        train_col,
        train_target,
        train_row,
        test_col,
        test_target,
        test_row,
    )
}

// ═══════════════════════════════════════════════════════════════════
// Linear Regression: scry vs smartcore vs linfa
// ═══════════════════════════════════════════════════════════════════

#[test]
fn regression_audit_linear() {
    let n = 2000;
    let n_features = 10;
    let (col_major, target, row_major) = gen_regression(n, n_features);
    let (train_col, train_target, train_row, _test_col, test_target, test_row) =
        split_data(&col_major, &target, &row_major);

    let n_iters = 50;

    // ── scry-learn ──
    let train_ds = scry_learn::dataset::Dataset::new(
        train_col,
        train_target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(),
        "y",
    );
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let mut lr = scry_learn::linear::LinearRegression::new();
        lr.fit(std::hint::black_box(&train_ds)).unwrap();
    }
    let scry_fit_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    let mut scry_lr = scry_learn::linear::LinearRegression::new();
    scry_lr.fit(&train_ds).unwrap();
    let scry_preds = scry_lr.predict(&test_row).unwrap();
    let scry_r2 = r2(&test_target, &scry_preds);
    let scry_mse = mse(&test_target, &scry_preds);

    // ── smartcore ──
    use smartcore::linalg::basic::matrix::DenseMatrix;
    let smart_train_x = DenseMatrix::from_2d_vec(&train_row).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = smartcore::linear::linear_regression::LinearRegression::fit(
            std::hint::black_box(&smart_train_x),
            std::hint::black_box(&train_target),
            Default::default(),
        )
        .unwrap();
    }
    let smart_fit_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    let smart_model = smartcore::linear::linear_regression::LinearRegression::fit(
        &smart_train_x,
        &train_target,
        Default::default(),
    )
    .unwrap();
    let smart_test_x = DenseMatrix::from_2d_vec(&test_row).unwrap();
    let smart_preds: Vec<f64> = smart_model.predict(&smart_test_x).unwrap();
    let smart_r2 = r2(&test_target, &smart_preds);
    let smart_mse = mse(&test_target, &smart_preds);

    // ── linfa ──
    use linfa::prelude::{Fit, Predict};
    let train_flat: Vec<f64> = train_row.iter().flat_map(|r| r.iter().copied()).collect();
    let train_x_nd =
        ndarray::Array2::from_shape_vec((train_row.len(), n_features), train_flat).unwrap();
    let train_y_nd = ndarray::Array1::from_vec(train_target);
    let linfa_train_ds = linfa::Dataset::new(train_x_nd, train_y_nd);

    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = linfa_linear::LinearRegression::default()
            .fit(std::hint::black_box(&linfa_train_ds))
            .unwrap();
    }
    let linfa_fit_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    let linfa_model = linfa_linear::LinearRegression::default()
        .fit(&linfa_train_ds)
        .unwrap();
    let test_flat: Vec<f64> = test_row.iter().flat_map(|r| r.iter().copied()).collect();
    let test_x_nd =
        ndarray::Array2::from_shape_vec((test_row.len(), n_features), test_flat).unwrap();
    let linfa_preds_nd = linfa_model.predict(&test_x_nd);
    let linfa_preds: Vec<f64> = linfa_preds_nd.to_vec();
    let linfa_r2 = r2(&test_target, &linfa_preds);
    let linfa_mse = mse(&test_target, &linfa_preds);

    println!("\n{}", "=".repeat(80));
    println!("REGRESSION AUDIT — LinearRegression ({n}x{n_features}, 80/20 split, seed=42)");
    println!(
        "  scry-learn   R2={:.6}  MSE={:.4}  fit={:.2}us  checksum=0x{:016x}",
        scry_r2,
        scry_mse,
        scry_fit_us,
        prediction_checksum(&scry_preds)
    );
    println!(
        "  smartcore    R2={:.6}  MSE={:.4}  fit={:.2}us  checksum=0x{:016x}",
        smart_r2,
        smart_mse,
        smart_fit_us,
        prediction_checksum(&smart_preds)
    );
    println!(
        "  linfa        R2={:.6}  MSE={:.4}  fit={:.2}us  checksum=0x{:016x}",
        linfa_r2,
        linfa_mse,
        linfa_fit_us,
        prediction_checksum(&linfa_preds)
    );

    // Regression threshold: linear regression on y = sum(j*xj) + noise should
    // achieve near-perfect R² on the test set.
    assert!(
        scry_r2 >= 0.99,
        "scry LinearRegression R² regression: {scry_r2:.6} < 0.99"
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// Decision Tree Regressor: scry vs smartcore
// ═══════════════════════════════════════════════════════════════════

#[test]
fn regression_audit_decision_tree() {
    let n = 2000;
    let n_features = 10;
    let (col_major, target, row_major) = gen_regression(n, n_features);
    let (train_col, train_target, train_row, _test_col, test_target, test_row) =
        split_data(&col_major, &target, &row_major);

    let n_iters = 20;

    // ── scry-learn ──
    let train_ds = scry_learn::dataset::Dataset::new(
        train_col,
        train_target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(),
        "y",
    );
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let mut dt = scry_learn::tree::DecisionTreeRegressor::new().max_depth(8);
        dt.fit(std::hint::black_box(&train_ds)).unwrap();
    }
    let scry_fit_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    let mut scry_dt = scry_learn::tree::DecisionTreeRegressor::new().max_depth(8);
    scry_dt.fit(&train_ds).unwrap();
    let scry_preds = scry_dt.predict(&test_row).unwrap();
    let scry_r2 = r2(&test_target, &scry_preds);
    let scry_mse = mse(&test_target, &scry_preds);

    // ── smartcore ──
    use smartcore::linalg::basic::matrix::DenseMatrix;
    let smart_train_x = DenseMatrix::from_2d_vec(&train_row).unwrap();
    let smart_params =
        smartcore::tree::decision_tree_regressor::DecisionTreeRegressorParameters::default()
            .with_max_depth(8);
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::fit(
            std::hint::black_box(&smart_train_x),
            std::hint::black_box(&train_target),
            smart_params.clone(),
        )
        .unwrap();
    }
    let smart_fit_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    let smart_model = smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::fit(
        &smart_train_x,
        &train_target,
        smart_params,
    )
    .unwrap();
    let smart_test_x = DenseMatrix::from_2d_vec(&test_row).unwrap();
    let smart_preds: Vec<f64> = smart_model.predict(&smart_test_x).unwrap();
    let smart_r2 = r2(&test_target, &smart_preds);
    let smart_mse = mse(&test_target, &smart_preds);

    println!("\n{}", "=".repeat(80));
    println!("REGRESSION AUDIT — DecisionTreeRegressor ({n}x{n_features}, depth=8, 80/20 split, seed=42)");
    println!(
        "  scry-learn   R2={:.6}  MSE={:.4}  fit={:.2}us  checksum=0x{:016x}",
        scry_r2,
        scry_mse,
        scry_fit_us,
        prediction_checksum(&scry_preds)
    );
    println!(
        "  smartcore    R2={:.6}  MSE={:.4}  fit={:.2}us  checksum=0x{:016x}",
        smart_r2,
        smart_mse,
        smart_fit_us,
        prediction_checksum(&smart_preds)
    );

    // Regression threshold: depth-8 tree on purely linear synthetic data is fundamentally
    // limited by axis-aligned splits (both scry and smartcore achieve ~0.59).
    // This guards against regressions, not absolute quality.
    assert!(
        scry_r2 >= 0.50,
        "scry DTRegressor R² regression: {scry_r2:.6} < 0.50"
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// Lasso: scry vs linfa-elasticnet
// ═══════════════════════════════════════════════════════════════════

#[test]
fn regression_audit_lasso() {
    let n = 2000;
    let n_features = 10;
    let (col_major, target, row_major) = gen_regression(n, n_features);
    let (train_col, train_target, train_row, _test_col, test_target, test_row) =
        split_data(&col_major, &target, &row_major);

    // ── scry-learn ──
    let train_ds = scry_learn::dataset::Dataset::new(
        train_col,
        train_target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(),
        "y",
    );
    let mut scry_lasso = scry_learn::linear::LassoRegression::new()
        .alpha(0.1)
        .max_iter(1000);
    scry_lasso.fit(&train_ds).unwrap();
    let scry_preds = scry_lasso.predict(&test_row).unwrap();
    let scry_r2 = r2(&test_target, &scry_preds);
    let scry_mse = mse(&test_target, &scry_preds);

    // ── linfa-elasticnet (l1_ratio=1.0 = Lasso) ──
    // linfa-elasticnet's coordinate descent assumes standardized features and uses
    // per-sample penalty scaling (like R's glmnet). scry-learn handles raw features
    // and uses total regularization (like sklearn). To compare fairly:
    //   1. Standardize features before passing to linfa
    //   2. Convert penalty: linfa_penalty = scry_alpha / n_train
    use linfa::prelude::{Fit, Predict};
    let n_train = train_row.len();

    // Compute per-feature mean and std from training data
    let mut feat_mean = vec![0.0f64; n_features];
    let mut feat_std = vec![0.0f64; n_features];
    for j in 0..n_features {
        let mean = train_row.iter().map(|r| r[j]).sum::<f64>() / n_train as f64;
        let var = train_row.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n_train as f64;
        feat_mean[j] = mean;
        feat_std[j] = var.sqrt().max(1e-10);
    }
    let standardize = |rows: &[Vec<f64>]| -> Vec<f64> {
        rows.iter()
            .flat_map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, &x)| (x - feat_mean[j]) / feat_std[j])
            })
            .collect::<Vec<f64>>()
    };

    let train_flat = standardize(&train_row);
    let train_x_nd = ndarray::Array2::from_shape_vec((n_train, n_features), train_flat).unwrap();
    let train_y_nd = ndarray::Array1::from_vec(train_target);
    let linfa_train_ds = linfa::Dataset::new(train_x_nd, train_y_nd);

    let linfa_model = linfa_elasticnet::ElasticNet::params()
        .penalty(0.1 / n_train as f64)
        .l1_ratio(1.0) // Pure Lasso
        .fit(&linfa_train_ds)
        .unwrap();
    let test_flat = standardize(&test_row);
    let test_x_nd =
        ndarray::Array2::from_shape_vec((test_row.len(), n_features), test_flat).unwrap();
    let linfa_preds_nd = linfa_model.predict(&test_x_nd);
    let linfa_preds: Vec<f64> = linfa_preds_nd.to_vec();
    let linfa_r2 = r2(&test_target, &linfa_preds);
    let linfa_mse = mse(&test_target, &linfa_preds);

    println!("\n{}", "=".repeat(80));
    println!("REGRESSION AUDIT — Lasso ({n}x{n_features}, alpha=0.1, 80/20 split, seed=42)");
    println!(
        "  scry-learn   R2={:.6}  MSE={:.4}  checksum=0x{:016x}",
        scry_r2,
        scry_mse,
        prediction_checksum(&scry_preds)
    );
    println!(
        "  linfa        R2={:.6}  MSE={:.4}  checksum=0x{:016x}",
        linfa_r2,
        linfa_mse,
        prediction_checksum(&linfa_preds)
    );

    // Regression threshold: Lasso on synthetic linear data should achieve high R²
    assert!(
        scry_r2 >= 0.95,
        "scry Lasso R² regression: {scry_r2:.6} < 0.95"
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// ElasticNet: scry vs linfa-elasticnet
// ═══════════════════════════════════════════════════════════════════

#[test]
fn regression_audit_elastic_net() {
    let n = 2000;
    let n_features = 10;
    let (col_major, target, row_major) = gen_regression(n, n_features);
    let (train_col, train_target, train_row, _test_col, test_target, test_row) =
        split_data(&col_major, &target, &row_major);

    // ── scry-learn ──
    let train_ds = scry_learn::dataset::Dataset::new(
        train_col,
        train_target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(),
        "y",
    );
    let mut scry_en = scry_learn::linear::ElasticNet::new()
        .alpha(0.1)
        .l1_ratio(0.5)
        .max_iter(1000);
    scry_en.fit(&train_ds).unwrap();
    let scry_preds = scry_en.predict(&test_row).unwrap();
    let scry_r2 = r2(&test_target, &scry_preds);
    let scry_mse = mse(&test_target, &scry_preds);

    // ── linfa-elasticnet ──
    // Same standardization approach as Lasso test above (see comment there).
    use linfa::prelude::{Fit, Predict};
    let n_train = train_row.len();

    let mut feat_mean = vec![0.0f64; n_features];
    let mut feat_std = vec![0.0f64; n_features];
    for j in 0..n_features {
        let mean = train_row.iter().map(|r| r[j]).sum::<f64>() / n_train as f64;
        let var = train_row.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / n_train as f64;
        feat_mean[j] = mean;
        feat_std[j] = var.sqrt().max(1e-10);
    }
    let standardize = |rows: &[Vec<f64>]| -> Vec<f64> {
        rows.iter()
            .flat_map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, &x)| (x - feat_mean[j]) / feat_std[j])
            })
            .collect::<Vec<f64>>()
    };

    let train_flat = standardize(&train_row);
    let train_x_nd = ndarray::Array2::from_shape_vec((n_train, n_features), train_flat).unwrap();
    let train_y_nd = ndarray::Array1::from_vec(train_target);
    let linfa_train_ds = linfa::Dataset::new(train_x_nd, train_y_nd);

    let linfa_model = linfa_elasticnet::ElasticNet::params()
        .penalty(0.1 / n_train as f64)
        .l1_ratio(0.5)
        .fit(&linfa_train_ds)
        .unwrap();
    let test_flat = standardize(&test_row);
    let test_x_nd =
        ndarray::Array2::from_shape_vec((test_row.len(), n_features), test_flat).unwrap();
    let linfa_preds_nd = linfa_model.predict(&test_x_nd);
    let linfa_preds: Vec<f64> = linfa_preds_nd.to_vec();
    let linfa_r2 = r2(&test_target, &linfa_preds);
    let linfa_mse = mse(&test_target, &linfa_preds);

    println!("\n{}", "=".repeat(80));
    println!("REGRESSION AUDIT — ElasticNet ({n}x{n_features}, alpha=0.1, l1_ratio=0.5, 80/20 split, seed=42)");
    println!(
        "  scry-learn   R2={:.6}  MSE={:.4}  checksum=0x{:016x}",
        scry_r2,
        scry_mse,
        prediction_checksum(&scry_preds)
    );
    println!(
        "  linfa        R2={:.6}  MSE={:.4}  checksum=0x{:016x}",
        linfa_r2,
        linfa_mse,
        prediction_checksum(&linfa_preds)
    );

    // Regression threshold: ElasticNet on synthetic linear data should achieve high R²
    assert!(
        scry_r2 >= 0.95,
        "scry ElasticNet R² regression: {scry_r2:.6} < 0.95"
    );
    println!();
}
