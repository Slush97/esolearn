#![allow(
    clippy::items_after_statements,
    clippy::default_trait_access,
    clippy::needless_range_loop,
    clippy::cast_possible_wrap
)]
//! 3-way fairness audit: scry-learn vs smartcore 0.4.9 vs linfa-ensemble
//! Also verifies accuracy & raw timing outside of criterion.
//!
//! Run: cargo test --test `benchmark_audit` -p scry-learn --release -- --nocapture

use std::time::Instant;

fn gen_classification(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let half = n / 2;
    let mut features_col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];

    for j in 0..n_features {
        let offset = 3.0 + j as f64 * 0.5;
        for i in 0..half {
            features_col_major[j][i] = rng.f64() * 2.0;
        }
        for i in half..n {
            features_col_major[j][i] = rng.f64() * 2.0 + offset;
            target[i] = 1.0;
        }
    }

    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_features).map(|j| features_col_major[j][i]).collect())
        .collect();

    (row_major, target)
}

fn transpose(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if rows.is_empty() {
        return vec![];
    }
    let n_cols = rows[0].len();
    let n_rows = rows.len();
    (0..n_cols)
        .map(|j| (0..n_rows).map(|i| rows[i][j]).collect())
        .collect()
}

fn accuracy_f64(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| (t - p).abs() < 1e-9)
        .count();
    correct as f64 / y_true.len() as f64
}

/// FNV-1a hash of prediction vector for cross-machine reproducibility verification.
fn prediction_checksum(preds: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &p in preds {
        h ^= p.to_bits();
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Convert row-major Vec<Vec<f64>> + Vec<f64> target into linfa Dataset (ndarray-based)
fn to_linfa_dataset(
    features: &[Vec<f64>],
    target: &[f64],
) -> linfa::DatasetBase<ndarray::Array2<f64>, ndarray::Array1<usize>> {
    let n = features.len();
    let m = features[0].len();
    let flat: Vec<f64> = features.iter().flat_map(|r| r.iter().copied()).collect();
    let x = ndarray::Array2::from_shape_vec((n, m), flat).unwrap();
    let y = ndarray::Array1::from_vec(target.iter().map(|&t| t as usize).collect());
    linfa::Dataset::new(x, y)
}

#[test]
fn audit_dt_predict_fairness() {
    let (features, target) = gen_classification(1000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    // ── scry-learn ──
    let data = scry_learn::prelude::Dataset::new(
        transpose(&features),
        target.clone(),
        (0..10).map(|i| format!("f{i}")).collect(),
        "target",
    );
    let mut scry_dt = scry_learn::prelude::DecisionTreeClassifier::new();
    scry_dt.fit(&data).unwrap();

    // ── smartcore 0.4.9 ──
    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
    let smart_dt = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
        &x,
        &target_i32,
        Default::default(),
    )
    .unwrap();

    // ── linfa-trees ──
    use linfa::prelude::{Fit, Predict};
    let linfa_ds = to_linfa_dataset(&features, &target);
    let linfa_dt = linfa_trees::DecisionTree::params().fit(&linfa_ds).unwrap();

    // Verify accuracy
    let scry_preds = scry_dt.predict(&features).unwrap();
    let smart_preds_i32: Vec<i32> = smart_dt.predict(&x).unwrap();
    let smart_preds: Vec<f64> = smart_preds_i32.iter().map(|&p| p as f64).collect();
    let linfa_preds_arr = linfa_dt.predict(&linfa_ds);
    let linfa_preds: Vec<f64> = linfa_preds_arr.iter().map(|&p| p as f64).collect();

    let scry_acc = accuracy_f64(&target, &scry_preds);
    let smart_acc = accuracy_f64(&target, &smart_preds);
    let linfa_acc = accuracy_f64(&target, &linfa_preds);

    println!("\n{}", "=".repeat(65));
    println!("DECISION TREE ACCURACY (1K×10, train=test, timing only — NOT generalization)");
    println!(
        "  scry-learn:          {:.1}%  checksum=0x{:016x}",
        scry_acc * 100.0,
        prediction_checksum(&scry_preds)
    );
    println!(
        "  smartcore 0.4.9:     {:.1}%  checksum=0x{:016x}",
        smart_acc * 100.0,
        prediction_checksum(&smart_preds)
    );
    println!(
        "  linfa-trees 0.7:     {:.1}%  checksum=0x{:016x}",
        linfa_acc * 100.0,
        prediction_checksum(&linfa_preds)
    );

    assert!(scry_acc > 0.95);
    assert!(smart_acc > 0.95);
    assert!(linfa_acc > 0.95);

    // ── Raw timing ──
    let n_iters = 1000;

    // Warmup
    for _ in 0..2 {
        std::hint::black_box(scry_dt.predict(std::hint::black_box(&features)).unwrap());
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        std::hint::black_box(scry_dt.predict(std::hint::black_box(&features)).unwrap());
    }
    let scry_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    for _ in 0..2 {
        std::hint::black_box(smart_dt.predict(std::hint::black_box(&x)).unwrap());
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        std::hint::black_box(smart_dt.predict(std::hint::black_box(&x)).unwrap());
    }
    let smart_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    for _ in 0..2 {
        std::hint::black_box(linfa_dt.predict(std::hint::black_box(&linfa_ds)));
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        std::hint::black_box(linfa_dt.predict(std::hint::black_box(&linfa_ds)));
    }
    let linfa_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    println!("\nDECISION TREE PREDICT LATENCY (1K samples, {n_iters} iters)");
    println!("  scry-learn:          {scry_us:.2} µs");
    println!(
        "  smartcore 0.4.9:     {:.2} µs  ({:.1}× slower)",
        smart_us,
        smart_us / scry_us
    );
    println!(
        "  linfa-trees 0.7:     {:.2} µs  ({:.1}× slower)",
        linfa_us,
        linfa_us / scry_us
    );
    println!();
}

#[test]
fn audit_rf_predict_fairness() {
    let (features, target) = gen_classification(2000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let n_trees = 100;

    // ── scry-learn ──
    let data = scry_learn::prelude::Dataset::new(
        transpose(&features),
        target.clone(),
        (0..10).map(|i| format!("f{i}")).collect(),
        "target",
    );
    let mut scry_rf = scry_learn::prelude::RandomForestClassifier::new()
        .n_estimators(n_trees)
        .max_depth(8);
    scry_rf.fit(&data).unwrap();

    // ── smartcore 0.4.9 ──
    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
    let params =
        smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(n_trees as u16)
            .with_max_depth(8);
    let smart_rf = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
        &x,
        &target_i32,
        params,
    )
    .unwrap();

    // ── linfa-ensemble ──
    use linfa::prelude::{Fit, Predict};
    let linfa_ds = to_linfa_dataset(&features, &target);
    let linfa_rf = linfa_ensemble::RandomForestParams::new(
        linfa_trees::DecisionTree::params().max_depth(Some(8)),
    )
    .ensemble_size(n_trees)
    .bootstrap_proportion(0.7)
    .feature_proportion(0.3)
    .fit(&linfa_ds)
    .unwrap();

    // Verify accuracy
    let scry_preds = scry_rf.predict(&features).unwrap();
    let smart_preds_i32: Vec<i32> = smart_rf.predict(&x).unwrap();
    let smart_preds: Vec<f64> = smart_preds_i32.iter().map(|&p| p as f64).collect();
    let linfa_preds_arr = linfa_rf.predict(&linfa_ds);
    let linfa_preds: Vec<f64> = linfa_preds_arr.iter().map(|&p| p as f64).collect();

    let scry_acc = accuracy_f64(&target, &scry_preds);
    let smart_acc = accuracy_f64(&target, &smart_preds);
    let linfa_acc = accuracy_f64(&target, &linfa_preds);

    println!("\n{}", "=".repeat(65));
    println!("RANDOM FOREST ACCURACY (2K×10, 100t, max_depth=8, train=test, timing only — NOT generalization)");
    println!(
        "  scry-learn:              {:.1}%  checksum=0x{:016x}",
        scry_acc * 100.0,
        prediction_checksum(&scry_preds)
    );
    println!(
        "  smartcore 0.4.9:         {:.1}%  checksum=0x{:016x}",
        smart_acc * 100.0,
        prediction_checksum(&smart_preds)
    );
    println!(
        "  linfa-ensemble 0.8:      {:.1}%  checksum=0x{:016x}",
        linfa_acc * 100.0,
        prediction_checksum(&linfa_preds)
    );

    assert!(scry_acc > 0.90);
    assert!(smart_acc > 0.90);
    assert!(linfa_acc > 0.90);

    // ── Predict timing ──
    let n_iters = 200;

    // Warmup
    for _ in 0..2 {
        std::hint::black_box(scry_rf.predict(std::hint::black_box(&features)).unwrap());
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        std::hint::black_box(scry_rf.predict(std::hint::black_box(&features)).unwrap());
    }
    let scry_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    for _ in 0..2 {
        std::hint::black_box(smart_rf.predict(std::hint::black_box(&x)).unwrap());
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        std::hint::black_box(smart_rf.predict(std::hint::black_box(&x)).unwrap());
    }
    let smart_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    for _ in 0..2 {
        std::hint::black_box(linfa_rf.predict(std::hint::black_box(&linfa_ds)));
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        std::hint::black_box(linfa_rf.predict(std::hint::black_box(&linfa_ds)));
    }
    let linfa_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    println!("\nRANDOM FOREST PREDICT (2K samples, 100t, {n_iters} iters)");
    println!("  scry-learn:              {scry_us:.2} µs");
    println!(
        "  smartcore 0.4.9:         {:.2} µs  ({:.1}× slower)",
        smart_us,
        smart_us / scry_us
    );
    println!(
        "  linfa-ensemble 0.8:      {:.2} µs  ({:.1}× slower)",
        linfa_us,
        linfa_us / scry_us
    );

    // ── Train timing ──
    let n_train_iters = 10;

    let t0 = Instant::now();
    for _ in 0..n_train_iters {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features),
            target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(),
            "target",
        );
        let mut rf = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(n_trees)
            .max_depth(8);
        rf.fit(std::hint::black_box(&d)).unwrap();
    }
    let scry_train_ms = t0.elapsed().as_nanos() as f64 / n_train_iters as f64 / 1_000_000.0;

    let t0 = Instant::now();
    for _ in 0..n_train_iters {
        let x2 = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
        let p2 = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(n_trees as u16).with_max_depth(8);
        let _ = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
            std::hint::black_box(&x2),
            std::hint::black_box(&target_i32),
            p2,
        )
        .unwrap();
    }
    let smart_train_ms = t0.elapsed().as_nanos() as f64 / n_train_iters as f64 / 1_000_000.0;

    let t0 = Instant::now();
    for _ in 0..n_train_iters {
        let ds = to_linfa_dataset(&features, &target);
        let _ = linfa_ensemble::RandomForestParams::new(
            linfa_trees::DecisionTree::params().max_depth(Some(8)),
        )
        .ensemble_size(n_trees)
        .bootstrap_proportion(0.7)
        .feature_proportion(0.3)
        .fit(std::hint::black_box(&ds))
        .unwrap();
    }
    let linfa_train_ms = t0.elapsed().as_nanos() as f64 / n_train_iters as f64 / 1_000_000.0;

    println!("\nRANDOM FOREST TRAIN (2K×10, 100t, {n_train_iters} iters)");
    println!("  scry-learn:              {scry_train_ms:.2} ms");
    println!(
        "  smartcore 0.4.9:         {:.2} ms  ({:.1}× slower)",
        smart_train_ms,
        smart_train_ms / scry_train_ms
    );
    println!(
        "  linfa-ensemble 0.8:      {:.2} ms  ({:.1}× slower)",
        linfa_train_ms,
        linfa_train_ms / scry_train_ms
    );
    println!(
        "  Note: linfa uses bootstrap=0.7, features=0.3 (non-default, differs from scry/smartcore)"
    );
    println!();
}

#[test]
fn audit_pca_fairness() {
    // Generate a 2K×20 dataset (enough to stress-test eigendecomposition).
    let n_samples = 2000;
    let n_features = 20;
    let n_components = 5;

    let (features_row, target) = gen_classification(n_samples, n_features);
    let features_col = transpose(&features_row);

    // ══════════════════════════════════════════════════════════════
    // FIT TIMING
    // ══════════════════════════════════════════════════════════════

    let n_fit_iters = 50;

    // ── scry-learn ──
    let scry_ds = scry_learn::prelude::Dataset::new(
        features_col,
        target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(),
        "target",
    );
    // Warmup
    for _ in 0..2 {
        let mut pca = scry_learn::prelude::Pca::with_n_components(n_components);
        scry_learn::preprocess::Transformer::fit(&mut pca, std::hint::black_box(&scry_ds)).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_fit_iters {
        let mut pca = scry_learn::prelude::Pca::with_n_components(n_components);
        scry_learn::preprocess::Transformer::fit(&mut pca, std::hint::black_box(&scry_ds)).unwrap();
        std::hint::black_box(&pca);
    }
    let scry_fit_us = t0.elapsed().as_nanos() as f64 / n_fit_iters as f64 / 1000.0;

    // ── smartcore 0.4.9 ──
    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features_row).unwrap();
    let smart_params =
        smartcore::decomposition::pca::PCAParameters::default().with_n_components(n_components);

    for _ in 0..2 {
        let pca =
            smartcore::decomposition::pca::PCA::fit(std::hint::black_box(&x), smart_params.clone())
                .unwrap();
        std::hint::black_box(&pca);
    }
    let t0 = Instant::now();
    for _ in 0..n_fit_iters {
        let pca =
            smartcore::decomposition::pca::PCA::fit(std::hint::black_box(&x), smart_params.clone())
                .unwrap();
        std::hint::black_box(&pca);
    }
    let smart_fit_us = t0.elapsed().as_nanos() as f64 / n_fit_iters as f64 / 1000.0;

    // ── linfa-reduction ──
    use linfa::prelude::{Fit, Predict};
    let flat: Vec<f64> = features_row
        .iter()
        .flat_map(|r| r.iter().copied())
        .collect();
    let x_nd = ndarray::Array2::from_shape_vec((n_samples, n_features), flat).unwrap();
    let linfa_ds = linfa::Dataset::new(
        x_nd.clone(),
        ndarray::Array1::<usize>::from_vec(target.iter().map(|&t| t as usize).collect()),
    );

    for _ in 0..2 {
        let pca = linfa_reduction::Pca::params(n_components)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
        std::hint::black_box(&pca);
    }
    let t0 = Instant::now();
    for _ in 0..n_fit_iters {
        let pca = linfa_reduction::Pca::params(n_components)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
        std::hint::black_box(&pca);
    }
    let linfa_fit_us = t0.elapsed().as_nanos() as f64 / n_fit_iters as f64 / 1000.0;

    // ══════════════════════════════════════════════════════════════
    // TRANSFORM TIMING
    // ══════════════════════════════════════════════════════════════

    let n_transform_iters = 200;

    // Fit all three once for transform benchmarks.
    let mut scry_pca = scry_learn::prelude::Pca::with_n_components(n_components);
    scry_learn::preprocess::Transformer::fit(&mut scry_pca, &scry_ds).unwrap();

    let smart_pca = smartcore::decomposition::pca::PCA::fit(&x, smart_params).unwrap();

    let linfa_pca = linfa_reduction::Pca::params(n_components)
        .fit(&linfa_ds)
        .unwrap();

    // Warmup transforms
    for _ in 0..2 {
        let mut ds = scry_ds.clone();
        scry_learn::preprocess::Transformer::transform(&scry_pca, std::hint::black_box(&mut ds))
            .unwrap();
    }
    // scry-learn transform
    let t0 = Instant::now();
    for _ in 0..n_transform_iters {
        let mut ds = scry_ds.clone();
        scry_learn::preprocess::Transformer::transform(&scry_pca, std::hint::black_box(&mut ds))
            .unwrap();
        std::hint::black_box(&ds);
    }
    let scry_transform_us = t0.elapsed().as_nanos() as f64 / n_transform_iters as f64 / 1000.0;

    // smartcore transform
    for _ in 0..2 {
        let result = smart_pca.transform(std::hint::black_box(&x)).unwrap();
        std::hint::black_box(&result);
    }
    let t0 = Instant::now();
    for _ in 0..n_transform_iters {
        let result = smart_pca.transform(std::hint::black_box(&x)).unwrap();
        std::hint::black_box(&result);
    }
    let smart_transform_us = t0.elapsed().as_nanos() as f64 / n_transform_iters as f64 / 1000.0;

    // linfa transform (uses predict)
    for _ in 0..2 {
        let result = linfa_pca.predict(std::hint::black_box(&x_nd));
        std::hint::black_box(&result);
    }
    let t0 = Instant::now();
    for _ in 0..n_transform_iters {
        let result = linfa_pca.predict(std::hint::black_box(&x_nd));
        std::hint::black_box(&result);
    }
    let linfa_transform_us = t0.elapsed().as_nanos() as f64 / n_transform_iters as f64 / 1000.0;

    // ══════════════════════════════════════════════════════════════
    // CORRECTNESS: Explained variance ratio
    // ══════════════════════════════════════════════════════════════

    let scry_ratios = scry_pca.explained_variance_ratio();
    let linfa_ratios = linfa_pca.explained_variance_ratio();

    // smartcore doesn't expose explained_variance_ratio directly.

    // Verify scry-learn and linfa agree on variance ratios.
    for i in 0..n_components {
        let diff = (scry_ratios[i] - linfa_ratios[i]).abs();
        assert!(
            diff < 0.05,
            "PC{} variance ratio mismatch: scry={:.4} linfa={:.4}",
            i + 1,
            scry_ratios[i],
            linfa_ratios[i],
        );
    }

    // ══════════════════════════════════════════════════════════════
    // REPORT
    // ══════════════════════════════════════════════════════════════

    println!("\n{}", "=".repeat(65));
    println!("PCA FIT ({n_samples}×{n_features} → {n_components} components, {n_fit_iters} iters)");
    println!("  scry-learn:          {scry_fit_us:.2} µs");
    println!(
        "  smartcore 0.4.9:     {:.2} µs  ({:.2}×)",
        smart_fit_us,
        smart_fit_us / scry_fit_us
    );
    println!(
        "  linfa-reduction 0.8: {:.2} µs  ({:.2}×)",
        linfa_fit_us,
        linfa_fit_us / scry_fit_us
    );

    println!(
        "\nPCA TRANSFORM ({n_samples}×{n_features} → {n_components}, {n_transform_iters} iters)"
    );
    println!("  scry-learn:          {scry_transform_us:.2} µs");
    println!(
        "  smartcore 0.4.9:     {:.2} µs  ({:.2}×)",
        smart_transform_us,
        smart_transform_us / scry_transform_us
    );
    println!(
        "  linfa-reduction 0.8: {:.2} µs  ({:.2}×)",
        linfa_transform_us,
        linfa_transform_us / scry_transform_us
    );

    println!("\nEXPLAINED VARIANCE RATIOS (top {n_components}):");
    print!("  scry-learn:          [");
    for (i, &r) in scry_ratios.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{r:.4}");
    }
    println!("]");
    print!("  linfa-reduction:     [");
    for (i, &r) in linfa_ratios.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{r:.4}");
    }
    println!("]");

    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// GBT Benchmark: scry-learn vs smartcore XGRegressor
// ═══════════════════════════════════════════════════════════════════════════

fn gen_regression(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];
    // y = sum of features + noise
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n_features {
            let v = rng.f64() * 10.0;
            col_major[j][i] = v;
            sum += v * (j as f64 + 1.0);
        }
        target[i] = sum + rng.f64() * 0.1;
    }
    let row_major: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_features).map(|j| col_major[j][i]).collect())
        .collect();
    (col_major, target, row_major)
}

#[test]
fn audit_ensemble_regression_fairness() {
    let n_samples = 2000;
    let n_features = 10;
    let n_estimators = 100;
    let n_iters = 5;

    let (col_major, target, row_major) = gen_regression(n_samples, n_features);

    // ── scry-learn GBT Regressor ──
    let scry_ds = scry_learn::dataset::Dataset::new(
        col_major,
        target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(),
        "y",
    );

    // Warmup
    for _ in 0..2 {
        let mut gbr = scry_learn::tree::GradientBoostingRegressor::new()
            .n_estimators(n_estimators)
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(std::hint::black_box(&scry_ds)).unwrap();
    }
    let scry_start = Instant::now();
    for _ in 0..n_iters {
        let mut gbr = scry_learn::tree::GradientBoostingRegressor::new()
            .n_estimators(n_estimators)
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(std::hint::black_box(&scry_ds)).unwrap();
    }
    let scry_fit_us = scry_start.elapsed().as_micros() as f64 / n_iters as f64;

    // Fit once for predict + accuracy.
    let mut scry_gbr = scry_learn::tree::GradientBoostingRegressor::new()
        .n_estimators(n_estimators)
        .learning_rate(0.1)
        .max_depth(3);
    scry_gbr.fit(&scry_ds).unwrap();

    for _ in 0..2 {
        let _ = scry_gbr.predict(std::hint::black_box(&row_major)).unwrap();
    }
    let scry_pred_start = Instant::now();
    for _ in 0..n_iters {
        let _ = scry_gbr.predict(std::hint::black_box(&row_major)).unwrap();
    }
    let scry_pred_us = scry_pred_start.elapsed().as_micros() as f64 / n_iters as f64;

    let scry_preds = scry_gbr.predict(&row_major).unwrap();
    let scry_mse: f64 = scry_preds
        .iter()
        .zip(target.iter())
        .map(|(&p, &y)| (p - y).powi(2))
        .sum::<f64>()
        / n_samples as f64;

    // ── smartcore RF Regressor ──
    use smartcore::ensemble::random_forest_regressor::RandomForestRegressor as SmartRFR;
    use smartcore::linalg::basic::matrix::DenseMatrix;

    // smartcore 0.4 doesn't expose a clean GBT API — use RF regressor as proxy
    // for ensemble regression timing comparison.
    let smart_x = DenseMatrix::from_2d_vec(&row_major).unwrap();

    for _ in 0..2 {
        let _model = SmartRFR::fit(
            std::hint::black_box(&smart_x),
            std::hint::black_box(&target),
            Default::default(),
        )
        .unwrap();
    }
    let smart_fit_start = Instant::now();
    for _ in 0..n_iters {
        let _model = SmartRFR::fit(
            std::hint::black_box(&smart_x),
            std::hint::black_box(&target),
            Default::default(),
        )
        .unwrap();
    }
    let smart_fit_us = smart_fit_start.elapsed().as_micros() as f64 / n_iters as f64;

    let smart_model = SmartRFR::fit(&smart_x, &target, Default::default()).unwrap();
    for _ in 0..2 {
        let _ = smart_model.predict(std::hint::black_box(&smart_x)).unwrap();
    }
    let smart_pred_start = Instant::now();
    for _ in 0..n_iters {
        let _ = smart_model.predict(std::hint::black_box(&smart_x)).unwrap();
    }
    let smart_pred_us = smart_pred_start.elapsed().as_micros() as f64 / n_iters as f64;

    // ── Report ──
    println!("\n{}", "=".repeat(65));
    println!("ENSEMBLE REGRESSION — TIMING ONLY (NOT a like-for-like comparison)");
    println!("  scry = GradientBoostingRegressor (100 trees, lr=0.1, depth=3)");
    println!("  smartcore = RandomForestRegressor (default params)");
    println!("  smartcore 0.4 lacks a GBT API — RF shown for reference only.");
    println!("  Accuracy/MSE numbers are NOT comparable across different algorithms.");
    println!();
    println!("  FIT ({n_samples}x{n_features}, {n_estimators} estimators, {n_iters} iters)");
    println!("    scry-learn GBT:      {scry_fit_us:.2} us");
    println!("    smartcore RF 0.4:    {smart_fit_us:.2} us");

    println!("\n  PREDICT ({n_samples} samples, {n_iters} iters)");
    println!("    scry-learn GBT:      {scry_pred_us:.2} us");
    println!("    smartcore RF 0.4:    {smart_pred_us:.2} us");

    println!(
        "\n  scry-learn GBT train MSE (self-only, no comparison): {scry_mse:.4}"
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// Logistic Regression Benchmark: scry-learn vs smartcore vs linfa-logistic
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn audit_logreg_fairness() {
    let (features, target) = gen_classification(1000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();
    let target_bool: Vec<bool> = target.iter().map(|&t| t > 0.5).collect();

    // ── scry-learn ──
    let data = scry_learn::prelude::Dataset::new(
        transpose(&features),
        target.clone(),
        (0..10).map(|i| format!("f{i}")).collect(),
        "target",
    );
    let mut scry_lr = scry_learn::prelude::LogisticRegression::new()
        .max_iter(200)
        .learning_rate(0.1);
    scry_lr.fit(&data).unwrap();

    // ── smartcore 0.4.9 ──
    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
    let smart_lr = smartcore::linear::logistic_regression::LogisticRegression::fit(
        &x,
        &target_i32,
        Default::default(),
    )
    .unwrap();

    // ── linfa-logistic ──
    use linfa::prelude::{Fit, Predict};
    let flat: Vec<f64> = features.iter().flat_map(|r| r.iter().copied()).collect();
    let x_nd = ndarray::Array2::from_shape_vec((1000, 10), flat).unwrap();
    let y_nd = ndarray::Array1::from_vec(target_bool);
    let linfa_ds = linfa::Dataset::new(x_nd, y_nd);
    let linfa_lr = linfa_logistic::LogisticRegression::default()
        .max_iterations(200)
        .fit(&linfa_ds)
        .unwrap();

    // Verify accuracy.
    let matrix = data.feature_matrix();
    let scry_preds = scry_lr.predict(&matrix).unwrap();
    let smart_preds_i32: Vec<i32> = smart_lr.predict(&x).unwrap();
    let smart_preds: Vec<f64> = smart_preds_i32.iter().map(|&p| p as f64).collect();
    let linfa_preds_arr = linfa_lr.predict(&linfa_ds);
    let linfa_preds: Vec<f64> = linfa_preds_arr
        .iter()
        .map(|&p| if p { 1.0 } else { 0.0 })
        .collect();

    let scry_acc = accuracy_f64(&target, &scry_preds);
    let smart_acc = accuracy_f64(&target, &smart_preds);
    let linfa_acc = accuracy_f64(&target, &linfa_preds);

    println!("\n{}", "=".repeat(65));
    println!("LOGISTIC REGRESSION ACCURACY (1K×10, train=test, timing only — NOT generalization)");
    println!(
        "  scry-learn:          {:.1}%  checksum=0x{:016x}",
        scry_acc * 100.0,
        prediction_checksum(&scry_preds)
    );
    println!(
        "  smartcore 0.4.9:     {:.1}%  checksum=0x{:016x}",
        smart_acc * 100.0,
        prediction_checksum(&smart_preds)
    );
    println!(
        "  linfa-logistic 0.8:  {:.1}%  checksum=0x{:016x}",
        linfa_acc * 100.0,
        prediction_checksum(&linfa_preds)
    );

    assert!(scry_acc > 0.90);
    assert!(smart_acc > 0.90);
    assert!(linfa_acc > 0.90);

    // ── Train timing ──
    let n_iters = 50;

    // Warmup
    for _ in 0..2 {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features),
            target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(),
            "target",
        );
        let mut lr = scry_learn::prelude::LogisticRegression::new()
            .max_iter(200)
            .learning_rate(0.1);
        lr.fit(std::hint::black_box(&d)).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features),
            target.clone(),
            (0..10).map(|i| format!("f{i}")).collect(),
            "target",
        );
        let mut lr = scry_learn::prelude::LogisticRegression::new()
            .max_iter(200)
            .learning_rate(0.1);
        lr.fit(std::hint::black_box(&d)).unwrap();
    }
    let scry_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    for _ in 0..2 {
        let x2 = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
        let _ = smartcore::linear::logistic_regression::LogisticRegression::fit(
            std::hint::black_box(&x2),
            std::hint::black_box(&target_i32),
            Default::default(),
        )
        .unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let x2 = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
        let _ = smartcore::linear::logistic_regression::LogisticRegression::fit(
            std::hint::black_box(&x2),
            std::hint::black_box(&target_i32),
            Default::default(),
        )
        .unwrap();
    }
    let smart_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    for _ in 0..2 {
        let _ = linfa_logistic::LogisticRegression::default()
            .max_iterations(200)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = linfa_logistic::LogisticRegression::default()
            .max_iterations(200)
            .fit(std::hint::black_box(&linfa_ds))
            .unwrap();
    }
    let linfa_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    println!("\nLOGISTIC REGRESSION TRAIN (1K×10, {n_iters} iters)");
    println!("  scry-learn:          {scry_us:.2} µs");
    println!(
        "  smartcore 0.4.9:     {:.2} µs  ({:.1}×)",
        smart_us,
        smart_us / scry_us
    );
    println!(
        "  linfa-logistic 0.8:  {:.2} µs  ({:.1}×)",
        linfa_us,
        linfa_us / scry_us
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// KNN Benchmark: scry-learn vs smartcore
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn audit_knn_fairness() {
    let (features, target) = gen_classification(1000, 10);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    // ── scry-learn ──
    let data = scry_learn::prelude::Dataset::new(
        transpose(&features),
        target.clone(),
        (0..10).map(|i| format!("f{i}")).collect(),
        "target",
    );
    let mut scry_knn = scry_learn::prelude::KnnClassifier::new().k(5);
    scry_knn.fit(&data).unwrap();

    // ── smartcore 0.4.9 ──
    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&features).unwrap();
    let smart_knn = smartcore::neighbors::knn_classifier::KNNClassifier::fit(
        &x,
        &target_i32,
        smartcore::neighbors::knn_classifier::KNNClassifierParameters::default().with_k(5),
    )
    .unwrap();

    // Verify accuracy.
    let matrix = data.feature_matrix();
    let scry_preds = scry_knn.predict(&matrix).unwrap();
    let smart_preds_i32: Vec<i32> = smart_knn.predict(&x).unwrap();
    let smart_preds: Vec<f64> = smart_preds_i32.iter().map(|&p| p as f64).collect();

    let scry_acc = accuracy_f64(&target, &scry_preds);
    let smart_acc = accuracy_f64(&target, &smart_preds);

    println!("\n{}", "=".repeat(65));
    println!("KNN ACCURACY (1K×10, k=5, train=test, timing only — NOT generalization)");
    println!(
        "  scry-learn:          {:.1}%  checksum=0x{:016x}",
        scry_acc * 100.0,
        prediction_checksum(&scry_preds)
    );
    println!(
        "  smartcore 0.4.9:     {:.1}%  checksum=0x{:016x}",
        smart_acc * 100.0,
        prediction_checksum(&smart_preds)
    );

    assert!(scry_acc > 0.90);
    assert!(smart_acc > 0.90);

    // ── Predict timing ──
    let n_iters = 100;

    // Warmup
    for _ in 0..2 {
        std::hint::black_box(scry_knn.predict(std::hint::black_box(&matrix)).unwrap());
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        std::hint::black_box(scry_knn.predict(std::hint::black_box(&matrix)).unwrap());
    }
    let scry_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    for _ in 0..2 {
        std::hint::black_box(smart_knn.predict(std::hint::black_box(&x)).unwrap());
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        std::hint::black_box(smart_knn.predict(std::hint::black_box(&x)).unwrap());
    }
    let smart_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    println!("\nKNN PREDICT (1K samples, k=5, {n_iters} iters)");
    println!("  scry-learn:          {scry_us:.2} µs");
    println!(
        "  smartcore 0.4.9:     {:.2} µs  ({:.1}×)",
        smart_us,
        smart_us / scry_us
    );

    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// K-Means Benchmark: scry-learn vs linfa-clustering
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn audit_kmeans_fairness() {
    use linfa::prelude::Fit;

    let n_samples = 2000;
    let n_features = 10;
    let k = 3;

    let (features, target) = gen_classification(n_samples, n_features);

    // ── scry-learn ──
    let data = scry_learn::prelude::Dataset::new(
        transpose(&features),
        target.clone(),
        (0..n_features).map(|i| format!("f{i}")).collect(),
        "target",
    );
    let mut scry_km = scry_learn::prelude::KMeans::new(k)
        .seed(42)
        .max_iter(100)
        .n_init(1);
    scry_km.fit(&data).unwrap();
    let scry_inertia = scry_km.inertia();

    // ── linfa-clustering ──
    let flat: Vec<f64> = features.iter().flat_map(|r| r.iter().copied()).collect();
    let x_nd = ndarray::Array2::from_shape_vec((n_samples, n_features), flat).unwrap();
    let linfa_ds = linfa::DatasetBase::from(x_nd.clone());
    let _linfa_km = linfa_clustering::KMeans::params_with_rng(k, rand::thread_rng())
        .max_n_iterations(100)
        .fit(&linfa_ds)
        .unwrap();

    println!("\n{}", "=".repeat(65));
    println!("K-MEANS CLUSTERING ({n_samples}×{n_features}, k={k})");
    println!("  scry-learn inertia:      {scry_inertia:.2}");
    println!("  (linfa-clustering uses a different inertia metric)");

    // ── Train timing ──
    let n_iters = 20;

    // Warmup
    for _ in 0..2 {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features),
            target.clone(),
            (0..n_features).map(|i| format!("f{i}")).collect(),
            "target",
        );
        let mut km = scry_learn::prelude::KMeans::new(k)
            .seed(42)
            .max_iter(100)
            .n_init(1);
        km.fit(std::hint::black_box(&d)).unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let d = scry_learn::prelude::Dataset::new(
            transpose(&features),
            target.clone(),
            (0..n_features).map(|i| format!("f{i}")).collect(),
            "target",
        );
        let mut km = scry_learn::prelude::KMeans::new(k)
            .seed(42)
            .max_iter(100)
            .n_init(1);
        km.fit(std::hint::black_box(&d)).unwrap();
    }
    let scry_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    for _ in 0..2 {
        let ds = linfa::DatasetBase::from(x_nd.clone());
        let _ = linfa_clustering::KMeans::params_with_rng(k, rand::thread_rng())
            .max_n_iterations(100)
            .fit(std::hint::black_box(&ds))
            .unwrap();
    }
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let ds = linfa::DatasetBase::from(x_nd.clone());
        let _ = linfa_clustering::KMeans::params_with_rng(k, rand::thread_rng())
            .max_n_iterations(100)
            .fit(std::hint::black_box(&ds))
            .unwrap();
    }
    let linfa_us = t0.elapsed().as_nanos() as f64 / n_iters as f64 / 1000.0;

    println!("\nK-MEANS TRAIN ({n_samples}×{n_features}, k={k}, {n_iters} iters)");
    println!("  scry-learn:              {scry_us:.2} µs");
    println!(
        "  linfa-clustering 0.8:    {:.2} µs  ({:.2}×)",
        linfa_us,
        linfa_us / scry_us
    );

    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// DBSCAN fairness audit — scry-only profiling at multiple data sizes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn audit_dbscan_fairness() {
    let sizes = [500, 2_000, 10_000];
    let n_features = 10;
    let n_iters = 5;

    println!("\n{}", "═".repeat(72));
    println!("  DBSCAN PROFILING — scry-learn (eps=1.0, min_samples=5)");
    println!("{}", "═".repeat(72));
    println!("  {:>8} {:>14} {:>14}", "N", "Fit time(ms)", "Clusters");
    println!("  {}", "─".repeat(40));

    for &n in &sizes {
        let (features, target) = gen_classification(n, n_features);

        let data = scry_learn::prelude::Dataset::new(
            transpose(&features),
            target,
            (0..n_features).map(|i| format!("f{i}")).collect(),
            "target",
        );

        let mut total_ms = 0.0;
        let mut n_clusters = 0;

        for _ in 0..n_iters {
            let mut m = scry_learn::prelude::Dbscan::new(1.0, 5);
            let t0 = std::time::Instant::now();
            m.fit(std::hint::black_box(&data)).unwrap();
            total_ms += t0.elapsed().as_secs_f64() * 1000.0;
            n_clusters = m.n_clusters();
        }

        let avg_ms = total_ms / n_iters as f64;
        println!("  {n:>8} {avg_ms:>12.2}ms {n_clusters:>14}",);
    }

    println!("\n  Note: DBSCAN uses KD-tree for Euclidean distance with ≤ 20 features.");
    println!("  Sublinear scaling demonstrates KD-tree O(n log n) vs brute-force O(n²).");
    println!();
}
