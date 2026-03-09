#![allow(clippy::needless_range_loop)]
//! Numerical stability tests — deterministic coverage of degenerate-but-valid inputs.
//!
//! These complement fuzz testing (which hits edge cases randomly) with reproducible,
//! documented assertions on soundness: finite output, no panics, correct error returns.
//!
//! Run: cargo test --test `numerical_stability` -p scry-learn --release -- --nocapture

use scry_learn::dataset::Dataset;
use scry_learn::linear::{LassoRegression, LinearRegression, LogisticRegression};
use scry_learn::metrics::{f1_score, Average};
use scry_learn::preprocess::{StandardScaler, Transformer};
use scry_learn::tree::{DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier};

fn make_dataset(cols: Vec<Vec<f64>>, target: Vec<f64>) -> Dataset {
    let n_features = cols.len();
    let names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    Dataset::new(cols, target, names, "target")
}

// ═══════════════════════════════════════════════════════════════════
// 1. Near-singular matrix (99.99% correlated features)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn near_singular_linear_regression() {
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 200;
    let base: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    // Second feature is base + tiny noise → near-perfect correlation
    let noisy: Vec<f64> = base.iter().map(|&v| v + rng.f64() * 1e-8).collect();
    let target: Vec<f64> = base.iter().map(|&v| 2.0 * v + 1.0).collect();

    let ds = make_dataset(vec![base, noisy], target);
    let mut model = LinearRegression::new();
    let result = model.fit(&ds);

    // Must not panic. May produce large coefficients but they must be finite.
    if matches!(result, Ok(())) {
        let coeffs = model.coefficients();
        for (i, &c) in coeffs.iter().enumerate() {
            assert!(c.is_finite(), "coefficient {i} is not finite: {c}");
        }
        assert!(model.intercept().is_finite(), "intercept is not finite");

        let preds = model.predict(&[vec![5.0, 5.0]]).unwrap();
        assert!(
            preds[0].is_finite(),
            "prediction is not finite: {}",
            preds[0]
        );
    }
    // Err is also acceptable — singular matrix detection.
    println!("near_singular_linear_regression: {:?}", result.is_ok());
}

// ═══════════════════════════════════════════════════════════════════
// 2. Extreme scale disparity (1e-10 to 1e10 without preprocessing)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn extreme_scale_disparity() {
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 200;
    // Feature 1: ~1e-10 scale, Feature 2: ~1e10 scale
    let tiny: Vec<f64> = (0..n).map(|_| rng.f64() * 1e-10).collect();
    let huge: Vec<f64> = (0..n).map(|_| rng.f64() * 1e10).collect();
    let target: Vec<f64> = tiny
        .iter()
        .zip(huge.iter())
        .map(|(&t, &h)| t * 1e10 + h * 1e-10)
        .collect();

    let ds = make_dataset(vec![tiny, huge], target);

    // LinearRegression
    let mut lr = LinearRegression::new();
    if matches!(lr.fit(&ds), Ok(())) {
        let preds = lr.predict(&[vec![5e-11, 5e9]]).unwrap();
        assert!(
            preds[0].is_finite(),
            "LR prediction not finite: {}",
            preds[0]
        );
    }

    // DecisionTree should handle scale-invariant splitting
    let half = n / 2;
    let cls_target: Vec<f64> = (0..n).map(|i| if i < half { 0.0 } else { 1.0 }).collect();
    let ds_cls = make_dataset(
        vec![
            (0..n).map(|_| rng.f64() * 1e-10).collect(),
            (0..n)
                .map(|i| {
                    if i < half {
                        rng.f64()
                    } else {
                        rng.f64() + 1e10
                    }
                })
                .collect(),
        ],
        cls_target,
    );
    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&ds_cls).unwrap();
    let row = vec![5e-11, 5e9];
    let pred = dt.predict(&[row]).unwrap();
    assert!(pred[0].is_finite());

    println!("extreme_scale_disparity: passed");
}

// ═══════════════════════════════════════════════════════════════════
// 3. Severe class imbalance (99:1)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn severe_class_imbalance() {
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 1000;
    let n_minority = 10; // 1% minority class
    let n_features = 5;

    let mut cols = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];

    for j in 0..n_features {
        for i in 0..n {
            cols[j][i] = rng.f64() * 10.0;
        }
    }
    // Minority class: shift features to make them separable
    for i in (n - n_minority)..n {
        target[i] = 1.0;
        for j in 0..n_features {
            cols[j][i] += 20.0;
        }
    }

    let ds = make_dataset(cols, target.clone());

    // DT
    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&ds).unwrap();
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n_features).map(|j| ds.features[j][i]).collect())
        .collect();
    let preds = dt.predict(&rows).unwrap();

    // Report F1-macro, not accuracy (accuracy is meaningless at 99:1)
    let f1 = f1_score(&target, &preds, Average::Macro);
    println!("severe_class_imbalance DT F1-macro: {f1:.4}");
    // DT should be able to find the cluster — F1 > 0 at minimum
    assert!(
        f1 > 0.0,
        "F1-macro should be > 0 on separable imbalanced data"
    );

    // RF
    let mut rf = RandomForestClassifier::new().n_estimators(50).max_depth(5);
    rf.fit(&ds).unwrap();
    let preds_rf = rf.predict(&rows).unwrap();
    let f1_rf = f1_score(&target, &preds_rf, Average::Macro);
    println!("severe_class_imbalance RF F1-macro: {f1_rf:.4}");
    assert!(f1_rf > 0.0);
}

// ═══════════════════════════════════════════════════════════════════
// 4. Single-class training data
// ═══════════════════════════════════════════════════════════════════

#[test]
fn single_class_input() {
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 100;
    let col: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
    let target = vec![1.0; n]; // all same class

    let ds = make_dataset(vec![col], target);

    // DT: should not panic. May produce a single-leaf tree.
    let mut dt = DecisionTreeClassifier::new();
    let result = dt.fit(&ds);
    println!("single_class DT fit: {:?}", result.is_ok());
    if result.is_ok() {
        let preds = dt.predict(&[vec![0.5]]).unwrap();
        assert!(preds[0].is_finite(), "prediction should be finite");
        println!("single_class DT prediction: {}", preds[0]);
    }

    // LogReg on single-class: should return Err (need >= 2 classes).
    let mut lr = LogisticRegression::new().max_iter(50);
    let lr_result = lr.fit(&ds);
    assert!(
        lr_result.is_err(),
        "LogReg on single-class should return Err"
    );
    println!("single_class LogReg fit: Err — correct behavior");
}

// ═══════════════════════════════════════════════════════════════════
// 5. Zero-variance columns
// ═══════════════════════════════════════════════════════════════════

#[test]
fn zero_variance_columns() {
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 200;
    let half = n / 2;

    let constant_col = vec![42.0; n]; // zero variance
    let normal_col: Vec<f64> = (0..n)
        .map(|i| if i < half { rng.f64() } else { rng.f64() + 5.0 })
        .collect();
    let target: Vec<f64> = (0..n).map(|i| if i < half { 0.0 } else { 1.0 }).collect();

    let ds = make_dataset(vec![constant_col, normal_col], target);

    // StandardScaler must not divide by zero
    let mut scaler = StandardScaler::new();
    scaler.fit(&ds).unwrap();
    let mut ds_scaled = ds.clone();
    scaler.transform(&mut ds_scaled).unwrap();
    // Constant column should become all zeros (0-mean / 1-std-as-safeguard)
    for &v in &ds_scaled.features[0] {
        assert!(v.is_finite(), "scaled zero-variance value not finite: {v}");
    }

    // DT on zero-variance data: should still work (ignoring the constant feature)
    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&ds).unwrap();
    let preds = dt.predict(&[vec![42.0, 3.0]]).unwrap();
    assert!(preds[0].is_finite());

    println!("zero_variance_columns: passed");
}

// ═══════════════════════════════════════════════════════════════════
// 6. Near-zero-variance (1e-15)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn near_zero_variance() {
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 200;
    // Column with variance ~1e-30
    let col: Vec<f64> = (0..n).map(|_| 1.0 + rng.f64() * 1e-15).collect();
    let target: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();

    let ds = make_dataset(vec![col], target);

    let mut scaler = StandardScaler::new();
    scaler.fit(&ds).unwrap();
    let mut ds_scaled = ds.clone();
    scaler.transform(&mut ds_scaled).unwrap();
    for &v in &ds_scaled.features[0] {
        assert!(
            v.is_finite(),
            "near-zero-variance scaled value not finite: {v}"
        );
    }

    println!("near_zero_variance: passed");
}

// ═══════════════════════════════════════════════════════════════════
// 7. p >> n (1000 features, 50 samples)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn high_dimensional_p_gt_n() {
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 50;
    let p = 1000;

    let cols: Vec<Vec<f64>> = (0..p)
        .map(|_| (0..n).map(|_| rng.f64()).collect())
        .collect();
    let target: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();

    let ds = make_dataset(cols, target);

    // LinearRegression on underdetermined system — should not panic.
    // Normal equations may be singular; Ridge should handle it.
    let mut lr_ridge = LinearRegression::new().alpha(1.0);
    let result = lr_ridge.fit(&ds);
    println!("p>>n LinearRegression(alpha=1.0) fit: {:?}", result.is_ok());
    if result.is_ok() {
        let coeffs = lr_ridge.coefficients();
        let finite_count = coeffs.iter().filter(|c| c.is_finite()).count();
        assert_eq!(
            finite_count, p,
            "all coefficients should be finite with regularization"
        );
    }

    // Lasso on p >> n should produce sparse solution
    let mut lasso = LassoRegression::new().alpha(1.0).max_iter(100);
    let lasso_result = lasso.fit(&ds);
    println!("p>>n Lasso fit: {:?}", lasso_result.is_ok());
    if lasso_result.is_ok() {
        let coeffs = lasso.coefficients();
        let zero_count = coeffs.iter().filter(|&&c| c.abs() < 1e-10).count();
        println!(
            "  Lasso sparsity: {zero_count}/{p} coefficients are zero"
        );
        for &c in coeffs {
            assert!(c.is_finite(), "Lasso coefficient not finite: {c}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// 8. Constant target (regression)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn constant_target_regression() {
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 200;
    let col: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let target = vec![7.0; n]; // constant target

    let ds = make_dataset(vec![col], target);

    // LinearRegression: should learn intercept=7, coefficients~0
    let mut lr = LinearRegression::new();
    lr.fit(&ds).unwrap();
    assert!(
        (lr.intercept() - 7.0).abs() < 0.1,
        "intercept should be ~7.0, got {}",
        lr.intercept()
    );
    assert!(
        lr.coefficients()[0].abs() < 0.1,
        "coefficient should be ~0, got {}",
        lr.coefficients()[0]
    );

    // DTRegressor: should predict ~7.0 for any input
    let mut dt = DecisionTreeRegressor::new();
    dt.fit(&ds).unwrap();
    let preds = dt.predict(&[vec![5.0]]).unwrap();
    assert!(
        (preds[0] - 7.0).abs() < 0.5,
        "DTRegressor should predict ~7.0, got {}",
        preds[0]
    );

    println!("constant_target_regression: passed");
}
