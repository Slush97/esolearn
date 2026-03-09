// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tests that NaN/Inf values in input data are rejected by `fit()`.

use scry_learn::dataset::Dataset;
use scry_learn::error::ScryLearnError;

fn make_clean_dataset() -> Dataset {
    Dataset::new(
        vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 3.0, 8.0, 2.0, 7.0]],
        vec![2.0, 4.0, 6.0, 3.0, 5.0],
        vec!["f1".into(), "f2".into()],
        "target",
    )
}

fn make_nan_feature_dataset() -> Dataset {
    Dataset::new(
        vec![
            vec![1.0, f64::NAN, 3.0, 4.0, 5.0],
            vec![5.0, 3.0, 8.0, 2.0, 7.0],
        ],
        vec![2.0, 4.0, 6.0, 3.0, 5.0],
        vec!["f1".into(), "f2".into()],
        "target",
    )
}

fn make_inf_target_dataset() -> Dataset {
    Dataset::new(
        vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 3.0, 8.0, 2.0, 7.0]],
        vec![2.0, f64::INFINITY, 6.0, 3.0, 5.0],
        vec!["f1".into(), "f2".into()],
        "target",
    )
}

fn make_neg_inf_feature_dataset() -> Dataset {
    Dataset::new(
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, f64::NEG_INFINITY, 8.0, 2.0, 7.0],
        ],
        vec![2.0, 4.0, 6.0, 3.0, 5.0],
        vec!["f1".into(), "f2".into()],
        "target",
    )
}

// ── validate_finite tests ───────────────────────────────────────────

#[test]
fn clean_data_passes_validation() {
    let data = make_clean_dataset();
    assert!(data.validate_finite().is_ok());
}

#[test]
fn nan_feature_fails_validation() {
    let data = make_nan_feature_dataset();
    let err = data.validate_finite().unwrap_err();
    assert!(matches!(err, ScryLearnError::InvalidData(_)));
    let msg = err.to_string();
    assert!(
        msg.contains("NaN"),
        "error message should mention NaN: {msg}"
    );
    assert!(
        msg.contains("f1"),
        "error message should mention column name: {msg}"
    );
}

#[test]
fn inf_target_fails_validation() {
    let data = make_inf_target_dataset();
    let err = data.validate_finite().unwrap_err();
    assert!(matches!(err, ScryLearnError::InvalidData(_)));
    let msg = err.to_string();
    assert!(
        msg.contains("inf"),
        "error message should mention inf: {msg}"
    );
    assert!(
        msg.contains("target"),
        "error message should mention target: {msg}"
    );
}

#[test]
fn neg_inf_feature_fails_validation() {
    let data = make_neg_inf_feature_dataset();
    let err = data.validate_finite().unwrap_err();
    assert!(matches!(err, ScryLearnError::InvalidData(_)));
}

// ── validate_no_inf tests (for imputer) ─────────────────────────────

#[test]
fn nan_allowed_by_validate_no_inf() {
    let data = make_nan_feature_dataset();
    assert!(
        data.validate_no_inf().is_ok(),
        "NaN should be allowed by validate_no_inf"
    );
}

#[test]
fn inf_rejected_by_validate_no_inf() {
    let data = make_inf_target_dataset();
    let err = data.validate_no_inf().unwrap_err();
    assert!(matches!(err, ScryLearnError::InvalidData(_)));
}

// ── Model fit() rejects NaN ─────────────────────────────────────────

#[test]
fn linear_regression_rejects_nan() {
    use scry_learn::linear::LinearRegression;
    let data = make_nan_feature_dataset();
    let mut model = LinearRegression::new();
    let err = model.fit(&data).unwrap_err();
    assert!(matches!(err, ScryLearnError::InvalidData(_)));
}

#[test]
fn linear_regression_rejects_inf_target() {
    use scry_learn::linear::LinearRegression;
    let data = make_inf_target_dataset();
    let mut model = LinearRegression::new();
    let err = model.fit(&data).unwrap_err();
    assert!(matches!(err, ScryLearnError::InvalidData(_)));
}

#[test]
fn random_forest_rejects_nan() {
    use scry_learn::tree::RandomForestClassifier;
    // RF needs classification targets
    let mut data = make_nan_feature_dataset();
    data.target = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    let mut model = RandomForestClassifier::new().n_estimators(5);
    let err = model.fit(&data).unwrap_err();
    assert!(matches!(err, ScryLearnError::InvalidData(_)));
}

#[test]
fn kmeans_rejects_nan() {
    use scry_learn::cluster::KMeans;
    let data = make_nan_feature_dataset();
    let mut model = KMeans::new(2);
    let err = model.fit(&data).unwrap_err();
    assert!(matches!(err, ScryLearnError::InvalidData(_)));
}

#[test]
fn logistic_regression_rejects_inf() {
    use scry_learn::linear::LogisticRegression;
    let mut data = make_neg_inf_feature_dataset();
    data.target = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    let mut model = LogisticRegression::new().max_iter(10);
    let err = model.fit(&data).unwrap_err();
    assert!(matches!(err, ScryLearnError::InvalidData(_)));
}

#[test]
fn clean_data_fit_succeeds() {
    use scry_learn::linear::LinearRegression;
    let data = make_clean_dataset();
    let mut model = LinearRegression::new();
    model
        .fit(&data)
        .expect("clean data should fit successfully");
}
