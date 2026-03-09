//! Integration test: end-to-end sparse Dataset workflow.
//!
//! Tests sparse construction, splitting, densification, and model fit/predict
//! through the sparse → dense auto-conversion path.

use scry_learn::dataset::Dataset;
use scry_learn::sparse::CscMatrix;
use scry_learn::split::train_test_split;

#[test]
fn test_sparse_end_to_end() {
    // 10 samples × 3 features, mostly zeros.
    let col0 = vec![1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 7.0, 0.0, 0.0, 10.0];
    let col1 = vec![0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 8.0, 0.0, 0.0];
    let col2 = vec![0.0, 0.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 9.0, 0.0];
    let csc = CscMatrix::from_dense(&[col0, col1, col2]);
    let target = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    let ds = Dataset::from_sparse(
        csc,
        target,
        vec!["f0".into(), "f1".into(), "f2".into()],
        "label",
    );
    assert!(ds.is_sparse());
    assert_eq!(ds.n_samples(), 10);
    assert_eq!(ds.n_features(), 3);

    // Train/test split preserves sparsity.
    let (train, test) = train_test_split(&ds, 0.3, 42);
    assert!(train.is_sparse());
    assert!(test.is_sparse());
    assert_eq!(train.n_samples() + test.n_samples(), 10);
    assert_eq!(train.n_features(), 3);
    assert_eq!(test.n_features(), 3);

    // CSC accessor works on split subsets.
    let train_csc = train.sparse_csc().expect("train should be sparse");
    assert_eq!(train_csc.n_rows(), train.n_samples());
    assert_eq!(train_csc.n_cols(), 3);

    let test_csc = test.sparse_csc().expect("test should be sparse");
    assert_eq!(test_csc.n_rows(), test.n_samples());
    assert_eq!(test_csc.n_cols(), 3);
}

#[test]
fn test_sparse_ensure_dense_after_split() {
    let col0 = vec![1.0, 0.0, 2.0, 0.0];
    let col1 = vec![0.0, 3.0, 0.0, 4.0];
    let csc = CscMatrix::from_dense(&[col0, col1]);
    let target = vec![0.0, 1.0, 0.0, 1.0];

    let ds = Dataset::from_sparse(csc, target, vec!["a".into(), "b".into()], "t");
    let (mut train, _test) = train_test_split(&ds, 0.5, 42);

    // Densify and verify features are accessible.
    train.ensure_dense();
    assert_eq!(train.features.len(), 2);
    assert_eq!(train.n_samples(), train.features[0].len());
}

// ═════════════════════════════════════════════════════════════════════════
// Sparse → model fit/predict end-to-end
// ═════════════════════════════════════════════════════════════════════════

/// Build a sparse classification dataset: two separable clusters.
fn sparse_clf_dataset() -> Dataset {
    // 20 samples × 4 features, ~50% sparse
    let col0 = vec![
        1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0,
        0.0, 10.0,
    ];
    let col1 = vec![
        0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0, 9.0, 0.0,
        10.0, 0.0,
    ];
    let col2 = vec![
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5,
        5.5, 5.5,
    ];
    let col3 = vec![0.0; 20];
    let csc = CscMatrix::from_dense(&[col0, col1, col2, col3]);
    let target = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0,
    ];
    Dataset::from_sparse(
        csc,
        target,
        vec!["f0".into(), "f1".into(), "f2".into(), "f3".into()],
        "class",
    )
}

#[test]
fn test_sparse_decision_tree_fit_predict() {
    // DT on sparse data: must densify first (DT builder indexes features directly).
    let ds = sparse_clf_dataset();
    let (mut train, test) = train_test_split(&ds, 0.3, 42);
    train.ensure_dense();

    let mut dt = scry_learn::tree::DecisionTreeClassifier::new().max_depth(5);
    dt.fit(&train).unwrap();
    let test_rows = test.feature_matrix();
    let preds = dt.predict(&test_rows).unwrap();
    assert_eq!(preds.len(), test.n_samples());

    // On this separable data, accuracy should be reasonable.
    let correct = preds
        .iter()
        .zip(test.target.iter())
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();
    let acc = correct as f64 / preds.len() as f64;
    assert!(acc >= 0.5, "DT on sparse data: accuracy {acc:.2} < 0.5");
}

#[test]
fn test_sparse_knn_fit_predict() {
    // KNN trained on sparse data must use predict_sparse() — verify that path works.
    let ds = sparse_clf_dataset();
    let (train, test) = train_test_split(&ds, 0.3, 42);

    let mut knn = scry_learn::neighbors::KnnClassifier::new().k(3);
    knn.fit(&train).unwrap();

    // Dense predict should return an error explaining sparse path is needed.
    let test_rows = test.feature_matrix();
    let dense_result = knn.predict(&test_rows);
    assert!(
        dense_result.is_err(),
        "predict on sparse-trained KNN should return error"
    );

    // predict_sparse should work. Build CSR from test data.
    let test_csr = test.sparse_csr().expect("test should have sparse CSR");
    let preds = knn.predict_sparse(&test_csr).unwrap();
    assert_eq!(preds.len(), test.n_samples());
}

#[test]
fn test_sparse_logistic_regression_fit_predict() {
    let ds = sparse_clf_dataset();
    let (train, test) = train_test_split(&ds, 0.3, 42);

    let mut lr = scry_learn::linear::LogisticRegression::new().max_iter(100);
    lr.fit(&train).unwrap();
    let test_rows = test.feature_matrix();
    let preds = lr.predict(&test_rows).unwrap();
    assert_eq!(preds.len(), test.n_samples());
}

#[test]
fn test_sparse_standard_scaler() {
    let ds = sparse_clf_dataset();
    let mut scaler = scry_learn::preprocess::StandardScaler::new();
    scry_learn::preprocess::Transformer::fit(&mut scaler, &ds).unwrap();
    let mut ds_copy = ds;
    scry_learn::preprocess::Transformer::transform(&scaler, &mut ds_copy).unwrap();

    // Verify scaled values are finite.
    for col in &ds_copy.features {
        for &v in col {
            assert!(v.is_finite(), "scaled sparse value not finite: {v}");
        }
    }
}
