//! Stress tests for edge cases that could cause panics or infinite loops.
//!
//! These tests verify that models handle degenerate inputs gracefully
//! (returning errors, not panicking or hanging).
//!
//! Run:
//!   cargo test --test `stress_edge_cases` -p scry-learn --release -- --nocapture

use scry_learn::dataset::Dataset;

fn make_dataset(features: Vec<Vec<f64>>, target: Vec<f64>) -> Dataset {
    let names: Vec<String> = (0..features.len()).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "target")
}

// ═══════════════════════════════════════════════════════════════════
// KMeans edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn kmeans_identical_points() {
    // All points identical — must not panic (empty cluster handling)
    let n = 50;
    let features = vec![vec![1.0; n], vec![2.0; n]];
    let data = make_dataset(features, vec![0.0; n]);

    let mut km = scry_learn::cluster::KMeans::new(3).seed(42);
    // Should complete without panic (may produce degenerate clusters)
    let result = km.fit(&data);
    // Either succeeds or returns an error — both are acceptable
    if matches!(result, Ok(())) {
        assert_eq!(km.labels().len(), n);
    }
}

// ═══════════════════════════════════════════════════════════════════
// DBSCAN edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn dbscan_tiny_eps_all_noise() {
    // Very small eps — all points should be noise
    let features = vec![
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
    ];
    let data = make_dataset(features, vec![0.0; 5]);

    let mut db = scry_learn::cluster::Dbscan::new(1e-10, 2);
    db.fit(&data).unwrap();
    // With tiny eps, everything is noise → 0 clusters
    assert_eq!(db.n_clusters(), 0, "tiny eps should yield 0 clusters");
    // All labels should be -1 (noise)
    for &label in db.labels() {
        assert_eq!(label, -1, "all points should be noise with tiny eps");
    }
}

// ═══════════════════════════════════════════════════════════════════
// MLP edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn mlp_single_iteration() {
    // 1 iteration should not panic
    let features = vec![vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 0.0, 1.0, 1.0]];
    let target = vec![0.0, 0.0, 1.0, 1.0];
    let data = make_dataset(features, target);

    let mut clf = scry_learn::neural::MLPClassifier::new()
        .hidden_layers(&[4])
        .max_iter(1)
        .seed(42);
    clf.fit(&data).unwrap();

    let preds = clf.predict(&[vec![0.0, 0.0], vec![2.0, 1.0]]).unwrap();
    assert_eq!(preds.len(), 2);
}

#[test]
fn mlp_empty_dataset_error() {
    let data = make_dataset(vec![Vec::<f64>::new()], Vec::new());
    let mut clf = scry_learn::neural::MLPClassifier::new();
    assert!(clf.fit(&data).is_err(), "empty dataset should error");
}

// ═══════════════════════════════════════════════════════════════════
// IsolationForest edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn iforest_identical_points() {
    // All identical — scores should be uniform
    let rows: Vec<Vec<f64>> = (0..50).map(|_| vec![1.0, 2.0, 3.0]).collect();

    let mut ifo = scry_learn::anomaly::IsolationForest::new()
        .n_estimators(50)
        .seed(42);
    ifo.fit(&rows).unwrap();

    let scores = ifo.predict(&rows);
    assert_eq!(scores.len(), 50);
    // All scores should be very similar
    let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    assert!(
        range < 0.1,
        "identical points should have similar anomaly scores, range={range:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Lasso edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn lasso_high_alpha_zero_coefficients() {
    // Very high alpha should drive all coefficients to zero
    let features = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    let target = vec![1.0, 2.0, 3.0, 4.0];
    let data = make_dataset(features, target);

    let mut model = scry_learn::linear::LassoRegression::new().alpha(1000.0);
    model.fit(&data).unwrap();

    let preds = model.predict(&[vec![1.0, 5.0], vec![4.0, 8.0]]).unwrap();
    assert_eq!(preds.len(), 2);
    // With very high alpha, predictions should be similar (near mean of target)
    let diff = (preds[0] - preds[1]).abs();
    assert!(
        diff < 1.0,
        "high alpha lasso should produce near-constant predictions, diff={diff:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// LinearSVR edge cases
// ═══════════════════════════════════════════════════════════════════

#[test]
fn linear_svr_max_iter_1() {
    // max_iter=1 should not panic (convergence failure is acceptable)
    let features = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let target = vec![2.0, 4.0, 6.0, 8.0];
    let data = make_dataset(features, target);

    let mut model = scry_learn::svm::LinearSVR::new()
        .c(1.0)
        .epsilon(0.1)
        .max_iter(1);
    // Should complete without panic
    let result = model.fit(&data);
    if matches!(result, Ok(())) {
        let preds = model.predict(&[vec![2.5]]).unwrap();
        assert_eq!(preds.len(), 1);
    }
}
