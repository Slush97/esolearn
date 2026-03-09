//! Serialization roundtrip tests: verify all models can be serialized and
//! deserialized via `serde_json`, and that the deserialized model produces
//! identical predictions.
//!
//! These tests require the `serde` feature flag:
//! ```sh
//! cargo test --features serde --test serialization
//! ```

#![cfg(feature = "serde")]

use scry_learn::dataset::Dataset;
use scry_learn::linear::{ElasticNet, LassoRegression, LinearRegression, LogisticRegression};
use scry_learn::naive_bayes::GaussianNb;
use scry_learn::neighbors::KnnClassifier;
use scry_learn::preprocess::{Pca, StandardScaler, Transformer};
use scry_learn::tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
};

// ─────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────

fn classification_data() -> Dataset {
    let mut f1 = Vec::new();
    let mut f2 = Vec::new();
    let mut target = Vec::new();
    for i in 0..30 {
        f1.push(i as f64 % 5.0);
        f2.push(i as f64 % 3.0);
        target.push(if i < 15 { 0.0 } else { 1.0 });
    }
    Dataset::new(vec![f1, f2], target, vec!["x".into(), "y".into()], "class")
}

fn regression_data() -> Dataset {
    let n = 50;
    let mut rng = fastrand::Rng::with_seed(42);
    let x: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
    Dataset::new(vec![x], y, vec!["x".into()], "y")
}

/// Serialize model to JSON, deserialize back, and assert predictions match.
macro_rules! roundtrip_test {
    ($name:ident, $model_type:ty, $setup:expr, $data_fn:ident) => {
        #[test]
        fn $name() {
            let data = $data_fn();
            let mut model: $model_type = $setup;

            // Some models need fit(&Dataset), some fit(&mut Dataset).
            model.fit(&data).unwrap();

            let features = data.feature_matrix();
            let preds_before = model.predict(&features).unwrap();

            let json = serde_json::to_string(&model).expect("serialize failed");
            assert!(!json.is_empty(), "serialized JSON should not be empty");

            let restored: $model_type = serde_json::from_str(&json).expect("deserialize failed");
            let preds_after = restored.predict(&features).unwrap();

            assert_eq!(
                preds_before.len(),
                preds_after.len(),
                "prediction length mismatch"
            );
            for (i, (a, b)) in preds_before.iter().zip(preds_after.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-10,
                    "prediction mismatch at index {i}: before={a}, after={b}"
                );
            }
        }
    };
}

// ─────────────────────────────────────────────────────────────────
// Roundtrip tests for all model types
// ─────────────────────────────────────────────────────────────────

// Trees
roundtrip_test!(
    serde_decision_tree_classifier,
    DecisionTreeClassifier,
    DecisionTreeClassifier::new(),
    classification_data
);

roundtrip_test!(
    serde_decision_tree_regressor,
    DecisionTreeRegressor,
    DecisionTreeRegressor::new(),
    regression_data
);

roundtrip_test!(
    serde_random_forest_classifier,
    RandomForestClassifier,
    RandomForestClassifier::new().n_estimators(10).seed(42),
    classification_data
);

roundtrip_test!(
    serde_random_forest_regressor,
    RandomForestRegressor,
    RandomForestRegressor::new().n_estimators(10).seed(42),
    regression_data
);

roundtrip_test!(
    serde_gbt_regressor,
    GradientBoostingRegressor,
    GradientBoostingRegressor::new()
        .n_estimators(20)
        .learning_rate(0.1)
        .max_depth(3),
    regression_data
);

roundtrip_test!(
    serde_gbt_classifier,
    GradientBoostingClassifier,
    GradientBoostingClassifier::new()
        .n_estimators(20)
        .learning_rate(0.1)
        .max_depth(3),
    classification_data
);

// Linear
roundtrip_test!(
    serde_linear_regression,
    LinearRegression,
    LinearRegression::new(),
    regression_data
);

roundtrip_test!(
    serde_lasso_regression,
    LassoRegression,
    LassoRegression::new().alpha(0.1).max_iter(2000),
    regression_data
);

roundtrip_test!(
    serde_elastic_net,
    ElasticNet,
    ElasticNet::new().alpha(0.1).l1_ratio(0.5).max_iter(2000),
    regression_data
);

// Neighbors
roundtrip_test!(
    serde_knn_classifier,
    KnnClassifier,
    KnnClassifier::new().k(3),
    classification_data
);

// Naive Bayes
roundtrip_test!(
    serde_gaussian_nb,
    GaussianNb,
    GaussianNb::new(),
    classification_data
);

// ─────────────────────────────────────────────────────────────────
// Logistic Regression (needs mut for fit)
// ─────────────────────────────────────────────────────────────────

#[test]
fn serde_logistic_regression() {
    let data = classification_data();
    let mut model = LogisticRegression::new().learning_rate(0.1).max_iter(500);
    model.fit(&data).unwrap();

    let features = data.feature_matrix();
    let preds_before = model.predict(&features).unwrap();

    let json = serde_json::to_string(&model).expect("serialize failed");
    let restored: LogisticRegression = serde_json::from_str(&json).expect("deserialize failed");
    let preds_after = restored.predict(&features).unwrap();

    assert_eq!(preds_before.len(), preds_after.len());
    for (i, (a, b)) in preds_before.iter().zip(preds_after.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "prediction mismatch at {i}: {a} vs {b}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// Preprocessor roundtrips
// ─────────────────────────────────────────────────────────────────

#[test]
fn serde_standard_scaler() {
    let data = regression_data();
    let mut scaler = StandardScaler::new();
    scaler.fit(&data).unwrap();

    let json = serde_json::to_string(&scaler).expect("serialize failed");
    let restored: StandardScaler = serde_json::from_str(&json).expect("deserialize failed");

    let mut d1 = data.clone();
    let mut d2 = data.clone();
    scaler.transform(&mut d1).unwrap();
    restored.transform(&mut d2).unwrap();

    for (a, b) in d1.features[0].iter().zip(d2.features[0].iter()) {
        assert!((a - b).abs() < 1e-10, "scaler mismatch: {a} vs {b}");
    }
}

#[test]
fn serde_pca() {
    let data = classification_data();
    let mut pca = Pca::with_n_components(1);
    pca.fit(&data).unwrap();

    let json = serde_json::to_string(&pca).expect("serialize failed");
    let restored: Pca = serde_json::from_str(&json).expect("deserialize failed");

    let mut d1 = data.clone();
    let mut d2 = data.clone();
    pca.transform(&mut d1).unwrap();
    restored.transform(&mut d2).unwrap();

    for (a, b) in d1.features[0].iter().zip(d2.features[0].iter()) {
        assert!((a - b).abs() < 1e-10, "PCA mismatch: {a} vs {b}");
    }
}

// ─────────────────────────────────────────────────────────────────
// Dataset roundtrip
// ─────────────────────────────────────────────────────────────────

#[test]
fn serde_dataset() {
    let data = regression_data();
    let json = serde_json::to_string(&data).expect("serialize failed");
    let restored: Dataset = serde_json::from_str(&json).expect("deserialize failed");

    assert_eq!(data.n_samples(), restored.n_samples());
    assert_eq!(data.n_features(), restored.n_features());
    assert_eq!(data.feature_names, restored.feature_names);
    assert_eq!(data.target_name, restored.target_name);
    for (a, b) in data.target.iter().zip(restored.target.iter()) {
        assert!((a - b).abs() < 1e-10);
    }
}
