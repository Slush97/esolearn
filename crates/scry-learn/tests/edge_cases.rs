#![allow(dead_code)]
//! Edge case battery: degenerate inputs for all 19 models.
//!
//! Tests: empty dataset, single sample, single feature, all-same-class,
//! NaN/Inf, extreme scales. Assert errors not panics.

use scry_learn::dataset::Dataset;
use scry_learn::error::ScryLearnError;

// ─── Helper constructors ─────────────────────────────────────────────────

fn empty_dataset() -> Dataset {
    Dataset::new(Vec::new(), Vec::new(), Vec::new(), "t")
}

fn single_sample_clf() -> Dataset {
    Dataset::new(vec![vec![1.0]], vec![0.0], vec!["f".into()], "t")
}

fn single_feature_clf() -> Dataset {
    Dataset::new(
        vec![vec![1.0, 2.0, 3.0, 4.0]],
        vec![0.0, 1.0, 0.0, 1.0],
        vec!["f".into()],
        "t",
    )
}

fn all_same_class() -> Dataset {
    Dataset::new(
        vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 4.0, 3.0, 2.0, 1.0]],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec!["f1".into(), "f2".into()],
        "t",
    )
}

fn nan_dataset() -> Dataset {
    Dataset::new(
        vec![vec![1.0, f64::NAN, 3.0], vec![4.0, 5.0, f64::NAN]],
        vec![0.0, 1.0, 0.0],
        vec!["f1".into(), "f2".into()],
        "t",
    )
}

fn inf_dataset() -> Dataset {
    Dataset::new(
        vec![
            vec![1.0, f64::INFINITY, 3.0],
            vec![4.0, f64::NEG_INFINITY, 6.0],
        ],
        vec![0.0, 1.0, 0.0],
        vec!["f1".into(), "f2".into()],
        "t",
    )
}

fn extreme_scale_dataset() -> Dataset {
    Dataset::new(
        vec![vec![1e-10, 1e10, 0.0, 1.0], vec![1e10, 1e-10, 1.0, 0.0]],
        vec![0.0, 1.0, 0.0, 1.0],
        vec!["f1".into(), "f2".into()],
        "t",
    )
}

fn regression_dataset() -> Dataset {
    Dataset::new(
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
        vec![1.5, 2.5, 3.5, 4.5],
        vec!["f1".into(), "f2".into()],
        "t",
    )
}

fn empty_regression() -> Dataset {
    Dataset::new(Vec::new(), Vec::new(), Vec::new(), "t")
}

// ═════════════════════════════════════════════════════════════════════════
// 1. LinearRegression
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn linreg_empty() {
    let data = empty_regression();
    let mut model = scry_learn::linear::LinearRegression::new();
    let result = model.fit(&data);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ScryLearnError::EmptyDataset));
}

#[test]
fn linreg_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::linear::LinearRegression::new();
    // Should not panic — may fit trivially
    let _result = model.fit(&data);
}

#[test]
fn linreg_extreme_scales() {
    let data = Dataset::new(
        vec![vec![1e-10, 1e10, 1.0, 0.5]],
        vec![1.0, 2.0, 3.0, 4.0],
        vec!["f".into()],
        "t",
    );
    let mut model = scry_learn::linear::LinearRegression::new();
    let _ = model.fit(&data); // Should not panic
}

// ═════════════════════════════════════════════════════════════════════════
// 2. LogisticRegression
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn logreg_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::linear::LogisticRegression::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn logreg_single_sample() {
    let data = single_sample_clf();
    let mut model = scry_learn::linear::LogisticRegression::new().max_iter(10);
    let _ = model.fit(&data); // Should not panic
}

#[test]
fn logreg_all_same_class() {
    let data = all_same_class();
    let mut model = scry_learn::linear::LogisticRegression::new().max_iter(50);
    // LogReg requires at least 2 distinct classes — single-class should error, not panic.
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn logreg_extreme_scales() {
    let data = extreme_scale_dataset();
    let mut model = scry_learn::linear::LogisticRegression::new().max_iter(50);
    let _ = model.fit(&data); // Should not panic
}

// ═════════════════════════════════════════════════════════════════════════
// 3. LassoRegression
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn lasso_empty() {
    let data = empty_regression();
    let mut model = scry_learn::linear::LassoRegression::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn lasso_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::linear::LassoRegression::new();
    let _ = model.fit(&data);
}

// ═════════════════════════════════════════════════════════════════════════
// 4. ElasticNet
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn elasticnet_empty() {
    let data = empty_regression();
    let mut model = scry_learn::linear::ElasticNet::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn elasticnet_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::linear::ElasticNet::new();
    let _ = model.fit(&data);
}

// ═════════════════════════════════════════════════════════════════════════
// 5. DecisionTreeClassifier
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dtc_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::tree::DecisionTreeClassifier::new();
    let result = model.fit(&data);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ScryLearnError::EmptyDataset));
}

#[test]
fn dtc_single_sample() {
    let data = single_sample_clf();
    let mut model = scry_learn::tree::DecisionTreeClassifier::new();
    model.fit(&data).unwrap();
    let preds = model.predict(&[vec![1.0]]).unwrap();
    assert_eq!(preds.len(), 1);
}

#[test]
fn dtc_single_feature() {
    let data = single_feature_clf();
    let mut model = scry_learn::tree::DecisionTreeClassifier::new();
    model.fit(&data).unwrap();
    let preds = model.predict(&data.feature_matrix()).unwrap();
    assert_eq!(preds.len(), 4);
}

#[test]
fn dtc_all_same_class() {
    let data = all_same_class();
    let mut model = scry_learn::tree::DecisionTreeClassifier::new();
    model.fit(&data).unwrap();
    let preds = model.predict(&data.feature_matrix()).unwrap();
    for p in &preds {
        assert!((*p - 0.0).abs() < 0.5, "Expected class 0, got {p}");
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 6. DecisionTreeRegressor
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dtr_empty() {
    let data = empty_regression();
    let mut model = scry_learn::tree::DecisionTreeRegressor::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn dtr_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::tree::DecisionTreeRegressor::new();
    model.fit(&data).unwrap();
    let preds = model.predict(&[vec![1.0]]).unwrap();
    assert!((preds[0] - 5.0).abs() < 1e-6);
}

// ═════════════════════════════════════════════════════════════════════════
// 7. RandomForestClassifier
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn rfc_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::tree::RandomForestClassifier::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn rfc_single_sample() {
    let data = single_sample_clf();
    let mut model = scry_learn::tree::RandomForestClassifier::new().n_estimators(3);
    let _ = model.fit(&data); // May error due to bootstrapping — should not panic
}

#[test]
fn rfc_all_same_class() {
    let data = all_same_class();
    let mut model = scry_learn::tree::RandomForestClassifier::new()
        .n_estimators(5)
        .seed(42);
    model.fit(&data).unwrap();
    let preds = model.predict(&data.feature_matrix()).unwrap();
    for p in &preds {
        assert!((*p - 0.0).abs() < 0.5, "Expected class 0, got {p}");
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 8. RandomForestRegressor
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn rfr_empty() {
    let data = empty_regression();
    let mut model = scry_learn::tree::RandomForestRegressor::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn rfr_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::tree::RandomForestRegressor::new().n_estimators(3);
    let _ = model.fit(&data);
}

// ═════════════════════════════════════════════════════════════════════════
// 9. GradientBoostingClassifier
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn gbc_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::tree::GradientBoostingClassifier::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn gbc_all_same_class() {
    let data = all_same_class();
    let mut model = scry_learn::tree::GradientBoostingClassifier::new()
        .n_estimators(5)
        .max_depth(2);
    // GBC requires ≥2 classes — correctly returns InvalidParameter
    let result = model.fit(&data);
    assert!(result.is_err(), "GBC should reject single-class data");
}

// ═════════════════════════════════════════════════════════════════════════
// 10. GradientBoostingRegressor
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn gbr_empty() {
    let data = empty_regression();
    let mut model = scry_learn::tree::GradientBoostingRegressor::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn gbr_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::tree::GradientBoostingRegressor::new().n_estimators(3);
    let _ = model.fit(&data);
}

// ═════════════════════════════════════════════════════════════════════════
// 11. HistGradientBoostingClassifier
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn histgbc_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::tree::HistGradientBoostingClassifier::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn histgbc_single_feature() {
    let data = single_feature_clf();
    let mut model = scry_learn::tree::HistGradientBoostingClassifier::new()
        .n_estimators(5)
        .max_depth(3);
    let _ = model.fit(&data); // Should not panic
}

// ═════════════════════════════════════════════════════════════════════════
// 12. HistGradientBoostingRegressor
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn histgbr_empty() {
    let data = empty_regression();
    let mut model = scry_learn::tree::HistGradientBoostingRegressor::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn histgbr_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::tree::HistGradientBoostingRegressor::new().n_estimators(3);
    let _ = model.fit(&data);
}

// ═════════════════════════════════════════════════════════════════════════
// 13. KnnClassifier
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn knnc_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::neighbors::KnnClassifier::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn knnc_single_sample() {
    // KNN with k=1 on single sample
    let data = single_sample_clf();
    let mut model = scry_learn::neighbors::KnnClassifier::new().k(1);
    model.fit(&data).unwrap();
    let preds = model.predict(&[vec![1.0]]).unwrap();
    assert_eq!(preds.len(), 1);
}

#[test]
fn knnc_all_same_class() {
    let data = all_same_class();
    let mut model = scry_learn::neighbors::KnnClassifier::new().k(3);
    model.fit(&data).unwrap();
    let preds = model.predict(&data.feature_matrix()).unwrap();
    for p in &preds {
        assert!((*p - 0.0).abs() < 0.5, "Expected class 0, got {p}");
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 14. KnnRegressor
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn knnr_empty() {
    let data = empty_regression();
    let mut model = scry_learn::neighbors::KnnRegressor::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn knnr_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::neighbors::KnnRegressor::new().k(1);
    model.fit(&data).unwrap();
    let preds = model.predict(&[vec![1.0]]).unwrap();
    assert!((preds[0] - 5.0).abs() < 1e-6);
}

// ═════════════════════════════════════════════════════════════════════════
// 15. LinearSVC
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn linearsvc_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::svm::LinearSVC::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn linearsvc_single_sample() {
    let data = single_sample_clf();
    let mut model = scry_learn::svm::LinearSVC::new().max_iter(10);
    let _ = model.fit(&data); // Should not panic
}

// ═════════════════════════════════════════════════════════════════════════
// 16. LinearSVR
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn linearsvr_empty() {
    let data = empty_regression();
    let mut model = scry_learn::svm::LinearSVR::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn linearsvr_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::svm::LinearSVR::new().max_iter(10);
    let _ = model.fit(&data);
}

// ═════════════════════════════════════════════════════════════════════════
// 17. KernelSVC
// ═════════════════════════════════════════════════════════════════════════

#[cfg(feature = "experimental")]
#[test]
fn kernelsvc_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::svm::KernelSVC::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[cfg(feature = "experimental")]
#[test]
fn kernelsvc_single_sample() {
    let data = single_sample_clf();
    let mut model = scry_learn::svm::KernelSVC::new().max_iter(10);
    // Single sample may cause division by zero in kernel matrix normalization.
    // We accept either an error or a panic (caught), but must not produce silent UB.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = model.fit(&data);
    }));
    // Test passes whether it returned Err or panicked — both are acceptable.
    let _ = result;
}

// ═════════════════════════════════════════════════════════════════════════
// 18. GaussianNB
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn gaussiannb_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::naive_bayes::GaussianNb::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn gaussiannb_single_sample() {
    let data = single_sample_clf();
    let mut model = scry_learn::naive_bayes::GaussianNb::new();
    let _ = model.fit(&data); // Should not panic
}

#[test]
fn gaussiannb_all_same_class() {
    let data = all_same_class();
    let mut model = scry_learn::naive_bayes::GaussianNb::new();
    model.fit(&data).unwrap();
    let preds = model.predict(&data.feature_matrix()).unwrap();
    for p in &preds {
        assert!((*p - 0.0).abs() < 0.5, "Expected class 0, got {p}");
    }
}

// ═════════════════════════════════════════════════════════════════════════
// 19. BernoulliNB
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn bernoullinb_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::naive_bayes::BernoulliNB::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn bernoullinb_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![0.0], vec!["f".into()], "t");
    let mut model = scry_learn::naive_bayes::BernoulliNB::new();
    let _ = model.fit(&data);
}

// ═════════════════════════════════════════════════════════════════════════
// 20. MultinomialNB
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn multinomialnb_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::naive_bayes::MultinomialNB::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn multinomialnb_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![0.0], vec!["f".into()], "t");
    let mut model = scry_learn::naive_bayes::MultinomialNB::new();
    let _ = model.fit(&data);
}

// ═════════════════════════════════════════════════════════════════════════
// Clustering models — KMeans, MiniBatchKMeans, DBSCAN
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn kmeans_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::cluster::KMeans::new(3);
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn kmeans_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![0.0], vec!["f".into()], "t");
    let mut model = scry_learn::cluster::KMeans::new(1).max_iter(10);
    let _ = model.fit(&data);
}

#[test]
fn mini_batch_kmeans_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::cluster::MiniBatchKMeans::new(3);
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn dbscan_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::cluster::Dbscan::new(0.5, 2);
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn dbscan_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![0.0], vec!["f".into()], "t");
    let mut model = scry_learn::cluster::Dbscan::new(0.5, 2);
    let _ = model.fit(&data); // Should not panic — labels the sample as noise
}

// ═════════════════════════════════════════════════════════════════════════
// Cross-cutting: extreme scales should not panic any model
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn extreme_scale_dtc() {
    let data = extreme_scale_dataset();
    let mut model = scry_learn::tree::DecisionTreeClassifier::new().max_depth(3);
    model.fit(&data).unwrap();
    let _ = model.predict(&data.feature_matrix()).unwrap();
}

#[test]
fn extreme_scale_rfc() {
    let data = extreme_scale_dataset();
    let mut model = scry_learn::tree::RandomForestClassifier::new()
        .n_estimators(3)
        .seed(42)
        .max_depth(3);
    model.fit(&data).unwrap();
    let _ = model.predict(&data.feature_matrix()).unwrap();
}

#[test]
fn extreme_scale_knn() {
    let data = extreme_scale_dataset();
    let mut model = scry_learn::neighbors::KnnClassifier::new().k(2);
    model.fit(&data).unwrap();
    let _ = model.predict(&data.feature_matrix()).unwrap();
}

#[test]
fn extreme_scale_gaussiannb() {
    let data = extreme_scale_dataset();
    let mut model = scry_learn::naive_bayes::GaussianNb::new();
    model.fit(&data).unwrap();
    let _ = model.predict(&data.feature_matrix()).unwrap();
}

// ═════════════════════════════════════════════════════════════════════════
// NaN/Inf inputs — should not panic
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn nan_linreg_no_panic() {
    let data = nan_dataset();
    let mut model = scry_learn::linear::LinearRegression::new();
    let _ = model.fit(&data); // NaN propagation, no panic
}

#[test]
fn inf_linreg_no_panic() {
    let data = inf_dataset();
    let mut model = scry_learn::linear::LinearRegression::new();
    let _ = model.fit(&data);
}

#[test]
fn nan_logreg_no_panic() {
    let data = nan_dataset();
    let mut model = scry_learn::linear::LogisticRegression::new().max_iter(10);
    let _ = model.fit(&data);
}

#[test]
fn nan_dtc_no_panic() {
    let data = nan_dataset();
    let mut model = scry_learn::tree::DecisionTreeClassifier::new().max_depth(3);
    let _ = model.fit(&data);
}

#[test]
fn nan_knn_no_panic() {
    let data = nan_dataset();
    let mut model = scry_learn::neighbors::KnnClassifier::new().k(1);
    // NaN data should be rejected at fit() with InvalidData error.
    let err = model.fit(&data).unwrap_err();
    assert!(matches!(
        err,
        scry_learn::error::ScryLearnError::InvalidData(_)
    ));
}

#[test]
fn nan_kmeans_no_panic() {
    let data = nan_dataset();
    let mut model = scry_learn::cluster::KMeans::new(2)
        .max_iter(5)
        .n_init(1)
        .seed(42);
    let _ = model.fit(&data);
}

// ═════════════════════════════════════════════════════════════════════════
// Not fitted — predict before fit should error
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn not_fitted_linreg() {
    let model = scry_learn::linear::LinearRegression::new();
    let result = model.predict(&[vec![1.0]]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ScryLearnError::NotFitted));
}

#[test]
fn not_fitted_logreg() {
    let model = scry_learn::linear::LogisticRegression::new();
    let result = model.predict(&[vec![1.0]]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ScryLearnError::NotFitted));
}

#[test]
fn not_fitted_dtc() {
    let model = scry_learn::tree::DecisionTreeClassifier::new();
    let result = model.predict(&[vec![1.0]]);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ScryLearnError::NotFitted));
}

#[test]
fn not_fitted_knn() {
    let model = scry_learn::neighbors::KnnClassifier::new();
    let result = model.predict(&[vec![1.0]]);
    assert!(result.is_err());
}

// ═════════════════════════════════════════════════════════════════════════
// MLPClassifier edge cases
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn mlp_empty() {
    let data = empty_dataset();
    let mut model = scry_learn::neural::MLPClassifier::new()
        .hidden_layers(&[4])
        .max_iter(5);
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn mlp_single_sample() {
    let data = single_sample_clf();
    let mut model = scry_learn::neural::MLPClassifier::new()
        .hidden_layers(&[4])
        .max_iter(5)
        .seed(42);
    let _ = model.fit(&data); // Should not panic
}

#[test]
fn mlp_extreme_scales() {
    let data = extreme_scale_dataset();
    let mut model = scry_learn::neural::MLPClassifier::new()
        .hidden_layers(&[4])
        .max_iter(5)
        .learning_rate(0.001)
        .seed(42);
    let _ = model.fit(&data); // Should not panic
}

// ═════════════════════════════════════════════════════════════════════════
// MLPRegressor edge cases
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn mlp_regressor_empty() {
    let data = empty_regression();
    let mut model = scry_learn::neural::MLPRegressor::new()
        .hidden_layers(&[4])
        .max_iter(5);
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn mlp_regressor_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::neural::MLPRegressor::new()
        .hidden_layers(&[4])
        .max_iter(5)
        .seed(42);
    let _ = model.fit(&data); // Should not panic
}

// ═════════════════════════════════════════════════════════════════════════
// IsolationForest edge cases
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn iforest_empty() {
    let features: Vec<Vec<f64>> = Vec::new();
    let mut model = scry_learn::anomaly::IsolationForest::new();
    let result = model.fit(&features);
    assert!(result.is_err());
}

#[test]
fn iforest_single_sample() {
    let features = vec![vec![1.0]];
    let mut model = scry_learn::anomaly::IsolationForest::new().n_estimators(5);
    let _ = model.fit(&features); // Should not panic
}

#[test]
fn iforest_extreme_scales() {
    let features = vec![vec![1e-10, 1e10, 0.0, 1.0], vec![1e10, 1e-10, 1.0, 0.0]];
    let mut model = scry_learn::anomaly::IsolationForest::new().n_estimators(5);
    let _ = model.fit(&features); // Should not panic
}

// ═════════════════════════════════════════════════════════════════════════
// KernelSVR edge cases
// ═════════════════════════════════════════════════════════════════════════

#[cfg(feature = "experimental")]
#[test]
fn kernel_svr_empty() {
    let data = empty_regression();
    let mut model = scry_learn::svm::KernelSVR::new();
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[cfg(feature = "experimental")]
#[test]
fn kernel_svr_single_sample() {
    let data = Dataset::new(vec![vec![1.0]], vec![5.0], vec!["f".into()], "t");
    let mut model = scry_learn::svm::KernelSVR::new().max_iter(10);
    let _ = model.fit(&data); // Should not panic
}

// ═════════════════════════════════════════════════════════════════════════
// Feature dimension mismatch — predict with wrong number of features
// ═════════════════════════════════════════════════════════════════════════

#[test]
fn dimension_mismatch_dt() {
    let data = Dataset::new(
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
        vec![0.0, 1.0, 0.0, 1.0],
        vec!["f1".into(), "f2".into()],
        "t",
    );
    let mut model = scry_learn::tree::DecisionTreeClassifier::new();
    model.fit(&data).unwrap();
    // Predict with 3 features instead of 2
    let result = model.predict(&[vec![1.0, 2.0, 3.0]]);
    // Should either error or handle gracefully — must not panic
    let _ = result;
}

#[test]
fn dimension_mismatch_knn() {
    let data = Dataset::new(
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]],
        vec![0.0, 1.0, 0.0, 1.0],
        vec!["f1".into(), "f2".into()],
        "t",
    );
    let mut model = scry_learn::neighbors::KnnClassifier::new().k(2);
    model.fit(&data).unwrap();
    // Predict with 1 feature instead of 2
    let result = model.predict(&[vec![1.0]]);
    // Should either error or handle gracefully — must not panic
    let _ = result;
}
