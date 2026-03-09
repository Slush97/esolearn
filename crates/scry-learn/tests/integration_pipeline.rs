//! Integration pipeline tests: every model family through fit → predict → metric.
//!
//! Embedded datasets (Iris + synthetic regression), 80/20 split, seed 42.
//! Target: all tests pass in < 30s on a laptop.

use scry_learn::cluster::{silhouette_score, KMeans, MiniBatchKMeans};
use scry_learn::dataset::Dataset;
use scry_learn::linear::{
    ElasticNet, LassoRegression, LinearRegression, LogisticRegression, Ridge,
};
use scry_learn::metrics::{accuracy, mean_squared_error, r2_score};
use scry_learn::naive_bayes::{BernoulliNB, GaussianNb};
use scry_learn::neighbors::{KnnClassifier, KnnRegressor};
use scry_learn::neural::{MLPClassifier, MLPRegressor};
use scry_learn::pipeline::Pipeline;
use scry_learn::preprocess::{MinMaxScaler, Pca, StandardScaler, Transformer};
use scry_learn::split::train_test_split;
use scry_learn::svm::{LinearSVC, LinearSVR};
#[cfg(feature = "experimental")]
use scry_learn::svm::{Kernel, KernelSVC, KernelSVR};
use scry_learn::tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
};

// ─────────────────────────────────────────────────────────────────
// Embedded Iris dataset (sklearn.datasets.load_iris)
// ─────────────────────────────────────────────────────────────────

fn iris_dataset() -> Dataset {
    let sepal_length = vec![
        5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1,
        5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5.0,
        5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0, 7.0, 6.4, 6.9, 5.5,
        6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1,
        6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5,
        5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3,
        6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0, 6.9, 5.6, 7.7, 6.3, 6.7, 7.2,
        6.2, 6.1, 6.4, 7.2, 7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6.0, 6.9, 6.7, 6.9, 5.8, 6.8,
        6.7, 6.7, 6.3, 6.5, 6.2, 5.9,
    ];
    let sepal_width = vec![
        3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5,
        3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2,
        3.5, 3.6, 3.0, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3, 3.2, 3.2, 3.1, 2.3,
        2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7, 2.2, 2.5, 3.2, 2.8,
        2.5, 2.8, 3.2, 3.0, 2.8, 3.0, 2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3.0, 3.4, 3.1, 2.3, 3.0, 2.5,
        2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9,
        2.5, 3.6, 3.2, 2.7, 3.0, 2.5, 2.8, 3.2, 3.0, 3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2,
        2.8, 3.0, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4, 3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2,
        3.3, 3.0, 2.5, 3.0, 3.4, 3.0,
    ];
    let petal_length = vec![
        1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4,
        1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2,
        1.3, 1.4, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0,
        4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0,
        4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0,
        4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3,
        5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0,
        4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9,
        5.7, 5.2, 5.0, 5.2, 5.4, 5.1,
    ];
    let petal_width = vec![
        0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3,
        0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2,
        0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3,
        1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3,
        1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3,
        1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8,
        1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 1.8, 1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8,
        1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3,
        2.5, 2.3, 1.9, 2.0, 2.3, 1.8,
    ];
    let target: Vec<f64> = (0..150)
        .map(|i| {
            if i < 50 {
                0.0
            } else if i < 100 {
                1.0
            } else {
                2.0
            }
        })
        .collect();

    Dataset::new(
        vec![sepal_length, sepal_width, petal_length, petal_width],
        target,
        vec![
            "sepal_length".into(),
            "sepal_width".into(),
            "petal_length".into(),
            "petal_width".into(),
        ],
        "species",
    )
}

// ─────────────────────────────────────────────────────────────────
// Synthetic regression dataset: y = x0 + 2*x1 + noise
// ─────────────────────────────────────────────────────────────────

fn regression_dataset() -> Dataset {
    let mut rng = fastrand::Rng::with_seed(42);
    let n = 200;
    let mut x0 = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);
    for _ in 0..n {
        let v0 = rng.f64() * 10.0 - 5.0;
        let v1 = rng.f64() * 10.0 - 5.0;
        let v2 = rng.f64() * 10.0 - 5.0;
        let noise = (rng.f64() - 0.5) * 0.5;
        x0.push(v0);
        x1.push(v1);
        x2.push(v2);
        target.push(v0 + 2.0 * v1 + noise);
    }
    Dataset::new(
        vec![x0, x1, x2],
        target,
        vec!["x0".into(), "x1".into(), "x2".into()],
        "y",
    )
}

// ═════════════════════════════════════════════════════════════════
// Classifiers (Iris, accuracy > 0.8)
// ═════════════════════════════════════════════════════════════════

#[test]
fn classifier_decision_tree() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = DecisionTreeClassifier::new();
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("DecisionTreeClassifier accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[test]
fn classifier_random_forest() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = RandomForestClassifier::new().n_estimators(10).seed(42);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("RandomForestClassifier accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[test]
fn classifier_gradient_boosting() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = GradientBoostingClassifier::new()
        .n_estimators(20)
        .learning_rate(0.1)
        .max_depth(3);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("GradientBoostingClassifier accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[test]
fn classifier_hist_gradient_boosting() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = HistGradientBoostingClassifier::new()
        .n_estimators(20)
        .learning_rate(0.1);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("HistGradientBoostingClassifier accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[test]
fn classifier_knn() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = KnnClassifier::new().k(5);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("KnnClassifier accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[test]
fn classifier_gaussian_nb() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = GaussianNb::new();
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("GaussianNb accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[test]
fn classifier_logistic_regression() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = LogisticRegression::new().learning_rate(0.1).max_iter(1000);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("LogisticRegression accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[test]
fn classifier_linear_svc() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = LinearSVC::new().c(1.0).max_iter(2000);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("LinearSVC accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[cfg(feature = "experimental")]
#[test]
#[ignore] // Kernel SVM is O(n²); too slow in debug mode — run with --release --ignored
fn classifier_kernel_svc() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = KernelSVC::new()
        .kernel(Kernel::RBF { gamma: 1.0 })
        .c(10.0)
        .max_iter(100);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("KernelSVC accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[test]
fn classifier_mlp() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = MLPClassifier::new()
        .hidden_layers(&[10])
        .learning_rate(0.1)
        .max_iter(200)
        .seed(42);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("MLPClassifier accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}

#[test]
fn classifier_bernoulli_nb() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut model = BernoulliNB::new().binarize(Some(3.0));
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("BernoulliNB accuracy: {acc:.3}");
    // BernoulliNB with binarized continuous features is weaker; accept > 0.5
    assert!(acc > 0.5, "accuracy {acc} <= 0.5");
}

// ═════════════════════════════════════════════════════════════════
// Regressors (synthetic y = x0 + 2*x1 + noise, R² > 0.8)
// ═════════════════════════════════════════════════════════════════

#[test]
fn regressor_linear() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = LinearRegression::new();
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("LinearRegression R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_ridge() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = Ridge::new(1.0);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("Ridge R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_lasso() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = LassoRegression::new().alpha(0.1).max_iter(5000);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("LassoRegression R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_elastic_net() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = ElasticNet::new().alpha(0.1).l1_ratio(0.5).max_iter(5000);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("ElasticNet R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_decision_tree() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = DecisionTreeRegressor::new();
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("DecisionTreeRegressor R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_random_forest() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = RandomForestRegressor::new().n_estimators(10).seed(42);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("RandomForestRegressor R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_gradient_boosting() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = GradientBoostingRegressor::new()
        .n_estimators(20)
        .learning_rate(0.1)
        .max_depth(3);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("GradientBoostingRegressor R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_hist_gradient_boosting() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = HistGradientBoostingRegressor::new()
        .n_estimators(20)
        .learning_rate(0.1);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("HistGradientBoostingRegressor R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_knn() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = KnnRegressor::new().k(5);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("KnnRegressor R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_mlp() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = MLPRegressor::new()
        .hidden_layers(&[10])
        .learning_rate(0.01)
        .max_iter(500)
        .seed(42);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    let mse = mean_squared_error(&test.target, &preds);
    println!("MLPRegressor R²: {r2:.3}, MSE: {mse:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[test]
fn regressor_linear_svr() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = LinearSVR::new().c(10.0).max_iter(2000);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("LinearSVR R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

#[cfg(feature = "experimental")]
#[test]
#[ignore] // Kernel SVM is O(n²); too slow in debug mode — run with --release --ignored
fn regressor_kernel_svr() {
    let (train, test) = train_test_split(&regression_dataset(), 0.2, 42);
    let mut model = KernelSVR::new()
        .kernel(Kernel::RBF { gamma: 0.1 })
        .c(10.0)
        .max_iter(100);
    model.fit(&train).unwrap();
    let preds = model.predict(&test.feature_matrix()).unwrap();
    let r2 = r2_score(&test.target, &preds);
    println!("KernelSVR R²: {r2:.3}");
    assert!(r2 > 0.8, "R² {r2} <= 0.8");
}

// ═════════════════════════════════════════════════════════════════
// Clustering (Iris features, silhouette > 0.4)
// ═════════════════════════════════════════════════════════════════

#[test]
fn cluster_kmeans() {
    let data = iris_dataset();
    let mut model = KMeans::new(3).seed(42).max_iter(100);
    model.fit(&data).unwrap();
    let labels = model.labels();
    let features = data.feature_matrix();
    let sil = silhouette_score(&features, labels);
    println!("KMeans silhouette: {sil:.3}");
    assert!(sil > 0.4, "silhouette {sil} <= 0.4");
}

#[test]
fn cluster_mini_batch_kmeans() {
    let data = iris_dataset();
    let mut model = MiniBatchKMeans::new(3).seed(42).batch_size(20);
    model.fit(&data).unwrap();
    let labels = model.labels();
    let features = data.feature_matrix();
    let sil = silhouette_score(&features, labels);
    println!("MiniBatchKMeans silhouette: {sil:.3}");
    assert!(sil > 0.4, "silhouette {sil} <= 0.4");
}

// ═════════════════════════════════════════════════════════════════
// Preprocessing round-trips
// ═════════════════════════════════════════════════════════════════

#[test]
fn preprocess_standard_scaler() {
    let data = iris_dataset();
    let (mut train, _) = train_test_split(&data, 0.2, 42);
    let mut scaler = StandardScaler::new();
    scaler.fit(&train).unwrap();
    scaler.transform(&mut train).unwrap();

    // After transform, each feature column should have mean ≈ 0, std ≈ 1
    for col_idx in 0..train.n_features() {
        let col = &train.features[col_idx];
        let mean = col.iter().sum::<f64>() / col.len() as f64;
        let std = (col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64).sqrt();
        assert!(mean.abs() < 1e-10, "feature {col_idx} mean {mean} not ≈ 0");
        assert!(
            (std - 1.0).abs() < 0.1,
            "feature {col_idx} std {std} not ≈ 1"
        );
    }
}

#[test]
fn preprocess_minmax_scaler() {
    let data = iris_dataset();
    let (mut train, _) = train_test_split(&data, 0.2, 42);
    let mut scaler = MinMaxScaler::new();
    scaler.fit(&train).unwrap();
    scaler.transform(&mut train).unwrap();

    for col_idx in 0..train.n_features() {
        let col = &train.features[col_idx];
        let min = col.iter().copied().fold(f64::INFINITY, f64::min);
        let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(min >= -1e-10, "feature {col_idx} min {min} < 0");
        assert!(max <= 1.0 + 1e-10, "feature {col_idx} max {max} > 1");
    }
}

#[test]
fn preprocess_pca_reduce_dims() {
    let data = iris_dataset();
    let mut pca = Pca::with_n_components(2);
    pca.fit(&data).unwrap();
    let mut reduced = data.clone();
    pca.transform(&mut reduced).unwrap();

    assert_eq!(
        reduced.n_features(),
        2,
        "PCA should reduce to 2 components, got {}",
        reduced.n_features()
    );
    assert_eq!(reduced.n_samples(), data.n_samples());
}

// ═════════════════════════════════════════════════════════════════
// Pipeline: StandardScaler + RandomForestClassifier
// ═════════════════════════════════════════════════════════════════

#[test]
fn pipeline_scaler_rf() {
    let (train, test) = train_test_split(&iris_dataset(), 0.2, 42);
    let mut pipeline = Pipeline::new()
        .add_transformer(StandardScaler::new())
        .set_model(RandomForestClassifier::new().n_estimators(10).seed(42));

    pipeline.fit(&train).unwrap();
    let preds = pipeline.predict(&test).unwrap();
    let acc = accuracy(&test.target, &preds);
    println!("Pipeline(StandardScaler + RF) accuracy: {acc:.3}");
    assert!(acc > 0.8, "accuracy {acc} <= 0.8");
}
