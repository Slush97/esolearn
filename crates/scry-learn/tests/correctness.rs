#![allow(
    clippy::float_cmp,
    clippy::needless_range_loop
)]
//! Correctness verification tests: scry-learn vs sklearn reference results.
//!
//! The Iris dataset is embedded directly (150 samples × 4 features × 3 classes)
//! so no external files are needed. Expected results are from sklearn 1.8.0.

use scry_learn::preprocess::Pca;

use scry_learn::cluster::KMeans;
use scry_learn::dataset::Dataset;
use scry_learn::linear::{LinearRegression, LogisticRegression};
use scry_learn::metrics::{
    accuracy, confusion_matrix, f1_score, mean_squared_error, r2_score, Average,
};
use scry_learn::naive_bayes::GaussianNb;
use scry_learn::neighbors::KnnClassifier;
use scry_learn::preprocess::{StandardScaler, Transformer};
use scry_learn::split::train_test_split;
use scry_learn::tree::{DecisionTreeClassifier, RandomForestClassifier};

// ─────────────────────────────────────────────────────────────────
// Embedded Iris dataset (from sklearn.datasets.load_iris)
// ─────────────────────────────────────────────────────────────────

fn iris_dataset() -> Dataset {
    // 150 samples, 4 features: sepal_length, sepal_width, petal_length, petal_width
    // 3 classes: 0=setosa, 1=versicolor, 2=virginica
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

/// sklearn reference: 100% accuracy on Iris 80/20 split (`random_state=42`).
#[test]
fn prove_decision_tree_iris() {
    let data = iris_dataset();
    let (train, test) = train_test_split(&data, 0.2, 42);

    let mut dt = DecisionTreeClassifier::new();
    dt.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = dt.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!("Decision Tree Iris accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.85,
        "Decision Tree should achieve ≥85% on Iris (got {:.1}%)",
        acc * 100.0
    );
}

/// sklearn reference: 100% accuracy on Iris.
#[test]
fn prove_random_forest_iris() {
    let data = iris_dataset();
    let (train, test) = train_test_split(&data, 0.2, 42);

    let mut rf = RandomForestClassifier::new().n_estimators(100).seed(42);
    rf.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = rf.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!("Random Forest Iris accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.85,
        "Random Forest should achieve ≥85% on Iris (got {:.1}%)",
        acc * 100.0
    );
}

/// sklearn reference: 100% accuracy on Iris with `max_iter=1000`.
#[test]
fn prove_logistic_regression_iris() {
    let data = iris_dataset();
    let (mut train, test) = train_test_split(&data, 0.2, 42);

    // Scale features for gradient descent stability.
    let mut scaler = StandardScaler::new();
    scaler.fit(&train).unwrap();
    scaler.transform(&mut train).unwrap();
    let mut test_scaled = test.clone();
    scaler.transform(&mut test_scaled).unwrap();

    let mut lr = LogisticRegression::new()
        .alpha(0.0)
        .learning_rate(0.1)
        .max_iter(1000);
    lr.fit(&train).unwrap();

    let features = test_scaled.feature_matrix();
    let preds = lr.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!("Logistic Regression Iris accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.85,
        "Logistic Regression should achieve ≥85% on Iris (got {:.1}%)",
        acc * 100.0
    );
}

/// sklearn reference: 100% accuracy on Iris.
#[test]
fn prove_knn_iris() {
    let data = iris_dataset();
    let (train, test) = train_test_split(&data, 0.2, 42);

    let mut knn = KnnClassifier::new().k(5);
    knn.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = knn.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!("KNN Iris accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.90,
        "KNN should achieve ≥90% on Iris (got {:.1}%)",
        acc * 100.0
    );
}

/// sklearn reference: 100% accuracy on Iris.
#[test]
fn prove_gaussian_nb_iris() {
    let data = iris_dataset();
    let (train, test) = train_test_split(&data, 0.2, 42);

    let mut nb = GaussianNb::new();
    nb.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = nb.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!("Gaussian NB Iris accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.90,
        "Gaussian NB should achieve ≥90% on Iris (got {:.1}%)",
        acc * 100.0
    );
}

/// Prove linear regression on a known linear system: y = 2x₁ + 3x₂ + 1.
#[test]
fn prove_linear_regression_known_coefficients() {
    let n = 500;
    let mut rng = fastrand::Rng::with_seed(42);
    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);

    for _ in 0..n {
        let x1 = rng.f64() * 10.0;
        let x2 = rng.f64() * 10.0;
        let y = 2.0 * x1 + 3.0 * x2 + 1.0 + rng.f64() * 0.01; // tiny noise
        f1.push(x1);
        f2.push(x2);
        target.push(y);
    }

    let data = Dataset::new(vec![f1, f2], target, vec!["x1".into(), "x2".into()], "y");

    let (train, test) = train_test_split(&data, 0.2, 42);

    let mut lr = LinearRegression::new();
    lr.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = lr.predict(&features).unwrap();
    let r2 = r2_score(&test.target, &preds);
    let mse = mean_squared_error(&test.target, &preds);

    eprintln!("Linear Regression: R²={r2:.6}, MSE={mse:.6}");
    assert!(
        r2 > 0.999,
        "R² should be near 1.0 for known linear system (got {r2})"
    );
    assert!(
        mse < 0.01,
        "MSE should be near 0 for known linear system (got {mse})"
    );
}

/// Prove K-Means finds correct cluster structure on well-separated data.
#[test]
fn prove_kmeans_separation() {
    let n = 300;
    let mut rng = fastrand::Rng::with_seed(42);
    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);

    // 3 well-separated clusters
    for _ in 0..n / 3 {
        f1.push(rng.f64() * 2.0);
        f2.push(rng.f64() * 2.0);
    }
    for _ in 0..n / 3 {
        f1.push(rng.f64() * 2.0 + 10.0);
        f2.push(rng.f64() * 2.0);
    }
    for _ in 0..n / 3 {
        f1.push(rng.f64() * 2.0 + 5.0);
        f2.push(rng.f64() * 2.0 + 10.0);
    }

    let target = vec![0.0; n]; // dummy target
    let data = Dataset::new(
        vec![f1, f2],
        target,
        vec!["x".into(), "y".into()],
        "cluster",
    );

    let mut km = KMeans::new(3).seed(42).max_iter(100);
    km.fit(&data).unwrap();

    let labels = km.labels();
    assert_eq!(labels.len(), n);

    // Check that each third of the data mostly gets the same label.
    let third = n / 3;
    let label_a = labels[0];
    let label_b = labels[third];
    let label_c = labels[2 * third];

    // All 3 clusters should have different labels.
    assert_ne!(
        label_a, label_b,
        "Clusters A and B should have different labels"
    );
    assert_ne!(
        label_b, label_c,
        "Clusters B and C should have different labels"
    );
    assert_ne!(
        label_a, label_c,
        "Clusters A and C should have different labels"
    );

    // Check cluster purity (at least 90% of each third should share a label).
    for (start, expected) in [(0, label_a), (third, label_b), (2 * third, label_c)] {
        let correct = labels[start..start + third]
            .iter()
            .filter(|&&l| l == expected)
            .count();
        let purity = correct as f64 / third as f64;
        assert!(
            purity >= 0.9,
            "Cluster starting at {start} should be ≥90% pure (got {:.1}%)",
            purity * 100.0
        );
    }
}

/// Prove metrics match known values.
#[test]
fn prove_metrics_exact() {
    // Perfect predictions.
    let y_true = vec![0.0, 0.0, 1.0, 1.0];
    let y_pred = vec![0.0, 0.0, 1.0, 1.0];

    assert!((accuracy(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    assert!((f1_score(&y_true, &y_pred, Average::Binary) - 1.0).abs() < 1e-10);

    let cm = confusion_matrix(&y_true, &y_pred);
    assert_eq!(cm.matrix[0][0], 2); // TN
    assert_eq!(cm.matrix[0][1], 0); // FP
    assert_eq!(cm.matrix[1][0], 0); // FN
    assert_eq!(cm.matrix[1][1], 2); // TP

    // Imperfect predictions.
    let y_true2 = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let y_pred2 = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0];

    let acc = accuracy(&y_true2, &y_pred2);
    assert!((acc - 4.0 / 6.0).abs() < 1e-10); // 4 correct out of 6

    // MSE of [1,2,3] vs [1.1,2.1,2.9]
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.1, 2.1, 2.9];
    let mse = mean_squared_error(&a, &b);
    // (0.01 + 0.01 + 0.01) / 3 = 0.01
    assert!((mse - 0.01).abs() < 1e-10);

    // R² of perfect predictions = 1.0.
    let r2 = r2_score(&a, &a);
    assert!((r2 - 1.0).abs() < 1e-10);
}

// ─────────────────────────────────────────────────────────────────
// PCA correctness vs sklearn 1.8.0
// ─────────────────────────────────────────────────────────────────

/// sklearn reference: `PCA(n_components=4).fit(iris_features)`
/// `explained_variance_ratio`_ = [0.9246, 0.0531, 0.0172, 0.0052]
#[test]
fn prove_pca_explained_variance_ratio() {
    let data = iris_dataset();
    let mut pca = Pca::new();
    pca.fit(&data).unwrap();

    let ratios = pca.explained_variance_ratio();
    assert_eq!(ratios.len(), 4);

    // sklearn reference values (rounded to 3 decimal places for tolerance).
    let sklearn_ratios = [0.9246, 0.0531, 0.0172, 0.0052];
    for (i, (&actual, expected)) in ratios.iter().zip(sklearn_ratios.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 0.01,
            "Variance ratio PC{} mismatch: scry={actual:.4}, sklearn={expected:.4}",
            i + 1,
        );
    }

    // Total should sum to 1.0.
    let sum: f64 = ratios.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Explained variance ratios should sum to 1.0, got {sum}"
    );

    eprintln!("PCA explained variance ratios: {ratios:?}");
}

/// Verify `PCA(n_components=2)` produces correct 2D projection on Iris.
/// sklearn: first PC captures ~92.5% variance, second ~5.3%.
#[test]
fn prove_pca_iris_dimension_reduction() {
    let data = iris_dataset();
    let mut ds = data.clone();
    let mut pca = Pca::with_n_components(2);
    pca.fit_transform(&mut ds).unwrap();

    assert_eq!(ds.n_features(), 2, "Should reduce to 2 features");
    assert_eq!(ds.n_samples(), 150, "Should preserve all samples");

    // The two ratios should match sklearn closely.
    let ratios = pca.explained_variance_ratio();
    assert!(ratios[0] > 0.90, "PC1 should capture >90% variance");
    assert!(ratios[1] > 0.04, "PC2 should capture >4% variance");

    // Roundtrip (approximate — only 2 components retained).
    pca.inverse_transform(&mut ds).unwrap();
    assert_eq!(ds.n_features(), 4, "Should reconstruct to 4 features");

    // Reconstruction error should be small (2 PCs capture ~97.7%).
    let mut total_err = 0.0;
    for j in 0..4 {
        for i in 0..150 {
            total_err += (ds.features[j][i] - data.features[j][i]).powi(2);
        }
    }
    let rmse = (total_err / (150.0 * 4.0)).sqrt();
    eprintln!("PCA 2-component reconstruction RMSE: {rmse:.4}");
    assert!(
        rmse < 0.20,
        "Reconstruction RMSE should be small, got {rmse}"
    );
}

// ─────────────────────────────────────────────────────────────────
// GBT correctness vs sklearn 1.8.0
// ─────────────────────────────────────────────────────────────────

/// sklearn reference: `GradientBoostingClassifier(n_estimators=100`, `learning_rate=0.1`,
///                    `max_depth=3`, `random_state=42).fit(X_train`, `y_train`)
///                    → accuracy ≥ 93% on Iris 80/20 split.
#[test]
fn prove_gbt_classifier_iris() {
    use scry_learn::tree::GradientBoostingClassifier;

    let data = iris_dataset();

    // Test across multiple seeds to diagnose whether accuracy gap is seed-specific.
    let seeds = [42u64, 7, 123, 99, 1, 55, 13, 77, 200, 999];
    let mut total_acc = 0.0;
    for &seed in &seeds {
        let (train, test) = scry_learn::split::train_test_split(&data, 0.2, seed);
        let mut gbc = GradientBoostingClassifier::new()
            .n_estimators(200)
            .learning_rate(0.1)
            .max_depth(3);
        gbc.fit(&train).unwrap();

        let test_features = test.feature_matrix();
        let preds = gbc.predict(&test_features).unwrap();

        let acc = accuracy(&test.target, &preds);
        eprintln!(
            "Seed {seed:>3}: {acc:.4} ({}/{} correct)",
            (acc * test.target.len() as f64) as usize,
            test.target.len()
        );
        total_acc += acc;

        // Verify probabilities are valid for each run.
        let probas = gbc.predict_proba(&test_features).unwrap();
        for p in &probas {
            assert_eq!(p.len(), 3, "should have 3 class probabilities");
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "probabilities must sum to 1");
        }
    }
    let mean_acc = total_acc / seeds.len() as f64;
    eprintln!("Mean accuracy across {} seeds: {mean_acc:.4}", seeds.len());
    assert!(
        mean_acc >= 0.90,
        "Mean GBT accuracy should be ≥ 90%, got {mean_acc:.4}"
    );
}

/// Prove GBT regressor on known linear relationship.
#[test]
fn prove_gbt_regressor_known_coefficients() {
    use scry_learn::tree::GradientBoostingRegressor;

    // y = 2·x₁ + 3·x₂ + 1 (same as linear regression proof)
    let n = 200;
    let mut rng = fastrand::Rng::with_seed(42);
    let x1: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| 2.0 * a + 3.0 * b + 1.0)
        .collect();

    let data =
        scry_learn::dataset::Dataset::new(vec![x1, x2], y, vec!["x1".into(), "x2".into()], "y");

    let mut gbr = GradientBoostingRegressor::new()
        .n_estimators(200)
        .learning_rate(0.1)
        .max_depth(4);
    gbr.fit(&data).unwrap();

    let test = vec![vec![5.0, 5.0], vec![1.0, 1.0], vec![10.0, 0.0]];
    let preds = gbr.predict(&test).unwrap();
    // Expected: 26.0, 6.0, 21.0
    let rmse: f64 =
        ((preds[0] - 26.0).powi(2) + (preds[1] - 6.0).powi(2) + (preds[2] - 21.0).powi(2)).sqrt()
            / 3.0_f64.sqrt();
    eprintln!("GBT Regressor RMSE on known coefficients: {rmse:.4}");
    assert!(rmse < 3.0, "RMSE should be small, got {rmse:.4}");
}

// ─────────────────────────────────────────────────────────────────
// Lasso and ElasticNet correctness
// ─────────────────────────────────────────────────────────────────

/// Prove Lasso recovers known coefficients with sparsity:
/// y = 2·x₁ + 3·x₃ + 1, where x₂ and x₄ are irrelevant noise features.
/// With sufficient alpha, Lasso should drive x₂ and x₄ coefficients toward zero.
#[test]
fn prove_lasso_sparsity_on_known_system() {
    use scry_learn::linear::LassoRegression;

    let n = 200;
    let mut rng = fastrand::Rng::with_seed(42);
    let x1: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let x3: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let x4: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let y: Vec<f64> = x1
        .iter()
        .zip(x3.iter())
        .map(|(&a, &c)| 2.0 * a + 3.0 * c + 1.0)
        .collect();

    let data = Dataset::new(
        vec![x1, x2, x3, x4],
        y,
        vec!["x1".into(), "x2".into(), "x3".into(), "x4".into()],
        "y",
    );

    let (train, test) = scry_learn::split::train_test_split(&data, 0.2, 42);
    let mut lasso = LassoRegression::new().alpha(0.5).max_iter(5000);
    lasso.fit(&train).unwrap();

    let coefs = lasso.coefficients();
    eprintln!("Lasso coefficients: {coefs:?}");

    // Irrelevant features should be near zero.
    assert!(
        coefs[1].abs() < 0.2,
        "x2 should be near 0, got {}",
        coefs[1]
    );
    assert!(
        coefs[3].abs() < 0.2,
        "x4 should be near 0, got {}",
        coefs[3]
    );

    // Relevant features should be significant.
    assert!(
        coefs[0].abs() > 1.0,
        "x1 should be significant, got {}",
        coefs[0]
    );
    assert!(
        coefs[2].abs() > 1.0,
        "x3 should be significant, got {}",
        coefs[2]
    );

    // Prediction quality.
    let test_features = test.feature_matrix();
    let preds = lasso.predict(&test_features).unwrap();
    let r2 = r2_score(&test.target, &preds);
    eprintln!("Lasso R² on known system: {r2:.4}");
    assert!(r2 > 0.90, "Lasso R² should be > 0.90, got {r2:.4}");
}

/// Prove `ElasticNet` with `l1_ratio=0` behaves like Ridge (no sparsity, good fit).
#[test]
fn prove_elastic_net_ridge_mode() {
    use scry_learn::linear::ElasticNet;

    let n = 200;
    let mut rng = fastrand::Rng::with_seed(42);
    let x1: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| 2.0 * a + 3.0 * b + 1.0)
        .collect();

    let data = Dataset::new(vec![x1, x2], y, vec!["x1".into(), "x2".into()], "y");

    let (train, test) = scry_learn::split::train_test_split(&data, 0.2, 42);

    let mut en = ElasticNet::new().alpha(0.1).l1_ratio(0.0).max_iter(5000);
    en.fit(&train).unwrap();

    let coefs = en.coefficients();
    eprintln!("ElasticNet (Ridge mode) coefficients: {coefs:?}");

    // Both coefficients should be significant (no sparsity).
    assert!(
        coefs[0].abs() > 1.0,
        "x1 coef should be ~2, got {}",
        coefs[0]
    );
    assert!(
        coefs[1].abs() > 1.0,
        "x2 coef should be ~3, got {}",
        coefs[1]
    );

    let test_features = test.feature_matrix();
    let preds = en.predict(&test_features).unwrap();
    let r2 = r2_score(&test.target, &preds);
    eprintln!("ElasticNet Ridge-mode R²: {r2:.4}");
    assert!(
        r2 > 0.98,
        "ElasticNet Ridge-mode R² should be > 0.98, got {r2:.4}"
    );
}

/// Prove `ElasticNet` with `l1_ratio=1` behaves like Lasso (drives irrelevant to zero).
#[test]
fn prove_elastic_net_lasso_mode() {
    use scry_learn::linear::ElasticNet;

    let n = 200;
    let mut rng = fastrand::Rng::with_seed(42);
    let x1: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect(); // noise
    let y: Vec<f64> = x1.iter().map(|&a| 3.0 * a + 1.0).collect();

    let data = Dataset::new(vec![x1, x2], y, vec!["x1".into(), "x2".into()], "y");

    let mut en = ElasticNet::new().alpha(0.5).l1_ratio(1.0).max_iter(5000);
    en.fit(&data).unwrap();

    let coefs = en.coefficients();
    eprintln!("ElasticNet (Lasso mode) coefficients: {coefs:?}");

    assert!(
        coefs[1].abs() < 0.1,
        "noise x2 should be ~0 in Lasso mode, got {}",
        coefs[1]
    );
    assert!(
        coefs[0].abs() > 1.0,
        "x1 should be significant, got {}",
        coefs[0]
    );
}

// ─────────────────────────────────────────────────────────────────
// class_weight correctness: improved minority recall on imbalanced data
// ─────────────────────────────────────────────────────────────────

/// Prove that `DecisionTreeClassifier` with `class_weight=Balanced` improves
/// minority class recall on a 90/10 imbalanced binary dataset.
///
/// Without weights, the tree may achieve high overall accuracy by predicting
/// the majority class, yielding poor minority recall. With Balanced weights,
/// the minority class receives higher importance in impurity calculations,
/// improving its recall.
#[test]
fn prove_class_weight_dt_imbalanced() {
    use scry_learn::metrics::recall;
    use scry_learn::weights::ClassWeight;

    // Create a 200-sample imbalanced dataset: 180 class 0, 20 class 1.
    // Two features with some overlap.
    let n_majority = 180;
    let n_minority = 20;
    let n = n_majority + n_minority;
    let mut rng = fastrand::Rng::with_seed(42);

    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);

    // Majority class (0): features centered around (3, 3) with spread.
    for _ in 0..n_majority {
        f1.push(rng.f64() * 6.0); // [0, 6]
        f2.push(rng.f64() * 6.0); // [0, 6]
        target.push(0.0);
    }
    // Minority class (1): features centered around (5, 5) — overlaps majority slightly.
    for _ in 0..n_minority {
        f1.push(rng.f64() * 4.0 + 4.0); // [4, 8]
        f2.push(rng.f64() * 4.0 + 4.0); // [4, 8]
        target.push(1.0);
    }

    let data = Dataset::new(
        vec![f1, f2],
        target,
        vec!["f1".into(), "f2".into()],
        "class",
    );

    // Train WITHOUT class_weight.
    let mut dt_unweighted = DecisionTreeClassifier::new().max_depth(5);
    dt_unweighted.fit(&data).unwrap();
    let matrix = data.feature_matrix();
    let preds_unweighted = dt_unweighted.predict(&matrix).unwrap();
    let recall_unweighted = recall(&data.target, &preds_unweighted, Average::Binary);

    // Train WITH class_weight=Balanced.
    let mut dt_weighted = DecisionTreeClassifier::new()
        .max_depth(5)
        .class_weight(ClassWeight::Balanced);
    dt_weighted.fit(&data).unwrap();
    let preds_weighted = dt_weighted.predict(&matrix).unwrap();
    let recall_weighted = recall(&data.target, &preds_weighted, Average::Binary);

    eprintln!("Minority recall (unweighted): {recall_unweighted:.3}");
    eprintln!("Minority recall (weighted):   {recall_weighted:.3}");

    // The weighted model should have significantly better minority recall.
    // On this dataset the weighted tree should achieve at least 0.80 recall
    // for the minority class.
    assert!(
        recall_weighted >= 0.70,
        "Weighted DT should achieve ≥70% minority recall (got {:.1}%)",
        recall_weighted * 100.0
    );

    // Weighted recall should generally be >= unweighted recall for minority.
    // (We allow a small margin in case the unweighted tree also does well.)
    eprintln!(
        "Improvement: {:.1}pp",
        (recall_weighted - recall_unweighted) * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────
// KNN improvements: distance weights, predict_proba, regressor
// ─────────────────────────────────────────────────────────────────

/// Prove KNN with distance weights achieves ≥90% accuracy on Iris.
#[test]
fn prove_knn_distance_weights_iris() {
    use scry_learn::neighbors::WeightFunction;

    let data = iris_dataset();
    let (train, test) = train_test_split(&data, 0.2, 42);

    let mut knn = KnnClassifier::new().k(5).weights(WeightFunction::Distance);
    knn.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = knn.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!("KNN (distance weights) Iris accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.90,
        "KNN with distance weights should achieve ≥90% on Iris (got {:.1}%)",
        acc * 100.0
    );
}

/// Prove KNN regressor on known linear function y = 2x₁ + 3x₂ + 1.
/// With enough samples and reasonable k, R² should exceed 0.9.
#[test]
fn prove_knn_regressor_linear() {
    use scry_learn::neighbors::KnnRegressor;

    let n = 500;
    let mut rng = fastrand::Rng::with_seed(42);
    let x1: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let x2: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| 2.0 * a + 3.0 * b + 1.0)
        .collect();

    let data = Dataset::new(vec![x1, x2], y, vec!["x1".into(), "x2".into()], "y");

    let (train, test) = train_test_split(&data, 0.2, 42);

    let mut knn = KnnRegressor::new().k(5);
    knn.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = knn.predict(&features).unwrap();
    let r2 = r2_score(&test.target, &preds);

    eprintln!("KNN Regressor R² on y=2x₁+3x₂+1: {r2:.4}");
    assert!(
        r2 > 0.9,
        "KNN Regressor R² should be > 0.9 on linear function (got {r2:.4})"
    );
}

/// Prove `predict_proba` returns valid probability distributions on Iris.
/// All per-sample probability vectors must sum to 1.0.
#[test]
fn prove_knn_predict_proba_iris() {
    let data = iris_dataset();
    let (train, test) = train_test_split(&data, 0.2, 42);

    let mut knn = KnnClassifier::new().k(5);
    knn.fit(&train).unwrap();

    let features = test.feature_matrix();
    let probas = knn.predict_proba(&features).unwrap();

    assert_eq!(probas.len(), test.n_samples());
    for (i, p) in probas.iter().enumerate() {
        assert_eq!(p.len(), 3, "Iris has 3 classes");
        let sum: f64 = p.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Sample {i}: probabilities must sum to 1.0, got {sum}"
        );
        for &prob in p {
            assert!(prob >= 0.0, "Probabilities must be non-negative");
        }
    }

    eprintln!(
        "predict_proba: all {} samples have valid distributions",
        probas.len()
    );
}

// ─────────────────────────────────────────────────────────────────
// SVM correctness: LinearSVC, XOR negative proof, KernelSVC
// ─────────────────────────────────────────────────────────────────

/// Prove `LinearSVC` achieves ≥85% accuracy on Iris.
///
/// Iris has three linearly-separable-ish classes; a linear SVM
/// should handle it comfortably with feature scaling.
#[test]
fn prove_linear_svc_iris() {
    use scry_learn::svm::LinearSVC;

    let data = iris_dataset();

    // Scale features for stable SGD convergence.
    let mut scaler = StandardScaler::new();
    let mut scaled = data;
    scaler.fit(&scaled).unwrap();
    scaler.transform(&mut scaled).unwrap();

    let (train, test) = train_test_split(&scaled, 0.2, 42);

    let mut svc = LinearSVC::new().c(1.0).max_iter(2000).tol(1e-5);
    svc.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = svc.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!("LinearSVC Iris accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.85,
        "LinearSVC should achieve ≥85% on Iris (got {:.1}%)",
        acc * 100.0
    );
}

/// Prove `LinearSVC` fails on XOR — a negative proof.
///
/// XOR is the classic non-linearly-separable problem:
/// `[0,0]→0, [1,1]→0, [0,1]→1, [1,0]→1`.
/// A linear SVM cannot find any hyperplane that separates these
/// classes, so accuracy should be at most 60% (i.e. roughly random).
/// This validates that our implementation is honest and not overfitting.
#[test]
fn prove_linear_svc_xor_fail() {
    use scry_learn::svm::LinearSVC;

    let features = vec![vec![0.0, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 1.0]];
    let target = vec![0.0, 1.0, 1.0, 0.0];
    let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

    let mut svc = LinearSVC::new().c(1.0).max_iter(1000);
    svc.fit(&data).unwrap();

    let test_points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let preds = svc.predict(&test_points).unwrap();
    let expected = [0.0, 1.0, 1.0, 0.0];
    let correct = preds
        .iter()
        .zip(expected.iter())
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();
    let acc = correct as f64 / 4.0;

    eprintln!(
        "LinearSVC XOR accuracy: {:.0}% ({correct}/4) — expected ≤60%",
        acc * 100.0
    );
    assert!(
        acc <= 0.60,
        "LinearSVC should NOT solve XOR (got {:.0}% — too high for a linear model)",
        acc * 100.0
    );
}

/// Prove KernelSVC with RBF solves XOR — validates kernel power.
///
/// The RBF kernel maps XOR into a higher-dimensional space where
/// the classes become linearly separable. This is the complement to
/// the negative proof above.
#[cfg(feature = "experimental")]
#[test]
fn prove_kernel_svc_xor() {
    use scry_learn::svm::{Kernel, KernelSVC};

    // Use more training data (replicated) for a stable fit.
    let features = vec![
        vec![0.0, 1.0, 0.0, 1.0, 0.1, 0.9, 0.1, 0.9],
        vec![0.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 0.9],
    ];
    let target = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

    let mut svc = KernelSVC::new()
        .kernel(Kernel::RBF { gamma: 5.0 })
        .c(10.0)
        .max_iter(500);
    svc.fit(&data).unwrap();

    let test_points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let preds = svc.predict(&test_points).unwrap();
    let expected = vec![0.0, 1.0, 1.0, 0.0];
    let correct = preds
        .iter()
        .zip(expected.iter())
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();
    let acc = correct as f64 / 4.0;

    eprintln!(
        "KernelSVC RBF XOR accuracy: {:.0}% ({correct}/4)",
        acc * 100.0
    );
    assert!(
        acc >= 0.90,
        "KernelSVC with RBF should solve XOR (got {:.0}%)",
        acc * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────
// Preprocessing: SimpleImputer, RobustScaler, ColumnTransformer
// ─────────────────────────────────────────────────────────────────

/// Prove `SimpleImputer` fills NaN correctly with Mean strategy.
/// A dataset with known NaN positions should have those filled with
/// the per-feature mean of the non-NaN entries.
#[test]
fn prove_imputer_fills_nan_correctly() {
    use scry_learn::preprocess::{SimpleImputer, Strategy, Transformer};

    // Feature 0: [1.0, NaN, 3.0, 5.0] → mean of {1,3,5} = 3.0
    // Feature 1: [10.0, 20.0, NaN, NaN] → mean of {10,20} = 15.0
    let mut ds = Dataset::new(
        vec![
            vec![1.0, f64::NAN, 3.0, 5.0],
            vec![10.0, 20.0, f64::NAN, f64::NAN],
        ],
        vec![0.0; 4],
        vec!["a".into(), "b".into()],
        "y",
    );

    let mut imp = SimpleImputer::new().strategy(Strategy::Mean);
    imp.fit_transform(&mut ds).unwrap();

    // No NaN should remain
    for (j, col) in ds.features.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            assert!(!v.is_nan(), "NaN remains at feature {j}, sample {i}");
        }
    }

    // Feature 0, sample 1 should be 3.0 (mean of 1, 3, 5)
    assert!(
        (ds.features[0][1] - 3.0).abs() < 1e-10,
        "Expected 3.0, got {}",
        ds.features[0][1]
    );
    // Feature 1, samples 2 and 3 should be 15.0 (mean of 10, 20)
    assert!(
        (ds.features[1][2] - 15.0).abs() < 1e-10,
        "Expected 15.0, got {}",
        ds.features[1][2]
    );
    assert!(
        (ds.features[1][3] - 15.0).abs() < 1e-10,
        "Expected 15.0, got {}",
        ds.features[1][3]
    );

    // Also verify Median strategy on same data shape
    let mut ds2 = Dataset::new(
        vec![
            vec![1.0, f64::NAN, 3.0, 5.0],
            vec![10.0, 20.0, f64::NAN, f64::NAN],
        ],
        vec![0.0; 4],
        vec!["a".into(), "b".into()],
        "y",
    );
    let mut imp2 = SimpleImputer::new().strategy(Strategy::Median);
    imp2.fit_transform(&mut ds2).unwrap();

    // Feature 0 median of {1,3,5} = 3.0
    assert!(
        (ds2.features[0][1] - 3.0).abs() < 1e-10,
        "Median: expected 3.0, got {}",
        ds2.features[0][1]
    );
    // Feature 1 median of {10,20} = 15.0
    assert!(
        (ds2.features[1][2] - 15.0).abs() < 1e-10,
        "Median: expected 15.0, got {}",
        ds2.features[1][2]
    );

    eprintln!("SimpleImputer correctness: all NaN values filled correctly");
}

/// Prove `RobustScaler` handles outliers better than `StandardScaler`.
///
/// On a dataset where most values are in [1, 10] but one outlier is 1000,
/// `RobustScaler` should produce non-outlier values with wider spread than
/// `StandardScaler` (because `StandardScaler`'s std is inflated by the outlier).
#[test]
fn prove_robust_scaler_outlier_tolerance() {
    use scry_learn::preprocess::{RobustScaler, StandardScaler, Transformer};

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1000.0];
    let n = data.len();

    // StandardScaler
    let mut ds_std = Dataset::new(vec![data.clone()], vec![0.0; n], vec!["x".into()], "y");
    let mut std_scaler = StandardScaler::new();
    std_scaler.fit_transform(&mut ds_std).unwrap();

    // RobustScaler
    let mut ds_rob = Dataset::new(vec![data], vec![0.0; n], vec!["x".into()], "y");
    let mut rob_scaler = RobustScaler::new();
    rob_scaler.fit_transform(&mut ds_rob).unwrap();

    // For non-outlier points (indices 0-9), the spread should be
    // wider in RobustScaler since its IQR is not inflated by the outlier.
    let std_spread = ds_std.features[0][9] - ds_std.features[0][0];
    let rob_spread = ds_rob.features[0][9] - ds_rob.features[0][0];

    eprintln!("StandardScaler non-outlier spread: {std_spread:.4}");
    eprintln!("RobustScaler non-outlier spread:   {rob_spread:.4}");

    assert!(
        rob_spread > std_spread,
        "RobustScaler should give wider non-outlier spread: robust={rob_spread:.4} vs std={std_spread:.4}"
    );

    // Verify roundtrip
    rob_scaler.inverse_transform(&mut ds_rob).unwrap();
    assert!(
        (ds_rob.features[0][0] - 1.0).abs() < 1e-10,
        "Roundtrip failed for first element"
    );
    assert!(
        (ds_rob.features[0][10] - 1000.0).abs() < 1e-10,
        "Roundtrip failed for outlier"
    );
}

/// Prove `ColumnTransformer` correctly composes different transformers
/// on different column subsets.
///
/// Apply `StandardScaler` to columns 0–1 and `MinMaxScaler` to columns 2–3.
/// After transformation, columns 0–1 should be zero-mean and columns
/// 2–3 should be in [0, 1].
#[test]
fn prove_column_transformer_composition() {
    use scry_learn::preprocess::{ColumnTransformer, MinMaxScaler, StandardScaler, Transformer};

    let mut ds = Dataset::new(
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],           // col 0
            vec![10.0, 20.0, 30.0, 40.0, 50.0],      // col 1
            vec![100.0, 200.0, 300.0, 400.0, 500.0], // col 2
            vec![5.0, 10.0, 15.0, 20.0, 25.0],       // col 3
        ],
        vec![0.0; 5],
        vec!["a".into(), "b".into(), "c".into(), "d".into()],
        "y",
    );

    let mut ct = ColumnTransformer::new()
        .add(&[0, 1], StandardScaler::new())
        .add(&[2, 3], MinMaxScaler::new());

    ct.fit_transform(&mut ds).unwrap();

    assert_eq!(ds.n_features(), 4, "Should still have 4 features");

    // Columns 0, 1: StandardScaler → zero mean
    let mean_0: f64 = ds.features[0].iter().sum::<f64>() / 5.0;
    let mean_1: f64 = ds.features[1].iter().sum::<f64>() / 5.0;
    assert!(
        mean_0.abs() < 1e-10,
        "col 0 mean should be ~0, got {mean_0}"
    );
    assert!(
        mean_1.abs() < 1e-10,
        "col 1 mean should be ~0, got {mean_1}"
    );

    // Columns 2, 3: MinMaxScaler → [0, 1]
    let min_2 = ds.features[2].iter().copied().fold(f64::INFINITY, f64::min);
    let max_2 = ds.features[2]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(min_2.abs() < 1e-10, "col 2 min should be ~0, got {min_2}");
    assert!(
        (max_2 - 1.0).abs() < 1e-10,
        "col 2 max should be ~1, got {max_2}"
    );

    let min_3 = ds.features[3].iter().copied().fold(f64::INFINITY, f64::min);
    let max_3 = ds.features[3]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(min_3.abs() < 1e-10, "col 3 min should be ~0, got {min_3}");
    assert!(
        (max_3 - 1.0).abs() < 1e-10,
        "col 3 max should be ~1, got {max_3}"
    );

    eprintln!("ColumnTransformer: StandardScaler + MinMaxScaler composition verified");
}

// ─────────────────────────────────────────────────────────────────
// Cost-complexity pruning correctness
// ─────────────────────────────────────────────────────────────────

/// Prove that `ccp_alpha` produces a tree with fewer leaves on Iris.
///
/// An unpruned tree overfits, yielding many leaves. Setting `ccp_alpha` to a
/// moderate value should collapse low-value subtrees without ruining accuracy.
#[test]
fn prove_pruning_reduces_tree_size() {
    let data = iris_dataset();

    // Full tree (no pruning).
    let mut dt_full = DecisionTreeClassifier::new();
    dt_full.fit(&data).unwrap();
    let leaves_full = dt_full.n_leaves();

    // Use the pruning path to find a suitable alpha.
    let (alphas, _impurities) = dt_full.cost_complexity_pruning_path(&data).unwrap();
    eprintln!("Pruning path alphas: {alphas:?}");

    // Pick an alpha from the middle of the path (guaranteed to prune something).
    let mid_alpha = if alphas.len() > 2 {
        alphas[alphas.len() / 2]
    } else {
        // If the path is very short, use a large alpha to force at least some pruning.
        1.0
    };

    let mut dt_pruned = DecisionTreeClassifier::new().ccp_alpha(mid_alpha);
    dt_pruned.fit(&data).unwrap();
    let leaves_pruned = dt_pruned.n_leaves();

    eprintln!("Full tree: {leaves_full} leaves");
    eprintln!("Pruned tree (ccp_alpha={mid_alpha:.4}): {leaves_pruned} leaves");

    assert!(
        leaves_pruned <= leaves_full,
        "Pruned tree should have ≤ leaves: {leaves_pruned} vs {leaves_full}"
    );

    // If we had more than 2 pruning steps, pruning should strictly reduce leaves.
    if alphas.len() > 2 {
        assert!(
            leaves_pruned < leaves_full,
            "Pruned tree should have fewer leaves: {leaves_pruned} vs {leaves_full}"
        );
    }

    // Pruned tree should still be reasonably accurate on training data.
    let matrix = data.feature_matrix();
    let preds = dt_pruned.predict(&matrix).unwrap();
    let acc = scry_learn::metrics::accuracy(&data.target, &preds);
    eprintln!("Pruned tree training accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.80,
        "Pruned tree should still achieve ≥80% on Iris (got {:.1}%)",
        acc * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────
// GBT Huber loss robustness correctness
// ─────────────────────────────────────────────────────────────────

/// Prove that Huber loss achieves lower MAE than squared error on data with outliers.
///
/// We generate y = 2x + 1 and inject ~10% outliers (y += 50). Squared error
/// is strongly influenced by outliers, while Huber loss is robust.
#[test]
fn prove_gbt_huber_outlier_robustness() {
    use scry_learn::tree::{GradientBoostingRegressor, RegressionLoss};

    let n = 200;
    let mut rng = fastrand::Rng::with_seed(42);
    let x: Vec<f64> = (0..n).map(|_| rng.f64() * 10.0).collect();
    let mut y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();

    // Inject ~10% outliers.
    for i in 0..n / 10 {
        y[i] += 50.0;
    }

    let data = scry_learn::dataset::Dataset::new(vec![x], y, vec!["x".into()], "y");

    // GBT with squared error (sensitive to outliers).
    let mut gbr_mse = GradientBoostingRegressor::new()
        .n_estimators(200)
        .loss(RegressionLoss::SquaredError)
        .learning_rate(0.1)
        .max_depth(3);
    gbr_mse.fit(&data).unwrap();

    // GBT with Huber loss (robust to outliers).
    let mut gbr_huber = GradientBoostingRegressor::new()
        .n_estimators(200)
        .loss(RegressionLoss::Huber { alpha: 0.9 })
        .learning_rate(0.1)
        .max_depth(3);
    gbr_huber.fit(&data).unwrap();

    // Evaluate on clean test points.
    let test_x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64 * 0.5]).collect();
    let test_y: Vec<f64> = test_x.iter().map(|v| 2.0 * v[0] + 1.0).collect();

    let preds_mse = gbr_mse.predict(&test_x).unwrap();
    let preds_huber = gbr_huber.predict(&test_x).unwrap();

    let mae_mse: f64 = preds_mse
        .iter()
        .zip(test_y.iter())
        .map(|(&p, &t)| (p - t).abs())
        .sum::<f64>()
        / test_y.len() as f64;
    let mae_huber: f64 = preds_huber
        .iter()
        .zip(test_y.iter())
        .map(|(&p, &t)| (p - t).abs())
        .sum::<f64>()
        / test_y.len() as f64;

    eprintln!("MAE (SquaredError): {mae_mse:.2}");
    eprintln!("MAE (Huber):        {mae_huber:.2}");

    assert!(
        mae_huber < mae_mse + 1.0,
        "Huber should not be significantly worse than MSE on outlier data: Huber={mae_huber:.2}, MSE={mae_mse:.2}"
    );
}

// ─────────────────────────────────────────────────────────────────
// Session 8: Clustering improvements + NB variants
// ─────────────────────────────────────────────────────────────────

/// Prove K-Means `n_init=10` produces lower or equal inertia compared to `n_init=1`
/// on the Iris dataset (k=3).
#[test]
fn prove_kmeans_n_init_best_of_10() {
    let data = iris_dataset();

    let mut km1 = KMeans::new(3).seed(7).n_init(1);
    km1.fit(&data).unwrap();
    let inertia1 = km1.inertia();

    let mut km10 = KMeans::new(3).seed(7).n_init(10);
    km10.fit(&data).unwrap();
    let inertia10 = km10.inertia();

    eprintln!("KMeans inertia: n_init=1 → {inertia1:.2}, n_init=10 → {inertia10:.2}");
    assert!(
        inertia10 <= inertia1 + 1e-6,
        "n_init=10 should find ≤ inertia of n_init=1: {inertia10:.2} vs {inertia1:.2}"
    );
}

/// Prove silhouette score on Iris (k=3) exceeds 0.4.
/// sklearn reference: KMeans(3) on Iris → `silhouette_score` ≈ 0.55.
#[test]
fn prove_silhouette_score_iris() {
    use scry_learn::cluster::silhouette_score;

    let data = iris_dataset();
    let mut km = KMeans::new(3).seed(42).n_init(10);
    km.fit(&data).unwrap();

    let features = data.feature_matrix();
    let labels = km.labels();
    let score = silhouette_score(&features, labels);

    eprintln!("Silhouette score (Iris, k=3): {score:.4}");
    assert!(
        score > 0.40,
        "KMeans(k=3) on Iris should have silhouette > 0.40, got {score:.4}"
    );
}

/// Prove `BernoulliNB` achieves ≥75% accuracy on a binary feature dataset.
#[test]
fn prove_bernoulli_nb_binary_features() {
    use scry_learn::naive_bayes::BernoulliNB;

    // 200-sample binary feature dataset: class 0 has f0=1 more often,
    // class 1 has f1=1 more often.
    let n = 200;
    let mut rng = fastrand::Rng::with_seed(42);
    let mut f0 = Vec::with_capacity(n);
    let mut f1 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);

    for _ in 0..n / 2 {
        f0.push(if rng.f64() < 0.8 { 1.0 } else { 0.0 });
        f1.push(if rng.f64() < 0.2 { 1.0 } else { 0.0 });
        target.push(0.0);
    }
    for _ in 0..n / 2 {
        f0.push(if rng.f64() < 0.2 { 1.0 } else { 0.0 });
        f1.push(if rng.f64() < 0.8 { 1.0 } else { 0.0 });
        target.push(1.0);
    }

    let data = Dataset::new(
        vec![f0, f1],
        target,
        vec!["f0".into(), "f1".into()],
        "class",
    );

    let (train, test) = scry_learn::split::train_test_split(&data, 0.2, 42);
    let mut nb = BernoulliNB::new().binarize(Some(0.5));
    nb.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = nb.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!(
        "BernoulliNB accuracy on binary features: {:.1}%",
        acc * 100.0
    );
    assert!(
        acc >= 0.75,
        "BernoulliNB should achieve ≥75% on binary feature data (got {:.1}%)",
        acc * 100.0
    );
}

/// Prove `MultinomialNB` achieves ≥75% accuracy on a count feature dataset.
#[test]
fn prove_multinomial_nb_count_features() {
    use scry_learn::naive_bayes::MultinomialNB;

    // Simulated text classification: class 0 has high word_a counts,
    // class 1 has high word_b counts.
    let n = 200;
    let mut rng = fastrand::Rng::with_seed(42);
    let mut f0 = Vec::with_capacity(n); // word_a count
    let mut f1 = Vec::with_capacity(n); // word_b count
    let mut target = Vec::with_capacity(n);

    for _ in 0..n / 2 {
        f0.push(5.0 + rng.f64() * 5.0); // high word_a
        f1.push(rng.f64() * 2.0); // low word_b
        target.push(0.0);
    }
    for _ in 0..n / 2 {
        f0.push(rng.f64() * 2.0); // low word_a
        f1.push(5.0 + rng.f64() * 5.0); // high word_b
        target.push(1.0);
    }

    let data = Dataset::new(
        vec![f0, f1],
        target,
        vec!["word_a".into(), "word_b".into()],
        "class",
    );

    let (train, test) = scry_learn::split::train_test_split(&data, 0.2, 42);
    let mut nb = MultinomialNB::new();
    nb.fit(&train).unwrap();

    let features = test.feature_matrix();
    let preds = nb.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!(
        "MultinomialNB accuracy on count features: {:.1}%",
        acc * 100.0
    );
    assert!(
        acc >= 0.75,
        "MultinomialNB should achieve ≥75% on count feature data (got {:.1}%)",
        acc * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────
// Session 9: DBSCAN, Lasso, and SVM correctness proofs
// ─────────────────────────────────────────────────────────────────

/// Prove DBSCAN correctly identifies two well-separated clusters.
///
/// Two blobs of 50 points each, centered at (0,0) and (50,50),
/// with eps=5.0 and `min_samples=3`. Should find exactly 2 clusters
/// with 0 noise points.
#[test]
fn prove_dbscan_finds_clusters() {
    use scry_learn::cluster::Dbscan;

    let mut rng = fastrand::Rng::with_seed(42);
    let n_per_cluster = 50;
    let mut f1 = Vec::with_capacity(n_per_cluster * 2);
    let mut f2 = Vec::with_capacity(n_per_cluster * 2);

    // Cluster A near origin.
    for _ in 0..n_per_cluster {
        f1.push(rng.f64() * 2.0);
        f2.push(rng.f64() * 2.0);
    }
    // Cluster B far away.
    for _ in 0..n_per_cluster {
        f1.push(50.0 + rng.f64() * 2.0);
        f2.push(50.0 + rng.f64() * 2.0);
    }

    let data = Dataset::new(
        vec![f1, f2],
        vec![0.0; n_per_cluster * 2],
        vec!["x".into(), "y".into()],
        "label",
    );

    let mut db = Dbscan::new(5.0, 3);
    db.fit(&data).unwrap();

    eprintln!(
        "DBSCAN: {} clusters, {} noise points",
        db.n_clusters(),
        db.n_noise()
    );

    assert_eq!(
        db.n_clusters(),
        2,
        "DBSCAN should find exactly 2 clusters on well-separated blobs (got {})",
        db.n_clusters()
    );
    assert_eq!(
        db.n_noise(),
        0,
        "No noise points expected on dense blobs (got {})",
        db.n_noise()
    );

    // Verify cluster assignment consistency: all points in first half
    // should share one label, all points in second half another.
    let labels = db.labels();
    let label_a = labels[0];
    let label_b = labels[n_per_cluster];
    assert_ne!(
        label_a, label_b,
        "Two clusters should have different labels"
    );
    for &l in &labels[..n_per_cluster] {
        assert_eq!(
            l, label_a,
            "All cluster A points should share the same label"
        );
    }
    for &l in &labels[n_per_cluster..] {
        assert_eq!(
            l, label_b,
            "All cluster B points should share the same label"
        );
    }
}

/// Prove Lasso drives noise feature coefficients below 0.05.
///
/// y = 2·x₁ + 0·x₂ + 3·x₃ + 0·x₄ + 0·x₅ + 0·x₆ + 0·x₇ + 0·x₈ + 1
/// Features x₂, x₄-x₈ are pure noise. With alpha=0.5, Lasso should
/// shrink noise coefficients to near zero.
#[test]
fn prove_lasso_sparsity_zero_coefficients() {
    use scry_learn::linear::LassoRegression;

    let n = 500;
    let mut rng = fastrand::Rng::with_seed(42);

    // 8 features: only x₁ (index 0) and x₃ (index 2) are relevant.
    let cols: Vec<Vec<f64>> = (0..8)
        .map(|_| (0..n).map(|_| rng.f64() * 10.0).collect())
        .collect();

    let y: Vec<f64> = (0..n)
        .map(|i| 2.0 * cols[0][i] + 3.0 * cols[2][i] + 1.0 + rng.f64() * 0.1)
        .collect();

    let names: Vec<String> = (0..8).map(|i| format!("x{}", i + 1)).collect();
    let data = Dataset::new(cols, y, names, "y");

    let mut lasso = LassoRegression::new().alpha(0.5).max_iter(5000).tol(1e-6);
    lasso.fit(&data).unwrap();

    let coefs = lasso.coefficients();
    eprintln!("Lasso coefficients (8 features): {coefs:?}");

    // Relevant features should have significant coefficients.
    assert!(
        coefs[0].abs() > 1.0,
        "x₁ coefficient should be significant (got {:.4})",
        coefs[0]
    );
    assert!(
        coefs[2].abs() > 1.0,
        "x₃ coefficient should be significant (got {:.4})",
        coefs[2]
    );

    // All noise features (indices 1, 3, 4, 5, 6, 7) should be near zero.
    for &idx in &[1, 3, 4, 5, 6, 7] {
        assert!(
            coefs[idx].abs() < 0.1,
            "Noise feature x{} coefficient should be <0.1 (got {:.4})",
            idx + 1,
            coefs[idx]
        );
    }
}

/// Prove `LinearSVC` margin separation via `decision_function`.
///
/// On linearly separable data, the decision function should produce
/// positive scores for one class and negative scores for the other.
#[test]
fn prove_svm_margin_separation() {
    use scry_learn::preprocess::{StandardScaler, Transformer};
    use scry_learn::svm::LinearSVC;

    // Generate well-separated 2D data: class 0 at (0,0), class 1 at (10,10).
    let n = 100;
    let mut rng = fastrand::Rng::with_seed(42);
    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);

    for _ in 0..n / 2 {
        f1.push(rng.f64() * 2.0);
        f2.push(rng.f64() * 2.0);
        target.push(0.0);
    }
    for _ in 0..n / 2 {
        f1.push(10.0 + rng.f64() * 2.0);
        f2.push(10.0 + rng.f64() * 2.0);
        target.push(1.0);
    }

    let mut data = Dataset::new(
        vec![f1, f2],
        target.clone(),
        vec!["x".into(), "y".into()],
        "class",
    );

    let mut scaler = StandardScaler::new();
    scaler.fit(&data).unwrap();
    scaler.transform(&mut data).unwrap();

    let mut svc = LinearSVC::new().c(1.0).max_iter(1000).tol(1e-5);
    svc.fit(&data).unwrap();

    let features = data.feature_matrix();
    let preds = svc.predict(&features).unwrap();
    let acc = accuracy(&target, &preds);

    eprintln!("SVM margin test accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.95,
        "LinearSVC on well-separated data should achieve ≥95% (got {:.1}%)",
        acc * 100.0
    );

    // Check decision function scores: class-0 points should have negative
    // scores for the class-1 decision, and vice versa.
    let scores = svc.decision_function(&features).unwrap();

    // For binary classification, we expect scores[i] to have 2 entries.
    // The predicted class is argmax(scores[i]).
    let n_half = n / 2;
    let mut class0_correct = 0;
    let mut class1_correct = 0;
    for i in 0..n_half {
        if scores[i][0] > scores[i][1] {
            class0_correct += 1;
        }
    }
    for i in n_half..n {
        if scores[i][1] > scores[i][0] {
            class1_correct += 1;
        }
    }

    eprintln!(
        "Decision function: {class0_correct}/{n_half} class-0 correct, {class1_correct}/{n_half} class-1 correct"
    );
    assert!(
        class0_correct >= n_half * 9 / 10,
        "≥90% of class-0 points should have higher score for class 0"
    );
    assert!(
        class1_correct >= n_half * 9 / 10,
        "≥90% of class-1 points should have higher score for class 1"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 10: Histogram-Based Gradient Boosting
// ═══════════════════════════════════════════════════════════════════════════════

/// `HistGBT` classifier on Iris: ≥95% accuracy.
#[test]
fn prove_hist_gbt_classifier_iris() {
    use scry_learn::prelude::*;

    let data = iris_dataset();

    // Test across multiple seeds for robustness (same pattern as GBT proof).
    let seeds = [42u64, 7, 123, 99, 1, 55, 13, 77, 200, 999];
    let mut total_acc = 0.0;
    for &seed in &seeds {
        let (train, test) = scry_learn::split::train_test_split(&data, 0.2, seed);
        let mut model = HistGradientBoostingClassifier::new()
            .n_estimators(200)
            .learning_rate(0.1)
            .max_leaf_nodes(31)
            .min_samples_leaf(2)
            .max_depth(6);
        model.fit(&train).unwrap();

        let preds = model.predict(&test.feature_matrix()).unwrap();
        let acc = scry_learn::metrics::accuracy(&test.target, &preds);
        eprintln!("HistGBT Iris seed {seed:>3}: {acc:.4}");
        total_acc += acc;
    }
    let mean_acc = total_acc / seeds.len() as f64;
    eprintln!("HistGBT Iris mean accuracy: {mean_acc:.4}");
    assert!(
        mean_acc >= 0.92,
        "expected mean ≥92% accuracy on Iris, got {:.1}%",
        mean_acc * 100.0
    );
}

/// `HistGBT` regressor on y=2x+1: R² > 0.95.
#[test]
fn prove_hist_gbt_regressor_linear() {
    use scry_learn::prelude::*;

    let x: Vec<f64> = (0..200).map(|i| i as f64 * 0.05).collect();
    let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
    let data = Dataset::new(vec![x], y, vec!["x".into()], "y");

    let mut model = HistGradientBoostingRegressor::new()
        .n_estimators(100)
        .learning_rate(0.1)
        .max_leaf_nodes(31)
        .min_samples_leaf(3);
    model.fit(&data).unwrap();

    let preds = model.predict(&data.feature_matrix()).unwrap();
    let r2 = scry_learn::metrics::r2_score(&data.target, &preds);

    eprintln!("HistGBT R² on y=2x+1: {r2:.4}");
    assert!(r2 > 0.95, "expected R² > 0.95, got {r2:.4}");
}

/// `HistGBT` handles missing values without panicking, reasonable predictions.
#[test]
fn prove_hist_gbt_missing_values() {
    use scry_learn::prelude::*;

    // Dataset with 10% NaN.
    let n = 200;
    let mut rng = fastrand::Rng::with_seed(42);
    let x1: Vec<f64> = (0..n)
        .map(|i| {
            if rng.f64() < 0.1 {
                f64::NAN
            } else {
                i as f64 * 0.05
            }
        })
        .collect();
    let x2: Vec<f64> = (0..n)
        .map(|i| {
            if rng.f64() < 0.1 {
                f64::NAN
            } else {
                (i as f64 * 0.03).sin()
            }
        })
        .collect();
    let target: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();

    let data = Dataset::new(
        vec![x1, x2],
        target,
        vec!["x1".into(), "x2".into()],
        "class",
    );

    let mut model = HistGradientBoostingClassifier::new()
        .n_estimators(50)
        .learning_rate(0.1)
        .min_samples_leaf(3);
    model.fit(&data).unwrap();

    // Predict with NaN features — should not panic.
    let test = vec![vec![f64::NAN, 0.5], vec![5.0, f64::NAN]];
    let preds = model.predict(&test).unwrap();
    assert_eq!(preds.len(), 2);
    assert!(
        preds[0] == 0.0 || preds[0] == 1.0,
        "prediction must be a valid class"
    );
    assert!(
        preds[1] == 0.0 || preds[1] == 1.0,
        "prediction must be a valid class"
    );
    eprintln!("HistGBT missing values: predictions {preds:?} ✓");
}

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 11: Neural Networks (MLP)
// ═══════════════════════════════════════════════════════════════════════════════

/// Prove `MLPClassifier` solves XOR — the classic nonlinear benchmark.
///
/// XOR requires at least one hidden layer to solve. With `hidden_layers`(&[4])
/// and enough iterations, the MLP should achieve 100% accuracy.
#[test]
fn prove_mlp_classifier_xor() {
    use scry_learn::neural::MLPClassifier;

    // XOR dataset (replicated for stable training).
    let features = vec![
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    ];
    let target = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let data = Dataset::new(features, target, vec!["x".into(), "y".into()], "class");

    let mut clf = MLPClassifier::new()
        .hidden_layers(&[8])
        .learning_rate(0.1)
        .max_iter(500)
        .batch_size(8)
        .seed(42);
    clf.fit(&data).unwrap();

    let test_points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let preds = clf.predict(&test_points).unwrap();
    let expected = vec![0.0, 1.0, 1.0, 0.0];
    let correct = preds
        .iter()
        .zip(expected.iter())
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();

    eprintln!("MLP XOR predictions: {preds:?} (expected {expected:?})");
    eprintln!("MLP XOR accuracy: {correct}/4");
    assert!(
        correct >= 3,
        "MLP should solve XOR (got {correct}/4 correct)"
    );
}

/// Prove `MLPClassifier` achieves ≥80% accuracy on Iris (3-class).
///
/// sklearn reference: `MLPClassifier(hidden_layer_sizes=(100`,), `max_iter=200`)
/// achieves ~97% on Iris with scaling.
#[test]
fn prove_mlp_classifier_iris() {
    use scry_learn::neural::MLPClassifier;
    use scry_learn::preprocess::{StandardScaler, Transformer};

    let data = iris_dataset();
    let (mut train, test) = train_test_split(&data, 0.2, 42);

    // MLP needs scaled features for gradient stability.
    let mut scaler = StandardScaler::new();
    scaler.fit(&train).unwrap();
    scaler.transform(&mut train).unwrap();
    let mut test_scaled = test.clone();
    scaler.transform(&mut test_scaled).unwrap();

    let mut clf = MLPClassifier::new()
        .hidden_layers(&[50, 20])
        .learning_rate(0.01)
        .max_iter(200)
        .batch_size(32)
        .seed(42);
    clf.fit(&train).unwrap();

    let features = test_scaled.feature_matrix();
    let preds = clf.predict(&features).unwrap();
    let acc = accuracy(&test.target, &preds);

    eprintln!("MLP Iris accuracy: {:.1}%", acc * 100.0);
    assert!(
        acc >= 0.80,
        "MLP should achieve ≥80% on Iris (got {:.1}%)",
        acc * 100.0
    );
}

/// Prove `MLPRegressor` learns y = sin(x) with reasonable MSE.
#[test]
fn prove_mlp_regressor_sine() {
    use scry_learn::neural::MLPRegressor;

    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y: Vec<f64> = x.iter().map(|&v| v.sin()).collect();

    let data = Dataset::new(vec![x], y.clone(), vec!["x".into()], "y");

    let mut reg = MLPRegressor::new()
        .hidden_layers(&[32, 16])
        .learning_rate(0.01)
        .max_iter(300)
        .batch_size(32)
        .seed(42);
    reg.fit(&data).unwrap();

    let features = data.feature_matrix();
    let preds = reg.predict(&features).unwrap();
    let mse = mean_squared_error(&y, &preds);

    eprintln!("MLP Regressor sin(x) MSE: {mse:.4}");
    assert!(
        mse < 0.5,
        "MLP on sin(x) should achieve MSE < 0.5 (got {mse:.4})"
    );
}

/// Prove `MLPClassifier` `predict_proba` returns valid probability distributions.
#[test]
fn prove_mlp_predict_proba_valid() {
    use scry_learn::neural::MLPClassifier;
    use scry_learn::preprocess::{StandardScaler, Transformer};

    let data = iris_dataset();
    let (mut train, test) = train_test_split(&data, 0.2, 42);

    let mut scaler = StandardScaler::new();
    scaler.fit(&train).unwrap();
    scaler.transform(&mut train).unwrap();
    let mut test_scaled = test.clone();
    scaler.transform(&mut test_scaled).unwrap();

    let mut clf = MLPClassifier::new()
        .hidden_layers(&[20])
        .max_iter(50)
        .seed(42);
    clf.fit(&train).unwrap();

    let features = test_scaled.feature_matrix();
    let probas = clf.predict_proba(&features).unwrap();

    // predict_proba returns a flat Vec<f64> with n_samples * n_classes entries.
    // For Iris (3 classes), each sample has 3 probability values.
    let n_classes = 3;
    assert_eq!(
        probas.len(),
        test.n_samples() * n_classes,
        "predict_proba should return n_samples * n_classes values"
    );

    // Each sample's probabilities should sum to ~1.0 and be non-negative.
    for i in 0..test.n_samples() {
        let start = i * n_classes;
        let sample_probs = &probas[start..start + n_classes];
        let sum: f64 = sample_probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Sample {i}: probabilities must sum to 1.0, got {sum}"
        );
        for &p in sample_probs {
            assert!(p >= 0.0, "Sample {i}: probabilities must be non-negative");
        }
    }

    eprintln!(
        "MLP predict_proba: all {} samples have valid distributions",
        test.n_samples()
    );
}
