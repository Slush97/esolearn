//! Real-world cross-library comparison: scry-learn vs linfa vs smartcore
//!
//! Tests each library on UCI datasets across classification, regression,
//! and clustering use cases. Reports accuracy, timing, and API observations.
//!
//! Run: `cargo run --example crosslib_comparison -p scry-learn --release`

#![allow(
    missing_docs,
    clippy::redundant_clone,
    clippy::default_trait_access,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::redundant_closure_for_method_calls,
    clippy::map_unwrap_or,
    clippy::cast_precision_loss
)]

use std::path::PathBuf;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// Data loading utilities
// ═══════════════════════════════════════════════════════════════════════════

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_features_csv(name: &str) -> (Vec<Vec<f64>>, Vec<String>) {
    let path = fixtures_dir().join(name);
    let mut rdr = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let headers: Vec<String> = rdr.headers().unwrap().iter().map(String::from).collect();
    let n_cols = headers.len();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        let row: Vec<f64> = record.iter().map(|s| s.parse::<f64>().unwrap()).collect();
        rows.push(row);
    }
    let mut cols = vec![vec![0.0; rows.len()]; n_cols];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            cols[j][i] = val;
        }
    }
    (cols, headers)
}

fn load_target_csv(name: &str) -> Vec<f64> {
    let path = fixtures_dir().join(name);
    let mut rdr = csv::Reader::from_path(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let mut target = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        target.push(record[0].parse::<f64>().unwrap());
    }
    target
}

/// Load a dataset from fixture CSVs. Returns (col_major_features, row_major_features, target, feature_names).
fn load_dataset(
    base: &str,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<String>) {
    let (cols, feat_names) = load_features_csv(&format!("{base}_features.csv"));
    let target = load_target_csv(&format!("{base}_target.csv"));
    let n_samples = target.len();
    let n_features = cols.len();

    let row_major: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| (0..n_features).map(|j| cols[j][i]).collect())
        .collect();

    (cols, row_major, target, feat_names)
}

/// Stratified train/test split (70/30) with deterministic seeding.
fn stratified_split(
    row_major: &[Vec<f64>],
    col_major: &[Vec<f64>],
    target: &[f64],
    ratio: f64,
) -> (
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<f64>,
) {
    let _n = target.len();
    let n_features = col_major.len();

    // Group indices by class
    let mut class_indices: std::collections::HashMap<i64, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &t) in target.iter().enumerate() {
        class_indices.entry(t as i64).or_default().push(i);
    }

    let mut train_idx = Vec::new();
    let mut test_idx = Vec::new();

    // Deterministic: sort classes then split each class proportionally
    let mut classes: Vec<i64> = class_indices.keys().copied().collect();
    classes.sort();

    for cls in classes {
        let indices = &class_indices[&cls];
        let split_at = (indices.len() as f64 * ratio).round() as usize;
        train_idx.extend_from_slice(&indices[..split_at]);
        test_idx.extend_from_slice(&indices[split_at..]);
    }

    let train_row: Vec<Vec<f64>> = train_idx.iter().map(|&i| row_major[i].clone()).collect();
    let test_row: Vec<Vec<f64>> = test_idx.iter().map(|&i| row_major[i].clone()).collect();
    let train_target: Vec<f64> = train_idx.iter().map(|&i| target[i]).collect();
    let test_target: Vec<f64> = test_idx.iter().map(|&i| target[i]).collect();

    let train_col: Vec<Vec<f64>> = (0..n_features)
        .map(|j| train_idx.iter().map(|&i| col_major[j][i]).collect())
        .collect();
    let test_col: Vec<Vec<f64>> = (0..n_features)
        .map(|j| test_idx.iter().map(|&i| col_major[j][i]).collect())
        .collect();

    (
        train_row,
        train_col,
        train_target,
        test_row,
        test_col,
        test_target,
    )
}

fn accuracy_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| (**a - **b).abs() < 0.5)
        .count();
    correct as f64 / y_true.len() as f64
}

fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|y| (y - mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mse: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>()
        / y_true.len() as f64;
    mse.sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Standardization (needed for linear models across all libraries)
// ═══════════════════════════════════════════════════════════════════════════

fn standardize(
    train_row: &[Vec<f64>],
    test_row: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n_features = train_row[0].len();
    let n_train = train_row.len();

    let mut means = vec![0.0; n_features];
    let mut stds = vec![0.0; n_features];

    for j in 0..n_features {
        let sum: f64 = train_row.iter().map(|r| r[j]).sum();
        means[j] = sum / n_train as f64;
        let var: f64 = train_row.iter().map(|r| (r[j] - means[j]).powi(2)).sum::<f64>()
            / n_train as f64;
        stds[j] = var.sqrt().max(1e-12);
    }

    let scale = |rows: &[Vec<f64>]| -> Vec<Vec<f64>> {
        rows.iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, &v)| (v - means[j]) / stds[j])
                    .collect()
            })
            .collect()
    };

    (scale(train_row), scale(test_row))
}

fn transpose_to_col_major(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if rows.is_empty() {
        return vec![];
    }
    let n_cols = rows[0].len();
    let n_rows = rows.len();
    (0..n_cols)
        .map(|j| (0..n_rows).map(|i| rows[i][j]).collect())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════

struct TestResult {
    library: String,
    model: String,
    metric_name: String,
    metric_value: f64,
    train_ms: f64,
    predict_ms: f64,
    status: String, // "OK" or error message
}

impl TestResult {
    fn ok(
        library: &str,
        model: &str,
        metric_name: &str,
        metric_value: f64,
        train_ms: f64,
        predict_ms: f64,
    ) -> Self {
        Self {
            library: library.to_string(),
            model: model.to_string(),
            metric_name: metric_name.to_string(),
            metric_value,
            train_ms,
            predict_ms,
            status: "OK".to_string(),
        }
    }

    fn err(library: &str, model: &str, error: &str) -> Self {
        Self {
            library: library.to_string(),
            model: model.to_string(),
            metric_name: "N/A".to_string(),
            metric_value: f64::NAN,
            train_ms: f64::NAN,
            predict_ms: f64::NAN,
            status: error.to_string(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// USE CASE 1: Classification — Breast Cancer (binary, 569 samples, 30 features)
// ═══════════════════════════════════════════════════════════════════════════

fn test_classification_breast_cancer() -> Vec<TestResult> {
    println!("\n  Loading breast_cancer dataset...");
    let (cols, rows, target, feat_names) = load_dataset("breast_cancer");
    let n_features = feat_names.len();
    let (train_row, _train_col, train_target, test_row, _test_col, test_target) =
        stratified_split(&rows, &cols, &target, 0.7);

    let (train_scaled, test_scaled) = standardize(&train_row, &test_row);
    let train_scaled_col = transpose_to_col_major(&train_scaled);
    let test_scaled_col = transpose_to_col_major(&test_scaled);
    let train_col = transpose_to_col_major(&train_row);

    let mut results = Vec::new();

    // --- Decision Tree ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(),
            train_target.clone(),
            feat_names.clone(),
            "target",
        );
        let mut dt = scry_learn::tree::DecisionTreeClassifier::new().max_depth(10);
        let t0 = Instant::now();
        dt.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = dt.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Decision Tree", "Accuracy", acc, train_ms, pred_ms));
    }
    // smartcore
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_row).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let params = smartcore::tree::decision_tree_classifier::DecisionTreeClassifierParameters::default()
            .with_max_depth(10);
        let t0 = Instant::now();
        let model = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(&x, &y, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_row).unwrap();
        let t0 = Instant::now();
        let preds: Vec<i32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "Decision Tree", "Accuracy", acc, train_ms, pred_ms));
    }
    // linfa: decision tree
    {
        use linfa::prelude::*;
        let flat: Vec<f64> = train_row.iter().flat_map(|r| r.iter().copied()).collect();
        let x = ndarray::Array2::from_shape_vec((train_row.len(), n_features), flat).unwrap();
        let y = ndarray::Array1::from_vec(train_target.iter().map(|&t| t as usize).collect::<Vec<_>>());
        let ds = linfa::Dataset::new(x, y);
        let t0 = Instant::now();
        let model = linfa_trees::DecisionTree::params()
            .max_depth(Some(10))
            .fit(&ds)
            .unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let flat_test: Vec<f64> = test_row.iter().flat_map(|r| r.iter().copied()).collect();
        let x_test = ndarray::Array2::from_shape_vec((test_row.len(), n_features), flat_test).unwrap();
        let ds_test = linfa::DatasetBase::from(x_test);
        let t0 = Instant::now();
        let preds = model.predict(&ds_test);
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("linfa", "Decision Tree", "Accuracy", acc, train_ms, pred_ms));
    }

    // --- Random Forest ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(),
            train_target.clone(),
            feat_names.clone(),
            "target",
        );
        let mut rf = scry_learn::tree::RandomForestClassifier::new()
            .n_estimators(100)
            .max_depth(10)
            .seed(42);
        let t0 = Instant::now();
        rf.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = rf.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Random Forest", "Accuracy", acc, train_ms, pred_ms));
    }
    // smartcore
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_row).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(100)
            .with_max_depth(10);
        let t0 = Instant::now();
        let model = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(&x, &y, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_row).unwrap();
        let t0 = Instant::now();
        let preds: Vec<i32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "Random Forest", "Accuracy", acc, train_ms, pred_ms));
    }
    // linfa: no random forest in linfa-trees, so skip
    results.push(TestResult::err("linfa", "Random Forest", "Not implemented in linfa"));

    // --- Logistic Regression (scaled data) ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_scaled_col.clone(),
            train_target.clone(),
            feat_names.clone(),
            "target",
        );
        let mut lr = scry_learn::linear::LogisticRegression::new()
            .max_iter(500)
            .learning_rate(0.01);
        let t0 = Instant::now();
        lr.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = lr.predict(&test_scaled).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Logistic Regression", "Accuracy", acc, train_ms, pred_ms));
    }
    // smartcore
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_scaled).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let t0 = Instant::now();
        let model = smartcore::linear::logistic_regression::LogisticRegression::fit(
            &x, &y, Default::default(),
        ).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_scaled).unwrap();
        let t0 = Instant::now();
        let preds: Vec<i32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "Logistic Regression", "Accuracy", acc, train_ms, pred_ms));
    }
    // linfa-logistic
    {
        use linfa::prelude::*;
        let flat: Vec<f64> = train_scaled.iter().flat_map(|r| r.iter().copied()).collect();
        let x = ndarray::Array2::from_shape_vec((train_scaled.len(), n_features), flat).unwrap();
        let y = ndarray::Array1::from_vec(train_target.iter().map(|&t| t > 0.5).collect::<Vec<_>>());
        let ds = linfa::Dataset::new(x, y);
        let t0 = Instant::now();
        let result = linfa_logistic::LogisticRegression::default()
            .max_iterations(500)
            .fit(&ds);
        match result {
            Ok(model) => {
                let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let flat_test: Vec<f64> = test_scaled.iter().flat_map(|r| r.iter().copied()).collect();
                let x_test = ndarray::Array2::from_shape_vec((test_scaled.len(), n_features), flat_test).unwrap();
                let ds_test = linfa::DatasetBase::from(x_test);
                let t0 = Instant::now();
                let preds = model.predict(&ds_test);
                let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let preds_f64: Vec<f64> = preds.iter().map(|&p| if p { 1.0 } else { 0.0 }).collect();
                let acc = accuracy_score(&test_target, &preds_f64);
                results.push(TestResult::ok("linfa", "Logistic Regression", "Accuracy", acc, train_ms, pred_ms));
            }
            Err(e) => results.push(TestResult::err("linfa", "Logistic Regression", &format!("{e}"))),
        }
    }

    // --- KNN ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_scaled_col.clone(),
            train_target.clone(),
            feat_names.clone(),
            "target",
        );
        let mut knn = scry_learn::neighbors::KnnClassifier::new().k(5);
        let t0 = Instant::now();
        knn.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = knn.predict(&test_scaled).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "KNN (k=5)", "Accuracy", acc, train_ms, pred_ms));
    }
    // smartcore
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_scaled).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let params = smartcore::neighbors::knn_classifier::KNNClassifierParameters::default().with_k(5);
        let t0 = Instant::now();
        let model = smartcore::neighbors::knn_classifier::KNNClassifier::fit(&x, &y, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_scaled).unwrap();
        let t0 = Instant::now();
        let preds: Vec<i32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "KNN (k=5)", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "KNN (k=5)", "linfa-nn has no classifier wrapper"));

    // --- Gaussian Naive Bayes ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(),
            train_target.clone(),
            feat_names.clone(),
            "target",
        );
        let mut gnb = scry_learn::naive_bayes::GaussianNb::new();
        let t0 = Instant::now();
        gnb.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = gnb.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Gaussian NB", "Accuracy", acc, train_ms, pred_ms));
    }
    // smartcore
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_row).unwrap();
        let y: Vec<u32> = train_target.iter().map(|&t| t as u32).collect();
        let t0 = Instant::now();
        let model = smartcore::naive_bayes::gaussian::GaussianNB::fit(&x, &y, Default::default()).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_row).unwrap();
        let t0 = Instant::now();
        let preds: Vec<u32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "Gaussian NB", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "Gaussian NB", "Not implemented in linfa"));

    // --- LinearSVC (scaled) ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_scaled_col.clone(),
            train_target.clone(),
            feat_names.clone(),
            "target",
        );
        let mut svc = scry_learn::svm::LinearSVC::new().c(1.0).max_iter(1000);
        let t0 = Instant::now();
        svc.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = svc.predict(&test_scaled).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "LinearSVC", "Accuracy", acc, train_ms, pred_ms));
    }
    // smartcore (kernel SVM with linear kernel — different algorithm)
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_scaled).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let knl = smartcore::svm::Kernels::linear();
        let params = smartcore::svm::svc::SVCParameters::default()
            .with_c(1.0)
            .with_kernel(knl);
        let t0 = Instant::now();
        let model = smartcore::svm::svc::SVC::fit(&x, &y, &params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_scaled).unwrap();
        let t0 = Instant::now();
        let preds_f64: Vec<f64> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "SVM (kernel=linear)", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "SVM", "linfa-svm requires additional setup"));

    // --- Gradient Boosting (scry-learn only — unique capability) ---
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(),
            train_target.clone(),
            feat_names.clone(),
            "target",
        );
        let mut gb = scry_learn::tree::GradientBoostingClassifier::new()
            .n_estimators(100)
            .max_depth(5)
            .learning_rate(0.1);
        let t0 = Instant::now();
        gb.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = gb.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Gradient Boosting", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("smartcore", "Gradient Boosting", "Not implemented"));
    results.push(TestResult::err("linfa", "Gradient Boosting", "Not implemented"));

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// USE CASE 2: Multi-class Classification — Wine (3 classes, 178 samples, 13 features)
// ═══════════════════════════════════════════════════════════════════════════

fn test_classification_wine() -> Vec<TestResult> {
    println!("\n  Loading wine dataset...");
    let (cols, rows, target, feat_names) = load_dataset("wine");
    let n_features = feat_names.len();
    let (train_row, _train_col, train_target, test_row, _test_col, test_target) =
        stratified_split(&rows, &cols, &target, 0.7);

    let (train_scaled, test_scaled) = standardize(&train_row, &test_row);
    let train_scaled_col = transpose_to_col_major(&train_scaled);
    let train_col = transpose_to_col_major(&train_row);

    let mut results = Vec::new();

    // --- Random Forest ---
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut rf = scry_learn::tree::RandomForestClassifier::new()
            .n_estimators(100).max_depth(10).seed(42);
        let t0 = Instant::now();
        rf.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = rf.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Random Forest", "Accuracy", acc, train_ms, pred_ms));
    }
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_row).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(100).with_max_depth(10);
        let t0 = Instant::now();
        let model = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(&x, &y, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_row).unwrap();
        let t0 = Instant::now();
        let preds: Vec<i32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "Random Forest", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "Random Forest", "Not implemented"));

    // --- Logistic Regression (multiclass) ---
    {
        let data = scry_learn::dataset::Dataset::new(
            train_scaled_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut lr = scry_learn::linear::LogisticRegression::new()
            .max_iter(500).learning_rate(0.01);
        let t0 = Instant::now();
        lr.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = lr.predict(&test_scaled).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Logistic Regression", "Accuracy", acc, train_ms, pred_ms));
    }
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_scaled).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let t0 = Instant::now();
        let model = smartcore::linear::logistic_regression::LogisticRegression::fit(
            &x, &y, Default::default(),
        ).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_scaled).unwrap();
        let t0 = Instant::now();
        let preds: Vec<i32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "Logistic Regression", "Accuracy", acc, train_ms, pred_ms));
    }
    // linfa-logistic only supports binary classification
    results.push(TestResult::err("linfa", "Logistic Regression", "Binary only (no multiclass)"));

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// USE CASE 3: Regression — California Housing (20640 samples, 8 features)
// ═══════════════════════════════════════════════════════════════════════════

fn test_regression_california() -> Vec<TestResult> {
    println!("\n  Loading california housing dataset...");
    let (cols, rows, target, feat_names) = load_dataset("california");
    let n_features = feat_names.len();

    // Simple 70/30 split (not stratified for regression)
    let split_at = (rows.len() as f64 * 0.7) as usize;
    let train_row = rows[..split_at].to_vec();
    let test_row = rows[split_at..].to_vec();
    let train_target = target[..split_at].to_vec();
    let test_target = target[split_at..].to_vec();

    let (train_scaled, test_scaled) = standardize(&train_row, &test_row);
    let train_scaled_col = transpose_to_col_major(&train_scaled);
    let train_col = transpose_to_col_major(&train_row);

    let mut results = Vec::new();

    // --- Ridge Regression ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_scaled_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut ridge = scry_learn::linear::Ridge::new(1.0);
        let t0 = Instant::now();
        ridge.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = ridge.predict(&test_scaled).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r2 = r2_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Ridge Regression", "R2", r2, train_ms, pred_ms));
    }
    // smartcore
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_scaled).unwrap();
        let params = smartcore::linear::ridge_regression::RidgeRegressionParameters::default()
            .with_alpha(1.0);
        let t0 = Instant::now();
        let model = smartcore::linear::ridge_regression::RidgeRegression::fit(&x, &train_target, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_scaled).unwrap();
        let t0 = Instant::now();
        let preds: Vec<f64> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r2 = r2_score(&test_target, &preds);
        results.push(TestResult::ok("smartcore", "Ridge Regression", "R2", r2, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "Ridge Regression", "Not implemented in linfa"));

    // --- Lasso Regression ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_scaled_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut lasso = scry_learn::linear::LassoRegression::new().alpha(0.1).max_iter(1000);
        let t0 = Instant::now();
        lasso.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = lasso.predict(&test_scaled).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r2 = r2_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Lasso Regression", "R2", r2, train_ms, pred_ms));
    }
    // linfa-elasticnet
    {
        use linfa::prelude::*;
        let flat: Vec<f64> = train_scaled.iter().flat_map(|r| r.iter().copied()).collect();
        let x = ndarray::Array2::from_shape_vec((train_scaled.len(), n_features), flat).unwrap();
        let y = ndarray::Array1::from_vec(train_target.clone());
        let ds = linfa::Dataset::new(x, y);
        let t0 = Instant::now();
        let result = linfa_elasticnet::ElasticNet::<f64>::lasso()
            .penalty(0.1)
            .max_iterations(1000)
            .fit(&ds);
        match result {
            Ok(model) => {
                let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let flat_test: Vec<f64> = test_scaled.iter().flat_map(|r| r.iter().copied()).collect();
                let x_test = ndarray::Array2::from_shape_vec((test_scaled.len(), n_features), flat_test).unwrap();
                let ds_test = linfa::DatasetBase::from(x_test);
                let t0 = Instant::now();
                let preds = model.predict(&ds_test);
                let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let r2 = r2_score(&test_target, preds.as_slice().unwrap());
                results.push(TestResult::ok("linfa", "Lasso (ElasticNet)", "R2", r2, train_ms, pred_ms));
            }
            Err(e) => results.push(TestResult::err("linfa", "Lasso (ElasticNet)", &format!("{e}"))),
        }
    }
    // smartcore lasso
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_scaled).unwrap();
        let params = smartcore::linear::lasso::LassoParameters::default().with_alpha(0.1);
        let t0 = Instant::now();
        let result = smartcore::linear::lasso::Lasso::fit(&x, &train_target, params);
        match result {
            Ok(model) => {
                let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_scaled).unwrap();
                let t0 = Instant::now();
                let preds: Vec<f64> = model.predict(&x_test).unwrap();
                let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let r2 = r2_score(&test_target, &preds);
                results.push(TestResult::ok("smartcore", "Lasso Regression", "R2", r2, train_ms, pred_ms));
            }
            Err(e) => results.push(TestResult::err("smartcore", "Lasso Regression", &format!("{e}"))),
        }
    }

    // --- Decision Tree Regressor ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut dt = scry_learn::tree::DecisionTreeRegressor::new().max_depth(10);
        let t0 = Instant::now();
        dt.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = dt.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r2 = r2_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Decision Tree Regressor", "R2", r2, train_ms, pred_ms));
    }
    // smartcore
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_row).unwrap();
        let params = smartcore::tree::decision_tree_regressor::DecisionTreeRegressorParameters::default()
            .with_max_depth(10);
        let t0 = Instant::now();
        let model = smartcore::tree::decision_tree_regressor::DecisionTreeRegressor::fit(&x, &train_target, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_row).unwrap();
        let t0 = Instant::now();
        let preds: Vec<f64> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r2 = r2_score(&test_target, &preds);
        results.push(TestResult::ok("smartcore", "Decision Tree Regressor", "R2", r2, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "Decision Tree Regressor", "linfa-trees: classification only"));

    // --- KNN Regressor ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            train_scaled_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut knn = scry_learn::neighbors::KnnRegressor::new().k(5);
        let t0 = Instant::now();
        knn.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = knn.predict(&test_scaled).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r2 = r2_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "KNN Regressor (k=5)", "R2", r2, train_ms, pred_ms));
    }
    // smartcore
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_scaled).unwrap();
        let params = smartcore::neighbors::knn_regressor::KNNRegressorParameters::default().with_k(5);
        let t0 = Instant::now();
        let model = smartcore::neighbors::knn_regressor::KNNRegressor::fit(&x, &train_target, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_scaled).unwrap();
        let t0 = Instant::now();
        let preds: Vec<f64> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let r2 = r2_score(&test_target, &preds);
        results.push(TestResult::ok("smartcore", "KNN Regressor (k=5)", "R2", r2, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "KNN Regressor", "Not implemented"));

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// USE CASE 4: Clustering — Iris (150 samples, 4 features, 3 true clusters)
// ═══════════════════════════════════════════════════════════════════════════

fn test_clustering_iris() -> Vec<TestResult> {
    println!("\n  Loading iris dataset for clustering...");
    let (cols, rows, target, feat_names) = load_dataset("iris");
    let n_features = feat_names.len();

    let mut results = Vec::new();

    // --- K-Means ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            cols.clone(), target.clone(), feat_names.clone(), "target",
        );
        let mut km = scry_learn::cluster::KMeans::new(3).seed(42).max_iter(100);
        let t0 = Instant::now();
        km.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let labels = km.predict(&rows).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        // Compute silhouette score
        let sil = scry_learn::cluster::silhouette_score(&rows, &labels);
        results.push(TestResult::ok("scry-learn", "K-Means (k=3)", "Silhouette", sil, train_ms, pred_ms));
    }
    // linfa-clustering
    {
        use linfa::prelude::*;
        use rand::SeedableRng;
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let x = ndarray::Array2::from_shape_vec((rows.len(), n_features), flat).unwrap();
        let ds = linfa::DatasetBase::from(x.clone());
        let rng = rand::rngs::SmallRng::seed_from_u64(42);
        let t0 = Instant::now();
        let result = linfa_clustering::KMeans::params_with_rng(3, rng)
            .max_n_iterations(100)
            .fit(&ds);
        match result {
            Ok(model) => {
                let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let ds_pred = linfa::DatasetBase::from(x);
                let t0 = Instant::now();
                let preds = model.predict(&ds_pred);
                let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let labels: Vec<usize> = preds.iter().map(|&l| l).collect();
                let sil = scry_learn::cluster::silhouette_score(&rows, &labels);
                results.push(TestResult::ok("linfa", "K-Means (k=3)", "Silhouette", sil, train_ms, pred_ms));
            }
            Err(e) => results.push(TestResult::err("linfa", "K-Means", &format!("{e}"))),
        }
    }
    // smartcore doesn't have k-means
    results.push(TestResult::err("smartcore", "K-Means", "Not implemented"));

    // --- DBSCAN ---
    // scry-learn
    {
        let data = scry_learn::dataset::Dataset::new(
            cols.clone(), target.clone(), feat_names.clone(), "target",
        );
        let mut dbscan = scry_learn::cluster::Dbscan::new(0.5, 5);
        let t0 = Instant::now();
        dbscan.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let labels = dbscan.predict(&rows).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let n_clusters = labels.iter().filter(|&&l| l >= 0).map(|l| *l as i64).collect::<std::collections::HashSet<_>>().len();
        // Filter noise for silhouette
        let non_noise: Vec<(usize, usize)> = labels.iter().enumerate()
            .filter(|(_, &l)| l >= 0)
            .map(|(i, &l)| (i, l as usize))
            .collect();
        let sil = if non_noise.len() > 1 {
            let filtered_rows: Vec<Vec<f64>> = non_noise.iter().map(|(i, _)| rows[*i].clone()).collect();
            let filtered_labels: Vec<usize> = non_noise.iter().map(|(_, l)| *l).collect();
            scry_learn::cluster::silhouette_score(&filtered_rows, &filtered_labels)
        } else {
            f64::NAN
        };
        results.push(TestResult::ok("scry-learn", &format!("DBSCAN ({n_clusters} clusters)"), "Silhouette", sil, train_ms, pred_ms));
    }
    // linfa DBSCAN
    {
        use linfa::prelude::*;
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let x = ndarray::Array2::from_shape_vec((rows.len(), n_features), flat).unwrap();
        let ds = linfa::DatasetBase::from(x);
        let t0 = Instant::now();
        let result = linfa_clustering::Dbscan::params(5)
            .tolerance(0.5)
            .transform(ds);
        match result {
            Ok(clustered) => {
                let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let labels: Vec<Option<usize>> = clustered.targets().iter().copied().collect();
                let n_clusters = labels.iter().filter_map(|l| *l).collect::<std::collections::HashSet<_>>().len();
                let non_noise: Vec<(usize, usize)> = labels.iter().enumerate()
                    .filter_map(|(i, l)| l.map(|c| (i, c)))
                    .collect();
                let sil = if non_noise.len() > 1 {
                    let filtered_rows: Vec<Vec<f64>> = non_noise.iter().map(|(i, _)| rows[*i].clone()).collect();
                    let filtered_labels: Vec<usize> = non_noise.iter().map(|(_, l)| *l).collect();
                    scry_learn::cluster::silhouette_score(&filtered_rows, &filtered_labels)
                } else {
                    f64::NAN
                };
                results.push(TestResult::ok("linfa", &format!("DBSCAN ({n_clusters} clusters)"), "Silhouette", sil, train_ms, 0.0));
            }
            Err(e) => results.push(TestResult::err("linfa", "DBSCAN", &format!("{e}"))),
        }
    }
    results.push(TestResult::err("smartcore", "DBSCAN", "Not implemented"));

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// USE CASE 5: High-dimensional Classification — Digits (1797 samples, 64 features, 10 classes)
// ═══════════════════════════════════════════════════════════════════════════

fn test_classification_digits() -> Vec<TestResult> {
    println!("\n  Loading digits dataset...");
    let (cols, rows, target, feat_names) = load_dataset("digits");
    let n_features = feat_names.len();
    let (train_row, _train_col, train_target, test_row, _test_col, test_target) =
        stratified_split(&rows, &cols, &target, 0.7);

    let (train_scaled, test_scaled) = standardize(&train_row, &test_row);
    let train_scaled_col = transpose_to_col_major(&train_scaled);
    let train_col = transpose_to_col_major(&train_row);

    let mut results = Vec::new();

    // --- Random Forest ---
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut rf = scry_learn::tree::RandomForestClassifier::new()
            .n_estimators(100).max_depth(20).seed(42);
        let t0 = Instant::now();
        rf.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = rf.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Random Forest", "Accuracy", acc, train_ms, pred_ms));
    }
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_row).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(100).with_max_depth(20);
        let t0 = Instant::now();
        let model = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(&x, &y, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_row).unwrap();
        let t0 = Instant::now();
        let preds: Vec<i32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "Random Forest", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "Random Forest", "Not implemented"));

    // --- KNN ---
    {
        let data = scry_learn::dataset::Dataset::new(
            train_scaled_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut knn = scry_learn::neighbors::KnnClassifier::new().k(3);
        let t0 = Instant::now();
        knn.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = knn.predict(&test_scaled).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "KNN (k=3)", "Accuracy", acc, train_ms, pred_ms));
    }
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_scaled).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let params = smartcore::neighbors::knn_classifier::KNNClassifierParameters::default().with_k(3);
        let t0 = Instant::now();
        let model = smartcore::neighbors::knn_classifier::KNNClassifier::fit(&x, &y, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_scaled).unwrap();
        let t0 = Instant::now();
        let preds: Vec<i32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "KNN (k=3)", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "KNN (k=3)", "No classifier wrapper"));

    // --- HistGradientBoosting (scry-learn exclusive) ---
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut hgb = scry_learn::tree::HistGradientBoostingClassifier::new()
            .n_estimators(100).max_depth(6).learning_rate(0.1);
        let t0 = Instant::now();
        hgb.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = hgb.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "HistGradientBoosting", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("smartcore", "HistGradientBoosting", "Not implemented"));
    results.push(TestResult::err("linfa", "HistGradientBoosting", "Not implemented"));

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// USE CASE 6: Large-scale Classification — Spambase (4601 samples, 57 features)
// ═══════════════════════════════════════════════════════════════════════════

fn test_classification_spambase() -> Vec<TestResult> {
    println!("\n  Loading spambase dataset...");
    let (cols, rows, target, feat_names) = load_dataset("spambase");
    let n_features = feat_names.len();
    let (train_row, _train_col, train_target, test_row, _test_col, test_target) =
        stratified_split(&rows, &cols, &target, 0.7);

    let train_col = transpose_to_col_major(&train_row);

    let mut results = Vec::new();

    // --- Random Forest ---
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut rf = scry_learn::tree::RandomForestClassifier::new()
            .n_estimators(100).max_depth(15).seed(42);
        let t0 = Instant::now();
        rf.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = rf.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Random Forest", "Accuracy", acc, train_ms, pred_ms));
    }
    {
        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&train_row).unwrap();
        let y: Vec<i32> = train_target.iter().map(|&t| t as i32).collect();
        let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
            .with_n_trees(100).with_max_depth(15);
        let t0 = Instant::now();
        let model = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(&x, &y, params).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let x_test = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&test_row).unwrap();
        let t0 = Instant::now();
        let preds: Vec<i32> = model.predict(&x_test).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let preds_f64: Vec<f64> = preds.iter().map(|&p| p as f64).collect();
        let acc = accuracy_score(&test_target, &preds_f64);
        results.push(TestResult::ok("smartcore", "Random Forest", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("linfa", "Random Forest", "Not implemented"));

    // --- Gradient Boosting (scry-learn exclusive) ---
    {
        let data = scry_learn::dataset::Dataset::new(
            train_col.clone(), train_target.clone(), feat_names.clone(), "target",
        );
        let mut gb = scry_learn::tree::GradientBoostingClassifier::new()
            .n_estimators(100).max_depth(5).learning_rate(0.1);
        let t0 = Instant::now();
        gb.fit(&data).unwrap();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let preds = gb.predict(&test_row).unwrap();
        let pred_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let acc = accuracy_score(&test_target, &preds);
        results.push(TestResult::ok("scry-learn", "Gradient Boosting", "Accuracy", acc, train_ms, pred_ms));
    }
    results.push(TestResult::err("smartcore", "Gradient Boosting", "Not implemented"));
    results.push(TestResult::err("linfa", "Gradient Boosting", "Not implemented"));

    results
}

// ═══════════════════════════════════════════════════════════════════════════
// Reporting
// ═══════════════════════════════════════════════════════════════════════════

fn print_results_table(title: &str, results: &[TestResult]) {
    println!("\n{}", "=".repeat(110));
    println!("  {title}");
    println!("{}", "=".repeat(110));
    println!(
        "  {:<14} {:<28} {:<10} {:<10} {:<12} {:<12} {}",
        "Library", "Model", "Metric", "Value", "Train(ms)", "Predict(ms)", "Status"
    );
    println!("  {}", "-".repeat(104));

    for r in results {
        if r.status == "OK" {
            println!(
                "  {:<14} {:<28} {:<10} {:<10.4} {:<12.2} {:<12.2} {}",
                r.library, r.model, r.metric_name, r.metric_value, r.train_ms, r.predict_ms, r.status
            );
        } else {
            println!(
                "  {:<14} {:<28} {:<10} {:<10} {:<12} {:<12} {}",
                r.library, r.model, "--", "--", "--", "--", r.status
            );
        }
    }
}

fn main() {
    // Force single-threaded for fair comparison
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CROSS-LIBRARY ML COMPARISON: scry-learn vs linfa vs smartcore                        ║");
    println!("║                    Real-World UCI Datasets — Single-Threaded for Fair Comparison                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // Use Case 1: Binary Classification
    let bc_results = test_classification_breast_cancer();
    print_results_table(
        "USE CASE 1: Binary Classification — Breast Cancer (569 samples, 30 features, 2 classes)",
        &bc_results,
    );

    // Use Case 2: Multi-class Classification
    let wine_results = test_classification_wine();
    print_results_table(
        "USE CASE 2: Multi-class Classification — Wine (178 samples, 13 features, 3 classes)",
        &wine_results,
    );

    // Use Case 3: Regression
    let cal_results = test_regression_california();
    print_results_table(
        "USE CASE 3: Regression — California Housing (20640 samples, 8 features)",
        &cal_results,
    );

    // Use Case 4: Clustering
    let cluster_results = test_clustering_iris();
    print_results_table(
        "USE CASE 4: Clustering — Iris (150 samples, 4 features, 3 true clusters)",
        &cluster_results,
    );

    // Use Case 5: High-dimensional
    let digits_results = test_classification_digits();
    print_results_table(
        "USE CASE 5: High-dimensional — Digits (1797 samples, 64 features, 10 classes)",
        &digits_results,
    );

    // Use Case 6: Large-scale
    let spam_results = test_classification_spambase();
    print_results_table(
        "USE CASE 6: Large-scale — Spambase (4601 samples, 57 features, 2 classes)",
        &spam_results,
    );

    // ── Summary ──
    let all_results: Vec<&TestResult> = bc_results.iter()
        .chain(wine_results.iter())
        .chain(cal_results.iter())
        .chain(cluster_results.iter())
        .chain(digits_results.iter())
        .chain(spam_results.iter())
        .collect();

    println!("\n{}", "=".repeat(110));
    println!("  ALGORITHM COVERAGE SUMMARY");
    println!("{}", "=".repeat(110));

    let libs = ["scry-learn", "smartcore", "linfa"];
    for lib in &libs {
        let total = all_results.iter().filter(|r| r.library == *lib).count();
        let ok = all_results.iter().filter(|r| r.library == *lib && r.status == "OK").count();
        let failed = total - ok;
        println!("  {:<14} Tested: {:<4} Succeeded: {:<4} Not available: {}", lib, total, ok, failed);
    }

    println!("\n{}", "=".repeat(110));
    println!("  Done. Run with: cargo run --example crosslib_comparison -p scry-learn --release");
    println!("{}", "=".repeat(110));
}
