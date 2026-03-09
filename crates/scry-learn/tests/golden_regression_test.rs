#![allow(
    clippy::items_after_statements,
    clippy::type_complexity,
    clippy::redundant_locals,
    dead_code
)]
//! Golden baseline regression test — verifies every model still hits
//! its expected accuracy/metric within tolerance.
//!
//! This is the **single source of truth** for algorithmic correctness.
//! If a code change causes this test to fail, it means a model's quality
//! has regressed and must be investigated before merging.
//!
//! Run:
//!   cargo test --test `golden_regression_test` -p scry-learn --release -- --nocapture

#[path = "../benches/benchmark_config.rs"]
mod benchmark_config;

use benchmark_config::*;
use scry_learn::metrics::{accuracy, adjusted_rand_index, roc_auc_score};
use scry_learn::preprocess::{StandardScaler, Transformer};
use scry_learn::split::{cross_val_score_stratified, train_test_split, ScoringFn};

// ═══════════════════════════════════════════════════════════════════
// Classification Golden Baselines (5-fold stratified CV)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn golden_classification_accuracy() {
    let scorer: ScoringFn = accuracy;
    let baselines = golden_baselines_classification();

    println!("\n{}", "═".repeat(80));
    println!("  GOLDEN BASELINE REGRESSION TEST — Classification (5-fold stratified CV)");
    println!("{}", "═".repeat(80));
    println!(
        "  {:<20} {:<16} {:>10} {:>10} {:>10}  Status",
        "Model", "Dataset", "Expected", "Actual", "Δ"
    );
    println!("  {}", "─".repeat(76));

    let mut failures = Vec::new();

    for baseline in &baselines {
        let data = load_dataset(baseline.dataset);

        // Apply scaling for models that need it.
        let (model_data, _scaler) = match baseline.model {
            "KNN" | "LogisticRegression" | "LinearSVC" => {
                let mut scaled = data.clone();
                let mut scaler = StandardScaler::new();
                scaler.fit(&scaled).unwrap();
                scaler.transform(&mut scaled).unwrap();
                (scaled, true)
            }
            _ => (data, false),
        };

        // Run 5-fold stratified CV with the canonical model config.
        let scores = match baseline.model {
            "DecisionTree" => {
                let model = scry_learn::tree::DecisionTreeClassifier::new()
                    .max_depth(configs::DT_MAX_DEPTH);
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            "RandomForest" => {
                let model = scry_learn::tree::RandomForestClassifier::new()
                    .n_estimators(configs::RF_N_ESTIMATORS)
                    .max_depth(configs::RF_MAX_DEPTH)
                    .seed(SEED);
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            "GradientBoosting" => {
                let model = scry_learn::tree::GradientBoostingClassifier::new()
                    .n_estimators(configs::GBT_N_ESTIMATORS)
                    .max_depth(configs::GBT_MAX_DEPTH)
                    .learning_rate(configs::GBT_LR);
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            "HistGBT" => {
                let model = scry_learn::tree::HistGradientBoostingClassifier::new()
                    .n_estimators(configs::HGBT_N_ESTIMATORS)
                    .max_depth(configs::HGBT_MAX_DEPTH)
                    .learning_rate(configs::HGBT_LR);
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            "GaussianNB" => {
                let model = scry_learn::naive_bayes::GaussianNb::new();
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            "KNN" => {
                let model = scry_learn::neighbors::KnnClassifier::new().k(configs::KNN_K);
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            "LogisticRegression" => {
                let model = scry_learn::linear::LogisticRegression::new()
                    .max_iter(configs::LOGREG_MAX_ITER)
                    .learning_rate(configs::LOGREG_LR);
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            "LinearSVC" => {
                let model = scry_learn::svm::LinearSVC::new()
                    .c(configs::SVC_C)
                    .max_iter(configs::SVC_MAX_ITER);
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            "MultinomialNB" => {
                let model = scry_learn::naive_bayes::MultinomialNB::new()
                    .alpha(configs::MNB_ALPHA);
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            "BernoulliNB" => {
                let model = scry_learn::naive_bayes::BernoulliNB::new()
                    .alpha(configs::BNB_ALPHA)
                    .binarize(Some(configs::BNB_BINARIZE));
                cross_val_score_stratified(&model, &model_data, 5, scorer, SEED).unwrap()
            }
            other => panic!("Unknown classification model in golden baselines: {other}"),
        };

        let actual = scores.iter().sum::<f64>() / scores.len() as f64;
        let delta = actual - baseline.expected;
        let passed = delta.abs() <= baseline.tolerance;

        let status = if passed { "✓ PASS" } else { "✗ FAIL" };
        println!(
            "  {:<20} {:<16} {:>10.4} {:>10.4} {:>+10.4}  {}",
            baseline.model, baseline.dataset, baseline.expected, actual, delta, status,
        );

        if !passed {
            failures.push(format!(
                "{}/{}: expected {:.4} ± {:.4}, got {:.4} (Δ = {:+.4})",
                baseline.model,
                baseline.dataset,
                baseline.expected,
                baseline.tolerance,
                actual,
                delta,
            ));
        }
    }

    println!();

    if !failures.is_empty() {
        println!("  FAILURES:");
        for f in &failures {
            println!("    • {f}");
        }
        panic!(
            "\n{} golden baseline(s) failed! See above for details.\n\
             If these are intentional (e.g. algorithm improvement), update the \
             golden baselines in benchmark_config.rs.",
            failures.len()
        );
    }

    println!("  All {} golden baselines passed ✓", baselines.len());
}

// ═══════════════════════════════════════════════════════════════════
// Regression Golden Baselines (California Housing, 80/20 split)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn golden_regression_r2() {
    let baselines = golden_baselines_regression();

    println!("\n{}", "═".repeat(80));
    println!("  GOLDEN BASELINE REGRESSION TEST — Regression (California Housing, 80/20)");
    println!("{}", "═".repeat(80));
    println!(
        "  {:<20} {:<16} {:>10} {:>10} {:>10}  Status",
        "Model", "Dataset", "Expected", "Actual", "Δ"
    );
    println!("  {}", "─".repeat(76));

    let data = load_dataset("california");

    // Apply StandardScaler
    let mut scaled = data;
    let mut scaler = StandardScaler::new();
    scaler.fit(&scaled).unwrap();
    scaler.transform(&mut scaled).unwrap();

    let (train, test) = train_test_split(&scaled, 0.2, SEED);

    let mut failures = Vec::new();

    for baseline in &baselines {
        let train_rows = to_row_major(&train);
        let test_rows = to_row_major(&test);

        let preds = match baseline.model {
            "LinearRegression" => {
                let mut model = scry_learn::linear::LinearRegression::new();
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "Lasso" => {
                let mut model =
                    scry_learn::linear::LassoRegression::new().alpha(configs::LASSO_ALPHA);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "ElasticNet" => {
                let mut model = scry_learn::linear::ElasticNet::new()
                    .alpha(configs::ENET_ALPHA)
                    .l1_ratio(configs::ENET_L1_RATIO);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "Ridge" => {
                let mut model =
                    scry_learn::linear::Ridge::new(configs::RIDGE_ALPHA);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "GBTRegressor" => {
                let mut model = scry_learn::tree::GradientBoostingRegressor::new()
                    .n_estimators(50)
                    .max_depth(5)
                    .learning_rate(0.1);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "KnnRegressor" => {
                let mut model =
                    scry_learn::neighbors::KnnRegressor::new().k(configs::KNN_K);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "DTRegressor" => {
                let mut model = scry_learn::tree::DecisionTreeRegressor::new()
                    .max_depth(configs::DTR_MAX_DEPTH);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "RFRegressor" => {
                let mut model = scry_learn::tree::RandomForestRegressor::new()
                    .n_estimators(configs::RFR_N_ESTIMATORS)
                    .max_depth(configs::RFR_MAX_DEPTH)
                    .seed(SEED);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "HistGBTRegressor" => {
                let mut model = scry_learn::tree::HistGradientBoostingRegressor::new()
                    .n_estimators(configs::HGBTR_N_ESTIMATORS)
                    .max_depth(configs::HGBTR_MAX_DEPTH)
                    .learning_rate(configs::HGBTR_LR);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "LinearSVR" => {
                let mut model = scry_learn::svm::LinearSVR::new()
                    .c(configs::SVR_C)
                    .epsilon(configs::SVR_EPSILON)
                    .max_iter(configs::SVR_MAX_ITER);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            "MLPRegressor" => {
                let mut model = scry_learn::neural::MLPRegressor::new()
                    .hidden_layers(configs::MLP_HIDDEN)
                    .max_iter(configs::MLP_MAX_ITER)
                    .learning_rate(configs::MLP_LR)
                    .seed(configs::MLP_SEED);
                model.fit(&train).unwrap();
                model.predict(&test_rows).unwrap()
            }
            other => panic!("Unknown regression model in golden baselines: {other}"),
        };

        let r2 = scry_learn::metrics::r2_score(&test.target, &preds);
        let delta = r2 - baseline.expected;
        let passed = delta.abs() <= baseline.tolerance;

        let status = if passed { "✓ PASS" } else { "✗ FAIL" };
        println!(
            "  {:<20} {:<16} {:>10.4} {:>10.4} {:>+10.4}  {}",
            baseline.model, baseline.dataset, baseline.expected, r2, delta, status,
        );

        if !passed {
            failures.push(format!(
                "{}/{}: expected R²={:.4} ± {:.4}, got {:.4} (Δ = {:+.4})",
                baseline.model,
                baseline.dataset,
                baseline.expected,
                baseline.tolerance,
                r2,
                delta,
            ));
        }

        let _ = (train_rows, test_rows); // suppress unused warnings
    }

    println!();

    if !failures.is_empty() {
        println!("  FAILURES:");
        for f in &failures {
            println!("    • {f}");
        }
        panic!(
            "\n{} golden regression baseline(s) failed! See above for details.",
            failures.len()
        );
    }

    println!(
        "  All {} golden regression baselines passed ✓",
        baselines.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Determinism Check — verify same seed produces identical predictions
// ═══════════════════════════════════════════════════════════════════

#[test]
fn golden_determinism_check() {
    println!("\n{}", "═".repeat(80));
    println!("  DETERMINISM CHECK — verify identical seeds → identical predictions");
    println!("{}", "═".repeat(80));

    let data = load_dataset("iris");
    let rows = to_row_major(&data);

    // Run the same model twice, check predictions match exactly.
    let models: Vec<(&str, Box<dyn Fn() -> Vec<f64>>)> = vec![
        (
            "DecisionTree",
            Box::new({
                let data = data.clone();
                let rows = rows.clone();
                move || {
                    let mut m = scry_learn::tree::DecisionTreeClassifier::new()
                        .max_depth(configs::DT_MAX_DEPTH);
                    m.fit(&data).unwrap();
                    m.predict(&rows).unwrap()
                }
            }),
        ),
        (
            "RandomForest",
            Box::new({
                let data = data.clone();
                let rows = rows.clone();
                move || {
                    let mut m = scry_learn::tree::RandomForestClassifier::new()
                        .n_estimators(configs::RF_N_ESTIMATORS)
                        .max_depth(configs::RF_MAX_DEPTH)
                        .seed(SEED);
                    m.fit(&data).unwrap();
                    m.predict(&rows).unwrap()
                }
            }),
        ),
        (
            "GaussianNB",
            Box::new({
                let data = data;
                let rows = rows;
                move || {
                    let mut m = scry_learn::naive_bayes::GaussianNb::new();
                    m.fit(&data).unwrap();
                    m.predict(&rows).unwrap()
                }
            }),
        ),
    ];

    for (name, run) in &models {
        let preds_a = run();
        let preds_b = run();

        let match_count = preds_a
            .iter()
            .zip(preds_b.iter())
            .filter(|(&a, &b)| a.to_bits() == b.to_bits())
            .count();

        let total = preds_a.len();
        let status = if match_count == total {
            "✓ bitwise identical"
        } else {
            "✗ NON-DETERMINISTIC"
        };

        println!("  {name:<20} {match_count}/{total} predictions match  {status}");
        assert_eq!(
            match_count, total,
            "{name} is not deterministic: {match_count}/{total} predictions differ!"
        );
    }

    println!("\n  All determinism checks passed ✓");
}

// ═══════════════════════════════════════════════════════════════════
// MLP Classifier Golden Baseline (80/20 split — too slow for 5-fold CV)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn golden_mlp_classifier() {
    println!("\n{}", "═".repeat(80));
    println!("  GOLDEN BASELINE — MLPClassifier (80/20 split, scaled)");
    println!("{}", "═".repeat(80));

    let datasets = &["iris", "wine", "breast_cancer"];
    let mut failures = Vec::new();

    for &ds_name in datasets {
        let data = load_dataset(ds_name);
        let mut scaled = data.clone();
        let mut scaler = StandardScaler::new();
        scaler.fit(&scaled).unwrap();
        scaler.transform(&mut scaled).unwrap();

        let (train, test) = train_test_split(&scaled, 0.2, SEED);
        let test_rows = to_row_major(&test);

        let mut model = scry_learn::neural::MLPClassifier::new()
            .hidden_layers(configs::MLP_HIDDEN)
            .max_iter(configs::MLP_MAX_ITER)
            .learning_rate(configs::MLP_LR)
            .seed(configs::MLP_SEED);
        model.fit(&train).unwrap();
        let preds = model.predict(&test_rows).unwrap();
        let acc = accuracy(&test.target, &preds);

        // MLP accuracy should be reasonable (> 0.70) on these datasets
        let min_acc = 0.70;
        let passed = acc >= min_acc;
        let status = if passed { "PASS" } else { "FAIL" };
        println!("  {ds_name:<20} accuracy={acc:.4}  (min={min_acc:.2})  {status}");

        if !passed {
            failures.push(format!("MLPClassifier/{ds_name}: accuracy={acc:.4} < {min_acc:.2}"));
        }
    }

    if !failures.is_empty() {
        for f in &failures {
            println!("    FAIL: {f}");
        }
        panic!("{} MLPClassifier baseline(s) failed", failures.len());
    }
    println!("  All MLPClassifier baselines passed");
}

// ═══════════════════════════════════════════════════════════════════
// Clustering Golden Baselines
// ═══════════════════════════════════════════════════════════════════

#[test]
fn golden_clustering_metrics() {
    println!("\n{}", "═".repeat(80));
    println!("  GOLDEN BASELINE — Clustering (silhouette + ARI)");
    println!("{}", "═".repeat(80));

    let mut failures = Vec::new();

    // Helper: convert labels to f64 for ARI
    fn labels_usize_to_f64(labels: &[usize]) -> Vec<f64> {
        labels.iter().map(|&l| l as f64).collect()
    }
    fn labels_i32_to_f64(labels: &[i32]) -> Vec<f64> {
        labels.iter().map(|&l| l as f64).collect()
    }

    for &ds_name in &["iris", "wine"] {
        let data = load_dataset(ds_name);
        let mut scaled = data.clone();
        let mut scaler = StandardScaler::new();
        scaler.fit(&scaled).unwrap();
        scaler.transform(&mut scaled).unwrap();

        let rows = to_row_major(&scaled);

        // KMeans
        {
            let mut model = scry_learn::cluster::KMeans::new(configs::KMEANS_K)
                .max_iter(configs::KMEANS_MAX_ITER)
                .seed(SEED);
            model.fit(&scaled).unwrap();
            let labels = labels_usize_to_f64(model.labels());
            let sil = scry_learn::cluster::silhouette_score(&rows, model.labels());
            let ari = adjusted_rand_index(&scaled.target, &labels);
            let passed = sil > 0.0;
            let status = if passed { "PASS" } else { "FAIL" };
            println!("  KMeans/{ds_name:16} sil={sil:.4} ari={ari:.4}  {status}");
            if !passed {
                failures.push(format!("KMeans/{ds_name}: silhouette={sil:.4} <= 0"));
            }
        }

        // MiniBatchKMeans
        {
            let mut model = scry_learn::cluster::MiniBatchKMeans::new(configs::MBKM_K)
                .batch_size(configs::MBKM_BATCH_SIZE)
                .seed(SEED);
            model.fit(&scaled).unwrap();
            let labels = labels_usize_to_f64(model.labels());
            let sil = scry_learn::cluster::silhouette_score(&rows, model.labels());
            let ari = adjusted_rand_index(&scaled.target, &labels);
            let passed = sil > 0.0;
            let status = if passed { "PASS" } else { "FAIL" };
            println!("  MiniBatchKMeans/{ds_name:10} sil={sil:.4} ari={ari:.4}  {status}");
            if !passed {
                failures.push(format!("MiniBatchKMeans/{ds_name}: silhouette={sil:.4} <= 0"));
            }
        }

        // DBSCAN — use dataset-specific eps (scaled features have different ranges)
        {
            let eps = match ds_name {
                "wine" => 3.0,
                _ => configs::DBSCAN_EPS,
            };
            let mut model = scry_learn::cluster::Dbscan::new(eps, configs::DBSCAN_MIN_SAMPLES);
            model.fit(&scaled).unwrap();
            let n_clusters = model.n_clusters();
            // For DBSCAN, just verify it finds at least 1 cluster
            let passed = n_clusters >= 1;
            let status = if passed { "PASS" } else { "FAIL" };
            println!("  DBSCAN/{ds_name:15} n_clusters={n_clusters}  {status}");
            if !passed {
                failures.push(format!("DBSCAN/{ds_name}: n_clusters={n_clusters} < 1"));
            }
        }

        // HDBSCAN
        {
            let mut model = scry_learn::cluster::Hdbscan::new()
                .min_cluster_size(configs::HDBSCAN_MIN_CLUSTER_SIZE);
            model.fit(&scaled).unwrap();
            let n_clusters = model.n_clusters();
            let passed = n_clusters >= 1;
            let status = if passed { "PASS" } else { "FAIL" };
            println!("  HDBSCAN/{ds_name:14} n_clusters={n_clusters}  {status}");
            if !passed {
                failures.push(format!("HDBSCAN/{ds_name}: n_clusters={n_clusters} < 1"));
            }
        }

        // Agglomerative
        {
            let mut model = scry_learn::cluster::AgglomerativeClustering::new(
                configs::AGGLO_N_CLUSTERS,
            );
            model.fit(&scaled).unwrap();
            let labels = labels_usize_to_f64(model.labels());
            let sil = scry_learn::cluster::silhouette_score(&rows, model.labels());
            let ari = adjusted_rand_index(&scaled.target, &labels);
            let passed = sil > 0.0;
            let status = if passed { "PASS" } else { "FAIL" };
            println!("  Agglomerative/{ds_name:9} sil={sil:.4} ari={ari:.4}  {status}");
            if !passed {
                failures.push(format!("Agglomerative/{ds_name}: silhouette={sil:.4} <= 0"));
            }
        }
    }

    println!();
    if !failures.is_empty() {
        for f in &failures {
            println!("    FAIL: {f}");
        }
        panic!("{} clustering baseline(s) failed", failures.len());
    }
    println!("  All clustering baselines passed");
}

// ═══════════════════════════════════════════════════════════════════
// Anomaly Detection Golden Baseline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn golden_anomaly_detection() {
    println!("\n{}", "═".repeat(80));
    println!("  GOLDEN BASELINE — Anomaly Detection (IsolationForest)");
    println!("{}", "═".repeat(80));

    let data = gen_anomaly(1000, 5, SEED);
    let rows = to_row_major(&data);

    let mut ifo = scry_learn::anomaly::IsolationForest::new()
        .n_estimators(configs::IFOREST_N_ESTIMATORS)
        .contamination(0.05)
        .seed(SEED);
    ifo.fit(&rows).unwrap();

    let scores = ifo.predict(&rows);
    let auc = roc_auc_score(&data.target, &scores);

    // Anomaly scores should separate normal from outlier well
    let min_auc = 0.90;
    let passed = auc >= min_auc;
    let status = if passed { "PASS" } else { "FAIL" };
    println!("  IsolationForest  ROC-AUC={auc:.4}  (min={min_auc:.2})  {status}");

    assert!(passed, "IsolationForest ROC-AUC={auc:.4} < {min_auc:.2}");
    println!("  Anomaly detection baseline passed");
}

// ═══════════════════════════════════════════════════════════════════
// Text Vectorizer Golden Baselines
// ═══════════════════════════════════════════════════════════════════

#[test]
fn golden_text_vectorizers() {
    println!("\n{}", "═".repeat(80));
    println!("  GOLDEN BASELINE — Text Vectorizers");
    println!("{}", "═".repeat(80));

    let docs = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat played with the dog",
        "a bird flew over the mat",
        "the bird sat on the log",
    ];

    // CountVectorizer
    {
        let mut cv = scry_learn::text::CountVectorizer::new();
        let matrix = cv.fit_transform(&docs);

        let vocab_size = cv.n_features();
        assert!(vocab_size > 0, "CountVectorizer: empty vocabulary");
        assert_eq!(matrix.n_rows(), 5, "CountVectorizer: wrong n_rows");
        assert_eq!(matrix.n_cols(), vocab_size, "CountVectorizer: cols != vocab");
        println!("  CountVectorizer: vocab_size={vocab_size}, matrix={}x{}", matrix.n_rows(), matrix.n_cols());
    }

    // TfidfVectorizer
    {
        let mut tfidf = scry_learn::text::TfidfVectorizer::new();
        let matrix = tfidf.fit_transform(&docs);

        let vocab_size = tfidf.n_features();
        assert!(vocab_size > 0, "TfidfVectorizer: empty vocabulary");
        assert_eq!(matrix.n_rows(), 5, "TfidfVectorizer: wrong n_rows");
        assert_eq!(matrix.n_cols(), vocab_size, "TfidfVectorizer: cols != vocab");

        // Check L2 norm of each row is approximately 1.0
        let dense = matrix.to_dense();
        for (i, row) in dense.iter().enumerate() {
            let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "TfidfVectorizer: row {i} L2 norm = {norm:.6}, expected ~1.0"
            );
        }
        println!("  TfidfVectorizer: vocab_size={vocab_size}, all rows L2-normalized");
    }

    println!("  All text vectorizer baselines passed");
}

// ═══════════════════════════════════════════════════════════════════
// Pipeline Golden Baseline
// ═══════════════════════════════════════════════════════════════════

#[test]
fn golden_pipeline() {
    println!("\n{}", "═".repeat(80));
    println!("  GOLDEN BASELINE — Pipeline (StandardScaler + RandomForest)");
    println!("{}", "═".repeat(80));

    let data = load_dataset("iris");
    let (train, test) = train_test_split(&data, 0.2, SEED);

    let mut pipeline = scry_learn::pipeline::Pipeline::new()
        .add_transformer(StandardScaler::new())
        .set_model(
            scry_learn::tree::RandomForestClassifier::new()
                .n_estimators(configs::RF_N_ESTIMATORS)
                .max_depth(configs::RF_MAX_DEPTH)
                .seed(SEED),
        );

    pipeline.fit(&train).unwrap();
    let preds = pipeline.predict(&test).unwrap();
    let acc = accuracy(&test.target, &preds);

    let min_acc = 0.85;
    let passed = acc >= min_acc;
    let status = if passed { "PASS" } else { "FAIL" };
    println!("  Pipeline accuracy={acc:.4}  (min={min_acc:.2})  {status}");

    assert!(passed, "Pipeline accuracy={acc:.4} < {min_acc:.2}");
    println!("  Pipeline baseline passed");
}
