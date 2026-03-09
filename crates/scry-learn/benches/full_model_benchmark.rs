#![allow(clippy::significant_drop_tightening)]
//! Full model coverage benchmark — trains and predicts with EVERY model family.
//!
//! This benchmark fills the gap where 18 of 31+ models had zero Criterion
//! measurements. For each model it measures:
//!   1. `fit()` latency at 2K synthetic samples
//!   2. `predict()` latency (1K-row batch)
//!
//! Run:  `cargo bench --bench full_model_benchmark -p scry-learn`
//! Dry-run:  `cargo bench --bench full_model_benchmark -p scry-learn -- --test`

#![allow(missing_docs)]

#[path = "benchmark_config.rs"]
mod benchmark_config;

use benchmark_config::{gen_classification, SEED, to_row_major, configs, gen_regression, gen_multiclass, gen_anomaly};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// ═══════════════════════════════════════════════════════════════════
// Classification Models (fit + predict)
// ═══════════════════════════════════════════════════════════════════

fn bench_classifiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("full/classifiers/fit");
    group.sample_size(10);

    let data = gen_classification(2000, 10, SEED);
    let rows = to_row_major(&data);

    // ── Decision Tree ──
    group.bench_function("DecisionTree/2k", |b| {
        b.iter(|| {
            let d = gen_classification(2000, 10, SEED);
            let mut m = scry_learn::tree::DecisionTreeClassifier::new()
                .max_depth(configs::DT_MAX_DEPTH);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── Random Forest ──
    group.bench_function("RandomForest/2k", |b| {
        b.iter(|| {
            let d = gen_classification(2000, 10, SEED);
            let mut m = scry_learn::tree::RandomForestClassifier::new()
                .n_estimators(configs::RF_N_ESTIMATORS)
                .max_depth(configs::RF_MAX_DEPTH)
                .seed(SEED);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── GBT Classifier ──
    group.bench_function("GradientBoosting/2k", |b| {
        b.iter(|| {
            let d = gen_classification(2000, 10, SEED);
            let mut m = scry_learn::tree::GradientBoostingClassifier::new()
                .n_estimators(configs::GBT_N_ESTIMATORS)
                .max_depth(configs::GBT_MAX_DEPTH)
                .learning_rate(configs::GBT_LR);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── HistGBT Classifier ──
    group.bench_function("HistGBT/2k", |b| {
        b.iter(|| {
            let d = gen_classification(2000, 10, SEED);
            let mut m = scry_learn::tree::HistGradientBoostingClassifier::new()
                .n_estimators(configs::HGBT_N_ESTIMATORS)
                .max_depth(configs::HGBT_MAX_DEPTH)
                .learning_rate(configs::HGBT_LR);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── KNN Classifier ──
    group.bench_function("KNN/2k", |b| {
        b.iter(|| {
            let d = gen_classification(2000, 10, SEED);
            let mut m = scry_learn::neighbors::KnnClassifier::new().k(configs::KNN_K);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── Gaussian NB ──
    group.bench_function("GaussianNB/2k", |b| {
        b.iter(|| {
            let d = gen_classification(2000, 10, SEED);
            let mut m = scry_learn::naive_bayes::GaussianNb::new();
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── Bernoulli NB ──
    group.bench_function("BernoulliNB/2k", |b| {
        b.iter(|| {
            let d = gen_classification(2000, 10, SEED);
            let mut m = scry_learn::naive_bayes::BernoulliNB::new();
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── Logistic Regression ──
    group.bench_function("LogisticReg/2k", |b| {
        b.iter(|| {
            let d = gen_classification(2000, 10, SEED);
            let mut m = scry_learn::linear::LogisticRegression::new()
                .max_iter(configs::LOGREG_MAX_ITER)
                .learning_rate(configs::LOGREG_LR);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── LinearSVC ──
    group.bench_function("LinearSVC/2k", |b| {
        b.iter(|| {
            let d = gen_classification(2000, 10, SEED);
            let mut m = scry_learn::svm::LinearSVC::new()
                .c(configs::SVC_C)
                .max_iter(configs::SVC_MAX_ITER);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // KernelSVC gated behind `experimental` feature — O(n^2) SMO.
    #[cfg(feature = "experimental")]
    {
        group.bench_function("KernelSVC_RBF/200", |b| {
            b.iter(|| {
                let d = gen_classification(200, 10, SEED);
                let mut m = scry_learn::svm::KernelSVC::new()
                    .kernel(scry_learn::svm::Kernel::RBF { gamma: configs::KSVC_GAMMA })
                    .c(configs::KSVC_C)
                    .max_iter(100);
                m.fit(black_box(&d)).unwrap();
            });
        });
    }

    group.finish();

    // ── Predict latency ──
    let mut pred_group = c.benchmark_group("full/classifiers/predict");
    pred_group.sample_size(20);

    // Pre-fit models for prediction benchmarking.
    let mut dt = scry_learn::tree::DecisionTreeClassifier::new()
        .max_depth(configs::DT_MAX_DEPTH);
    dt.fit(&data).unwrap();

    let mut rf = scry_learn::tree::RandomForestClassifier::new()
        .n_estimators(configs::RF_N_ESTIMATORS)
        .max_depth(configs::RF_MAX_DEPTH)
        .seed(SEED);
    rf.fit(&data).unwrap();

    let mut knn = scry_learn::neighbors::KnnClassifier::new().k(configs::KNN_K);
    knn.fit(&data).unwrap();

    let mut gnb = scry_learn::naive_bayes::GaussianNb::new();
    gnb.fit(&data).unwrap();

    pred_group.bench_function("DecisionTree/1k_batch", |b| {
        b.iter(|| dt.predict(black_box(&rows[..1000])).unwrap());
    });
    pred_group.bench_function("RandomForest/1k_batch", |b| {
        b.iter(|| rf.predict(black_box(&rows[..1000])).unwrap());
    });
    pred_group.bench_function("KNN/1k_batch", |b| {
        b.iter(|| knn.predict(black_box(&rows[..1000])).unwrap());
    });
    pred_group.bench_function("GaussianNB/1k_batch", |b| {
        b.iter(|| gnb.predict(black_box(&rows[..1000])).unwrap());
    });

    pred_group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Regression Models (all previously unbenchmarked)
// ═══════════════════════════════════════════════════════════════════

fn bench_regressors(c: &mut Criterion) {
    let mut group = c.benchmark_group("full/regressors/fit");
    group.sample_size(10);

    // ── Linear Regression ──
    group.bench_function("LinearRegression/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m = scry_learn::linear::LinearRegression::new();
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── Ridge ──
    group.bench_function("Ridge/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m = scry_learn::linear::Ridge::new(configs::RIDGE_ALPHA);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── Lasso ──
    group.bench_function("Lasso/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m =
                scry_learn::linear::LassoRegression::new().alpha(configs::LASSO_ALPHA);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── ElasticNet ──
    group.bench_function("ElasticNet/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m = scry_learn::linear::ElasticNet::new()
                .alpha(configs::ENET_ALPHA)
                .l1_ratio(configs::ENET_L1_RATIO);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── DT Regressor ──
    group.bench_function("DecisionTreeRegressor/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m = scry_learn::tree::DecisionTreeRegressor::new()
                .max_depth(configs::DT_MAX_DEPTH);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── RF Regressor ──
    group.bench_function("RandomForestRegressor/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m = scry_learn::tree::RandomForestRegressor::new()
                .n_estimators(configs::RF_N_ESTIMATORS)
                .max_depth(configs::RF_MAX_DEPTH)
                .seed(SEED);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── GBT Regressor ──
    group.bench_function("GBTRegressor/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m = scry_learn::tree::GradientBoostingRegressor::new()
                .n_estimators(50)
                .max_depth(5)
                .learning_rate(0.1);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── HistGBT Regressor ──
    group.bench_function("HistGBTRegressor/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m = scry_learn::tree::HistGradientBoostingRegressor::new()
                .n_estimators(configs::HGBT_N_ESTIMATORS)
                .max_depth(configs::HGBT_MAX_DEPTH)
                .learning_rate(configs::HGBT_LR);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── KNN Regressor ──
    group.bench_function("KnnRegressor/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m =
                scry_learn::neighbors::KnnRegressor::new().k(configs::KNN_K);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── LinearSVR ──
    group.bench_function("LinearSVR/2k", |b| {
        b.iter(|| {
            let d = gen_regression(2000, 10, SEED);
            let mut m = scry_learn::svm::LinearSVR::new();
            m.fit(black_box(&d)).unwrap();
        });
    });

    // KernelSVR gated behind `experimental` feature — O(n^2) SMO.
    #[cfg(feature = "experimental")]
    {
        group.bench_function("KernelSVR_RBF/200", |b| {
            b.iter(|| {
                let d = gen_regression(200, 10, SEED);
                let mut m = scry_learn::svm::KernelSVR::new()
                    .kernel(scry_learn::svm::Kernel::RBF { gamma: configs::KSVC_GAMMA });
                m.fit(black_box(&d)).unwrap();
            });
        });
    }

    group.finish();

    // ── Regressor predict latency ──
    let mut pred_group = c.benchmark_group("full/regressors/predict");
    pred_group.sample_size(20);

    let data = gen_regression(2000, 10, SEED);
    let rows = to_row_major(&data);

    let mut lr = scry_learn::linear::LinearRegression::new();
    lr.fit(&data).unwrap();

    let mut dt_r = scry_learn::tree::DecisionTreeRegressor::new()
        .max_depth(configs::DT_MAX_DEPTH);
    dt_r.fit(&data).unwrap();

    let mut rf_r = scry_learn::tree::RandomForestRegressor::new()
        .n_estimators(configs::RF_N_ESTIMATORS)
        .max_depth(configs::RF_MAX_DEPTH)
        .seed(SEED);
    rf_r.fit(&data).unwrap();

    let mut knn_r = scry_learn::neighbors::KnnRegressor::new().k(configs::KNN_K);
    knn_r.fit(&data).unwrap();

    pred_group.bench_function("LinearRegression/1k_batch", |b| {
        b.iter(|| lr.predict(black_box(&rows[..1000])).unwrap());
    });
    pred_group.bench_function("DTRegressor/1k_batch", |b| {
        b.iter(|| dt_r.predict(black_box(&rows[..1000])).unwrap());
    });
    pred_group.bench_function("RFRegressor/1k_batch", |b| {
        b.iter(|| rf_r.predict(black_box(&rows[..1000])).unwrap());
    });
    pred_group.bench_function("KnnRegressor/1k_batch", |b| {
        b.iter(|| knn_r.predict(black_box(&rows[..1000])).unwrap());
    });

    pred_group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Clustering Models (all previously unbenchmarked except KMeans)
// ═══════════════════════════════════════════════════════════════════

fn bench_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("full/clustering/fit");
    group.sample_size(10);

    // ── KMeans ──
    group.bench_function("KMeans/2k", |b| {
        b.iter(|| {
            let d = gen_multiclass(2000, 10, 3, SEED);
            let mut m = scry_learn::cluster::KMeans::new(configs::KMEANS_K)
                .seed(SEED)
                .max_iter(configs::KMEANS_MAX_ITER);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── MiniBatchKMeans ──
    group.bench_function("MiniBatchKMeans/2k", |b| {
        b.iter(|| {
            let d = gen_multiclass(2000, 10, 3, SEED);
            let mut m = scry_learn::cluster::MiniBatchKMeans::new(configs::KMEANS_K)
                .seed(SEED)
                .max_iter(configs::KMEANS_MAX_ITER);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── DBSCAN ──
    group.bench_function("DBSCAN/2k", |b| {
        b.iter(|| {
            let d = gen_multiclass(2000, 10, 3, SEED);
            let mut m = scry_learn::cluster::Dbscan::new(configs::DBSCAN_EPS, configs::DBSCAN_MIN_SAMPLES);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── Agglomerative ──
    group.bench_function("Agglomerative/500", |b| {
        b.iter(|| {
            let d = gen_multiclass(500, 10, 3, SEED);
            let mut m = scry_learn::cluster::AgglomerativeClustering::new(3);
            m.fit(black_box(&d)).unwrap();
        });
    });

    // ── HDBSCAN ──
    group.bench_function("HDBSCAN/500", |b| {
        b.iter(|| {
            let d = gen_multiclass(500, 10, 3, SEED);
            let mut m = scry_learn::cluster::Hdbscan::new().min_samples(5);
            m.fit(black_box(&d)).unwrap();
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Anomaly Detection (previously unbenchmarked)
// ═══════════════════════════════════════════════════════════════════

fn bench_anomaly(c: &mut Criterion) {
    let mut group = c.benchmark_group("full/anomaly");
    group.sample_size(10);

    let data = gen_anomaly(2000, 10, SEED);
    let rows = to_row_major(&data);

    // ── IsolationForest fit ──
    group.bench_function("IsolationForest_fit/2k", |b| {
        b.iter(|| {
            let mut m =
                scry_learn::anomaly::IsolationForest::new()
                    .n_estimators(configs::IFOREST_N_ESTIMATORS)
                    .seed(SEED);
            m.fit(black_box(&rows)).unwrap();
        });
    });

    // ── IsolationForest predict ──
    let mut iforest =
        scry_learn::anomaly::IsolationForest::new()
            .n_estimators(configs::IFOREST_N_ESTIMATORS)
            .seed(SEED);
    iforest.fit(&rows).unwrap();

    group.bench_function("IsolationForest_predict/2k", |b| {
        b.iter(|| {
            black_box(iforest.predict(black_box(&rows)));
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Preprocessing Throughput (Phase 6 preview)
// ═══════════════════════════════════════════════════════════════════

fn bench_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("full/preprocessing");
    group.sample_size(10);

    // ── StandardScaler ──
    group.bench_function("StandardScaler/10k", |b| {
        b.iter(|| {
            let d = gen_regression(10_000, 20, SEED);
            let mut scaler = scry_learn::preprocess::StandardScaler::new();
            scry_learn::preprocess::Transformer::fit(&mut scaler, &d).unwrap();
            let mut d2 = d.clone();
            scry_learn::preprocess::Transformer::transform(&scaler, &mut d2).unwrap();
        });
    });

    // ── MinMaxScaler ──
    group.bench_function("MinMaxScaler/10k", |b| {
        b.iter(|| {
            let d = gen_regression(10_000, 20, SEED);
            let mut scaler = scry_learn::preprocess::MinMaxScaler::new();
            scry_learn::preprocess::Transformer::fit(&mut scaler, &d).unwrap();
            let mut d2 = d.clone();
            scry_learn::preprocess::Transformer::transform(&scaler, &mut d2).unwrap();
        });
    });

    // ── PCA ──
    group.bench_function("PCA_5comp/10k_20d", |b| {
        b.iter(|| {
            let d = gen_regression(10_000, 20, SEED);
            let mut pca = scry_learn::preprocess::Pca::with_n_components(5);
            scry_learn::preprocess::Transformer::fit(&mut pca, &d).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_classifiers,
    bench_regressors,
    bench_clustering,
    bench_anomaly,
    bench_preprocessing,
);
criterion_main!(benches);
