//! Comprehensive ML algorithm benchmarks for `scry-learn`.
//!
//! Designed for laptop execution (Intel Core Ultra 7 155H, 22 threads, 14GiB).
//!
//! | Group | What it measures |
//! |-------|-----------------|
//! | 1 | All 8 algorithms — training throughput |
//! | 2 | All 8 algorithms — prediction latency |
//! | 3 | Dataset scaling (100 → 1K → 5K → 10K samples) |
//! | 4 | Feature scaling (2 → 10 → 25 → 50 features) |
//! | 5 | Random Forest tree-count scaling (10 → 50 → 100) |
//! | 6 | Preprocessing pipeline overhead |
//! | 7 | Metrics computation |
//! | 8 | End-to-end pipeline: load → preprocess → train → predict → evaluate |
//!
//! Run with: `cargo bench --bench ml_algorithms -p scry-learn`

#![allow(missing_docs)]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};

use scry_learn::cluster::{Dbscan, KMeans};
use scry_learn::dataset::Dataset;
use scry_learn::linear::{ElasticNet, LassoRegression, LinearRegression, LogisticRegression};
use scry_learn::metrics::{
    accuracy, classification_report, confusion_matrix, f1_score, mean_squared_error, r2_score,
    roc_auc_score, Average,
};
use scry_learn::naive_bayes::GaussianNb;
use scry_learn::neighbors::KnnClassifier;
use scry_learn::pipeline::Pipeline;
use scry_learn::preprocess::{StandardScaler, Transformer};
use scry_learn::split::train_test_split;
use scry_learn::svm::LinearSVC;
use scry_learn::tree::{DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier};

// ─────────────────────────────────────────────────────────────────
// Deterministic data generators
// ─────────────────────────────────────────────────────────────────

/// Generate a classification dataset with `n` samples and `n_features` features.
/// Two classes with clear nonlinear separation.
fn gen_classification(n: usize, n_features: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut features: Vec<Vec<f64>> = (0..n_features).map(|_| Vec::with_capacity(n)).collect();
    let mut target = Vec::with_capacity(n);

    for i in 0..n {
        let class = if i < n / 2 { 0.0 } else { 1.0 };
        for (j, col) in features.iter_mut().enumerate() {
            let offset = if class == 0.0 {
                0.0
            } else {
                3.0 + j as f64 * 0.5
            };
            col.push(rng.f64() * 2.0 + offset);
        }
        target.push(class);
    }

    let names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "class")
}

/// Generate a multiclass classification dataset with `n_classes` classes.
fn gen_multiclass(n: usize, n_features: usize, n_classes: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut features: Vec<Vec<f64>> = (0..n_features).map(|_| Vec::with_capacity(n)).collect();
    let mut target = Vec::with_capacity(n);
    let per_class = n / n_classes;

    for c in 0..n_classes {
        let count = if c == n_classes - 1 {
            n - per_class * c
        } else {
            per_class
        };
        for _ in 0..count {
            for (j, col) in features.iter_mut().enumerate() {
                let offset = c as f64 * 3.0 + j as f64 * 0.2;
                col.push(rng.f64() * 2.0 + offset);
            }
            target.push(c as f64);
        }
    }

    let names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "class")
}

/// Generate a regression dataset: y = `sum(x_i` * (i+1)) + noise.
fn gen_regression(n: usize, n_features: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut features: Vec<Vec<f64>> = (0..n_features).map(|_| Vec::with_capacity(n)).collect();
    let mut target = Vec::with_capacity(n);

    for _ in 0..n {
        let mut y = 0.0;
        for (j, col) in features.iter_mut().enumerate() {
            let x = rng.f64() * 10.0;
            col.push(x);
            y += x * (j + 1) as f64;
        }
        y += rng.f64() * 0.5; // small noise
        target.push(y);
    }

    let names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "y")
}

// ─────────────────────────────────────────────────────────────────
// Group 1: Training throughput — all algorithms
// ─────────────────────────────────────────────────────────────────

fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("train");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let clf_data = gen_classification(1000, 10, 42);
    let reg_data = gen_regression(1000, 10, 42);

    group.bench_function("decision_tree_clf/1k×10", |b| {
        b.iter(|| {
            let mut dt = DecisionTreeClassifier::new();
            dt.fit(black_box(&clf_data)).unwrap();
        });
    });

    group.bench_function("decision_tree_reg/1k×10", |b| {
        b.iter(|| {
            let mut dt = DecisionTreeRegressor::new();
            dt.fit(black_box(&reg_data)).unwrap();
        });
    });

    group.bench_function("random_forest_clf/1k×10/20trees", |b| {
        b.iter(|| {
            let mut rf = RandomForestClassifier::new().n_estimators(20).seed(42);
            rf.fit(black_box(&clf_data)).unwrap();
        });
    });

    group.bench_function("linear_regression/1k×10", |b| {
        b.iter(|| {
            let mut lr = LinearRegression::new();
            lr.fit(black_box(&reg_data)).unwrap();
        });
    });

    group.bench_function("logistic_regression/1k×10", |b| {
        b.iter(|| {
            let mut lr = LogisticRegression::new().max_iter(200).learning_rate(0.1);
            lr.fit(black_box(&clf_data)).unwrap();
        });
    });

    group.bench_function("knn_clf/1k×10", |b| {
        b.iter(|| {
            let mut knn = KnnClassifier::new().k(5);
            knn.fit(black_box(&clf_data)).unwrap();
        });
    });

    group.bench_function("kmeans/1k×10/k=3", |b| {
        b.iter(|| {
            let mut km = KMeans::new(3).seed(42).max_iter(100);
            km.fit(black_box(&clf_data)).unwrap();
        });
    });

    group.bench_function("dbscan/1k×10", |b| {
        b.iter(|| {
            let mut db = Dbscan::new(3.0, 5);
            db.fit(black_box(&clf_data)).unwrap();
        });
    });

    group.bench_function("gaussian_nb/1k×10", |b| {
        b.iter(|| {
            let mut nb = GaussianNb::new();
            nb.fit(black_box(&clf_data)).unwrap();
        });
    });

    group.bench_function("lasso/1k×10", |b| {
        b.iter(|| {
            let mut lasso = LassoRegression::new().alpha(0.1).max_iter(500);
            lasso.fit(black_box(&reg_data)).unwrap();
        });
    });

    group.bench_function("elastic_net/1k×10", |b| {
        b.iter(|| {
            let mut en = ElasticNet::new().alpha(0.1).l1_ratio(0.5).max_iter(500);
            en.fit(black_box(&reg_data)).unwrap();
        });
    });

    group.bench_function("linear_svc/1k×10", |b| {
        let mut scaler = StandardScaler::new();
        let mut scaled = clf_data.clone();
        scaler.fit(&scaled).unwrap();
        scaler.transform(&mut scaled).unwrap();
        b.iter(|| {
            let mut svc = LinearSVC::new().c(1.0).max_iter(500);
            svc.fit(black_box(&scaled)).unwrap();
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 2: Prediction latency — all algorithms
// ─────────────────────────────────────────────────────────────────

fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let clf_data = gen_classification(1000, 10, 42);
    let reg_data = gen_regression(1000, 10, 42);
    let clf_features = clf_data.feature_matrix();
    let reg_features = reg_data.feature_matrix();

    // Pre-train all models.
    let mut dt_clf = DecisionTreeClassifier::new();
    dt_clf.fit(&clf_data).unwrap();

    let mut dt_reg = DecisionTreeRegressor::new();
    dt_reg.fit(&reg_data).unwrap();

    let mut rf_clf = RandomForestClassifier::new().n_estimators(20).seed(42);
    rf_clf.fit(&clf_data).unwrap();

    let mut lr = LinearRegression::new();
    lr.fit(&reg_data).unwrap();

    let mut log_reg = LogisticRegression::new().max_iter(200).learning_rate(0.1);
    log_reg.fit(&clf_data).unwrap();

    let mut knn = KnnClassifier::new().k(5);
    knn.fit(&clf_data).unwrap();

    let mut nb = GaussianNb::new();
    nb.fit(&clf_data).unwrap();

    group.bench_function("decision_tree_clf/1k", |b| {
        b.iter(|| dt_clf.predict(black_box(&clf_features)).unwrap());
    });

    group.bench_function("decision_tree_reg/1k", |b| {
        b.iter(|| dt_reg.predict(black_box(&reg_features)).unwrap());
    });

    group.bench_function("random_forest_clf/1k", |b| {
        b.iter(|| rf_clf.predict(black_box(&clf_features)).unwrap());
    });

    group.bench_function("linear_regression/1k", |b| {
        b.iter(|| lr.predict(black_box(&reg_features)).unwrap());
    });

    group.bench_function("logistic_regression/1k", |b| {
        b.iter(|| log_reg.predict(black_box(&clf_features)).unwrap());
    });

    group.bench_function("knn/1k", |b| {
        b.iter(|| knn.predict(black_box(&clf_features)).unwrap());
    });

    group.bench_function("gaussian_nb/1k", |b| {
        b.iter(|| nb.predict(black_box(&clf_features)).unwrap());
    });

    let mut lasso = LassoRegression::new().alpha(0.1).max_iter(500);
    lasso.fit(&reg_data).unwrap();

    let mut en = ElasticNet::new().alpha(0.1).l1_ratio(0.5).max_iter(500);
    en.fit(&reg_data).unwrap();

    let mut scaler = StandardScaler::new();
    let mut scaled = clf_data.clone();
    scaler.fit(&scaled).unwrap();
    scaler.transform(&mut scaled).unwrap();
    let mut svc = LinearSVC::new().c(1.0).max_iter(500);
    svc.fit(&scaled).unwrap();
    let scaled_features = scaled.feature_matrix();

    group.bench_function("lasso/1k", |b| {
        b.iter(|| lasso.predict(black_box(&reg_features)).unwrap());
    });

    group.bench_function("elastic_net/1k", |b| {
        b.iter(|| en.predict(black_box(&reg_features)).unwrap());
    });

    group.bench_function("linear_svc/1k", |b| {
        b.iter(|| svc.predict(black_box(&scaled_features)).unwrap());
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 3: Dataset size scaling
// ─────────────────────────────────────────────────────────────────

fn bench_dataset_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/samples");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(3));
    group.sampling_mode(SamplingMode::Flat);

    let sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000];

    for &n in &sizes {
        let data = gen_classification(n, 10, 42);

        group.bench_with_input(BenchmarkId::new("decision_tree", n), &data, |b, data| {
            b.iter(|| {
                let mut dt = DecisionTreeClassifier::new().max_depth(10);
                dt.fit(black_box(data)).unwrap();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("random_forest/10t", n),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut rf = RandomForestClassifier::new()
                        .n_estimators(10)
                        .max_depth(8)
                        .seed(42);
                    rf.fit(black_box(data)).unwrap();
                });
            },
        );

        if n <= 5000 {
            // KNN training is O(1) but prediction is O(n²) — skip huge datasets
            group.bench_with_input(BenchmarkId::new("knn_predict", n), &data, |b, data| {
                let mut knn = KnnClassifier::new().k(5);
                knn.fit(data).unwrap();
                let features = data.feature_matrix();
                b.iter(|| knn.predict(black_box(&features)).unwrap());
            });
        }

        if n <= 10_000 {
            group.bench_with_input(
                BenchmarkId::new("logistic_regression", n),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut lr = LogisticRegression::new().max_iter(100).learning_rate(0.1);
                        lr.fit(black_box(data)).unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 4: Feature count scaling
// ─────────────────────────────────────────────────────────────────

fn bench_feature_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/features");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(3));
    group.sampling_mode(SamplingMode::Flat);

    let feature_counts = [2, 5, 10, 25, 50, 100, 500];

    for &nf in &feature_counts {
        let data = gen_classification(1000, nf, 42);

        group.bench_with_input(BenchmarkId::new("decision_tree", nf), &data, |b, data| {
            b.iter(|| {
                let mut dt = DecisionTreeClassifier::new().max_depth(10);
                dt.fit(black_box(data)).unwrap();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("linear_regression", nf),
            &gen_regression(1000, nf, 42),
            |b, data| {
                b.iter(|| {
                    let mut lr = LinearRegression::new();
                    lr.fit(black_box(data)).unwrap();
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("gaussian_nb", nf), &data, |b, data| {
            b.iter(|| {
                let mut nb = GaussianNb::new();
                nb.fit(black_box(data)).unwrap();
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 5: Random Forest tree count scaling
// ─────────────────────────────────────────────────────────────────

fn bench_forest_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling/forest_trees");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);

    let data = gen_classification(2000, 10, 42);
    let tree_counts = [5, 10, 25, 50, 100];

    for &n_trees in &tree_counts {
        group.bench_with_input(
            BenchmarkId::new("train", n_trees),
            &n_trees,
            |b, &n_trees| {
                b.iter(|| {
                    let mut rf = RandomForestClassifier::new()
                        .n_estimators(n_trees)
                        .max_depth(8)
                        .seed(42);
                    rf.fit(black_box(&data)).unwrap();
                });
            },
        );
    }

    // Prediction scaling with pre-trained forests.
    let features = data.feature_matrix();
    for &n_trees in &tree_counts {
        let mut rf = RandomForestClassifier::new()
            .n_estimators(n_trees)
            .max_depth(8)
            .seed(42);
        rf.fit(&data).unwrap();

        group.bench_with_input(BenchmarkId::new("predict", n_trees), &rf, |b, rf| {
            b.iter(|| rf.predict(black_box(&features)).unwrap());
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 6: Preprocessing overhead
// ─────────────────────────────────────────────────────────────────

fn bench_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    let data = gen_classification(5000, 20, 42);

    group.bench_function("standard_scaler/fit_transform/5k×20", |b| {
        b.iter(|| {
            let mut scaler = StandardScaler::new();
            let mut d = black_box(data.clone());
            scaler.fit(&d).unwrap();
            scaler.transform(&mut d).unwrap();
        });
    });

    group.bench_function("train_test_split/5k", |b| {
        b.iter(|| {
            train_test_split(black_box(&data), 0.2, 42);
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 7: Metrics computation
// ─────────────────────────────────────────────────────────────────

fn bench_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("metrics");
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(2));

    // Generate realistic predictions.
    let n = 10_000;
    let mut rng = fastrand::Rng::with_seed(42);
    let y_true: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
    let y_pred: Vec<f64> = y_true
        .iter()
        .map(|&y| if rng.f64() < 0.9 { y } else { 1.0 - y })
        .collect();
    let y_scores: Vec<f64> = y_true
        .iter()
        .map(|&y| y + (rng.f64() - 0.5) * 0.4)
        .collect();

    // Regression targets.
    let y_true_reg: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y_pred_reg: Vec<f64> = y_true_reg.iter().map(|&y| y + rng.f64() * 0.5).collect();

    group.bench_function("accuracy/10k", |b| {
        b.iter(|| accuracy(black_box(&y_true), black_box(&y_pred)));
    });

    group.bench_function("f1_macro/10k", |b| {
        b.iter(|| f1_score(black_box(&y_true), black_box(&y_pred), Average::Macro));
    });

    group.bench_function("confusion_matrix/10k", |b| {
        b.iter(|| confusion_matrix(black_box(&y_true), black_box(&y_pred)));
    });

    group.bench_function("classification_report/10k", |b| {
        b.iter(|| classification_report(black_box(&y_true), black_box(&y_pred)));
    });

    group.bench_function("roc_auc/10k", |b| {
        b.iter(|| roc_auc_score(black_box(&y_true), black_box(&y_scores)));
    });

    group.bench_function("mse/10k", |b| {
        b.iter(|| mean_squared_error(black_box(&y_true_reg), black_box(&y_pred_reg)));
    });

    group.bench_function("r2/10k", |b| {
        b.iter(|| r2_score(black_box(&y_true_reg), black_box(&y_pred_reg)));
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 8: End-to-end pipeline benchmark
// ─────────────────────────────────────────────────────────────────

fn bench_e2e_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_pipeline");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);

    let data = gen_classification(2000, 10, 42);
    let (train, test) = train_test_split(&data, 0.2, 42);

    group.bench_function("scaler+dt/2k×10", |b| {
        b.iter(|| {
            let mut pipeline = Pipeline::new()
                .add_transformer(StandardScaler::new())
                .set_model(DecisionTreeClassifier::new());
            pipeline.fit(black_box(&train)).unwrap();
            let preds = pipeline.predict(black_box(&test)).unwrap();
            black_box(accuracy(&test.target, &preds));
        });
    });

    group.bench_function("scaler+rf20/2k×10", |b| {
        b.iter(|| {
            let mut pipeline = Pipeline::new()
                .add_transformer(StandardScaler::new())
                .set_model(RandomForestClassifier::new().n_estimators(20).seed(42));
            pipeline.fit(black_box(&train)).unwrap();
            let preds = pipeline.predict(black_box(&test)).unwrap();
            black_box(accuracy(&test.target, &preds));
        });
    });

    group.bench_function("scaler+logreg/2k×10", |b| {
        b.iter(|| {
            let mut pipeline = Pipeline::new()
                .add_transformer(StandardScaler::new())
                .set_model(LogisticRegression::new().max_iter(200).learning_rate(0.1));
            pipeline.fit(black_box(&train)).unwrap();
            let preds = pipeline.predict(black_box(&test)).unwrap();
            black_box(accuracy(&test.target, &preds));
        });
    });

    group.bench_function("full_eval/rf+metrics/2k×10", |b| {
        b.iter(|| {
            let mut rf = RandomForestClassifier::new()
                .n_estimators(20)
                .max_depth(8)
                .seed(42);
            rf.fit(black_box(&train)).unwrap();
            let features = test.feature_matrix();
            let preds = rf.predict(black_box(&features)).unwrap();
            let _acc = accuracy(&test.target, &preds);
            let _report = classification_report(&test.target, &preds);
            let _cm = confusion_matrix(&test.target, &preds);
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 9: Multiclass benchmark (10-class synthetic dataset)
// ─────────────────────────────────────────────────────────────────

fn bench_multiclass(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiclass/10class");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(3));
    group.sampling_mode(SamplingMode::Flat);

    let data = gen_multiclass(1000, 10, 10, 42);
    let features = data.feature_matrix();

    // Scale data for algorithms that need it.
    let mut scaler = StandardScaler::new();
    let mut scaled = data.clone();
    scaler.fit(&scaled).unwrap();
    scaler.transform(&mut scaled).unwrap();
    let scaled_features = scaled.feature_matrix();

    group.bench_function("decision_tree/train", |b| {
        b.iter(|| {
            let mut dt = DecisionTreeClassifier::new().max_depth(10);
            dt.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("random_forest/train/20t", |b| {
        b.iter(|| {
            let mut rf = RandomForestClassifier::new()
                .n_estimators(20)
                .max_depth(8)
                .seed(42);
            rf.fit(black_box(&data)).unwrap();
        });
    });

    group.bench_function("logistic_regression/train", |b| {
        b.iter(|| {
            let mut lr = LogisticRegression::new().max_iter(200).learning_rate(0.1);
            lr.fit(black_box(&scaled)).unwrap();
        });
    });

    group.bench_function("knn/train+predict", |b| {
        b.iter(|| {
            let mut knn = KnnClassifier::new().k(5);
            knn.fit(black_box(&data)).unwrap();
            knn.predict(black_box(&features)).unwrap()
        });
    });

    group.bench_function("linear_svc/train", |b| {
        b.iter(|| {
            let mut svc = LinearSVC::new().c(1.0).max_iter(500);
            svc.fit(black_box(&scaled)).unwrap();
        });
    });

    // Pre-train for prediction benchmarks.
    let mut dt = DecisionTreeClassifier::new().max_depth(10);
    dt.fit(&data).unwrap();
    let mut rf = RandomForestClassifier::new()
        .n_estimators(20)
        .max_depth(8)
        .seed(42);
    rf.fit(&data).unwrap();
    let mut svc = LinearSVC::new().c(1.0).max_iter(500);
    svc.fit(&scaled).unwrap();

    group.bench_function("decision_tree/predict", |b| {
        b.iter(|| dt.predict(black_box(&features)).unwrap());
    });

    group.bench_function("random_forest/predict", |b| {
        b.iter(|| rf.predict(black_box(&features)).unwrap());
    });

    group.bench_function("linear_svc/predict", |b| {
        b.iter(|| svc.predict(black_box(&scaled_features)).unwrap());
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 10: HistGBT vs standard GBT scaling
// ─────────────────────────────────────────────────────────────────

fn bench_hist_gbt(c: &mut Criterion) {
    use scry_learn::tree::{GradientBoostingRegressor, HistGradientBoostingRegressor};

    let mut group = c.benchmark_group("hist_gbt_scaling");
    group.sample_size(10);

    for &n_rows in &[1_000, 5_000, 10_000] {
        let mut rng = fastrand::Rng::with_seed(42);
        let n_features = 10;
        let features: Vec<Vec<f64>> = (0..n_features)
            .map(|_| (0..n_rows).map(|_| rng.f64() * 10.0).collect())
            .collect();
        let target: Vec<f64> = (0..n_rows)
            .map(|i| {
                features
                    .iter()
                    .enumerate()
                    .map(|(j, col)| col[i] * (j as f64 + 1.0))
                    .sum::<f64>()
            })
            .collect();
        let names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
        let data = Dataset::new(features, target, names, "y");

        group.bench_function(format!("standard_gbt_{n_rows}"), |b| {
            b.iter(|| {
                let mut m = GradientBoostingRegressor::new()
                    .n_estimators(20)
                    .learning_rate(0.1)
                    .max_depth(4);
                m.fit(&data).unwrap();
            });
        });

        group.bench_function(format!("hist_gbt_{n_rows}"), |b| {
            b.iter(|| {
                let mut m = HistGradientBoostingRegressor::new()
                    .n_estimators(20)
                    .learning_rate(0.1)
                    .max_leaf_nodes(15)
                    .min_samples_leaf(5);
                m.fit(&data).unwrap();
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 11: Thread scaling — RandomForest parallel efficiency
// ─────────────────────────────────────────────────────────────────

fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    group.sampling_mode(SamplingMode::Flat);

    let n = 5000;
    let n_features = 10;
    let data = gen_classification(n, n_features, 42);

    for threads in [1, 2, 4, 8] {
        group.bench_with_input(BenchmarkId::new("rf_100t", threads), &threads, |b, &t| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap();
            pool.install(|| {
                b.iter(|| {
                    let mut rf = RandomForestClassifier::new().n_estimators(100).max_depth(8);
                    rf.fit(black_box(&data)).unwrap();
                });
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_training,
    bench_prediction,
    bench_dataset_scaling,
    bench_feature_scaling,
    bench_forest_scaling,
    bench_preprocessing,
    bench_metrics,
    bench_e2e_pipeline,
    bench_multiclass,
    bench_hist_gbt,
    bench_thread_scaling,
);
criterion_main!(benches);
