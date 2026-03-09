//! Industry benchmark suite: scry-learn on real UCI datasets + scaling.
//!
//! **Unlike `ml_algorithms.rs`** (synthetic data, train=test), this suite:
//! - Uses **real UCI CSV fixtures** (iris, wine, `breast_cancer`, digits)
//! - Runs **5-fold stratified cross-validation** for accuracy measurement
//! - Benchmarks training at **1K / 10K / 100K** row scales
//! - Measures **single-row prediction latency** separately
//!
//! Run:  `cargo bench --bench industry_benchmark -p scry-learn`
//!
//! Compare with:
//! - `python3 benches/python/bench_sklearn.py`
//! - `python3 benches/python/bench_xgboost.py`
//! - `python3 benches/python/bench_lightgbm.py`

#![allow(missing_docs)]

use std::path::PathBuf;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};

use scry_learn::dataset::Dataset;
use scry_learn::linear::LogisticRegression;
use scry_learn::metrics::accuracy;
use scry_learn::naive_bayes::GaussianNb;
use scry_learn::neighbors::KnnClassifier;
use scry_learn::preprocess::{StandardScaler, Transformer};
use scry_learn::split::{cross_val_score_stratified, ScoringFn};
use scry_learn::svm::LinearSVC;
use scry_learn::tree::{
    DecisionTreeClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
};

// ═══════════════════════════════════════════════════════════════════
// Fixture loading (same pattern as tests/golden_reference.rs)
// ═══════════════════════════════════════════════════════════════════

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Load a feature CSV into column-major format for Dataset.
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

    // Transpose to column-major.
    let mut cols = vec![vec![0.0; rows.len()]; n_cols];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            cols[j][i] = val;
        }
    }
    (cols, headers)
}

/// Load a target CSV.
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

/// Load a UCI dataset from CSV fixtures.
fn load_dataset(base: &str) -> Dataset {
    let (features, feat_names) = load_features_csv(&format!("{base}_features.csv"));
    let target = load_target_csv(&format!("{base}_target.csv"));
    Dataset::new(features, target, feat_names, "target")
}

// ═══════════════════════════════════════════════════════════════════
// Synthetic data generators (identical to ml_algorithms.rs)
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
// Group 1: 5-Fold CV Accuracy on UCI Datasets
// ═══════════════════════════════════════════════════════════════════

/// Run 5-fold stratified CV and print accuracy. Criterion measures the
/// wall-clock of the full CV, but the primary deliverable is the printed
/// accuracy for comparison against Python baselines.
fn bench_accuracy_cv(c: &mut Criterion) {
    let mut group = c.benchmark_group("accuracy_cv");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(10));
    group.sampling_mode(SamplingMode::Flat);

    let datasets = ["iris", "wine", "breast_cancer", "digits"];
    let scorer: ScoringFn = accuracy;

    for &ds_name in &datasets {
        let data = load_dataset(ds_name);

        // Decision Tree
        group.bench_with_input(
            BenchmarkId::new("decision_tree", ds_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let model = DecisionTreeClassifier::new().max_depth(10);
                    let scores = cross_val_score_stratified(&model, data, 5, scorer, 42).unwrap();
                    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                    black_box(mean)
                });
            },
        );

        // Random Forest
        group.bench_with_input(
            BenchmarkId::new("random_forest", ds_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let model = RandomForestClassifier::new()
                        .n_estimators(20)
                        .max_depth(10)
                        .seed(42);
                    let scores = cross_val_score_stratified(&model, data, 5, scorer, 42).unwrap();
                    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                    black_box(mean)
                });
            },
        );

        // Gradient Boosting
        group.bench_with_input(
            BenchmarkId::new("gradient_boosting", ds_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let model = GradientBoostingClassifier::new()
                        .n_estimators(100)
                        .max_depth(5)
                        .learning_rate(0.1);
                    let scores = cross_val_score_stratified(&model, data, 5, scorer, 42).unwrap();
                    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                    black_box(mean)
                });
            },
        );

        // HistGBT
        group.bench_with_input(BenchmarkId::new("hist_gbt", ds_name), &data, |b, data| {
            b.iter(|| {
                let model = HistGradientBoostingClassifier::new()
                    .n_estimators(100)
                    .max_depth(6)
                    .learning_rate(0.1);
                let scores = cross_val_score_stratified(&model, data, 5, scorer, 42).unwrap();
                let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                black_box(mean)
            });
        });

        // Logistic Regression (needs scaling)
        let mut scaled_data = data.clone();
        let mut scaler = StandardScaler::new();
        scaler.fit(&scaled_data).unwrap();
        scaler.transform(&mut scaled_data).unwrap();

        group.bench_with_input(
            BenchmarkId::new("logistic_regression", ds_name),
            &scaled_data,
            |b, data| {
                b.iter(|| {
                    let model = LogisticRegression::new().max_iter(500).learning_rate(0.01);
                    let scores = cross_val_score_stratified(&model, data, 5, scorer, 42).unwrap();
                    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                    black_box(mean)
                });
            },
        );

        // KNN (needs scaling for fair comparison)
        group.bench_with_input(BenchmarkId::new("knn", ds_name), &scaled_data, |b, data| {
            b.iter(|| {
                let model = KnnClassifier::new().k(5);
                let scores = cross_val_score_stratified(&model, data, 5, scorer, 42).unwrap();
                let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                black_box(mean)
            });
        });

        // Gaussian NB
        group.bench_with_input(
            BenchmarkId::new("gaussian_nb", ds_name),
            &data,
            |b, data| {
                b.iter(|| {
                    let model = GaussianNb::new();
                    let scores = cross_val_score_stratified(&model, data, 5, scorer, 42).unwrap();
                    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
                    black_box(mean)
                });
            },
        );
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Group 2: Training Throughput at Scale
// ═══════════════════════════════════════════════════════════════════

fn bench_training_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("industry_train");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);

    let sizes: &[usize] = &[1_000, 10_000, 100_000];

    for &n in sizes {
        let data = gen_classification(n, 10, 42);

        group.bench_with_input(BenchmarkId::new("decision_tree", n), &data, |b, data| {
            b.iter(|| {
                let mut dt = DecisionTreeClassifier::new().max_depth(10);
                dt.fit(black_box(data)).unwrap();
            });
        });

        group.bench_with_input(
            BenchmarkId::new("random_forest/20t", n),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut rf = RandomForestClassifier::new()
                        .n_estimators(20)
                        .max_depth(10)
                        .seed(42);
                    rf.fit(black_box(data)).unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gradient_boosting", n),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut gbt = GradientBoostingClassifier::new()
                        .n_estimators(50)
                        .max_depth(5)
                        .learning_rate(0.1);
                    gbt.fit(black_box(data)).unwrap();
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("hist_gbt", n), &data, |b, data| {
            b.iter(|| {
                let mut hgbt = HistGradientBoostingClassifier::new()
                    .n_estimators(50)
                    .max_depth(6)
                    .learning_rate(0.1);
                hgbt.fit(black_box(data)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("gaussian_nb", n), &data, |b, data| {
            b.iter(|| {
                let mut nb = GaussianNb::new();
                nb.fit(black_box(data)).unwrap();
            });
        });

        // KNN training is O(1) but prediction is O(n·k), skip large sizes
        if n <= 10_000 {
            group.bench_with_input(BenchmarkId::new("knn", n), &data, |b, data| {
                b.iter(|| {
                    let mut knn = KnnClassifier::new().k(5);
                    knn.fit(black_box(data)).unwrap();
                });
            });

            group.bench_with_input(
                BenchmarkId::new("logistic_regression", n),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut lr = LogisticRegression::new().max_iter(200).learning_rate(0.1);
                        lr.fit(black_box(data)).unwrap();
                    });
                },
            );

            let mut svc_scaler = StandardScaler::new();
            let mut svc_data = data.clone();
            svc_scaler.fit(&svc_data).unwrap();
            svc_scaler.transform(&mut svc_data).unwrap();

            group.bench_with_input(BenchmarkId::new("linear_svc", n), &svc_data, |b, data| {
                b.iter(|| {
                    let mut svc = LinearSVC::new().c(1.0).max_iter(500);
                    svc.fit(black_box(data)).unwrap();
                });
            });
        }
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Group 3: Single-Row Prediction Latency
// ═══════════════════════════════════════════════════════════════════

fn bench_prediction_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("industry_predict_single");
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let data = gen_classification(1000, 10, 42);

    // Pre-train models
    let mut dt = DecisionTreeClassifier::new().max_depth(10);
    dt.fit(&data).unwrap();

    let mut rf = RandomForestClassifier::new()
        .n_estimators(20)
        .max_depth(10)
        .seed(42);
    rf.fit(&data).unwrap();

    let mut gbt = GradientBoostingClassifier::new()
        .n_estimators(100)
        .max_depth(5)
        .learning_rate(0.1);
    gbt.fit(&data).unwrap();

    let mut hgbt = HistGradientBoostingClassifier::new()
        .n_estimators(100)
        .max_depth(6)
        .learning_rate(0.1);
    hgbt.fit(&data).unwrap();

    let mut lr = LogisticRegression::new().max_iter(200).learning_rate(0.1);
    lr.fit(&data).unwrap();

    let mut knn = KnnClassifier::new().k(5);
    knn.fit(&data).unwrap();

    let mut nb = GaussianNb::new();
    nb.fit(&data).unwrap();

    let mut scaler = StandardScaler::new();
    let mut scaled = data.clone();
    scaler.fit(&scaled).unwrap();
    scaler.transform(&mut scaled).unwrap();
    let mut svc = LinearSVC::new().c(1.0).max_iter(500);
    svc.fit(&scaled).unwrap();

    // Single-row input
    let single_row = vec![data.sample(0)];
    let scaled_row = vec![scaled.sample(0)];

    group.bench_function("decision_tree", |b| {
        b.iter(|| dt.predict(black_box(&single_row)).unwrap());
    });

    group.bench_function("random_forest", |b| {
        b.iter(|| rf.predict(black_box(&single_row)).unwrap());
    });

    group.bench_function("gradient_boosting", |b| {
        b.iter(|| gbt.predict(black_box(&single_row)).unwrap());
    });

    group.bench_function("hist_gbt", |b| {
        b.iter(|| hgbt.predict(black_box(&single_row)).unwrap());
    });

    group.bench_function("logistic_regression", |b| {
        b.iter(|| lr.predict(black_box(&single_row)).unwrap());
    });

    group.bench_function("knn", |b| {
        b.iter(|| knn.predict(black_box(&single_row)).unwrap());
    });

    group.bench_function("gaussian_nb", |b| {
        b.iter(|| nb.predict(black_box(&single_row)).unwrap());
    });

    group.bench_function("linear_svc", |b| {
        b.iter(|| svc.predict(black_box(&scaled_row)).unwrap());
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Group 4: Scaling — Time vs N_samples
// ═══════════════════════════════════════════════════════════════════

fn bench_scaling_curves(c: &mut Criterion) {
    let mut group = c.benchmark_group("industry_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);

    let sizes: &[usize] = &[100, 1_000, 5_000, 10_000, 50_000];

    for &n in sizes {
        let data = gen_classification(n, 10, 42);

        group.bench_with_input(BenchmarkId::new("dt_train", n), &data, |b, data| {
            b.iter(|| {
                let mut dt = DecisionTreeClassifier::new().max_depth(10);
                dt.fit(black_box(data)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("rf_train/10t", n), &data, |b, data| {
            b.iter(|| {
                let mut rf = RandomForestClassifier::new()
                    .n_estimators(10)
                    .max_depth(8)
                    .seed(42);
                rf.fit(black_box(data)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("hist_gbt_train", n), &data, |b, data| {
            b.iter(|| {
                let mut hgbt = HistGradientBoostingClassifier::new()
                    .n_estimators(20)
                    .max_depth(6)
                    .learning_rate(0.1);
                hgbt.fit(black_box(data)).unwrap();
            });
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Group 5: Cold Start — construct + fit + first predict
// ═══════════════════════════════════════════════════════════════════

fn bench_cold_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("industry_cold_start");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);

    let data = gen_classification(5_000, 10, 42);
    let single_row = vec![data.sample(0)];

    group.bench_function("decision_tree", |b| {
        b.iter(|| {
            let mut dt = DecisionTreeClassifier::new().max_depth(10);
            dt.fit(black_box(&data)).unwrap();
            dt.predict(black_box(&single_row)).unwrap()
        });
    });

    group.bench_function("random_forest/20t", |b| {
        b.iter(|| {
            let mut rf = RandomForestClassifier::new()
                .n_estimators(20)
                .max_depth(10)
                .seed(42);
            rf.fit(black_box(&data)).unwrap();
            rf.predict(black_box(&single_row)).unwrap()
        });
    });

    group.bench_function("hist_gbt", |b| {
        b.iter(|| {
            let mut hgbt = HistGradientBoostingClassifier::new()
                .n_estimators(50)
                .max_depth(6)
                .learning_rate(0.1);
            hgbt.fit(black_box(&data)).unwrap();
            hgbt.predict(black_box(&single_row)).unwrap()
        });
    });

    group.bench_function("logistic_regression", |b| {
        b.iter(|| {
            let mut lr = LogisticRegression::new().max_iter(200).learning_rate(0.1);
            lr.fit(black_box(&data)).unwrap();
            lr.predict(black_box(&single_row)).unwrap()
        });
    });

    group.bench_function("knn", |b| {
        b.iter(|| {
            let mut knn = KnnClassifier::new().k(5);
            knn.fit(black_box(&data)).unwrap();
            knn.predict(black_box(&single_row)).unwrap()
        });
    });

    group.bench_function("gaussian_nb", |b| {
        b.iter(|| {
            let mut nb = GaussianNb::new();
            nb.fit(black_box(&data)).unwrap();
            nb.predict(black_box(&single_row)).unwrap()
        });
    });

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Group 6: Batch Prediction Throughput
// ═══════════════════════════════════════════════════════════════════

fn bench_batch_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("industry_batch_predict");
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(3));

    let train = gen_classification(5_000, 10, 42);

    // Pre-train all models
    let mut dt = DecisionTreeClassifier::new().max_depth(10);
    dt.fit(&train).unwrap();

    let mut rf = RandomForestClassifier::new()
        .n_estimators(20)
        .max_depth(10)
        .seed(42);
    rf.fit(&train).unwrap();

    let mut hgbt = HistGradientBoostingClassifier::new()
        .n_estimators(50)
        .max_depth(6)
        .learning_rate(0.1);
    hgbt.fit(&train).unwrap();

    let mut knn = KnnClassifier::new().k(5);
    knn.fit(&train).unwrap();

    let mut nb = GaussianNb::new();
    nb.fit(&train).unwrap();

    let batch_sizes: &[usize] = &[1, 10, 100, 1_000];

    for &bs in batch_sizes {
        // Build batch input: bs rows sampled from training data
        let batch: Vec<Vec<f64>> = (0..bs)
            .map(|i| train.sample(i % train.n_samples()))
            .collect();

        group.bench_with_input(BenchmarkId::new("decision_tree", bs), &batch, |b, batch| {
            b.iter(|| dt.predict(black_box(batch)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("random_forest", bs), &batch, |b, batch| {
            b.iter(|| rf.predict(black_box(batch)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("hist_gbt", bs), &batch, |b, batch| {
            b.iter(|| hgbt.predict(black_box(batch)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("knn", bs), &batch, |b, batch| {
            b.iter(|| knn.predict(black_box(batch)).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("gaussian_nb", bs), &batch, |b, batch| {
            b.iter(|| nb.predict(black_box(batch)).unwrap());
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════
// Group 7: Memory Footprint (RSS Delta) — migrated from predict_latency.rs
// ═══════════════════════════════════════════════════════════════════

/// Read RSS (Resident Set Size) in kilobytes from `/proc/self/statm`.
fn rss_kb() -> usize {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: usize = statm
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    pages * 4 // page size = 4KB on x86_64
}

fn bench_model_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("industry_model_memory");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for &(label, n_rows) in &[("10k", 10_000), ("100k", 100_000)] {
        let data = gen_classification(n_rows, 50, 42);

        // Random Forest — RSS delta during fit
        let rss_before = rss_kb();
        let mut rf = RandomForestClassifier::new()
            .n_estimators(100)
            .max_depth(10)
            .seed(42);
        rf.fit(&data).unwrap();
        let rss_after = rss_kb();
        let rf_rss_delta_kb = rss_after.saturating_sub(rss_before);
        group.bench_function(BenchmarkId::new("rf_100trees_rss_kb", label), |b| {
            b.iter(|| black_box(rf_rss_delta_kb));
        });

        // GBT — RSS delta during fit
        let rss_before = rss_kb();
        let mut gbt = GradientBoostingClassifier::new()
            .n_estimators(100)
            .max_depth(5)
            .learning_rate(0.1);
        gbt.fit(&data).unwrap();
        let rss_after = rss_kb();
        let gbt_rss_delta_kb = rss_after.saturating_sub(rss_before);
        group.bench_function(BenchmarkId::new("gbt_100est_rss_kb", label), |b| {
            b.iter(|| black_box(gbt_rss_delta_kb));
        });

        // KNN — stores full training set
        let rss_before = rss_kb();
        let mut knn = KnnClassifier::new().k(5);
        knn.fit(&data).unwrap();
        let rss_after = rss_kb();
        let knn_rss_delta_kb = rss_after.saturating_sub(rss_before);
        group.bench_function(BenchmarkId::new("knn_k5_rss_kb", label), |b| {
            b.iter(|| black_box(knn_rss_delta_kb));
        });
    }

    group.finish();
}

// ═══════════════════════════════════════════════════════════════════

criterion_group!(
    benches,
    bench_accuracy_cv,
    bench_training_scale,
    bench_prediction_latency,
    bench_scaling_curves,
    bench_cold_start,
    bench_batch_predict,
    bench_model_memory,
);
criterion_main!(benches);
