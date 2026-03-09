//! Large-scale `DenseMatrix` benchmarks (Sprint 12C).
//!
//! Validates that contiguous column-major `DenseMatrix` gives cache-friendly
//! scaling at 100K/1M rows for PCA, `LinearRegression`, and tree models.
//!
//! | Group | What it measures |
//! |-------|-----------------|
//! | 1 | PCA transform scaling (100K, 1M, 10K x 10K) |
//! | 2 | `LinearRegression` fit scaling |
//! | 3 | Tree model fit scaling (DT + RF) |
//! | 4 | Throughput metrics (bytes/sec, rows/sec) |
//!
//! Run with: `cargo bench --bench scaling_benchmark -p scry-learn`

#![allow(missing_docs)]

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use scry_learn::dataset::Dataset;
use scry_learn::linear::LinearRegression;
use scry_learn::preprocess::{Pca, Transformer};
use scry_learn::tree::{DecisionTreeClassifier, RandomForestClassifier};

// ─────────────────────────────────────────────────────────────────
// Deterministic data generators
// ─────────────────────────────────────────────────────────────────

/// Generate a generic dataset with continuous target (for PCA / general use).
fn generate_dataset(n_rows: usize, n_cols: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let features: Vec<Vec<f64>> = (0..n_cols)
        .map(|_| (0..n_rows).map(|_| rng.f64() * 10.0 - 5.0).collect())
        .collect();
    let target: Vec<f64> = (0..n_rows).map(|_| (rng.f64() * 3.0).floor()).collect();
    let names: Vec<String> = (0..n_cols).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "target")
}

/// Generate a regression dataset with linear relationship plus noise.
fn generate_regression_dataset(n_rows: usize, n_cols: usize, seed: u64) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut features: Vec<Vec<f64>> = (0..n_cols).map(|_| Vec::with_capacity(n_rows)).collect();
    let mut target = Vec::with_capacity(n_rows);

    for _ in 0..n_rows {
        let mut y = 0.0;
        for (j, col) in features.iter_mut().enumerate() {
            let x = rng.f64() * 10.0;
            col.push(x);
            y += x * (j + 1) as f64;
        }
        y += rng.f64() * 0.5;
        target.push(y);
    }

    let names: Vec<String> = (0..n_cols).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "y")
}

/// Generate a classification dataset with `n_classes` classes.
fn generate_classification_dataset(
    n_rows: usize,
    n_cols: usize,
    n_classes: usize,
    seed: u64,
) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut features: Vec<Vec<f64>> = (0..n_cols).map(|_| Vec::with_capacity(n_rows)).collect();
    let mut target = Vec::with_capacity(n_rows);
    let per_class = n_rows / n_classes;

    for c in 0..n_classes {
        let count = if c == n_classes - 1 {
            n_rows - per_class * c
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

    let names: Vec<String> = (0..n_cols).map(|i| format!("f{i}")).collect();
    Dataset::new(features, target, names, "class")
}

// ─────────────────────────────────────────────────────────────────
// Group 1: PCA Transform Scaling
// ─────────────────────────────────────────────────────────────────

fn bench_pca_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("pca_transform_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(30));

    for (label, n_rows, n_cols, n_components) in [
        ("100k_50f_10c", 100_000, 50, 10),
        ("1m_100f_20c", 1_000_000, 100, 20),
        ("10k_10kf_50c", 10_000, 10_000, 50),
    ] {
        group.bench_function(label, |b| {
            let data = generate_dataset(n_rows, n_cols, 42);
            let mut pca = Pca::with_n_components(n_components);
            pca.fit(&data).unwrap();
            b.iter(|| {
                let mut d = data.clone();
                pca.transform(&mut d).unwrap();
                std::hint::black_box(&d);
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 2: LinearRegression Fit Scaling
// ─────────────────────────────────────────────────────────────────

fn bench_linreg_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("linreg_fit_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(30));

    for (label, n_rows, n_cols) in [
        ("100k_50f", 100_000, 50),
        ("1m_100f", 1_000_000, 100),
        ("10k_10kf", 10_000, 10_000),
    ] {
        group.bench_function(label, |b| {
            let data = generate_regression_dataset(n_rows, n_cols, 42);
            b.iter(|| {
                let mut model = LinearRegression::new();
                model.fit(&data).unwrap();
                std::hint::black_box(&model);
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 3: Tree Model Scaling (CART + Random Forest)
// ─────────────────────────────────────────────────────────────────

fn bench_tree_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_fit_scaling");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(30));

    for (label, n_rows, n_cols) in [("100k_50f", 100_000, 50), ("1m_100f", 1_000_000, 100)] {
        group.bench_function(BenchmarkId::new("dt_classifier", label), |b| {
            let data = generate_classification_dataset(n_rows, n_cols, 3, 42);
            b.iter(|| {
                let mut model = DecisionTreeClassifier::new().max_depth(10);
                model.fit(&data).unwrap();
                std::hint::black_box(&model);
            });
        });

        group.bench_function(BenchmarkId::new("rf_classifier", label), |b| {
            let data = generate_classification_dataset(n_rows, n_cols, 3, 42);
            b.iter(|| {
                let mut model = RandomForestClassifier::new().n_estimators(10).max_depth(10);
                model.fit(&data).unwrap();
                std::hint::black_box(&model);
            });
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────
// Group 4: Throughput Metrics
// ─────────────────────────────────────────────────────────────────

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(15));

    let n_rows: usize = 100_000;
    let n_cols: usize = 50;
    let data = generate_regression_dataset(n_rows, n_cols, 42);
    let bytes = (n_rows * n_cols * 8) as u64;

    group.throughput(Throughput::Bytes(bytes));
    group.bench_function("pca_fit_100k", |b| {
        b.iter(|| {
            let mut pca = Pca::with_n_components(10);
            pca.fit(&data).unwrap();
            std::hint::black_box(&pca);
        });
    });

    group.throughput(Throughput::Elements(n_rows as u64));
    group.bench_function("linreg_predict_100k", |b| {
        let mut model = LinearRegression::new();
        model.fit(&data).unwrap();
        let test_rows: Vec<Vec<f64>> = (0..n_rows)
            .map(|i| (0..n_cols).map(|j| data.features[j][i]).collect())
            .collect();
        b.iter(|| {
            std::hint::black_box(model.predict(&test_rows).unwrap());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pca_transform,
    bench_linreg_fit,
    bench_tree_scaling,
    bench_throughput,
);
criterion_main!(benches);
