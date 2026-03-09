#![allow(
    unsafe_code,
    clippy::items_after_statements,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::default_trait_access,
    clippy::redundant_locals
)]

//! Production readiness benchmarks — heap memory, allocation counts, scaling,
//! dimensionality, per-predict costs, and stress/edge-case tests.
//!
//! Uses a `GlobalAlloc` wrapper to measure real heap usage (not serialization proxies).
//!
//! Run:  cargo test --test `production_bench` -p scry-learn --release -- --nocapture

#[path = "tracking_alloc.rs"]
mod tracking_alloc;

use std::time::Instant;
use tracking_alloc::{format_bytes, format_count, AllocSnapshot, TrackingAllocator};

#[global_allocator]
static ALLOC: TrackingAllocator = TrackingAllocator::new();

// ═══════════════════════════════════════════════════════════════════════════
// Data generation
// ═══════════════════════════════════════════════════════════════════════════

fn gen_classification(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let half = n / 2;
    let mut col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];

    for j in 0..n_features {
        let offset = 3.0 + j as f64 * 0.5;
        for i in 0..half {
            col_major[j][i] = rng.f64() * 2.0;
        }
        for i in half..n {
            col_major[j][i] = rng.f64() * 2.0 + offset;
            target[i] = 1.0;
        }
    }
    (col_major, target)
}

fn gen_multiclass(n: usize, n_features: usize, n_classes: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];
    let per_class = n / n_classes;

    for i in 0..n {
        let class = (i / per_class).min(n_classes - 1);
        target[i] = class as f64;
        for j in 0..n_features {
            col_major[j][i] = rng.f64() * 2.0 + (class as f64) * 2.0 + (j as f64) * 0.3;
        }
    }
    (col_major, target)
}

fn gen_regression(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut col_major = vec![vec![0.0; n]; n_features];
    let mut target = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n_features {
            let v = rng.f64() * 10.0;
            col_major[j][i] = v;
            sum += v * (j as f64 + 1.0);
        }
        target[i] = sum + rng.f64() * 0.1;
    }
    (col_major, target)
}

fn make_dataset(
    col_major: Vec<Vec<f64>>,
    target: Vec<f64>,
    n_features: usize,
) -> scry_learn::dataset::Dataset {
    scry_learn::dataset::Dataset::new(
        col_major,
        target,
        (0..n_features).map(|i| format!("f{i}")).collect(),
        "target",
    )
}

/// Build row-major feature matrix from column-major data.
fn to_row_major(col_major: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if col_major.is_empty() {
        return vec![];
    }
    let n = col_major[0].len();
    let d = col_major.len();
    (0..n)
        .map(|i| (0..d).map(|j| col_major[j][i]).collect())
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Peak heap memory during training (per model)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_peak_heap_per_model() {
    let n = 1000;
    #[cfg(feature = "experimental")]
    let n_ksvc = 200; // KernelSVC is O(n²), use smaller dataset
    let d = 5;
    let (col_cls, target_cls) = gen_classification(n, d);
    let (col_reg, target_reg) = gen_regression(n, d);
    let rows_cls = to_row_major(&col_cls);

    println!("\n{}", "═".repeat(72));
    println!("  PEAK HEAP MEMORY DURING fit() — {n} samples × {d} features");
    println!("{}", "═".repeat(72));
    println!(
        "  {:<30} {:>12} {:>14} {:>10}",
        "Model", "Peak Heap", "Alloc Count", "Time"
    );
    println!("  {}", "─".repeat(68));

    struct Result {
        name: &'static str,
        peak: usize,
        allocs: usize,
        time_ms: f64,
    }
    let mut results: Vec<Result> = Vec::new();

    // ── DecisionTreeClassifier ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::DecisionTreeClassifier::new();
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "DecisionTree",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── RandomForestClassifier ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(20)
            .max_depth(8);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "RandomForest(20t)",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── GradientBoostingClassifier ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::GradientBoostingClassifier::new()
            .n_estimators(20)
            .learning_rate(0.1)
            .max_depth(3);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "GBT(20t)",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── HistGradientBoostingClassifier ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::HistGradientBoostingClassifier::new()
            .n_estimators(20)
            .learning_rate(0.1);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "HistGBT(20t)",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── KNN ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::KnnClassifier::new().k(5);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "KNN(k=5)",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── LogisticRegression ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::LogisticRegression::new()
            .max_iter(50)
            .learning_rate(0.1);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "LogisticReg",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── KMeans ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::KMeans::new(3)
            .seed(42)
            .max_iter(30)
            .n_init(1);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "KMeans(k=3)",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── PCA ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::Pca::with_n_components(3);
        scry_learn::prelude::Transformer::fit(&mut m, &ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "PCA(3 comp)",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── LinearSVC ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::LinearSVC::new();
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "LinearSVC",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── GaussianNB ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::GaussianNb::new();
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "GaussianNB",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── LinearRegression ──
    {
        let ds = make_dataset(col_reg.clone(), target_reg.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::LinearRegression::new();
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "LinearRegression",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── GBT Regressor ──
    let col_reg_saved = col_reg.clone();
    let target_reg_saved = target_reg.clone();
    {
        let ds = make_dataset(col_reg, target_reg, d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::GradientBoostingRegressor::new()
            .n_estimators(20)
            .learning_rate(0.1)
            .max_depth(3);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "GBT_Regressor(20t)",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── KernelSVC ── (O(n²) — use smaller dataset)
    #[cfg(feature = "experimental")]
    {
        let cols_small: Vec<Vec<f64>> = col_cls.iter().map(|c| c[..n_ksvc].to_vec()).collect();
        let tgt_small = target_cls[..n_ksvc].to_vec();
        let ds = make_dataset(cols_small, tgt_small, d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::KernelSVC::new()
            .kernel(scry_learn::prelude::Kernel::RBF { gamma: 0.1 });
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "KernelSVC",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── LassoRegression ──
    {
        let ds = make_dataset(col_reg_saved.clone(), target_reg_saved.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::LassoRegression::new().alpha(0.1);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "LassoRegression",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── ElasticNet ──
    {
        let ds = make_dataset(col_reg_saved, target_reg_saved, d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::ElasticNet::new()
            .alpha(0.1)
            .l1_ratio(0.5);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "ElasticNet",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── BernoulliNB ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::BernoulliNB::new();
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "BernoulliNB",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── MultinomialNB (needs non-negative features) ──
    {
        let col_abs: Vec<Vec<f64>> = col_cls
            .iter()
            .map(|c| c.iter().map(|v| v.abs()).collect())
            .collect();
        let ds = make_dataset(col_abs, target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::MultinomialNB::new();
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "MultinomialNB",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── DBSCAN ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::Dbscan::new(1.0, 5);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "DBSCAN",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── AgglomerativeClustering ──
    {
        let ds = make_dataset(col_cls.clone(), target_cls.clone(), d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::AgglomerativeClustering::new(3);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "AgglomerativeClustering",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // ── MiniBatchKMeans ──
    {
        let ds = make_dataset(col_cls, target_cls, d);
        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::MiniBatchKMeans::new(3).seed(42);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);
        results.push(Result {
            name: "MiniBatchKMeans",
            peak: delta.peak_increase,
            allocs: delta.alloc_count,
            time_ms: elapsed.as_secs_f64() * 1000.0,
        });
    }

    // Print results
    for r in &results {
        println!(
            "  {:<30} {:>12} {:>14} {:>8.1}ms",
            r.name,
            format_bytes(r.peak),
            format_count(r.allocs),
            r.time_ms,
        );
    }
    println!();

    // Sanity: no model should use more than 100 MB for 1K×5 data.
    let max_allowed = 100 * 1024 * 1024;
    for r in &results {
        assert!(
            r.peak < max_allowed,
            "{} used {} peak heap — exceeds 500 MB ceiling!",
            r.name,
            format_bytes(r.peak),
        );
    }

    // Drop everything to keep row_major alive for the test
    let _ = &rows_cls;
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Allocation count per model (detect allocation storms)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_alloc_count_per_model() {
    let n = 500;
    let d = 10;
    let (col, target) = gen_classification(n, d);

    println!("\n{}", "═".repeat(72));
    println!("  ALLOCATION ANALYSIS — {n} samples × {d} features");
    println!("{}", "═".repeat(72));
    println!(
        "  {:<30} {:>14} {:>14} {:>12}",
        "Model", "Allocs", "Deallocs", "Net Bytes"
    );
    println!("  {}", "─".repeat(68));

    let models: Vec<(&str, Box<dyn FnOnce()>)> = vec![
        (
            "DecisionTree",
            Box::new({
                let col = col.clone();
                let target = target.clone();
                move || {
                    let ds = make_dataset(col, target, d);
                    let mut m = scry_learn::prelude::DecisionTreeClassifier::new();
                    m.fit(&ds).unwrap();
                    std::hint::black_box(&m);
                }
            }),
        ),
        (
            "RandomForest(20t)",
            Box::new({
                let col = col.clone();
                let target = target.clone();
                move || {
                    let ds = make_dataset(col, target, d);
                    let mut m = scry_learn::prelude::RandomForestClassifier::new()
                        .n_estimators(20)
                        .max_depth(8);
                    m.fit(&ds).unwrap();
                    std::hint::black_box(&m);
                }
            }),
        ),
        (
            "GBT(20t)",
            Box::new({
                let col = col.clone();
                let target = target.clone();
                move || {
                    let ds = make_dataset(col, target, d);
                    let mut m = scry_learn::prelude::GradientBoostingClassifier::new()
                        .n_estimators(20)
                        .learning_rate(0.1)
                        .max_depth(3);
                    m.fit(&ds).unwrap();
                    std::hint::black_box(&m);
                }
            }),
        ),
        (
            "HistGBT(20t)",
            Box::new({
                let col = col.clone();
                let target = target.clone();
                move || {
                    let ds = make_dataset(col, target, d);
                    let mut m = scry_learn::prelude::HistGradientBoostingClassifier::new()
                        .n_estimators(20)
                        .learning_rate(0.1);
                    m.fit(&ds).unwrap();
                    std::hint::black_box(&m);
                }
            }),
        ),
        (
            "KNN(k=5)",
            Box::new({
                let col = col.clone();
                let target = target.clone();
                move || {
                    let ds = make_dataset(col, target, d);
                    let mut m = scry_learn::prelude::KnnClassifier::new().k(5);
                    m.fit(&ds).unwrap();
                    std::hint::black_box(&m);
                }
            }),
        ),
        (
            "LogisticReg",
            Box::new({
                let col = col.clone();
                let target = target.clone();
                move || {
                    let ds = make_dataset(col, target, d);
                    let mut m = scry_learn::prelude::LogisticRegression::new()
                        .max_iter(50)
                        .learning_rate(0.1);
                    m.fit(&ds).unwrap();
                    std::hint::black_box(&m);
                }
            }),
        ),
        (
            "KMeans(k=3)",
            Box::new({
                let col = col.clone();
                let target = target.clone();
                move || {
                    let ds = make_dataset(col, target, d);
                    let mut m = scry_learn::prelude::KMeans::new(3)
                        .seed(42)
                        .max_iter(30)
                        .n_init(1);
                    m.fit(&ds).unwrap();
                    std::hint::black_box(&m);
                }
            }),
        ),
        (
            "GaussianNB",
            Box::new({
                let col = col;
                let target = target;
                move || {
                    let ds = make_dataset(col, target, d);
                    let mut m = scry_learn::prelude::GaussianNb::new();
                    m.fit(&ds).unwrap();
                    std::hint::black_box(&m);
                }
            }),
        ),
    ];

    for (name, run) in models {
        let before = AllocSnapshot::now();
        run();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        println!(
            "  {:<30} {:>14} {:>14} {:>12}",
            name,
            format_count(delta.alloc_count),
            format_count(delta.dealloc_count),
            if delta.net_bytes >= 0 {
                format!("+{}", format_bytes(delta.net_bytes as usize))
            } else {
                format!("-{}", format_bytes((-delta.net_bytes) as usize))
            },
        );
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Memory scaling by N (samples)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_memory_scaling_by_n() {
    let d = 10;
    let sizes = [200, 500, 2_000, 5_000];

    println!("\n{}", "═".repeat(72));
    println!("  MEMORY SCALING BY SAMPLE COUNT (d={d})");
    println!("{}", "═".repeat(72));

    struct ScalingResult {
        model: &'static str,
        points: Vec<(usize, usize, f64)>, // (n, peak_bytes, train_ms)
    }

    let model_configs: Vec<(&str, Box<dyn Fn(usize) -> (usize, f64)>)> = vec![
        (
            "RandomForest(20t)",
            Box::new(move |n| {
                let (col, tgt) = gen_classification(n, d);
                let ds = make_dataset(col, tgt, d);
                let before = AllocSnapshot::now();
                let t0 = Instant::now();
                let mut m = scry_learn::prelude::RandomForestClassifier::new()
                    .n_estimators(20)
                    .max_depth(8);
                m.fit(&ds).unwrap();
                let elapsed = t0.elapsed();
                let after = AllocSnapshot::now();
                let delta = after.delta_from(before);
                std::hint::black_box(&m);
                (delta.peak_increase, elapsed.as_secs_f64() * 1000.0)
            }),
        ),
        (
            "GBT(20t)",
            Box::new(move |n| {
                let (col, tgt) = gen_classification(n, d);
                let ds = make_dataset(col, tgt, d);
                let before = AllocSnapshot::now();
                let t0 = Instant::now();
                let mut m = scry_learn::prelude::GradientBoostingClassifier::new()
                    .n_estimators(20)
                    .learning_rate(0.1)
                    .max_depth(3);
                m.fit(&ds).unwrap();
                let elapsed = t0.elapsed();
                let after = AllocSnapshot::now();
                let delta = after.delta_from(before);
                std::hint::black_box(&m);
                (delta.peak_increase, elapsed.as_secs_f64() * 1000.0)
            }),
        ),
        (
            "HistGBT(20t)",
            Box::new(move |n| {
                let (col, tgt) = gen_classification(n, d);
                let ds = make_dataset(col, tgt, d);
                let before = AllocSnapshot::now();
                let t0 = Instant::now();
                let mut m = scry_learn::prelude::HistGradientBoostingClassifier::new()
                    .n_estimators(20)
                    .learning_rate(0.1);
                m.fit(&ds).unwrap();
                let elapsed = t0.elapsed();
                let after = AllocSnapshot::now();
                let delta = after.delta_from(before);
                std::hint::black_box(&m);
                (delta.peak_increase, elapsed.as_secs_f64() * 1000.0)
            }),
        ),
        (
            "KNN(k=5)",
            Box::new(move |n| {
                let (col, tgt) = gen_classification(n, d);
                let ds = make_dataset(col, tgt, d);
                let before = AllocSnapshot::now();
                let t0 = Instant::now();
                let mut m = scry_learn::prelude::KnnClassifier::new().k(5);
                m.fit(&ds).unwrap();
                let elapsed = t0.elapsed();
                let after = AllocSnapshot::now();
                let delta = after.delta_from(before);
                std::hint::black_box(&m);
                (delta.peak_increase, elapsed.as_secs_f64() * 1000.0)
            }),
        ),
        (
            "LogisticReg",
            Box::new(move |n| {
                let (col, tgt) = gen_classification(n, d);
                let ds = make_dataset(col, tgt, d);
                let before = AllocSnapshot::now();
                let t0 = Instant::now();
                let mut m = scry_learn::prelude::LogisticRegression::new()
                    .max_iter(50)
                    .learning_rate(0.1);
                m.fit(&ds).unwrap();
                let elapsed = t0.elapsed();
                let after = AllocSnapshot::now();
                let delta = after.delta_from(before);
                std::hint::black_box(&m);
                (delta.peak_increase, elapsed.as_secs_f64() * 1000.0)
            }),
        ),
        (
            "LinearSVC",
            Box::new(move |n| {
                let (col, tgt) = gen_classification(n, d);
                let ds = make_dataset(col, tgt, d);
                let before = AllocSnapshot::now();
                let t0 = Instant::now();
                let mut m = scry_learn::prelude::LinearSVC::new();
                m.fit(&ds).unwrap();
                let elapsed = t0.elapsed();
                let after = AllocSnapshot::now();
                let delta = after.delta_from(before);
                std::hint::black_box(&m);
                (delta.peak_increase, elapsed.as_secs_f64() * 1000.0)
            }),
        ),
    ];

    let mut all_results: Vec<ScalingResult> = Vec::new();

    for (model_name, run_fn) in &model_configs {
        let mut points = Vec::new();
        for &n in &sizes {
            let (peak, time_ms) = run_fn(n);
            points.push((n, peak, time_ms));
        }
        all_results.push(ScalingResult {
            model: model_name,
            points,
        });
    }

    // Print table
    print!("  {:<20}", "Model");
    for &n in &sizes {
        print!(" {:>14}", format!("N={n}"));
    }
    println!();
    println!("  {}", "─".repeat(20 + sizes.len() * 15));

    for r in &all_results {
        print!("  {:<20}", r.model);
        for (_, peak, _) in &r.points {
            print!(" {:>14}", format_bytes(*peak));
        }
        println!();
    }

    // Print time row
    println!();
    print!("  {:<20}", "(train time)");
    for _ in &sizes {
        print!(" {:>14}", "");
    }
    println!();

    for r in &all_results {
        print!("  {:<20}", format!("  {} time", r.model));
        for (_, _, ms) in &r.points {
            print!(" {ms:>12.1}ms");
        }
        println!();
    }

    // Scaling assertion: peak at largest N shouldn't be more than 150× peak at smallest.
    // This is a generous bound — most algorithms should be roughly linear.
    for r in &all_results {
        let smallest_peak = r.points.first().unwrap().1;
        let largest_peak = r.points.last().unwrap().1;
        if smallest_peak > 0 {
            let ratio = largest_peak as f64 / smallest_peak as f64;
            let n_ratio = *sizes.last().unwrap() as f64 / *sizes.first().unwrap() as f64;
            println!(
                "\n  {} scaling: {:.1}× memory for {:.0}× data (efficiency: {:.2})",
                r.model,
                ratio,
                n_ratio,
                ratio / n_ratio,
            );
            assert!(
                ratio < 150.0 * n_ratio / n_ratio, // just ratio < 150
                "{} memory scaling is pathological: {:.1}× for {:.0}× data increase",
                r.model,
                ratio,
                n_ratio,
            );
        }
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Dimensionality scaling (vary d at fixed N)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dimensionality_scaling() {
    let n = 500;
    let dims = [5, 20, 50, 100];

    println!("\n{}", "═".repeat(72));
    println!("  DIMENSIONALITY SCALING (N={n}, varying features)");
    println!("{}", "═".repeat(72));

    // KNN — this is where KDTree vs brute-force heuristic matters
    println!("\n  KNN(k=5):");
    println!(
        "  {:<8} {:>14} {:>14} {:>10} {:>12}",
        "d", "Peak Heap", "Allocs", "Time", "Strategy"
    );
    println!("  {}", "─".repeat(60));

    for &d in &dims {
        let (col, tgt) = gen_classification(n, d);
        let ds = make_dataset(col, tgt, d);

        let strategy = if d < 20 { "KD-Tree" } else { "Brute" };

        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::KnnClassifier::new().k(5);
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);

        // Also measure predict time
        let rows = to_row_major(&ds.features);
        let sample = vec![rows[0].clone()];
        let pred_start = Instant::now();
        for _ in 0..50 {
            std::hint::black_box(m.predict(std::hint::black_box(&sample)).unwrap());
        }
        let pred_us = pred_start.elapsed().as_nanos() as f64 / 50.0 / 1000.0;

        println!(
            "  {:<8} {:>14} {:>14} {:>8.1}ms {:>12}  (predict: {:.1}µs)",
            d,
            format_bytes(delta.peak_increase),
            format_count(delta.alloc_count),
            elapsed.as_secs_f64() * 1000.0,
            strategy,
            pred_us,
        );

        std::hint::black_box(&m);
    }

    // PCA — memory should scale with d²
    println!("\n  PCA(5 components):");
    println!(
        "  {:<8} {:>14} {:>14} {:>10}",
        "d", "Peak Heap", "Allocs", "Time"
    );
    println!("  {}", "─".repeat(48));

    for &d in &dims {
        let (col, tgt) = gen_classification(n, d);
        let ds = make_dataset(col, tgt, d);
        let n_comp = 5.min(d);

        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::Pca::with_n_components(n_comp);
        scry_learn::prelude::Transformer::fit(&mut m, &ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);

        println!(
            "  {:<8} {:>14} {:>14} {:>8.1}ms",
            d,
            format_bytes(delta.peak_increase),
            format_count(delta.alloc_count),
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    // LinearRegression — scales with d for normal equation
    println!("\n  LinearRegression:");
    println!(
        "  {:<8} {:>14} {:>14} {:>10}",
        "d", "Peak Heap", "Allocs", "Time"
    );
    println!("  {}", "─".repeat(48));

    for &d in &dims {
        let (col, tgt) = gen_regression(n, d);
        let ds = make_dataset(col, tgt, d);

        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        let mut m = scry_learn::prelude::LinearRegression::new();
        m.fit(&ds).unwrap();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        std::hint::black_box(&m);

        println!(
            "  {:<8} {:>14} {:>14} {:>8.1}ms",
            d,
            format_bytes(delta.peak_increase),
            format_count(delta.alloc_count),
            elapsed.as_secs_f64() * 1000.0,
        );
    }

    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Per-predict allocation cost
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_predict_allocations() {
    let n = 200;
    let d = 10;
    let (col, target) = gen_classification(n, d);
    let rows = to_row_major(&col);
    let single_sample = vec![rows[0].clone()];
    let n_predicts = 50;

    println!("\n{}", "═".repeat(72));
    println!("  PER-PREDICT ALLOCATION COST — single sample, {n_predicts} iterations");
    println!("{}", "═".repeat(72));
    println!(
        "  {:<30} {:>14} {:>14} {:>14}",
        "Model", "Allocs/pred", "Bytes/pred", "Latency(µs)"
    );
    println!("  {}", "─".repeat(72));

    // Fit all models first
    let ds = make_dataset(col, target, d);

    let mut dt = scry_learn::prelude::DecisionTreeClassifier::new();
    dt.fit(&ds).unwrap();

    let mut rf = scry_learn::prelude::RandomForestClassifier::new()
        .n_estimators(20)
        .max_depth(8);
    rf.fit(&ds).unwrap();

    let mut gbt = scry_learn::prelude::GradientBoostingClassifier::new()
        .n_estimators(20)
        .learning_rate(0.1)
        .max_depth(3);
    gbt.fit(&ds).unwrap();

    let mut hgbt = scry_learn::prelude::HistGradientBoostingClassifier::new()
        .n_estimators(20)
        .learning_rate(0.1);
    hgbt.fit(&ds).unwrap();

    let mut knn = scry_learn::prelude::KnnClassifier::new().k(5);
    knn.fit(&ds).unwrap();

    let mut lr = scry_learn::prelude::LogisticRegression::new()
        .max_iter(50)
        .learning_rate(0.1);
    lr.fit(&ds).unwrap();

    let mut nb = scry_learn::prelude::GaussianNb::new();
    nb.fit(&ds).unwrap();

    let mut lsvc = scry_learn::prelude::LinearSVC::new();
    lsvc.fit(&ds).unwrap();

    // KernelSVC is O(n²), use smaller dataset
    #[cfg(feature = "experimental")]
    let ksvc = {
        let (col_ksvc, tgt_ksvc) = gen_classification(50, d);
        let ds_ksvc = make_dataset(col_ksvc, tgt_ksvc, d);
        let mut ksvc = scry_learn::prelude::KernelSVC::new()
            .kernel(scry_learn::prelude::Kernel::RBF { gamma: 0.1 });
        ksvc.fit(&ds_ksvc).unwrap();
        ksvc
    };

    let mut bnb = scry_learn::prelude::BernoulliNB::new();
    bnb.fit(&ds).unwrap();

    let (col_reg, target_reg) = gen_regression(n, d);
    let ds_reg = make_dataset(col_reg, target_reg, d);
    let rows_reg = to_row_major(&ds_reg.features);
    let reg_single = vec![rows_reg[0].clone()];

    let mut lasso = scry_learn::prelude::LassoRegression::new().alpha(0.1);
    lasso.fit(&ds_reg).unwrap();

    // Measure predict costs
    let predictors: Vec<(&str, Box<dyn Fn() -> Vec<f64>>)> = vec![
        (
            "DecisionTree",
            Box::new(|| dt.predict(&single_sample).unwrap()),
        ),
        (
            "RandomForest(20t)",
            Box::new(|| rf.predict(&single_sample).unwrap()),
        ),
        (
            "GBT(20t)",
            Box::new(|| gbt.predict(&single_sample).unwrap()),
        ),
        (
            "HistGBT(20t)",
            Box::new(|| hgbt.predict(&single_sample).unwrap()),
        ),
        (
            "KNN(k=5)",
            Box::new(|| knn.predict(&single_sample).unwrap()),
        ),
        (
            "LogisticReg",
            Box::new(|| lr.predict(&single_sample).unwrap()),
        ),
        (
            "GaussianNB",
            Box::new(|| nb.predict(&single_sample).unwrap()),
        ),
        (
            "LinearSVC",
            Box::new(|| lsvc.predict(&single_sample).unwrap()),
        ),
        #[cfg(feature = "experimental")]
        (
            "KernelSVC",
            Box::new(|| ksvc.predict(&single_sample).unwrap()),
        ),
        (
            "BernoulliNB",
            Box::new(|| bnb.predict(&single_sample).unwrap()),
        ),
        (
            "LassoRegression",
            Box::new(|| lasso.predict(&reg_single).unwrap()),
        ),
    ];

    for (name, predict_fn) in &predictors {
        // Warmup
        for _ in 0..10 {
            std::hint::black_box(predict_fn());
        }

        let before = AllocSnapshot::now();
        let t0 = Instant::now();
        for _ in 0..n_predicts {
            std::hint::black_box(predict_fn());
        }
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);

        let allocs_per = delta.alloc_count as f64 / n_predicts as f64;
        let peak_per = delta.peak_increase as f64 / n_predicts as f64;
        let latency_us = elapsed.as_nanos() as f64 / n_predicts as f64 / 1000.0;

        println!(
            "  {:<30} {:>14.1} {:>14} {:>12.1}",
            name,
            allocs_per,
            format_bytes(peak_per as usize),
            latency_us,
        );
    }

    println!();
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 6: Stress and edge-case tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_stress_edge_cases() {
    println!("\n{}", "═".repeat(72));
    println!("  STRESS & EDGE-CASE TESTS");
    println!("{}", "═".repeat(72));

    let mut passed = 0;
    let mut total = 0;

    macro_rules! stress_test {
        ($name:expr, $body:expr) => {{
            total += 1;
            let t0 = Instant::now();
            let before = AllocSnapshot::now();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
            let elapsed = t0.elapsed();
            let after = AllocSnapshot::now();
            let delta = after.delta_from(before);
            match result {
                Ok(_) => {
                    passed += 1;
                    println!(
                        "  ✅ {:<45} {:>8} {:>10.1}ms",
                        $name,
                        format_bytes(delta.peak_increase),
                        elapsed.as_secs_f64() * 1000.0,
                    );
                }
                Err(_) => {
                    println!("  ❌ {:<45} PANICKED", $name);
                }
            }
        }};
    }

    // 1. Single-sample training
    stress_test!("DT: single sample (N=1)", {
        let ds = make_dataset(vec![vec![1.0]], vec![0.0], 1);
        let mut m = scry_learn::prelude::DecisionTreeClassifier::new();
        m.fit(&ds).unwrap();
    });

    // 2. Two-sample training (minimum for split)
    stress_test!("DT: two samples (N=2)", {
        let ds = make_dataset(vec![vec![1.0, 2.0]], vec![0.0, 1.0], 1);
        let mut m = scry_learn::prelude::DecisionTreeClassifier::new();
        m.fit(&ds).unwrap();
    });

    // 3. High cardinality classification (100 classes)
    stress_test!("GBT: 100 classes (N=1000)", {
        let (col, tgt) = gen_multiclass(1000, 10, 100);
        let ds = make_dataset(col, tgt, 10);
        let mut m = scry_learn::prelude::GradientBoostingClassifier::new()
            .n_estimators(10)
            .learning_rate(0.1)
            .max_depth(3);
        m.fit(&ds).unwrap();
    });

    // 4. Wide-and-short data (N=50, d=500)
    stress_test!("RF: wide data (N=50, d=500)", {
        let (col, tgt) = gen_classification(50, 500);
        let ds = make_dataset(col, tgt, 500);
        let mut m = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(10)
            .max_depth(5);
        m.fit(&ds).unwrap();
    });

    // 5. Very deep tree
    stress_test!("DT: deep tree (max_depth=50, N=2K)", {
        let (col, tgt) = gen_classification(2_000, 10);
        let ds = make_dataset(col, tgt, 10);
        let mut m = scry_learn::prelude::DecisionTreeClassifier::new().max_depth(50);
        m.fit(&ds).unwrap();
    });

    // 6. KMeans with k = N/2 (many clusters)
    stress_test!("KMeans: k=100 on N=200", {
        let (col, tgt) = gen_classification(200, 5);
        let ds = make_dataset(col, tgt, 5);
        let mut m = scry_learn::prelude::KMeans::new(100)
            .seed(42)
            .max_iter(50)
            .n_init(1);
        m.fit(&ds).unwrap();
    });

    // 7. PCA with d > N (underdetermined)
    stress_test!("PCA: d=200 > N=50 (underdetermined)", {
        let (col, tgt) = gen_classification(50, 200);
        let ds = make_dataset(col, tgt, 200);
        let mut m = scry_learn::prelude::Pca::with_n_components(10);
        scry_learn::prelude::Transformer::fit(&mut m, &ds).unwrap();
    });

    // 8. LogReg with many iterations
    stress_test!("LogReg: 500 iterations (N=500)", {
        let (col, tgt) = gen_classification(500, 10);
        let ds = make_dataset(col, tgt, 10);
        let mut m = scry_learn::prelude::LogisticRegression::new()
            .max_iter(500)
            .learning_rate(0.01);
        m.fit(&ds).unwrap();
    });

    // 9. Large RF ensemble
    stress_test!("RF: 50 trees (N=500)", {
        let (col, tgt) = gen_classification(500, 10);
        let ds = make_dataset(col, tgt, 10);
        let mut m = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(50)
            .max_depth(8);
        m.fit(&ds).unwrap();
    });

    // 10. Constant features (zero variance)
    stress_test!("DT: constant features", {
        let col = vec![vec![1.0; 100]; 5]; // all features = 1.0
        let tgt: Vec<f64> = (0..100).map(|i| (i % 2) as f64).collect();
        let ds = make_dataset(col, tgt, 5);
        let mut m = scry_learn::prelude::DecisionTreeClassifier::new();
        m.fit(&ds).unwrap();
    });

    // 11. Moderate-scale training to catch OOM-type issues
    stress_test!("RF: 20t on N=2K, d=10", {
        let (col, tgt) = gen_classification(2_000, 10);
        let ds = make_dataset(col, tgt, 10);
        let mut m = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(20)
            .max_depth(10);
        m.fit(&ds).unwrap();
    });

    // 12. HistGBT moderate-scale
    stress_test!("HistGBT: 20t on N=5K, d=10", {
        let (col, tgt) = gen_classification(5_000, 10);
        let ds = make_dataset(col, tgt, 10);
        let mut m = scry_learn::prelude::HistGradientBoostingClassifier::new()
            .n_estimators(20)
            .learning_rate(0.1);
        m.fit(&ds).unwrap();
    });

    // 13. KNN with large k
    stress_test!("KNN: k=50 on N=500", {
        let (col, tgt) = gen_classification(500, 10);
        let rows = to_row_major(&col);
        let ds = make_dataset(col, tgt, 10);
        let mut m = scry_learn::prelude::KnnClassifier::new().k(50);
        m.fit(&ds).unwrap();
        let _ = m.predict(&rows).unwrap();
    });

    // 14. Predict on batch (many samples at once)
    stress_test!("RF: batch predict 2K samples", {
        let (col, tgt) = gen_classification(2_000, 10);
        let ds = make_dataset(col.clone(), tgt, 10);
        let mut m = scry_learn::prelude::RandomForestClassifier::new()
            .n_estimators(20)
            .max_depth(8);
        m.fit(&ds).unwrap();
        let rows = to_row_major(&col);
        let _ = m.predict(&rows).unwrap();
    });

    println!("\n  Result: {passed}/{total} passed");
    println!();
    assert_eq!(passed, total, "Some stress tests failed!");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 7: Throughput scaling (samples/sec for train and predict)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_throughput_scaling() {
    let d = 10;
    let sizes = [200, 500, 2_000];

    println!("\n{}", "═".repeat(110));
    println!("  THROUGHPUT SCALING — samples/second for train & predict (d={d})");
    println!("{}", "═".repeat(110));

    struct ThroughputRow {
        model: &'static str,
        train_sps: Vec<f64>, // samples/sec per size
        predict_sps: Vec<f64>,
    }

    let mut results: Vec<ThroughputRow> = Vec::new();

    // Helper: measure throughput for a model
    macro_rules! bench_model {
        ($name:expr, $sizes:expr, $build_cls:expr, $build_pred:expr, $max_n:expr) => {{
            let mut train_sps = Vec::new();
            let mut predict_sps = Vec::new();
            for &n in $sizes {
                if n > $max_n {
                    train_sps.push(0.0);
                    predict_sps.push(0.0);
                    continue;
                }
                let (col, target) = gen_classification(n, d);
                let rows = to_row_major(&col);

                // Train throughput
                let ds = make_dataset(col.clone(), target.clone(), d);
                let t0 = Instant::now();
                let model = ($build_cls)(&ds);
                let train_elapsed = t0.elapsed().as_secs_f64();
                train_sps.push(n as f64 / train_elapsed);

                // Predict throughput
                let t0 = Instant::now();
                ($build_pred)(&model, &rows);
                let pred_elapsed = t0.elapsed().as_secs_f64();
                predict_sps.push(n as f64 / pred_elapsed);
            }
            results.push(ThroughputRow {
                model: $name,
                train_sps,
                predict_sps,
            });
        }};
    }

    bench_model!(
        "DecisionTree",
        &sizes,
        |ds: &scry_learn::dataset::Dataset| {
            let mut m = scry_learn::prelude::DecisionTreeClassifier::new();
            m.fit(ds).unwrap();
            m
        },
        |m: &scry_learn::prelude::DecisionTreeClassifier, rows: &Vec<Vec<f64>>| {
            std::hint::black_box(m.predict(std::hint::black_box(rows)).unwrap());
        },
        2_000
    );

    bench_model!(
        "RandomForest(20t)",
        &sizes,
        |ds: &scry_learn::dataset::Dataset| {
            let mut m = scry_learn::prelude::RandomForestClassifier::new()
                .n_estimators(20)
                .max_depth(8);
            m.fit(ds).unwrap();
            m
        },
        |m: &scry_learn::prelude::RandomForestClassifier, rows: &Vec<Vec<f64>>| {
            std::hint::black_box(m.predict(std::hint::black_box(rows)).unwrap());
        },
        2_000
    );

    bench_model!(
        "GBT(20t)",
        &sizes,
        |ds: &scry_learn::dataset::Dataset| {
            let mut m = scry_learn::prelude::GradientBoostingClassifier::new()
                .n_estimators(20)
                .learning_rate(0.1)
                .max_depth(3);
            m.fit(ds).unwrap();
            m
        },
        |m: &scry_learn::prelude::GradientBoostingClassifier, rows: &Vec<Vec<f64>>| {
            std::hint::black_box(m.predict(std::hint::black_box(rows)).unwrap());
        },
        2_000
    );

    bench_model!(
        "HistGBT(20t)",
        &sizes,
        |ds: &scry_learn::dataset::Dataset| {
            let mut m = scry_learn::prelude::HistGradientBoostingClassifier::new()
                .n_estimators(20)
                .learning_rate(0.1);
            m.fit(ds).unwrap();
            m
        },
        |m: &scry_learn::prelude::HistGradientBoostingClassifier, rows: &Vec<Vec<f64>>| {
            std::hint::black_box(m.predict(std::hint::black_box(rows)).unwrap());
        },
        2_000
    );

    bench_model!(
        "KNN(k=5)",
        &sizes,
        |ds: &scry_learn::dataset::Dataset| {
            let mut m = scry_learn::prelude::KnnClassifier::new().k(5);
            m.fit(ds).unwrap();
            m
        },
        |m: &scry_learn::prelude::KnnClassifier, rows: &Vec<Vec<f64>>| {
            std::hint::black_box(m.predict(std::hint::black_box(rows)).unwrap());
        },
        2_000
    );

    bench_model!(
        "LogisticReg",
        &sizes,
        |ds: &scry_learn::dataset::Dataset| {
            let mut m = scry_learn::prelude::LogisticRegression::new()
                .max_iter(50)
                .learning_rate(0.1);
            m.fit(ds).unwrap();
            m
        },
        |m: &scry_learn::prelude::LogisticRegression, rows: &Vec<Vec<f64>>| {
            std::hint::black_box(m.predict(std::hint::black_box(rows)).unwrap());
        },
        2_000
    );

    bench_model!(
        "LinearSVC",
        &sizes,
        |ds: &scry_learn::dataset::Dataset| {
            let mut m = scry_learn::prelude::LinearSVC::new();
            m.fit(ds).unwrap();
            m
        },
        |m: &scry_learn::prelude::LinearSVC, rows: &Vec<Vec<f64>>| {
            std::hint::black_box(m.predict(std::hint::black_box(rows)).unwrap());
        },
        2_000
    );

    // KernelSVC removed from throughput test — O(n²) training is tested in test_peak_heap

    // Format helper
    fn fmt_sps(sps: f64) -> String {
        if sps <= 0.0 {
            "—".to_string()
        } else if sps >= 1_000_000.0 {
            format!("{:.1}M", sps / 1_000_000.0)
        } else if sps >= 1_000.0 {
            format!("{:.1}K", sps / 1_000.0)
        } else {
            format!("{sps:.0}")
        }
    }

    // Print train throughput table
    println!("\n  TRAIN throughput (samples/second):");
    print!("  {:<20}", "Model");
    for &n in &sizes {
        print!(" {:>14}", format!("N={n}"));
    }
    println!();
    println!("  {}", "─".repeat(20 + sizes.len() * 15));
    for r in &results {
        print!("  {:<20}", r.model);
        for &sps in &r.train_sps {
            print!(" {:>14}", fmt_sps(sps));
        }
        println!();
    }

    // Print predict throughput table
    println!("\n  PREDICT throughput (samples/second):");
    print!("  {:<20}", "Model");
    for &n in &sizes {
        print!(" {:>14}", format!("N={n}"));
    }
    println!();
    println!("  {}", "─".repeat(20 + sizes.len() * 15));
    for r in &results {
        print!("  {:<20}", r.model);
        for &sps in &r.predict_sps {
            print!(" {:>14}", fmt_sps(sps));
        }
        println!();
    }
    println!();

    // Sanity: all models should process at least 50 samples/sec for N=200
    for r in &results {
        if r.train_sps[0] > 0.0 {
            assert!(
                r.train_sps[0] > 50.0,
                "{} train throughput at N=200 is too low: {:.0} sps",
                r.model,
                r.train_sps[0],
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 8: Competitor memory comparison (smartcore + linfa)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_competitor_memory_comparison() {
    let n = 1000;
    let d = 10;
    let (col, target) = gen_classification(n, d);
    let rows = to_row_major(&col);
    let target_i32: Vec<i32> = target.iter().map(|&t| t as i32).collect();

    // Regression data for Lasso comparison
    let (col_reg, target_reg) = gen_regression(n, d);
    let rows_reg = to_row_major(&col_reg);

    println!("\n{}", "═".repeat(80));
    println!("  COMPETITOR MEMORY COMPARISON — {n} samples × {d} features");
    println!("{}", "═".repeat(80));
    println!(
        "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
        "Model", "scry-learn", "smartcore", "linfa", "Winner"
    );
    println!("  {}", "─".repeat(78));

    // Helper: measure peak heap for a closure
    fn measure<F: FnOnce()>(f: F) -> (usize, f64) {
        let before = AllocSnapshot::reset();
        let t0 = Instant::now();
        f();
        let elapsed = t0.elapsed();
        let after = AllocSnapshot::now();
        let delta = after.delta_from(before);
        (delta.peak_increase, elapsed.as_secs_f64() * 1000.0)
    }

    fn winner_str(scry: usize, other: usize) -> String {
        if scry == 0 && other == 0 {
            "tie".to_string()
        } else if scry <= other {
            let ratio = if scry > 0 {
                other as f64 / scry as f64
            } else {
                f64::INFINITY
            };
            format!("scry {ratio:.1}×")
        } else {
            let ratio = if other > 0 {
                scry as f64 / other as f64
            } else {
                f64::INFINITY
            };
            format!("comp {ratio:.1}×")
        }
    }

    // ── DecisionTree: scry vs smartcore ──
    {
        let ds = make_dataset(col.clone(), target.clone(), d);
        let (scry_peak, _) = measure(|| {
            let mut m = scry_learn::prelude::DecisionTreeClassifier::new();
            m.fit(&ds).unwrap();
            std::hint::black_box(&m);
        });

        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&rows).unwrap();
        let (smart_peak, _) = measure(|| {
            let m = smartcore::tree::decision_tree_classifier::DecisionTreeClassifier::fit(
                &x,
                &target_i32,
                Default::default(),
            )
            .unwrap();
            std::hint::black_box(&m);
        });

        println!(
            "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
            "DecisionTree",
            format_bytes(scry_peak),
            format_bytes(smart_peak),
            "N/A",
            winner_str(scry_peak, smart_peak),
        );
    }

    // ── RandomForest: scry vs smartcore vs linfa ──
    {
        let ds = make_dataset(col.clone(), target.clone(), d);
        let (scry_peak, _) = measure(|| {
            let mut m = scry_learn::prelude::RandomForestClassifier::new()
                .n_estimators(20)
                .max_depth(8);
            m.fit(&ds).unwrap();
            std::hint::black_box(&m);
        });

        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&rows).unwrap();
        let (smart_peak, _) = measure(|| {
            let params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
                .with_n_trees(20_u16)
                .with_max_depth(8);
            let m = smartcore::ensemble::random_forest_classifier::RandomForestClassifier::fit(
                &x,
                &target_i32,
                params,
            )
            .unwrap();
            std::hint::black_box(&m);
        });

        // linfa RF
        use linfa::prelude::Fit;
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let linfa_x = ndarray::Array2::from_shape_vec((n, d), flat).unwrap();
        let linfa_y = ndarray::Array1::from_vec(target.iter().map(|&t| t as usize).collect());
        let linfa_ds = linfa::Dataset::new(linfa_x, linfa_y);
        let (linfa_peak, _) = measure(|| {
            let m = linfa_ensemble::RandomForestParams::new(
                linfa_trees::DecisionTree::params().max_depth(Some(8)),
            )
            .ensemble_size(20)
            .bootstrap_proportion(0.7)
            .feature_proportion(0.3)
            .fit(&linfa_ds)
            .unwrap();
            std::hint::black_box(&m);
        });
        println!("  Note: linfa uses bootstrap=0.7, features=0.3 (non-default, differs from scry/smartcore)");

        println!(
            "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
            "RandomForest(20t)",
            format_bytes(scry_peak),
            format_bytes(smart_peak),
            format_bytes(linfa_peak),
            winner_str(scry_peak, smart_peak.min(linfa_peak)),
        );
    }

    // ── KNN: scry vs smartcore ──
    {
        let ds = make_dataset(col.clone(), target.clone(), d);
        let (scry_peak, _) = measure(|| {
            let mut m = scry_learn::prelude::KnnClassifier::new().k(5);
            m.fit(&ds).unwrap();
            std::hint::black_box(&m);
        });

        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&rows).unwrap();
        let (smart_peak, _) = measure(|| {
            let m = smartcore::neighbors::knn_classifier::KNNClassifier::fit(
                &x,
                &target_i32,
                smartcore::neighbors::knn_classifier::KNNClassifierParameters::default().with_k(5),
            )
            .unwrap();
            std::hint::black_box(&m);
        });

        println!(
            "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
            "KNN(k=5)",
            format_bytes(scry_peak),
            format_bytes(smart_peak),
            "N/A",
            winner_str(scry_peak, smart_peak),
        );
    }

    // ── LogisticRegression: scry vs smartcore ──
    {
        let ds = make_dataset(col.clone(), target.clone(), d);
        let (scry_peak, _) = measure(|| {
            let mut m = scry_learn::prelude::LogisticRegression::new()
                .max_iter(200)
                .learning_rate(0.1);
            m.fit(&ds).unwrap();
            std::hint::black_box(&m);
        });

        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&rows).unwrap();
        let (smart_peak, _) = measure(|| {
            let m = smartcore::linear::logistic_regression::LogisticRegression::fit(
                &x,
                &target_i32,
                Default::default(),
            )
            .unwrap();
            std::hint::black_box(&m);
        });

        println!(
            "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
            "LogisticReg",
            format_bytes(scry_peak),
            format_bytes(smart_peak),
            "N/A",
            winner_str(scry_peak, smart_peak),
        );
    }

    // ── PCA: scry vs smartcore vs linfa ──
    {
        let ds = make_dataset(col.clone(), target.clone(), d);
        let (scry_peak, _) = measure(|| {
            let mut m = scry_learn::prelude::Pca::with_n_components(5);
            scry_learn::prelude::Transformer::fit(&mut m, &ds).unwrap();
            std::hint::black_box(&m);
        });

        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&rows).unwrap();
        let (smart_peak, _) = measure(|| {
            let smart_params =
                smartcore::decomposition::pca::PCAParameters::default().with_n_components(5);
            let m = smartcore::decomposition::pca::PCA::fit(&x, smart_params).unwrap();
            std::hint::black_box(&m);
        });

        use linfa::prelude::Fit;
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let linfa_x = ndarray::Array2::from_shape_vec((n, d), flat).unwrap();
        let linfa_y =
            ndarray::Array1::<usize>::from_vec(target.iter().map(|&t| t as usize).collect());
        let linfa_ds = linfa::Dataset::new(linfa_x, linfa_y);
        let (linfa_peak, _) = measure(|| {
            let m = linfa_reduction::Pca::params(5).fit(&linfa_ds).unwrap();
            std::hint::black_box(&m);
        });

        println!(
            "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
            "PCA(5 comp)",
            format_bytes(scry_peak),
            format_bytes(smart_peak),
            format_bytes(linfa_peak),
            winner_str(scry_peak, smart_peak.min(linfa_peak)),
        );
    }

    // ── KMeans: scry vs linfa ──
    {
        let ds = make_dataset(col.clone(), target.clone(), d);
        let (scry_peak, _) = measure(|| {
            let mut m = scry_learn::prelude::KMeans::new(3)
                .seed(42)
                .max_iter(100)
                .n_init(1);
            m.fit(&ds).unwrap();
            std::hint::black_box(&m);
        });

        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let linfa_x = ndarray::Array2::from_shape_vec((n, d), flat).unwrap();
        let linfa_ds = linfa::DatasetBase::from(linfa_x);
        let (linfa_peak, _) = measure(|| {
            use linfa::prelude::Fit;
            let m = linfa_clustering::KMeans::params_with_rng(3, rand::thread_rng())
                .max_n_iterations(100)
                .fit(&linfa_ds)
                .unwrap();
            std::hint::black_box(&m);
        });

        println!(
            "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
            "KMeans(k=3)",
            format_bytes(scry_peak),
            "N/A",
            format_bytes(linfa_peak),
            winner_str(scry_peak, linfa_peak),
        );
    }

    // ── LinearSVC: scry vs smartcore SVC ──
    {
        let ds = make_dataset(col.clone(), target.clone(), d);
        let (scry_peak, _) = measure(|| {
            let mut m = scry_learn::prelude::LinearSVC::new();
            m.fit(&ds).unwrap();
            std::hint::black_box(&m);
        });

        let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_vec(&rows).unwrap();
        let (smart_peak, _) = measure(|| {
            let knl = smartcore::svm::Kernels::linear();
            let params = smartcore::svm::svc::SVCParameters::default()
                .with_c(1.0)
                .with_kernel(knl);
            let m = smartcore::svm::svc::SVC::fit(&x, &target_i32, &params).unwrap();
            std::hint::black_box(&m);
        });

        println!(
            "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
            "LinearSVC",
            format_bytes(scry_peak),
            format_bytes(smart_peak),
            "N/A",
            winner_str(scry_peak, smart_peak),
        );
    }

    // ── Lasso: scry vs linfa-elasticnet ──
    {
        let ds = make_dataset(col_reg, target_reg.clone(), d);
        let (scry_peak, _) = measure(|| {
            let mut m = scry_learn::prelude::LassoRegression::new().alpha(0.1);
            m.fit(&ds).unwrap();
            std::hint::black_box(&m);
        });

        let flat: Vec<f64> = rows_reg.iter().flat_map(|r| r.iter().copied()).collect();
        let linfa_x = ndarray::Array2::from_shape_vec((n, d), flat).unwrap();
        let linfa_y = ndarray::Array1::from_vec(target_reg);
        let linfa_ds = linfa::Dataset::new(linfa_x, linfa_y);
        let (linfa_peak, _) = measure(|| {
            use linfa::prelude::Fit;
            let m = linfa_elasticnet::ElasticNet::<f64>::lasso()
                .penalty(0.1)
                .fit(&linfa_ds)
                .unwrap();
            std::hint::black_box(&m);
        });

        println!(
            "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
            "Lasso(α=0.1)",
            format_bytes(scry_peak),
            "N/A",
            format_bytes(linfa_peak),
            winner_str(scry_peak, linfa_peak),
        );
    }

    // ── GaussianNB: scry only (no direct competitor equivalent) ──
    {
        let ds = make_dataset(col, target, d);
        let (scry_peak, _) = measure(|| {
            let mut m = scry_learn::prelude::GaussianNb::new();
            m.fit(&ds).unwrap();
            std::hint::black_box(&m);
        });

        println!(
            "  {:<20} {:>16} {:>16} {:>16}  {:>8}",
            "GaussianNB",
            format_bytes(scry_peak),
            "N/A",
            "N/A",
            "—",
        );
    }

    println!();
    println!("  Note: All libraries use the same GlobalAlloc tracker. Data");
    println!("  conversion (DenseMatrix, ndarray) is performed outside the");
    println!("  measurement closures to isolate model-fitting memory.");
    println!();
}
