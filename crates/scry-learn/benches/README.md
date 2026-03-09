# scry-learn Benchmarks

> Last updated: 2026-02-16

This directory contains Criterion benchmark suites and Python baselines
for scry-learn's ML algorithms.

## Rust Criterion Benchmarks

| File | Focus | Run Command |
|------|-------|-------------|
| `ml_algorithms.rs` | **Synthetic data**: training throughput, prediction latency, dataset/feature/forest scaling, preprocessing, metrics, e2e pipelines, multiclass, HistGBT, thread scaling (11 groups) | `cargo bench --bench ml_algorithms -p scry-learn` |
| `industry_benchmark.rs` | **UCI datasets**: 5-fold CV accuracy, training at scale, single-row predict latency, scaling curves, cold start, batch predict, memory footprint (7 groups) | `cargo bench --bench industry_benchmark -p scry-learn` |
| `competitor_bench.rs` | **Cross-library comparison**: scry vs smartcore vs linfa for DT, RF, LogReg, KNN, K-Means, SVM, Lasso | `cargo bench --bench competitor_bench -p scry-learn` |
| `scaling_benchmark.rs` | **Large-scale** (100K–1M rows): PCA, LinearRegression, tree model scaling, throughput metrics | `cargo bench --bench scaling_benchmark -p scry-learn` |

### Key Distinctions

- **`ml_algorithms.rs`** uses _synthetic_ data (deterministic generation) — tests algorithmic behavior and scaling characteristics in isolation.
- **`industry_benchmark.rs`** uses _real UCI_ datasets (Iris, Wine, Breast Cancer, Digits) — measures accuracy and production-profile performance.
- **`competitor_bench.rs`** is the only file comparing against other Rust ML crates (smartcore, linfa).
- **`scaling_benchmark.rs`** goes to much larger dataset sizes (1M rows) than `ml_algorithms.rs`.

## Python Baselines

Located in `python/`:

| File | Focus |
|------|-------|
| `bench_sklearn.py` | Classification: 5-fold CV accuracy, training time, prediction latency (8 models × 4 UCI datasets) |
| `bench_sklearn_regression.py` | Regression: R², RMSE, MAE, training time, predict latency (6 models × California Housing) |
| `bench_xgboost.py` | XGBoost HistGBT comparison on UCI datasets |
| `bench_lightgbm.py` | LightGBM HistGBT comparison on UCI datasets |
| `bench_memory.py` | Memory footprint comparison |
| `supply_chain_benchmark.py` | Multi-library supply chain benchmark |

### Running Python Baselines

```bash
cd crates/scry-learn/benches/python
python3 -m venv .venv
.venv/bin/pip install scikit-learn xgboost lightgbm numpy
.venv/bin/python3 bench_sklearn.py
.venv/bin/python3 bench_sklearn_regression.py
```

## Quick Vitals (Integration Test)

The fastest way to get a comprehensive benchmark snapshot is:

```bash
cargo test --test quick_vitals -p scry-learn --release -- --nocapture
```

This runs 9 sections in ~5s covering classification multi-metric, regression,
confusion matrix, prediction latency, concurrent inference, serialization,
training throughput, cold start, and memory footprint.
