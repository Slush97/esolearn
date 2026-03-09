#!/usr/bin/env python3
"""
Industry benchmark: LightGBM head-to-head with scry-learn HistGBT.

Measures 5-fold CV accuracy on UCI datasets, training throughput
on larger synthetic data, and prediction latency.

Usage:
    python3 bench_lightgbm.py

Output:
    lightgbm_results.json — structured results for comparison
"""

import json
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import cross_val_score, StratifiedKFold


# ─────────────────────────────────────────────────────────────────
# Synthetic data generators (match Rust implementations)
# ─────────────────────────────────────────────────────────────────

def gen_classification(n: int, n_features: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.zeros((n, n_features))
    y = np.zeros(n, dtype=int)
    for j in range(n_features):
        offset = 3.0 + j * 0.5
        X[:half, j] = rng.random(half) * 2.0
        X[half:, j] = rng.random(n - half) * 2.0 + offset
    y[half:] = 1
    return X, y


# ─────────────────────────────────────────────────────────────────
# UCI dataset loaders
# ─────────────────────────────────────────────────────────────────

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}


def fmt_time(us: float) -> str:
    if us < 1:
        return f"{us * 1000:.1f} ns"
    if us < 1000:
        return f"{us:.1f} µs"
    if us < 1_000_000:
        return f"{us / 1000:.2f} ms"
    return f"{us / 1_000_000:.2f} s"


def time_fn(fn, n_runs=5, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        fn()
        elapsed_us = (time.perf_counter_ns() - start) / 1000
        times.append(elapsed_us)
    times.sort()
    return times[len(times) // 2]


def prediction_latency(model, X, n_iters=5_000):
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(X), size=n_iters)
    times = []
    for idx in indices:
        row = X[idx : idx + 1]
        start = time.perf_counter_ns()
        model.predict(row)
        elapsed = (time.perf_counter_ns() - start) / 1000
        times.append(elapsed)
    times.sort()
    n = len(times)
    return {
        "p50_us": times[n // 2],
        "p95_us": times[int(n * 0.95)],
        "p99_us": times[int(n * 0.99)],
    }


# ─────────────────────────────────────────────────────────────────
# LightGBM configuration (comparable to scry HistGBT)
# ─────────────────────────────────────────────────────────────────

def make_lgb(n_classes):
    """Create LGBMClassifier with settings comparable to scry HistGBT."""
    params = dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,  # FAIRNESS: match Rust single-thread (RAYON_NUM_THREADS=1)
        verbose=-1,
    )
    if n_classes > 2:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes
    return lgb.LGBMClassifier(**params)


# ─────────────────────────────────────────────────────────────────
# Section 1: 5-Fold CV Accuracy
# ─────────────────────────────────────────────────────────────────

def run_accuracy_benchmarks():
    print("\n" + "=" * 72)
    print("  SECTION 1: LightGBM 5-Fold Stratified CV Accuracy")
    print("=" * 72)

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ds_name, loader in DATASETS.items():
        data = loader()
        X, y = data.data, data.target.astype(float)
        n_classes = len(np.unique(y))
        print(f"\n  Dataset: {ds_name} ({X.shape[0]}×{X.shape[1]}, {n_classes} classes)")

        model = make_lgb(n_classes)
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
            mean_acc = scores.mean()
            std_acc = scores.std()
            fold_str = ", ".join(f"{s:.3f}" for s in scores)
            print(f"  LightGBM: {mean_acc:.4f} ± {std_acc:.4f}  [{fold_str}]")
            results[ds_name] = {
                "mean_accuracy": round(float(mean_acc), 6),
                "std_accuracy": round(float(std_acc), 6),
                "fold_scores": [round(float(s), 6) for s in scores],
            }
        except Exception as e:
            print(f"  LightGBM: FAILED — {e}")
            results[ds_name] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────
# Section 2: Training Throughput
# ─────────────────────────────────────────────────────────────────

def run_training_benchmarks():
    print("\n" + "=" * 72)
    print("  SECTION 2: LightGBM Training Throughput")
    print("=" * 72)

    sizes = [1_000, 10_000, 100_000]
    results = {}

    for n in sizes:
        X, y = gen_classification(n, 10)
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=1, verbose=-1,  # FAIRNESS: single-thread
        )
        median_us = time_fn(lambda X=X, y=y: model.fit(X, y))
        rows_per_sec = n / (median_us / 1e6) if median_us > 0 else 0
        print(f"  {n:>7,} × 10: {fmt_time(median_us):>12}  ({rows_per_sec:,.0f} rows/s)")
        results[str(n)] = {
            "median_us": round(median_us, 2),
            "rows_per_sec": round(rows_per_sec, 0),
        }

    return results


# ─────────────────────────────────────────────────────────────────
# Section 3: Prediction Latency
# ─────────────────────────────────────────────────────────────────

def run_prediction_benchmarks():
    print("\n" + "=" * 72)
    print("  SECTION 3: LightGBM Single-Row Prediction Latency")
    print("=" * 72)

    X, y = gen_classification(1000, 10)
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, verbose=-1,
    )
    model.fit(X, y)
    latency = prediction_latency(model, X)
    print(f"  p50: {fmt_time(latency['p50_us']):>10}  "
          f"p95: {fmt_time(latency['p95_us']):>10}  "
          f"p99: {fmt_time(latency['p99_us']):>10}")
    return latency


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print(f"  LightGBM Industry Benchmark — v{lgb.__version__}")
    print(f"  NumPy {np.__version__}")
    print("=" * 72)

    all_results = {
        "accuracy_cv": run_accuracy_benchmarks(),
        "training": run_training_benchmarks(),
        "prediction_latency": run_prediction_benchmarks(),
    }

    out_path = Path(__file__).parent / "lightgbm_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
