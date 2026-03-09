#!/usr/bin/env python3
"""
sklearn memory & performance benchmark — companion to scry-learn's production_bench.rs.

Measures peak heap (via tracemalloc), training time, and prediction latency
for the same models and data shapes used in the Rust benchmarks, so numbers
are directly comparable.

Usage:
    python3 crates/scry-learn/tests/sklearn_bench.py
    python3 crates/scry-learn/tests/sklearn_bench.py --json   # machine-readable output
"""

import argparse
import gc
import json
import sys
import time
import tracemalloc

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


# ═══════════════════════════════════════════════════════════════════════════
# Data generation — MUST match Rust gen_classification / gen_regression exactly
# ═══════════════════════════════════════════════════════════════════════════

def gen_classification(n, n_features, seed=42):
    """Mirror Rust gen_classification: two clusters separated by offset."""
    rng = np.random.RandomState(seed)
    half = n // 2
    X = np.zeros((n, n_features))
    y = np.zeros(n)
    for j in range(n_features):
        offset = 3.0 + j * 0.5
        X[:half, j] = rng.rand(half) * 2.0
        X[half:, j] = rng.rand(n - half) * 2.0 + offset
    y[half:] = 1.0
    return X, y


def gen_regression(n, n_features, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, n_features) * 10.0
    y = np.sum(X * np.arange(1, n_features + 1), axis=1) + rng.rand(n) * 0.1
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# Measurement helpers
# ═══════════════════════════════════════════════════════════════════════════

def measure_fit(model, X, y):
    """Measure peak memory and time for model.fit(X, y)."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, elapsed


def measure_predict(model, X_single, n_iters=1000):
    """Measure per-predict latency and memory."""
    # Warmup
    for _ in range(10):
        model.predict(X_single)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        model.predict(X_single)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / n_iters, elapsed / n_iters


def fmt_bytes(b):
    if b >= 1_073_741_824:
        return f"{b / 1_073_741_824:.2f} GB"
    if b >= 1_048_576:
        return f"{b / 1_048_576:.2f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark suite
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmarks():
    results = {}

    # ── Test 1: Peak heap per model (5K × 10, same as Rust) ──
    n, d = 5000, 10
    X_cls, y_cls = gen_classification(n, d)
    X_reg, y_reg = gen_regression(n, d)
    X_single = X_cls[:1]

    models_cls = [
        ("DecisionTree", DecisionTreeClassifier(random_state=42)),
        ("RandomForest(100t)", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
        ("GBT(100t)", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
        ("HistGBT(100t)", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)),
        ("KNN(k=5)", KNeighborsClassifier(n_neighbors=5)),
        ("LogisticReg", LogisticRegression(max_iter=200, random_state=42)),
        ("KMeans(k=3)", KMeans(n_clusters=3, max_iter=100, n_init=1, random_state=42)),
        ("PCA(5 comp)", PCA(n_components=5)),
        ("LinearSVC", LinearSVC(random_state=42, max_iter=1000)),
        ("GaussianNB", GaussianNB()),
        ("BernoulliNB", BernoulliNB()),
        ("MultinomialNB", MultinomialNB()),
        ("DBSCAN", DBSCAN(eps=1.0, min_samples=5)),
        ("AgglomerativeClustering", AgglomerativeClustering(n_clusters=3)),
    ]

    models_reg = [
        ("LinearRegression", LinearRegression()),
        ("GBT_Regressor(100t)", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
        ("Lasso", Lasso(alpha=0.1)),
        ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5)),
    ]

    peak_heap_results = []

    for name, model in models_cls:
        if name == "MultinomialNB":
            peak, elapsed = measure_fit(model, np.abs(X_cls), y_cls)
        elif name.startswith(("KMeans", "PCA", "DBSCAN", "Agglomerative")):
            peak, elapsed = measure_fit(model, X_cls, None)
        else:
            peak, elapsed = measure_fit(model, X_cls, y_cls)
        peak_heap_results.append({
            "model": name,
            "peak_bytes": peak,
            "train_ms": elapsed * 1000,
        })

    for name, model in models_reg:
        peak, elapsed = measure_fit(model, X_reg, y_reg)
        peak_heap_results.append({
            "model": name,
            "peak_bytes": peak,
            "train_ms": elapsed * 1000,
        })

    results["peak_heap"] = peak_heap_results

    # ── Test 2: Per-predict allocation cost ──
    predict_results = []
    predict_models = [
        ("DecisionTree", DecisionTreeClassifier(random_state=42)),
        ("RandomForest(100t)", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
        ("GBT(100t)", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
        ("HistGBT(100t)", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42)),
        ("KNN(k=5)", KNeighborsClassifier(n_neighbors=5)),
        ("LogisticReg", LogisticRegression(max_iter=200, random_state=42)),
        ("GaussianNB", GaussianNB()),
        ("LinearSVC", LinearSVC(random_state=42, max_iter=1000)),
        ("BernoulliNB", BernoulliNB()),
    ]

    for name, model in predict_models:
        model.fit(X_cls, y_cls)
        bytes_per, time_per = measure_predict(model, X_single, n_iters=1000)
        predict_results.append({
            "model": name,
            "bytes_per_predict": bytes_per,
            "latency_us": time_per * 1_000_000,
        })

    results["predict"] = predict_results

    # ── Test 3: Memory scaling by N ──
    sizes = [500, 2000, 10_000, 50_000]
    scaling_models = [
        ("RandomForest(50t)", lambda: RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)),
        ("GBT(50t)", lambda: GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)),
        ("HistGBT(50t)", lambda: HistGradientBoostingClassifier(max_iter=50, learning_rate=0.1, random_state=42)),
        ("KNN(k=5)", lambda: KNeighborsClassifier(n_neighbors=5)),
        ("LogisticReg", lambda: LogisticRegression(max_iter=200, random_state=42)),
        ("LinearSVC", lambda: LinearSVC(random_state=42, max_iter=1000)),
        ("DBSCAN", lambda: DBSCAN(eps=1.0, min_samples=5)),
    ]

    scaling_results = []
    for name, make_model in scaling_models:
        points = []
        for n_s in sizes:
            X_s, y_s = gen_classification(n_s, 10)
            model = make_model()
            y_fit = None if name == "DBSCAN" else y_s
            peak, elapsed = measure_fit(model, X_s, y_fit)
            points.append({
                "n": n_s,
                "peak_bytes": peak,
                "train_ms": elapsed * 1000,
            })
        scaling_results.append({"model": name, "points": points})

    results["scaling"] = scaling_results

    # ── Test 4: Dimensionality scaling ──
    dims = [5, 20, 100, 500]
    dim_results = []

    for dd in dims:
        X_d, y_d = gen_classification(1000, dd)
        model = KNeighborsClassifier(n_neighbors=5)
        peak, elapsed = measure_fit(model, X_d, y_d)
        # Predict single sample
        single = X_d[:1]
        model_fitted = KNeighborsClassifier(n_neighbors=5)
        model_fitted.fit(X_d, y_d)
        _, pred_time = measure_predict(model_fitted, single, n_iters=100)
        dim_results.append({
            "d": dd,
            "model": "KNN(k=5)",
            "peak_bytes": peak,
            "train_ms": elapsed * 1000,
            "predict_us": pred_time * 1_000_000,
        })

    results["dimensionality"] = dim_results

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════

def print_table(results):
    print()
    print("=" * 78)
    print("  SKLEARN PRODUCTION BENCHMARK (tracemalloc)")
    print(f"  sklearn {__import__('sklearn').__version__}, "
          f"numpy {np.__version__}, "
          f"Python {sys.version.split()[0]}")
    print("=" * 78)

    # Peak heap
    print()
    print("  PEAK HEAP DURING fit() — 5000 × 10")
    print(f"  {'Model':<30} {'Peak Heap':>12} {'Train Time':>12}")
    print(f"  {'─' * 56}")
    for r in results["peak_heap"]:
        print(f"  {r['model']:<30} {fmt_bytes(r['peak_bytes']):>12} {r['train_ms']:>10.1f}ms")

    # Per-predict
    print()
    print("  PER-PREDICT COST — single sample, 1000 iterations")
    print(f"  {'Model':<30} {'Bytes/pred':>14} {'Latency(µs)':>14}")
    print(f"  {'─' * 60}")
    for r in results["predict"]:
        print(f"  {r['model']:<30} {fmt_bytes(r['bytes_per_predict']):>14} {r['latency_us']:>12.1f}")

    # Memory scaling
    print()
    print("  MEMORY SCALING BY SAMPLE COUNT (d=10)")
    header = f"  {'Model':<20}"
    for s in [500, 2000, 10_000, 50_000]:
        header += f" {'N=' + str(s):>14}"
    print(header)
    print(f"  {'─' * (20 + 4 * 15)}")
    for sr in results["scaling"]:
        row = f"  {sr['model']:<20}"
        for p in sr["points"]:
            row += f" {fmt_bytes(p['peak_bytes']):>14}"
        print(row)

    # Time scaling
    print()
    for sr in results["scaling"]:
        row = f"  {sr['model'] + ' time':<20}"
        for p in sr["points"]:
            row += f" {p['train_ms']:>12.1f}ms"
        print(row)

    # Dimensionality
    print()
    print("  DIMENSIONALITY SCALING — KNN(k=5), N=1000")
    print(f"  {'d':<8} {'Peak Heap':>14} {'Train Time':>12} {'Predict(µs)':>14}")
    print(f"  {'─' * 50}")
    for r in results["dimensionality"]:
        print(f"  {r['d']:<8} {fmt_bytes(r['peak_bytes']):>14} {r['train_ms']:>10.1f}ms {r['predict_us']:>12.1f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="sklearn production benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    results = run_benchmarks()

    if args.json:
        # Output JSON for consumption by Rust test
        json_path = "sklearn_bench_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {json_path}")
    else:
        print_table(results)


if __name__ == "__main__":
    main()
