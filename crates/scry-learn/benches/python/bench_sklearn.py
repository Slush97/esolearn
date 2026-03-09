#!/usr/bin/env python3
"""
Industry benchmark: scikit-learn baselines for scry-learn comparison.

Measures 5-fold stratified CV accuracy on real UCI datasets, training
throughput, and single-row prediction latency for all model families.

Usage:
    python3 bench_sklearn.py

Output:
    sklearn_cv_results.json — structured results for cross-language comparison
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB


# ─────────────────────────────────────────────────────────────────
# Synthetic data generators (match Rust implementations exactly)
# ─────────────────────────────────────────────────────────────────

def gen_classification(n: int, n_features: int, seed: int = 42):
    """Generate classification data matching Rust gen_classification()."""
    rng = np.random.default_rng(seed)
    half = n // 2
    X = np.zeros((n, n_features))
    y = np.zeros(n)
    for j in range(n_features):
        offset = 3.0 + j * 0.5
        X[:half, j] = rng.random(half) * 2.0
        X[half:, j] = rng.random(n - half) * 2.0 + offset
    y[half:] = 1.0
    return X, y


def gen_regression(n: int, n_features: int, seed: int = 42):
    """Generate regression data matching Rust gen_regression()."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_features)) * 10.0
    y = np.zeros(n)
    for j in range(n_features):
        y += X[:, j] * (j + 1)
    y += rng.random(n) * 0.5
    return X, y


# ─────────────────────────────────────────────────────────────────
# UCI dataset loaders
# ─────────────────────────────────────────────────────────────────

DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "adult": None,  # loaded from CSV fixtures via load_uci()
}

# Path to CSV fixtures (shared with Rust benchmarks for data identity)
FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"


def load_uci(name: str):
    """Load a UCI dataset, returning (X, y)."""
    if name == "adult":
        # Load from CSV fixtures (same data as Rust benchmarks)
        import csv
        features_path = FIXTURES_DIR / "adult_features.csv"
        target_path = FIXTURES_DIR / "adult_target.csv"
        with open(features_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            X = np.array([[float(v) for v in row] for row in reader])
        with open(target_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            y = np.array([float(row[0]) for row in reader])
        return X, y
    loader = DATASETS[name]
    data = loader()
    return data.data, data.target.astype(float)


# ─────────────────────────────────────────────────────────────────
# Model configurations (matching scry-learn defaults where possible)
# ─────────────────────────────────────────────────────────────────

def get_classifiers():
    """Return dict of name → (model, needs_scaling)."""
    return {
        "decision_tree": (DecisionTreeClassifier(max_depth=10, random_state=42), False),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=20, max_depth=10, random_state=42, n_jobs=1
            ),
            False,
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ),
            False,
        ),
        "hist_gbt": (
            HistGradientBoostingClassifier(
                max_iter=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            False,
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=200, solver="lbfgs", random_state=42),
            True,
        ),
        "knn": (KNeighborsClassifier(n_neighbors=5), True),
        "linear_svc": (
            LinearSVC(max_iter=2000, random_state=42, dual="auto"),
            True,
        ),
        "gaussian_nb": (GaussianNB(), False),
    }


# ─────────────────────────────────────────────────────────────────
# Benchmarking helpers
# ─────────────────────────────────────────────────────────────────

def time_fn(fn, n_runs=5, warmup=2):
    """Time a function, return median microseconds."""
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


def prediction_latency(model, X, n_iters=10_000):
    """Measure single-row prediction latency, return (p50, p95, p99) in µs."""
    # Pick a random subset of rows to predict
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


def fmt_time(us: float) -> str:
    """Format microseconds to human-readable."""
    if us < 1:
        return f"{us * 1000:.1f} ns"
    if us < 1000:
        return f"{us:.1f} µs"
    if us < 1_000_000:
        return f"{us / 1000:.2f} ms"
    return f"{us / 1_000_000:.2f} s"


# ─────────────────────────────────────────────────────────────────
# Section 1: 5-Fold Stratified CV Accuracy on UCI Datasets
# ─────────────────────────────────────────────────────────────────

def run_accuracy_benchmarks():
    """5-fold stratified CV accuracy for each model × dataset."""
    print("\n" + "=" * 72)
    print("  SECTION 1: 5-Fold Stratified CV Accuracy (UCI Datasets)")
    print("=" * 72)

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ds_name in DATASETS:
        X, y = load_uci(ds_name)
        print(f"\n  Dataset: {ds_name} ({X.shape[0]} samples, {X.shape[1]} features, "
              f"{len(np.unique(y))} classes)")
        print(f"  {'Model':<25} {'Mean Acc':>10} {'Std':>8} {'Folds':>30}")
        print("  " + "-" * 75)

        for model_name, (model, needs_scaling) in get_classifiers().items():
            # KMeans is unsupervised — skip accuracy CV
            if needs_scaling:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", model),
                ])
            else:
                pipe = model

            try:
                scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
                mean_acc = scores.mean()
                std_acc = scores.std()
                fold_str = ", ".join(f"{s:.3f}" for s in scores)
                print(f"  {model_name:<25} {mean_acc:>9.4f} {std_acc:>7.4f}  [{fold_str}]")
                results[f"{model_name}/{ds_name}"] = {
                    "mean_accuracy": round(float(mean_acc), 6),
                    "std_accuracy": round(float(std_acc), 6),
                    "fold_scores": [round(float(s), 6) for s in scores],
                }
            except Exception as e:
                print(f"  {model_name:<25} {'FAILED':>10}  {e}")
                results[f"{model_name}/{ds_name}"] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────
# Section 2: Training Throughput
# ─────────────────────────────────────────────────────────────────

def run_training_benchmarks():
    """Training wall-clock time at various dataset sizes."""
    print("\n" + "=" * 72)
    print("  SECTION 2: Training Throughput")
    print("=" * 72)

    sizes = [1_000, 10_000, 100_000]
    results = {}

    for n in sizes:
        X, y = gen_classification(n, 10)
        print(f"\n  Size: {n:,} × 10 features")
        print(f"  {'Model':<25} {'Median Time':>12}")
        print("  " + "-" * 40)

        for model_name, (model, needs_scaling) in get_classifiers().items():
            if needs_scaling:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X

            try:
                median_us = time_fn(lambda X=X_scaled, y=y, m=model: m.fit(X, y))
                print(f"  {model_name:<25} {fmt_time(median_us):>12}")
                results[f"{model_name}/{n}"] = {
                    "median_us": round(median_us, 2),
                    "n_samples": n,
                }
            except Exception as e:
                print(f"  {model_name:<25} {'FAILED':>12}  {e}")

    return results


# ─────────────────────────────────────────────────────────────────
# Section 3: Single-Row Prediction Latency
# ─────────────────────────────────────────────────────────────────

def run_prediction_benchmarks():
    """Single-row prediction latency (p50/p95/p99)."""
    print("\n" + "=" * 72)
    print("  SECTION 3: Single-Row Prediction Latency (1K training set)")
    print("=" * 72)

    X, y = gen_classification(1000, 10)
    results = {}

    print(f"\n  {'Model':<25} {'p50':>10} {'p95':>10} {'p99':>10}")
    print("  " + "-" * 58)

    for model_name, (model, needs_scaling) in get_classifiers().items():
        if needs_scaling:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        try:
            model.fit(X_scaled, y)
            latency = prediction_latency(model, X_scaled, n_iters=5_000)
            print(f"  {model_name:<25} {fmt_time(latency['p50_us']):>10} "
                  f"{fmt_time(latency['p95_us']):>10} {fmt_time(latency['p99_us']):>10}")
            results[model_name] = latency
        except Exception as e:
            print(f"  {model_name:<25} {'FAILED':>10}  {e}")

    return results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    import sklearn
    print("=" * 72)
    print(f"  scikit-learn Industry Benchmark — v{sklearn.__version__}")
    print(f"  NumPy {np.__version__}")
    print("=" * 72)

    all_results = {}
    all_results["accuracy_cv"] = run_accuracy_benchmarks()
    all_results["training"] = run_training_benchmarks()
    all_results["prediction_latency"] = run_prediction_benchmarks()

    # Write JSON
    out_path = Path(__file__).parent / "sklearn_cv_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
