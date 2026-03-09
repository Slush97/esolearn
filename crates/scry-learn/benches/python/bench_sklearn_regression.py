#!/usr/bin/env python3
"""
Industry benchmark: scikit-learn regression baselines on California Housing.

Companion to bench_sklearn.py (classification). Measures R², RMSE, MAE,
training time, and single-row prediction latency for 6 regression models.

Usage:
    python3 bench_sklearn_regression.py

Output:
    sklearn_regression_results.json — structured results for cross-language comparison
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

# Models
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ─────────────────────────────────────────────────────────────────
# Dataset — loaded from local CSV fixtures (no network needed)
# ─────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


def _load_csv_features(name: str) -> np.ndarray:
    """Load a feature CSV into a numpy array."""
    import csv
    path = FIXTURES_DIR / name
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = [[float(v) for v in row] for row in reader]
    return np.array(rows)


def _load_csv_target(name: str) -> np.ndarray:
    """Load a target CSV into a numpy array."""
    import csv
    path = FIXTURES_DIR / name
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        values = [float(row[0]) for row in reader]
    return np.array(values)


def load_california():
    """Load California Housing from local fixtures, return (X_train, X_test, y_train, y_test) scaled."""
    X = _load_csv_features("california_features.csv")
    y = _load_csv_target("california_target.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────
# Model configurations
# ─────────────────────────────────────────────────────────────────

def get_regressors():
    """Return dict of name → model."""
    return {
        "linear_regression": LinearRegression(),
        "lasso": Lasso(alpha=0.01, max_iter=1000, random_state=42),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=42),
        "knn_regressor": KNeighborsRegressor(n_neighbors=5),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "ridge": Ridge(alpha=1.0),
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
# Section 1: Regression Metrics on California Housing
# ─────────────────────────────────────────────────────────────────

def run_regression_metrics():
    """R², RMSE, MAE for each regressor on California Housing (80/20 split)."""
    print("\n" + "=" * 72)
    print("  SECTION 1: Regression Metrics — California Housing (80/20 split)")
    print("=" * 72)

    X_train, X_test, y_train, y_test = load_california()
    results = {}

    print(f"\n  {'Model':<25} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-" * 52)

    for name, model in get_regressors().items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = root_mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            print(f"  {name:<25} {r2:>8.4f} {rmse:>8.4f} {mae:>8.4f}")
            results[name] = {
                "r2": round(float(r2), 6),
                "rmse": round(float(rmse), 6),
                "mae": round(float(mae), 6),
            }
        except Exception as e:
            print(f"  {name:<25} {'FAILED':>8}  {e}")
            results[name] = {"error": str(e)}

    return results


# ─────────────────────────────────────────────────────────────────
# Section 2: Training Time
# ─────────────────────────────────────────────────────────────────

def run_training_benchmarks():
    """Training wall-clock time (median of 5 runs)."""
    print("\n" + "=" * 72)
    print("  SECTION 2: Training Time (California Housing, full train set)")
    print("=" * 72)

    X_train, _, y_train, _ = load_california()
    results = {}

    print(f"\n  {'Model':<25} {'Median Time':>12}")
    print("  " + "-" * 40)

    for name, model in get_regressors().items():
        try:
            # Fresh model each time
            model_cls = model.__class__
            params = model.get_params()

            def train():
                m = model_cls(**params)
                m.fit(X_train, y_train)

            median_us = time_fn(train)
            print(f"  {name:<25} {fmt_time(median_us):>12}")
            results[name] = {
                "median_us": round(median_us, 2),
                "n_samples": len(X_train),
            }
        except Exception as e:
            print(f"  {name:<25} {'FAILED':>12}  {e}")

    return results


# ─────────────────────────────────────────────────────────────────
# Section 3: Single-Row Prediction Latency
# ─────────────────────────────────────────────────────────────────

def run_prediction_benchmarks():
    """Single-row prediction latency (p50/p95/p99)."""
    print("\n" + "=" * 72)
    print("  SECTION 3: Single-Row Prediction Latency")
    print("=" * 72)

    X_train, X_test, y_train, _ = load_california()
    results = {}

    print(f"\n  {'Model':<25} {'p50':>10} {'p95':>10} {'p99':>10}")
    print("  " + "-" * 58)

    for name, model in get_regressors().items():
        try:
            model.fit(X_train, y_train)
            latency = prediction_latency(model, X_test, n_iters=5_000)
            print(f"  {name:<25} {fmt_time(latency['p50_us']):>10} "
                  f"{fmt_time(latency['p95_us']):>10} {fmt_time(latency['p99_us']):>10}")
            results[name] = latency
        except Exception as e:
            print(f"  {name:<25} {'FAILED':>10}  {e}")

    return results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    import sklearn
    print("=" * 72)
    print(f"  scikit-learn Regression Benchmark — v{sklearn.__version__}")
    print(f"  NumPy {np.__version__}")
    print(f"  Dataset: California Housing (20640 samples, 8 features)")
    print("=" * 72)

    all_results = {}
    all_results["regression_metrics"] = run_regression_metrics()
    all_results["training_time"] = run_training_benchmarks()
    all_results["prediction_latency"] = run_prediction_benchmarks()

    # Write JSON
    out_path = Path(__file__).parent / "sklearn_regression_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
