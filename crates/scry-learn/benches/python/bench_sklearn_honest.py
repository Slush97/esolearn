#!/usr/bin/env python3
"""
Honest Benchmark — sklearn counterpart.

Loads the EXACT SAME UCI CSV fixture files used by the Rust benchmark to
eliminate RNG mismatch.  Enforces single-threaded execution at TWO levels:
  1. n_jobs=1 for joblib parallelism (RF, KNN)
  2. OMP/MKL/OPENBLAS env vars = "1" for BLAS-level threading (LogReg, KMeans)

Measures memory via tracemalloc (actual heap), not RSS.

Usage:
    python3 bench_sklearn_honest.py

Output:
    honest_sklearn_results.json
"""

# ── MUST be set BEFORE importing numpy/scipy/sklearn ──
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────
# Configuration — MUST match honest_bench.rs constants exactly
# ─────────────────────────────────────────────────────────────────

DT_MAX_DEPTH = 10
RF_N_TREES = 20
RF_MAX_DEPTH = 10
LR_MAX_ITER = 200
KNN_K = 5
KM_K = 3
KM_MAX_ITER = 100
LASSO_ALPHA = 0.01
LASSO_MAX_ITER = 1000

DATASETS = ["iris", "wine", "breast_cancer"]
FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


# ─────────────────────────────────────────────────────────────────
# Data loading — same CSV fixtures as Rust
# ─────────────────────────────────────────────────────────────────

def load_dataset(name: str):
    """Load UCI dataset from CSV fixtures. Returns (X, y)."""
    feat_path = FIXTURES_DIR / f"{name}_features.csv"
    targ_path = FIXTURES_DIR / f"{name}_target.csv"
    with open(feat_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        X = np.array([[float(v) for v in row] for row in reader])
    with open(targ_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        y = np.array([float(row[0]) for row in reader])
    return X, y


# ─────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────

def time_fn(fn, n_runs=30, warmup=3):
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
    return {
        "median_us": round(times[len(times) // 2], 2),
        "p5_us": round(times[max(0, len(times) // 20)], 2),
        "p95_us": round(times[int(len(times) * 0.95)], 2),
    }


def fmt_time(us: float) -> str:
    if us < 1:
        return f"{us * 1000:.1f} ns"
    if us < 1000:
        return f"{us:.1f} µs"
    if us < 1_000_000:
        return f"{us / 1000:.2f} ms"
    return f"{us / 1_000_000:.2f} s"


def accuracy(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) < 1e-9)


# ─────────────────────────────────────────────────────────────────
# §1 Cold Start
# ─────────────────────────────────────────────────────────────────

def run_cold_start():
    print("\n" + "=" * 65)
    print("  §1 COLD START (construct → fit → predict 1 row)")
    print("=" * 65)

    X, y = load_dataset("iris")
    single_row = X[:1]
    results = {}

    # DT
    t = time_fn(lambda: DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=42)
                .fit(X, y).predict(single_row))
    print(f"  DT sklearn:               {fmt_time(t['median_us']):>12}")
    results["cold_start/dt"] = t

    # KNN
    t = time_fn(lambda: KNeighborsClassifier(n_neighbors=KNN_K, n_jobs=1)
                .fit(X, y).predict(single_row))
    print(f"  KNN sklearn:              {fmt_time(t['median_us']):>12}")
    results["cold_start/knn"] = t

    # RF
    t = time_fn(lambda: RandomForestClassifier(
        n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH,
        random_state=42, n_jobs=1).fit(X, y).predict(single_row))
    print(f"  RF sklearn:               {fmt_time(t['median_us']):>12}")
    results["cold_start/rf"] = t

    return results


# ─────────────────────────────────────────────────────────────────
# §2 Training Throughput
# ─────────────────────────────────────────────────────────────────

def run_training():
    print("\n" + "=" * 65)
    print("  §2 TRAINING THROUGHPUT (n_jobs=1)")
    print("=" * 65)

    results = {}

    for ds_name in DATASETS:
        X, y = load_dataset(ds_name)
        print(f"\n  Dataset: {ds_name} ({X.shape[0]}×{X.shape[1]})")

        # DT
        t = time_fn(lambda: DecisionTreeClassifier(
            max_depth=DT_MAX_DEPTH, random_state=42).fit(X, y))
        print(f"    DT:     {fmt_time(t['median_us']):>12}")
        results[f"train/dt/{ds_name}"] = t

        # RF
        t = time_fn(lambda: RandomForestClassifier(
            n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH,
            random_state=42, n_jobs=1).fit(X, y), n_runs=10)
        print(f"    RF:     {fmt_time(t['median_us']):>12}")
        results[f"train/rf/{ds_name}"] = t

        # KNN (fit only)
        t = time_fn(lambda: KNeighborsClassifier(
            n_neighbors=KNN_K, n_jobs=1).fit(X, y))
        print(f"    KNN:    {fmt_time(t['median_us']):>12}")
        results[f"train/knn/{ds_name}"] = t

    # Logistic Regression (breast_cancer, standardized)
    X, y = load_dataset("breast_cancer")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    t = time_fn(lambda: LogisticRegression(
        max_iter=LR_MAX_ITER, solver="lbfgs", random_state=42
    ).fit(X_std, y))
    print(f"\n  LogReg (breast_cancer, standardized): {fmt_time(t['median_us']):>12}")
    results["train/logreg/breast_cancer"] = t

    # K-Means
    for ds_name in DATASETS:
        X, y = load_dataset(ds_name)
        t = time_fn(lambda: KMeans(
            n_clusters=KM_K, max_iter=KM_MAX_ITER,
            random_state=42, n_init=1).fit(X))
        print(f"  KMeans/{ds_name}: {fmt_time(t['median_us']):>12}")
        results[f"train/kmeans/{ds_name}"] = t

    return results


# ─────────────────────────────────────────────────────────────────
# §3 Prediction Latency
# ─────────────────────────────────────────────────────────────────

def run_prediction():
    print("\n" + "=" * 65)
    print("  §3 PREDICTION LATENCY (batch, n_jobs=1)")
    print("=" * 65)

    results = {}

    for ds_name in DATASETS:
        X, y = load_dataset(ds_name)
        print(f"\n  Dataset: {ds_name}")

        # DT
        dt = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=42).fit(X, y)
        preds = dt.predict(X)
        acc_dt = accuracy(y, preds)

        t = time_fn(lambda: dt.predict(X))
        print(f"    DT:     {fmt_time(t['median_us']):>12}  acc={acc_dt:.4f}")
        results[f"predict/dt/{ds_name}"] = {**t, "accuracy": round(float(acc_dt), 6)}

        # RF
        rf = RandomForestClassifier(
            n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH,
            random_state=42, n_jobs=1).fit(X, y)
        preds_rf = rf.predict(X)
        acc_rf = accuracy(y, preds_rf)

        t = time_fn(lambda: rf.predict(X))
        print(f"    RF:     {fmt_time(t['median_us']):>12}  acc={acc_rf:.4f}")
        results[f"predict/rf/{ds_name}"] = {**t, "accuracy": round(float(acc_rf), 6)}

        # KNN
        knn = KNeighborsClassifier(n_neighbors=KNN_K, n_jobs=1).fit(X, y)
        preds_knn = knn.predict(X)
        acc_knn = accuracy(y, preds_knn)

        t = time_fn(lambda: knn.predict(X))
        print(f"    KNN:    {fmt_time(t['median_us']):>12}  acc={acc_knn:.4f}")
        results[f"predict/knn/{ds_name}"] = {**t, "accuracy": round(float(acc_knn), 6)}

    return results


# ─────────────────────────────────────────────────────────────────
# §4 Memory Footprint (tracemalloc)
# ─────────────────────────────────────────────────────────────────

def fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.2f} MB"


def run_memory():
    print("\n" + "=" * 65)
    print("  §4 MEMORY FOOTPRINT (tracemalloc, breast_cancer)")
    print("=" * 65)

    X, y = load_dataset("breast_cancer")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    results = {}
    models = {
        "DT": lambda: DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=42).fit(X, y),
        "RF": lambda: RandomForestClassifier(
            n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH,
            random_state=42, n_jobs=1).fit(X, y),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=KNN_K, n_jobs=1).fit(X, y),
        "LogReg": lambda: LogisticRegression(
            max_iter=LR_MAX_ITER, solver="lbfgs", random_state=42
        ).fit(X_std, y),
    }

    for name, factory in models.items():
        tracemalloc.start()
        model = factory()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  {name:<15} current={fmt_bytes(current):>10}  peak={fmt_bytes(peak):>10}")
        results[f"memory/{name.lower()}"] = {
            "current_bytes": current,
            "peak_bytes": peak,
        }
        del model  # noqa

    return results


# ─────────────────────────────────────────────────────────────────
# §5 Accuracy Parity (for comparison with Rust results)
# ─────────────────────────────────────────────────────────────────

def run_accuracy():
    print("\n" + "=" * 65)
    print("  §5 ACCURACY (train=test, for parity comparison only)")
    print("=" * 65)

    results = {}
    for ds_name in DATASETS:
        X, y = load_dataset(ds_name)
        print(f"\n  Dataset: {ds_name} ({X.shape[0]}×{X.shape[1]})")

        models = {
            "DT": DecisionTreeClassifier(max_depth=DT_MAX_DEPTH, random_state=42),
            "RF": RandomForestClassifier(
                n_estimators=RF_N_TREES, max_depth=RF_MAX_DEPTH,
                random_state=42, n_jobs=1),
            "KNN": KNeighborsClassifier(n_neighbors=KNN_K, n_jobs=1),
        }

        for model_name, model in models.items():
            model.fit(X, y)
            preds = model.predict(X)
            acc = accuracy(y, preds)
            print(f"    {model_name:<10} {acc:.4f} ({acc*100:.1f}%)")
            results[f"accuracy/{model_name.lower()}/{ds_name}"] = round(float(acc), 6)

    return results


# ─────────────────────────────────────────────────────────────────
# §6 Lasso (california housing)
# ─────────────────────────────────────────────────────────────────

def run_lasso():
    print("\n" + "=" * 65)
    print("  §6 LASSO (California Housing)")
    print("=" * 65)

    X, y = load_dataset("california")
    results = {}

    t = time_fn(lambda: Lasso(alpha=LASSO_ALPHA, max_iter=LASSO_MAX_ITER).fit(X, y), n_runs=30)
    print(f"  Lasso sklearn:  {fmt_time(t['median_us']):>12}")
    results["lasso/train"] = t

    return results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    import sklearn
    print("=" * 65)
    print(f"  Honest Benchmark — sklearn {sklearn.__version__}")
    print(f"  NumPy {np.__version__}")
    print(f"  Single-threaded: n_jobs=1 + OMP/MKL/OPENBLAS_NUM_THREADS=1")
    print(f"  Data: real UCI CSV fixtures (same as Rust)")
    print("=" * 65)

    all_results = {
        "meta": {
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "n_jobs": 1,
            "data_source": "UCI CSV fixtures (shared with Rust)",
        }
    }

    all_results["cold_start"] = run_cold_start()
    all_results["training"] = run_training()
    all_results["prediction"] = run_prediction()
    all_results["memory"] = run_memory()
    all_results["accuracy"] = run_accuracy()
    all_results["lasso"] = run_lasso()

    out_path = Path(__file__).parent / "honest_sklearn_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
