#!/usr/bin/env python3
"""
Head-to-head benchmark: scikit-learn vs scry-learn.

Mirrors the exact same datasets, configurations, and measurements
from the Rust benchmark suite (benches/ml_algorithms.rs).

Usage: /tmp/scry-bench/bin/python3 benches/sklearn_comparison.py
"""

import time
import json
import numpy as np
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────
# Deterministic data generators (MUST match Rust implementations)
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
    y += rng.random(n) * 0.5  # noise
    return X, y


# ─────────────────────────────────────────────────────────────────
# Benchmark harness
# ─────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    median_us: float  # microseconds
    min_us: float
    max_us: float
    iters: int


def bench(name: str, fn, n_iters: int = 20, warmup: int = 3) -> BenchResult:
    """Run a benchmark with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(n_iters):
        start = time.perf_counter_ns()
        fn()
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed / 1000)  # to microseconds

    times.sort()
    median = times[len(times) // 2]
    return BenchResult(
        name=name,
        median_us=median,
        min_us=times[0],
        max_us=times[-1],
        iters=n_iters,
    )


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
# Benchmarks
# ─────────────────────────────────────────────────────────────────

def run_training_benchmarks():
    """Group 1: Training throughput — all algorithms (1K×10)."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.naive_bayes import GaussianNB

    X_clf, y_clf = gen_classification(1000, 10)
    X_reg, y_reg = gen_regression(1000, 10)

    results = []

    results.append(bench("decision_tree_clf/1k×10", lambda: (
        DecisionTreeClassifier().fit(X_clf, y_clf)
    )))

    results.append(bench("decision_tree_reg/1k×10", lambda: (
        DecisionTreeRegressor().fit(X_reg, y_reg)
    )))

    results.append(bench("random_forest_clf/1k×10/20trees", lambda: (
        RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1).fit(X_clf, y_clf)
    )))

    results.append(bench("linear_regression/1k×10", lambda: (
        LinearRegression().fit(X_reg, y_reg)
    )))

    results.append(bench("logistic_regression/1k×10", lambda: (
        LogisticRegression(max_iter=200, solver='lbfgs').fit(X_clf, y_clf)
    )))

    results.append(bench("knn_clf/1k×10", lambda: (
        KNeighborsClassifier(n_neighbors=5).fit(X_clf, y_clf)
    )))

    results.append(bench("kmeans/1k×10/k=3", lambda: (
        KMeans(n_clusters=3, random_state=42, max_iter=100, n_init=1).fit(X_clf)
    )))

    results.append(bench("dbscan/1k×10", lambda: (
        DBSCAN(eps=3.0, min_samples=5).fit(X_clf)
    )))

    results.append(bench("gaussian_nb/1k×10", lambda: (
        GaussianNB().fit(X_clf, y_clf)
    )))

    return results


def run_prediction_benchmarks():
    """Group 2: Prediction latency — all algorithms (1K samples)."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    X_clf, y_clf = gen_classification(1000, 10)
    X_reg, y_reg = gen_regression(1000, 10)

    # Pre-train
    dt_clf = DecisionTreeClassifier().fit(X_clf, y_clf)
    dt_reg = DecisionTreeRegressor().fit(X_reg, y_reg)
    rf = RandomForestClassifier(n_estimators=20, random_state=42).fit(X_clf, y_clf)
    lr = LinearRegression().fit(X_reg, y_reg)
    log_reg = LogisticRegression(max_iter=200).fit(X_clf, y_clf)
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_clf, y_clf)
    nb = GaussianNB().fit(X_clf, y_clf)

    results = []

    results.append(bench("decision_tree_clf/1k", lambda: dt_clf.predict(X_clf), n_iters=50))
    results.append(bench("decision_tree_reg/1k", lambda: dt_reg.predict(X_reg), n_iters=50))
    results.append(bench("random_forest_clf/1k", lambda: rf.predict(X_clf), n_iters=50))
    results.append(bench("linear_regression/1k", lambda: lr.predict(X_reg), n_iters=50))
    results.append(bench("logistic_regression/1k", lambda: log_reg.predict(X_clf), n_iters=50))
    results.append(bench("knn/1k", lambda: knn.predict(X_clf), n_iters=30))
    results.append(bench("gaussian_nb/1k", lambda: nb.predict(X_clf), n_iters=50))

    return results


def run_scaling_benchmarks():
    """Group 3: Dataset size scaling."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    sizes = [100, 500, 1000, 5000, 10000]
    results = []

    for n in sizes:
        X, y = gen_classification(n, 10)

        results.append(bench(f"decision_tree/{n}", lambda X=X, y=y: (
            DecisionTreeClassifier(max_depth=10).fit(X, y)
        ), n_iters=10))

        results.append(bench(f"random_forest/10t/{n}", lambda X=X, y=y: (
            RandomForestClassifier(n_estimators=10, max_depth=8, random_state=42, n_jobs=-1).fit(X, y)
        ), n_iters=10))

    return results


def run_forest_scaling():
    """Group 5: Random Forest tree count scaling."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = gen_classification(2000, 10)
    tree_counts = [5, 10, 25, 50, 100]
    results = []

    for n_trees in tree_counts:
        results.append(bench(f"rf_train/{n_trees}t", lambda n=n_trees: (
            RandomForestClassifier(n_estimators=n, max_depth=8, random_state=42, n_jobs=-1).fit(X, y)
        ), n_iters=10))

    # Prediction scaling
    for n_trees in tree_counts:
        model = RandomForestClassifier(n_estimators=n_trees, max_depth=8, random_state=42).fit(X, y)
        results.append(bench(f"rf_predict/{n_trees}t", lambda m=model: m.predict(X), n_iters=20))

    return results


def run_metrics_benchmarks():
    """Group 7: Metrics computation."""
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix,
        classification_report, roc_auc_score,
        mean_squared_error, r2_score,
    )

    n = 10_000
    rng = np.random.default_rng(42)
    y_true = np.array([0.0 if i < n // 2 else 1.0 for i in range(n)])
    y_pred = np.where(rng.random(n) < 0.9, y_true, 1.0 - y_true)
    y_scores = y_true + (rng.random(n) - 0.5) * 0.4

    y_true_reg = np.arange(n, dtype=float) * 0.1
    y_pred_reg = y_true_reg + rng.random(n) * 0.5

    results = []

    results.append(bench("accuracy/10k", lambda: accuracy_score(y_true, y_pred), n_iters=50))
    results.append(bench("f1_macro/10k", lambda: f1_score(y_true, y_pred, average='macro'), n_iters=50))
    results.append(bench("confusion_matrix/10k", lambda: confusion_matrix(y_true, y_pred), n_iters=50))
    results.append(bench("classification_report/10k", lambda: classification_report(y_true, y_pred), n_iters=50))
    results.append(bench("roc_auc/10k", lambda: roc_auc_score(y_true, y_scores), n_iters=50))
    results.append(bench("mse/10k", lambda: mean_squared_error(y_true_reg, y_pred_reg), n_iters=50))
    results.append(bench("r2/10k", lambda: r2_score(y_true_reg, y_pred_reg), n_iters=50))

    return results


def run_correctness_verification():
    """Verify algorithm correctness against sklearn on known datasets."""
    from sklearn.datasets import load_iris, make_classification, make_regression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

    print("\n" + "=" * 70)
    print("CORRECTNESS VERIFICATION — sklearn reference results")
    print("=" * 70)

    # Iris dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    print(f"\nIris dataset: {X_train.shape[0]} train, {X_test.shape[0]} test, "
          f"{iris.data.shape[1]} features, {len(set(iris.target))} classes")

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest (100t)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Gaussian NB": GaussianNB(),
    }

    print(f"\n{'Algorithm':<25} {'Accuracy':>10} {'Notes'}")
    print("-" * 55)
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"{name:<25} {acc:>9.1%}  {'✓ reference' }")

    # Regression
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=5, random_state=42)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    lr = LinearRegression().fit(Xr_train, yr_train)
    y_pred = lr.predict(Xr_test)
    r2 = r2_score(yr_test, y_pred)
    mse = mean_squared_error(yr_test, y_pred)
    print(f"\nLinear Regression (500×10): R²={r2:.4f}, MSE={mse:.2f}")

    # Clustering
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(iris.data)
    print(f"\nK-Means (Iris, k=3): inertia={km.inertia_:.2f}, "
          f"cluster sizes={np.bincount(km.labels_).tolist()}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("scikit-learn Benchmark Suite")
    print(f"sklearn version: ", end="")
    import sklearn
    print(sklearn.__version__)
    print("=" * 70)

    all_groups = [
        ("TRAINING (1K×10)", run_training_benchmarks),
        ("PREDICTION (1K samples)", run_prediction_benchmarks),
        ("DATASET SCALING", run_scaling_benchmarks),
        ("FOREST TREE SCALING (2K×10)", run_forest_scaling),
        ("METRICS (10K samples)", run_metrics_benchmarks),
    ]

    all_results = {}

    for group_name, group_fn in all_groups:
        print(f"\n{'─' * 70}")
        print(f"  {group_name}")
        print(f"{'─' * 70}")
        results = group_fn()
        for r in results:
            print(f"  {r.name:<45} {fmt_time(r.median_us):>12}  "
                  f"[{fmt_time(r.min_us)} .. {fmt_time(r.max_us)}]")
            all_results[r.name] = r.median_us

    # Correctness verification
    run_correctness_verification()

    # Save results as JSON for comparison
    with open("benches/sklearn_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to benches/sklearn_results.json")


if __name__ == "__main__":
    main()
