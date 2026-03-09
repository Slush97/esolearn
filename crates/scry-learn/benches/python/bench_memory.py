#!/usr/bin/env python3
"""Memory footprint benchmark for sklearn models (comparison baseline).

Usage:
    python bench_memory.py
"""

import tracemalloc
import pickle
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA


def synthetic_clf(n_samples, n_features, n_classes):
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, size=n_samples)
    return X, y


def synthetic_reg(n_samples, n_features):
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    y = X @ rng.randn(n_features) + rng.randn(n_samples) * 0.1
    return X, y


def bench_model(name, fit_fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    model = fit_fn()
    fit_ms = (time.perf_counter() - t0) * 1000
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kb /= 1024

    serialized = pickle.dumps(model)
    return {
        "model": name,
        "peak_kb": peak_kb,
        "pickle_bytes": len(serialized),
        "fit_ms": fit_ms,
    }


def main():
    sizes = [1_000, 10_000, 100_000]
    n_features = 10
    n_classes = 3

    results = []

    for n in sizes:
        print(f"\n--- {n} samples x {n_features} features ---")
        X_clf, y_clf = synthetic_clf(n, n_features, n_classes)
        X_reg, y_reg = synthetic_reg(n, n_features)

        models = [
            ("DecisionTreeClassifier", lambda: DecisionTreeClassifier(max_depth=10).fit(X_clf, y_clf)),
            ("RandomForestClassifier", lambda: RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42).fit(X_clf, y_clf)),
            ("GradientBoostingClassifier", lambda: GradientBoostingClassifier(n_estimators=10, max_depth=5, random_state=42).fit(X_clf, y_clf)),
            ("LinearRegression", lambda: LinearRegression().fit(X_reg, y_reg)),
            ("LogisticRegression", lambda: LogisticRegression(max_iter=200).fit(X_clf, y_clf)),
            ("KNeighborsClassifier", lambda: KNeighborsClassifier(n_neighbors=5).fit(X_clf, y_clf)),
            ("KMeans", lambda: KMeans(n_clusters=n_classes, random_state=42, n_init=10).fit(X_clf)),
            ("GaussianNB", lambda: GaussianNB().fit(X_clf, y_clf)),
            ("PCA", lambda: PCA().fit(X_clf)),
        ]

        for name, fit_fn in models:
            r = bench_model(name, fit_fn)
            r["n_samples"] = n
            results.append(r)

    # Print table
    print(f"\n{'Model':<30} {'Samples':>10} {'Peak mem':>12} {'Pickle':>14} {'Fit (ms)':>10}")
    print("-" * 78)
    for r in results:
        sz = r["pickle_bytes"]
        if sz < 1024:
            sz_str = f"{sz} B"
        elif sz < 1024 * 1024:
            sz_str = f"{sz / 1024:.1f} KB"
        else:
            sz_str = f"{sz / (1024 * 1024):.1f} MB"

        pk = r["peak_kb"]
        if pk < 1024:
            pk_str = f"{pk:.0f} KB"
        else:
            pk_str = f"{pk / 1024:.1f} MB"

        print(f"{r['model']:<30} {r['n_samples']:>10} {pk_str:>12} {sz_str:>14} {r['fit_ms']:>10.1f}")


if __name__ == "__main__":
    main()
