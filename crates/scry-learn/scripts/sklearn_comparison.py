#!/usr/bin/env python3
"""
Fair side-by-side comparison: scry-learn vs scikit-learn.

Methodology:
  - Same UCI datasets (iris, wine, breast_cancer, digits)
  - Same hyperparameters (from benchmark_config.rs)
  - Same evaluation: 5-fold StratifiedKFold, seed=42
  - Models that need scaling (KNN, LogReg) get StandardScaler per fold
  - No cherry-picking: every model × dataset pair from the golden baselines

Usage:
  pip install scikit-learn
  python3 crates/scry-learn/scripts/sklearn_comparison.py

Then compare with:
  cargo test --test golden_regression_test -p scry-learn --release -- --nocapture
"""

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Regression
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge

# ═══════════════════════════════════════════════════════════════════
# Hyperparameters — identical to benchmark_config.rs
# ═══════════════════════════════════════════════════════════════════

SEED = 42

CONFIGS = {
    "DecisionTree":      lambda: DecisionTreeClassifier(max_depth=10, random_state=SEED),
    "RandomForest":      lambda: RandomForestClassifier(n_estimators=20, max_depth=10, random_state=SEED),
    "GradientBoosting":  lambda: GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=SEED),
    "HistGBT":           lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.1, random_state=SEED),
    "GaussianNB":        lambda: GaussianNB(),
    "KNN":               lambda: KNeighborsClassifier(n_neighbors=5, weights="uniform"),
}

DATASETS = {
    "iris":          load_iris,
    "wine":          load_wine,
    "breast_cancer": load_breast_cancer,
    "digits":        load_digits,
}

# Models that require feature scaling
NEEDS_SCALING = {"KNN"}

# Which models run on which datasets (matches golden_baselines_classification)
BASELINES = [
    # iris
    ("DecisionTree",     "iris"),
    ("RandomForest",     "iris"),
    ("GradientBoosting", "iris"),
    ("HistGBT",          "iris"),
    ("GaussianNB",       "iris"),
    ("KNN",              "iris"),
    # wine
    ("DecisionTree",     "wine"),
    ("RandomForest",     "wine"),
    ("GradientBoosting", "wine"),
    ("HistGBT",          "wine"),
    ("GaussianNB",       "wine"),
    # breast_cancer
    ("DecisionTree",     "breast_cancer"),
    ("RandomForest",     "breast_cancer"),
    ("GradientBoosting", "breast_cancer"),
    ("HistGBT",          "breast_cancer"),
    ("GaussianNB",       "breast_cancer"),
    # digits
    ("DecisionTree",     "digits"),
    ("RandomForest",     "digits"),
    ("GradientBoosting", "digits"),
    ("HistGBT",          "digits"),
    ("GaussianNB",       "digits"),
    ("KNN",              "digits"),
]


def cross_val_score_stratified(model_fn, X, y, n_folds=5, seed=42, scale=False):
    """5-fold stratified CV, matching scry-learn's cross_val_score_stratified."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return scores


def main():
    # Cache loaded datasets
    loaded = {}
    for name, loader in DATASETS.items():
        data = loader()
        loaded[name] = (data.data, data.target)

    print()
    print("=" * 84)
    print("  sklearn BASELINE — Classification (5-fold stratified CV, seed=42)")
    print("=" * 84)
    print(f"  {'Model':<20} {'Dataset':<16} {'Mean':>10} {'Std':>10} {'Per-fold scores'}")
    print(f"  {'─' * 80}")

    results = {}
    for model_name, dataset_name in BASELINES:
        X, y = loaded[dataset_name]
        scale = model_name in NEEDS_SCALING
        model_fn = CONFIGS[model_name]

        scores = cross_val_score_stratified(model_fn, X, y, scale=scale)
        mean = np.mean(scores)
        std = np.std(scores)
        results[(model_name, dataset_name)] = mean

        fold_str = " ".join(f"{s:.4f}" for s in scores)
        print(f"  {model_name:<20} {dataset_name:<16} {mean:>10.4f} {std:>10.4f} [{fold_str}]")

    # ═══════════════════════════════════════════════════════════════════
    # Side-by-side comparison with scry's known outputs
    # ═══════════════════════════════════════════════════════════════════

    # scry-learn scores from golden test (deterministic, seed=42)
    scry_scores = {
        ("DecisionTree",     "iris"):          0.9533,
        ("RandomForest",     "iris"):          0.9533,
        ("GradientBoosting", "iris"):          0.9533,
        ("HistGBT",          "iris"):          0.9333,
        ("GaussianNB",       "iris"):          0.9600,
        ("KNN",              "iris"):          0.9533,
        ("DecisionTree",     "wine"):          0.9037,
        ("RandomForest",     "wine"):          0.9495,
        ("GradientBoosting", "wine"):          0.9552,
        ("HistGBT",          "wine"):          0.9776,
        ("GaussianNB",       "wine"):          0.9776,
        ("DecisionTree",     "breast_cancer"): 0.9526,
        ("RandomForest",     "breast_cancer"): 0.9596,
        ("GradientBoosting", "breast_cancer"): 0.9474,
        ("HistGBT",          "breast_cancer"): 0.9667,
        ("GaussianNB",       "breast_cancer"): 0.9333,
        ("DecisionTree",     "digits"):        0.8559,
        ("RandomForest",     "digits"):        0.9466,
        ("GradientBoosting", "digits"):        0.9566,
        ("HistGBT",          "digits"):        0.9716,
        ("GaussianNB",       "digits"):        0.8392,
        ("KNN",              "digits"):        0.9755,
    }

    print()
    print("=" * 84)
    print("  SIDE-BY-SIDE COMPARISON: scry-learn vs scikit-learn")
    print("=" * 84)
    print(f"  {'Model':<20} {'Dataset':<16} {'sklearn':>10} {'scry':>10} {'Δ (scry-sk)':>12}  {'Note'}")
    print(f"  {'─' * 80}")

    for model_name, dataset_name in BASELINES:
        sk = results[(model_name, dataset_name)]
        scry = scry_scores.get((model_name, dataset_name))
        if scry is None:
            continue
        delta = scry - sk
        note = ""
        if abs(delta) < 0.005:
            note = "~match"
        elif abs(delta) < 0.02:
            note = "minor"
        elif delta > 0:
            note = "scry higher"
        else:
            note = "sklearn higher"
        print(
            f"  {model_name:<20} {dataset_name:<16} {sk:>10.4f} {scry:>10.4f} {delta:>+12.4f}  {note}"
        )

    print()
    print("  Notes:")
    print("  - Both use 5-fold StratifiedKFold, seed=42")
    print("  - Differences come from: RNG (fastrand vs numpy), tie-breaking,")
    print("    ceil vs floor for max_features, bootstrap sampling, init predictions")
    print("  - These are implementation differences, not mathematical errors")
    print()


if __name__ == "__main__":
    main()
