#!/usr/bin/env python3
"""Generate golden-reference fixtures for scry-learn from sklearn.

Produces CSV datasets and JSON sklearn predictions for:
  15 UCI/OpenML datasets + California Housing regression.

Usage:
    pip install scikit-learn numpy pandas
    python generate_fixtures.py
"""
import json
import csv
import os
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score as cv
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
import pandas as pd

SEED = 42
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_csv(name, data, header=None):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for row in data:
            if hasattr(row, "__iter__") and not isinstance(row, str):
                w.writerow([f"{v:.10g}" for v in row])
            else:
                w.writerow([f"{row:.10g}"])
    print(f"  Wrote {path} ({len(data)} rows)")

def load_and_save_classification(loader_fn, name):
    ds = loader_fn()
    X, y = ds.data, ds.target
    feat_names = [f"f{i}" for i in range(X.shape[1])]
    save_csv(f"{name}_features.csv", X, header=feat_names)
    save_csv(f"{name}_target.csv", y.reshape(-1, 1), header=["target"])
    return X, y

def load_and_save_regression(loader_fn, name):
    ds = loader_fn()
    X, y = ds.data, ds.target
    feat_names = [f"f{i}" for i in range(X.shape[1])]
    save_csv(f"{name}_features.csv", X, header=feat_names)
    save_csv(f"{name}_target.csv", y.reshape(-1, 1), header=["target"])
    return X, y

def load_openml_numeric(openml_name, local_name, version=1):
    """Load an OpenML dataset, keep only numeric columns, label-encode target."""
    print(f"Loading {openml_name} from OpenML...")
    bunch = fetch_openml(name=openml_name, version=version, as_frame=True, parser="auto")
    df = bunch.data
    target_raw = bunch.target

    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].values.astype(np.float64)

    # Drop rows with NaNs
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    target_raw = target_raw[mask]

    # Label-encode target to integers
    le = LabelEncoder()
    y = le.fit_transform(target_raw).astype(np.float64)

    feat_names = [f"f{i}" for i in range(X.shape[1])]
    save_csv(f"{local_name}_features.csv", X, header=feat_names)
    save_csv(f"{local_name}_target.csv", y.reshape(-1, 1), header=["target"])
    print(f"  {local_name}: {X.shape[0]} samples, {X.shape[1]} numeric features, {len(np.unique(y))} classes")
    return X, y

def compute_cv(results, key, estimator, X, y):
    """Compute stratified 5-fold CV and store in results dict."""
    scores = cv(estimator, X, y, cv=5, scoring="accuracy")
    results[key] = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "folds": scores.tolist(),
    }
    print(f"  {key}: {scores.mean():.3f} ± {scores.std():.3f}")

def main():
    results = {}

    # ═══════════════════════════════════════════════════════
    # Original 5 datasets (sklearn built-in)
    # ═══════════════════════════════════════════════════════

    # ── Iris (150×4, 3-class) ──
    print("Loading Iris...")
    X_iris, y_iris = load_and_save_classification(datasets.load_iris, "iris")

    # ── Wine (178×13, 3-class) ──
    print("Loading Wine...")
    X_wine, y_wine = load_and_save_classification(datasets.load_wine, "wine")

    # ── Breast Cancer (569×30, binary) ──
    print("Loading Breast Cancer...")
    X_bc, y_bc = load_and_save_classification(
        datasets.load_breast_cancer, "breast_cancer"
    )

    # ── Digits (1797×64, 10-class) ──
    print("Loading Digits...")
    X_dig, y_dig = load_and_save_classification(datasets.load_digits, "digits")

    # ── California Housing (20640×8, regression) ──
    print("Loading California Housing...")
    X_cal, y_cal = load_and_save_regression(
        datasets.fetch_california_housing, "california"
    )

    # ═══════════════════════════════════════════════════════
    # 10 new OpenML datasets (numeric-only)
    # ═══════════════════════════════════════════════════════

    X_adult, y_adult = load_openml_numeric("adult", "adult", version=2)
    X_spam, y_spam = load_openml_numeric("spambase", "spambase", version=1)
    X_wineq, y_wineq = load_openml_numeric("wine-quality-red", "wine_quality", version=1)
    X_glass, y_glass = load_openml_numeric("glass", "glass", version=1)
    X_iono, y_iono = load_openml_numeric("ionosphere", "ionosphere", version=1)
    X_vehicle, y_vehicle = load_openml_numeric("vehicle", "vehicle", version=1)
    X_segment, y_segment = load_openml_numeric("segment", "segment", version=1)
    X_sonar, y_sonar = load_openml_numeric("sonar", "sonar", version=1)
    X_haberman, y_haberman = load_openml_numeric("haberman", "haberman", version=1)
    X_ecoli, y_ecoli = load_openml_numeric("ecoli", "ecoli", version=1)

    # ═══════════════════════════════════════════════════════
    # Train sklearn models and record predictions (original)
    # ═══════════════════════════════════════════════════════

    # --- LogisticRegression on Iris ---
    print("Training LogisticRegression on Iris...")
    scaler_iris = StandardScaler().fit(X_iris)
    X_iris_s = scaler_iris.transform(X_iris)
    logreg = LogisticRegression(random_state=SEED, max_iter=200, solver="lbfgs")
    logreg.fit(X_iris_s, y_iris)
    lr_preds = logreg.predict(X_iris_s).tolist()
    lr_acc = float(np.mean(logreg.predict(X_iris_s) == y_iris))
    results["logreg_iris"] = {
        "predictions": lr_preds,
        "accuracy": lr_acc,
        "params": {"max_iter": 200, "solver": "lbfgs", "random_state": SEED},
    }

    # --- DecisionTreeClassifier on Iris ---
    print("Training DecisionTree on Iris...")
    dt = DecisionTreeClassifier(random_state=SEED, max_depth=5)
    dt.fit(X_iris, y_iris)
    dt_preds = dt.predict(X_iris).tolist()
    dt_acc = float(np.mean(dt.predict(X_iris) == y_iris))
    results["dt_iris"] = {
        "predictions": dt_preds,
        "accuracy": dt_acc,
        "params": {"max_depth": 5, "random_state": SEED},
    }

    # --- DecisionTreeClassifier on Wine ---
    print("Training DecisionTree on Wine...")
    dt_wine = DecisionTreeClassifier(random_state=SEED, max_depth=5)
    dt_wine.fit(X_wine, y_wine)
    results["dt_wine"] = {
        "predictions": dt_wine.predict(X_wine).tolist(),
        "accuracy": float(np.mean(dt_wine.predict(X_wine) == y_wine)),
        "params": {"max_depth": 5, "random_state": SEED},
    }

    # --- DecisionTreeClassifier on Breast Cancer ---
    print("Training DecisionTree on Breast Cancer...")
    dt_bc = DecisionTreeClassifier(random_state=SEED, max_depth=5)
    dt_bc.fit(X_bc, y_bc)
    results["dt_breast_cancer"] = {
        "predictions": dt_bc.predict(X_bc).tolist(),
        "accuracy": float(np.mean(dt_bc.predict(X_bc) == y_bc)),
        "params": {"max_depth": 5, "random_state": SEED},
    }

    # --- KNN on Iris ---
    print("Training KNN on Iris...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_iris, y_iris)
    knn_preds = knn.predict(X_iris).tolist()
    knn_acc = float(np.mean(knn.predict(X_iris) == y_iris))
    results["knn_iris"] = {
        "predictions": knn_preds,
        "accuracy": knn_acc,
        "params": {"n_neighbors": 5},
    }

    # --- KNN on Wine ---
    print("Training KNN on Wine...")
    knn_wine = KNeighborsClassifier(n_neighbors=5)
    knn_wine.fit(X_wine, y_wine)
    results["knn_wine"] = {
        "predictions": knn_wine.predict(X_wine).tolist(),
        "accuracy": float(np.mean(knn_wine.predict(X_wine) == y_wine)),
        "params": {"n_neighbors": 5},
    }

    # --- KMeans on Iris ---
    print("Training KMeans on Iris...")
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10, max_iter=300)
    km.fit(X_iris)
    km_labels = km.labels_.tolist()
    km_inertia = float(km.inertia_)
    results["kmeans_iris"] = {
        "labels": km_labels,
        "inertia": km_inertia,
        "params": {"n_clusters": 3, "random_state": SEED, "n_init": 10},
    }

    # --- LinearRegression on California ---
    print("Training LinearRegression on California Housing...")
    scaler_cal = StandardScaler().fit(X_cal)
    X_cal_s = scaler_cal.transform(X_cal)
    lr_reg = LinearRegression()
    lr_reg.fit(X_cal_s, y_cal)
    lr_reg_preds = lr_reg.predict(X_cal_s).tolist()
    lr_reg_r2 = float(lr_reg.score(X_cal_s, y_cal))
    results["linreg_california"] = {
        "predictions": lr_reg_preds,
        "r2_score": lr_reg_r2,
        "params": {},
    }

    # --- StandardScaler on Iris ---
    print("Computing StandardScaler on Iris...")
    scaler = StandardScaler().fit(X_iris)
    transformed = scaler.transform(X_iris)
    results["scaler_iris"] = {
        "means": scaler.mean_.tolist(),
        "stds": scaler.scale_.tolist(),
        "first_5_transformed": transformed[:5].tolist(),
    }

    # --- PCA on Iris (2 components) ---
    print("Computing PCA on Iris...")
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_iris)
    results["pca_iris"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "first_5_transformed": X_pca[:5].tolist(),
        "params": {"n_components": 2},
    }

    # ═══════════════════════════════════════════════════════
    # 5-fold CV accuracies for parity table
    # ═══════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("Computing 5-fold CV accuracies for parity table...")
    print("=" * 60)

    # ── Original 5 datasets ──

    print("\n--- Iris ---")
    pipe_lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200, random_state=SEED))])
    cv_lr_iris = cv(pipe_lr, X_iris, y_iris, cv=5, scoring="accuracy")
    results["cv_logreg_iris"] = {"mean": float(cv_lr_iris.mean()), "std": float(cv_lr_iris.std()), "folds": cv_lr_iris.tolist()}

    compute_cv(results, "cv_dt_iris", DecisionTreeClassifier(max_depth=5, random_state=SEED), X_iris, y_iris)
    compute_cv(results, "cv_knn_iris", KNeighborsClassifier(n_neighbors=5), X_iris, y_iris)

    print("\n--- Wine ---")
    compute_cv(results, "cv_dt_wine", DecisionTreeClassifier(max_depth=5, random_state=SEED), X_wine, y_wine)
    compute_cv(results, "cv_knn_wine", KNeighborsClassifier(n_neighbors=5), X_wine, y_wine)

    print("\n--- Breast Cancer ---")
    compute_cv(results, "cv_dt_bc", DecisionTreeClassifier(max_depth=5, random_state=SEED), X_bc, y_bc)
    pipe_lr_bc = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200, random_state=SEED))])
    cv_lr_bc = cv(pipe_lr_bc, X_bc, y_bc, cv=5)
    results["cv_logreg_bc"] = {"mean": float(cv_lr_bc.mean()), "std": float(cv_lr_bc.std()), "folds": cv_lr_bc.tolist()}

    print("\n--- Digits ---")
    compute_cv(results, "cv_dt_digits", DecisionTreeClassifier(max_depth=15, random_state=SEED), X_dig, y_dig)
    compute_cv(results, "cv_knn_digits", KNeighborsClassifier(n_neighbors=5), X_dig, y_dig)

    print("\n--- California Housing (R²) ---")
    pipe_cal = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    cv_lr_cal = cv(pipe_cal, X_cal, y_cal, cv=5, scoring="r2")
    results["cv_linreg_california"] = {"mean": float(cv_lr_cal.mean()), "std": float(cv_lr_cal.std()), "folds": cv_lr_cal.tolist()}

    # ── 10 new datasets — DT + KNN CV for each ──

    new_datasets = [
        ("adult",       X_adult,   y_adult),
        ("spambase",    X_spam,    y_spam),
        ("wine_quality", X_wineq,  y_wineq),
        ("glass",       X_glass,   y_glass),
        ("ionosphere",  X_iono,    y_iono),
        ("vehicle",     X_vehicle, y_vehicle),
        ("segment",     X_segment, y_segment),
        ("sonar",       X_sonar,   y_sonar),
        ("haberman",    X_haberman, y_haberman),
        ("ecoli",       X_ecoli,   y_ecoli),
    ]

    for name, X, y in new_datasets:
        print(f"\n--- {name} ---")
        compute_cv(results, f"cv_dt_{name}",
                   DecisionTreeClassifier(max_depth=5, random_state=SEED), X, y)
        compute_cv(results, f"cv_knn_{name}",
                   KNeighborsClassifier(n_neighbors=5), X, y)

    # Save all results as JSON
    out_path = os.path.join(OUT_DIR, "sklearn_predictions.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Wrote {out_path}")
    print(f"   Keys ({len(results.keys())}): {list(results.keys())}")

if __name__ == "__main__":
    main()
