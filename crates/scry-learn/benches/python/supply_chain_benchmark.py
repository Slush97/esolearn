#!/usr/bin/env python3
"""
Supply Chain ML Benchmark — Competitor Library Evaluation
==========================================================

Benchmarks industry-grade ML libraries on the REAL DataCo SMART
Supply Chain dataset (Kaggle) — ~180k orders, 53 features.

Task: Binary classification — predict `Late_delivery_risk` (0/1).

Libraries benchmarked:
  1. scikit-learn  (RF, GBT, HistGBT, LogReg)
  2. XGBoost       (XGBClassifier)
  3. LightGBM      (LGBMClassifier)
  4. FLAML         (AutoML — auto-selects learner + hyperparams)

Libraries documented but unavailable (no Python 3.14 wheels):
  - CatBoost, H2O AutoML

Usage:
  .venv/bin/python3 supply_chain_benchmark.py
"""

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer

warnings.filterwarnings("ignore")

SEED = 42
DATA_PATH = Path(__file__).parent / "data" / "DataCoSupplyChainDataset.csv"

# ─────────────────────────────────────────────────────────────────
# 1. Load & Clean Real DataCo SMART Dataset
# ─────────────────────────────────────────────────────────────────

# Features selected for ML (avoiding leakage — no "Delivery Status",
# no "Days for shipping (real)" since that reveals the outcome)
NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Price",
    "Latitude",
    "Longitude",
]

CATEGORICAL_FEATURES = [
    "Type",              # Payment type: DEBIT, TRANSFER, CASH, PAYMENT
    "Category Name",     # Product category
    "Customer Segment",  # Consumer, Corporate, Home Office
    "Market",            # Region market
    "Order Region",      # Geographic region
    "Order Status",      # COMPLETE, PENDING, etc.
    "Shipping Mode",     # Standard, First Class, etc.
]

TARGET = "Late_delivery_risk"


def load_dataco() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and clean the real DataCo SMART dataset."""
    print("📦 Loading DataCo SMART Supply Chain dataset...")
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    print(f"   Raw shape: {df.shape}")

    # Select features + target
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available = [f for f in all_features if f in df.columns]
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"   ⚠️  Missing columns (skipped): {missing}")

    sub = df[available + [TARGET]].copy()
    sub = sub.dropna()
    print(f"   Clean shape: {sub.shape}")

    y = sub[TARGET].values
    print(f"   Target: {TARGET}  (1={y.sum():,} late, 0={len(y)-y.sum():,} on-time, "
          f"late_rate={y.mean():.1%})")

    # Encode categoricals as integers
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in sub.columns]
    for col in cat_cols:
        le = LabelEncoder()
        sub[col] = le.fit_transform(sub[col].astype(str))

    X = sub[available].values
    feature_names = available
    print(f"   Features: {len(feature_names)}")

    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────
# 2. Model Definitions — Default & Tuned
# ─────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    library: str
    model_name: str
    config: str
    accuracy: float
    f1: float
    roc_auc: float
    train_time_s: float
    best_params: dict


def get_sklearn_models():
    return {
        "sklearn_RF_default": {
            "model": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
            "config": "default", "library": "scikit-learn",
            "params": {"n_estimators": 100, "max_depth": None},
        },
        "sklearn_GBT_default": {
            "model": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
            "config": "default", "library": "scikit-learn",
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        },
        "sklearn_HistGBT_default": {
            "model": HistGradientBoostingClassifier(max_iter=100, random_state=SEED),
            "config": "default", "library": "scikit-learn",
            "params": {"max_iter": 100, "learning_rate": 0.1},
        },
        "sklearn_LR_default": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]),
            "config": "default", "library": "scikit-learn",
            "params": {"C": 1.0, "max_iter": 1000},
        },

        # ── Tuned ──
        "sklearn_RF_tuned": {
            "model": RandomForestClassifier(
                n_estimators=300, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=SEED, n_jobs=-1
            ),
            "config": "tuned", "library": "scikit-learn",
            "params": {"n_estimators": 300, "max_depth": 20,
                       "min_samples_split": 5, "min_samples_leaf": 2},
        },
        "sklearn_GBT_tuned": {
            "model": GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=SEED
            ),
            "config": "tuned", "library": "scikit-learn",
            "params": {"n_estimators": 200, "max_depth": 5,
                       "learning_rate": 0.05, "subsample": 0.8},
        },
        "sklearn_HistGBT_tuned": {
            "model": HistGradientBoostingClassifier(
                max_iter=300, max_depth=8, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=0.1, random_state=SEED
            ),
            "config": "tuned", "library": "scikit-learn",
            "params": {"max_iter": 300, "max_depth": 8,
                       "learning_rate": 0.05, "l2_regularization": 0.1},
        },
    }


def get_xgboost_models():
    from xgboost import XGBClassifier
    return {
        "xgboost_default": {
            "model": XGBClassifier(
                n_estimators=100, eval_metric="logloss",
                verbosity=0, random_state=SEED, n_jobs=-1
            ),
            "config": "default", "library": "XGBoost",
            "params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3},
        },
        "xgboost_tuned": {
            "model": XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
                min_child_weight=3, eval_metric="logloss",
                verbosity=0, random_state=SEED, n_jobs=-1
            ),
            "config": "tuned", "library": "XGBoost",
            "params": {"n_estimators": 300, "max_depth": 8,
                       "learning_rate": 0.05, "subsample": 0.8,
                       "colsample_bytree": 0.8, "reg_alpha": 0.1,
                       "reg_lambda": 1.0, "gamma": 0.1, "min_child_weight": 3},
        },
    }


def get_lightgbm_models():
    from lightgbm import LGBMClassifier
    return {
        "lightgbm_default": {
            "model": LGBMClassifier(n_estimators=100, random_state=SEED, verbose=-1, n_jobs=-1),
            "config": "default", "library": "LightGBM",
            "params": {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1},
        },
        "lightgbm_tuned": {
            "model": LGBMClassifier(
                n_estimators=300, num_leaves=63, max_depth=10,
                learning_rate=0.05, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1
            ),
            "config": "tuned", "library": "LightGBM",
            "params": {"n_estimators": 300, "num_leaves": 63,
                       "max_depth": 10, "learning_rate": 0.05,
                       "min_child_samples": 20, "subsample": 0.8,
                       "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
        },
    }


# ─────────────────────────────────────────────────────────────────
# 3. Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_model(name, spec, X, y, cv) -> ModelResult:
    model = spec["model"]
    library = spec["library"]
    config = spec["config"]
    params = spec["params"]
    fit_kwargs = spec.get("fit_kwargs", {})

    print(f"  {name:<35s}", end="", flush=True)
    start = time.perf_counter()

    if library == "FLAML":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        model.fit(X_train, y_train, **fit_kwargs)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        elapsed = time.perf_counter() - start
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        if hasattr(model, "best_config"):
            params = {**params, **model.best_config}
        if hasattr(model, "best_estimator"):
            params["best_learner"] = type(model.best_estimator).__name__
    else:
        scoring = {
            "accuracy": "accuracy",
            "f1": make_scorer(f1_score, average="binary"),
            "roc_auc": "roc_auc",
        }
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.perf_counter() - start
        acc = cv_results["test_accuracy"].mean()
        f1 = cv_results["test_f1"].mean()
        auc = cv_results["test_roc_auc"].mean()

    print(f"  acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  ({elapsed:.1f}s)")
    return ModelResult(library, name, config, acc, f1, auc, elapsed, params)


# ─────────────────────────────────────────────────────────────────
# 4. Hyperparameter Reference
# ─────────────────────────────────────────────────────────────────

HYPERPARAM_REFERENCE = {
    "scikit-learn RandomForest": {
        "n_estimators": "Number of trees (50-500, default 100)",
        "max_depth": "Max tree depth (None=unlimited, try 5-30)",
        "min_samples_split": "Min samples to split a node (2-20)",
        "min_samples_leaf": "Min samples in a leaf (1-10)",
        "max_features": "Features per split ('sqrt', 'log2', float)",
        "class_weight": "Handle imbalance ('balanced' or dict)",
    },
    "scikit-learn GradientBoosting": {
        "n_estimators": "Boosting rounds (100-500)",
        "learning_rate": "Shrinkage (0.01-0.3, default 0.1)",
        "max_depth": "Tree depth (3-10, default 3)",
        "subsample": "Row sampling per tree (0.5-1.0)",
    },
    "scikit-learn HistGradientBoosting": {
        "max_iter": "Boosting iterations (100-500)",
        "learning_rate": "Shrinkage (0.01-0.3)",
        "max_depth": "Tree depth (None or 3-15)",
        "min_samples_leaf": "Min samples per leaf (5-50)",
        "l2_regularization": "L2 penalty (0.0-10.0)",
    },
    "XGBoost": {
        "n_estimators": "Boosting rounds (100-1000)",
        "max_depth": "Tree depth (3-12, default 6)",
        "learning_rate": "eta / shrinkage (0.01-0.3)",
        "subsample": "Row sampling (0.5-1.0)",
        "colsample_bytree": "Column sampling per tree (0.3-1.0)",
        "reg_alpha": "L1 regularization (0-10)",
        "reg_lambda": "L2 regularization (0-10, default 1)",
        "gamma": "Min loss reduction for split (0-5)",
        "min_child_weight": "Min instance weight in leaf (1-10)",
    },
    "LightGBM": {
        "n_estimators": "Boosting rounds (100-1000)",
        "num_leaves": "Max leaves per tree (15-127, default 31)",
        "max_depth": "Depth limit (-1=no limit, 3-15)",
        "learning_rate": "Shrinkage (0.01-0.3)",
        "min_child_samples": "Min data in a leaf (5-100)",
        "subsample": "Bagging fraction (0.5-1.0)",
        "colsample_bytree": "Feature fraction (0.3-1.0)",
        "reg_alpha": "L1 regularization (0-10)",
        "reg_lambda": "L2 regularization (0-10)",
    },
    "FLAML": {
        "time_budget": "Total tuning time in seconds (30-600)",
        "metric": "Optimization target ('accuracy', 'f1', 'roc_auc')",
        "estimator_list": "Learners to try (['lgbm', 'xgboost', 'rf'])",
        "NOTE": "Auto-tunes ALL hyperparameters of selected learner",
    },
    "CatBoost (no py3.14 wheel)": {
        "iterations": "Boosting rounds (100-1000, default 1000)",
        "depth": "Tree depth (4-10, default 6)",
        "learning_rate": "Shrinkage (auto if unset)",
        "l2_leaf_reg": "L2 regularization (1-10, default 3)",
        "border_count": "Histogram bins (32-255)",
        "bagging_temperature": "Bayesian bootstrap (0-10)",
        "NOTE": "Native categorical support — no encoding needed",
    },
    "H2O AutoML (no py3.14 wheel)": {
        "max_runtime_secs": "Time budget (60-3600)",
        "max_models": "Max models to train (10-50)",
        "sort_metric": "'AUC', 'logloss', 'accuracy'",
        "NOTE": "Trains + stacks GBMs, RFs, GLMs, XGBoost, DL automatically",
    },
}


# ─────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────

def banner(text: str):
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}")


def main():
    banner("Supply Chain ML Benchmark — DataCo SMART (Real Data)")

    X, y, feature_names = load_dataco()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Collect all models
    all_models = {}
    all_models.update(get_sklearn_models())
    all_models.update(get_xgboost_models())
    all_models.update(get_lightgbm_models())

    # FLAML
    try:
        from flaml import AutoML
        flaml_model = AutoML()
        all_models["flaml_automl_120s"] = {
            "model": flaml_model,
            "config": "automl_120s", "library": "FLAML",
            "params": {"time_budget": 120, "metric": "accuracy"},
            "fit_kwargs": {
                "task": "classification", "time_budget": 120,
                "metric": "accuracy", "seed": SEED, "verbose": 0,
            },
        }
    except ImportError:
        print("  ⚠️  FLAML not installed — skipping")

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        all_models["catboost_tuned"] = {
            "model": CatBoostClassifier(
                iterations=300, depth=8, learning_rate=0.05,
                l2_leaf_reg=5, verbose=0, random_state=SEED
            ),
            "config": "tuned", "library": "CatBoost",
            "params": {"iterations": 300, "depth": 8, "learning_rate": 0.05},
        }
    except ImportError:
        print("  ⚠️  CatBoost not available (no py3.14 wheel)")

    # H2O
    try:
        import h2o
    except ImportError:
        print("  ⚠️  H2O not available (no py3.14 wheel)")

    # Run
    banner("Training Models (5-fold Stratified CV, ~180k samples)")
    results: list[ModelResult] = []
    for name, spec in all_models.items():
        try:
            results.append(evaluate_model(name, spec, X, y, cv))
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    results.sort(key=lambda r: r.roc_auc, reverse=True)

    # ── Ranked Results ──
    banner("Results — Ranked by ROC-AUC")
    print(f"{'#':<4} {'Library':<14} {'Model':<36} {'Cfg':<10} "
          f"{'Acc':<8} {'F1':<8} {'AUC':<8} {'Time':<7}")
    print("-" * 95)
    for i, r in enumerate(results, 1):
        print(f"{i:<4} {r.library:<14} {r.model_name:<36} {r.config:<10} "
              f"{r.accuracy:<8.4f} {r.f1:<8.4f} {r.roc_auc:<8.4f} {r.train_time_s:<7.1f}")

    # ── Best per library ──
    banner("Best Model Per Library")
    seen = set()
    for r in results:
        if r.library not in seen:
            seen.add(r.library)
            print(f"  {r.library:<14} → {r.model_name:<36} "
                  f"acc={r.accuracy:.4f}  f1={r.f1:.4f}  auc={r.roc_auc:.4f}")

    # ── Tuning impact ──
    banner("Tuning Impact — Default → Tuned")
    defaults = {}
    tuned = {}
    for r in results:
        key = r.model_name.rsplit("_", 1)[0]
        if r.config == "default":
            defaults[key] = r
        elif r.config == "tuned":
            tuned[key] = r

    for key in sorted(defaults.keys() & tuned.keys()):
        d, t = defaults[key], tuned[key]
        print(f"  {key:<32} Δacc={t.accuracy-d.accuracy:+.4f}  "
              f"Δf1={t.f1-d.f1:+.4f}  Δauc={t.roc_auc-d.roc_auc:+.4f}")

    # ── Hyperparameter reference ──
    banner("Hyperparameter Reference — All Libraries")
    for lib, params in HYPERPARAM_REFERENCE.items():
        print(f"\n  📋 {lib}")
        for p, desc in params.items():
            print(f"     {p:<25} {desc}")

    # ── Ergonomics ──
    banner("Library Ergonomics & Recommendations")
    for lib, stars, note in [
        ("scikit-learn", "⭐⭐⭐⭐⭐",
         "Gold-standard API. Best docs. Pipelines + GridSearchCV. "
         "HistGBT competitive with XGBoost/LightGBM."),
        ("XGBoost", "⭐⭐⭐⭐",
         "Industry workhorse. Many knobs. reg_alpha/reg_lambda crucial. "
         "GPU support via tree_method='gpu_hist'."),
        ("LightGBM", "⭐⭐⭐⭐",
         "Fastest on large data. num_leaves is THE knob (not max_depth). "
         "Native categoricals. Leaf-wise → easy to overfit."),
        ("FLAML", "⭐⭐⭐⭐⭐",
         "Zero-config AutoML. Set time_budget + metric, done. "
         "Auto-selects learner (often LightGBM). Microsoft-backed."),
        ("CatBoost", "⭐⭐⭐⭐",
         "Native categorical handling. Ordered boosting reduces overfitting. "
         "Often best accuracy. No py3.14 wheel yet."),
        ("H2O AutoML", "⭐⭐⭐⭐",
         "Enterprise-grade. Trains + stacks multiple families. "
         "Requires JVM. No py3.14 support yet."),
    ]:
        print(f"\n  {lib} — {stars}\n    {note}")

    # ── Save ──
    output = {
        "dataset": {
            "name": "DataCo SMART Supply Chain (Kaggle — real data)",
            "source": "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis",
            "samples": int(len(y)),
            "features": len(feature_names),
            "feature_names": feature_names,
            "target": TARGET,
            "late_rate": float(y.mean()),
        },
        "results": [asdict(r) for r in results],
        "hyperparameter_reference": HYPERPARAM_REFERENCE,
        "libraries_tested": list({r.library for r in results}),
        "libraries_unavailable": {
            "CatBoost": "No wheel for Python 3.14",
            "H2O": "No wheel for Python 3.14",
        },
    }
    out_path = Path(__file__).parent / "supply_chain_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
