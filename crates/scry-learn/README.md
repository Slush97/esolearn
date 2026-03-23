# scry-learn

[![Crates.io](https://img.shields.io/crates/v/scry-learn.svg)](https://crates.io/crates/scry-learn)
[![docs.rs](https://img.shields.io/docsrs/scry-learn)](https://docs.rs/scry-learn)
[![License](https://img.shields.io/crates/l/scry-learn.svg)](https://github.com/Slush97/esolearn/blob/main/LICENSE-MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/Slush97/esolearn/ci.yml?label=tests)](https://github.com/Slush97/esolearn)

**Production-grade machine learning in pure Rust.**

scry-learn is a scikit-learn-inspired ML toolkit with 23+ models, built-in explainability (TreeSHAP), and optional GPU acceleration — no Python, no BLAS, no LAPACK. Single binary, `cargo add`, done.

```rust
use scry_learn::prelude::*;

let data = Dataset::from_csv("iris.csv", "species")?;
let (train, test) = train_test_split(&data, 0.2, 42);

let mut rf = RandomForestClassifier::new()
    .n_estimators(100)
    .max_depth(10);
rf.fit(&train)?;

let preds = rf.predict(&test)?;
println!("{}", classification_report(&test.target, &preds));
```

---

## Why scry-learn?

- **No native dependencies.** Pure Rust — no Python runtime, no BLAS/LAPACK, no system libraries. `cargo build` just works.
- **Explainability built in.** TreeSHAP and permutation importance ship with the crate, not as a separate package.
- **Honest benchmarks.** We benchmark against linfa and smartcore with counting allocators, single-thread enforcement, and accuracy parity gates. [Run them yourself.](#benchmarks)
- **Column-major layout.** Data is stored column-first internally, giving tree-based models a genuine algorithmic advantage on feature scans.
- **`deny(unsafe_code)`.** The entire crate compiles with zero `unsafe` blocks.

---

## Algorithms

<details>
<summary><strong>Classification & Regression</strong></summary>

| Model | Classification | Regression |
|-------|:-:|:-:|
| Decision Tree (CART) | ✓ | ✓ |
| Random Forest | ✓ | ✓ |
| Gradient Boosting | ✓ | ✓ |
| Histogram Gradient Boosting | ✓ | ✓ |
| Linear / Logistic Regression | ✓ | ✓ |
| Ridge | — | ✓ |
| Lasso | — | ✓ |
| ElasticNet | — | ✓ |
| Linear SVM | ✓ | ✓ |
| Kernel SVM | ✓* | ✓* |
| K-Nearest Neighbors | ✓ | ✓ |
| Gaussian Naive Bayes | ✓ | — |
| Multinomial Naive Bayes | ✓ | — |
| Bernoulli Naive Bayes | ✓ | — |
| MLP Neural Network | ✓ | ✓ |

*\* Kernel SVM requires `features = ["experimental"]`*

</details>

<details>
<summary><strong>Clustering</strong></summary>

| Algorithm | Notes |
|-----------|-------|
| K-Means | k-means++ init, configurable max_iter |
| Mini-Batch K-Means | Streaming-friendly variant |
| DBSCAN | Density-based, automatic cluster count |
| HDBSCAN | Hierarchical density-based |
| Agglomerative | Ward / complete / average / single linkage |

</details>

<details>
<summary><strong>Preprocessing & Feature Engineering</strong></summary>

- **Scaling:** StandardScaler, MinMaxScaler, RobustScaler, Normalizer (L1/L2)
- **Encoding:** OneHotEncoder, LabelEncoder
- **Imputation:** SimpleImputer (mean, median, most-frequent, constant)
- **Dimensionality:** PCA, VarianceThreshold, SelectKBest (f_classif)
- **Transforms:** PolynomialFeatures, ColumnTransformer, Pipeline

</details>

<details>
<summary><strong>Model Selection & Metrics</strong></summary>

- **Search:** GridSearchCV, RandomizedSearchCV, BayesSearchCV
- **Validation:** cross_val_score, stratified k-fold, group k-fold, time series split, repeated CV
- **Classification metrics:** accuracy, precision, recall, F1, balanced accuracy, Cohen's kappa, confusion matrix, ROC AUC, PR curve, log loss
- **Regression metrics:** MSE, MAPE, R², explained variance
- **Clustering metrics:** silhouette score, Calinski-Harabasz, Davies-Bouldin, adjusted Rand index
- **Calibration:** Platt scaling, isotonic regression

</details>

<details>
<summary><strong>Explainability</strong></summary>

- **TreeSHAP** — exact Shapley values for tree ensembles in polynomial time (Lundberg & Lee, 2018)
- **Permutation importance** — model-agnostic feature importance with configurable repeats (Breiman, 2001)

```rust
use scry_learn::prelude::*;

let shap_values = ensemble_tree_shap(&rf, &test.features);
let importance = permutation_importance(&rf, &test, accuracy, 5);
```

</details>

<details>
<summary><strong>Text / NLP</strong></summary>

- **CountVectorizer** — n-gram term counts, min/max document frequency, sparse CSR output
- **TfidfVectorizer** — TF-IDF with L1/L2 normalization, sublinear TF, smooth IDF
- **Tokenizer** — zero-dependency whitespace/punctuation-aware tokenizer

</details>

<details>
<summary><strong>Anomaly Detection</strong></summary>

- **Isolation Forest** — unsupervised anomaly detection via random partitioning

</details>

---

## sklearn → scry-learn

If you know scikit-learn, you already know scry-learn.

| scikit-learn (Python) | scry-learn (Rust) |
|---|---|
| `from sklearn.ensemble import RandomForestClassifier` | `use scry_learn::prelude::*;` |
| `rf = RandomForestClassifier(n_estimators=100)` | `let mut rf = RandomForestClassifier::new().n_estimators(100);` |
| `rf.fit(X_train, y_train)` | `rf.fit(&train)?;` |
| `rf.predict(X_test)` | `rf.predict(&test)?` |
| `cross_val_score(rf, X, y, cv=5)` | `cross_val_score(&rf, &data, 5, accuracy)` |
| `GridSearchCV(rf, param_grid, cv=5)` | `GridSearchCV::new(rf, param_grid, 5, accuracy)` |
| `shap.TreeExplainer(rf).shap_values(X)` | `ensemble_tree_shap(&rf, &features)` |
| `StandardScaler().fit_transform(X)` | `StandardScaler::new().fit_transform(&mut data)?` |

---

## Benchmarks

scry-learn ships with rigorous cross-library benchmarks against [linfa](https://github.com/rust-ml/linfa) and [smartcore](https://github.com/smartcorelib/smartcore). Our benchmarking infrastructure enforces:

- **Real UCI datasets only** — no synthetic data with RNG bias
- **Counting allocator** — actual heap bytes, not RSS estimates
- **Single-thread enforcement** — asserted programmatically, not assumed via env var
- **Accuracy parity gates** — timing only reported when all libraries converge within ε=3%
- **Identical preprocessing** — matched standardization across libraries

Run them yourself:

```bash
cargo bench --bench fair_bench -p scry-learn
cargo bench --bench honest_bench -p scry-learn

# Extended scaling curves (500 / 2K / 10K samples)
cargo bench --bench fair_bench -p scry-learn --features extended-bench
```

---

## Install

```toml
[dependencies]
scry-learn = "0.7"
```

### Optional features

| Feature | What it enables |
|---------|----------------|
| `csv` | `Dataset::from_csv()` file loading |
| `serde` | Serialize / deserialize models |
| `gpu` | GPU-accelerated operations via wgpu compute shaders |
| `polars` | Polars DataFrame interop |
| `mmap` | Memory-mapped dataset loading for large-scale data |
| `experimental` | Kernel SVM (RBF, polynomial kernels) |

```toml
scry-learn = { version = "0.7", features = ["csv", "serde"] }
```

---

## Examples

```bash
# 5-fold stratified CV across 8 models on 4 UCI datasets
cargo run --example industry_report -p scry-learn --release

# Head-to-head comparison vs linfa and smartcore
cargo run --example crosslib_comparison -p scry-learn --release
```

---

## Test suite

905+ tests covering correctness, convergence, numerical stability, and cross-library parity.

```bash
cargo test -p scry-learn
```

| Test suite | What it validates |
|------------|-------------------|
| `correctness` | sklearn reference accuracy verification |
| `convergence` | Monotonic improvement and max_iter stability |
| `numerical_stability` | NaN/Inf handling, gradient norm tracking |
| `mathematical_invariants` | SHAP additivity (Σφᵢ = pred − E[f(x)]) |
| `golden_regression_test` | Deterministic snapshot tests |
| `statistical_robustness` | Bootstrap confidence interval validity |
| `edge_cases` | Empty datasets, single samples, NaN/Inf inputs |
| `production_bench` | Heap memory, allocation counts, scaling curves |
| `memory_crosslib` | Heap usage comparison across libraries |

---

## Contributing

Contributions welcome. Please open an issue before large PRs.

## License

MIT OR Apache-2.0
