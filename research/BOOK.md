# Data Science with Rust: The scry-learn Book

> A beginner-friendly, business-oriented guide to end-to-end data science and
> machine learning using the scry-learn ecosystem.

**Status:** Planning
**Target audience:** Developers entering data science, Python DS practitioners exploring Rust
**Prerequisite:** Basic programming experience (not necessarily Rust)

---

## Goals

1. Teach data science and ML concepts through real business problems
2. Show scry-learn as a complete, production-viable alternative to scikit-learn
3. Every chapter produces runnable code and visualizations
4. No "left as exercise" gaps — all code is complete

## Format

Each chapter follows a consistent structure:
- **Business context** — a realistic problem statement
- **Data exploration** — load, inspect, visualize
- **Model building** — preprocessing, fitting, predicting
- **Evaluation** — metrics, plots, interpretation
- **Business takeaway** — what would you tell the stakeholder?

Companion code lives in `examples/book/chXX_name/` with a shared `datasets/` directory.

---

## Part I — Foundations

### Chapter 1: Why Rust for Data Science
- The case for performance, safety, and reproducibility
- When Rust beats Python (and when it doesn't)
- Setting up: cargo, scry-learn, esoc-chart
- Hello world: load CSV, print summary stats, save a scatter plot

### Chapter 2: Working with Data
- `Dataset::from_csv()` — loading and inspecting
- Column types, missing values, basic statistics
- Subsetting, filtering, reshaping
- Optional: Polars interop for larger workflows (`polars` feature)

### Chapter 3: Visualization Essentials
- Express API one-liners: `scatter()`, `line()`, `bar()`, `histogram()`
- Customization: titles, axis labels, themes (light/dark/publication)
- Saving SVG and PNG output
- Grammar API intro for when you need more control
- **Business scenario:** Sales dashboard — monthly revenue line chart, product bar chart, regional scatter

### Chapter 4: Data Cleaning & Preprocessing
- `SimpleImputer` — handling missing values (mean/median/mode/constant)
- `StandardScaler`, `MinMaxScaler`, `RobustScaler` — when to use which
- `LabelEncoder`, `OneHotEncoder` — categorical features
- `ColumnTransformer` — applying different transforms to different columns
- **Business scenario:** Messy customer survey data — clean, encode, normalize

---

## Part II — Core Machine Learning

### Chapter 5: Linear Regression
- OLS, coefficients, intercept interpretation
- `LinearRegression`, `Ridge`, `LassoRegression`, `ElasticNet`
- `r2_score`, `mean_squared_error`, `mean_absolute_error`
- Residual plots, prediction vs actual scatter
- **Business scenario:** Predicting house prices / quarterly revenue

### Chapter 6: Classification Fundamentals
- Binary classification, decision boundaries
- `LogisticRegression` — penalties (L1/L2/ElasticNet), solvers (LBFGS/SGD)
- `confusion_matrix`, `classification_report`, `accuracy`, `precision`, `recall`, `f1_score`
- ROC curves and AUC via interop visualization
- **Business scenario:** Customer churn prediction

### Chapter 7: Decision Trees & Random Forests
- How trees split, overfitting, pruning
- `DecisionTreeClassifier`, `DecisionTreeRegressor`
- `RandomForestClassifier`, `RandomForestRegressor`
- Feature importance via `permutation_importance()`
- **Business scenario:** Loan approval — which features matter most?

### Chapter 8: Model Evaluation & Selection
- `train_test_split`, `stratified_split`
- `cross_val_score`, `k_fold`, `stratified_k_fold`
- `time_series_split` for temporal data
- Bias-variance tradeoff, learning curves
- Precision-recall tradeoffs, threshold tuning
- **Business scenario:** Comparing 3 models for the same problem

### Chapter 9: Pipelines
- `Pipeline` — chaining preprocessing + model
- `ColumnTransformer` inside pipelines
- `PolynomialFeatures` for non-linear relationships
- Reproducibility: same pipeline for train and inference
- **Business scenario:** Productionizing a credit scoring model

---

## Part III — Advanced Methods

### Chapter 10: Gradient Boosting
- Boosting vs bagging intuition
- `GradientBoostingClassifier`, `GradientBoostingRegressor`
- `HistGradientBoostingClassifier` — when and why it's faster
- `RegressionLoss`, `SplitCriterion` configuration
- **Business scenario:** Sales forecasting with many features

### Chapter 11: Clustering & Segmentation
- Unsupervised learning concepts
- `KMeans`, `MiniBatchKMeans` — centroid methods
- `Dbscan`, `Hdbscan` — density methods
- `AgglomerativeClustering` — hierarchical with linkage options
- `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`
- Scatter plots colored by cluster assignment
- **Business scenario:** Customer segmentation for marketing

### Chapter 12: Anomaly Detection
- Normal vs abnormal, contamination rates
- `IsolationForest` — how isolation trees work
- Threshold tuning, precision at low recall
- **Business scenario:** Fraud detection in transaction data

### Chapter 13: Text Analysis
- Bag of words and TF-IDF intuition
- `CountVectorizer`, `TfidfVectorizer` — sparse CSR output
- `sparse_to_dataset()` — converting for model consumption
- `MultinomialNB`, `BernoulliNB` for text classification
- **Business scenario:** Product review sentiment analysis

### Chapter 14: Neural Networks
- Perceptrons, layers, activations
- `MLPClassifier`, `MLPRegressor`
- `Activation` (ReLU, Sigmoid, Tanh, SoftMax, LeakyReLU)
- `OptimizerKind` (SGD, Adam, RMSProp)
- `TrainingCallback` for monitoring convergence
- When to use MLPs vs tree methods
- **Business scenario:** Handwritten digit recognition (tabular features)

### Chapter 15: SVMs & Nearest Neighbors
- Margins, kernels, distance metrics
- `LinearSVC`, `KernelSVC` (polynomial/RBF)
- `KnnClassifier`, `KnnRegressor`, `KdTree`
- `DistanceMetric` (Euclidean, Manhattan, Minkowski)
- `WeightFunction` (uniform, distance-weighted)
- **Business scenario:** Image feature classification

---

## Part IV — Production & Business Impact

### Chapter 16: Hyperparameter Tuning
- `GridSearchCV` — exhaustive search
- `RandomizedSearchCV` — sampling from distributions
- `BayesSearchCV` — Bayesian optimization
- `ParamGrid`, `ParamSpace`, `ParamDistribution`
- `Tunable` trait — models expose their knobs
- **Business scenario:** Squeezing the last 2% accuracy for a high-stakes model

### Chapter 17: Model Explainability
- Why explainability matters for business trust
- `tree_shap()`, `ensemble_tree_shap()` — SHAP values for trees
- `permutation_importance()` — model-agnostic importance
- Communicating results to non-technical stakeholders
- **Business scenario:** Regulatory compliance — explaining a loan denial

### Chapter 18: Probability Calibration
- Why raw scores aren't probabilities
- `PlattScaling` (sigmoid), `IsotonicRegression` (PAV)
- `CalibratedClassifierCV` — wrapping any classifier
- When calibration matters (risk scoring, medical, insurance)
- **Business scenario:** Insurance risk pricing

### Chapter 19: Ensemble Strategies
- `VotingClassifier` — majority and soft voting
- `StackingClassifier` — meta-learner on top of base models
- When ensembles help vs overfit
- **Business scenario:** Kaggle-style model stacking for max performance

### Chapter 20: End-to-End Case Study
- Raw messy data to business recommendation
- Full pipeline: ingest, clean, explore, feature engineer, model, tune, explain, present
- Multiple model comparison with cross-validation
- Final visualization dashboard (express + grammar API)
- **Business scenario:** Predicting employee attrition — present to the C-suite

---

## Appendices

### Appendix A: Rust Crash Course for Data Scientists
- Ownership, borrowing, lifetimes — just enough to be productive
- Iterators, closures, error handling
- Cargo basics, features, workspace structure

### Appendix B: scry-learn vs scikit-learn API Map
- Side-by-side comparison table of equivalent APIs
- Key differences in naming, patterns, idioms
- Migration tips for Python practitioners

### Appendix C: Performance & Benchmarks
- When Rust's performance advantage matters
- Benchmark results vs Python/scikit-learn
- Memory usage patterns, zero-copy loading

### Appendix D: esoc-chart Reference
- Express API quick reference (all functions + builder methods)
- Grammar API for advanced charts
- Theme customization
- Output format options

---

## esoc-chart Requirements for the Book

The express API is the primary visualization tool. Current audit shows all 9 chart
functions are implemented and working. Before writing begins, these gaps need closing:

### Must-fix (blocks book chapters)
- [ ] Migrate interop module from legacy Figure API to grammar/express API
- [ ] Add `.color()` builder method on scatter/line for manual color override
- [ ] Add `.legend_position()` or `.show_legend()` to express builders
- [ ] Add heatmap support (needed for confusion matrix — Ch6, Ch17)
- [ ] Expose `.annotate()` (hline/vline/text) on express builders (not just grammar)

### Should-fix (improves book quality)
- [ ] Add `.x_range()` / `.y_range()` to express builders for axis limits
- [ ] Better error messages for mismatched array lengths (fail at builder, not compile)
- [ ] Expose faceting on all chart types, not just scatter
- [ ] Add training curve helper to interop (loss/accuracy over epochs — Ch14)
- [ ] Add feature importance bar chart helper to interop (Ch7, Ch17)

### Nice-to-have
- [ ] Custom tick labels in express
- [ ] Multi-panel figure layout (2x2 grid of charts)
- [ ] Horizontal bar chart via `.flip()` in express
- [ ] Color palette override in theme

---

## Datasets (sourced — 4.1 MB total)

All datasets live in `datasets/`. Total size is 4.1 MB, well under the 10 MB shipping target.

### Downloaded (open license)

| Chapter | Dataset | File | Rows | License |
|---------|---------|------|------|---------|
| 1, 3 | Iris | `iris/iris.csv` | 150 | CC BY 4.0 (UCI) |
| 6 | Telco Customer Churn | `telco_churn/telco_churn.csv` | 7,043 | Apache 2.0 (IBM) |
| 7 | German Credit | `german_credit/german_credit.csv` | 1,000 | CC BY 4.0 (UCI) |
| 10 | Online Retail | `online_retail/online_retail.csv` | 10,000 | CC BY 4.0 (UCI) |
| 11 | Wholesale Customers | `wholesale_customers/wholesale_customers.csv` | 440 | CC BY 4.0 (UCI) |
| 13 | SMS Spam Collection | `sms_spam/sms_spam.csv` | 5,574 | CC BY 4.0 (UCI) |
| 14, 15 | Optdigits | `optdigits/optdigits.csv` | 5,620 | CC BY 4.0 (UCI) |
| 20 | HR Employee Attrition | `hr_attrition/hr_attrition.csv` | 1,470 | Apache 2.0 (IBM) |

### Generated (we own, seed=42, reproducible via generate.py)

| Chapter | Dataset | File | Rows |
|---------|---------|------|------|
| 2, 4 | Customer Survey (messy) | `customer_survey/customer_survey.csv` | 500 |
| 5 | Housing Prices | `housing/housing.csv` | 1,500 |
| 9 | Credit Scoring | `credit_scoring/credit_scoring.csv` | 2,000 |
| 10 | Retail Sales | `retail_sales/retail_sales.csv` | 5,053 |
| 12 | Fraud Transactions | `fraud/fraud.csv` | 5,000 |

### Chapter → dataset mapping

| Chapter | Primary dataset | Also reuses |
|---------|----------------|-------------|
| 1 | Iris | — |
| 2 | Customer Survey | — |
| 3 | Iris | — |
| 4 | Customer Survey | — |
| 5 | Housing Prices | — |
| 6 | Telco Churn | — |
| 7 | German Credit | — |
| 8 | (reuses Ch5-7 datasets) | Housing, Telco, German Credit |
| 9 | Credit Scoring | — |
| 10 | Online Retail + Retail Sales | — |
| 11 | Wholesale Customers | — |
| 12 | Fraud Transactions | — |
| 13 | SMS Spam | — |
| 14 | Optdigits | — |
| 15 | (reuses Ch14) | Optdigits |
| 16-19 | (reuses earlier datasets) | — |
| 20 | HR Attrition | — |

---

## Open Questions

- [ ] Physical book, online book, or both? (mdBook for web version?)
- [ ] Do we need a `scry-learn-datasets` crate with built-in loaders?
- [ ] Should chapters include exercises / challenge problems?
- [ ] License for book content vs code?
- [ ] Review process — who reviews for technical accuracy?
- [ ] What Rust edition / MSRV do we target?
