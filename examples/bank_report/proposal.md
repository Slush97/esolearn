# Project Proposal: Universal Bank Personal Loan Campaign — Predictive Modeling

## 1. Introduction

Universal Bank serves a diverse customer base, the majority of whom are liability customers (depositors) with a smaller proportion holding loan products. A previous blanket marketing campaign offering personal loans to the entire customer base achieved only a 9.6% acceptance rate. This project proposes a data-driven approach to identify which customers are most likely to accept a personal loan offer, enabling targeted campaigns that reduce marketing costs and improve conversion rates.

## 2. Problem Statement

Given a customer's demographic and banking attributes, can we predict whether they will accept a personal loan offer? Solving this binary classification problem will allow the bank to:

- Focus marketing resources on high-probability prospects
- Reduce customer fatigue from irrelevant offers
- Increase campaign acceptance rates and overall ROI
- Grow the loan portfolio by converting existing depositors

The dataset is significantly imbalanced (90.4% declined, 9.6% accepted), which requires careful model selection and evaluation beyond simple accuracy.

## 3. Dataset

The Universal Bank dataset contains 5,000 customer records with 14 variables. Each record represents a customer targeted in the previous campaign, with the outcome (accepted/declined) as the target variable.

| Feature          | Type        | Description                                |
|------------------|-------------|--------------------------------------------|
| Age              | Continuous  | Customer age in years                      |
| Experience       | Continuous  | Years of professional experience           |
| Income           | Continuous  | Annual income ($K)                         |
| Family           | Ordinal     | Family size (1–4)                          |
| CCAvg            | Continuous  | Avg. monthly credit card spending ($K)     |
| Education        | Ordinal     | 1 = Undergrad, 2 = Graduate, 3 = Advanced  |
| Mortgage         | Continuous  | House mortgage value ($K)                  |
| Securities Acct  | Binary      | Has a securities account?                  |
| CD Account       | Binary      | Has a certificate of deposit account?      |
| Online           | Binary      | Uses internet banking?                     |
| CreditCard       | Binary      | Uses a bank credit card?                   |
| **Personal Loan** | **Binary** | **Target — accepted the loan offer?**      |

Two non-predictive columns (ID, ZIP Code) are dropped before modeling, leaving 11 features.

## 4. Methodology

### 4.1 Data Cleaning & Preprocessing

1. Drop non-predictive columns (ID, ZIP Code)
2. Clip 52 negative Experience values to 0 (data entry errors)
3. Assess and impute missing values if present (none found)
4. Retain legitimate outliers in Income/Mortgage (high-net-worth customers)
5. Standardize features (StandardScaler) for distance-sensitive models; tree-based models use raw features

### 4.2 Exploratory Data Analysis

- Summary statistics for all features
- Target class distribution analysis
- Feature distribution histograms (Income, CCAvg, Age, Mortgage)
- Pairwise scatter plots colored by loan acceptance to reveal clustering patterns

### 4.3 Feature Selection

- Domain-based exclusion of ID and ZIP Code
- Permutation feature importance (Breiman, 2001) on the trained model using held-out test data (5 repeats, seed = 42) to rank predictive variables

### 4.4 Model Training & Evaluation

Seven classification models are trained and compared using 5-fold stratified cross-validation (80/20 train/test split, seed = 42):

| Model                  | Description                              |
|------------------------|------------------------------------------|
| Decision Tree          | Max depth 8                              |
| Random Forest          | 100 estimators, max depth 10             |
| Gradient Boosting      | 100 estimators, max depth 5, lr = 0.1    |
| K-Nearest Neighbors    | k = 5, on standardized features          |
| Logistic Regression    | 500 iterations, lr = 0.01, standardized  |
| Linear SVC             | C = 1.0, 1000 iterations, standardized   |
| Gaussian Naive Bayes   | Default parameters                       |

Evaluation metrics: accuracy, precision, recall, F1-score, ROC AUC, and confusion matrices.

## 5. Results

### 5.1 Cross-Validation Comparison

| Model                 | Mean Accuracy | Std    |
|-----------------------|---------------|--------|
| Gradient Boosting     | 0.9882        | 0.0033 |
| Random Forest         | 0.9880        | 0.0014 |
| Decision Tree         | 0.9820        | 0.0055 |
| KNN (k=5)            | 0.9602        | 0.0037 |
| Linear SVC           | 0.9512        | 0.0049 |
| Logistic Regression   | 0.9504        | 0.0031 |
| Gaussian Naive Bayes  | 0.8838        | 0.0035 |

Tree-based ensembles dominate, with Random Forest showing the lowest variance across folds.

### 5.2 Best Model — Random Forest (Test Set)

| Metric     | Declined (0) | Accepted (1) |
|------------|-------------|--------------|
| Precision  | 0.992       | 0.978        |
| Recall     | 0.998       | 0.927        |
| F1-Score   | 0.995       | 0.952        |

- **Test accuracy: 99.1%**
- **ROC AUC: 0.998**
- Improves minority-class recall by +29.2 percentage points over Logistic Regression (92.7% vs. 63.5%)

### 5.3 Feature Importance (Top Predictors)

1. **Income** — 0.181 mean accuracy decrease (dominant predictor)
2. **Education** — 0.084
3. **Family** — 0.062
4. **CCAvg** — 0.014
5. **CD Account** — 0.006

## 6. Conclusions & Recommendations

1. **Deploy the Random Forest model** for targeted campaigns — 99.1% accuracy and 0.998 AUC using only standard customer data
2. **Prioritize three customer segments:** high-income/high-spending customers (>$100K income, >$3K/mo CC spend), advanced-degree holders, and CD account holders
3. **Expected impact:** achieve comparable loan conversion with ~10% of the marketing effort, dramatically improving cost efficiency
4. **Monitor and retrain** periodically as customer behavior shifts
5. **Future work:** feature engineering (Income/CCAvg ratio, interaction terms), hyperparameter tuning, SHAP-based explanations for individual predictions

## 7. Tools & Technologies

- **scry-learn** — Rust-based ML library for model training and evaluation
- **esoc-chart** — Visualization library for generating report figures
- **Dataset:** Universal Bank (5,000 records, public dataset)
