//! Generate all charts and statistics for the Universal Bank LaTeX report.
//!
//! Outputs SVG charts to `report/figures/` and prints stats to stdout.

use std::path::PathBuf;
use std::time::Instant;

use esoc_chart::prelude::*;
use esoc_chart::v2::{self, bar, boxplot, histogram, pie, NewTheme};
use esoc_chart::grammar::coord::CoordSystem;

use scry_learn::dataset::Dataset;
use scry_learn::explain::permutation_importance;
use scry_learn::linear::LogisticRegression;
use scry_learn::metrics::{
    accuracy, classification_report, confusion_matrix, roc_curve,
};
use scry_learn::naive_bayes::GaussianNb;
use scry_learn::neighbors::KnnClassifier;
use scry_learn::preprocess::{SimpleImputer, StandardScaler, Transformer};
use scry_learn::split::{cross_val_score_stratified, train_test_split, ScoringFn};
use scry_learn::svm::LinearSVC;
use scry_learn::tree::{
    DecisionTreeClassifier, GradientBoostingClassifier, RandomForestClassifier,
};

fn datasets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("datasets")
}

fn figures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("report")
        .join("figures")
}

/// Convert column-major Dataset features to row-major for predict().
fn to_row_major(data: &Dataset) -> Vec<Vec<f64>> {
    let n = data.n_samples();
    let m = data.n_features();
    (0..n)
        .map(|i| (0..m).map(|j| data.features[j][i]).collect())
        .collect()
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let fig_dir = figures_dir();
    std::fs::create_dir_all(&fig_dir)?;

    let theme = NewTheme::light();

    // ═══════════════════════════════════════════════════════════════
    //  1. Load & clean
    // ═══════════════════════════════════════════════════════════════
    let csv_path = datasets_dir().join("universal_bank.csv");
    let mut data = Dataset::from_csv(csv_path.to_str().unwrap(), "Personal Loan")?;

    println!("=== RAW DATA ===");
    println!("Samples: {}, Features: {}", data.n_samples(), data.n_features());
    println!("Feature names: {:?}", data.feature_names);
    data.describe();

    // Drop ID and ZIP Code
    let drop_cols: Vec<&str> = vec!["ID", "ZIP Code"];
    let keep: Vec<usize> = data
        .feature_names
        .iter()
        .enumerate()
        .filter(|(_, name)| !drop_cols.contains(&name.as_str()))
        .map(|(i, _)| i)
        .collect();

    let kept_names: Vec<String> = keep.iter().map(|&i| data.feature_names[i].clone()).collect();
    let kept_features: Vec<Vec<f64>> = keep.iter().map(|&i| data.features[i].clone()).collect();
    data = Dataset::new(kept_features, data.target.clone(), kept_names, &data.target_name);

    // Clip negative Experience
    if let Some(exp_idx) = data.feature_names.iter().position(|n| n == "Experience") {
        let neg_count = data.features[exp_idx].iter().filter(|&&v| v < 0.0).count();
        if neg_count > 0 {
            println!("\nClipped {} negative Experience values to 0.", neg_count);
            for v in &mut data.features[exp_idx] {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }

    // Impute NaN
    let has_nan = data.features.iter().any(|col| col.iter().any(|v| v.is_nan()));
    if has_nan {
        let mut imputer = SimpleImputer::new();
        imputer.fit(&data)?;
        imputer.transform(&mut data)?;
        println!("Imputed NaN values with column means.");
    }

    println!("\n=== CLEANED DATA ===");
    println!("Samples: {}, Features: {}", data.n_samples(), data.n_features());
    data.describe();

    // ═══════════════════════════════════════════════════════════════
    //  2. Class distribution chart
    // ═══════════════════════════════════════════════════════════════
    let positive = data.target.iter().filter(|&&v| v == 1.0).count();
    let negative = data.target.iter().filter(|&&v| v == 0.0).count();
    println!("\n=== CLASS DISTRIBUTION ===");
    println!("Declined (0): {} ({:.1}%)", negative, 100.0 * negative as f64 / data.n_samples() as f64);
    println!("Accepted (1): {} ({:.1}%)", positive, 100.0 * positive as f64 / data.n_samples() as f64);

    // Pie chart
    pie(
        &[negative as f64, positive as f64],
        &["Declined", "Accepted"],
    )
    .title("Personal Loan — Class Distribution")
    .theme(theme.clone())
    .size(500.0, 500.0)
    .build()
    .save_svg(fig_dir.join("class_distribution.svg").to_str().unwrap())?;

    // Bar chart version
    bar(
        &["Declined (0)", "Accepted (1)"],
        &[negative as f64, positive as f64],
    )
    .title("Personal Loan — Class Counts")
    .x_label("Class")
    .y_label("Count")
    .theme(theme.clone())
    .size(600.0, 400.0)
    .build()
    .save_svg(fig_dir.join("class_bar.svg").to_str().unwrap())?;

    // ═══════════════════════════════════════════════════════════════
    //  3. Feature histograms
    // ═══════════════════════════════════════════════════════════════
    let continuous_features = ["Age", "Experience", "Income", "CCAvg", "Mortgage"];
    for feat_name in &continuous_features {
        if let Some(idx) = data.feature_names.iter().position(|n| n == feat_name) {
            histogram(&data.features[idx])
                .bins(30)
                .title(&format!("Distribution of {feat_name}"))
                .x_label(*feat_name)
                .y_label("Frequency")
                .theme(theme.clone())
                .size(600.0, 400.0)
                .build()
                .save_svg(
                    fig_dir
                        .join(format!("hist_{}.svg", feat_name.to_lowercase()))
                        .to_str()
                        .unwrap(),
                )?;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  4. Key scatter plots
    // ═══════════════════════════════════════════════════════════════
    let class_labels: Vec<String> = data
        .target
        .iter()
        .map(|&v| if v == 1.0 { "Accepted".to_string() } else { "Declined".to_string() })
        .collect();

    // Income vs CCAvg
    if let (Some(inc_idx), Some(cc_idx)) = (
        data.feature_names.iter().position(|n| n == "Income"),
        data.feature_names.iter().position(|n| n == "CCAvg"),
    ) {
        v2::scatter(&data.features[inc_idx], &data.features[cc_idx])
            .color_by(&class_labels)
            .title("Income vs Credit Card Spending")
            .x_label("Income ($K)")
            .y_label("CCAvg ($K/month)")
            .theme(theme.clone())
            .size(700.0, 500.0)
            .build()
            .save_svg(fig_dir.join("scatter_income_ccavg.svg").to_str().unwrap())?;
    }

    // Income distribution by Education Level (boxplot)
    if let (Some(inc_idx), Some(edu_idx)) = (
        data.feature_names.iter().position(|n| n == "Income"),
        data.feature_names.iter().position(|n| n == "Education"),
    ) {
        let edu_labels: Vec<String> = data.features[edu_idx]
            .iter()
            .map(|&v| match v as i64 {
                1 => "Undergraduate".to_string(),
                2 => "Graduate".to_string(),
                3 => "Advanced/Professional".to_string(),
                other => format!("Level {other}"),
            })
            .collect();
        boxplot(&edu_labels, &data.features[inc_idx])
            .title("Income Distribution by Education Level")
            .x_label("Education Level")
            .y_label("Income ($K)")
            .theme(theme.clone())
            .size(700.0, 500.0)
            .build()
            .save_svg(fig_dir.join("boxplot_income_education.svg").to_str().unwrap())?;
    }

    // Age vs Income
    if let (Some(age_idx), Some(inc_idx)) = (
        data.feature_names.iter().position(|n| n == "Age"),
        data.feature_names.iter().position(|n| n == "Income"),
    ) {
        v2::scatter(&data.features[age_idx], &data.features[inc_idx])
            .color_by(&class_labels)
            .title("Age vs Income")
            .x_label("Age")
            .y_label("Income ($K)")
            .theme(theme.clone())
            .size(700.0, 500.0)
            .build()
            .save_svg(fig_dir.join("scatter_age_income.svg").to_str().unwrap())?;
    }

    // ═══════════════════════════════════════════════════════════════
    //  5. Normalize & split
    // ═══════════════════════════════════════════════════════════════
    let mut scaled = data.clone();
    let mut scaler = StandardScaler::new();
    scaler.fit(&scaled)?;
    scaler.transform(&mut scaled)?;

    let (train, test) = train_test_split(&data, 0.2, 42);
    let (train_s, test_s) = train_test_split(&scaled, 0.2, 42);

    println!("\n=== TRAIN/TEST SPLIT ===");
    println!("Train: {}, Test: {}", train.n_samples(), test.n_samples());

    // ═══════════════════════════════════════════════════════════════
    //  6. Cross-validation comparison
    // ═══════════════════════════════════════════════════════════════
    println!("\n=== 5-FOLD STRATIFIED CV ===");
    let scorer: ScoringFn = accuracy;

    struct CvResult {
        name: &'static str,
        mean: f64,
        std: f64,
        time_ms: f64,
    }

    let mut results: Vec<CvResult> = Vec::new();

    macro_rules! run_cv {
        ($name:expr, $model:expr, $data:expr) => {{
            let start = Instant::now();
            let scores = cross_val_score_stratified(&$model, &$data, 5, scorer, 42)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let var = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
            println!("  {:25} {:.4} ± {:.4}  ({:.1} ms)", $name, mean, var.sqrt(), elapsed);
            results.push(CvResult { name: $name, mean, std: var.sqrt(), time_ms: elapsed });
        }};
    }

    run_cv!("Decision Tree", DecisionTreeClassifier::new().max_depth(8), data);
    run_cv!("Random Forest", RandomForestClassifier::new().n_estimators(50).max_depth(10).seed(42), data);
    run_cv!("Gradient Boosting", GradientBoostingClassifier::new().n_estimators(100).max_depth(5).learning_rate(0.1), data);
    run_cv!("Gaussian NB", GaussianNb::new(), data);
    run_cv!("Logistic Regression", LogisticRegression::new().max_iter(500).learning_rate(0.01), scaled);
    run_cv!("KNN (k=5)", KnnClassifier::new().k(5), scaled);
    run_cv!("LinearSVC", LinearSVC::new().c(1.0).max_iter(1000), scaled);

    // Model comparison bar chart
    let model_names: Vec<&str> = results.iter().map(|r| r.name).collect();
    let model_accs: Vec<f64> = results.iter().map(|r| r.mean).collect();
    let model_stds: Vec<f64> = results.iter().map(|r| r.std).collect();
    bar(&model_names, &model_accs)
        .error_bars(&model_stds)
        .title("5-Fold CV Accuracy by Model")
        .x_label("Model")
        .y_label("Mean Accuracy")
        .theme(theme.clone())
        .size(900.0, 500.0)
        .build()
        .save_svg(fig_dir.join("model_comparison.svg").to_str().unwrap())?;

    // ═══════════════════════════════════════════════════════════════
    //  7. Best model — Random Forest test evaluation
    // ═══════════════════════════════════════════════════════════════
    println!("\n=== RANDOM FOREST TEST EVALUATION ===");
    let mut rf = RandomForestClassifier::new()
        .n_estimators(100)
        .max_depth(10)
        .seed(42);
    rf.fit(&train)?;
    let test_rows = to_row_major(&test);
    let preds = rf.predict(&test_rows)?;
    let probas = rf.predict_proba(&test_rows)?;

    let report = classification_report(&test.target, &preds);
    println!("{report}");

    // Confusion matrix chart
    let cm = confusion_matrix(&test.target, &preds);
    cm.figure()
        .save_svg(fig_dir.join("confusion_matrix_rf.svg").to_str().unwrap())?;

    // Classification report grouped bar
    report
        .figure()
        .save_svg(fig_dir.join("classification_report_rf.svg").to_str().unwrap())?;

    // ROC curve (binary — use P(class=1) scores)
    let scores: Vec<f64> = probas.iter().map(|p| p.get(1).copied().unwrap_or(0.0)).collect();
    let roc = roc_curve(&test.target, &scores);
    println!("ROC AUC: {:.4}", roc.auc);
    roc.roc_figure()
        .save_svg(fig_dir.join("roc_curve_rf.svg").to_str().unwrap())?;

    // ═══════════════════════════════════════════════════════════════
    //  8. Feature importance (permutation-based)
    // ═══════════════════════════════════════════════════════════════
    println!("\n=== PERMUTATION FEATURE IMPORTANCE ===");
    let pi = permutation_importance(
        &test.features,
        &test.target,
        &|feats: &[Vec<f64>]| {
            let n = feats[0].len();
            let m = feats.len();
            let rows: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..m).map(|j| feats[j][i]).collect())
                .collect();
            rf.predict(&rows).unwrap()
        },
        accuracy,
        5,
        42,
    );

    // Sort by importance descending
    let mut feat_imp: Vec<(String, f64)> = data
        .feature_names
        .iter()
        .zip(pi.importances_mean.iter())
        .map(|(name, &imp)| (name.clone(), imp))
        .collect();
    feat_imp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (name, imp) in &feat_imp {
        println!("  {:20} {:.4}", name, imp);
    }

    // Reverse so highest importance is at top when rendered as horizontal bars
    let fi_names: Vec<&str> = feat_imp.iter().rev().map(|(n, _)| n.as_str()).collect();
    let fi_vals: Vec<f64> = feat_imp.iter().rev().map(|(_, v)| *v).collect();
    bar(&fi_names, &fi_vals)
        .title("Permutation Feature Importance (Random Forest)")
        .x_label("Feature")
        .y_label("Mean Accuracy Decrease")
        .theme(theme.clone())
        .size(900.0, 500.0)
        .build()
        .coord(CoordSystem::Flipped)
        .save_svg(fig_dir.join("feature_importance.svg").to_str().unwrap())?;

    // ═══════════════════════════════════════════════════════════════
    //  9. Surrogate model — Interpretable ICP rules
    // ═══════════════════════════════════════════════════════════════
    println!("\n=== SURROGATE MODEL — ICP RULES ===");

    // Generate RF predictions on training set (surrogate labels)
    let train_rows = to_row_major(&train);
    let rf_train_preds = rf.predict(&train_rows)?;

    // Build surrogate dataset: unscaled features + RF predictions as target
    let surrogate_data = Dataset::new(
        train.features.clone(),
        rf_train_preds,
        data.feature_names.clone(),
        "RF_prediction",
    );

    // Fit shallow decision tree on RF predictions
    let mut surrogate = DecisionTreeClassifier::new().max_depth(3);
    surrogate.fit(&surrogate_data)?;

    // Measure fidelity (agreement with RF on test set)
    let surrogate_preds = surrogate.predict(&test_rows)?;
    let fidelity = surrogate_preds
        .iter()
        .zip(preds.iter())
        .filter(|(s, r)| (*s - *r).abs() < 1e-6)
        .count() as f64
        / test.n_samples() as f64;
    let surrogate_acc = accuracy(&test.target, &surrogate_preds);
    println!(
        "Surrogate fidelity to RF: {:.2}%",
        fidelity * 100.0
    );
    println!(
        "Surrogate accuracy (vs true labels): {:.2}%",
        surrogate_acc * 100.0
    );
    println!(
        "Surrogate depth: {}, leaves: {}",
        surrogate.depth(),
        surrogate.n_leaves()
    );

    // Detect integer-valued features for clean threshold formatting
    let is_integer_feature: Vec<bool> = data
        .feature_names
        .iter()
        .enumerate()
        .map(|(i, _)| data.features[i].iter().all(|v| *v == v.floor()))
        .collect();

    // Walk the FlatTree to extract root-to-leaf rules
    let ft = surrogate.flat_tree().unwrap();
    let nodes = &ft.nodes;
    let predictions_ft = &ft.predictions;
    let node_counts = &ft.node_counts;
    let total_train = train.n_samples() as f64;

    struct PathEntry {
        node_idx: usize,
        conditions: Vec<String>,
    }

    let mut stack = vec![PathEntry {
        node_idx: 0,
        conditions: vec![],
    }];
    // (conditions, prediction, coverage%)
    let mut rules: Vec<(Vec<String>, f64, f64)> = vec![];

    while let Some(PathEntry {
        node_idx,
        conditions,
    }) = stack.pop()
    {
        let node = &nodes[node_idx];
        if node.right == u32::MAX {
            // Leaf: feature_idx is repurposed as leaf data index
            let leaf_idx = node.feature_idx as usize;
            let pred = predictions_ft[leaf_idx];
            let coverage = if node_idx < node_counts.len() {
                node_counts[node_idx] as f64 / total_train * 100.0
            } else {
                0.0
            };
            rules.push((conditions, pred, coverage));
        } else {
            let feat_idx = node.feature_idx as usize;
            let feat_name = &data.feature_names[feat_idx];
            let threshold = node.threshold;
            let is_int = is_integer_feature[feat_idx];

            // Right child (feature > threshold)
            let mut right_conds = conditions.clone();
            if is_int {
                right_conds.push(format!(
                    "{} >= {}",
                    feat_name,
                    threshold.floor() as i64 + 1
                ));
            } else {
                right_conds.push(format!("{} > {:.1}", feat_name, threshold));
            }
            stack.push(PathEntry {
                node_idx: node.right as usize,
                conditions: right_conds,
            });

            // Left child (feature <= threshold)
            let mut left_conds = conditions;
            if is_int {
                left_conds.push(format!(
                    "{} <= {}",
                    feat_name,
                    threshold.floor() as i64
                ));
            } else {
                left_conds.push(format!("{} <= {:.1}", feat_name, threshold));
            }
            stack.push(PathEntry {
                node_idx: node_idx + 1,
                conditions: left_conds,
            });
        }
    }

    // Sort: positive class (ACCEPTED) first, then by coverage descending
    rules.sort_by(|a, b| {
        let a_pos = a.1 == 1.0;
        let b_pos = b.1 == 1.0;
        b_pos.cmp(&a_pos).then(b.2.partial_cmp(&a.2).unwrap())
    });

    println!("\n--- ICP Rules (Personal Loan Acceptance) ---");
    let mut rule_num = 0;
    for (conditions, pred, coverage) in &rules {
        let label = if *pred == 1.0 {
            "ACCEPTED"
        } else {
            "DECLINED"
        };
        let marker = if *pred == 1.0 { ">>>" } else { "   " };
        rule_num += 1;
        let cond_str = if conditions.is_empty() {
            "Always".to_string()
        } else {
            conditions.join(" AND ")
        };
        println!(
            "{marker} Rule {rule_num}: {cond_str}\n       -> {label} (covers {coverage:.1}% of training data)"
        );
    }

    // ═══════════════════════════════════════════════════════════════
    // 10. Logistic Regression test evaluation
    // ═══════════════════════════════════════════════════════════════
    println!("\n=== LOGISTIC REGRESSION TEST EVALUATION ===");
    let mut lr = LogisticRegression::new().max_iter(500).learning_rate(0.01);
    lr.fit(&train_s)?;
    let test_s_rows = to_row_major(&test_s);
    let preds_lr = lr.predict(&test_s_rows)?;
    let probas_lr = lr.predict_proba(&test_s_rows)?;

    let report_lr = classification_report(&test_s.target, &preds_lr);
    println!("{report_lr}");

    let cm_lr = confusion_matrix(&test_s.target, &preds_lr);
    cm_lr.figure()
        .save_svg(fig_dir.join("confusion_matrix_lr.svg").to_str().unwrap())?;

    let scores_lr: Vec<f64> = probas_lr.iter().map(|p| p.get(1).copied().unwrap_or(0.0)).collect();
    let roc_lr = roc_curve(&test_s.target, &scores_lr);
    println!("ROC AUC: {:.4}", roc_lr.auc);
    roc_lr.roc_figure()
        .save_svg(fig_dir.join("roc_curve_lr.svg").to_str().unwrap())?;

    println!("\n=== CHARTS SAVED ===");
    println!("All figures written to: {}", fig_dir.display());

    Ok(())
}
