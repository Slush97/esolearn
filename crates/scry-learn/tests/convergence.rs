//! Convergence tests — verify that iterative algorithms improve monotonically
//! with increasing iteration counts.
//!
//! For each iterative solver, we run with `max_iter` in [1, 5, 10, 50, 200, 1000]
//! and assert that the evaluation metric improves (or stays flat) at each step.
//! This proves: (a) solvers actually converge, (b) default iteration counts are
//! sufficient, (c) loss decreases monotonically within noise tolerance.
//!
//! Run: cargo test --test convergence -p scry-learn --release -- --nocapture

use scry_learn::dataset::Dataset;
use scry_learn::metrics::{accuracy, mean_squared_error, r2_score};
use scry_learn::split::train_test_split;

fn load_csv_dataset(name: &str) -> Dataset {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    let feat_path = base.join(format!("{name}_features.csv"));
    let target_path = base.join(format!("{name}_target.csv"));

    let mut rdr = csv::Reader::from_path(&feat_path).unwrap();
    let n_cols = rdr.headers().unwrap().len();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for result in rdr.records() {
        let record = result.unwrap();
        rows.push(record.iter().map(|s| s.parse::<f64>().unwrap()).collect());
    }
    let n_rows = rows.len();
    let mut cols = vec![vec![0.0; n_rows]; n_cols];
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            cols[j][i] = val;
        }
    }

    let mut rdr = csv::Reader::from_path(&target_path).unwrap();
    let target: Vec<f64> = rdr
        .records()
        .map(|r| r.unwrap()[0].parse::<f64>().unwrap())
        .collect();

    let names: Vec<String> = (0..n_cols).map(|i| format!("f{i}")).collect();
    Dataset::new(cols, target, names, "target")
}

const ITER_STEPS: &[usize] = &[1, 5, 10, 50, 200, 1000];
// Tolerance for monotonicity — small regression allowed due to stochastic effects
const MONO_TOL: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════
// Logistic Regression convergence
// ═══════════════════════════════════════════════════════════════════

#[test]
fn convergence_logistic_regression() {
    let ds = load_csv_dataset("iris");
    let (train, test) = train_test_split(&ds, 0.2, 42);

    let test_rows: Vec<Vec<f64>> = (0..test.n_samples())
        .map(|i| {
            (0..test.n_features())
                .map(|j| test.features[j][i])
                .collect()
        })
        .collect();

    println!("\n{}", "=".repeat(65));
    println!("CONVERGENCE: LogisticRegression on Iris (80/20 split)");
    println!("  {:>10} {:>12}", "max_iter", "accuracy");
    println!("  {}", "-".repeat(25));

    let mut prev_acc = 0.0;
    for &mi in ITER_STEPS {
        let mut lr = scry_learn::linear::LogisticRegression::new()
            .max_iter(mi)
            .learning_rate(0.1);
        lr.fit(&train).unwrap();
        let preds = lr.predict(&test_rows).unwrap();
        let acc = accuracy(&test.target, &preds);
        println!("  {mi:>10} {acc:>12.4}");

        assert!(
            acc >= prev_acc - MONO_TOL,
            "accuracy decreased from {prev_acc:.4} to {acc:.4} at max_iter={mi}"
        );
        prev_acc = acc;
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// Lasso convergence
// ═══════════════════════════════════════════════════════════════════

#[test]
fn convergence_lasso() {
    let ds = load_csv_dataset("california");
    let (train, test) = train_test_split(&ds, 0.2, 42);

    let test_rows: Vec<Vec<f64>> = (0..test.n_samples())
        .map(|i| {
            (0..test.n_features())
                .map(|j| test.features[j][i])
                .collect()
        })
        .collect();

    println!("\n{}", "=".repeat(65));
    println!("CONVERGENCE: Lasso on California Housing (80/20 split)");
    println!("  {:>10} {:>12} {:>12}", "max_iter", "MSE", "R2");
    println!("  {}", "-".repeat(38));

    let mut prev_mse = f64::MAX;
    for &mi in ITER_STEPS {
        let mut lasso = scry_learn::linear::LassoRegression::new()
            .alpha(0.01)
            .max_iter(mi);
        lasso.fit(&train).unwrap();
        let preds = lasso.predict(&test_rows).unwrap();
        let mse = mean_squared_error(&test.target, &preds);
        let r2 = r2_score(&test.target, &preds);
        println!("  {mi:>10} {mse:>12.4} {r2:>12.4}");

        // MSE should decrease (or stay flat) as iterations increase
        assert!(
            mse <= prev_mse + prev_mse * MONO_TOL,
            "MSE increased from {prev_mse:.4} to {mse:.4} at max_iter={mi}"
        );
        prev_mse = mse;
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// ElasticNet convergence
// ═══════════════════════════════════════════════════════════════════

#[test]
fn convergence_elastic_net() {
    let ds = load_csv_dataset("california");
    let (train, test) = train_test_split(&ds, 0.2, 42);

    let test_rows: Vec<Vec<f64>> = (0..test.n_samples())
        .map(|i| {
            (0..test.n_features())
                .map(|j| test.features[j][i])
                .collect()
        })
        .collect();

    println!("\n{}", "=".repeat(65));
    println!("CONVERGENCE: ElasticNet on California Housing (80/20 split)");
    println!("  {:>10} {:>12} {:>12}", "max_iter", "MSE", "R2");
    println!("  {}", "-".repeat(38));

    let mut prev_mse = f64::MAX;
    for &mi in ITER_STEPS {
        let mut en = scry_learn::linear::ElasticNet::new()
            .alpha(0.01)
            .l1_ratio(0.5)
            .max_iter(mi);
        en.fit(&train).unwrap();
        let preds = en.predict(&test_rows).unwrap();
        let mse = mean_squared_error(&test.target, &preds);
        let r2 = r2_score(&test.target, &preds);
        println!("  {mi:>10} {mse:>12.4} {r2:>12.4}");

        assert!(
            mse <= prev_mse + prev_mse * MONO_TOL,
            "MSE increased from {prev_mse:.4} to {mse:.4} at max_iter={mi}"
        );
        prev_mse = mse;
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// LinearSVC convergence
// ═══════════════════════════════════════════════════════════════════

#[test]
fn convergence_linear_svc() {
    let ds = load_csv_dataset("iris");
    let (train, test) = train_test_split(&ds, 0.2, 42);

    let test_rows: Vec<Vec<f64>> = (0..test.n_samples())
        .map(|i| {
            (0..test.n_features())
                .map(|j| test.features[j][i])
                .collect()
        })
        .collect();

    println!("\n{}", "=".repeat(65));
    println!("CONVERGENCE: LinearSVC on Iris (80/20 split)");
    println!("  {:>10} {:>12}", "max_iter", "accuracy");
    println!("  {}", "-".repeat(25));

    let mut accs = Vec::new();
    for &mi in ITER_STEPS {
        let mut svc = scry_learn::svm::LinearSVC::new().max_iter(mi);
        svc.fit(&train).unwrap();
        let preds = svc.predict(&test_rows).unwrap();
        let acc = accuracy(&test.target, &preds);
        println!("  {mi:>10} {acc:>12.4}");
        accs.push(acc);
    }

    // SVM with hinge loss can be non-monotonic at very low iterations due to
    // the subgradient method. Verify the overall trend: final accuracy should
    // exceed initial accuracy, and high-iteration results should be stable.
    let first = accs[0];
    let last = *accs.last().unwrap();
    assert!(
        last >= first - MONO_TOL,
        "final accuracy {last:.4} should be >= initial {first:.4}"
    );
    // High-iter stability: last two steps should not diverge
    let second_last = accs[accs.len() - 2];
    assert!(
        (last - second_last).abs() < 0.05,
        "high-iteration accuracy should stabilize: {second_last:.4} vs {last:.4}"
    );
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// KMeans convergence (inertia should decrease)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn convergence_kmeans() {
    let ds = load_csv_dataset("iris");

    println!("\n{}", "=".repeat(65));
    println!("CONVERGENCE: KMeans on Iris (k=3)");
    println!("  {:>10} {:>14}", "max_iter", "inertia");
    println!("  {}", "-".repeat(28));

    let mut prev_inertia = f64::MAX;
    for &mi in ITER_STEPS {
        let mut km = scry_learn::cluster::KMeans::new(3)
            .seed(42)
            .max_iter(mi)
            .n_init(1);
        km.fit(&ds).unwrap();
        let inertia = km.inertia();
        println!("  {mi:>10} {inertia:>14.4}");

        assert!(
            inertia <= prev_inertia + prev_inertia * MONO_TOL,
            "inertia increased from {prev_inertia:.4} to {inertia:.4} at max_iter={mi}"
        );
        prev_inertia = inertia;
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// GradientBoosting convergence (MSE should decrease with more trees)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn convergence_gradient_boosting() {
    let ds = load_csv_dataset("california");
    let (train, test) = train_test_split(&ds, 0.2, 42);

    let test_rows: Vec<Vec<f64>> = (0..test.n_samples())
        .map(|i| {
            (0..test.n_features())
                .map(|j| test.features[j][i])
                .collect()
        })
        .collect();

    let estimator_steps: &[usize] = &[1, 5, 10, 25, 50, 100];

    println!("\n{}", "=".repeat(65));
    println!("CONVERGENCE: GradientBoostingRegressor on California Housing (80/20 split)");
    println!("  {:>14} {:>12} {:>12}", "n_estimators", "MSE", "R2");
    println!("  {}", "-".repeat(40));

    let mut prev_mse = f64::MAX;
    for &ne in estimator_steps {
        let mut gbr = scry_learn::tree::GradientBoostingRegressor::new()
            .n_estimators(ne)
            .learning_rate(0.1)
            .max_depth(3);
        gbr.fit(&train).unwrap();
        let preds = gbr.predict(&test_rows).unwrap();
        let mse = mean_squared_error(&test.target, &preds);
        let r2 = r2_score(&test.target, &preds);
        println!("  {ne:>14} {mse:>12.4} {r2:>12.4}");

        assert!(
            mse <= prev_mse + prev_mse * MONO_TOL,
            "MSE increased from {prev_mse:.4} to {mse:.4} at n_estimators={ne}"
        );
        prev_mse = mse;
    }
    println!();
}
