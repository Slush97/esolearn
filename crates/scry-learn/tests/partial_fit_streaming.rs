//! Integration test: streaming simulation with `partial_fit`.
//!
//! Generates 10K samples, splits into 10 batches of 1K, trains
//! incrementally, and verifies final model accuracy.

use scry_learn::cluster::MiniBatchKMeans;
use scry_learn::dataset::Dataset;
use scry_learn::linear::{LogisticRegression, Solver};
use scry_learn::naive_bayes::GaussianNb;
use scry_learn::partial_fit::PartialFit;

/// Generate `n` linearly separable samples with 2 features.
/// Class 0 centered at (-2, -2), class 1 centered at (2, 2).
fn make_batch(n: usize, rng: &mut fastrand::Rng) -> Dataset {
    let mut f0 = Vec::with_capacity(n);
    let mut f1 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);
    for _ in 0..n / 2 {
        f0.push(-2.0 + rng.f64() * 2.0 - 1.0);
        f1.push(-2.0 + rng.f64() * 2.0 - 1.0);
        target.push(0.0);
    }
    for _ in n / 2..n {
        f0.push(2.0 + rng.f64() * 2.0 - 1.0);
        f1.push(2.0 + rng.f64() * 2.0 - 1.0);
        target.push(1.0);
    }
    Dataset::new(vec![f0, f1], target, vec!["x".into(), "y".into()], "class")
}

#[test]
fn streaming_logistic_regression() {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut model = LogisticRegression::new()
        .solver(Solver::GradientDescent)
        .learning_rate(0.1)
        .alpha(0.0);

    // Train on 10 batches of 1000 samples.
    for _ in 0..10 {
        let batch = make_batch(1000, &mut rng);
        model.partial_fit(&batch).unwrap();
    }

    // Test on fresh data.
    let test = make_batch(200, &mut rng);
    let matrix = test.feature_matrix();
    let preds = model.predict(&matrix).unwrap();
    let acc = preds
        .iter()
        .zip(test.target.iter())
        .filter(|(p, t)| (*p - *t).abs() < 1e-6)
        .count() as f64
        / test.n_samples() as f64;

    assert!(
        acc >= 0.85,
        "streaming LogReg: expected >= 85% accuracy, got {:.1}%",
        acc * 100.0
    );
}

#[test]
fn streaming_gaussian_nb() {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut model = GaussianNb::new();

    for _ in 0..10 {
        let batch = make_batch(1000, &mut rng);
        model.partial_fit(&batch).unwrap();
    }

    let test = make_batch(200, &mut rng);
    let matrix = test.feature_matrix();
    let preds = model.predict(&matrix).unwrap();
    let acc = preds
        .iter()
        .zip(test.target.iter())
        .filter(|(p, t)| (*p - *t).abs() < 1e-6)
        .count() as f64
        / test.n_samples() as f64;

    assert!(
        acc >= 0.85,
        "streaming GaussianNB: expected >= 85% accuracy, got {:.1}%",
        acc * 100.0
    );
}

#[test]
fn streaming_mini_batch_kmeans() {
    let mut rng = fastrand::Rng::with_seed(42);
    let mut model = MiniBatchKMeans::new(2).seed(42);

    for _ in 0..10 {
        let batch = make_batch(1000, &mut rng);
        model.partial_fit(&batch).unwrap();
    }

    // Centroids should be near (-2,-2) and (2,2).
    let c = model.centroids();
    let has_neg = c.iter().any(|ci| ci[0] < 0.0 && ci[1] < 0.0);
    let has_pos = c.iter().any(|ci| ci[0] > 0.0 && ci[1] > 0.0);
    assert!(
        has_neg,
        "expected a centroid in negative quadrant, got {c:?}"
    );
    assert!(
        has_pos,
        "expected a centroid in positive quadrant, got {c:?}"
    );
}
