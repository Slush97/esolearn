// SPDX-License-Identifier: MIT OR Apache-2.0
//! Benchmark the GPU-resident tensor forward pass.
//!
//! Trains an MLP classifier on a synthetic dataset large enough to trigger
//! the GPU dispatch path (batch * `max_dim` >= 4096), then compares
//! forward-pass throughput for inference.
//!
//! Usage:
//!   cargo run -p scry-learn --example `gpu_tensor_bench` --release [--features scry-gpu]

use std::time::Instant;

use scry_learn::dataset::Dataset;
use scry_learn::neural::{Activation, MLPClassifier};

fn main() {
    let n_samples = 10_000;
    let n_features = 128;
    let n_classes = 10;
    let batch_size = 256;

    println!("═══ GPU-resident tensor forward pass benchmark ═══\n");
    println!("  samples:   {n_samples}");
    println!("  features:  {n_features}");
    println!("  classes:   {n_classes}");
    println!("  batch:     {batch_size}");
    println!("  hidden:    [256, 128, 64]");
    println!();

    // ── Synthetic dataset ──
    let (features, target) = make_dataset(n_samples, n_features, n_classes, 42);
    let feat_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let ds = Dataset::new(features, target, feat_names, "class");

    // ── Train ──
    println!("Training (5 epochs)...");
    let mut clf = MLPClassifier::new()
        .hidden_layers(&[256, 128, 64])
        .activation(Activation::Relu)
        .learning_rate(0.001)
        .max_iter(5)
        .batch_size(batch_size)
        .seed(42);

    let t0 = Instant::now();
    clf.fit(&ds).expect("training failed");
    let train_time = t0.elapsed();
    println!("  train time: {:.1}ms", train_time.as_secs_f64() * 1000.0);
    println!(
        "  final loss: {:.4}",
        clf.loss_curve.last().copied().unwrap_or(f64::NAN)
    );
    println!();

    // ── Inference benchmark ──
    // Build row-major feature matrix for predict (one Vec per sample)
    let features_for_predict: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| (0..n_features).map(|j| ds.features[j][i]).collect())
        .collect();

    // Warm up
    let _ = clf.predict(&features_for_predict);

    let n_iters = 20;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = clf.predict(&features_for_predict);
    }
    let predict_time = t0.elapsed();
    let per_predict = predict_time.as_secs_f64() * 1000.0 / n_iters as f64;
    let throughput = n_samples as f64 / (per_predict / 1000.0);

    println!("Inference ({n_iters} iterations):");
    println!("  total:      {:.1}ms", predict_time.as_secs_f64() * 1000.0);
    println!("  per batch:  {per_predict:.2}ms");
    println!("  throughput: {throughput:.0} samples/sec");
    println!();

    // ── Accuracy ──
    let preds = clf.predict(&features_for_predict).expect("predict failed");
    let correct = preds
        .iter()
        .zip(&ds.target)
        .filter(|(p, t)| (**p - *t).abs() < 0.5)
        .count();
    println!(
        "Train accuracy: {:.1}% ({correct}/{n_samples})",
        100.0 * correct as f64 / n_samples as f64
    );

    // ── Summary ──
    println!();
    println!("With --features scry-gpu: forward pass uses GPU-resident tensors");
    println!("  (1 upload → N GPU matmuls → 1 download per batch)");
    println!("Without: forward pass is pure CPU");
}

/// Generate a synthetic classification dataset (column-major features).
fn make_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = SimpleRng(seed);
    let mut features = vec![vec![0.0; n_samples]; n_features];
    let mut target = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let class = i % n_classes;
        target.push(class as f64);
        for (f, col) in features.iter_mut().enumerate() {
            let signal = (class * n_features + f) as f64 * 0.01;
            let noise = rng.normal() * 0.5;
            col[i] = signal + noise;
        }
    }

    (features, target)
}

struct SimpleRng(u64);
impl SimpleRng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        // Box-Muller
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
