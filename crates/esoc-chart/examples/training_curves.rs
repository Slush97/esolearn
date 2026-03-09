// SPDX-License-Identifier: MIT OR Apache-2.0
//! Simulate training/validation loss curves and plot them.
//!
//! This is a common visualization in ML: watching loss and accuracy
//! converge over training epochs.

use esoc_chart::prelude::*;

fn main() -> Result<()> {
    // ── Simulate training curves ─────────────────────────────────────
    let epochs: Vec<f64> = (1..=50).map(f64::from).collect();

    // Training loss: exponential decay + noise
    let mut rng = SimpleRng::new(42);
    let train_loss: Vec<f64> = epochs
        .iter()
        .map(|&e| 2.0 * (-e / 15.0).exp() + 0.05 + rng.normal() * 0.02)
        .collect();

    // Validation loss: decays slower, starts overfitting around epoch 30
    let val_loss: Vec<f64> = epochs
        .iter()
        .map(|&e| {
            let base = 2.0 * (-e / 20.0).exp() + 0.1;
            let overfit = if e > 30.0 { (e - 30.0) * 0.005 } else { 0.0 };
            base + overfit + rng.normal() * 0.03
        })
        .collect();

    // Training accuracy: rises from ~50% to ~98%
    let train_acc: Vec<f64> = epochs
        .iter()
        .map(|&e| (0.98 - 0.48 * (-e / 12.0).exp()).min(1.0) + rng.normal() * 0.01)
        .collect();

    // Validation accuracy: rises but plateaus earlier
    let val_acc: Vec<f64> = epochs
        .iter()
        .map(|&e| {
            let base = 0.93 - 0.43 * (-e / 18.0).exp();
            let overfit = if e > 30.0 { -(e - 30.0) * 0.002 } else { 0.0 };
            (base + overfit + rng.normal() * 0.015).min(1.0)
        })
        .collect();

    // ── Plot 1: Loss curves ──────────────────────────────────────────
    let mut fig = Figure::new()
        .size(750.0, 500.0)
        .title("Training & Validation Loss");

    let ax = fig.add_axes();
    ax.x_label("Epoch").y_label("Loss");
    ax.line(&epochs, &train_loss)
        .label("Train Loss")
        .color(Color::from_hex("#1f77b4").unwrap())
        .width(2.0)
        .done();
    ax.line(&epochs, &val_loss)
        .label("Val Loss")
        .color(Color::from_hex("#ff7f0e").unwrap())
        .width(2.0)
        .dash(&[8.0, 4.0])
        .done();

    fig.save_svg("training_loss.svg")?;
    println!("Saved training_loss.svg");

    // ── Plot 2: Accuracy curves ──────────────────────────────────────
    let mut fig2 = Figure::new()
        .size(750.0, 500.0)
        .title("Training & Validation Accuracy");

    let ax2 = fig2.add_axes();
    ax2.x_label("Epoch").y_label("Accuracy").y_range(0.4, 1.05);
    ax2.line(&epochs, &train_acc)
        .label("Train Accuracy")
        .color(Color::from_hex("#2ca02c").unwrap())
        .width(2.0)
        .done();
    ax2.line(&epochs, &val_acc)
        .label("Val Accuracy")
        .color(Color::from_hex("#d62728").unwrap())
        .width(2.0)
        .dash(&[8.0, 4.0])
        .done();

    fig2.save_svg("training_accuracy.svg")?;
    println!("Saved training_accuracy.svg");

    Ok(())
}

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
