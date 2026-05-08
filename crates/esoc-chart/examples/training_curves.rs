// SPDX-License-Identifier: MIT OR Apache-2.0
//! Simulate training/validation loss curves and plot them.
//!
//! This is a common visualization in ML: watching loss and accuracy
//! converge over training epochs.

use esoc_chart::express::line;

fn main() -> esoc_chart::error::Result<()> {
    let epochs: Vec<f64> = (1..=50).map(f64::from).collect();

    let mut rng = SimpleRng::new(42);
    let train_loss: Vec<f64> = epochs
        .iter()
        .map(|&e| 2.0 * (-e / 15.0).exp() + 0.05 + rng.normal() * 0.02)
        .collect();

    let val_loss: Vec<f64> = epochs
        .iter()
        .map(|&e| {
            let base = 2.0 * (-e / 20.0).exp() + 0.1;
            let overfit = if e > 30.0 { (e - 30.0) * 0.005 } else { 0.0 };
            base + overfit + rng.normal() * 0.03
        })
        .collect();

    let train_acc: Vec<f64> = epochs
        .iter()
        .map(|&e| (0.98 - 0.48 * (-e / 12.0).exp()).min(1.0) + rng.normal() * 0.01)
        .collect();

    let val_acc: Vec<f64> = epochs
        .iter()
        .map(|&e| {
            let base = 0.93 - 0.43 * (-e / 18.0).exp();
            let overfit = if e > 30.0 { -(e - 30.0) * 0.002 } else { 0.0 };
            (base + overfit + rng.normal() * 0.015).min(1.0)
        })
        .collect();

    let (loss_x, loss_y, loss_series) = stack_series(
        &epochs,
        &[("Train Loss", &train_loss), ("Val Loss", &val_loss)],
    );
    line(&loss_x, &loss_y)
        .color_by(&loss_series)
        .title("Training & Validation Loss")
        .x_label("Epoch")
        .y_label("Loss")
        .size(750.0, 500.0)
        .save_svg("training_loss.svg")?;
    println!("Saved training_loss.svg");

    let (acc_x, acc_y, acc_series) = stack_series(
        &epochs,
        &[("Train Accuracy", &train_acc), ("Val Accuracy", &val_acc)],
    );
    line(&acc_x, &acc_y)
        .color_by(&acc_series)
        .title("Training & Validation Accuracy")
        .x_label("Epoch")
        .y_label("Accuracy")
        .y_domain(0.4, 1.05)
        .size(750.0, 500.0)
        .save_svg("training_accuracy.svg")?;
    println!("Saved training_accuracy.svg");

    Ok(())
}

fn stack_series(
    x: &[f64],
    series: &[(&str, &Vec<f64>)],
) -> (Vec<f64>, Vec<f64>, Vec<String>) {
    let cap = x.len() * series.len();
    let mut xs = Vec::with_capacity(cap);
    let mut ys = Vec::with_capacity(cap);
    let mut labels = Vec::with_capacity(cap);
    for (name, ys_in) in series {
        for (&xi, &yi) in x.iter().zip(ys_in.iter()) {
            xs.push(xi);
            ys.push(yi);
            labels.push((*name).to_string());
        }
    }
    (xs, ys, labels)
}

struct SimpleRng(u64);
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
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
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
