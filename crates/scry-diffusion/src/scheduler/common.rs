// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared scheduler primitives — β schedule generation and timestep-index
//! resolution. Used by every concrete sampler in this module.

use crate::error::{Error, Result};

/// Beta schedule shape. `ScaledLinear` is what SD 1.5 ships.
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    /// `linspace(beta_start^0.5, beta_end^0.5, T)` then squared. SD 1.5 default.
    ScaledLinear,
    /// Plain linear `linspace(beta_start, beta_end, T)`.
    Linear,
}

/// Build the per-step β schedule for the configured shape.
#[must_use]
pub(crate) fn build_betas(
    num_train_timesteps: u32,
    beta_start: f32,
    beta_end: f32,
    schedule: BetaSchedule,
) -> Vec<f32> {
    let t = num_train_timesteps as usize;
    let mut betas = Vec::with_capacity(t);
    match schedule {
        BetaSchedule::ScaledLinear => {
            let start = beta_start.sqrt();
            let end = beta_end.sqrt();
            for i in 0..t {
                let v = lerp(start, end, i, t);
                betas.push(v * v);
            }
        }
        BetaSchedule::Linear => {
            for i in 0..t {
                betas.push(lerp(beta_start, beta_end, i, t));
            }
        }
    }
    betas
}

/// `linspace(a, b, n)[i]`. Matches NumPy / PyTorch's formula for `n >= 2`;
/// returns `a` when `n == 1`.
#[allow(clippy::cast_precision_loss)]
fn lerp(a: f32, b: f32, i: usize, n: usize) -> f32 {
    if n <= 1 {
        a
    } else {
        a + (b - a) * (i as f32) / ((n - 1) as f32)
    }
}

/// Round a floating-point timestep to its integer α-cumprod index, validating bounds.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn timestep_to_index(timestep: f32, table_len: usize) -> Result<u32> {
    if !timestep.is_finite() || timestep < 0.0 {
        return Err(Error::Scheduler(format!(
            "timestep {timestep} must be finite and non-negative"
        )));
    }
    let rounded = timestep.round();
    if rounded as usize >= table_len {
        return Err(Error::Scheduler(format!(
            "timestep {timestep} out of range for {table_len}-step schedule"
        )));
    }
    Ok(rounded as u32)
}
