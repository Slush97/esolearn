// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pure-Rust L-BFGS optimizer for smooth unconstrained optimization.
//!
//! Implements the limited-memory BFGS algorithm with two-loop recursion
//! (Nocedal & Wright, 2006) and strong-Wolfe backtracking line search.
//!
//! This is the gold-standard optimizer for smooth convex problems such as
//! logistic regression — scikit-learn uses it as their default solver.

use std::collections::VecDeque;

/// Configuration for the L-BFGS minimizer.
pub(crate) struct LbfgsConfig {
    /// Maximum number of L-BFGS iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the gradient infinity-norm.
    pub tolerance: f64,
    /// Number of correction pairs to store (default 10).
    pub history_size: usize,
    /// Whether to enforce the strong Wolfe curvature condition in the line
    /// search. When `false`, the line search accepts a step as soon as the
    /// Armijo (sufficient decrease) condition is met — cheaper per iteration
    /// and sufficient for smooth convex objectives like logistic regression.
    /// Default: `true` (full strong-Wolfe, safe for all smooth problems).
    pub wolfe: bool,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tolerance: crate::constants::STRICT_TOL,
            history_size: 10,
            wolfe: true,
        }
    }
}

/// Minimize `f(x)` using L-BFGS.
///
/// # Arguments
///
/// * `x0` — initial parameter vector (mutated in-place to the solution).
/// * `eval` — closure `|x: &[f64], grad: &mut [f64]| -> f64` that computes the
///   function value and writes the gradient into `grad`.
/// * `config` — optimizer configuration.
///
/// # Returns
///
/// The final objective value.
pub(crate) fn minimize(
    x0: &mut [f64],
    mut eval: impl FnMut(&[f64], &mut [f64]) -> f64,
    config: &LbfgsConfig,
) -> f64 {
    let n = x0.len();
    let mut grad = vec![0.0; n];
    let mut f = eval(x0, &mut grad);

    // History of correction pairs (s_k, y_k) and ρ_k = 1 / (y_k · s_k).
    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(config.history_size);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(config.history_size);
    let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(config.history_size);

    let mut x_prev = vec![0.0; n];
    let mut grad_prev = vec![0.0; n];
    let mut direction = vec![0.0; n];
    let mut x_trial = vec![0.0; n];
    let mut grad_trial = vec![0.0; n];
    let mut best_x = vec![0.0; n];
    let mut best_grad = vec![0.0; n];
    let mut s_k = vec![0.0; n];
    let mut y_k = vec![0.0; n];

    // Stagnation detection: stop if function value barely changes for several
    // consecutive iterations (relative improvement < tol).  A patience of 5
    // prevents premature exit on large datasets where the loss plateau is
    // gradual but the model has not yet converged (e.g. logistic regression
    // on adult).  scipy uses patience=10; 5 is a pragmatic middle ground.
    const STAGNATION_PATIENCE: u32 = 5;
    let mut stagnant_count: u32 = 0;

    for _iter in 0..config.max_iter {
        // Check convergence: ||grad||_inf < tolerance.
        let grad_norm = grad.iter().fold(0.0_f64, |mx, &g| mx.max(g.abs()));
        if grad_norm < config.tolerance {
            break;
        }

        // Save previous state.
        x_prev.copy_from_slice(x0);
        grad_prev.copy_from_slice(&grad);

        // ─── Two-loop recursion ───────────────────────────────────
        // Compute search direction d = -H_k * grad  (approximate).
        direction.copy_from_slice(&grad);

        let m = s_hist.len();
        let mut alphas = vec![0.0; m];

        // First loop (reverse).
        for i in (0..m).rev() {
            let s = &s_hist[i];
            let rho = rho_hist[i];
            let mut dot = 0.0;
            for j in 0..n {
                dot += s[j] * direction[j];
            }
            alphas[i] = rho * dot;
            let y = &y_hist[i];
            for j in 0..n {
                direction[j] -= alphas[i] * y[j];
            }
        }

        // Initial Hessian scaling: H_0 = γ I where γ = (s_{k-1} · y_{k-1}) / (y_{k-1} · y_{k-1}).
        if m > 0 {
            let s_last = &s_hist[m - 1];
            let y_last = &y_hist[m - 1];
            let mut sy = 0.0;
            let mut yy = 0.0;
            for j in 0..n {
                sy += s_last[j] * y_last[j];
                yy += y_last[j] * y_last[j];
            }
            let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
            for d in &mut direction {
                *d *= gamma;
            }
        }

        // Second loop (forward).
        for i in 0..m {
            let y = &y_hist[i];
            let rho = rho_hist[i];
            let mut dot = 0.0;
            for j in 0..n {
                dot += y[j] * direction[j];
            }
            let beta = rho * dot;
            let s = &s_hist[i];
            for j in 0..n {
                direction[j] += (alphas[i] - beta) * s[j];
            }
        }

        // Negate to get descent direction.
        for d in &mut direction {
            *d = -*d;
        }

        // ─── Backtracking line search ────────────────────────────
        // Armijo (sufficient decrease):  f(x + α·d) ≤ f(x) + c₁·α·(∇f · d)
        // Strong Wolfe (curvature):      |∇f(x + α·d) · d| ≤ c₂·|∇f · d|
        // When config.wolfe == false, only the Armijo condition is checked.
        let c_armijo = crate::constants::ARMIJO_C;
        let c_wolfe = crate::constants::WOLFE_C2;
        let mut step = 1.0;

        let dir_deriv: f64 = grad
            .iter()
            .zip(direction.iter())
            .map(|(&g, &d)| g * d)
            .sum();

        // Guard: if direction is not a descent direction, fall back to steepest descent.
        if dir_deriv >= 0.0 {
            for (d, g) in direction.iter_mut().zip(grad.iter()) {
                *d = -*g;
            }
            // No line search needed for steepest descent; use a small step.
            step = crate::constants::STEEPEST_DESCENT_SCALE / grad_norm.max(1.0);
        }

        let dir_deriv_ls: f64 = grad
            .iter()
            .zip(direction.iter())
            .map(|(&g, &d)| g * d)
            .sum();
        let abs_dir_deriv = dir_deriv_ls.abs();

        // Track the first (largest) Armijo-satisfying step as fallback
        // in case the Wolfe curvature condition is never met.
        let mut best_armijo_f: f64 = 0.0;
        let mut has_best = false;
        let mut accepted = false;

        // For cubic interpolation: track the previous backtrack's (step, f_trial).
        let mut prev_step = 0.0_f64;
        let mut prev_f_trial = 0.0_f64;

        let f_prev = f;

        for ls in 0..crate::constants::LINE_SEARCH_MAX_ITER {
            for j in 0..n {
                x_trial[j] = x0[j] + step * direction[j];
            }
            let f_trial = eval(&x_trial, &mut grad_trial);
            let armijo_threshold = f + c_armijo * step * dir_deriv_ls;

            if f_trial <= armijo_threshold {
                if !config.wolfe {
                    // Armijo-only mode: accept immediately.
                    f = f_trial;
                    x0.copy_from_slice(&x_trial);
                    grad.copy_from_slice(&grad_trial);
                    accepted = true;
                    break;
                }
                // Full Wolfe mode: record as fallback if first.
                if !has_best {
                    best_armijo_f = f_trial;
                    best_grad.copy_from_slice(&grad_trial);
                    best_x.copy_from_slice(&x_trial);
                    has_best = true;
                }
                // Check strong Wolfe curvature condition.
                let trial_deriv: f64 = grad_trial
                    .iter()
                    .zip(direction.iter())
                    .map(|(&g, &d)| g * d)
                    .sum();
                if trial_deriv.abs() <= c_wolfe * abs_dir_deriv {
                    f = f_trial;
                    x0.copy_from_slice(&x_trial);
                    grad.copy_from_slice(&grad_trial);
                    accepted = true;
                    break;
                }
            }

            // Cubic interpolation for next step size (Nocedal & Wright §3.5).
            // Requires two prior function evaluations at different step sizes.
            if ls > 0 {
                // We have (prev_step, prev_f_trial) and (step, f_trial).
                // Fit a cubic to interpolate the minimizer.
                let d1 = dir_deriv_ls;
                let fa = prev_f_trial - f - d1 * prev_step;
                let fb = f_trial - f - d1 * step;
                let denom = (prev_step * prev_step * step * step) * (step - prev_step);
                if denom.abs() > 1e-30 {
                    let a = (step * step * fa - prev_step * prev_step * fb) / denom;
                    let b =
                        (-step * step * step * fa + prev_step * prev_step * prev_step * fb) / denom;
                    let disc = b * b - 3.0 * a * d1;
                    if a.abs() > 1e-30 && disc >= 0.0 {
                        let cubic_min = (-b + disc.sqrt()) / (3.0 * a);
                        // Only use if the cubic minimizer is in a reasonable range.
                        if cubic_min > 0.1 * step && cubic_min < 0.9 * step {
                            prev_step = step;
                            prev_f_trial = f_trial;
                            step = cubic_min;
                            continue;
                        }
                    }
                }
            }

            // Fall back to 0.5× contraction.
            prev_step = step;
            prev_f_trial = f_trial;
            step *= crate::constants::LINE_SEARCH_BACKTRACK;
        }

        // Fallback: if Wolfe was never satisfied but Armijo was, accept
        // the first (largest) Armijo step to guarantee progress.
        if !accepted && has_best {
            f = best_armijo_f;
            x0.copy_from_slice(&best_x);
            grad.copy_from_slice(&best_grad);
        }

        // Stagnation check: |f_prev - f| < tol * |f_prev|.
        if f_prev.abs() > 0.0 && (f_prev - f).abs() < config.tolerance * f_prev.abs() {
            stagnant_count += 1;
            if stagnant_count >= STAGNATION_PATIENCE {
                break;
            }
        } else {
            stagnant_count = 0;
        }
        let mut sy = 0.0;
        for j in 0..n {
            s_k[j] = x0[j] - x_prev[j];
            y_k[j] = grad[j] - grad_prev[j];
            sy += s_k[j] * y_k[j];
        }

        // Only add to history if curvature condition holds.
        if sy > crate::constants::LBFGS_CURVATURE_THRESH {
            if s_hist.len() == config.history_size {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            rho_hist.push_back(1.0 / sy);
            s_hist.push_back(s_k.clone());
            y_hist.push_back(y_k.clone());
        }
    }

    f
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    /// Minimum at (1, 1) with f = 0.
    #[test]
    fn test_rosenbrock() {
        let mut x = vec![-1.0, 1.0];
        let config = LbfgsConfig {
            max_iter: 500,
            tolerance: 1e-10,
            history_size: 10,
            ..Default::default()
        };

        let f = minimize(
            &mut x,
            |x, grad| {
                let a = 1.0 - x[0];
                let b = x[1] - x[0] * x[0];
                grad[0] = -2.0 * a - 400.0 * x[0] * b;
                grad[1] = 200.0 * b;
                a * a + 100.0 * b * b
            },
            &config,
        );

        assert!(f < 1e-10, "expected f ≈ 0, got {f}");
        assert!((x[0] - 1.0).abs() < 1e-4, "expected x[0] ≈ 1, got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "expected x[1] ≈ 1, got {}", x[1]);
    }

    /// Simple quadratic: f(x) = 0.5 * x^T x, gradient = x, minimum at origin.
    #[test]
    fn test_quadratic() {
        let mut x = vec![3.0, -4.0, 5.0];
        let config = LbfgsConfig::default();

        let f = minimize(
            &mut x,
            |x, grad| {
                let mut val = 0.0;
                for (g, &xi) in grad.iter_mut().zip(x.iter()) {
                    *g = xi;
                    val += 0.5 * xi * xi;
                }
                val
            },
            &config,
        );

        assert!(f < 1e-12, "expected f ≈ 0, got {f}");
        for (i, &xi) in x.iter().enumerate() {
            assert!(xi.abs() < 1e-6, "expected x[{i}] ≈ 0, got {xi}");
        }
    }
}
