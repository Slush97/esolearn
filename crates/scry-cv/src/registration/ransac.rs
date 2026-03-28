// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generic RANSAC framework for robust model estimation.

use crate::rng::FastRng;

/// Configuration for the RANSAC algorithm.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct RansacConfig {
    /// Maximum iterations.
    pub max_iterations: u32,
    /// Inlier threshold (interpretation depends on the model).
    pub threshold: f64,
    /// Confidence level (0..1) for adaptive iteration count.
    pub confidence: f64,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            threshold: 3.0,
            confidence: 0.999,
            seed: 42,
        }
    }
}

/// Result of RANSAC estimation.
#[derive(Clone, Debug)]
pub struct RansacResult<M> {
    /// The best model found.
    pub model: M,
    /// Boolean mask: `true` for inliers.
    pub inlier_mask: Vec<bool>,
    /// Number of inliers.
    pub n_inliers: usize,
}

/// Trait for models estimatable via RANSAC.
pub trait RansacModel: Clone {
    /// Type of a single data point.
    type Point;

    /// Minimum number of points needed to estimate the model.
    const MIN_SAMPLES: usize;

    /// Estimate a model from exactly `MIN_SAMPLES` points.
    /// Returns `None` if the configuration is degenerate.
    fn estimate(points: &[Self::Point]) -> Option<Self>;

    /// Compute the error for a single point against this model.
    fn error(&self, point: &Self::Point) -> f64;
}

/// Run RANSAC on a dataset, returning the best model and inlier mask.
pub fn ransac<M: RansacModel>(
    data: &[M::Point],
    config: &RansacConfig,
) -> Option<RansacResult<M>>
where
    M::Point: Clone,
{
    if data.len() < M::MIN_SAMPLES {
        return None;
    }

    let mut rng = FastRng::new(config.seed);
    let n = data.len();
    let mut best: Option<RansacResult<M>> = None;
    let mut adaptive_max = config.max_iterations;

    for iter in 0..adaptive_max {
        if iter >= adaptive_max {
            break;
        }

        // Random sample of unique indices
        let mut indices = Vec::with_capacity(M::MIN_SAMPLES);
        while indices.len() < M::MIN_SAMPLES {
            let idx = rng.usize(0..n);
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }

        let model = estimate_from_indices::<M>(data, &indices);
        let model = match model {
            Some(m) => m,
            None => continue,
        };

        // Count inliers
        let mut inlier_mask = vec![false; n];
        let mut n_inliers = 0;
        for (i, point) in data.iter().enumerate() {
            if model.error(point) < config.threshold {
                inlier_mask[i] = true;
                n_inliers += 1;
            }
        }

        let is_better = match &best {
            Some(b) => n_inliers > b.n_inliers,
            None => n_inliers >= M::MIN_SAMPLES,
        };

        if is_better {
            best = Some(RansacResult {
                model,
                inlier_mask,
                n_inliers,
            });

            // Adaptive iteration count
            let inlier_ratio = n_inliers as f64 / n as f64;
            if inlier_ratio > 0.0 {
                let p_fail =
                    1.0 - inlier_ratio.powi(M::MIN_SAMPLES as i32);
                if p_fail > 0.0 && p_fail < 1.0 {
                    let new_max =
                        ((1.0 - config.confidence).ln() / p_fail.ln()).ceil() as u32;
                    adaptive_max = adaptive_max.min(new_max.max(10));
                }
            }
        }
    }

    best
}

/// Helper: estimate model from indexed data points.
///
/// Since `RansacModel::estimate` takes `&[Point]`, we need the data to be
/// index-accessible. This function works with any slice.
fn estimate_from_indices<M: RansacModel>(data: &[M::Point], indices: &[usize]) -> Option<M>
where
    M::Point: Clone,
{
    let subset: Vec<M::Point> = indices.iter().map(|&i| data[i].clone()).collect();
    M::estimate(&subset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test model: fit a line y = mx + b
    #[derive(Clone, Debug)]
    struct LineModel {
        m: f64,
        b: f64,
    }

    impl RansacModel for LineModel {
        type Point = (f64, f64);
        const MIN_SAMPLES: usize = 2;

        fn estimate(points: &[Self::Point]) -> Option<Self> {
            let (x0, y0) = points[0];
            let (x1, y1) = points[1];
            let dx = x1 - x0;
            if dx.abs() < 1e-10 {
                return None;
            }
            let m = (y1 - y0) / dx;
            let b = y0 - m * x0;
            Some(Self { m, b })
        }

        fn error(&self, point: &Self::Point) -> f64 {
            let (x, y) = *point;
            (y - (self.m * x + self.b)).abs()
        }
    }

    #[test]
    fn ransac_finds_line() {
        // Line y = 2x + 1, with some outliers
        let mut data: Vec<(f64, f64)> = (0..50)
            .map(|i| {
                let x = i as f64;
                (x, 2.0 * x + 1.0)
            })
            .collect();
        // Add outliers
        data.push((10.0, 100.0));
        data.push((20.0, -50.0));
        data.push((30.0, 200.0));

        let config = RansacConfig {
            max_iterations: 100,
            threshold: 0.5,
            confidence: 0.99,
            seed: 42,
        };

        let result = ransac::<LineModel>(&data, &config).unwrap();
        assert!(
            (result.model.m - 2.0).abs() < 0.1,
            "slope should be ~2.0, got {}",
            result.model.m
        );
        assert!(
            (result.model.b - 1.0).abs() < 0.5,
            "intercept should be ~1.0, got {}",
            result.model.b
        );
        assert!(result.n_inliers >= 48, "should have >= 48 inliers, got {}", result.n_inliers);
    }
}
