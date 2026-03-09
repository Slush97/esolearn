// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mini-Batch K-Means clustering.
//!
//! Uses random mini-batches for centroid updates instead of full-data passes.
//! Much faster on large datasets with slightly worse cluster quality.
//!
//! # Example
//!
//! ```
//! use scry_learn::cluster::MiniBatchKMeans;
//! use scry_learn::dataset::Dataset;
//!
//! let data = Dataset::new(
//!     vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]],
//!     vec![0.0; 4],
//!     vec!["x".into(), "y".into()],
//!     "label",
//! );
//!
//! let mut mbk = MiniBatchKMeans::new(2).batch_size(2).seed(42);
//! mbk.fit(&data).unwrap();
//! assert_eq!(mbk.labels().len(), 4);
//! ```

use super::kmeans::kmeans_plus_plus;
use crate::dataset::Dataset;
use crate::distance::euclidean_sq;
use crate::error::{Result, ScryLearnError};
use crate::partial_fit::PartialFit;

/// Mini-Batch K-Means clustering.
///
/// Approximates standard K-Means by updating centroids using random mini-batches
/// of the data at each iteration, rather than the full dataset.
/// This is significantly faster for large datasets while producing similar results.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MiniBatchKMeans {
    k: usize,
    batch_size: usize,
    max_iter: usize,
    tolerance: f64,
    seed: u64,
    centroids: Vec<Vec<f64>>,
    labels: Vec<usize>,
    inertia: f64,
    n_iter: usize,
    fitted: bool,
    // Per-centroid update counts for streaming average (used by partial_fit).
    centroid_counts: Vec<u64>,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl MiniBatchKMeans {
    /// Create a Mini-Batch K-Means model with k clusters.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            batch_size: 1024,
            max_iter: 100,
            tolerance: 0.0,
            seed: 42,
            centroids: Vec::new(),
            labels: Vec::new(),
            inertia: f64::INFINITY,
            n_iter: 0,
            fitted: false,
            centroid_counts: Vec::new(),
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the mini-batch size (default 1024).
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    /// Set maximum iterations.
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set convergence tolerance.
    pub fn tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }

    /// Alias for [`tolerance`](Self::tolerance) (sklearn convention).
    pub fn tol(self, t: f64) -> Self {
        self.tolerance(t)
    }

    /// Set random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Fit the model on a dataset (uses features only, ignores target).
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
        let m = data.n_features();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.k == 0 || self.k > n {
            return Err(ScryLearnError::InvalidParameter(format!(
                "k must be between 1 and n_samples ({}), got {}",
                n, self.k
            )));
        }

        let rows = data.feature_matrix();
        let mut rng = crate::rng::FastRng::new(self.seed);
        let effective_batch = self.batch_size.min(n);

        // K-means++ initialization.
        let mut centroids = kmeans_plus_plus(&rows, self.k, self.seed);

        // Per-centroid update counts for streaming average.
        let mut centroid_counts = vec![0_u64; self.k];
        let mut prev_inertia = f64::INFINITY;

        for iter in 0..self.max_iter {
            // Sample a mini-batch.
            let batch_indices: Vec<usize> = (0..effective_batch).map(|_| rng.usize(0..n)).collect();

            // Assign batch samples to nearest centroid.
            let mut assignments = Vec::with_capacity(effective_batch);
            for &idx in &batch_indices {
                let mut best_c = 0;
                let mut best_dist = f64::INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = euclidean_sq(&rows[idx], centroid);
                    if d < best_dist {
                        best_dist = d;
                        best_c = c;
                    }
                }
                assignments.push(best_c);
            }

            // Update centroids with streaming average.
            for (batch_i, &idx) in batch_indices.iter().enumerate() {
                let c = assignments[batch_i];
                centroid_counts[c] += 1;
                let lr = 1.0 / centroid_counts[c] as f64;
                for j in 0..m {
                    centroids[c][j] += lr * (rows[idx][j] - centroids[c][j]);
                }
            }

            // Compute full inertia periodically (every 10 iters or last iter).
            if iter % 10 == 0 || iter == self.max_iter - 1 {
                let mut inertia = 0.0;
                for row in &rows {
                    let mut best_dist = f64::INFINITY;
                    for centroid in &centroids {
                        let d = euclidean_sq(row, centroid);
                        if d < best_dist {
                            best_dist = d;
                        }
                    }
                    inertia += best_dist;
                }

                self.n_iter = iter + 1;
                self.inertia = inertia;

                if self.tolerance > 0.0 && (prev_inertia - inertia).abs() < self.tolerance {
                    break;
                }
                prev_inertia = inertia;
            }
        }

        // Final assignment of all points.
        self.labels = rows
            .iter()
            .map(|row| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        euclidean_sq(row, a)
                            .partial_cmp(&euclidean_sq(row, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(0, |(idx, _)| idx)
            })
            .collect();

        self.centroids = centroids;
        self.fitted = true;
        Ok(())
    }

    /// Predict cluster assignments for new data.
    pub fn predict(&self, features: &[Vec<f64>]) -> Result<Vec<usize>> {
        crate::version::check_schema_version(self._schema_version)?;
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(features
            .iter()
            .map(|row| {
                self.centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        euclidean_sq(row, a)
                            .partial_cmp(&euclidean_sq(row, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(0, |(idx, _)| idx)
            })
            .collect())
    }

    /// Get the cluster centroids.
    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }

    /// Get cluster labels for training data.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Sum of squared distances to the nearest centroid.
    pub fn inertia(&self) -> f64 {
        self.inertia
    }

    /// Number of iterations run.
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl PartialFit for MiniBatchKMeans {
    /// Update centroids with a streaming average over the given batch.
    ///
    /// On the first call, initializes centroids via K-Means++ on the batch.
    /// Subsequent calls assign each sample to the nearest centroid and
    /// update it with a decaying learning rate (`1 / count`).
    fn partial_fit(&mut self, data: &Dataset) -> Result<()> {
        let n = data.n_samples();
        if n == 0 {
            // No-op on empty batch if already initialized; error if not.
            if self.is_initialized() {
                return Ok(());
            }
            return Err(ScryLearnError::EmptyDataset);
        }

        let rows = data.feature_matrix();
        let m = data.n_features();

        if !self.is_initialized() {
            if self.k == 0 || self.k > n {
                return Err(ScryLearnError::InvalidParameter(format!(
                    "k must be between 1 and n_samples ({}), got {}",
                    n, self.k
                )));
            }
            self.centroids = kmeans_plus_plus(&rows, self.k, self.seed);
            self.centroid_counts = vec![0_u64; self.k];
        }

        // Assign each sample to nearest centroid and update with streaming average.
        for row in &rows {
            let mut best_c = 0;
            let mut best_dist = f64::INFINITY;
            for (c, centroid) in self.centroids.iter().enumerate() {
                let d = euclidean_sq(row, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_c = c;
                }
            }
            self.centroid_counts[best_c] += 1;
            let lr = 1.0 / self.centroid_counts[best_c] as f64;
            for j in 0..m {
                self.centroids[best_c][j] += lr * (row[j] - self.centroids[best_c][j]);
            }
        }

        // Assign labels and compute inertia for this batch.
        self.labels = rows
            .iter()
            .map(|row| {
                self.centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        euclidean_sq(row, a)
                            .partial_cmp(&euclidean_sq(row, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map_or(0, |(idx, _)| idx)
            })
            .collect();

        self.inertia = rows
            .iter()
            .map(|row| {
                self.centroids
                    .iter()
                    .map(|c| euclidean_sq(row, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .sum();

        self.fitted = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        !self.centroids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mini_batch_kmeans_two_blobs() {
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        for i in 0..30 {
            f1.push(i as f64 % 3.0);
            f2.push(i as f64 % 3.0);
        }
        for i in 0..30 {
            f1.push(100.0 + i as f64 % 3.0);
            f2.push(100.0 + i as f64 % 3.0);
        }

        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; 60],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut mbk = MiniBatchKMeans::new(2).seed(42).batch_size(20);
        mbk.fit(&data).unwrap();

        let labels = mbk.labels();
        let first_label = labels[0];
        assert!(labels[..30].iter().all(|&l| l == first_label));
        assert!(labels[30..].iter().all(|&l| l != first_label));
    }

    #[test]
    fn test_mini_batch_kmeans_predict() {
        let data = Dataset::new(
            vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]],
            vec![0.0; 4],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut mbk = MiniBatchKMeans::new(2).seed(42).batch_size(4);
        mbk.fit(&data).unwrap();

        let pred = mbk.predict(&[vec![1.0, 1.0], vec![9.0, 9.0]]).unwrap();
        assert_ne!(
            pred[0], pred[1],
            "nearby and far points should be in different clusters"
        );
    }

    #[test]
    fn test_partial_fit_is_initialized() {
        let mut mbk = MiniBatchKMeans::new(2);
        assert!(!mbk.is_initialized());

        let data = Dataset::new(
            vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]],
            vec![0.0; 4],
            vec!["x".into(), "y".into()],
            "label",
        );
        mbk.partial_fit(&data).unwrap();
        assert!(mbk.is_initialized());
    }

    #[test]
    fn test_partial_fit_convergence() {
        // Two well-separated blobs, fed in batches.
        let mut mbk = MiniBatchKMeans::new(2).seed(42);

        // Batch 1: cluster A around (1, 1)
        let b1 = Dataset::new(
            vec![vec![0.5, 1.0, 1.5], vec![0.5, 1.0, 1.5]],
            vec![0.0; 3],
            vec!["x".into(), "y".into()],
            "label",
        );
        // Batch 2: cluster B around (10, 10)
        let b2 = Dataset::new(
            vec![vec![9.5, 10.0, 10.5], vec![9.5, 10.0, 10.5]],
            vec![0.0; 3],
            vec!["x".into(), "y".into()],
            "label",
        );

        mbk.partial_fit(&b1).unwrap();
        mbk.partial_fit(&b2).unwrap();

        // Centroids should be near the two cluster centers.
        let c = mbk.centroids();
        let c0_near_1 = c
            .iter()
            .any(|ci| (ci[0] - 1.0).abs() < 3.0 && (ci[1] - 1.0).abs() < 3.0);
        let c1_near_10 = c
            .iter()
            .any(|ci| (ci[0] - 10.0).abs() < 3.0 && (ci[1] - 10.0).abs() < 3.0);
        assert!(c0_near_1, "expected a centroid near (1,1), got {c:?}");
        assert!(c1_near_10, "expected a centroid near (10,10), got {c:?}");

        // Should predict correctly.
        let pred = mbk.predict(&[vec![1.0, 1.0], vec![10.0, 10.0]]).unwrap();
        assert_ne!(pred[0], pred[1], "different clusters expected");
    }
}
