// SPDX-License-Identifier: MIT OR Apache-2.0
//! K-Means clustering with k-means++ initialization.
//!
//! # Example
//!
//! ```
//! use scry_learn::cluster::KMeans;
//! use scry_learn::dataset::Dataset;
//!
//! let data = Dataset::new(
//!     vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]],
//!     vec![0.0; 4],
//!     vec!["x".into(), "y".into()],
//!     "label",
//! );
//!
//! let mut km = KMeans::new(2).n_init(10).seed(42);
//! km.fit(&data).unwrap();
//! assert_eq!(km.labels().len(), 4);
//! ```

use rayon::prelude::*;

use crate::constants::KMEANS_PAR_THRESHOLD;
use crate::dataset::Dataset;
use crate::distance::euclidean_sq;
use crate::error::{Result, ScryLearnError};

/// K-Means clustering.
///
/// Uses k-means++ initialization for better convergence.
/// When `n_init > 1` (default 10), the algorithm runs multiple times
/// with different random seeds and keeps the result with the lowest inertia.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KMeans {
    k: usize,
    max_iter: usize,
    tolerance: f64,
    seed: u64,
    n_init: usize,
    centroids: Vec<Vec<f64>>,
    labels: Vec<usize>,
    inertia: f64,
    n_iter: usize,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl KMeans {
    /// Create a K-Means model with k clusters.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 300,
            tolerance: 1e-4,
            seed: 42,
            n_init: 10,
            centroids: Vec::new(),
            labels: Vec::new(),
            inertia: f64::INFINITY,
            n_iter: 0,
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set maximum iterations per run.
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

    /// Set the number of independent runs with different random seeds.
    ///
    /// The result with the lowest inertia is kept. Default is 10, matching sklearn.
    /// Set to 1 for a single run (faster but less reliable).
    pub fn n_init(mut self, n: usize) -> Self {
        self.n_init = n.max(1);
        self
    }

    /// Fit the model on a dataset (uses features only, ignores target).
    ///
    /// When `n_init > 1`, runs K-Means multiple times and keeps the best.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
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
        let m = data.n_features();

        let mut best_centroids: Option<Vec<Vec<f64>>> = None;
        let mut best_labels: Option<Vec<usize>> = None;
        let mut best_inertia = f64::INFINITY;
        let mut best_n_iter = 0;

        for run in 0..self.n_init {
            let run_seed = self.seed.wrapping_add(run as u64);
            let (centroids, labels, inertia, n_iter) = self.run_once(&rows, n, m, run_seed);

            if inertia < best_inertia {
                best_centroids = Some(centroids);
                best_labels = Some(labels);
                best_inertia = inertia;
                best_n_iter = n_iter;
            }
        }

        self.centroids = best_centroids.unwrap_or_default();
        self.labels = best_labels.unwrap_or_default();
        self.inertia = best_inertia;
        self.n_iter = best_n_iter;
        self.fitted = true;
        Ok(())
    }

    /// Run a single K-Means pass with the given seed.
    #[allow(clippy::type_complexity)]
    fn run_once(
        &self,
        rows: &[Vec<f64>],
        n: usize,
        m: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<usize>, f64, usize) {
        let mut centroids = kmeans_plus_plus(rows, self.k, seed);
        let mut labels = vec![0usize; n];
        let mut prev_inertia = f64::INFINITY;
        let mut final_inertia = f64::INFINITY;
        let mut final_n_iter = 0;
        let use_par = n * self.k >= KMEANS_PAR_THRESHOLD;

        for iter in 0..self.max_iter {
            // Assignment step.
            let inertia;
            if use_par {
                let results: Vec<(usize, f64)> = rows
                    .par_iter()
                    .map(|row| {
                        let mut best_dist = f64::INFINITY;
                        let mut best_c = 0;
                        for (c, centroid) in centroids.iter().enumerate() {
                            let d = euclidean_sq(row, centroid);
                            if d < best_dist {
                                best_dist = d;
                                best_c = c;
                            }
                        }
                        (best_c, best_dist)
                    })
                    .collect();
                inertia = results.iter().map(|(_, d)| d).sum();
                for (i, (c, _)) in results.into_iter().enumerate() {
                    labels[i] = c;
                }
            } else {
                let mut seq_inertia = 0.0;
                for (i, row) in rows.iter().enumerate() {
                    let mut best_dist = f64::INFINITY;
                    let mut best_c = 0;
                    for (c, centroid) in centroids.iter().enumerate() {
                        let d = euclidean_sq(row, centroid);
                        if d < best_dist {
                            best_dist = d;
                            best_c = c;
                        }
                    }
                    labels[i] = best_c;
                    seq_inertia += best_dist;
                }
                inertia = seq_inertia;
            }

            // Update step.
            let mut new_centroids = vec![vec![0.0; m]; self.k];
            let mut counts = vec![0usize; self.k];

            for (i, row) in rows.iter().enumerate() {
                let c = labels[i];
                counts[c] += 1;
                for (j, &val) in row.iter().enumerate() {
                    new_centroids[c][j] += val;
                }
            }

            // Normalize non-empty centroids.
            for c in 0..self.k {
                if counts[c] > 0 {
                    for val in &mut new_centroids[c] {
                        *val /= counts[c] as f64;
                    }
                }
            }

            // Reinitialize empty centroids: pick the data point farthest
            // from its nearest occupied centroid (sklearn's approach).
            for c in 0..self.k {
                if counts[c] == 0 {
                    let mut max_dist = f64::NEG_INFINITY;
                    let mut best_idx = 0;
                    for (i, row) in rows.iter().enumerate() {
                        let min_dist = new_centroids
                            .iter()
                            .enumerate()
                            .filter(|&(ci, _)| ci != c && (counts[ci] > 0 || ci < c))
                            .map(|(_, cen)| euclidean_sq(row, cen))
                            .fold(f64::INFINITY, f64::min);
                        if min_dist > max_dist {
                            max_dist = min_dist;
                            best_idx = i;
                        }
                    }
                    new_centroids[c] = rows[best_idx].clone();
                }
            }

            // Check convergence.
            let shift: f64 = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| euclidean_sq(old, new))
                .sum();

            centroids = new_centroids;
            final_n_iter = iter + 1;
            final_inertia = inertia;

            if (prev_inertia - inertia).abs() < self.tolerance || shift < self.tolerance {
                break;
            }
            prev_inertia = inertia;
        }

        (centroids, labels, final_inertia, final_n_iter)
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

    /// Transform data into cluster-distance space.
    ///
    /// Returns a `n_samples × k` matrix where each value is the Euclidean
    /// distance from the sample to each centroid.
    ///
    /// # Example
    ///
    /// ```
    /// # use scry_learn::cluster::KMeans;
    /// # use scry_learn::dataset::Dataset;
    /// # let data = Dataset::new(
    /// #     vec![vec![0.0, 10.0], vec![0.0, 10.0]],
    /// #     vec![0.0; 2], vec!["x".into(), "y".into()], "l",
    /// # );
    /// # let mut km = KMeans::new(2).n_init(1).seed(42);
    /// # km.fit(&data).unwrap();
    /// let distances = km.transform(&[vec![5.0, 5.0]]).unwrap();
    /// assert_eq!(distances[0].len(), 2); // one distance per centroid
    /// ```
    pub fn transform(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(ScryLearnError::NotFitted);
        }
        Ok(features
            .iter()
            .map(|row| {
                self.centroids
                    .iter()
                    .map(|c| euclidean_sq(row, c).sqrt())
                    .collect()
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

    /// Number of iterations to converge.
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

/// K-means++ initialization: select initial centroids to be spread apart.
pub(crate) fn kmeans_plus_plus(rows: &[Vec<f64>], k: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = crate::rng::FastRng::new(seed);
    let n = rows.len();
    let mut centroids = Vec::with_capacity(k);

    // Pick first centroid randomly.
    centroids.push(rows[rng.usize(0..n)].clone());

    for _ in 1..k {
        // Compute distances to nearest centroid.
        let mut dists: Vec<f64> = rows
            .iter()
            .map(|row| {
                centroids
                    .iter()
                    .map(|c| euclidean_sq(row, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();

        // Weighted random selection proportional to D².
        let total: f64 = dists.iter().sum();
        if total < 1e-12 {
            centroids.push(rows[rng.usize(0..n)].clone());
            continue;
        }
        for d in &mut dists {
            *d /= total;
        }

        let r = rng.f64();
        let mut cumsum = 0.0;
        let mut selected = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= r {
                selected = i;
                break;
            }
        }
        centroids.push(rows[selected].clone());
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_two_blobs() {
        // Two well-separated clusters.
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        let mut target = Vec::new();
        for i in 0..30 {
            f1.push(i as f64 % 3.0);
            f2.push(i as f64 % 3.0);
            target.push(0.0);
        }
        for i in 0..30 {
            f1.push(100.0 + i as f64 % 3.0);
            f2.push(100.0 + i as f64 % 3.0);
            target.push(1.0);
        }

        let data = Dataset::new(vec![f1, f2], target, vec!["x".into(), "y".into()], "label");

        let mut km = KMeans::new(2).seed(42).n_init(1);
        km.fit(&data).unwrap();

        // All points in the same blob should have the same label.
        let labels = km.labels();
        let first_label = labels[0];
        assert!(labels[..30].iter().all(|&l| l == first_label));
        assert!(labels[30..].iter().all(|&l| l != first_label));
    }

    #[test]
    fn test_kmeans_predict() {
        let data = Dataset::new(
            vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]],
            vec![0.0; 4],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut km = KMeans::new(2).seed(42).n_init(1);
        km.fit(&data).unwrap();

        let pred = km.predict(&[vec![1.0, 1.0], vec![9.0, 9.0]]).unwrap();
        assert_ne!(
            pred[0], pred[1],
            "nearby and far points should be in different clusters"
        );
    }

    #[test]
    fn test_kmeans_n_init_improves_inertia() {
        // n_init=10 should produce inertia ≤ n_init=1.
        let mut rng = crate::rng::FastRng::new(7);
        let n = 100;
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        for _ in 0..n / 2 {
            f1.push(rng.f64() * 5.0);
            f2.push(rng.f64() * 5.0);
        }
        for _ in 0..n / 2 {
            f1.push(20.0 + rng.f64() * 5.0);
            f2.push(20.0 + rng.f64() * 5.0);
        }
        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; n],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut km1 = KMeans::new(3).seed(7).n_init(1);
        km1.fit(&data).unwrap();
        let inertia1 = km1.inertia();

        let mut km10 = KMeans::new(3).seed(7).n_init(10);
        km10.fit(&data).unwrap();
        let inertia10 = km10.inertia();

        assert!(
            inertia10 <= inertia1 + 1e-6,
            "n_init=10 inertia ({inertia10:.4}) should be ≤ n_init=1 ({inertia1:.4})"
        );
    }

    #[test]
    fn test_kmeans_empty_cluster_reinit() {
        // Pathological case: 3 clusters requested but data has 2 clear blobs.
        // With bad initialization, one centroid can get zero assigned points.
        // After fix, empty centroids should be reinitialized, not left at [0,0].
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        for _ in 0..50 {
            f1.push(0.0);
            f2.push(0.0);
        }
        for _ in 0..50 {
            f1.push(100.0);
            f2.push(100.0);
        }
        let data = Dataset::new(vec![f1, f2], vec![0.0; 100], vec!["x".into(), "y".into()], "l");

        let mut km = KMeans::new(3).seed(42).n_init(1);
        km.fit(&data).unwrap();

        // No centroid should be at the origin [0,0] unless a cluster actually lives there.
        // At least one centroid should be near (100,100) and at least one near (0,0).
        let centroids = km.centroids();
        assert_eq!(centroids.len(), 3);
        let has_near_origin = centroids.iter().any(|c| c[0] < 50.0 && c[1] < 50.0);
        let has_near_far = centroids.iter().any(|c| c[0] > 50.0 && c[1] > 50.0);
        assert!(has_near_origin, "should have centroid near (0,0)");
        assert!(has_near_far, "should have centroid near (100,100)");

        // All 100 points should be assigned to some cluster.
        assert_eq!(km.labels().len(), 100);
    }

    #[test]
    fn test_kmeans_transform_shape() {
        let data = Dataset::new(
            vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]],
            vec![0.0; 4],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut km = KMeans::new(2).seed(42).n_init(1);
        km.fit(&data).unwrap();

        let dists = km.transform(&[vec![5.0, 5.0], vec![0.0, 0.0]]).unwrap();
        assert_eq!(dists.len(), 2, "should have 2 samples");
        assert_eq!(
            dists[0].len(),
            2,
            "should have distance to each of 2 centroids"
        );
        // All distances should be non-negative.
        for row in &dists {
            for &d in row {
                assert!(d >= 0.0, "distance should be non-negative");
            }
        }
    }
}
