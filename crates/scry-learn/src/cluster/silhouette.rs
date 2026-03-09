// SPDX-License-Identifier: MIT OR Apache-2.0
//! Silhouette scoring for clustering quality.
//!
//! The silhouette coefficient measures how similar a sample is to its own cluster
//! compared to other clusters. Values range from -1 to +1, where higher is better.
//!
//! # Example
//!
//! ```
//! use scry_learn::cluster::silhouette_score;
//!
//! let features = vec![
//!     vec![0.0, 0.0], vec![1.0, 0.0], vec![0.5, 0.0], // cluster 0
//!     vec![10.0, 10.0], vec![11.0, 10.0], vec![10.5, 10.0], // cluster 1
//! ];
//! let labels = vec![0, 0, 0, 1, 1, 1];
//! let score = silhouette_score(&features, &labels);
//! assert!(score > 0.7);
//! ```

use crate::distance::euclidean_sq;

/// Compute the mean silhouette coefficient across all samples.
///
/// # Arguments
/// - `features` — row-major feature matrix (`n_samples` rows).
/// - `labels` — cluster assignment for each sample.
///
/// # Returns
/// Mean silhouette score in `[-1, 1]`. Higher means better-defined clusters.
///
/// # Panics
/// Panics if `features` and `labels` have different lengths.
pub fn silhouette_score(features: &[Vec<f64>], labels: &[usize]) -> f64 {
    let scores = silhouette_samples(features, labels);
    if scores.is_empty() {
        return 0.0;
    }
    scores.iter().sum::<f64>() / scores.len() as f64
}

/// Compute per-sample silhouette coefficients.
///
/// For each sample *i*:
/// - `a(i)` = mean Euclidean distance to all other samples in the same cluster.
/// - `b(i)` = mean Euclidean distance to the nearest other cluster.
/// - `s(i) = (b(i) - a(i)) / max(a(i), b(i))`.
///
/// Samples in clusters of size 1 get a silhouette of 0.
pub fn silhouette_samples(features: &[Vec<f64>], labels: &[usize]) -> Vec<f64> {
    assert_eq!(
        features.len(),
        labels.len(),
        "features and labels must have the same length"
    );

    let n = features.len();
    if n <= 1 {
        return vec![0.0; n];
    }

    // Find unique clusters.
    let max_label = labels.iter().copied().max().unwrap_or(0);
    let n_clusters = max_label + 1;

    if n_clusters <= 1 {
        return vec![0.0; n];
    }

    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let my_label = labels[i];

        // Compute mean distance to each cluster.
        let mut cluster_dist_sum = vec![0.0_f64; n_clusters];
        let mut cluster_count = vec![0usize; n_clusters];

        for j in 0..n {
            if i == j {
                continue;
            }
            let d = euclidean_sq(&features[i], &features[j]).sqrt();
            cluster_dist_sum[labels[j]] += d;
            cluster_count[labels[j]] += 1;
        }

        // a(i): mean distance within own cluster.
        let a = if cluster_count[my_label] > 0 {
            cluster_dist_sum[my_label] / cluster_count[my_label] as f64
        } else {
            0.0 // singleton cluster
        };

        // b(i): min mean distance to any other cluster.
        let mut b = f64::INFINITY;
        for c in 0..n_clusters {
            if c == my_label || cluster_count[c] == 0 {
                continue;
            }
            let mean_d = cluster_dist_sum[c] / cluster_count[c] as f64;
            if mean_d < b {
                b = mean_d;
            }
        }

        if b == f64::INFINITY {
            // Only one cluster with points — silhouette is 0.
            scores.push(0.0);
        } else {
            let max_ab = a.max(b);
            if max_ab < 1e-15 {
                scores.push(0.0);
            } else {
                scores.push((b - a) / max_ab);
            }
        }
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silhouette_perfect_clusters() {
        // Two well-separated blobs.
        let features = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.5, 0.5],
            vec![100.0, 100.0],
            vec![101.0, 100.0],
            vec![100.5, 100.5],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let score = silhouette_score(&features, &labels);
        assert!(
            score > 0.90,
            "well-separated clusters should have silhouette > 0.90, got {score:.4}"
        );
    }

    #[test]
    fn test_silhouette_single_cluster() {
        let features = vec![vec![1.0], vec![2.0], vec![3.0]];
        let labels = vec![0, 0, 0];
        let score = silhouette_score(&features, &labels);
        assert!(
            score.abs() < 1e-10,
            "single cluster should have silhouette 0, got {score}"
        );
    }

    #[test]
    fn test_silhouette_samples_length() {
        let features = vec![vec![0.0], vec![10.0], vec![0.5], vec![10.5]];
        let labels = vec![0, 1, 0, 1];
        let scores = silhouette_samples(&features, &labels);
        assert_eq!(scores.len(), 4);
        for &s in &scores {
            assert!(
                (-1.0..=1.0).contains(&s),
                "silhouette must be in [-1, 1], got {s}"
            );
        }
    }
}
