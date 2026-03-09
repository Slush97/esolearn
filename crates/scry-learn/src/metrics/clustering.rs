// SPDX-License-Identifier: MIT OR Apache-2.0
//! Clustering evaluation metrics.
//!
//! These metrics compare cluster assignments against ground truth
//! (supervised evaluation) or assess cluster quality from the data alone
//! (unsupervised evaluation).

use std::collections::HashMap;

/// Map f64 labels to contiguous 0-based indices.
fn label_to_index(labels: &[f64]) -> (Vec<usize>, usize) {
    let mut map: HashMap<i64, usize> = HashMap::new();
    let mut next = 0usize;
    let indices: Vec<usize> = labels
        .iter()
        .map(|&v| {
            let key = v as i64;
            *map.entry(key).or_insert_with(|| {
                let idx = next;
                next += 1;
                idx
            })
        })
        .collect();
    (indices, next)
}

/// Adjusted Rand Index — similarity between two clusterings, adjusted for chance.
///
/// Returns a value in `[-1, 1]` where:
/// - 1.0 means perfect match
/// - 0.0 means random agreement
/// - negative values indicate worse than random
pub fn adjusted_rand_index(labels_true: &[f64], labels_pred: &[f64]) -> f64 {
    let n = labels_true.len();
    if n == 0 {
        return 0.0;
    }

    let (true_idx, n_true) = label_to_index(labels_true);
    let (pred_idx, n_pred) = label_to_index(labels_pred);

    // Build contingency matrix
    let mut contingency = vec![vec![0i64; n_pred]; n_true];
    for i in 0..n {
        contingency[true_idx[i]][pred_idx[i]] += 1;
    }

    // Row and column sums
    let a: Vec<i64> = contingency.iter().map(|row| row.iter().sum()).collect();
    let b: Vec<i64> = (0..n_pred)
        .map(|j| contingency.iter().map(|row| row[j]).sum())
        .collect();

    // Compute sums of C(n, 2) = n*(n-1)/2
    let comb2 = |x: i64| -> i64 { x * (x - 1) / 2 };

    let sum_comb_c: i64 = contingency
        .iter()
        .flat_map(|row| row.iter())
        .map(|&x| comb2(x))
        .sum();
    let sum_comb_a: i64 = a.iter().map(|&x| comb2(x)).sum();
    let sum_comb_b: i64 = b.iter().map(|&x| comb2(x)).sum();
    let n_i64 = i64::try_from(n).expect("sample count overflows i64");
    let comb_n = comb2(n_i64);

    if comb_n == 0 {
        return 0.0;
    }

    let expected = sum_comb_a as f64 * sum_comb_b as f64 / comb_n as f64;
    let max_index = (sum_comb_a as f64 + sum_comb_b as f64) / 2.0;

    if (max_index - expected).abs() < 1e-15 {
        return if (sum_comb_c as f64 - expected).abs() < 1e-15 {
            1.0
        } else {
            0.0
        };
    }

    (sum_comb_c as f64 - expected) / (max_index - expected)
}

/// Calinski-Harabasz Index — ratio of between-cluster to within-cluster dispersion.
///
/// Higher values indicate better-defined clusters. Features are provided
/// in column-major format (one `Vec<f64>` per feature).
pub fn calinski_harabasz_score(features: &[Vec<f64>], labels: &[f64]) -> f64 {
    let n = labels.len();
    if n == 0 || features.is_empty() {
        return 0.0;
    }
    let n_features = features.len();

    let (label_idx, k) = label_to_index(labels);

    if k <= 1 || k >= n {
        return 0.0;
    }

    // Global centroid
    let global_centroid: Vec<f64> = features
        .iter()
        .map(|col| col.iter().sum::<f64>() / n as f64)
        .collect();

    // Per-cluster centroids and counts
    let mut cluster_sums = vec![vec![0.0_f64; n_features]; k];
    let mut cluster_counts = vec![0usize; k];

    for i in 0..n {
        let ci = label_idx[i];
        cluster_counts[ci] += 1;
        for f in 0..n_features {
            cluster_sums[ci][f] += features[f][i];
        }
    }

    let cluster_centroids: Vec<Vec<f64>> = cluster_sums
        .iter()
        .zip(cluster_counts.iter())
        .map(|(sums, &cnt)| {
            if cnt == 0 {
                vec![0.0; n_features]
            } else {
                sums.iter().map(|s| s / cnt as f64).collect()
            }
        })
        .collect();

    // Between-cluster dispersion (B)
    let mut bgss = 0.0;
    for ci in 0..k {
        let cnt = cluster_counts[ci] as f64;
        let dist_sq: f64 = cluster_centroids[ci]
            .iter()
            .zip(global_centroid.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        bgss += cnt * dist_sq;
    }

    // Within-cluster dispersion (W)
    let mut wgss = 0.0;
    for i in 0..n {
        let ci = label_idx[i];
        for f in 0..n_features {
            wgss += (features[f][i] - cluster_centroids[ci][f]).powi(2);
        }
    }

    if wgss < 1e-15 {
        return 0.0;
    }

    (bgss / (k as f64 - 1.0)) / (wgss / (n as f64 - k as f64))
}

/// Davies-Bouldin Index — average worst-case similarity ratio between clusters.
///
/// Lower values indicate better separation. Features are provided
/// in column-major format (one `Vec<f64>` per feature).
pub fn davies_bouldin_score(features: &[Vec<f64>], labels: &[f64]) -> f64 {
    let n = labels.len();
    if n == 0 || features.is_empty() {
        return 0.0;
    }
    let n_features = features.len();

    let (label_idx, k) = label_to_index(labels);

    if k <= 1 {
        return 0.0;
    }

    // Cluster centroids
    let mut cluster_sums = vec![vec![0.0_f64; n_features]; k];
    let mut cluster_counts = vec![0usize; k];

    for i in 0..n {
        let ci = label_idx[i];
        cluster_counts[ci] += 1;
        for f in 0..n_features {
            cluster_sums[ci][f] += features[f][i];
        }
    }

    let centroids: Vec<Vec<f64>> = cluster_sums
        .iter()
        .zip(cluster_counts.iter())
        .map(|(sums, &cnt)| {
            if cnt == 0 {
                vec![0.0; n_features]
            } else {
                sums.iter().map(|s| s / cnt as f64).collect()
            }
        })
        .collect();

    // Average intra-cluster distance (scatter) for each cluster
    let mut scatter = vec![0.0_f64; k];
    for i in 0..n {
        let ci = label_idx[i];
        let dist: f64 = (0..n_features)
            .map(|f| (features[f][i] - centroids[ci][f]).powi(2))
            .sum::<f64>()
            .sqrt();
        scatter[ci] += dist;
    }
    for ci in 0..k {
        if cluster_counts[ci] > 0 {
            scatter[ci] /= cluster_counts[ci] as f64;
        }
    }

    // DB = mean of max R_ij for each cluster i
    let mut db_sum = 0.0;
    for i in 0..k {
        let mut max_r = f64::NEG_INFINITY;
        for j in 0..k {
            if i == j {
                continue;
            }
            let d_ij: f64 = centroids[i]
                .iter()
                .zip(centroids[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let r = if d_ij < 1e-15 {
                0.0
            } else {
                (scatter[i] + scatter[j]) / d_ij
            };
            if r > max_r {
                max_r = r;
            }
        }
        db_sum += max_r;
    }

    db_sum / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ari_identical() {
        let labels = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let ari = adjusted_rand_index(&labels, &labels);
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "identical ARI should be 1.0, got {ari}"
        );
    }

    #[test]
    fn test_ari_permuted() {
        // Same clustering, different label names → ARI = 1.0
        let true_labels = vec![0.0, 0.0, 1.0, 1.0];
        let pred_labels = vec![5.0, 5.0, 3.0, 3.0];
        let ari = adjusted_rand_index(&true_labels, &pred_labels);
        assert!(
            (ari - 1.0).abs() < 1e-10,
            "permuted ARI should be 1.0, got {ari}"
        );
    }

    #[test]
    fn test_ari_random() {
        // Completely mismatched → ARI should be near 0 or negative
        let true_labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let pred_labels = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let ari = adjusted_rand_index(&true_labels, &pred_labels);
        assert!(ari < 0.5, "random ARI should be low, got {ari}");
    }

    #[test]
    fn test_calinski_harabasz_well_separated() {
        // Two well-separated clusters in 2D
        let f0 = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2];
        let f1 = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let ch = calinski_harabasz_score(&[f0, f1], &labels);
        assert!(ch > 100.0, "well-separated CH should be high, got {ch}");
    }

    #[test]
    fn test_davies_bouldin_well_separated() {
        // Two well-separated clusters → DB should be low (close to 0)
        let f0 = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2];
        let f1 = vec![0.0, 0.1, 0.2, 10.0, 10.1, 10.2];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let db = davies_bouldin_score(&[f0, f1], &labels);
        assert!(db < 0.5, "well-separated DB should be low, got {db}");
    }

    #[test]
    fn test_davies_bouldin_overlapping() {
        // Overlapping clusters → DB should be higher
        let f0 = vec![0.0, 1.0, 2.0, 1.5, 2.5, 3.0];
        let f1 = vec![0.0, 1.0, 2.0, 1.5, 2.5, 3.0];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let db = davies_bouldin_score(&[f0, f1], &labels);
        // DB should be significantly higher for overlapping clusters
        assert!(db > 0.3, "overlapping DB should be non-trivial, got {db}");
    }
}
