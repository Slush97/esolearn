// SPDX-License-Identifier: MIT OR Apache-2.0
//! Agglomerative (hierarchical) clustering.
//!
//! Bottom-up clustering that starts with each sample as its own cluster
//! and merges the closest pair until `n_clusters` remain.
//!
//! # Example
//!
//! ```
//! use scry_learn::cluster::AgglomerativeClustering;
//! use scry_learn::dataset::Dataset;
//!
//! let data = Dataset::new(
//!     vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]],
//!     vec![0.0; 4],
//!     vec!["x".into(), "y".into()],
//!     "label",
//! );
//!
//! let mut model = AgglomerativeClustering::new(2);
//! model.fit(&data).unwrap();
//! assert_eq!(model.labels().len(), 4);
//! ```

use crate::dataset::Dataset;
use crate::distance::euclidean_sq;
use crate::error::{Result, ScryLearnError};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Inter-cluster distance measure for agglomerative clustering.
///
/// Controls how the distance between two clusters is computed when
/// deciding which pair to merge next.
///
/// # Example
///
/// ```
/// use scry_learn::cluster::{AgglomerativeClustering, Linkage};
///
/// let model = AgglomerativeClustering::new(3).linkage(Linkage::Ward);
/// ```
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Linkage {
    /// Minimum distance between any pair of points across two clusters.
    Single,
    /// Maximum distance between any pair of points across two clusters.
    Complete,
    /// Mean distance between all pairs of points across two clusters.
    Average,
    /// Minimize the total within-cluster variance after merging (most common).
    #[default]
    Ward,
}

/// A single merge event in the dendrogram.
///
/// Records which two clusters were merged and at what distance.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MergeStep {
    /// Index of the first cluster merged.
    pub cluster_a: usize,
    /// Index of the second cluster merged.
    pub cluster_b: usize,
    /// Distance at which the merge occurred.
    pub distance: f64,
    /// Number of samples in the merged cluster.
    pub size: usize,
}

/// Agglomerative (hierarchical) clustering.
///
/// Starts with each sample as its own cluster and iteratively merges the
/// closest pair until `n_clusters` remain. Supports four linkage criteria.
///
/// # Example
///
/// ```
/// use scry_learn::cluster::AgglomerativeClustering;
/// use scry_learn::dataset::Dataset;
///
/// let data = Dataset::new(
///     vec![vec![0.0, 0.0, 10.0, 10.0], vec![0.0, 0.0, 10.0, 10.0]],
///     vec![0.0; 4],
///     vec!["x".into(), "y".into()],
///     "label",
/// );
///
/// let mut model = AgglomerativeClustering::new(2);
/// model.fit(&data).unwrap();
/// assert_eq!(model.labels().len(), 4);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct AgglomerativeClustering {
    n_clusters: usize,
    linkage: Linkage,
    labels: Vec<usize>,
    children: Vec<MergeStep>,
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl AgglomerativeClustering {
    /// Create a new agglomerative clustering model.
    ///
    /// # Arguments
    ///
    /// * `n_clusters` — target number of clusters.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Ward,
            labels: Vec::new(),
            children: Vec::new(),
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the linkage criterion.
    pub fn linkage(mut self, l: Linkage) -> Self {
        self.linkage = l;
        self
    }

    /// Fit the model on a dataset.
    ///
    /// Uses the features and ignores the target column. Computes the
    /// full O(n²) pairwise distance matrix, then greedily merges the
    /// closest cluster pair using a priority queue.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if self.n_clusters == 0 || self.n_clusters > n {
            return Err(ScryLearnError::InvalidParameter(format!(
                "n_clusters must be between 1 and n_samples ({}), got {}",
                n, self.n_clusters
            )));
        }

        let rows = data.feature_matrix();
        let n_features = data.n_features();

        // Compute pairwise squared-Euclidean distance matrix (upper triangle).
        let mut dist = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = euclidean_sq(&rows[i], &rows[j]);
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }

        // Track which cluster each original sample belongs to.
        // cluster_id[i] = current cluster for original sample i.
        let mut cluster_of = (0..n).collect::<Vec<usize>>();

        // Members of each cluster (indexed by cluster id).
        let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        // Centroids for Ward linkage.
        let mut centroids: Vec<Vec<f64>> = rows.clone();

        // Priority queue: (neg_distance, cluster_a, cluster_b).
        // We use a max-heap with negated distances to get a min-heap.
        let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();

        // Populate initial distances.
        for i in 0..n {
            for j in (i + 1)..n {
                let d = self.linkage_distance(i, j, &dist, &members, &centroids, n_features);
                heap.push(MergeCandidate {
                    neg_dist: -d,
                    a: i,
                    b: j,
                });
            }
        }

        let mut active: Vec<bool> = vec![true; n];
        let mut n_active = n;
        let mut next_cluster_id = n; // new clusters get IDs >= n
        let mut children = Vec::new();

        while n_active > self.n_clusters {
            // Pop the closest pair (skip stale entries).
            let merge = loop {
                let Some(candidate) = heap.pop() else {
                    break None;
                };
                if active[candidate.a] && active[candidate.b] {
                    break Some(candidate);
                }
            };

            let Some(merge) = merge else { break };

            let ca = merge.a;
            let cb = merge.b;
            let merge_dist = -merge.neg_dist;

            // Create a new merged cluster.
            let new_id = next_cluster_id;
            next_cluster_id += 1;

            // Merge members.
            let mut new_members = std::mem::take(&mut members[ca]);
            new_members.extend(std::mem::take(&mut members[cb]));
            let new_size = new_members.len();

            children.push(MergeStep {
                cluster_a: ca,
                cluster_b: cb,
                distance: merge_dist.sqrt(),
                size: new_size,
            });

            // Compute new centroid (for Ward).
            let new_centroid = if matches!(self.linkage, Linkage::Ward) {
                let mut c = vec![0.0; n_features];
                for &idx in &new_members {
                    for (j, &v) in rows[idx].iter().enumerate() {
                        c[j] += v;
                    }
                }
                for v in &mut c {
                    *v /= new_size as f64;
                }
                c
            } else {
                Vec::new()
            };

            // Deactivate old clusters.
            active[ca] = false;
            active[cb] = false;

            // Expand storage for the new cluster.
            while active.len() <= new_id {
                active.push(false);
                members.push(Vec::new());
                centroids.push(Vec::new());
                // Expand dist matrix
                for row in &mut dist {
                    row.push(f64::INFINITY);
                }
                dist.push(vec![f64::INFINITY; dist[0].len()]);
            }

            active[new_id] = true;
            members[new_id] = new_members;
            centroids[new_id] = new_centroid;

            // Compute distances from new cluster to all remaining active clusters.
            for other in 0..active.len() {
                if !active[other] || other == new_id {
                    continue;
                }
                let d = self.compute_merged_distance(
                    ca, cb, other, &dist, &members, &centroids, n_features, &rows,
                );
                dist[new_id][other] = d;
                dist[other][new_id] = d;
                heap.push(MergeCandidate {
                    neg_dist: -d,
                    a: new_id.min(other),
                    b: new_id.max(other),
                });
            }

            // Update cluster_of for merged members.
            for &idx in &members[new_id] {
                cluster_of[idx] = new_id;
            }

            n_active -= 1;
        }

        // Assign final labels 0..n_clusters-1.
        let active_ids: Vec<usize> = active
            .iter()
            .enumerate()
            .filter(|(_, &a)| a)
            .map(|(i, _)| i)
            .collect();

        let mut labels = vec![0usize; n];
        for (label, &cid) in active_ids.iter().enumerate() {
            for &sample in &members[cid] {
                labels[sample] = label;
            }
        }

        self.labels = labels;
        self.children = children;
        self.fitted = true;
        Ok(())
    }

    /// Compute linkage distance between two clusters.
    fn linkage_distance(
        &self,
        a: usize,
        b: usize,
        dist: &[Vec<f64>],
        members: &[Vec<usize>],
        centroids: &[Vec<f64>],
        _n_features: usize,
    ) -> f64 {
        match self.linkage {
            Linkage::Single => {
                let mut min_d = f64::INFINITY;
                for &i in &members[a] {
                    for &j in &members[b] {
                        let d = dist[i][j];
                        if d < min_d {
                            min_d = d;
                        }
                    }
                }
                min_d
            }
            Linkage::Complete => {
                let mut max_d = 0.0_f64;
                for &i in &members[a] {
                    for &j in &members[b] {
                        let d = dist[i][j];
                        if d > max_d {
                            max_d = d;
                        }
                    }
                }
                max_d
            }
            Linkage::Average => {
                let mut sum = 0.0;
                let count = members[a].len() * members[b].len();
                for &i in &members[a] {
                    for &j in &members[b] {
                        sum += dist[i][j];
                    }
                }
                if count > 0 {
                    sum / count as f64
                } else {
                    0.0
                }
            }
            Linkage::Ward => {
                // Ward distance = size_a * size_b / (size_a + size_b) * ||c_a - c_b||²
                let sa = members[a].len() as f64;
                let sb = members[b].len() as f64;
                let d: f64 = centroids[a]
                    .iter()
                    .zip(centroids[b].iter())
                    .map(|(ca, cb)| (ca - cb).powi(2))
                    .sum();
                sa * sb / (sa + sb) * d
            }
        }
    }

    /// Compute distance from a newly merged cluster (ca+cb) to another cluster.
    #[allow(clippy::too_many_arguments)]
    fn compute_merged_distance(
        &self,
        ca: usize,
        cb: usize,
        other: usize,
        dist: &[Vec<f64>],
        members: &[Vec<usize>],
        _centroids: &[Vec<f64>],
        _n_features: usize,
        _rows: &[Vec<f64>],
    ) -> f64 {
        match self.linkage {
            Linkage::Single => dist[ca][other].min(dist[cb][other]),
            Linkage::Complete => dist[ca][other].max(dist[cb][other]),
            Linkage::Average => {
                let na = members[ca].len() as f64;
                let nb = members[cb].len() as f64;
                (na * dist[ca][other] + nb * dist[cb][other]) / (na + nb)
            }
            Linkage::Ward => {
                // Lance-Williams formula for Ward's method
                let na = members[ca].len() as f64;
                let nb = members[cb].len() as f64;
                let nc = members[other].len() as f64;
                let total = na + nb + nc;
                ((na + nc) * dist[ca][other] + (nb + nc) * dist[cb][other] - nc * dist[ca][cb])
                    / total
            }
        }
    }

    /// Get cluster labels for training data.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Number of clusters.
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Merge history (dendrogram data).
    ///
    /// Each entry records which two clusters were merged and at what distance.
    pub fn children(&self) -> &[MergeStep] {
        &self.children
    }
}

/// Priority queue entry for cluster merging.
#[derive(Clone, Copy)]
struct MergeCandidate {
    neg_dist: f64, // negated so BinaryHeap (max-heap) gives us the minimum
    a: usize,
    b: usize,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.neg_dist == other.neg_dist
    }
}

impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.neg_dist
            .partial_cmp(&other.neg_dist)
            .unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agglomerative_three_clusters() {
        // Three well-separated clusters.
        let mut rng = crate::rng::FastRng::new(0);
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        for _ in 0..10 {
            f1.push(rng.f64() * 2.0);
            f2.push(rng.f64() * 2.0);
        }
        for _ in 0..10 {
            f1.push(50.0 + rng.f64() * 2.0);
            f2.push(50.0 + rng.f64() * 2.0);
        }
        for _ in 0..10 {
            f1.push(100.0 + rng.f64() * 2.0);
            f2.push(100.0 + rng.f64() * 2.0);
        }

        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; 30],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut model = AgglomerativeClustering::new(3);
        model.fit(&data).unwrap();

        let labels = model.labels();
        assert_eq!(labels.len(), 30);

        // All points in the same group should have the same label.
        let label_a = labels[0];
        assert!(
            labels[..10].iter().all(|&l| l == label_a),
            "Cluster A inconsistent"
        );

        let label_b = labels[10];
        assert!(
            labels[10..20].iter().all(|&l| l == label_b),
            "Cluster B inconsistent"
        );

        let label_c = labels[20];
        assert!(
            labels[20..].iter().all(|&l| l == label_c),
            "Cluster C inconsistent"
        );

        // All three labels should be distinct.
        assert_ne!(label_a, label_b);
        assert_ne!(label_a, label_c);
        assert_ne!(label_b, label_c);
    }

    #[test]
    fn test_agglomerative_linkage_variants() {
        let data = Dataset::new(
            vec![vec![0.0, 1.0, 5.0, 6.0], vec![0.0, 0.0, 0.0, 0.0]],
            vec![0.0; 4],
            vec!["x".into(), "y".into()],
            "label",
        );

        for linkage in [
            Linkage::Single,
            Linkage::Complete,
            Linkage::Average,
            Linkage::Ward,
        ] {
            let mut model = AgglomerativeClustering::new(2).linkage(linkage);
            model.fit(&data).unwrap();
            assert_eq!(model.labels().len(), 4, "Failed for {linkage:?}");
        }
    }

    #[test]
    fn test_agglomerative_ward_vs_single() {
        // Ward and Single should produce different merge histories
        // on data where they disagree.
        let data = Dataset::new(
            vec![
                vec![0.0, 1.0, 3.0, 10.0, 11.0, 13.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            vec![0.0; 6],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut ward = AgglomerativeClustering::new(2).linkage(Linkage::Ward);
        ward.fit(&data).unwrap();

        let mut single = AgglomerativeClustering::new(2).linkage(Linkage::Single);
        single.fit(&data).unwrap();

        // Both should output valid labels of length 6.
        assert_eq!(ward.labels().len(), 6);
        assert_eq!(single.labels().len(), 6);

        // The merge histories should have the right number of steps.
        assert_eq!(ward.children().len(), 4); // 6 - 2 merges
        assert_eq!(single.children().len(), 4);
    }

    #[test]
    fn test_agglomerative_single_cluster() {
        let data = Dataset::new(
            vec![vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]],
            vec![0.0; 3],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut model = AgglomerativeClustering::new(1);
        model.fit(&data).unwrap();
        assert!(
            model.labels().iter().all(|&l| l == 0),
            "All should be cluster 0"
        );
    }
}
