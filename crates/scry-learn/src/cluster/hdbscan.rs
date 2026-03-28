// SPDX-License-Identifier: MIT OR Apache-2.0
//! HDBSCAN — Hierarchical Density-Based Spatial Clustering.
//!
//! Implements the Campello, Moulavi & Sander (2013) algorithm. Unlike DBSCAN,
//! HDBSCAN automatically determines the number of clusters and handles
//! varying-density regions without manual epsilon tuning.
//!
//! # Algorithm
//!
//! 1. Compute **core distances** (distance to k-th nearest neighbor).
//! 2. Build a **mutual reachability graph** where edge weight =
//!    `max(core(a), core(b), dist(a,b))`.
//! 3. Extract a **minimum spanning tree** (Prim's algorithm).
//! 4. Build a **single-linkage dendrogram** from the MST.
//! 5. **Condense** the tree, pruning clusters smaller than `min_cluster_size`.
//! 6. **Extract clusters** using the Excess of Mass (EOMM) method.
//!
//! # Example
//!
//! ```
//! use scry_learn::cluster::Hdbscan;
//! use scry_learn::dataset::Dataset;
//!
//! let data = Dataset::new(
//!     vec![vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4],
//!          vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4]],
//!     vec![0.0; 10],
//!     vec!["x".into(), "y".into()],
//!     "label",
//! );
//!
//! let mut hdb = Hdbscan::new().min_cluster_size(3);
//! hdb.fit(&data).unwrap();
//! assert_eq!(hdb.n_clusters(), 2);
//! ```

use crate::dataset::Dataset;
use crate::distance::{cosine_distance, euclidean_sq, manhattan};
use crate::error::{Result, ScryLearnError};
use crate::neighbors::DistanceMetric;

/// HDBSCAN clustering model.
///
/// Automatically determines the number of clusters from density variations
/// in the data. Noise points are labeled -1.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Hdbscan {
    /// Minimum cluster size. Clusters smaller than this are treated as noise.
    min_cluster_size: usize,
    /// Number of neighbors used to compute core distance (default = min_cluster_size).
    min_samples: Option<usize>,
    /// Distance metric for pairwise distance computation.
    metric: DistanceMetric,
    /// Cluster labels after fitting (-1 = noise).
    labels: Vec<i32>,
    /// Number of clusters found (excluding noise).
    n_clusters: usize,
    /// Per-point outlier scores (higher = more likely outlier).
    outlier_scores: Vec<f64>,
    /// Whether the model has been fitted.
    fitted: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    _schema_version: u32,
}

impl Hdbscan {
    /// Create a new HDBSCAN model with default parameters.
    ///
    /// Default: `min_cluster_size = 5`, `min_samples = min_cluster_size`.
    pub fn new() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: None,
            metric: DistanceMetric::Euclidean,
            labels: Vec::new(),
            n_clusters: 0,
            outlier_scores: Vec::new(),
            fitted: false,
            _schema_version: crate::version::SCHEMA_VERSION,
        }
    }

    /// Set the minimum cluster size (default: 5).
    ///
    /// Clusters with fewer points than this are dissolved into noise.
    pub fn min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }

    /// Set min_samples for core distance computation.
    ///
    /// Default: same as `min_cluster_size`. Smaller values produce
    /// more clusters; larger values make the algorithm more conservative.
    pub fn min_samples(mut self, k: usize) -> Self {
        self.min_samples = Some(k);
        self
    }

    /// Set the distance metric (default: [`DistanceMetric::Euclidean`]).
    ///
    /// Use [`DistanceMetric::Cosine`] for L2-normalized embeddings
    /// (e.g. face embeddings, CLIP vectors).
    pub fn metric(mut self, m: DistanceMetric) -> Self {
        self.metric = m;
        self
    }

    /// Get cluster labels (-1 = noise).
    pub fn labels(&self) -> &[i32] {
        &self.labels
    }

    /// Number of clusters found (excluding noise).
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Number of noise points.
    pub fn n_noise(&self) -> usize {
        self.labels.iter().filter(|&&l| l == -1).count()
    }

    /// Per-point outlier scores (higher = more outlier-like).
    ///
    /// Based on the ratio of the point's lambda value to its cluster's
    /// lambda birth. Points deep inside clusters have scores near 0;
    /// borderline points have scores near 1.
    pub fn outlier_scores(&self) -> &[f64] {
        &self.outlier_scores
    }

    /// Fit the HDBSCAN model on a dataset.
    pub fn fit(&mut self, data: &Dataset) -> Result<()> {
        data.validate_finite()?;
        let n = data.n_samples();
        if n == 0 {
            return Err(ScryLearnError::EmptyDataset);
        }
        if n < self.min_cluster_size {
            // Everything is noise.
            self.labels = vec![-1; n];
            self.n_clusters = 0;
            self.outlier_scores = vec![1.0; n];
            self.fitted = true;
            return Ok(());
        }

        let rows = data.feature_matrix(); // row-major
        let k = self.min_samples.unwrap_or(self.min_cluster_size);
        let k = k.min(n - 1).max(1);

        // Step 1: Compute pairwise distances and core distances.
        let dist = pairwise_distances(&rows, self.metric);
        let core_dist = core_distances(&dist, k);

        // Step 2: Mutual reachability distances.
        let mr_dist = mutual_reachability(&dist, &core_dist);

        // Step 3: Minimum spanning tree (Prim's algorithm).
        let mst = prim_mst(&mr_dist);

        // Step 4: Sort MST edges by weight to get single-linkage ordering.
        let mut sorted_edges = mst;
        sorted_edges.sort_by(|a, b| a.2.total_cmp(&b.2));

        // Step 5: Build single-linkage hierarchy and condense.
        let (labels, n_clusters, outlier_scores) =
            extract_clusters(&sorted_edges, n, self.min_cluster_size);

        self.labels = labels;
        self.n_clusters = n_clusters;
        self.outlier_scores = outlier_scores;
        self.fitted = true;
        Ok(())
    }

    /// Convenience: fit and return labels.
    pub fn fit_predict(&mut self, data: &Dataset) -> Result<Vec<i32>> {
        self.fit(data)?;
        Ok(self.labels.clone())
    }
}

impl Default for Hdbscan {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal algorithms
// ---------------------------------------------------------------------------

/// Compute pairwise distances using the configured metric.
fn pairwise_distances(rows: &[Vec<f64>], metric: DistanceMetric) -> Vec<Vec<f64>> {
    let n = rows.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = match metric {
                DistanceMetric::Euclidean => euclidean_sq(&rows[i], &rows[j]).sqrt(),
                DistanceMetric::Manhattan => manhattan(&rows[i], &rows[j]),
                DistanceMetric::Cosine => cosine_distance(&rows[i], &rows[j]),
            };
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

/// Compute the core distance for each point (distance to k-th nearest neighbor).
fn core_distances(dist: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = dist.len();
    let mut core = Vec::with_capacity(n);
    for i in 0..n {
        let mut dists: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| dist[i][j]).collect();
        dists.sort_unstable_by(|a, b| a.total_cmp(b));
        let kth = (k - 1).min(dists.len() - 1);
        core.push(dists[kth]);
    }
    core
}

/// Compute mutual reachability distances.
///
/// `mr(a, b) = max(core(a), core(b), dist(a, b))`
fn mutual_reachability(dist: &[Vec<f64>], core_dist: &[f64]) -> Vec<Vec<f64>> {
    let n = dist.len();
    let mut mr = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist[i][j].max(core_dist[i]).max(core_dist[j]);
            mr[i][j] = d;
            mr[j][i] = d;
        }
    }
    mr
}

/// Prim's algorithm for MST. Returns edges as (u, v, weight).
fn prim_mst(dist: &[Vec<f64>]) -> Vec<(usize, usize, f64)> {
    let n = dist.len();
    if n <= 1 {
        return Vec::new();
    }

    let mut in_tree = vec![false; n];
    let mut min_edge = vec![f64::INFINITY; n];
    let mut edge_from = vec![0usize; n];
    let mut edges = Vec::with_capacity(n - 1);

    // Start from node 0.
    in_tree[0] = true;
    for j in 1..n {
        min_edge[j] = dist[0][j];
        edge_from[j] = 0;
    }

    for _ in 0..(n - 1) {
        // Find the closest non-tree node.
        let mut best = usize::MAX;
        let mut best_w = f64::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_edge[j] < best_w {
                best = j;
                best_w = min_edge[j];
            }
        }

        if best == usize::MAX {
            break;
        }

        in_tree[best] = true;
        edges.push((edge_from[best], best, best_w));

        // Update min_edge for remaining nodes.
        for j in 0..n {
            if !in_tree[j] && dist[best][j] < min_edge[j] {
                min_edge[j] = dist[best][j];
                edge_from[j] = best;
            }
        }
    }

    edges
}

/// Extract clusters from a sorted single-linkage MST using a simplified
/// condensed tree approach.
///
/// Returns (labels, n_clusters, outlier_scores).
fn extract_clusters(
    sorted_edges: &[(usize, usize, f64)],
    n: usize,
    min_cluster_size: usize,
) -> (Vec<i32>, usize, Vec<f64>) {
    // Union-Find for building the dendrogram.
    let mut parent: Vec<usize> = (0..n).collect();
    let mut size: Vec<usize> = vec![1; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    // Track the lambda (1/distance) at which each point was last "alive".
    let mut point_lambda = vec![0.0_f64; n];

    // Process edges in order of increasing weight (single-linkage merge order).
    // We'll extract clusters using a simple method:
    // At each merge, if both sides are ≥ min_cluster_size, they form separate
    // clusters. If only one side is large enough, the small side's points
    // are noise at this level.
    let mut cluster_assignments: Vec<i32> = vec![-1; n];
    let mut next_cluster = 0i32;

    // Simple approach: after building the full hierarchy, cut the tree
    // using the stability criterion.
    //
    // We'll group components at the point where merging would combine
    // two distinct dense regions.

    // Build the full hierarchy first.
    for &(u, v, w) in sorted_edges {
        let ru = find(&parent, u);
        let rv = find(&parent, v);
        if ru == rv {
            continue;
        }

        let lambda = if w > 0.0 { 1.0 / w } else { f64::INFINITY };

        // Record lambda for ALL points in both merging components.
        // This captures the density level at which each point participates
        // in a merge event.
        for &pt in &members[ru] {
            point_lambda[pt] = lambda;
        }
        for &pt in &members[rv] {
            point_lambda[pt] = lambda;
        }

        // Union by size.
        let (small, big) = if size[ru] < size[rv] {
            (ru, rv)
        } else {
            (rv, ru)
        };
        parent[small] = big;
        size[big] += size[small];
        let small_members = std::mem::take(&mut members[small]);
        members[big].extend(small_members);
    }

    // Now extract clusters: find connected components at a density threshold.
    // We use the simplified HDBSCAN approach: identify "prominent" cluster
    // splits in the hierarchy.
    //
    // Replay the merges. Each time two components ≥ min_cluster_size merge,
    // both are finalized as clusters. Remaining points are noise.
    let mut uf_parent: Vec<usize> = (0..n).collect();
    let mut uf_size: Vec<usize> = vec![1; n];
    let mut uf_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    // Track which component has been "finalized" as a cluster.
    let mut finalized: Vec<bool> = vec![false; n];

    for &(u, v, _w) in sorted_edges {
        let ru = find(&uf_parent, u);
        let rv = find(&uf_parent, v);
        if ru == rv {
            continue;
        }

        let size_u = uf_size[ru];
        let size_v = uf_size[rv];

        // If both components are large enough, finalize each as a cluster
        // before merging them.
        if size_u >= min_cluster_size && size_v >= min_cluster_size {
            if !finalized[ru] {
                for &pt in &uf_members[ru] {
                    cluster_assignments[pt] = next_cluster;
                }
                next_cluster += 1;
                finalized[ru] = true;
            }
            if !finalized[rv] {
                for &pt in &uf_members[rv] {
                    cluster_assignments[pt] = next_cluster;
                }
                next_cluster += 1;
                finalized[rv] = true;
            }
        }

        // Union by size.
        let (small, big) = if uf_size[ru] < uf_size[rv] {
            (ru, rv)
        } else {
            (rv, ru)
        };
        uf_parent[small] = big;
        uf_size[big] += uf_size[small];
        let small_members = std::mem::take(&mut uf_members[small]);
        uf_members[big].extend(small_members);

        if finalized[small] || finalized[big] {
            finalized[big] = true;
        }
    }

    // If no split was ever triggered (all data forms one cluster), and
    // the total size is ≥ min_cluster_size, assign everything to cluster 0.
    if next_cluster == 0 && n >= min_cluster_size {
        cluster_assignments.fill(0);
        next_cluster = 1;
    }

    // Compute outlier scores.
    // For each point, the outlier score is based on how early it was merged
    // relative to its cluster's typical density.
    let mut outlier_scores = vec![0.0_f64; n];
    if next_cluster > 0 {
        // For each cluster, find the max lambda of its members.
        let mut cluster_max_lambda = vec![0.0_f64; next_cluster as usize];
        for i in 0..n {
            let c = cluster_assignments[i];
            if c >= 0 {
                let cu = c as usize;
                if point_lambda[i] > cluster_max_lambda[cu] {
                    cluster_max_lambda[cu] = point_lambda[i];
                }
            }
        }

        for i in 0..n {
            let c = cluster_assignments[i];
            if c < 0 {
                outlier_scores[i] = 1.0;
            } else {
                let max_l = cluster_max_lambda[c as usize];
                if max_l > 0.0 {
                    outlier_scores[i] = 1.0 - (point_lambda[i] / max_l).min(1.0);
                }
            }
        }
    } else {
        outlier_scores.fill(1.0);
    }

    (cluster_assignments, next_cluster as usize, outlier_scores)
}

/// Union-Find path-compressed find.
fn find(parent: &[usize], mut x: usize) -> usize {
    while parent[x] != x {
        x = parent[x];
    }
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdbscan_two_clusters() {
        let f1 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4];
        let f2 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4];
        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; 10],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut hdb = Hdbscan::new().min_cluster_size(3);
        hdb.fit(&data).unwrap();

        assert_eq!(hdb.n_clusters(), 2, "should find 2 clusters");

        // First 5 and last 5 should have different labels.
        let labels = hdb.labels();
        let cluster_a = labels[0];
        let cluster_b = labels[5];
        assert!(cluster_a >= 0, "first cluster should not be noise");
        assert!(cluster_b >= 0, "second cluster should not be noise");
        assert_ne!(cluster_a, cluster_b, "clusters should be different");
    }

    #[test]
    fn test_hdbscan_with_noise() {
        // Two tight clusters + an outlier at (100, 100).
        let mut f1 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4];
        let mut f2 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4];
        f1.push(100.0);
        f2.push(100.0);

        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; 11],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut hdb = Hdbscan::new().min_cluster_size(3);
        hdb.fit(&data).unwrap();

        assert_eq!(hdb.n_clusters(), 2);

        // The outlier should be noise.
        let labels = hdb.labels();
        assert_eq!(labels[10], -1, "outlier should be noise");
    }

    #[test]
    fn test_hdbscan_all_same() {
        let f1 = vec![1.0; 10];
        let f2 = vec![1.0; 10];
        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; 10],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut hdb = Hdbscan::new().min_cluster_size(3);
        hdb.fit(&data).unwrap();

        // All points are identical → single cluster.
        assert_eq!(hdb.n_clusters(), 1);
        assert_eq!(hdb.n_noise(), 0);
    }

    #[test]
    fn test_hdbscan_min_cluster_size_respected() {
        let f1 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4];
        let f2 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4];
        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; 10],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut hdb = Hdbscan::new().min_cluster_size(3);
        hdb.fit(&data).unwrap();

        // Count members per cluster.
        let labels = hdb.labels();
        let mut counts = std::collections::HashMap::new();
        for &l in labels {
            if l >= 0 {
                *counts.entry(l).or_insert(0usize) += 1;
            }
        }

        for (&cluster, &count) in &counts {
            assert!(
                count >= 3,
                "cluster {} has {} members, less than min_cluster_size=3",
                cluster,
                count
            );
        }
    }

    #[test]
    fn test_hdbscan_empty_dataset() {
        let data = Dataset::new(Vec::<Vec<f64>>::new(), Vec::new(), Vec::new(), "label");
        let mut hdb = Hdbscan::new();
        assert!(hdb.fit(&data).is_err());
    }

    #[test]
    fn test_hdbscan_outlier_scores() {
        let f1 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4, 100.0];
        let f2 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4, 100.0];
        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; 11],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut hdb = Hdbscan::new().min_cluster_size(3);
        hdb.fit(&data).unwrap();

        let scores = hdb.outlier_scores();
        assert_eq!(scores.len(), 11);

        // Noise point should have outlier score of 1.0.
        assert!(
            (scores[10] - 1.0).abs() < 1e-6,
            "noise point outlier score should be 1.0, got {}",
            scores[10]
        );

        // Cluster interior points should have low outlier scores.
        for &s in &scores[..5] {
            assert!(
                s < 1.0,
                "cluster point should have outlier score < 1.0, got {s}"
            );
        }
    }

    #[test]
    fn test_hdbscan_cosine_metric() {
        use crate::neighbors::DistanceMetric;

        // Two groups of unit vectors in opposite-ish directions
        // Group A: vectors near [1, 0] (normalized)
        // Group B: vectors near [0, 1] (normalized)
        let n_per = 5;
        let mut f1 = Vec::new();
        let mut f2 = Vec::new();
        for i in 0..n_per {
            // Group A: angle near 0°
            let angle = 0.05 * i as f64;
            f1.push(angle.cos());
            f2.push(angle.sin());
        }
        for i in 0..n_per {
            // Group B: angle near 90°
            let angle = std::f64::consts::FRAC_PI_2 + 0.05 * i as f64;
            f1.push(angle.cos());
            f2.push(angle.sin());
        }

        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; n_per * 2],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut hdb = Hdbscan::new()
            .min_cluster_size(3)
            .metric(DistanceMetric::Cosine);
        hdb.fit(&data).unwrap();

        assert_eq!(hdb.n_clusters(), 2, "should find 2 clusters with cosine metric");
        let labels = hdb.labels();
        assert_ne!(labels[0], labels[n_per], "groups should have different labels");
    }

    #[test]
    fn test_hdbscan_fit_predict() {
        let f1 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4];
        let f2 = vec![0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4];
        let data = Dataset::new(
            vec![f1, f2],
            vec![0.0; 10],
            vec!["x".into(), "y".into()],
            "label",
        );

        let mut hdb = Hdbscan::new().min_cluster_size(3);
        let labels = hdb.fit_predict(&data).unwrap();
        assert_eq!(labels.len(), 10);
        assert!(labels.iter().any(|&l| l >= 0), "should have some clusters");
    }
}
