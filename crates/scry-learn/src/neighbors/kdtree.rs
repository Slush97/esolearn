// SPDX-License-Identifier: MIT OR Apache-2.0
//! KD-tree spatial index for fast k-nearest-neighbor queries.
//!
//! Provides O(n log n) build and O(log n) average-case query versus brute-force
//! O(n) per query. Significant speedup for datasets with moderate dimensionality
//! (< 20 features). Uses a flat arena layout for cache-friendly traversal.
//!
//! # Algorithm
//!
//! - **Build**: Recursively partition points by the median along the axis with
//!   the widest spread, cycling through dimensions. Stored as a flat `Vec` of
//!   nodes (arena allocation).
//! - **Query**: Depth-first traversal with branch pruning: if the splitting
//!   plane distance exceeds the current worst distance in the k-neighbor heap,
//!   the far subtree is skipped entirely.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::distance::euclidean_sq;

/// A KD-tree for fast nearest-neighbor lookup in Euclidean space.
///
/// Stores points in a flat arena (no `Box`/`Rc`). Each node is either a
/// `Split` (with a splitting dimension, value, and child indices) or a `Leaf`
/// (storing a single point index into the original training data).
///
/// # Example
///
/// ```
/// use scry_learn::neighbors::KdTree;
///
/// let points = vec![
///     vec![0.0, 0.0],
///     vec![1.0, 0.0],
///     vec![0.0, 1.0],
///     vec![10.0, 10.0],
/// ];
/// let tree = KdTree::build(&points);
///
/// // Query 2 nearest neighbors to (0.5, 0.5)
/// let neighbors = tree.query_k_nearest(&[0.5, 0.5], 2, &points);
/// // Returns (squared_distance, original_index) pairs, sorted nearest-first
/// assert_eq!(neighbors.len(), 2);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct KdTree {
    nodes: Vec<KdNode>,
    n_dims: usize,
}

/// A single node in the KD-tree arena.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum KdNode {
    /// Internal split node.
    Split {
        /// Dimension to split on.
        dim: usize,
        /// Split value (median along `dim`).
        value: f64,
        /// Index of left child in the arena.
        left: usize,
        /// Index of right child in the arena.
        right: usize,
    },
    /// Leaf node containing a single point.
    Leaf {
        /// Index into the original point array.
        point_idx: usize,
    },
}

/// Max-heap entry for the k-nearest neighbor search.
///
/// We negate the ordering so that `BinaryHeap` (a max-heap) keeps the
/// *farthest* neighbor on top, making it easy to check whether a new
/// candidate is closer than the current worst.
#[derive(Clone, Copy)]
struct HeapEntry {
    dist_sq: f64,
    idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Largest distance first (max-heap behavior).
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(Ordering::Equal)
    }
}

impl KdTree {
    /// Build a KD-tree from the given points.
    ///
    /// `points` is row-major: `points[sample_idx][feature_idx]`.
    /// Returns an empty tree if `points` is empty.
    ///
    /// # Complexity
    ///
    /// O(n log n) time, O(n) space.
    pub fn build(points: &[Vec<f64>]) -> Self {
        if points.is_empty() {
            return Self {
                nodes: Vec::new(),
                n_dims: 0,
            };
        }

        let n_dims = points[0].len();
        let mut nodes = Vec::with_capacity(2 * points.len());
        let indices: Vec<usize> = (0..points.len()).collect();
        Self::build_recursive(points, &indices, 0, n_dims, &mut nodes);

        Self { nodes, n_dims }
    }

    /// Recursively build the tree, returning the index of the created node.
    fn build_recursive(
        points: &[Vec<f64>],
        indices: &[usize],
        depth: usize,
        n_dims: usize,
        nodes: &mut Vec<KdNode>,
    ) -> usize {
        debug_assert!(!indices.is_empty());

        if indices.len() == 1 {
            let node_idx = nodes.len();
            nodes.push(KdNode::Leaf {
                point_idx: indices[0],
            });
            return node_idx;
        }

        // Pick split dimension: cycle through dimensions.
        // Using widest-spread heuristic for better balance.
        let dim = Self::best_split_dim(points, indices, n_dims, depth);

        // Sort indices by the split dimension and pick median.
        let mut sorted = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            points[a][dim]
                .partial_cmp(&points[b][dim])
                .unwrap_or(Ordering::Equal)
        });

        let median = sorted.len() / 2;
        let split_value = points[sorted[median]][dim];

        // Reserve a slot for this node (we'll fill it after children are built).
        let this_idx = nodes.len();
        nodes.push(KdNode::Leaf { point_idx: 0 }); // placeholder

        let left_indices = &sorted[..median];
        let right_indices = &sorted[median..];

        // Handle edge case: if all values are the same on this dim, just split in half.
        let left_idx = if left_indices.is_empty() {
            // All points have the same value — create a leaf from the first point.
            let leaf_idx = nodes.len();
            nodes.push(KdNode::Leaf {
                point_idx: right_indices[0],
            });
            // Adjust right to skip the first.
            leaf_idx
        } else {
            Self::build_recursive(points, left_indices, depth + 1, n_dims, nodes)
        };

        let right_idx = if right_indices.is_empty() {
            // Should not happen given median logic, but safety.
            let leaf_idx = nodes.len();
            nodes.push(KdNode::Leaf {
                point_idx: left_indices[left_indices.len() - 1],
            });
            leaf_idx
        } else if left_indices.is_empty() && right_indices.len() > 1 {
            // All points went right, skip the one used for left.
            Self::build_recursive(points, &right_indices[1..], depth + 1, n_dims, nodes)
        } else {
            Self::build_recursive(points, right_indices, depth + 1, n_dims, nodes)
        };

        nodes[this_idx] = KdNode::Split {
            dim,
            value: split_value,
            left: left_idx,
            right: right_idx,
        };

        this_idx
    }

    /// Pick the dimension with the widest spread among the given indices.
    ///
    /// Falls back to cycling (`depth % n_dims`) if all spreads are zero.
    #[allow(clippy::needless_range_loop)] // iterating dimensions, not points
    fn best_split_dim(
        points: &[Vec<f64>],
        indices: &[usize],
        n_dims: usize,
        depth: usize,
    ) -> usize {
        let mut best_dim = depth % n_dims;
        let mut best_spread = -1.0_f64;

        for d in 0..n_dims {
            let (min_v, max_v) =
                indices
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &idx| {
                        let v = points[idx][d];
                        (lo.min(v), hi.max(v))
                    });
            let spread = max_v - min_v;
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }

        best_dim
    }

    /// Query the k nearest neighbors to `query`.
    ///
    /// Returns `(squared_distance, point_index)` pairs sorted by distance
    /// (nearest first). `points` must be the same array used to build the tree.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`.
    pub fn query_k_nearest(
        &self,
        query: &[f64],
        k: usize,
        points: &[Vec<f64>],
    ) -> Vec<(f64, usize)> {
        assert!(k > 0, "k must be at least 1");

        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);
        self.search(0, query, k, points, &mut heap);

        // Drain heap into sorted result (nearest first).
        let mut result: Vec<(f64, usize)> = heap.into_iter().map(|e| (e.dist_sq, e.idx)).collect();
        // Sort by (distance, index) — lower index wins on ties, matching
        // brute-force tie-breaking and sklearn behavior.
        result.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        result
    }

    /// Recursive search with branch pruning.
    fn search(
        &self,
        node_idx: usize,
        query: &[f64],
        k: usize,
        points: &[Vec<f64>],
        heap: &mut BinaryHeap<HeapEntry>,
    ) {
        match &self.nodes[node_idx] {
            KdNode::Leaf { point_idx } => {
                let dist_sq = euclidean_sq(query, &points[*point_idx]);
                if heap.len() < k {
                    heap.push(HeapEntry {
                        dist_sq,
                        idx: *point_idx,
                    });
                } else if let Some(worst) = heap.peek() {
                    if dist_sq < worst.dist_sq {
                        heap.pop();
                        heap.push(HeapEntry {
                            dist_sq,
                            idx: *point_idx,
                        });
                    }
                }
            }
            KdNode::Split {
                dim,
                value,
                left,
                right,
            } => {
                let diff = query[*dim] - value;
                let (near, far) = if diff <= 0.0 {
                    (*left, *right)
                } else {
                    (*right, *left)
                };

                // Always search the near side.
                self.search(near, query, k, points, heap);

                // Prune: only search far side if splitting plane is closer
                // than the current k-th nearest distance.
                let plane_dist_sq = diff * diff;
                let should_search_far = heap.len() < k
                    || heap
                        .peek()
                        .is_none_or(|worst| plane_dist_sq < worst.dist_sq);

                if should_search_far {
                    self.search(far, query, k, points, heap);
                }
            }
        }
    }

    /// Query all points within a squared-distance radius.
    ///
    /// Returns indices of all points `p` where
    /// `euclidean_sq(query, p) <= radius_sq`. Uses the same
    /// branch-pruning strategy as [`KdTree::query_k_nearest`].
    ///
    /// # Example
    ///
    /// ```
    /// use scry_learn::neighbors::KdTree;
    ///
    /// let points = vec![
    ///     vec![0.0, 0.0],
    ///     vec![1.0, 0.0],
    ///     vec![10.0, 10.0],
    /// ];
    /// let tree = KdTree::build(&points);
    ///
    /// // radius² = 4.0 → radius = 2.0
    /// let within = tree.query_radius(&[0.5, 0.0], 4.0, &points);
    /// assert_eq!(within.len(), 2); // indices 0 and 1
    /// ```
    pub fn query_radius(&self, query: &[f64], radius_sq: f64, points: &[Vec<f64>]) -> Vec<usize> {
        let mut result = Vec::new();
        if !self.nodes.is_empty() {
            self.search_radius(0, query, radius_sq, points, &mut result);
        }
        result
    }

    /// Recursive radius search with branch pruning.
    fn search_radius(
        &self,
        node_idx: usize,
        query: &[f64],
        radius_sq: f64,
        points: &[Vec<f64>],
        result: &mut Vec<usize>,
    ) {
        match &self.nodes[node_idx] {
            KdNode::Leaf { point_idx } => {
                if euclidean_sq(query, &points[*point_idx]) <= radius_sq {
                    result.push(*point_idx);
                }
            }
            KdNode::Split {
                dim,
                value,
                left,
                right,
            } => {
                let diff = query[*dim] - value;
                let (near, far) = if diff <= 0.0 {
                    (*left, *right)
                } else {
                    (*right, *left)
                };

                // Always search the near side.
                self.search_radius(near, query, radius_sq, points, result);

                // Prune: only search far side if splitting plane is within radius.
                let plane_dist_sq = diff * diff;
                if plane_dist_sq <= radius_sq {
                    self.search_radius(far, query, radius_sq, points, result);
                }
            }
        }
    }

    /// Returns `true` if the tree is empty (no points).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of dimensions the tree was built with.
    pub fn n_dims(&self) -> usize {
        self.n_dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kdtree_empty() {
        let tree = KdTree::build(&[]);
        assert!(tree.is_empty());
        let result = tree.query_k_nearest(&[0.0, 0.0], 1, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_kdtree_single_point() {
        let points = vec![vec![1.0, 2.0]];
        let tree = KdTree::build(&points);
        let result = tree.query_k_nearest(&[0.0, 0.0], 1, &points);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 0); // index 0
        assert!((result[0].0 - 5.0).abs() < 1e-9); // 1² + 2² = 5
    }

    #[test]
    fn test_kdtree_two_clusters() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 10.0],
            vec![10.0, 11.0],
        ];
        let tree = KdTree::build(&points);

        // Query near origin — should find cluster A.
        let result = tree.query_k_nearest(&[0.5, 0.5], 3, &points);
        assert_eq!(result.len(), 3);
        for (_, idx) in &result {
            assert!(*idx < 3, "Expected cluster A indices, got {idx}");
        }

        // Query near (10,10) — should find cluster B.
        let result = tree.query_k_nearest(&[10.5, 10.5], 3, &points);
        assert_eq!(result.len(), 3);
        for (_, idx) in &result {
            assert!(*idx >= 3, "Expected cluster B indices, got {idx}");
        }
    }

    #[test]
    fn test_kdtree_k_larger_than_n() {
        let points = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let tree = KdTree::build(&points);
        let result = tree.query_k_nearest(&[0.0, 0.0], 5, &points);
        assert_eq!(result.len(), 2, "Should return all points when k > n");
    }

    #[test]
    fn test_kdtree_sorted_nearest_first() {
        let points = vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![2.0, 0.0]];
        let tree = KdTree::build(&points);
        let result = tree.query_k_nearest(&[1.0, 0.0], 3, &points);
        // Distances: 0→1, 2→1, 5→16
        assert!(result[0].0 <= result[1].0);
        assert!(result[1].0 <= result[2].0);
    }

    #[test]
    fn test_kdtree_matches_brute_force() {
        // 100 random-ish points in 5D, verify KD-tree returns same neighbors
        // as brute-force for several queries.
        let mut points = Vec::new();
        let mut seed = 12345u64;
        for _ in 0..100 {
            let mut p = Vec::new();
            for _ in 0..5 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                p.push((seed >> 33) as f64 / 1e9);
            }
            points.push(p);
        }

        let tree = KdTree::build(&points);

        for q_idx in [0, 25, 50, 75, 99] {
            let query = &points[q_idx];
            let k = 7;

            // Brute-force k-nearest.
            let mut dists: Vec<(f64, usize)> = points
                .iter()
                .enumerate()
                .map(|(i, p)| (euclidean_sq(query, p), i))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let brute: Vec<usize> = dists.iter().take(k).map(|(_, i)| *i).collect();

            let kd_result = tree.query_k_nearest(query, k, &points);
            let kd: Vec<usize> = kd_result.iter().map(|(_, i)| *i).collect();

            assert_eq!(
                brute, kd,
                "KD-tree and brute-force disagree for query point {q_idx}"
            );
        }
    }

    #[test]
    fn test_kdtree_duplicate_points() {
        let points = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![5.0, 5.0],
        ];
        let tree = KdTree::build(&points);
        let result = tree.query_k_nearest(&[1.0, 1.0], 3, &points);
        assert_eq!(result.len(), 3);
        // All three should be exact matches (distance = 0).
        for (dist, _) in &result {
            assert!(*dist < 1e-9);
        }
    }

    #[test]
    fn test_kdtree_high_dim() {
        // 20-dimensional data — still within the recommended range.
        let mut points = Vec::new();
        let mut seed = 42u64;
        for _ in 0..50 {
            let mut p = Vec::new();
            for _ in 0..20 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                p.push((seed >> 33) as f64 / 1e9);
            }
            points.push(p);
        }

        let tree = KdTree::build(&points);
        let result = tree.query_k_nearest(&points[0], 5, &points);
        assert_eq!(result.len(), 5);
        // First result should be the query point itself (distance 0).
        assert!((result[0].0).abs() < 1e-9);
        assert_eq!(result[0].1, 0);
    }

    #[test]
    fn test_kdtree_query_radius() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 10.0],
        ];
        let tree = KdTree::build(&points);

        // radius² = 2.0 → includes points within sqrt(2) ≈ 1.414 of query (0.5, 0.5)
        let mut result = tree.query_radius(&[0.5, 0.5], 2.0, &points);
        result.sort_unstable();
        assert_eq!(result, vec![0, 1, 2, 3], "Should find the 4 nearby points");

        // Verify against brute-force on larger random data.
        let mut rng_points = Vec::new();
        let mut seed = 99u64;
        for _ in 0..200 {
            let mut p = Vec::new();
            for _ in 0..3 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                p.push((seed >> 33) as f64 / 1e9);
            }
            rng_points.push(p);
        }
        let tree2 = KdTree::build(&rng_points);
        let query = &[2.0, 2.0, 2.0];
        let radius_sq = 1.0;

        let mut kd_result = tree2.query_radius(query, radius_sq, &rng_points);
        kd_result.sort_unstable();

        let brute: Vec<usize> = rng_points
            .iter()
            .enumerate()
            .filter(|(_, p)| euclidean_sq(query, p) <= radius_sq)
            .map(|(i, _)| i)
            .collect();

        assert_eq!(
            kd_result, brute,
            "KD-tree radius and brute-force should agree"
        );
    }
}
