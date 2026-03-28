// SPDX-License-Identifier: MIT OR Apache-2.0
//! Brute-force descriptor matcher supporting both Hamming and L2 distance.

use crate::features::keypoint::{BinaryDescriptor, FloatDescriptor};
use crate::matching::match_types::DMatch;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

macro_rules! maybe_par_iter {
    ($range:expr) => {{
        #[cfg(feature = "rayon")]
        {
            $range.into_par_iter()
        }
        #[cfg(not(feature = "rayon"))]
        {
            $range.into_iter()
        }
    }};
}

/// Find the best match for each query descriptor in the train set (Hamming distance).
pub fn match_binary(query: &[BinaryDescriptor], train: &[BinaryDescriptor]) -> Vec<DMatch> {
    maybe_par_iter!(0..query.len())
        .filter_map(|qi| {
            let qd = &query[qi];
            let mut best_dist = u32::MAX;
            let mut best_idx = 0;
            for (ti, td) in train.iter().enumerate() {
                let dist = qd.hamming_distance(td);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = ti;
                }
            }
            if best_dist < u32::MAX {
                Some(DMatch {
                    query_idx: qi,
                    train_idx: best_idx,
                    distance: best_dist as f32,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Find the k nearest matches for each query (Hamming distance).
pub fn knn_match_binary(
    query: &[BinaryDescriptor],
    train: &[BinaryDescriptor],
    k: usize,
) -> Vec<Vec<DMatch>> {
    maybe_par_iter!(0..query.len())
        .map(|qi| {
            let qd = &query[qi];
            let mut dists: Vec<(usize, u32)> = train
                .iter()
                .enumerate()
                .map(|(ti, td)| (ti, qd.hamming_distance(td)))
                .collect();
            dists.sort_unstable_by_key(|&(_, d)| d);
            dists
                .into_iter()
                .take(k)
                .map(|(ti, d)| DMatch {
                    query_idx: qi,
                    train_idx: ti,
                    distance: d as f32,
                })
                .collect()
        })
        .collect()
}

/// Find the best match for each query descriptor (L2 distance).
pub fn match_float(query: &[FloatDescriptor], train: &[FloatDescriptor]) -> Vec<DMatch> {
    maybe_par_iter!(0..query.len())
        .filter_map(|qi| {
            let qd = &query[qi];
            let mut best_dist = f32::INFINITY;
            let mut best_idx = 0;
            for (ti, td) in train.iter().enumerate() {
                let dist = qd.l2_distance(td);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = ti;
                }
            }
            if best_dist.is_finite() {
                Some(DMatch {
                    query_idx: qi,
                    train_idx: best_idx,
                    distance: best_dist,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Find the k nearest matches for each query (L2 distance).
pub fn knn_match_float(
    query: &[FloatDescriptor],
    train: &[FloatDescriptor],
    k: usize,
) -> Vec<Vec<DMatch>> {
    maybe_par_iter!(0..query.len())
        .map(|qi| {
            let qd = &query[qi];
            let mut dists: Vec<(usize, f32)> = train
                .iter()
                .enumerate()
                .map(|(ti, td)| (ti, qd.l2_distance(td)))
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists
                .into_iter()
                .take(k)
                .map(|(ti, d)| DMatch {
                    query_idx: qi,
                    train_idx: ti,
                    distance: d,
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_self_match() {
        let descs = vec![
            BinaryDescriptor {
                data: vec![0x00, 0xFF],
            },
            BinaryDescriptor {
                data: vec![0xFF, 0x00],
            },
            BinaryDescriptor {
                data: vec![0xAB, 0xCD],
            },
        ];
        let matches = match_binary(&descs, &descs);
        assert_eq!(matches.len(), 3);
        for m in &matches {
            assert_eq!(m.query_idx, m.train_idx);
            assert_eq!(m.distance, 0.0);
        }
    }

    #[test]
    fn float_self_match() {
        let descs = vec![
            FloatDescriptor {
                data: vec![1.0, 0.0],
            },
            FloatDescriptor {
                data: vec![0.0, 1.0],
            },
        ];
        let matches = match_float(&descs, &descs);
        assert_eq!(matches.len(), 2);
        for m in &matches {
            assert_eq!(m.query_idx, m.train_idx);
            assert!(m.distance < 1e-6);
        }
    }

    #[test]
    fn knn_returns_k() {
        let descs = vec![
            BinaryDescriptor { data: vec![0x00] },
            BinaryDescriptor { data: vec![0x01] },
            BinaryDescriptor { data: vec![0xFF] },
        ];
        let knn = knn_match_binary(&descs, &descs, 2);
        assert_eq!(knn.len(), 3);
        for matches in &knn {
            assert_eq!(matches.len(), 2);
            assert_eq!(matches[0].distance, 0.0);
        }
    }
}
