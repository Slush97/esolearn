// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lowe's ratio test for filtering descriptor matches.

use crate::matching::match_types::DMatch;

/// Apply Lowe's ratio test to k-nearest-neighbor matches.
///
/// For each query, if the best match distance is less than `ratio * second_best_distance`,
/// the match is kept. Otherwise it is discarded as ambiguous.
///
/// Typical ratio: 0.75 (Lowe's original recommendation).
pub fn ratio_test(knn_matches: &[Vec<DMatch>], ratio: f32) -> Vec<DMatch> {
    knn_matches
        .iter()
        .filter_map(|matches| {
            if matches.len() < 2 {
                return matches.first().cloned();
            }
            let best = &matches[0];
            let second = &matches[1];
            if second.distance > f32::EPSILON && best.distance / second.distance < ratio {
                Some(best.clone())
            } else {
                None
            }
        })
        .collect()
}

/// Apply cross-check filtering: keep only matches where both A→B and B→A agree.
pub fn cross_check(matches_ab: &[DMatch], matches_ba: &[DMatch]) -> Vec<DMatch> {
    matches_ab
        .iter()
        .filter(|m_ab| {
            matches_ba.iter().any(|m_ba| {
                m_ba.query_idx == m_ab.train_idx && m_ba.train_idx == m_ab.query_idx
            })
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_test_filters_ambiguous() {
        let knn = vec![
            // Good match: 10 / 50 = 0.2 < 0.75
            vec![
                DMatch { query_idx: 0, train_idx: 1, distance: 10.0 },
                DMatch { query_idx: 0, train_idx: 2, distance: 50.0 },
            ],
            // Ambiguous match: 40 / 45 = 0.89 > 0.75
            vec![
                DMatch { query_idx: 1, train_idx: 3, distance: 40.0 },
                DMatch { query_idx: 1, train_idx: 4, distance: 45.0 },
            ],
        ];
        let good = ratio_test(&knn, 0.75);
        assert_eq!(good.len(), 1);
        assert_eq!(good[0].query_idx, 0);
    }

    #[test]
    fn cross_check_filters() {
        let ab = vec![
            DMatch { query_idx: 0, train_idx: 1, distance: 5.0 },
            DMatch { query_idx: 1, train_idx: 2, distance: 10.0 },
        ];
        let ba = vec![
            DMatch { query_idx: 1, train_idx: 0, distance: 5.0 }, // agrees with ab[0]
            DMatch { query_idx: 2, train_idx: 3, distance: 8.0 }, // disagrees with ab[1]
        ];
        let good = cross_check(&ab, &ba);
        assert_eq!(good.len(), 1);
        assert_eq!(good[0].query_idx, 0);
    }
}
