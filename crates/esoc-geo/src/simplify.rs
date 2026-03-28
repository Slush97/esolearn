// SPDX-License-Identifier: MIT OR Apache-2.0
//! Ramer-Douglas-Peucker polygon simplification.

use crate::geometry::GeoPoint;

/// Perpendicular distance from point `p` to the line segment `a`→`b`.
fn perpendicular_distance(p: GeoPoint, a: GeoPoint, b: GeoPoint) -> f64 {
    let dx = b.lon - a.lon;
    let dy = b.lat - a.lat;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-20 {
        // a and b are the same point
        let ex = p.lon - a.lon;
        let ey = p.lat - a.lat;
        return ex.hypot(ey);
    }
    ((dy * p.lon - dx * p.lat + b.lon * a.lat - b.lat * a.lon).abs()) / len_sq.sqrt()
}

/// Simplify a sequence of points using the Ramer-Douglas-Peucker algorithm.
///
/// `epsilon` is the maximum allowed perpendicular distance. Points farther
/// than `epsilon` from the simplified line are kept.
///
/// Uses an iterative (stack-based) implementation to avoid stack overflow
/// on large inputs.
pub fn simplify(points: &[GeoPoint], epsilon: f64) -> Vec<GeoPoint> {
    let n = points.len();
    if n <= 2 {
        return points.to_vec();
    }

    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    // Stack of (start, end) index pairs to process
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];

    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue;
        }

        let mut max_dist = 0.0;
        let mut max_idx = start;

        for i in (start + 1)..end {
            let d = perpendicular_distance(points[i], points[start], points[end]);
            if d > max_dist {
                max_dist = d;
                max_idx = i;
            }
        }

        if max_dist > epsilon {
            keep[max_idx] = true;
            stack.push((start, max_idx));
            stack.push((max_idx, end));
        }
    }

    points
        .iter()
        .zip(keep.iter())
        .filter(|(_, &k)| k)
        .map(|(&p, _)| p)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_collapses() {
        let points = vec![
            GeoPoint::new(0.0, 0.0),
            GeoPoint::new(1.0, 0.0),
            GeoPoint::new(2.0, 0.0),
            GeoPoint::new(3.0, 0.0),
            GeoPoint::new(4.0, 0.0),
        ];
        let result = simplify(&points, 0.1);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], points[0]);
        assert_eq!(result[1], points[4]);
    }

    #[test]
    fn l_shape_keeps_corner() {
        let points = vec![
            GeoPoint::new(0.0, 0.0),
            GeoPoint::new(1.0, 0.0),
            GeoPoint::new(2.0, 0.0),
            GeoPoint::new(2.0, 1.0),
            GeoPoint::new(2.0, 2.0),
        ];
        let result = simplify(&points, 0.1);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], points[0]);
        assert_eq!(result[1], GeoPoint::new(2.0, 0.0));
        assert_eq!(result[2], points[4]);
    }

    #[test]
    fn two_points_unchanged() {
        let points = vec![GeoPoint::new(0.0, 0.0), GeoPoint::new(5.0, 5.0)];
        let result = simplify(&points, 1.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn empty_input() {
        let result = simplify(&[], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn single_point() {
        let points = vec![GeoPoint::new(1.0, 2.0)];
        let result = simplify(&points, 1.0);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn large_epsilon_collapses() {
        let points = vec![
            GeoPoint::new(0.0, 0.0),
            GeoPoint::new(1.0, 5.0),
            GeoPoint::new(2.0, 0.0),
            GeoPoint::new(3.0, 5.0),
            GeoPoint::new(4.0, 0.0),
        ];
        let result = simplify(&points, 100.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn zero_epsilon_keeps_all() {
        let points = vec![
            GeoPoint::new(0.0, 0.0),
            GeoPoint::new(1.0, 1.0),
            GeoPoint::new(2.0, 0.0),
        ];
        let result = simplify(&points, 0.0);
        // With epsilon=0, any deviation > 0 is kept
        assert_eq!(result.len(), 3);
    }
}
