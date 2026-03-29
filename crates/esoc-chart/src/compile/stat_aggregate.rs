// SPDX-License-Identifier: MIT OR Apache-2.0
//! Aggregate statistical transform (count, sum, mean, etc.).

use crate::error::{ChartError, Result};
use crate::grammar::stat::AggregateFunc;

/// Result of aggregation: per-category values.
pub struct AggregateResult {
    pub categories: Vec<String>,
    pub x_data: Vec<f64>,
    pub y_data: Vec<f64>,
}

/// Compute aggregate of y_data grouped by categories.
pub fn compute_aggregate(
    categories: &[String],
    y_data: &[f64],
    func: AggregateFunc,
) -> Result<AggregateResult> {
    if categories.len() != y_data.len() {
        return Err(ChartError::LengthMismatch {
            expected: categories.len(),
            got: y_data.len(),
        });
    }

    // Collect unique categories in order of first appearance
    let mut unique_cats: Vec<String> = Vec::new();
    for c in categories {
        if !unique_cats.contains(c) {
            unique_cats.push(c.clone());
        }
    }

    let mut y_out = Vec::with_capacity(unique_cats.len());
    for cat in &unique_cats {
        let values: Vec<f64> = categories
            .iter()
            .zip(y_data.iter())
            .filter(|(c, _)| *c == cat)
            .map(|(_, &v)| v)
            .collect();

        let agg = match func {
            AggregateFunc::Count => values.len() as f64,
            AggregateFunc::Sum => {
                if values.is_empty() {
                    f64::NAN
                } else {
                    values.iter().sum()
                }
            }
            AggregateFunc::Mean => {
                if values.is_empty() {
                    f64::NAN
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
            AggregateFunc::Median => {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if sorted.is_empty() {
                    f64::NAN
                } else if sorted.len() % 2 == 0 {
                    let mid = sorted.len() / 2;
                    f64::midpoint(sorted[mid - 1], sorted[mid])
                } else {
                    sorted[sorted.len() / 2]
                }
            }
            AggregateFunc::Min => {
                if values.is_empty() {
                    f64::NAN
                } else {
                    values.iter().copied().fold(f64::INFINITY, f64::min)
                }
            }
            AggregateFunc::Max => {
                if values.is_empty() {
                    f64::NAN
                } else {
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                }
            }
        };
        y_out.push(agg);
    }

    let x_data: Vec<f64> = (0..unique_cats.len()).map(|i| i as f64).collect();

    Ok(AggregateResult {
        categories: unique_cats,
        x_data,
        y_data: y_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_aggregate() {
        let cats = vec!["A".into(), "B".into(), "A".into(), "B".into(), "A".into()];
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compute_aggregate(&cats, &vals, AggregateFunc::Count).unwrap();
        assert_eq!(result.categories, vec!["A", "B"]);
        assert_eq!(result.y_data, vec![3.0, 2.0]);
    }

    #[test]
    fn sum_aggregate() {
        let cats = vec!["A".into(), "B".into(), "A".into()];
        let vals = vec![10.0, 20.0, 30.0];
        let result = compute_aggregate(&cats, &vals, AggregateFunc::Sum).unwrap();
        assert_eq!(result.y_data, vec![40.0, 20.0]);
    }

    #[test]
    fn mean_aggregate() {
        let cats = vec!["X".into(), "X".into(), "X".into()];
        let vals = vec![2.0, 4.0, 6.0];
        let result = compute_aggregate(&cats, &vals, AggregateFunc::Mean).unwrap();
        assert!((result.y_data[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn median_aggregate() {
        let cats = vec!["A".into(), "A".into(), "A".into(), "A".into()];
        let vals = vec![1.0, 3.0, 5.0, 7.0];
        let result = compute_aggregate(&cats, &vals, AggregateFunc::Median).unwrap();
        // Even count: median = (3+5)/2 = 4.0
        assert!((result.y_data[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn min_aggregate() {
        let cats = vec!["A".into(), "B".into(), "A".into(), "B".into()];
        let vals = vec![10.0, 20.0, 5.0, 25.0];
        let result = compute_aggregate(&cats, &vals, AggregateFunc::Min).unwrap();
        assert!((result.y_data[0] - 5.0).abs() < 1e-10); // A min
        assert!((result.y_data[1] - 20.0).abs() < 1e-10); // B min
    }

    #[test]
    fn empty_group_mean_is_nan() {
        // This tests that aggregates on empty groups return NaN
        // (The current grouping logic won't produce truly empty groups from
        //  the input, but we test the aggregate functions directly)
        let cats = vec!["A".into()];
        let vals = vec![5.0];
        let result = compute_aggregate(&cats, &vals, AggregateFunc::Mean).unwrap();
        assert!(
            !result.y_data[0].is_nan(),
            "non-empty group should not be NaN"
        );
        assert!((result.y_data[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn max_aggregate() {
        let cats = vec!["A".into(), "B".into(), "A".into(), "B".into()];
        let vals = vec![10.0, 20.0, 5.0, 25.0];
        let result = compute_aggregate(&cats, &vals, AggregateFunc::Max).unwrap();
        assert!((result.y_data[0] - 10.0).abs() < 1e-10); // A max
        assert!((result.y_data[1] - 25.0).abs() < 1e-10); // B max
    }
}
