// SPDX-License-Identifier: MIT OR Apache-2.0
//! Box plot statistical summary.

use crate::error::{ChartError, Result};

/// Five-number summary + outliers for one category.
#[derive(Clone, Debug)]
pub struct BoxPlotSummary {
    /// Category label.
    pub category: String,
    /// Median (Q2).
    pub median: f64,
    /// First quartile (Q1, 25th percentile).
    pub q1: f64,
    /// Third quartile (Q3, 75th percentile).
    pub q3: f64,
    /// Lower whisker (min value >= Q1 - 1.5*IQR).
    pub whisker_lo: f64,
    /// Upper whisker (max value <= Q3 + 1.5*IQR).
    pub whisker_hi: f64,
    /// Outlier values outside whisker range.
    pub outliers: Vec<f64>,
}

/// Compute box plot summaries grouped by category.
pub fn compute_boxplot(categories: &[String], y_data: &[f64]) -> Result<Vec<BoxPlotSummary>> {
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

    let mut results = Vec::with_capacity(unique_cats.len());
    for cat in &unique_cats {
        let mut values: Vec<f64> = categories
            .iter()
            .zip(y_data.iter())
            .filter(|(c, _)| *c == cat)
            .map(|(_, &v)| v)
            .collect();

        if values.is_empty() {
            continue;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = percentile(&values, 0.5);
        let q1 = percentile(&values, 0.25);
        let q3 = percentile(&values, 0.75);
        let iqr = q3 - q1;
        let lo_fence = q1 - 1.5 * iqr;
        let hi_fence = q3 + 1.5 * iqr;

        let whisker_lo = values
            .iter()
            .copied()
            .find(|&v| v >= lo_fence)
            .unwrap_or(q1);
        let whisker_hi = values
            .iter()
            .rev()
            .copied()
            .find(|&v| v <= hi_fence)
            .unwrap_or(q3);

        let outliers: Vec<f64> = values
            .iter()
            .copied()
            .filter(|&v| v < lo_fence || v > hi_fence)
            .collect();

        results.push(BoxPlotSummary {
            category: cat.clone(),
            median,
            q1,
            q3,
            whisker_lo,
            whisker_hi,
            outliers,
        });
    }

    Ok(results)
}

/// Linear interpolation percentile (exclusive method).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let n = sorted.len() as f64;
    let idx = p * (n - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if lo == hi || hi >= sorted.len() {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_boxplot() {
        let cats: Vec<String> = vec!["A".into(); 9];
        let data: Vec<f64> = (1..=9).map(|x| x as f64).collect();
        let summaries = compute_boxplot(&cats, &data).unwrap();
        assert_eq!(summaries.len(), 1);
        let s = &summaries[0];
        assert!((s.median - 5.0).abs() < 1e-10);
        assert!((s.q1 - 3.0).abs() < 1e-10);
        assert!((s.q3 - 7.0).abs() < 1e-10);
    }

    #[test]
    fn outlier_detection() {
        let mut cats = vec!["A".into(); 10];
        let mut data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        // Add an outlier
        cats.push("A".into());
        data.push(100.0);

        let summaries = compute_boxplot(&cats, &data).unwrap();
        let s = &summaries[0];
        assert!(s.outliers.contains(&100.0));
    }

    #[test]
    fn multi_category() {
        let cats = vec!["A".into(), "B".into(), "A".into(), "B".into(), "A".into(), "B".into()];
        let data = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        let summaries = compute_boxplot(&cats, &data).unwrap();
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].category, "A");
        assert_eq!(summaries[1].category, "B");
        assert!((summaries[0].median - 2.0).abs() < 1e-10);
        assert!((summaries[1].median - 20.0).abs() < 1e-10);
    }

    #[test]
    fn all_same_values() {
        let cats = vec!["X".into(); 5];
        let data = vec![7.0; 5];
        let summaries = compute_boxplot(&cats, &data).unwrap();
        let s = &summaries[0];
        assert!((s.median - 7.0).abs() < 1e-10);
        assert!((s.q1 - 7.0).abs() < 1e-10);
        assert!((s.q3 - 7.0).abs() < 1e-10);
        assert!(s.outliers.is_empty());
    }
}
