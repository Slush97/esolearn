// SPDX-License-Identifier: MIT OR Apache-2.0
//! Binning transform for histograms.

/// Compute histogram bins from data.
///
/// Returns `(bin_centers, counts)` where each bin center is the midpoint
/// of the bin range and counts is the number of values in each bin.
pub fn compute_bins(data: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    if data.is_empty() {
        return (vec![], vec![]);
    }

    let bins = bins.max(1);

    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Single-value edge case
    if (max - min).abs() < 1e-15 {
        return (vec![min], vec![data.len() as f64]);
    }

    let bin_width = (max - min) / bins as f64;
    let mut counts = vec![0.0; bins];

    for &v in data {
        let mut idx = ((v - min) / bin_width) as usize;
        // Clamp the max value into the last bin
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1.0;
    }

    let centers: Vec<f64> = (0..bins)
        .map(|i| min + (i as f64 + 0.5) * bin_width)
        .collect();

    (centers, counts)
}

/// Default bin count using Sturges' rule: ceil(log2(n)) + 1
pub fn sturges_bins(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    ((n as f64).log2().ceil() as usize + 1).max(1)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn bins_known_data() {
        // 9 values [1..9] into 3 bins: [1,2,3], [4,5,6], [7,8,9]
        let data: Vec<f64> = (1..=9).map(f64::from).collect();
        let (centers, counts) = compute_bins(&data, 3);
        assert_eq!(centers.len(), 3);
        assert_eq!(counts.len(), 3);
        assert_eq!(counts[0], 3.0); // 1,2,3
        assert_eq!(counts[1], 3.0); // 4,5,6
        assert_eq!(counts[2], 3.0); // 7,8,9
    }

    #[test]
    fn sturges_default_n100() {
        let bins = sturges_bins(100);
        assert_eq!(bins, 8); // ceil(log2(100)) + 1 = ceil(6.64) + 1 = 7 + 1 = 8
    }

    #[test]
    fn empty_data() {
        let (centers, counts) = compute_bins(&[], 5);
        assert!(centers.is_empty());
        assert!(counts.is_empty());
    }

    #[test]
    fn single_value() {
        let (centers, counts) = compute_bins(&[42.0, 42.0, 42.0], 5);
        assert_eq!(centers.len(), 1);
        assert_eq!(counts.len(), 1);
        assert_eq!(centers[0], 42.0);
        assert_eq!(counts[0], 3.0);
    }
}
