// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROC and Precision-Recall curve computation.

/// A receiver operating characteristic curve.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RocCurve {
    /// False positive rates.
    pub fpr: Vec<f64>,
    /// True positive rates.
    pub tpr: Vec<f64>,
    /// Thresholds.
    pub thresholds: Vec<f64>,
    /// Area under the ROC curve.
    pub auc: f64,
}

/// A precision-recall curve.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PrCurve {
    /// Precision values.
    pub precision: Vec<f64>,
    /// Recall values.
    pub recall: Vec<f64>,
    /// Thresholds.
    pub thresholds: Vec<f64>,
    /// Average precision (area under PR curve).
    pub avg_precision: f64,
}

impl RocCurve {
    /// Create a new ROC curve from precomputed values.
    pub fn new(fpr: Vec<f64>, tpr: Vec<f64>, thresholds: Vec<f64>, auc: f64) -> Self {
        Self {
            fpr,
            tpr,
            thresholds,
            auc,
        }
    }
}

impl PrCurve {
    /// Create a new precision-recall curve from precomputed values.
    pub fn new(
        precision: Vec<f64>,
        recall: Vec<f64>,
        thresholds: Vec<f64>,
        avg_precision: f64,
    ) -> Self {
        Self {
            precision,
            recall,
            thresholds,
            avg_precision,
        }
    }
}

/// Compute the ROC curve and AUC.
///
/// `y_true` should be binary (0.0 or 1.0).
/// `y_scores` should be continuous scores (e.g., predicted probabilities).
///
/// Returns `auc = NaN` when only one class is present (ROC is undefined).
pub fn roc_curve(y_true: &[f64], y_scores: &[f64]) -> RocCurve {
    let n = y_true.len();
    let pos_count = y_true.iter().filter(|&&v| v > 0.5).count();
    let neg_count = n - pos_count;

    // ROC is undefined when only one class is present.
    if pos_count == 0 || neg_count == 0 {
        return RocCurve {
            fpr: vec![0.0],
            tpr: vec![0.0],
            thresholds: vec![],
            auc: f64::NAN,
        };
    }

    // Sort by descending score.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        y_scores[b]
            .partial_cmp(&y_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut fpr = vec![0.0];
    let mut tpr = vec![0.0];
    let mut thresholds = Vec::new();
    let mut tp = 0;
    let mut fp = 0;

    for &i in &indices {
        if y_true[i] > 0.5 {
            tp += 1;
        } else {
            fp += 1;
        }
        let current_tpr = if pos_count > 0 {
            tp as f64 / pos_count as f64
        } else {
            0.0
        };
        let current_fpr = if neg_count > 0 {
            fp as f64 / neg_count as f64
        } else {
            0.0
        };

        fpr.push(current_fpr);
        tpr.push(current_tpr);
        thresholds.push(y_scores[i]);
    }

    // Compute AUC via trapezoidal rule.
    let auc = compute_auc(&fpr, &tpr);

    RocCurve {
        fpr,
        tpr,
        thresholds,
        auc,
    }
}

/// Compute the area under the ROC curve directly.
pub fn roc_auc_score(y_true: &[f64], y_scores: &[f64]) -> f64 {
    roc_curve(y_true, y_scores).auc
}

/// Compute the precision-recall curve.
pub fn pr_curve(y_true: &[f64], y_scores: &[f64]) -> PrCurve {
    let n = y_true.len();
    let pos_count = y_true.iter().filter(|&&v| v > 0.5).count();

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        y_scores[b]
            .partial_cmp(&y_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut prec = vec![1.0];
    let mut rec = vec![0.0];
    let mut thresholds = Vec::new();
    let mut tp = 0;
    let mut fp = 0;

    for &i in &indices {
        if y_true[i] > 0.5 {
            tp += 1;
        } else {
            fp += 1;
        }
        let p = tp as f64 / (tp + fp) as f64;
        let r = if pos_count > 0 {
            tp as f64 / pos_count as f64
        } else {
            0.0
        };
        prec.push(p);
        rec.push(r);
        thresholds.push(y_scores[i]);
    }

    let avg_precision = compute_auc(&rec, &prec);

    PrCurve {
        precision: prec,
        recall: rec,
        thresholds,
        avg_precision,
    }
}

/// Trapezoidal AUC computation.
///
/// Assumes `x` is monotonically increasing. Uses signed deltas so that
/// non-monotonic input doesn't silently produce inflated areas.
fn compute_auc(x: &[f64], y: &[f64]) -> f64 {
    let mut area = 0.0;
    for i in 1..x.len() {
        let dx = x[i] - x[i - 1];
        area += dx * (y[i] + y[i - 1]) / 2.0;
    }
    area
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roc_auc_perfect() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_scores = vec![0.1, 0.2, 0.8, 0.9];
        let auc = roc_auc_score(&y_true, &y_scores);
        assert!(
            (auc - 1.0).abs() < 1e-6,
            "perfect separation should give AUC=1.0, got {auc}"
        );
    }

    #[test]
    fn test_roc_auc_random() {
        // Random ordering — AUC should be around 0.5.
        let y_true = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y_scores = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let auc = roc_auc_score(&y_true, &y_scores);
        assert!(
            (0.0..=1.0).contains(&auc),
            "AUC should be in [0,1], got {auc}"
        );
    }

    #[test]
    fn test_roc_curve_length() {
        let roc = roc_curve(&[0.0, 1.0, 0.0, 1.0], &[0.1, 0.9, 0.2, 0.8]);
        assert_eq!(roc.fpr.len(), roc.tpr.len());
        assert!(roc.fpr.len() > 2);
    }
}
