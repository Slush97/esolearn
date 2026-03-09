// SPDX-License-Identifier: MIT OR Apache-2.0
//! Classification metrics: accuracy, precision, recall, F1, confusion matrix.

use std::collections::HashMap;
use std::fmt;

/// Averaging strategy for multi-class metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Average {
    /// Binary classification (positive class = 1.0).
    Binary,
    /// Unweighted mean across all classes.
    Macro,
    /// Weighted mean by class support (number of true instances).
    Weighted,
}

/// A confusion matrix.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ConfusionMatrix {
    /// The matrix: `matrix[true_class][predicted_class]`.
    pub matrix: Vec<Vec<usize>>,
    /// Class labels.
    pub labels: Vec<String>,
}

/// Per-class metrics.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ClassMetrics {
    /// Precision for this class.
    pub precision: f64,
    /// Recall for this class.
    pub recall: f64,
    /// F1-score for this class.
    pub f1: f64,
    /// Number of true instances (support).
    pub support: usize,
}

/// A full classification report.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ClassificationReport {
    /// Overall accuracy.
    pub accuracy: f64,
    /// Per-class metrics.
    pub per_class: Vec<(String, ClassMetrics)>,
    /// Macro-averaged metrics.
    pub macro_avg: ClassMetrics,
    /// Weighted-averaged metrics.
    pub weighted_avg: ClassMetrics,
    /// Total number of samples.
    pub total_support: usize,
}

impl fmt::Display for ClassificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:>15} {:>10} {:>10} {:>10} {:>10}",
            "", "precision", "recall", "f1-score", "support"
        )?;
        writeln!(f)?;
        for (label, m) in &self.per_class {
            writeln!(
                f,
                "{:>15} {:>10.4} {:>10.4} {:>10.4} {:>10}",
                label, m.precision, m.recall, m.f1, m.support
            )?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "{:>15} {:>10.4} {:>10.4} {:>10.4} {:>10}",
            "accuracy", "", "", self.accuracy, self.total_support
        )?;
        writeln!(
            f,
            "{:>15} {:>10.4} {:>10.4} {:>10.4} {:>10}",
            "macro avg",
            self.macro_avg.precision,
            self.macro_avg.recall,
            self.macro_avg.f1,
            self.total_support
        )?;
        writeln!(
            f,
            "{:>15} {:>10.4} {:>10.4} {:>10.4} {:>10}",
            "weighted avg",
            self.weighted_avg.precision,
            self.weighted_avg.recall,
            self.weighted_avg.f1,
            self.total_support
        )?;
        Ok(())
    }
}

/// Compute accuracy: fraction of correct predictions.
pub fn accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| (*t - *p).abs() < 1e-6)
        .count();
    correct as f64 / y_true.len() as f64
}

/// Compute precision from a pre-built confusion matrix.
fn precision_from_cm(cm: &ConfusionMatrix, avg: Average) -> f64 {
    let n = cm.matrix.len();
    match avg {
        Average::Binary => {
            let tp = if n >= 2 { cm.matrix[1][1] } else { 0 };
            let fp = (0..n)
                .map(|i| if i == 1 { 0 } else { cm.matrix[i][1] })
                .sum::<usize>();
            if tp + fp == 0 {
                0.0
            } else {
                tp as f64 / (tp + fp) as f64
            }
        }
        Average::Macro => {
            let mut total = 0.0;
            for c in 0..n {
                let tp = cm.matrix[c][c];
                let fp: usize = (0..n)
                    .map(|i| if i == c { 0 } else { cm.matrix[i][c] })
                    .sum();
                total += if tp + fp == 0 {
                    0.0
                } else {
                    tp as f64 / (tp + fp) as f64
                };
            }
            total / n as f64
        }
        Average::Weighted => {
            let mut total = 0.0;
            let mut total_support = 0;
            for c in 0..n {
                let support: usize = cm.matrix[c].iter().sum();
                let tp = cm.matrix[c][c];
                let fp: usize = (0..n)
                    .map(|i| if i == c { 0 } else { cm.matrix[i][c] })
                    .sum();
                let p = if tp + fp == 0 {
                    0.0
                } else {
                    tp as f64 / (tp + fp) as f64
                };
                total += p * support as f64;
                total_support += support;
            }
            if total_support == 0 {
                0.0
            } else {
                total / total_support as f64
            }
        }
    }
}

/// Compute recall from a pre-built confusion matrix.
fn recall_from_cm(cm: &ConfusionMatrix, avg: Average) -> f64 {
    let n = cm.matrix.len();
    match avg {
        Average::Binary => {
            let tp = if n >= 2 { cm.matrix[1][1] } else { 0 };
            let fn_ = if n >= 2 {
                (0..n)
                    .map(|j| if j == 1 { 0 } else { cm.matrix[1][j] })
                    .sum::<usize>()
            } else {
                0
            };
            if tp + fn_ == 0 {
                0.0
            } else {
                tp as f64 / (tp + fn_) as f64
            }
        }
        Average::Macro => {
            let mut total = 0.0;
            for c in 0..n {
                let tp = cm.matrix[c][c];
                let support: usize = cm.matrix[c].iter().sum();
                total += if support == 0 {
                    0.0
                } else {
                    tp as f64 / support as f64
                };
            }
            total / n as f64
        }
        Average::Weighted => {
            let mut total = 0.0;
            let mut total_support = 0;
            for c in 0..n {
                let support: usize = cm.matrix[c].iter().sum();
                let tp = cm.matrix[c][c];
                let r = if support == 0 {
                    0.0
                } else {
                    tp as f64 / support as f64
                };
                total += r * support as f64;
                total_support += support;
            }
            if total_support == 0 {
                0.0
            } else {
                total / total_support as f64
            }
        }
    }
}

/// Compute precision (builds confusion matrix internally).
pub fn precision(y_true: &[f64], y_pred: &[f64], avg: Average) -> f64 {
    let cm = confusion_matrix(y_true, y_pred);
    precision_from_cm(&cm, avg)
}

/// Compute recall (builds confusion matrix internally).
pub fn recall(y_true: &[f64], y_pred: &[f64], avg: Average) -> f64 {
    let cm = confusion_matrix(y_true, y_pred);
    recall_from_cm(&cm, avg)
}

/// Compute F1 score.
///
/// For `Binary`, computes `2 * precision * recall / (precision + recall)`.
/// For `Macro`, computes per-class F1 scores then averages (matching sklearn).
/// For `Weighted`, computes per-class F1 scores then takes a support-weighted average.
pub fn f1_score(y_true: &[f64], y_pred: &[f64], avg: Average) -> f64 {
    let cm = confusion_matrix(y_true, y_pred);
    let n = cm.matrix.len();

    match avg {
        Average::Binary => {
            let p = precision_from_cm(&cm, Average::Binary);
            let r = recall_from_cm(&cm, Average::Binary);
            if p + r == 0.0 {
                0.0
            } else {
                2.0 * p * r / (p + r)
            }
        }
        Average::Macro => {
            let mut total_f1 = 0.0;
            for c in 0..n {
                let tp = cm.matrix[c][c];
                let fp: usize = (0..n)
                    .map(|i| if i == c { 0 } else { cm.matrix[i][c] })
                    .sum();
                let support: usize = cm.matrix[c].iter().sum();
                let p = if tp + fp == 0 {
                    0.0
                } else {
                    tp as f64 / (tp + fp) as f64
                };
                let r = if support == 0 {
                    0.0
                } else {
                    tp as f64 / support as f64
                };
                total_f1 += if p + r == 0.0 {
                    0.0
                } else {
                    2.0 * p * r / (p + r)
                };
            }
            total_f1 / n as f64
        }
        Average::Weighted => {
            let mut total_f1 = 0.0;
            let mut total_support = 0;
            for c in 0..n {
                let tp = cm.matrix[c][c];
                let fp: usize = (0..n)
                    .map(|i| if i == c { 0 } else { cm.matrix[i][c] })
                    .sum();
                let support: usize = cm.matrix[c].iter().sum();
                let p = if tp + fp == 0 {
                    0.0
                } else {
                    tp as f64 / (tp + fp) as f64
                };
                let r = if support == 0 {
                    0.0
                } else {
                    tp as f64 / support as f64
                };
                let f = if p + r == 0.0 {
                    0.0
                } else {
                    2.0 * p * r / (p + r)
                };
                total_f1 += f * support as f64;
                total_support += support;
            }
            if total_support == 0 {
                0.0
            } else {
                total_f1 / total_support as f64
            }
        }
    }
}

/// Build a confusion matrix from true and predicted labels.
pub fn confusion_matrix(y_true: &[f64], y_pred: &[f64]) -> ConfusionMatrix {
    let mut classes: Vec<i64> = y_true
        .iter()
        .chain(y_pred.iter())
        .map(|&v| v as i64)
        .collect();
    classes.sort_unstable();
    classes.dedup();

    let n = classes.len();
    let mut matrix = vec![vec![0usize; n]; n];
    let labels: Vec<String> = classes
        .iter()
        .map(std::string::ToString::to_string)
        .collect();

    // O(1) lookup per sample instead of O(k) linear scan.
    let class_map: HashMap<i64, usize> = classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let ti = class_map.get(&(t as i64)).copied().unwrap_or(0);
        let pi = class_map.get(&(p as i64)).copied().unwrap_or(0);
        matrix[ti][pi] += 1;
    }

    ConfusionMatrix { matrix, labels }
}

/// Generate a full classification report (like sklearn's `classification_report`).
pub fn classification_report(y_true: &[f64], y_pred: &[f64]) -> ClassificationReport {
    let cm = confusion_matrix(y_true, y_pred);
    let n = cm.matrix.len();
    let total: usize = cm.matrix.iter().flat_map(|r| r.iter()).sum();

    let mut per_class = Vec::with_capacity(n);
    let mut macro_p = 0.0;
    let mut macro_r = 0.0;
    let mut macro_f = 0.0;
    let mut weighted_p = 0.0;
    let mut weighted_r = 0.0;
    let mut weighted_f = 0.0;

    for c in 0..n {
        let tp = cm.matrix[c][c];
        let support: usize = cm.matrix[c].iter().sum();
        let fp: usize = (0..n)
            .map(|i| if i == c { 0 } else { cm.matrix[i][c] })
            .sum();

        let p = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let r = if support == 0 {
            0.0
        } else {
            tp as f64 / support as f64
        };
        let f = if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        };

        per_class.push((
            cm.labels[c].clone(),
            ClassMetrics {
                precision: p,
                recall: r,
                f1: f,
                support,
            },
        ));

        macro_p += p;
        macro_r += r;
        macro_f += f;
        weighted_p += p * support as f64;
        weighted_r += r * support as f64;
        weighted_f += f * support as f64;
    }

    let n_f = n as f64;
    let total_f = total as f64;

    ClassificationReport {
        accuracy: accuracy(y_true, y_pred),
        per_class,
        macro_avg: ClassMetrics {
            precision: macro_p / n_f,
            recall: macro_r / n_f,
            f1: macro_f / n_f,
            support: total,
        },
        weighted_avg: ClassMetrics {
            precision: if total > 0 { weighted_p / total_f } else { 0.0 },
            recall: if total > 0 { weighted_r / total_f } else { 0.0 },
            f1: if total > 0 { weighted_f / total_f } else { 0.0 },
            support: total,
        },
        total_support: total,
    }
}

/// Compute log-loss (cross-entropy loss) for probabilistic predictions.
///
/// # Arguments
/// - `y_true` — true class labels (0-indexed integers as f64)
/// - `y_prob` — predicted probability vectors, one per sample
///
/// Probabilities are clipped to `[1e-15, 1 - 1e-15]` to avoid `log(0)`.
pub fn log_loss(y_true: &[f64], y_prob: &[Vec<f64>]) -> f64 {
    if y_true.is_empty() || y_prob.is_empty() {
        return 0.0;
    }
    let eps = 1e-15;
    let n = y_true.len();
    let mut total = 0.0;
    for (i, &label) in y_true.iter().enumerate() {
        let class_idx = label as usize;
        if class_idx < y_prob[i].len() {
            let p = y_prob[i][class_idx].clamp(eps, 1.0 - eps);
            total -= p.ln();
        }
    }
    total / n as f64
}

/// Balanced accuracy: mean per-class recall (macro recall).
///
/// Particularly useful when classes are imbalanced, since it weights
/// each class equally regardless of support.
pub fn balanced_accuracy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let cm = confusion_matrix(y_true, y_pred);
    let n = cm.matrix.len();
    let mut total_recall = 0.0;
    for c in 0..n {
        let support: usize = cm.matrix[c].iter().sum();
        let tp = cm.matrix[c][c];
        total_recall += if support == 0 {
            0.0
        } else {
            tp as f64 / support as f64
        };
    }
    total_recall / n as f64
}

/// Cohen's kappa coefficient — inter-rater agreement adjusted for chance.
///
/// Returns a value in `[-1, 1]` where 1 means perfect agreement,
/// 0 means agreement no better than chance, and negative values
/// mean worse than chance.
pub fn cohen_kappa_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let cm = confusion_matrix(y_true, y_pred);
    let n_classes = cm.matrix.len();
    let total: f64 = cm.matrix.iter().flat_map(|r| r.iter()).sum::<usize>() as f64;
    if total == 0.0 {
        return 0.0;
    }

    // Observed agreement
    let p_o: f64 = (0..n_classes).map(|c| cm.matrix[c][c] as f64).sum::<f64>() / total;

    // Expected agreement by chance
    let mut p_e = 0.0;
    for c in 0..n_classes {
        let row_sum: f64 = cm.matrix[c].iter().sum::<usize>() as f64;
        let col_sum: f64 = (0..n_classes).map(|r| cm.matrix[r][c] as f64).sum::<f64>();
        p_e += (row_sum * col_sum) / (total * total);
    }

    if (1.0 - p_e).abs() < 1e-15 {
        return if (p_o - 1.0).abs() < 1e-15 { 1.0 } else { 0.0 };
    }

    (p_o - p_e) / (1.0 - p_e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect() {
        assert!((accuracy(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_half() {
        assert!((accuracy(&[0.0, 1.0, 0.0, 1.0], &[0.0, 0.0, 0.0, 1.0]) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_binary() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 1.0, 0.0, 1.0];
        let cm = confusion_matrix(&y_true, &y_pred);
        assert_eq!(cm.matrix, vec![vec![1, 1], vec![1, 1]]);
    }

    #[test]
    fn test_classification_report_display() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let y_pred = vec![0.0, 0.0, 1.0, 2.0, 1.0, 2.0];
        let report = classification_report(&y_true, &y_pred);
        let output = format!("{report}");
        assert!(output.contains("accuracy"));
        assert!(output.contains("macro avg"));
    }

    #[test]
    fn test_f1_binary() {
        // TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
        let y_true = vec![0.0, 1.0, 1.0];
        let y_pred = vec![1.0, 1.0, 0.0];
        let f = f1_score(&y_true, &y_pred, Average::Binary);
        assert!((f - 0.5).abs() < 1e-6, "expected F1=0.5, got {f}");
    }

    // -----------------------------------------------------------------------
    // log_loss tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_loss_perfect() {
        // Perfect predictions → each true class has probability 1.0
        let y_true = vec![0.0, 1.0, 2.0];
        let y_prob = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ll = log_loss(&y_true, &y_prob);
        assert!(ll < 1e-10, "perfect log_loss should be ~0, got {ll}");
    }

    #[test]
    fn test_log_loss_random() {
        // Uniform random predictions → log_loss should be ln(3) ≈ 1.099
        let y_true = vec![0.0, 1.0, 2.0];
        let y_prob = vec![
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ];
        let ll = log_loss(&y_true, &y_prob);
        assert!(ll > 0.5, "random log_loss should be positive, got {ll}");
        assert!(
            (ll - 3.0_f64.ln()).abs() < 1e-6,
            "expected ~ln(3), got {ll}"
        );
    }

    // -----------------------------------------------------------------------
    // balanced_accuracy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_balanced_accuracy_perfect() {
        let ba = balanced_accuracy(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]);
        assert!((ba - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_balanced_accuracy_imbalanced() {
        // 90 class-0, 10 class-1. Predict all as 0.
        let mut y_true = vec![0.0; 90];
        y_true.extend(vec![1.0; 10]);
        let y_pred = vec![0.0; 100];

        let raw = accuracy(&y_true, &y_pred);
        let bal = balanced_accuracy(&y_true, &y_pred);

        // Raw accuracy = 0.90, balanced = (1.0 + 0.0)/2 = 0.50
        assert!((raw - 0.90).abs() < 1e-10);
        assert!((bal - 0.50).abs() < 1e-10);
        assert!(bal < raw, "balanced should be lower on imbalanced data");
    }

    // -----------------------------------------------------------------------
    // cohen_kappa tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cohen_kappa_perfect() {
        let kappa = cohen_kappa_score(&[0.0, 1.0, 2.0, 0.0, 1.0], &[0.0, 1.0, 2.0, 0.0, 1.0]);
        assert!(
            (kappa - 1.0).abs() < 1e-10,
            "perfect kappa should be 1.0, got {kappa}"
        );
    }

    #[test]
    fn test_cohen_kappa_chance() {
        // All predict class 0 on balanced data → kappa ≈ 0
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 0.0, 0.0, 0.0];
        let kappa = cohen_kappa_score(&y_true, &y_pred);
        assert!(
            kappa.abs() < 1e-10,
            "chance kappa should be ~0, got {kappa}"
        );
    }

    #[test]
    fn test_cohen_kappa_partial() {
        // Known: 3 agree out of 4 on binary
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_pred = vec![0.0, 0.0, 0.0, 1.0];
        let kappa = cohen_kappa_score(&y_true, &y_pred);
        // p_o = 3/4 = 0.75, row/col sums: [2,2] x [3,1]
        // p_e = (2*3)/(4*4) + (2*1)/(4*4) = 6/16 + 2/16 = 0.5
        // kappa = (0.75 - 0.5) / (1.0 - 0.5) = 0.5
        assert!(
            (kappa - 0.5).abs() < 1e-10,
            "expected kappa=0.5, got {kappa}"
        );
    }
}
