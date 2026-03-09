// SPDX-License-Identifier: MIT OR Apache-2.0
//! Classification, regression, and clustering metrics.

mod classification;
mod clustering;
mod regression;
mod roc;

pub use classification::{
    accuracy, balanced_accuracy, classification_report, cohen_kappa_score, confusion_matrix,
    f1_score, log_loss, precision, recall, Average, ClassMetrics, ClassificationReport,
    ConfusionMatrix,
};
pub use clustering::{adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score};
pub use regression::{
    explained_variance_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score, root_mean_squared_error,
};
pub use roc::{pr_curve, roc_auc_score, roc_curve, PrCurve, RocCurve};
