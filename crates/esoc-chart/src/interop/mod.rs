// SPDX-License-Identifier: MIT OR Apache-2.0
//! scry-learn integration (behind `scry-learn` feature flag).

pub mod classification;
pub mod confusion;
pub mod dataset;
pub mod roc;

pub use classification::{classification_report_figure, ClassificationReportExt};
pub use confusion::{confusion_matrix_figure, ConfusionMatrixExt};
pub use dataset::{scatter_dataset, DatasetExt};
pub use roc::{roc_curve_figure, RocCurveExt};
