// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROC curve visualization.

use crate::grammar::chart::Chart;
use crate::grammar::layer::{Layer, MarkType};

/// Extension trait for creating ROC curve figures.
pub trait RocCurveExt {
    /// Create a chart showing this ROC curve.
    fn roc_figure(&self) -> Chart;
}

impl RocCurveExt for scry_learn::metrics::RocCurve {
    fn roc_figure(&self) -> Chart {
        let auc_label = if self.auc.is_nan() {
            "ROC (AUC = N/A)".to_string()
        } else {
            format!("ROC (AUC = {:.3})", self.auc)
        };

        // ROC curve layer
        let roc_layer = Layer::new(MarkType::Line)
            .with_x(self.fpr.clone())
            .with_y(self.tpr.clone())
            .with_label("ROC Curve");

        // Diagonal reference line
        let diag_layer = Layer::new(MarkType::Line)
            .with_x(vec![0.0, 1.0])
            .with_y(vec![0.0, 1.0])
            .with_label("Random Classifier");

        Chart::new()
            .layer(roc_layer)
            .layer(diag_layer)
            .title(auc_label)
            .x_label("False Positive Rate")
            .y_label("True Positive Rate")
            .size(600.0, 600.0)
    }
}

/// Create a ROC curve chart from an `RocCurve`.
pub fn roc_curve_figure(roc: &scry_learn::metrics::RocCurve) -> Chart {
    roc.roc_figure()
}
