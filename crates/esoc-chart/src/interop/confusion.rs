// SPDX-License-Identifier: MIT OR Apache-2.0
//! Confusion matrix visualization.

use crate::express::heatmap;
use crate::grammar::chart::Chart;

/// Extension trait for creating confusion matrix figures.
pub trait ConfusionMatrixExt {
    /// Create a heatmap chart from this confusion matrix.
    fn figure(&self) -> Chart;
}

impl ConfusionMatrixExt for scry_learn::metrics::ConfusionMatrix {
    fn figure(&self) -> Chart {
        let data: Vec<Vec<f64>> = self
            .matrix
            .iter()
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect();

        heatmap(data)
            .annotate()
            .row_labels(self.labels.clone())
            .col_labels(self.labels.clone())
            .title("Confusion Matrix")
            .x_label("Predicted")
            .y_label("True")
            .size(600.0, 600.0)
            .build()
    }
}

/// Create a confusion matrix figure.
pub fn confusion_matrix_figure(cm: &scry_learn::metrics::ConfusionMatrix) -> Chart {
    cm.figure()
}
