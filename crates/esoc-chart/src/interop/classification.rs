// SPDX-License-Identifier: MIT OR Apache-2.0
//! Classification report visualization.

use crate::express::grouped_bar;
use crate::grammar::chart::Chart;
use crate::new_theme::NewTheme;

/// Extension trait for creating classification report figures.
pub trait ClassificationReportExt {
    /// Create a grouped bar chart from this classification report.
    fn figure(&self) -> Chart;

    /// Create a grouped bar chart from this classification report with a custom theme.
    fn figure_with_theme(&self, theme: NewTheme) -> Chart;
}

impl ClassificationReportExt for scry_learn::metrics::ClassificationReport {
    fn figure(&self) -> Chart {
        self.figure_with_theme(NewTheme::default())
    }

    fn figure_with_theme(&self, theme: NewTheme) -> Chart {
        let labels: Vec<String> = self
            .per_class
            .iter()
            .map(|(l, _)| {
                // Prefix purely numeric class labels for readability on the chart axis
                if l.parse::<f64>().is_ok() {
                    format!("Class {l}")
                } else {
                    l.clone()
                }
            })
            .collect();
        let precisions: Vec<f64> = self.per_class.iter().map(|(_, m)| m.precision).collect();
        let recalls: Vec<f64> = self.per_class.iter().map(|(_, m)| m.recall).collect();
        let f1s: Vec<f64> = self.per_class.iter().map(|(_, m)| m.f1).collect();

        let mut categories = Vec::with_capacity(labels.len() * 3);
        let mut groups = Vec::with_capacity(labels.len() * 3);
        let mut values = Vec::with_capacity(labels.len() * 3);

        for (i, label) in labels.iter().enumerate() {
            categories.push(label.clone());
            groups.push("Precision".to_string());
            values.push(precisions[i]);

            categories.push(label.clone());
            groups.push("Recall".to_string());
            values.push(recalls[i]);

            categories.push(label.clone());
            groups.push("F1-score".to_string());
            values.push(f1s[i]);
        }

        grouped_bar(&categories, &groups, &values)
            .title(format!(
                "Classification Report (accuracy: {:.3})",
                self.accuracy
            ))
            .x_label("Class")
            .y_label("Score")
            .theme(theme)
            .size(800.0, 500.0)
            .build()
    }
}

/// Create a classification report figure.
pub fn classification_report_figure(report: &scry_learn::metrics::ClassificationReport) -> Chart {
    report.figure()
}
