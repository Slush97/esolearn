// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dataset visualization helpers.

use crate::express::{histogram, scatter};
use crate::grammar::chart::Chart;

/// Extension trait for creating charts from scry-learn datasets.
pub trait DatasetExt {
    /// Create a scatter plot of two features.
    fn scatter_figure(&self, x_feature: usize, y_feature: usize) -> Chart;

    /// Create a histogram of a single feature.
    fn histogram_figure(&self, feature: usize) -> Chart;
}

impl DatasetExt for scry_learn::dataset::Dataset {
    fn scatter_figure(&self, x_feature: usize, y_feature: usize) -> Chart {
        let x = &self.features[x_feature];
        let y = &self.features[y_feature];

        let x_name = self
            .feature_names
            .get(x_feature)
            .cloned()
            .unwrap_or_default();
        let y_name = self
            .feature_names
            .get(y_feature)
            .cloned()
            .unwrap_or_default();

        if let Some(labels) = &self.class_labels {
            let class_labels: Vec<String> = self
                .target
                .iter()
                .map(|&v| {
                    labels
                        .get(v as usize)
                        .cloned()
                        .unwrap_or_else(|| format!("class {v}"))
                })
                .collect();

            scatter(x, y)
                .color_by(&class_labels)
                .title(format!("{x_name} vs {y_name}"))
                .x_label(&x_name)
                .y_label(&y_name)
                .build()
        } else {
            scatter(x, y)
                .title(format!("{x_name} vs {y_name}"))
                .x_label(&x_name)
                .y_label(&y_name)
                .build()
        }
    }

    fn histogram_figure(&self, feature: usize) -> Chart {
        let data = &self.features[feature];
        let name = self.feature_names.get(feature).cloned().unwrap_or_default();

        histogram(data)
            .title(&name)
            .x_label(&name)
            .y_label("Count")
            .build()
    }
}

/// Create a scatter plot figure from a dataset.
pub fn scatter_dataset(
    dataset: &scry_learn::dataset::Dataset,
    x_feature: usize,
    y_feature: usize,
) -> Chart {
    dataset.scatter_figure(x_feature, y_feature)
}
