// SPDX-License-Identifier: MIT OR Apache-2.0
//! K-Nearest Neighbors classifier and regressor.

pub mod kdtree;
mod knn;
pub use kdtree::KdTree;
pub use knn::{Algorithm, DistanceMetric, KnnClassifier, KnnRegressor, WeightFunction};
