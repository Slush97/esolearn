// SPDX-License-Identifier: MIT OR Apache-2.0
//! Anomaly detection algorithms.
//!
//! Currently provides [`IsolationForest`] for unsupervised outlier detection
//! using random isolation trees.

mod iforest;

pub use iforest::IsolationForest;
