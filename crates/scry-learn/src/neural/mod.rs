// SPDX-License-Identifier: MIT OR Apache-2.0
//! Neural network module — MLP and CNN layers.
//!
//! Provides [`MLPClassifier`] and [`MLPRegressor`] with an sklearn-compatible
//! builder API, GPU-accelerated forward pass, and training history tracking.
//!
//! Also provides CNN building blocks: [`Conv2D`], [`MaxPool2D`], [`Flatten`],
//! and the [`Layer`] trait for composing custom architectures.
//!
//! # GPU Acceleration
//!
//! When a GPU compute backend is available, the **forward pass** is dispatched
//! to it for MLP networks whose `batch × max_layer_dim` exceeds the internal
//! `GPU_THRESHOLD` (4096). The **backward pass is always executed on the CPU**
//! regardless of backend availability — gradient computation has not been
//! ported to GPU yet. This means training speed is bounded by CPU backward-pass
//! throughput even when a GPU accelerates inference.
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::prelude::*;
//!
//! let data = Dataset::from_csv("iris.csv", "species")?;
//! let (train, test) = train_test_split(&data, 0.2, 42);
//!
//! let mut clf = MLPClassifier::new()
//!     .hidden_layers(&[100, 50])
//!     .activation(Activation::Relu)
//!     .learning_rate(0.001)
//!     .seed(42);
//! clf.fit(&train)?;
//!
//! let preds = clf.predict(&test.features_row_major())?;
//! let acc = accuracy(&test.target, &preds);
//! println!("Accuracy: {acc:.2}%");
//!
//! // Inspect training history
//! let loss = clf.history().unwrap().epochs.last().unwrap().train_loss;
//! println!("Final loss: {loss:.4}");
//! ```

pub mod activation;
pub mod callback;
pub mod classifier;
#[cfg(feature = "experimental")]
pub mod conv;
pub mod dropout;
#[cfg(feature = "experimental")]
pub mod flatten;
pub(crate) mod layer;
pub(crate) mod network;
pub(crate) mod optimizer;
#[cfg(feature = "experimental")]
pub mod pool;
pub mod regressor;
pub mod traits;

pub use activation::Activation;
pub use callback::{CallbackAction, EpochMetrics, TrainingCallback, TrainingHistory};
pub use classifier::MLPClassifier;
#[cfg(feature = "experimental")]
pub use conv::Conv2D;
pub use dropout::DropoutLayer;
#[cfg(feature = "experimental")]
pub use flatten::Flatten;
pub use optimizer::{LearningRateSchedule, OptimizerKind};
#[cfg(feature = "experimental")]
pub use pool::MaxPool2D;
pub use regressor::MLPRegressor;
pub use traits::{BackwardOutput, Layer};
