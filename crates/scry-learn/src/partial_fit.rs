// SPDX-License-Identifier: MIT OR Apache-2.0
//! Incremental (online) learning trait.
//!
//! Models implementing [`PartialFit`] can be trained on data that arrives
//! in batches, without requiring all data in memory at once.
//!
//! # Example
//!
//! ```ignore
//! use scry_learn::prelude::*;
//!
//! let mut model = LogisticRegression::new()
//!     .solver(Solver::GradientDescent);
//! for batch in data_stream.chunks(10_000) {
//!     model.partial_fit(&batch)?;
//! }
//! let preds = model.predict(&test_features)?;
//! ```

use crate::dataset::Dataset;
use crate::error::Result;

/// Trait for models that support incremental (online) learning.
///
/// State from previous `partial_fit` calls is preserved and updated —
/// the model does **not** restart from scratch.
///
/// # Supported models
///
/// | Model | How it works |
/// |-------|-------------|
/// | `LogisticRegression` (GD) | One epoch of gradient descent per batch |
/// | `GaussianNb` | Accumulates sufficient statistics |
/// | `MiniBatchKMeans` | Streaming centroid updates |
/// | `MLPClassifier` | One epoch of mini-batch SGD |
/// | `MLPRegressor` | One epoch of mini-batch SGD |
///
/// Trees, Random Forest, and GBT are inherently batch algorithms and do
/// **not** support `partial_fit`.
pub trait PartialFit {
    /// Train incrementally on a single batch of data.
    ///
    /// On the first call, the model is initialized from the data dimensions
    /// and class structure. Subsequent calls update the existing model state.
    fn partial_fit(&mut self, data: &Dataset) -> Result<()>;

    /// Whether the model has been initialized (at least one `partial_fit` call).
    fn is_initialized(&self) -> bool;
}
