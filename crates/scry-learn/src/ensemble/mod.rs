// SPDX-License-Identifier: MIT OR Apache-2.0
//! Ensemble meta-learning methods.
//!
//! Provides [`VotingClassifier`] for majority/soft voting across multiple
//! models and [`StackingClassifier`] for stacked generalization with a
//! meta-learner trained on out-of-fold predictions.

mod stacking;

pub use stacking::{StackingClassifier, Voting, VotingClassifier};
