// SPDX-License-Identifier: MIT OR Apache-2.0
//! Match result types.

/// A single descriptor match between query and train sets.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DMatch {
    /// Index into the query descriptor set.
    pub query_idx: usize,
    /// Index into the train descriptor set.
    pub train_idx: usize,
    /// Distance between the matched descriptors.
    pub distance: f32,
}
