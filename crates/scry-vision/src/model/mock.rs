// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mock model for testing pipelines without real weights.

use crate::error::Result;
use crate::model::VisionModel;

/// A mock [`VisionModel`] that returns a fixed output regardless of input.
///
/// This is the standard test helper for any pipeline that takes a
/// `Box<dyn VisionModel>` — inject a `MockModel` with the expected output
/// tensor (flattened) to test preprocessing and postprocessing in isolation.
pub(crate) struct MockModel {
    pub output: Vec<f32>,
}

impl VisionModel for MockModel {
    fn forward(&self, _input: &[f32], _input_shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self.output.clone())
    }

    fn output_shape(&self, _input_shape: &[usize]) -> Vec<usize> {
        vec![self.output.len()]
    }
}
