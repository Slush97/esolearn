// SPDX-License-Identifier: MIT OR Apache-2.0
//! ONNX Runtime model wrapper.
//!
//! Wraps an `ort::Session` and implements [`VisionModel`].
//! Requires `feature = "onnx"`.

use std::path::Path;
use std::sync::Mutex;

use ort::session::Session;
use ort::value::TensorRef;

use crate::error::{Result, VisionError};
use crate::model::VisionModel;

/// An ONNX Runtime model.
///
/// Wraps a single-input, single-output ONNX model. For multi-input/output
/// models, use the `ort::Session` directly.
///
/// Uses a `Mutex` internally because ort 2.x's `Session::run` requires `&mut self`.
pub struct OnnxModel {
    session: Mutex<Session>,
}

impl OnnxModel {
    /// Load an ONNX model from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_file(path.as_ref()))
            .map_err(|e| VisionError::ModelLoad(e.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }

    /// Load an ONNX model from in-memory bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let session = Session::builder()
            .and_then(|mut b| b.commit_from_memory(bytes))
            .map_err(|e| VisionError::ModelLoad(e.to_string()))?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }
}

impl VisionModel for OnnxModel {
    fn forward(&self, input: &[f32], input_shape: &[usize]) -> Result<Vec<f32>> {
        let shape: Vec<i64> = input_shape.iter().map(|&d| d as i64).collect();
        let tensor = TensorRef::from_array_view((shape.as_slice(), input))
            .map_err(|e| VisionError::Inference(e.to_string()))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| VisionError::Inference("session lock poisoned".into()))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| VisionError::Inference(e.to_string()))?;

        let output = &outputs[0];
        let (_shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| VisionError::Inference(e.to_string()))?;
        Ok(data.to_vec())
    }

    fn output_shape(&self, _input_shape: &[usize]) -> Vec<usize> {
        // ONNX models typically have dynamic output shapes (batch dim = -1),
        // so static shape extraction is unreliable. Return empty.
        vec![]
    }
}
