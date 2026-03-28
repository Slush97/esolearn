// SPDX-License-Identifier: MIT OR Apache-2.0
//! Model abstraction layer.
//!
//! [`VisionModel`] is an object-safe trait that abstracts over native
//! (scry-llm `Tensor<B>`) and ONNX Runtime backends using `&[f32]` I/O.
//! The copy overhead is negligible compared to inference time.
//!
//! Implementations:
//! - [`NativeModel`] — wraps a scry-llm `Module<B>` with a forward function
//! - [`OnnxModel`] — wraps an `ort::Session` (requires `feature = "onnx"`)

pub mod native;
#[cfg(feature = "onnx")]
pub mod onnx;
#[cfg(test)]
pub(crate) mod mock;

pub use native::NativeModel;
#[cfg(feature = "onnx")]
pub use onnx::OnnxModel;

use crate::error::Result;

/// Object-safe model trait using f32 slices.
///
/// This trait uses `&[f32]` rather than `Tensor<B>` so it can be used as
/// `Box<dyn VisionModel>` — enabling runtime selection between native and
/// ONNX backends.
pub trait VisionModel: Send + Sync {
    /// Run a forward pass.
    ///
    /// - `input` — flattened f32 input tensor (e.g., `[C, H, W]` for a single image)
    /// - `input_shape` — the shape of the input tensor (e.g., `[1, 3, 224, 224]`)
    ///
    /// Returns the flattened output tensor.
    fn forward(&self, input: &[f32], input_shape: &[usize]) -> Result<Vec<f32>>;

    /// Compute the output shape for a given input shape (without running inference).
    fn output_shape(&self, input_shape: &[usize]) -> Vec<usize>;
}
