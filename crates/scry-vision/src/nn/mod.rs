// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vision-specific neural network layers (inference-only).
//!
//! These layers implement `Module<B>` for parameter enumeration and use
//! scry-llm's `MathBackend` for computation.

pub mod batchnorm;
pub mod conv2d;
pub mod patch_embed;
pub mod pool;

pub use batchnorm::BatchNorm2d;
pub use conv2d::Conv2d;
pub use patch_embed::PatchEmbedding;
pub use pool::{AdaptiveAvgPool2d, MaxPool2d};

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

/// Element-wise ReLU: `max(0, x)`.
pub fn relu<B: MathBackend>(input: &Tensor<B>) -> Tensor<B> {
    let data: Vec<f32> = input.to_vec().into_iter().map(|x| x.max(0.0)).collect();
    Tensor::from_vec(data, input.shape.clone())
}
