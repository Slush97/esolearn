// SPDX-License-Identifier: MIT OR Apache-2.0
//! Native model wrapper using scry-llm tensors.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::error::Result;
use crate::model::VisionModel;

/// A forward-pass function that takes an input tensor and returns an output tensor.
pub trait ForwardFn<B: MathBackend>: Send + Sync {
    fn forward(&self, input: Tensor<B>) -> Tensor<B>;
    fn output_shape(&self, input_shape: &[usize]) -> Vec<usize>;
}

/// Wraps a scry-llm-based model and exposes it as a [`VisionModel`].
///
/// The generic parameter `F` is the forward function/model struct that knows
/// how to run inference. This will be implemented by concrete model types
/// (YOLO, CLIP, SCRFD, etc.) in later phases.
pub struct NativeModel<B: MathBackend, F: ForwardFn<B>> {
    model: F,
    _marker: std::marker::PhantomData<B>,
}

impl<B: MathBackend, F: ForwardFn<B>> NativeModel<B, F> {
    pub fn new(model: F) -> Self {
        Self {
            model,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: MathBackend + Send + Sync + 'static, F: ForwardFn<B> + 'static> VisionModel for NativeModel<B, F> {
    fn forward(&self, input: &[f32], input_shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = Tensor::<B>::from_vec(input.to_vec(), Shape::new(input_shape));
        let output = self.model.forward(tensor);
        Ok(output.to_vec())
    }

    fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        self.model.output_shape(input_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    struct IdentityModel;

    impl ForwardFn<CpuBackend> for IdentityModel {
        fn forward(&self, input: Tensor<CpuBackend>) -> Tensor<CpuBackend> {
            input
        }

        fn output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
            input_shape.to_vec()
        }
    }

    #[test]
    fn native_model_identity() {
        let model = NativeModel::<CpuBackend, _>::new(IdentityModel);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let output = model.forward(&input, &shape).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn native_model_output_shape() {
        let model = NativeModel::<CpuBackend, _>::new(IdentityModel);
        assert_eq!(model.output_shape(&[1, 3, 224, 224]), vec![1, 3, 224, 224]);
    }
}
