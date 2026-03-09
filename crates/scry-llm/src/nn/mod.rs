pub mod attention;
pub mod embedding;
pub mod gpt2;
pub mod init;
pub mod kv_cache;
pub mod layernorm;
pub mod llama;
pub mod linear;
pub mod mlp;
pub mod rmsnorm;
pub mod transformer;

use crate::backend::MathBackend;
use crate::tensor::Tensor;

/// Trait for neural network modules (inference only).
pub trait Module<B: MathBackend> {
    fn parameters(&self) -> Vec<&Tensor<B>>;
}
