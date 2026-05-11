// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared application state for scry-studio.
//!
//! Loaded models live behind `Arc<Mutex<Option<T>>>` so command handlers
//! can hand the work off to `spawn_blocking` without cloning model weights.

use std::sync::Arc;

use parking_lot::Mutex;

#[cfg(not(feature = "scry-gpu-cuda"))]
pub type Backend = scry_llm::backend::cpu::CpuBackend;

#[cfg(feature = "scry-gpu-cuda")]
pub type Backend = scry_llm::backend::scry_gpu::ScryGpuBackend;

pub const BACKEND_NAME: &str = if cfg!(feature = "scry-gpu-cuda") {
    "scry-gpu (CUDA)"
} else {
    "CPU"
};

/// A loaded GPT-2 model and its paired tokenizer.
#[allow(dead_code)]
pub struct LoadedLlm {
    pub model: scry_llm::nn::gpt2::Gpt2Model<Backend>,
    pub tokenizer: scry_llm::tokenizer::HfTokenizer,
    pub config: scry_llm::nn::gpt2::Gpt2Config,
}

pub struct LoadedResnet {
    pub classifier: scry_vision::models::ResNetClassifier<Backend>,
    pub labels: Vec<String>,
}

#[cfg(feature = "onnx")]
pub struct LoadedScrfd {
    pub detector: scry_vision::models::ScrfdDetector,
}

pub type DiffusionPipeline = scry_diffusion::SdPipeline<
    Backend,
    scry_diffusion::text_encoder::clip_text::ClipTextEncoder<Backend>,
    scry_diffusion::scheduler::ddim::DdimScheduler,
>;

pub struct LoadedDiffusion {
    pub pipeline: DiffusionPipeline,
}

#[derive(Default)]
pub struct AppState {
    pub llm: Arc<Mutex<Option<LoadedLlm>>>,
    pub resnet: Arc<Mutex<Option<LoadedResnet>>>,
    #[cfg(feature = "onnx")]
    pub scrfd: Arc<Mutex<Option<LoadedScrfd>>>,
    pub diffusion: Arc<Mutex<Option<LoadedDiffusion>>>,
}

impl AppState {
    pub fn new() -> Self {
        Self::default()
    }
}
