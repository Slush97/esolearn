// SPDX-License-Identifier: MIT OR Apache-2.0
//! Diffusion-model inference on the scry stack.
//!
//! Phase 2 of the scry-vision GPU-residency roadmap: latent-diffusion image
//! generators. Stable Diffusion 1.5 is the first target; SDXL, SD-Turbo, and
//! related variants sit on top of the same building blocks (text encoder →
//! UNet → scheduler → VAE decoder). Every component goes through
//! [`scry_llm::backend::MathBackend`], so the same pipeline runs on CPU,
//! Vulkan, or CUDA without rewrites.
//!
//! See `HANDOFF.md` at the crate root for the milestone plan and where the
//! still-empty modules need real implementations.

#![warn(missing_docs)]
// The scaffold leans on acronyms (SD, SDXL, ViT, CLIP, UNet, VAE, NCHW, …)
// in module / item docs; treating them as code spans would just be noise.
#![allow(clippy::doc_markdown)]
// Constructors that store their config by-value trip this needlessly.
#![allow(clippy::needless_pass_by_value)]
// `to_device` is the workspace convention for "upload tensors to backend
// device" — semantically a mutation, not a value-consuming conversion.
// `&mut self` is correct here even though clippy expects `to_*` to take
// self by value.
#![allow(clippy::wrong_self_convention)]

pub mod conditioning;
pub mod error;
pub mod ops;
pub mod pipeline;
pub mod profile;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod unet;
pub mod vae;
pub mod weights;

pub use conditioning::{Conditioning, SdxlExtras};
pub use error::{Error, Result};
pub use pipeline::{GenerationParams, Img2ImgParams, SdPipeline};
