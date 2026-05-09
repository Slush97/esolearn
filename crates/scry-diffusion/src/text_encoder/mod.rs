// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text encoders that produce conditioning for the UNet.
//!
//! SD 1.5 / 2.x use a single CLIP-ViT-L text encoder ([`ClipTextEncoder`]).
//! SDXL concatenates CLIP-ViT-L with OpenCLIP-ViT-bigG into a 2048-d
//! conditioning ([`SdxlTextEncoder`]) and additionally produces a pooled
//! embedding for [`crate::SdxlExtras::pooled`].
//!
//! [`TextEncoder`] abstracts over both so the pipeline doesn't branch on
//! model family.

pub mod clip_text;

use scry_llm::backend::MathBackend;

use crate::conditioning::Conditioning;
use crate::error::Result;

/// Trait implemented by every text encoder variant the pipeline supports.
pub trait TextEncoder<B: MathBackend> {
    /// Run the encoder on a tokenized prompt and return conditioning ready
    /// for the UNet. Tokens must already be padded to the encoder's fixed
    /// sequence length (77 for CLIP).
    fn encode(&mut self, tokens: &[u32]) -> Result<Conditioning<B>>;

    /// Output embedding dimension. SD 1.5 = 768, SDXL = 2048
    /// (concatenation of CLIP-L and OpenCLIP-bigG).
    fn d_model(&self) -> usize;

    /// Pre-upload every parameter tensor to the backend's device-resident
    /// form. Default no-op for encoders that don't carry tensor parameters
    /// (or whose backend has no device residency, e.g. `CpuBackend`).
    fn to_device(&mut self) {}
}
