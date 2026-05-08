// SPDX-License-Identifier: MIT OR Apache-2.0
//! Safetensors weight loading and HF state-dict key mapping.
//!
//! HF Stable Diffusion checkpoints ship as safetensors files with PyTorch
//! state-dict naming conventions. Each subnet (UNet, VAE, text encoder)
//! lives in its own file when downloaded from `runwayml/stable-diffusion-v1-5`
//! / `stabilityai/stable-diffusion-xl-base-1.0` etc.:
//!
//! ```text
//! unet/diffusion_pytorch_model.safetensors
//! vae/diffusion_pytorch_model.safetensors
//! text_encoder/model.safetensors
//! tokenizer/{vocab.json, merges.txt}
//! ```
//!
//! See `HANDOFF.md` M2 for the implementation strategy. Pattern lifted from
//! `scry-vision/src/checkpoint.rs` (memmap → safetensors view → cast to f32).

#[cfg(feature = "safetensors")]
use crate::error::Result;

/// A view over a memory-mapped safetensors checkpoint.
#[cfg(feature = "safetensors")]
pub struct SafetensorsCheckpoint {
    // M2: hold the memmap + a SafeTensors view. Mirror `scry-vision`'s
    // approach so we can cast bf16/fp16/fp32 uniformly.
    _placeholder: (),
}

#[cfg(feature = "safetensors")]
impl SafetensorsCheckpoint {
    /// Memory-map a safetensors file and parse its header.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let _ = path;
        todo!("M2: memmap + SafeTensors::deserialize, mirroring scry-vision/checkpoint.rs")
    }

    /// Read a tensor by name and copy it into a `Vec<f32>` (casting from
    /// bf16/fp16 if needed).
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let _ = name;
        todo!("M2: read + cast to f32")
    }
}

/// HF → scry-diffusion key mapping for the CLIP text encoder. M3 fills this
/// in once the encoder structs exist.
#[cfg(feature = "safetensors")]
pub fn map_clip_text_keys(_hf_key: &str) -> Option<String> {
    todo!("M3: translate HF text_model.encoder.layers.{{i}}.* -> our naming")
}

/// HF → scry-diffusion key mapping for the UNet. M5.
#[cfg(feature = "safetensors")]
pub fn map_unet_keys(_hf_key: &str) -> Option<String> {
    todo!(
        "M5: translate HF unet.down_blocks.{{i}}.resnets.{{j}}.* / .attentions.{{j}}.* / \
         .downsamplers.0.* -> our naming. Same for mid_block, up_blocks."
    )
}

/// HF → scry-diffusion key mapping for the VAE decoder. M4.
#[cfg(feature = "safetensors")]
pub fn map_vae_keys(_hf_key: &str) -> Option<String> {
    todo!("M4: translate HF decoder.up_blocks.{{i}}.* + post_quant_conv -> our naming")
}
