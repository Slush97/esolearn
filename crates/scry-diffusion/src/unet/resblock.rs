// SPDX-License-Identifier: MIT OR Apache-2.0
//! UNet residual block.
//!
//! ```text
//!     in ─────────────────────────────────────────────────► +
//!      │                                                    ▲
//!      ├─ GroupNorm → SiLU → Conv2d ─┐                     skip_conv
//!      │                              │                       │
//!      │   time_embed ─ SiLU ─ Linear ┘  (broadcast-add)       │
//!      │                              │                        │
//!      └─ GroupNorm → SiLU → Conv2d ──┴──── (residual) ───────►
//! ```
//!
//! Each ResBlock takes a feature map and a timestep embedding, applies two
//! GroupNorm/SiLU/Conv2d sandwiches with the timestep injected between, and
//! adds a residual (1×1 conv when channel count changes, identity otherwise).
//!
//! Reference: HF `diffusers/src/diffusers/models/resnet.py::ResnetBlock2D`.

use scry_llm::backend::MathBackend;
use scry_llm::tensor::Tensor;

use crate::error::Result;

/// UNet residual block.
pub struct ResBlock<B: MathBackend> {
    /// Input channel count.
    pub in_channels: usize,
    /// Output channel count.
    pub out_channels: usize,
    /// Timestep embedding dim (input to the time-projection Linear).
    pub time_embed_dim: usize,
    // M6: norm1, conv1, time_proj_linear, norm2, conv2, conv_shortcut
    // (only when in_channels != out_channels). Field order mirrors HF
    // ResnetBlock2D for clean key mapping.
    _backend: std::marker::PhantomData<B>,
}

impl<B: MathBackend> ResBlock<B> {
    /// Construct a zero-initialized ResBlock.
    pub fn new(in_channels: usize, out_channels: usize, time_embed_dim: usize) -> Self {
        Self {
            in_channels,
            out_channels,
            time_embed_dim,
            _backend: std::marker::PhantomData,
        }
    }

    /// Forward pass with timestep injection.
    pub fn forward(&mut self, input: &Tensor<B>, time_embed: &Tensor<B>) -> Result<Tensor<B>> {
        let _ = (input, time_embed);
        todo!(
            "M6: GroupNorm + SiLU + Conv2d; broadcast-add timestep proj; GroupNorm + SiLU + \
             Conv2d; residual (with optional 1x1 skip conv)"
        )
    }
}
