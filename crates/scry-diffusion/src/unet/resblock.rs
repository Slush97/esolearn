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
use scry_vision::nn::conv2d::Conv2d;

use super::common::GroupNormParams;
use crate::error::Result;

/// UNet residual block.
pub struct ResBlock<B: MathBackend> {
    /// Input channel count.
    pub in_channels: usize,
    /// Output channel count.
    pub out_channels: usize,
    /// Timestep embedding dim (input to the time-projection Linear).
    pub time_embed_dim: usize,

    pub(crate) norm1: GroupNormParams<B>,
    pub(crate) conv1: Conv2d<B>,
    /// Time projection `Linear(time_embed_dim, out_channels)` in scry-llm
    /// `[in, out]` convention. SiLU is applied to the time embedding *before*
    /// this projection (HF `nonlinearity(temb)` then `time_emb_proj(temb)`).
    pub(crate) time_emb_proj_weight: Tensor<B>,
    pub(crate) time_emb_proj_bias: Tensor<B>,
    pub(crate) norm2: GroupNormParams<B>,
    pub(crate) conv2: Conv2d<B>,
    /// 1×1 channel-matching conv on the residual when `in != out`.
    /// Identity skip when `None`.
    pub(crate) conv_shortcut: Option<Conv2d<B>>,
}

impl<B: MathBackend> ResBlock<B> {
    /// Forward pass with timestep injection.
    pub fn forward(&mut self, input: &Tensor<B>, time_embed: &Tensor<B>) -> Result<Tensor<B>> {
        let _ = (input, time_embed);
        todo!(
            "M6: GroupNorm + SiLU + Conv2d; broadcast-add timestep proj; GroupNorm + SiLU + \
             Conv2d; residual (with optional 1x1 skip conv)"
        )
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> ResBlock<B> {
    /// Load one ResBlock at `prefix.*` (e.g.
    /// `down_blocks.0.resnets.0`). When `in_channels != out_channels`,
    /// `prefix.conv_shortcut.{weight,bias}` is required; otherwise it
    /// must be absent.
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        time_embed_dim: usize,
        num_norm_groups: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use scry_vision::checkpoint::load_conv2d_with_bias;

        use super::common::{load_group_norm, load_linear};
        use crate::error::Error;

        let norm1 = load_group_norm::<B>(
            view,
            &format!("{prefix}.norm1"),
            in_channels,
            num_norm_groups,
            consume,
        )?;
        let conv1 = load_conv2d_with_bias::<B>(
            view,
            &format!("{prefix}.conv1"),
            in_channels,
            out_channels,
            3,
            3,
            1,
            1,
        )
        .map_err(|e| Error::Llm(format!("load {prefix}.conv1: {e}")))?;
        consume(&format!("{prefix}.conv1.weight"));
        consume(&format!("{prefix}.conv1.bias"));

        let (time_emb_proj_weight, time_emb_proj_bias) = load_linear::<B>(
            view,
            &format!("{prefix}.time_emb_proj"),
            time_embed_dim,
            out_channels,
            true,
            consume,
        )?;

        let norm2 = load_group_norm::<B>(
            view,
            &format!("{prefix}.norm2"),
            out_channels,
            num_norm_groups,
            consume,
        )?;
        let conv2 = load_conv2d_with_bias::<B>(
            view,
            &format!("{prefix}.conv2"),
            out_channels,
            out_channels,
            3,
            3,
            1,
            1,
        )
        .map_err(|e| Error::Llm(format!("load {prefix}.conv2: {e}")))?;
        consume(&format!("{prefix}.conv2.weight"));
        consume(&format!("{prefix}.conv2.bias"));

        let conv_shortcut = if in_channels == out_channels {
            None
        } else {
            let key = format!("{prefix}.conv_shortcut");
            let conv =
                load_conv2d_with_bias::<B>(view, &key, in_channels, out_channels, 1, 1, 1, 0)
                    .map_err(|e| Error::Llm(format!("load {key}: {e}")))?;
            consume(&format!("{key}.weight"));
            consume(&format!("{key}.bias"));
            Some(conv)
        };

        Ok(Self {
            in_channels,
            out_channels,
            time_embed_dim,
            norm1,
            conv1,
            time_emb_proj_weight,
            time_emb_proj_bias,
            norm2,
            conv2,
            conv_shortcut,
        })
    }
}
