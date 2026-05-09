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
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_vision::nn::conv2d::Conv2d;

use super::common::{add_channel_bias, add_same, silu, GroupNormParams};
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
    /// Pre-upload every parameter tensor to the backend's device-resident form.
    /// No-op on `CpuBackend`; idempotent on any backend.
    pub fn to_device(&mut self) {
        self.norm1.to_device();
        self.conv1.to_device();
        B::to_device_in_place(&mut self.time_emb_proj_weight.data);
        B::to_device_in_place(&mut self.time_emb_proj_bias.data);
        self.norm2.to_device();
        self.conv2.to_device();
        if let Some(c) = self.conv_shortcut.as_mut() {
            c.to_device();
        }
    }

    /// Forward pass with timestep injection.
    ///
    /// Input is `[in_channels, H, W]`; `time_embed` is `[time_embed_dim]`.
    /// HF's `ResnetBlock2D` runs `nonlinearity → conv` (not `conv →
    /// nonlinearity`), and projects the time embedding through SiLU →
    /// Linear before broadcast-adding it into the channels of the
    /// post-conv1 activation.
    pub fn forward(&mut self, input: &Tensor<B>, time_embed: &Tensor<B>) -> Result<Tensor<B>> {
        use crate::profile::time_section;

        Ok(time_section("resblock.forward", || {
            // ---- Main branch: norm1 → silu → conv1 -----------------
            let h = self.norm1.forward(input);
            let h = silu(&h);
            let h = self.conv1.forward(&h);

            // ---- Time embedding projection (SiLU → Linear → channel-bias add).
            // HF: `temb = nonlinearity(temb); temb = self.time_emb_proj(temb)[:, :, None, None]`
            // — i.e. broadcast into the spatial dims of the post-conv1 activation.
            let t_act = silu(time_embed);
            let t_proj = B::matmul_bias(
                &t_act.data,
                &self.time_emb_proj_weight.data,
                &self.time_emb_proj_bias.data,
                1,
                self.time_embed_dim,
                self.out_channels,
                false,
                false,
            );
            let t_proj_tensor = Tensor::new(t_proj, Shape::new(&[self.out_channels]));
            let h = add_channel_bias(&h, &t_proj_tensor);

            // ---- norm2 → silu → conv2 ------------------------------
            let h = self.norm2.forward(&h);
            let h = silu(&h);
            let h = self.conv2.forward(&h);

            // ---- Residual: 1x1 shortcut conv when channels change, else identity.
            let shortcut = match &self.conv_shortcut {
                Some(c) => c.forward(input),
                None => super::common::clone_tensor(input),
            };
            add_same(&shortcut, &h)
        }))
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
