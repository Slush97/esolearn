// SPDX-License-Identifier: MIT OR Apache-2.0
//! UNet predicting noise (or v-prediction) per denoising step.
//!
//! Topology:
//! ```text
//!     conv_in
//!         │
//!     down_blocks[0..4]  ── DownBlock { 2× ResBlock + optional self+cross-attn + downsample }
//!         │
//!     mid_block          ── ResBlock + self+cross-attn + ResBlock
//!         │
//!     up_blocks[0..4]    ── UpBlock   { 3× ResBlock + optional self+cross-attn + upsample }
//!         │
//!     conv_norm_out (GroupNorm) → SiLU → conv_out
//! ```
//!
//! For SD 1.5: 4 down blocks, 1 mid block, 4 up blocks, channel multiplier
//! `[1, 2, 4, 4]` × `block_out_channels=320` (first block has 320, then 640,
//! 1280, 1280). Cross-attention sits in down blocks 0..2 and up blocks 1..3,
//! with 8 heads per attention block.
//!
//! For SDXL: same shape but 4 up/down (one is identity), bigger channels
//! (`[1, 2, 4]` × 320), more transformer layers per attention block, and
//! the timestep MLP also takes the SDXL extras (size, crop, target).
//!
//! Submodules:
//! - [`config`] — `UnetConfig` (parameterized so SDXL is a config swap).
//! - [`resblock`] — Residual block (GroupNorm + SiLU + Conv2d sandwich).
//! - [`attention`] — Spatial transformer block (self-attention + cross-attention).
//! - [`blocks`] — DownBlock / MidBlock / UpBlock orchestration.

pub mod attention;
pub mod blocks;
pub(crate) mod common;
pub mod config;
pub mod resblock;

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::conditioning::Conditioning;
use crate::error::{Error, Result};
use crate::ops::sinusoidal_timestep_embedding;

pub use config::UnetConfig;

use self::blocks::{DownBlock, MidBlock, UpBlock};
use self::common::{clone_tensor, silu, GroupNormParams};
use scry_vision::nn::conv2d::Conv2d;

/// Sinusoidal timestep embedding MLP `linear_1 → SiLU → linear_2`.
/// Inputs: a `[block_out_channels[0]]` sinusoidal embed; outputs the
/// `[time_embed_dim]` vector consumed by every ResBlock's `time_emb_proj`.
#[allow(clippy::struct_field_names)]
pub(crate) struct TimeEmbedding<B: MathBackend> {
    /// `linear_1.weight` in scry-llm `[in, out]` convention.
    pub(crate) linear_1_weight: Tensor<B>,
    pub(crate) linear_1_bias: Tensor<B>,
    pub(crate) linear_2_weight: Tensor<B>,
    pub(crate) linear_2_bias: Tensor<B>,
}

/// SD UNet predicting per-step noise (or v-prediction, depending on schedule).
pub struct Unet<B: MathBackend> {
    /// Architecture config.
    pub config: UnetConfig,

    pub(crate) conv_in: Conv2d<B>,
    pub(crate) time_embed: TimeEmbedding<B>,
    pub(crate) down_blocks: Vec<DownBlock<B>>,
    pub(crate) mid_block: MidBlock<B>,
    pub(crate) up_blocks: Vec<UpBlock<B>>,
    pub(crate) conv_norm_out: GroupNormParams<B>,
    pub(crate) conv_out: Conv2d<B>,
}

impl<B: MathBackend> Unet<B> {
    /// Forward pass: `(noisy_latent, timestep, conditioning) → predicted_noise`.
    ///
    /// `noisy_latent` is `[in_channels, h, w]` (batch=1 squeezed) — for SD 1.5
    /// latents are `[4, height/8, width/8]`. With CFG enabled the pipeline
    /// calls `forward` twice (uncond + cond) and combines the outputs.
    ///
    /// The forward mirrors HF `UNet2DConditionModel.forward`:
    ///
    /// 1. `conv_in` lifts the latent to `block_out_channels[0]` features.
    /// 2. `time_embedding(silu(linear_1(sin_emb(t))))` produces a
    ///    `[time_embed_dim]` vector consumed by every ResBlock.
    /// 3. Down blocks emit per-layer skip activations (one per
    ///    resnet/attention pair, plus one per downsampler). The conv_in
    ///    output itself is the first skip, matching diffusers'
    ///    `down_block_res_samples = (sample,)`.
    /// 4. Mid block (resnet → spatial-attention → resnet) at the deepest stage.
    /// 5. Up blocks pop skips in LIFO order, concat-along-channels at each
    ///    resnet input.
    /// 6. `conv_norm_out → SiLU → conv_out` projects back to
    ///    `out_channels` (= 4 for SD's noise-prediction head).
    pub fn forward(
        &mut self,
        noisy_latent: &Tensor<B>,
        timestep: f32,
        conditioning: &Conditioning<B>,
    ) -> Result<Tensor<B>> {
        let dims = noisy_latent.shape.dims();
        if dims.len() != 3 || dims[0] != self.config.in_channels {
            return Err(Error::Llm(format!(
                "unet forward: expected [{}, H, W] latent, got {:?}",
                self.config.in_channels, dims
            )));
        }

        // ---- 1. conv_in ---------------------------------------------
        let mut x = self.conv_in.forward(noisy_latent);

        // ---- 2. timestep embedding ---------------------------------
        // `block_out_channels[0]` sized sinusoidal embed → linear → silu →
        // linear → `[time_embed_dim]`.
        let bc0 = self.config.block_out_channels[0];
        let t_sin = sinusoidal_timestep_embedding::<B>(timestep, bc0, 10_000.0)?;
        let t1 = B::matmul_bias(
            &t_sin.data,
            &self.time_embed.linear_1_weight.data,
            &self.time_embed.linear_1_bias.data,
            1,
            bc0,
            self.config.time_embed_dim,
            false,
            false,
        );
        let t1_t: Tensor<B> = Tensor::new(t1, Shape::new(&[self.config.time_embed_dim]));
        let t1_silu = silu(&t1_t);
        let t2 = B::matmul_bias(
            &t1_silu.data,
            &self.time_embed.linear_2_weight.data,
            &self.time_embed.linear_2_bias.data,
            1,
            self.config.time_embed_dim,
            self.config.time_embed_dim,
            false,
            false,
        );
        let time_embed = Tensor::new(t2, Shape::new(&[self.config.time_embed_dim]));

        // ---- 3. down blocks ----------------------------------------
        // Diffusers' `down_block_res_samples = (sample,) + ...` — the
        // conv_in output itself is the first skip.
        let mut skips: Vec<Tensor<B>> = Vec::with_capacity(16);
        skips.push(clone_tensor(&x));
        for db in &mut self.down_blocks {
            let (out, mut emitted) = db.forward(&x, &time_embed, conditioning)?;
            x = out;
            skips.append(&mut emitted);
        }

        // ---- 4. mid block ------------------------------------------
        x = self.mid_block.forward(&x, &time_embed, conditioning)?;

        // ---- 5. up blocks ------------------------------------------
        for ub in &mut self.up_blocks {
            x = ub.forward(&x, &mut skips, &time_embed, conditioning)?;
        }
        if !skips.is_empty() {
            return Err(Error::Llm(format!(
                "unet forward: {} skips left unused after up blocks",
                skips.len()
            )));
        }

        // ---- 6. final norm + silu + conv_out -----------------------
        x = self.conv_norm_out.forward(&x);
        let x_silu = silu(&x);
        Ok(self.conv_out.forward(&x_silu))
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> Unet<B> {
    /// Load a UNet from a HF `unet/diffusion_pytorch_model.safetensors`.
    ///
    /// SD 1.5 ships 686 keys: 10 top-level (`conv_in`, `conv_out`,
    /// `conv_norm_out`, `time_embedding.linear_{1,2}`), 246 across four
    /// down blocks (2 resnets each, attention on the first three stages,
    /// downsamplers on the first three), 46 across the mid block (2
    /// resnets + 1 attention), and 384 across four up blocks (3 resnets
    /// each with conv_shortcut, attention on the last three stages,
    /// upsamplers on the first three).
    ///
    /// The loader asserts 100% key consumption, so a typo or missed
    /// branch surfaces with a list of missing keys rather than silently
    /// zero-initializing a tensor.
    #[allow(clippy::too_many_lines)]
    pub fn from_safetensors(
        config: UnetConfig,
        ckpt: &crate::weights::SafetensorsCheckpoint,
    ) -> Result<Self> {
        use std::collections::HashSet;

        use scry_vision::checkpoint::load_conv2d_with_bias;

        use self::blocks::StageParams;
        use self::common::{load_group_norm, load_linear, missing_keys_error};
        use crate::error::Error;

        let view = ckpt.tensors()?;
        let g = config.num_norm_groups;
        let mut consumed: HashSet<String> = HashSet::new();
        let mut consume = |key: &str| {
            consumed.insert(key.to_string());
        };

        let n_stages = config.block_out_channels.len();
        if n_stages < 2 {
            return Err(Error::Llm(format!(
                "unet: need at least 2 stages, got {n_stages}"
            )));
        }
        if config.attention_per_block.len() != n_stages
            || config.num_attention_heads.len() != n_stages
            || config.transformer_layers_per_block.len() != n_stages
        {
            return Err(Error::Llm(
                "unet config: per-stage vectors must match block_out_channels length".into(),
            ));
        }
        let bc = &config.block_out_channels;
        let head_ch_0 = bc[0];

        // ---- Top-level conv_in (in_channels → block_out_channels[0]). ----
        let conv_in =
            load_conv2d_with_bias::<B>(&view, "conv_in", config.in_channels, head_ch_0, 3, 3, 1, 1)
                .map_err(|e| Error::Llm(format!("load conv_in: {e}")))?;
        consume("conv_in.weight");
        consume("conv_in.bias");

        // ---- Time embedding MLP (Linear → SiLU → Linear). ----
        // HF stores `linear_1: [time_embed_dim, block_out_channels[0]]`
        // (PyTorch [out, in]) — the first linear up-projects the
        // sinusoidal embed from `bc[0]` to `time_embed_dim`. Second
        // linear keeps `time_embed_dim` as both input and output.
        let (linear_1_weight, linear_1_bias) = load_linear::<B>(
            &view,
            "time_embedding.linear_1",
            head_ch_0,
            config.time_embed_dim,
            true,
            &mut consume,
        )?;
        let (linear_2_weight, linear_2_bias) = load_linear::<B>(
            &view,
            "time_embedding.linear_2",
            config.time_embed_dim,
            config.time_embed_dim,
            true,
            &mut consume,
        )?;
        let time_embed = TimeEmbedding {
            linear_1_weight,
            linear_1_bias,
            linear_2_weight,
            linear_2_bias,
        };

        // ---- Down blocks ----
        // Each stage `i`: input channels = bc[max(i-1, 0)] (or bc[0] for
        // i=0 since conv_in produced bc[0]), output = bc[i]. Downsampler
        // present at `i < n_stages - 1`.
        let mut down_blocks = Vec::with_capacity(n_stages);
        let mut prev_ch = head_ch_0;
        for (i, &out_ch) in bc.iter().enumerate() {
            let params = StageParams {
                num_resnets: config.layers_per_block,
                in_channels: prev_ch,
                out_channels: out_ch,
                resnet_in_channels: None,
                time_embed_dim: config.time_embed_dim,
                num_norm_groups: g,
                has_attention: config.attention_per_block[i],
                num_attention_heads: config.num_attention_heads[i],
                cross_attention_dim: config.cross_attention_dim,
                transformer_layers: config.transformer_layers_per_block[i],
            };
            let has_downsampler = i + 1 < n_stages;
            down_blocks.push(DownBlock::from_safetensors(
                &view,
                &format!("down_blocks.{i}"),
                &params,
                has_downsampler,
                &mut consume,
            )?);
            prev_ch = out_ch;
        }

        // ---- Mid block ----
        // Mid stays at bc[-1] throughout. Always has attention with
        // `transformer_layers_per_block` matching the deepest stage.
        let deepest = bc[n_stages - 1];
        let mid_params = StageParams {
            num_resnets: 2,
            in_channels: deepest,
            out_channels: deepest,
            resnet_in_channels: None,
            time_embed_dim: config.time_embed_dim,
            num_norm_groups: g,
            has_attention: true,
            num_attention_heads: config.num_attention_heads[n_stages - 1],
            cross_attention_dim: config.cross_attention_dim,
            transformer_layers: config.transformer_layers_per_block[n_stages - 1],
        };
        let mid_block = MidBlock::from_safetensors(&view, "mid_block", &mid_params, &mut consume)?;

        // ---- Up blocks (mirror the down side, taking skip concats). ----
        //
        // The diffusers UNet records `n_stages × layers_per_block` skip
        // channels on the way down (one per resnet output) plus
        // `n_stages - 1` extra skips from each downsampler output, all
        // sharing channel widths with the corresponding down stage. The
        // up side has `layers_per_block + 1` resnets per stage, popping
        // one skip per resnet. Per stage `i` (0 = deepest = post-mid),
        // the resnet input channels are:
        //
        //     resnets_per_up = layers_per_block + 1
        //     stage_out_ch   = bc[n_stages - 1 - i]
        //     prev_stage_out = bc[n_stages - 1 - (i - 1)]   (or `deepest` for i=0)
        //     next_skip_lvl  = bc[n_stages - 1 - i - 1]     (or 0 for i = n_stages-1)
        //
        // Resnet 0 takes (prev_up_out, skip_at_stage_out_ch) -> stage_out_ch
        // Resnet 1..k-1 takes (stage_out_ch, skip_at_stage_out_ch) -> stage_out_ch
        // Resnet k=last takes (stage_out_ch, skip_from_one_level_shallower) -> stage_out_ch
        //
        // The "one level shallower" handoff is what HF calls the
        // `output_states[-1]` of the previous (shallower) down block —
        // see diffusers `unet_2d_blocks.py::CrossAttnUpBlock2D.forward`.
        let resnets_per_up = config.layers_per_block + 1;
        let mut up_blocks = Vec::with_capacity(n_stages);
        let mut up_in_ch = deepest;
        for i in 0..n_stages {
            // Reverse-indexed: deepest first.
            let stage_idx = n_stages - 1 - i;
            let stage_out = bc[stage_idx];
            // Skip channel widths popped at this stage. The first
            // (layers_per_block) resnets pop a skip from this stage
            // (channel = stage_out); the last resnet pops a skip from
            // the *next-shallower* down stage (which produced output at
            // bc[stage_idx - 1] before downsampling). For the shallowest
            // stage (stage_idx == 0) the conv_in skip is also at bc[0].
            let shallower = if stage_idx == 0 {
                bc[0]
            } else {
                bc[stage_idx - 1]
            };
            let mut resnet_in_channels = Vec::with_capacity(resnets_per_up);
            for r in 0..resnets_per_up {
                let prev = if r == 0 { up_in_ch } else { stage_out };
                let skip = if r + 1 == resnets_per_up {
                    shallower
                } else {
                    stage_out
                };
                resnet_in_channels.push(prev + skip);
            }

            let attention_at_down = config.attention_per_block[stage_idx];
            let params = StageParams {
                num_resnets: resnets_per_up,
                in_channels: stage_out, // unused; resnet_in_channels overrides
                out_channels: stage_out,
                resnet_in_channels: Some(resnet_in_channels),
                time_embed_dim: config.time_embed_dim,
                num_norm_groups: g,
                has_attention: attention_at_down,
                num_attention_heads: config.num_attention_heads[stage_idx],
                cross_attention_dim: config.cross_attention_dim,
                transformer_layers: config.transformer_layers_per_block[stage_idx],
            };
            // Upsampler at all stages except the shallowest (i == n_stages-1).
            let has_upsampler = i + 1 < n_stages;
            up_blocks.push(UpBlock::from_safetensors(
                &view,
                &format!("up_blocks.{i}"),
                &params,
                has_upsampler,
                &mut consume,
            )?);
            up_in_ch = stage_out;
        }

        // ---- Output norm + conv (back to `out_channels` latent dims). ----
        let conv_norm_out =
            load_group_norm::<B>(&view, "conv_norm_out", head_ch_0, g, &mut consume)?;
        let conv_out = load_conv2d_with_bias::<B>(
            &view,
            "conv_out",
            head_ch_0,
            config.out_channels,
            3,
            3,
            1,
            1,
        )
        .map_err(|e| Error::Llm(format!("load conv_out: {e}")))?;
        consume("conv_out.weight");
        consume("conv_out.bias");

        // ---- 100% consumption check ----
        let all: HashSet<String> = view.names().into_iter().cloned().collect();
        if let Some(err) = missing_keys_error("unet loader", &all, &consumed) {
            return Err(err);
        }
        let extra: Vec<String> = consumed.difference(&all).cloned().collect();
        if !extra.is_empty() {
            let mut sorted = extra;
            sorted.sort();
            return Err(Error::Llm(format!(
                "unet loader: {} consumed keys absent from checkpoint (loader bug): {}",
                sorted.len(),
                sorted
                    .iter()
                    .take(8)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        Ok(Self {
            config,
            conv_in,
            time_embed,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }
}
