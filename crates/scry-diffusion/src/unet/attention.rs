// SPDX-License-Identifier: MIT OR Apache-2.0
//! Spatial transformer block — self-attention + cross-attention + GeGLU MLP.
//!
//! Each cross-attention block in SD's UNet is a "spatial transformer" stack:
//!
//! ```text
//!     in (NCHW) ──► reshape to (N, H*W, C) ──┐
//!                                             │
//!     ┌─ LayerNorm → self-attn(Q=K=V=x) ─ + ─┤        (image attends to itself)
//!     │                                       │
//!     ├─ LayerNorm → cross-attn(Q=x, K/V=ctx) ┤        (image attends to text)
//!     │                                       │
//!     └─ LayerNorm → GeGLU MLP ─────────── + ─┘
//!                                             │
//!     reshape back to (N, C, H, W)  ◄─────────┘
//! ```
//!
//! - **Self-attention** is just `scry-vision`'s ViT attention used at a
//!   different shape (image tokens instead of patches).
//! - **Cross-attention** has Q from the latent and K/V from the conditioning
//!   embeddings — the only shape change vs self-attention is that K/V come
//!   from a different tensor with a different sequence length.
//! - **GeGLU** = chunk dim in two, gate one half by GELU(other half) — needs
//!   a small MathBackend helper or two existing matmul + GELU calls.
//!
//! `transformer_layers_per_block` (from `UnetConfig`) controls how many of
//! these stacks are nested per attention block. SD 1.5 = 1; SDXL deepest
//! stage = 10 (the bulk of SDXL's parameter count lives here).
//!
//! HF naming convention (mirrors `diffusers/models/attention_processor.py`
//! and `attention.py::Transformer2DModel`):
//!
//! - `proj_in` / `proj_out`: 1×1 Conv2d (SD 1.5 uses `use_linear_projection=False`).
//! - `attn1` is self-attention with `to_q`, `to_k`, `to_v` (no bias) and
//!   `to_out.0` (with bias). `attn2` is cross-attention with the same
//!   layout but K/V projecting from the conditioning dim.
//! - `ff.net.0.proj` is the GEGLU first linear (`d -> 2*d_ff`). `ff.net.1`
//!   is dropout (no params, ignored at inference). `ff.net.2` is the
//!   second linear (`d_ff -> d`).

use scry_llm::backend::MathBackend;
use scry_llm::nn::layernorm::LayerNormModule;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_vision::nn::conv2d::Conv2d;

use super::common::{
    add_same, exact_gelu, matmul_bias_2d, transpose_chw_to_hwc, transpose_hwc_to_chw,
    GroupNormParams,
};
use crate::conditioning::Conditioning;
use crate::error::Result;

/// Self- or cross-attention layer. SD's UNet uses the same `Attention`
/// shape for both `attn1` (self) and `attn2` (cross); the only difference
/// is whether K/V project from `d_model` (self) or `cross_dim` (cross).
///
/// Q/K/V have no bias in SD 1.5; the output projection (`to_out.0`) does.
/// Weights are stored in scry-llm `[in, out]` convention.
pub(crate) struct Attention<B: MathBackend> {
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) d_model: usize,
    pub(crate) cross_dim: usize,

    /// `[d_model, inner_dim]` where `inner_dim = num_heads * head_dim`.
    pub(crate) q_weight: Tensor<B>,
    /// `[cross_dim, inner_dim]`.
    pub(crate) k_weight: Tensor<B>,
    /// `[cross_dim, inner_dim]`.
    pub(crate) v_weight: Tensor<B>,
    /// `[inner_dim, d_model]`.
    pub(crate) out_weight: Tensor<B>,
    /// `[d_model]`.
    pub(crate) out_bias: Tensor<B>,
}

/// GeGLU feed-forward block (`ff.*` in HF). Two linears with a GELU gate
/// in between.
///
/// HF stores the input projection as a single `[2·d_ff, d_model]` weight
/// whose output is chunked along the last dim into `(values, gate)`. We
/// **split that into two independent `[d_model, d_ff]` matmuls at load
/// time** — empirically, cuBLAS bf16 GemmEx picks a faster algorithm for
/// two narrower matmuls than for one with 2× wider M, and splitting also
/// eliminates the post-projection `gather_columns` step (each matmul
/// produces exactly its half of the output).
pub(crate) struct GeGluFf<B: MathBackend> {
    pub(crate) d_model: usize,
    pub(crate) d_ff: usize,

    /// First half of HF's `proj_in` — the **values**. Shape
    /// `[d_model, d_ff]`, scry-llm `[in, out]` convention.
    pub(crate) proj_values_weight: Tensor<B>,
    /// `[d_ff]`.
    pub(crate) proj_values_bias: Tensor<B>,
    /// Second half of HF's `proj_in` — the **gate**. Shape `[d_model, d_ff]`.
    pub(crate) proj_gate_weight: Tensor<B>,
    /// `[d_ff]`.
    pub(crate) proj_gate_bias: Tensor<B>,
    /// `proj_out`: `[d_ff, d_model]`.
    pub(crate) proj_out_weight: Tensor<B>,
    /// `[d_model]`.
    pub(crate) proj_out_bias: Tensor<B>,
}

/// One transformer-2D layer (self + cross + MLP, all with pre-LN).
pub struct BasicTransformerBlock<B: MathBackend> {
    /// Inner channel count.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Cross-attention conditioning dim (768 SD 1.5, 2048 SDXL).
    pub cross_attention_dim: usize,

    pub(crate) norm1: LayerNormModule<B>,
    pub(crate) attn1: Attention<B>,
    pub(crate) norm2: LayerNormModule<B>,
    pub(crate) attn2: Attention<B>,
    pub(crate) norm3: LayerNormModule<B>,
    pub(crate) ff: GeGluFf<B>,
}

impl<B: MathBackend> Attention<B> {
    /// Pre-upload every Q/K/V/out parameter to the backend's device-resident form.
    pub(crate) fn to_device(&mut self) {
        B::to_device_in_place(&mut self.q_weight.data);
        B::to_device_in_place(&mut self.k_weight.data);
        B::to_device_in_place(&mut self.v_weight.data);
        B::to_device_in_place(&mut self.out_weight.data);
        B::to_device_in_place(&mut self.out_bias.data);
    }

    /// Multi-head attention. `q_input` is `[n_q, d_model]`; `kv_input` is
    /// `[n_kv, cross_dim]`. For self-attention, pass the same tensor and
    /// `cross_dim == d_model`. Returns `[n_q, d_model]`.
    ///
    /// Implementation: pre-permute Q/K/V to `[h, n, d_h]` once via
    /// `B::reshape_for_heads` (CUDA kernel on `ScryGpuBackend`), then
    /// run all heads in two `B::matmul_strided_batched` calls (cuBLAS
    /// strided batched gemm). Replaces a `num_heads`-deep loop that
    /// previously called `gather_columns`/`scatter_columns` per head —
    /// those have host-roundtrip defaults on `ScryGpuBackend`, which
    /// the profile identified as the dominant cost in attention.
    fn forward(&self, q_input: &Tensor<B>, kv_input: &Tensor<B>) -> Tensor<B> {
        use crate::profile::time_section;

        let inner_dim = self.num_heads * self.head_dim;
        let n_q = q_input.shape.dims()[0];
        let n_kv = kv_input.shape.dims()[0];

        let (q, k, v) = time_section("attn.qkv_proj", || {
            let q = B::matmul(
                &q_input.data,
                &self.q_weight.data,
                n_q,
                self.d_model,
                inner_dim,
                false,
                false,
            );
            let k = B::matmul(
                &kv_input.data,
                &self.k_weight.data,
                n_kv,
                self.cross_dim,
                inner_dim,
                false,
                false,
            );
            let v = B::matmul(
                &kv_input.data,
                &self.v_weight.data,
                n_kv,
                self.cross_dim,
                inner_dim,
                false,
                false,
            );
            (q, k, v)
        });

        let (q_h, k_h, v_h) = time_section("attn.reshape_in", || {
            (
                B::reshape_for_heads(&q, 1, n_q, self.num_heads, self.head_dim),
                B::reshape_for_heads(&k, 1, n_kv, self.num_heads, self.head_dim),
                B::reshape_for_heads(&v, 1, n_kv, self.num_heads, self.head_dim),
            )
        });

        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let scores = time_section("attn.scores", || {
            B::matmul_strided_batched(
                &q_h,
                &k_h,
                self.num_heads,
                n_q,
                self.head_dim,
                n_kv,
                false,
                true,
            )
        });
        let attn = time_section("attn.softmax", || {
            B::scaled_softmax(&scores, scale, &Shape::new(&[self.num_heads * n_q, n_kv]))
        });
        let out_per_head = time_section("attn.values", || {
            B::matmul_strided_batched(
                &attn,
                &v_h,
                self.num_heads,
                n_q,
                n_kv,
                self.head_dim,
                false,
                false,
            )
        });
        let head_concat = time_section("attn.reshape_out", || {
            B::reshape_from_heads(&out_per_head, 1, n_q, self.num_heads, self.head_dim)
        });

        let out = time_section("attn.out_proj", || {
            B::matmul_bias(
                &head_concat,
                &self.out_weight.data,
                &self.out_bias.data,
                n_q,
                inner_dim,
                self.d_model,
                false,
                false,
            )
        });
        Tensor::new(out, Shape::new(&[n_q, self.d_model]))
    }
}

impl<B: MathBackend> GeGluFf<B> {
    /// Pre-upload every projection weight/bias to the backend's device-resident form.
    pub(crate) fn to_device(&mut self) {
        B::to_device_in_place(&mut self.proj_values_weight.data);
        B::to_device_in_place(&mut self.proj_values_bias.data);
        B::to_device_in_place(&mut self.proj_gate_weight.data);
        B::to_device_in_place(&mut self.proj_gate_bias.data);
        B::to_device_in_place(&mut self.proj_out_weight.data);
        B::to_device_in_place(&mut self.proj_out_bias.data);
    }

    /// GeGLU feed-forward on `[n, d_model]`. Returns `[n, d_model]`.
    ///
    /// Two independent `[d_model, d_ff]` matmuls produce values and gate
    /// directly — no `gather_columns` step needed. This is faster than
    /// one wide `[d_model, 2·d_ff]` matmul + chunk at SD shapes
    /// (cuBLAS bf16 GemmEx prefers the narrower-M algorithm).
    fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        use crate::profile::time_section;

        let n = input.shape.dims()[0];

        let values_t = time_section("ff.proj_values", || {
            matmul_bias_2d::<B>(
                input,
                &self.proj_values_weight,
                &self.proj_values_bias,
                n,
                self.d_model,
                self.d_ff,
            )
        });
        let gate_t = time_section("ff.proj_gate", || {
            matmul_bias_2d::<B>(
                input,
                &self.proj_gate_weight,
                &self.proj_gate_bias,
                n,
                self.d_model,
                self.d_ff,
            )
        });

        let gated_t = time_section("ff.gate", || {
            // HF GeGLU uses `F.gelu(approximate="none")` (erf-based exact GELU).
            // `B::gelu` is the tanh approximation, which drifts enough across
            // SD 1.5's 16 transformer blocks to push the M6 1e-3 parity gate.
            let gelu_gate = time_section("ff.gate.exact_gelu", || exact_gelu(&gate_t));
            let gated = B::mul_elementwise(&values_t.data, &gelu_gate.data);
            Tensor::new(gated, Shape::new(&[n, self.d_ff]))
        });

        time_section("ff.proj_out", || {
            matmul_bias_2d::<B>(
                &gated_t,
                &self.proj_out_weight,
                &self.proj_out_bias,
                n,
                self.d_ff,
                self.d_model,
            )
        })
    }
}

impl<B: MathBackend> BasicTransformerBlock<B> {
    /// Pre-upload every parameter tensor to the backend's device-resident form.
    pub fn to_device(&mut self) {
        self.norm1.to_device();
        self.attn1.to_device();
        self.norm2.to_device();
        self.attn2.to_device();
        self.norm3.to_device();
        self.ff.to_device();
    }

    /// Forward pass: latent ⊕ self-attn → ⊕ cross-attn(text) → ⊕ MLP.
    ///
    /// Input/output `[n, d_model]` (image tokens flattened from `[H*W, C]`).
    pub fn forward(
        &mut self,
        latent: &Tensor<B>,
        conditioning: &Conditioning<B>,
    ) -> Result<Tensor<B>> {
        use crate::profile::time_section;

        // Self-attention with pre-LN, residual-added.
        let attn1_out = time_section("xfblock.self_attn", || {
            let n1 = self.norm1.forward(latent);
            self.attn1.forward(&n1, &n1)
        });
        let x = add_same(latent, &attn1_out);

        // Cross-attention with pre-LN; K/V come from the conditioning
        // text embeddings `[seq_len, cross_dim]`.
        let attn2_out = time_section("xfblock.cross_attn", || {
            let n2 = self.norm2.forward(&x);
            self.attn2.forward(&n2, &conditioning.embeddings)
        });
        let x = add_same(&x, &attn2_out);

        // FF MLP with pre-LN.
        let ff_out = time_section("xfblock.ff", || {
            let n3 = self.norm3.forward(&x);
            self.ff.forward(&n3)
        });
        Ok(add_same(&x, &ff_out))
    }
}

/// Spatial transformer block — wraps `transformer_layers_per_block` stacks
/// with the NCHW ↔ (N, H*W, C) reshape and a 1×1 conv on entry/exit.
pub struct SpatialTransformer<B: MathBackend> {
    /// Channel count of the feature map entering the block.
    pub channels: usize,
    /// Stacks of `BasicTransformerBlock` to apply (= `transformer_layers_per_block`).
    pub transformer_blocks: Vec<BasicTransformerBlock<B>>,

    pub(crate) norm: GroupNormParams<B>,
    pub(crate) proj_in: Conv2d<B>,
    pub(crate) proj_out: Conv2d<B>,
}

impl<B: MathBackend> SpatialTransformer<B> {
    /// Pre-upload every parameter tensor to the backend's device-resident form.
    pub fn to_device(&mut self) {
        self.norm.to_device();
        self.proj_in.to_device();
        self.proj_out.to_device();
        for b in &mut self.transformer_blocks {
            b.to_device();
        }
    }

    /// Forward: GroupNorm → 1×1 conv in → reshape `[C, H, W]` to `[H*W, C]`
    /// → N × `BasicTransformerBlock` → reshape back → 1×1 conv out →
    /// residual add. Input/output `[C, H, W]`.
    pub fn forward(
        &mut self,
        feature_map: &Tensor<B>,
        conditioning: &Conditioning<B>,
    ) -> Result<Tensor<B>> {
        use crate::profile::time_section;

        let dims = feature_map.shape.dims();
        debug_assert_eq!(dims.len(), 3);
        let (c, h, w) = (dims[0], dims[1], dims[2]);
        debug_assert_eq!(c, self.channels);
        let n = h * w;

        // Pre-norm + 1×1 in projection, still NCHW.
        let normed = time_section("xfrm.norm", || self.norm.forward(feature_map));
        let proj_in = time_section("xfrm.proj_in", || self.proj_in.forward(&normed));

        // [C, H, W] -> [H*W, C] for the transformer blocks.
        let mut x = time_section("xfrm.transpose_in", || {
            transpose_chw_to_hwc::<B>(&proj_in, c, n)
        });
        for block in &mut self.transformer_blocks {
            x = block.forward(&x, conditioning)?;
        }

        // [H*W, C] -> [C, H, W] and 1×1 out projection, then residual.
        let x_chw = time_section("xfrm.transpose_out", || {
            transpose_hwc_to_chw::<B>(&x, n, c, h, w)
        });
        let proj_out = time_section("xfrm.proj_out", || self.proj_out.forward(&x_chw));
        Ok(time_section("xfrm.residual", || {
            add_same(feature_map, &proj_out)
        }))
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> Attention<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        d_model: usize,
        cross_dim: usize,
        num_heads: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use super::common::load_linear;
        use crate::error::Error;

        if d_model % num_heads != 0 {
            return Err(Error::Llm(format!(
                "{prefix}: d_model={d_model} not divisible by num_heads={num_heads}"
            )));
        }
        let head_dim = d_model / num_heads;
        let inner_dim = num_heads * head_dim;

        // SD attention has bias=False on Q/K/V; bias=True on to_out.0.
        let (q_weight, _) = load_linear::<B>(
            view,
            &format!("{prefix}.to_q"),
            d_model,
            inner_dim,
            false,
            consume,
        )?;
        let (k_weight, _) = load_linear::<B>(
            view,
            &format!("{prefix}.to_k"),
            cross_dim,
            inner_dim,
            false,
            consume,
        )?;
        let (v_weight, _) = load_linear::<B>(
            view,
            &format!("{prefix}.to_v"),
            cross_dim,
            inner_dim,
            false,
            consume,
        )?;
        // HF `to_out` is a `nn.ModuleList([Linear(inner, d), Dropout])`
        // — the linear is at `to_out.0` and the dropout has no params.
        let (out_weight, out_bias) = load_linear::<B>(
            view,
            &format!("{prefix}.to_out.0"),
            inner_dim,
            d_model,
            true,
            consume,
        )?;

        Ok(Self {
            num_heads,
            head_dim,
            d_model,
            cross_dim,
            q_weight,
            k_weight,
            v_weight,
            out_weight,
            out_bias,
        })
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> GeGluFf<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        d_model: usize,
        d_ff: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use super::common::{load_linear, load_split2_linear};

        // HF stores `ff.net.0.proj` as one [2*d_ff, d_model] tensor; we
        // split it column-wise into independent values + gate matmuls
        // at load time. First half = values, second half = gate (matches
        // HF `chunk(2, dim=-1)` semantics).
        let ((proj_values_weight, proj_values_bias), (proj_gate_weight, proj_gate_bias)) =
            load_split2_linear::<B>(
                view,
                &format!("{prefix}.net.0.proj"),
                d_model,
                d_ff,
                true,
                consume,
            )?;
        let (proj_out_weight, proj_out_bias) = load_linear::<B>(
            view,
            &format!("{prefix}.net.2"),
            d_ff,
            d_model,
            true,
            consume,
        )?;
        Ok(Self {
            d_model,
            d_ff,
            proj_values_weight,
            proj_values_bias,
            proj_gate_weight,
            proj_gate_bias,
            proj_out_weight,
            proj_out_bias,
        })
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> BasicTransformerBlock<B> {
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        d_model: usize,
        num_heads: usize,
        cross_attention_dim: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use scry_vision::checkpoint::load_tensor;

        use crate::error::Error;

        // SD 1.5 BasicTransformerBlock follows pre-LN / parallel stack:
        //   norm1 -> attn1(self)        -> +residual
        //   norm2 -> attn2(cross,ctx)   -> +residual
        //   norm3 -> ff(GeGLU)          -> +residual
        let norm1 = LayerNormModule {
            gamma: load_tensor::<B>(view, &format!("{prefix}.norm1.weight"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm1.weight: {e}")))?,
            beta: load_tensor::<B>(view, &format!("{prefix}.norm1.bias"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm1.bias: {e}")))?,
            eps: 1e-5,
        };
        consume(&format!("{prefix}.norm1.weight"));
        consume(&format!("{prefix}.norm1.bias"));

        let attn1 = Attention::from_safetensors(
            view,
            &format!("{prefix}.attn1"),
            d_model,
            d_model,
            num_heads,
            consume,
        )?;

        let norm2 = LayerNormModule {
            gamma: load_tensor::<B>(view, &format!("{prefix}.norm2.weight"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm2.weight: {e}")))?,
            beta: load_tensor::<B>(view, &format!("{prefix}.norm2.bias"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm2.bias: {e}")))?,
            eps: 1e-5,
        };
        consume(&format!("{prefix}.norm2.weight"));
        consume(&format!("{prefix}.norm2.bias"));

        let attn2 = Attention::from_safetensors(
            view,
            &format!("{prefix}.attn2"),
            d_model,
            cross_attention_dim,
            num_heads,
            consume,
        )?;

        let norm3 = LayerNormModule {
            gamma: load_tensor::<B>(view, &format!("{prefix}.norm3.weight"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm3.weight: {e}")))?,
            beta: load_tensor::<B>(view, &format!("{prefix}.norm3.bias"), &[d_model])
                .map_err(|e| Error::Llm(format!("load {prefix}.norm3.bias: {e}")))?,
            eps: 1e-5,
        };
        consume(&format!("{prefix}.norm3.weight"));
        consume(&format!("{prefix}.norm3.bias"));

        // SD 1.5 fixes ff_inner_dim = 4 * d_model (FeedForward `mult=4`).
        let d_ff = 4 * d_model;
        let ff = GeGluFf::from_safetensors(view, &format!("{prefix}.ff"), d_model, d_ff, consume)?;

        Ok(Self {
            d_model,
            num_heads,
            cross_attention_dim,
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
        })
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> SpatialTransformer<B> {
    /// Load one `Transformer2DModel` block at `prefix.*` (e.g.
    /// `down_blocks.0.attentions.0`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_safetensors(
        view: &safetensors::SafeTensors<'_>,
        prefix: &str,
        channels: usize,
        num_heads: usize,
        cross_attention_dim: usize,
        transformer_layers: usize,
        num_norm_groups: usize,
        consume: &mut impl FnMut(&str),
    ) -> Result<Self> {
        use scry_vision::checkpoint::load_conv2d_with_bias;

        use super::common::load_group_norm;
        use crate::error::Error;

        // Outer norm + 1×1 in/out projections (SD 1.5 uses Conv2d, not Linear).
        let norm = load_group_norm::<B>(
            view,
            &format!("{prefix}.norm"),
            channels,
            num_norm_groups,
            consume,
        )?;
        let proj_in = load_conv2d_with_bias::<B>(
            view,
            &format!("{prefix}.proj_in"),
            channels,
            channels,
            1,
            1,
            1,
            0,
        )
        .map_err(|e| Error::Llm(format!("load {prefix}.proj_in: {e}")))?;
        consume(&format!("{prefix}.proj_in.weight"));
        consume(&format!("{prefix}.proj_in.bias"));
        let proj_out = load_conv2d_with_bias::<B>(
            view,
            &format!("{prefix}.proj_out"),
            channels,
            channels,
            1,
            1,
            1,
            0,
        )
        .map_err(|e| Error::Llm(format!("load {prefix}.proj_out: {e}")))?;
        consume(&format!("{prefix}.proj_out.weight"));
        consume(&format!("{prefix}.proj_out.bias"));

        let mut transformer_blocks = Vec::with_capacity(transformer_layers);
        for i in 0..transformer_layers {
            transformer_blocks.push(BasicTransformerBlock::from_safetensors(
                view,
                &format!("{prefix}.transformer_blocks.{i}"),
                channels,
                num_heads,
                cross_attention_dim,
                consume,
            )?);
        }

        Ok(Self {
            channels,
            transformer_blocks,
            norm,
            proj_in,
            proj_out,
        })
    }
}
