# scry-diffusion — Handoff

This crate is a **scaffold**. Every public type compiles, every method body is
`todo!()`. Your job is to fill them in, in the milestone order below, and ship
end-to-end Stable Diffusion 1.5 txt2img on the scry stack.

## Goal

End-to-end **Stable Diffusion 1.5 text-to-image** on `MathBackend`, with
correctness validated against HF `diffusers` and competitive perf on
`ScryGpuBackend`. SDXL should require new configs + a second text encoder
+ extra timestep conditioning, **not a rewrite**.

## Non-goals (for the first cut)

- img2img, inpainting, controlnet, LoRA, IP-adapter — defer.
- SDXL output quality — scaffold leaves the hooks; first SDXL run is M11.
- Training. We are inference-only.
- Vulkan-path optimization. Per the project memory, CUDA is the perf target;
  Vulkan stays correct (default trait impls + on-CPU fallbacks).

## Architecture (and where SDXL plugs in)

```
GenerationParams ─► Tokenizer ─► TextEncoder ──► Conditioning
                                                     │
                                                     ▼
                                  Scheduler ◄── UNet (CFG-batched)
                                                     │
                                                     ▼
                                                 VaeDecoder ──► RGB tensor
```

The five seams that are **already** SDXL-ready:

| Seam | SD 1.5 | SDXL upgrade |
|---|---|---|
| `TextEncoder` trait | `ClipTextEncoder` (CLIP-L) | `SdxlTextEncoder` (CLIP-L ++ OpenCLIP-bigG) |
| `Conditioning` | `embeddings` only | `embeddings` + `extras: SdxlExtras` |
| `UnetConfig` | `UnetConfig::sd_1_5()` preset | `UnetConfig::sdxl_base()` preset (already written) |
| `VaeDecoderConfig` | `sd_1_5()` preset | `sdxl()` preset (different scaling factor) |
| `Scheduler` trait | `DdimScheduler` | works unchanged for SDXL |

The seams not yet touched (do not pre-build them — wait for SD 1.5 to land):

- SDXL refiner (second-stage UNet that takes the first-stage output as init).
- LoRA / IP-adapter weight merging at load time.

## Missing primitives (M1 — must land before any model code can run)

These need to land in `scry-llm` (trait method + tests) and **on CUDA in
`scry-gpu`** (kernel + override). Default CPU impl is fine for correctness;
the CUDA override is what makes inference usable.

| Op | Why we need it | Where it goes |
|---|---|---|
| `group_norm` | Used everywhere in UNet / VAE. Channels split into 32 groups, mean/var per `(batch, group)`, then per-channel affine. | Trait: `scry-llm/src/backend/mod.rs::MathBackend`. Kernel: pattern matches `LAYERNORM_CUDA` (one block per `[batch, group]`, two block-wide reductions for mean/var, third pass normalize+affine). |
| `silu` | Activation in every UNet ResBlock. `x * sigmoid(x)`. | Trait method + elementwise CUDA kernel. Register a key in `ScryCtx` like `gelu`. |
| `upsample_2d_nearest` | UNet UpBlocks and VAE decoder. NCHW, integer scale factor, no interpolation. | Trait method + CUDA kernel (one thread per output element, `c, oh, ow → c, oh/scale, ow/scale`). |
| `geglu` (or compose) | UNet MLPs use GeGLU instead of plain MLP: chunk last dim, gate one half through GELU, multiply. | Either: add a trait method (mirrors `swiglu`), **or** compose from two existing matmuls + `gelu` + `mul_elementwise`. Compose first; promote to a kernel only if the bench says so. |

`scry-llm` already has: `matmul`, `matmul_strided_batched`, `softmax`,
`scaled_softmax`, `layernorm`, `batchnorm_2d_inference`, `gelu`, `relu`,
`conv2d_forward`, `im2col_2d`, `max_pool_2d`, `adaptive_avg_pool_2d`,
`embedding`, `mul_elementwise`, `scale`, `concat_rows`, `add`, `rmsnorm`,
`rope`, `swiglu`. Cross-attention reuses `matmul_strided_batched` and
`scaled_softmax`; do **not** add a new "cross_attention" trait method.

For each new kernel: one CUDA shader in `scry-gpu/src/shaders.rs`, one
`Device::*_async` wrapper in `scry-gpu/src/device.rs`, one `MathBackend`
method in `scry-llm/src/backend/mod.rs` with a CPU default, one
`ScryGpuBackend` override in `scry-llm/src/backend/scry_gpu.rs` with a
threshold gate, and one `gpu_*_matches_cpu_within_tolerance` test. **Mirror
the existing patterns** — e.g. `gpu_layernorm_matches_cpu_within_tolerance`
for the GroupNorm test shape.

## Milestone plan

Each milestone is a separate commit (or PR). Earlier milestones gate later
ones — don't skip ahead. Numerical-equivalence tests against `CpuBackend`
or HF `diffusers` are part of the "done" criterion, not optional polish.

**M1 — Missing kernels** (`scry-gpu` + `scry-llm`)

- GroupNorm: kernel, trait method, override, test.
- SiLU: kernel, trait method, override, test.
- `upsample_2d_nearest`: kernel, trait method, override, test.
- Update `crates/scry-diffusion/src/ops.rs` to delegate to the new
  `MathBackend` methods (drop the local `todo!()` wrappers).

**M2 — Tokenizer + safetensors loader**

- `Tokenizer`: pick one of the two paths in `src/tokenizer.rs` (DIY BPE
  recommended — zero new C deps; pattern after the openai-clip Python impl).
- `SafetensorsCheckpoint`: mirror `crates/scry-vision/src/checkpoint.rs`
  (memmap → `SafeTensors::deserialize` → cast helpers for f32 / bf16 / fp16).
- Acquire a local SD 1.5 snapshot for testing — link
  `runwayml/stable-diffusion-v1-5` into `crates/scry-diffusion/.assets/sd-1-5/`
  via `huggingface-cli download`. Document the path conventions in a
  `crates/scry-diffusion/ASSETS.md` that's gitignored.

**M3 — CLIP text encoder forward**

- Implement `ClipTextEncoder::encode` (token embed + positional embed → 12
  causal-attention blocks → final LN). Fill in `weights::map_clip_text_keys`.
- Validate: write `examples/check_text_encoder.rs` that loads HF weights and
  encodes a fixed prompt, then compare to a Python reference dumping
  `text_model(prompt).last_hidden_state` to a numpy file. Tolerance: 1e-4
  abs, since this is a single forward through a fp32 transformer.

**M4 — VAE decoder forward**

- Implement `VaeDecoder::decode`, `weights::map_vae_keys`. No attention in
  SD 1.5 base VAE's mid-block — the spatial attention is self-only,
  which is the same shape as a one-stack `BasicTransformerBlock` with
  `cross_attention_dim = channels`.
- Validate: feed a fixed noise latent through both our decoder and HF's,
  compare pixels within 1e-3 abs.

**M5 — UNet weight loading + plumbing**

- Wire up the structs (`Unet`, `DownBlock`, `MidBlock`, `UpBlock`,
  `ResBlock`, `SpatialTransformer`, `BasicTransformerBlock`) with real
  `Conv2d`, `Linear`, `LayerNorm` etc. fields.
- Fill in `weights::map_unet_keys`. Print the 100% of HF keys consumed
  on a successful load, error otherwise — silent missing keys are the most
  common bug here.

**M6 — UNet forward**

- Implement `Unet::forward` end-to-end: `conv_in` → time-embed MLP →
  `down_blocks` → `mid_block` → `up_blocks` (with skip concats) →
  `conv_norm_out` → SiLU → `conv_out`.
- Validate: a single forward at `t=981, latent=fixed_noise, conditioning=fixed_text`
  matches HF UNet output within 1e-3 abs.

**M7 — DDIM scheduler**

- Implement `DdimScheduler::new`, `set_timesteps`, `step`. Match HF's
  `scaled_linear` beta schedule exactly.
- Validate: given a fixed noise + a fixed sequence of "predicted noise"
  vectors, our `step` chain matches HF's `step` chain bit-for-bit (this
  is a pure-CPU numerical test, no GPU involved).

**M8 — End-to-end correctness (no perf yet)**

- Implement `Txt2ImgPipeline::generate` — tokenize, encode (uncond + cond),
  init noise (deterministic seed), 30-step DDIM loop with CFG batch=2,
  VAE decode, clamp, rescale to `[0, 1]`.
- Validate: same prompt + same seed produces a near-identical image to
  `diffusers.StableDiffusionPipeline` (HF). Use `PSNR > 35 dB` or pixel-MAE
  `< 0.01` as the success criterion (some drift from
  GroupNorm-precision differences is unavoidable).
- Write `examples/txt2img.rs` properly — CLI args for prompt, output path,
  seed, steps, CFG scale.

**M9a — `python/bench_pytorch.py`** (done 2026-05-08)

Builds the diffusers `StableDiffusionPipeline` programmatically from the
per-component dirs (no `model_index.json` in the snapshot — we keep the
asset tree minimal). Reports per-step median latency via
`callback_on_step_end` and total wall-clock; supports fp32/fp16/bf16.

**M9b — `examples/bench_sd.rs`** (done 2026-05-08)

Backend type is cfg-selected: `CpuBackend` by default, `ScryGpuBackend`
under `--features scry-gpu-cuda`. Exposes `--bf16-matmul` and `--no-cudnn`
toggles for the perf-pass sweep.

**M9c — Perf pass** (host-roundtrip wins exhausted; F/G/H remaining)

Baseline numbers landed 2026-05-08 (RTX 5070 Ti, prompt + seed
matched between Python and Rust):

| Backend | Size | Steps | Total | Per-step |
|---|---|---|---|---|
| CPU | 64×64 | 2 | 8.78 s | 3815 ms |
| GPU bf16 (start of perf pass) | 512×512 | 2 | 5.21 s | 2066 ms |
| GPU bf16 (post-matmul_bias) | 512×512 | 30 | 23.0 s | ~700 ms |
| GPU bf16 (post-mul_elementwise) | 512×512 | 30 | 17.0 s | 537 ms |
| GPU bf16 (post-to_device cascade) | 512×512 | 30 | **5.68 s** | **159 ms** |
| **PyTorch fp16** | **512×512** | **30** | **1.04 s** | **32.9 ms** |

After 10 profile-driven commits, our GPU bf16 path runs the same
generation in **5.68 s vs 63.3 s** at the perf pass start — an **11.1×
total speedup** with M6 HF parity bit-identical at 1.549e-4 throughout.
Gap to PyTorch fp16: 60× → **4.8×**.

**The big lesson** that ran through the entire perf pass: the original
"residency-bound" framing was wrong. The actual issue was that *many*
`MathBackend` trait methods had host-roundtrip defaults on
`ScryGpuBackend` — every `B::foo(...)` looks identical in source
whether `foo` is a CUDA kernel or a `to_vec` + scalar loop +
`from_vec`. Profile, find the host roundtrip masquerading as a kernel
call, lift it onto the GPU. Repeat.

Wins (each one a separate commit, gates green):

| Commit | Change | Per-step |
|---|---|---|
| `4408375` | Phase 1: Scheduler\<B\> (latent stays on device) | 2048 |
| `c63fbe8` | `MathBackend::gelu_exact` + CUDA shader | 1737 |
| `b48edc2` | GeGLU `proj_in` split (no gather_columns) | 1480 |
| `32148f0` | Batched attention (eliminates per-head gather/scatter) | 1113 |
| `26a42fa` | `MathBackend::transpose_2d` on GPU | 877 |
| `d52527e` | `clone_tensor` via `B::scale(1.0)` | 859 |
| `50ca256` | `concat_rows` CUDA shader (skip-concat on device) | 851 |
| `b8170e3` | `matmul_bias` on GPU + cascade fix | 717 |
| `4054fc0` | `mul_elementwise` on GPU + VAE helpers + cascade fix | 537 |
| _next_ | **`to_device` cascade — UNet/VAE/CLIP weights pre-uploaded once** | **159** |

**The audit that found the matmul_bias landmine.** After landing the
seven wins above we were ~hand-waving about "easy wins exhausted" — the
profile said per-section work was at floor and only multi-week levers
(CFG batching, FlashAttention, CUDA graphs) remained. A systematic
audit changed that: `awk` over `mod.rs` found 9 `MathBackend` trait
methods with `Self::to_vec` in their default body but no
`ScryGpuBackend` override. Of those, **`matmul_bias` was on the
hottest path possible** — every dense linear in the UNet (~104 calls
per forward) was paying a host roundtrip + re-upload cost.

The kicker: the `BIAS_ADD_CUDA` kernel had been sitting in
`scry-gpu/src/shaders.rs` since the original conv-bias work, **fully
implemented but never wired into scry-llm**. Wiring it took ~30 minutes
and produced a bigger step-time delta than any of the previous wins:

  Per-step:    851 ms  →  717 ms   (-16%)
  Per-call:    ff.proj_values    3.07 ms  →  0.80 ms  (3.8× faster)
               ff.proj_gate      2.84 ms  →  0.76 ms  (3.7× faster)
               ff.gate.exact_gelu  0.50 ms  →  0.027 ms  (**18× faster**)
               xfblock.cross_attn  1.13 ms  →  0.79 ms  (-30%)

Why the cascade? `matmul_bias`'s default returned `Cpu` storage
variant. Every consuming op then had to call `as_gpu_buffer(Cpu(v))`,
which uploads from host. Fixing matmul_bias didn't just remove its own
roundtrip — it removed a **re-upload from every downstream op**.

**Lesson for future-you.** When the profile says "easy wins exhausted",
audit the trait. The pattern of ad-hoc fixes (one host-roundtrip at a
time) is necessary but not sufficient. The systematic check is:

```
awk '/^    fn / { fn = $0 } /Self::to_vec/ && fn { print fn; fn = "" }' \
    crates/scry-llm/src/backend/mod.rs | sort -u  # 25 methods
awk '/^    fn / { sub(/.*fn /,""); sub(/[\(\<].*$/,""); print }' \
    crates/scry-llm/src/backend/scry_gpu.rs | sort -u  # GPU overrides
# diff the two — anything in (1) but not (2) is a latent host roundtrip.
```

**Round 2 audit lesson (2026-05-08, second session).** The "missing
override" check (above) is necessary but not sufficient — it misses the
**fake-override pattern**, where the GPU method exists but its body is
just `cpu(CpuBackend::foo(&a.as_vec()...))`. Both the matmul_bias and
mul_elementwise landmines were this pattern. The check that catches
them:

```
awk '
  /^    fn / { fn = $0; body = ""; in_fn = 1; next }
  in_fn && /^    \}/ {
    if (body ~ /cpu\(CpuBackend::/ && body !~ /gpu_/ && body !~ /persistent/) {
      print fn
    }
    in_fn = 0; body = ""; next
  }
  in_fn { body = body " | " $0 }
' crates/scry-llm/src/backend/scry_gpu.rs
```

Run this after every perf pass. Six methods on `ScryGpuBackend` still
match the pattern — `embedding`, `rmsnorm`, `rope`, `rope_with_freqs_preloaded`,
`swiglu`, `repeat_kv`. None are on SD's hot path (the first is CLIP-only,
called ~2× per generation; the other five are Llama-only). They become
landmines for SDXL or future model work, not SD 1.5.

## The mul_elementwise win (2026-05-08, second session)

Round 2 of the systematic audit found `ScryGpuBackend::mul_elementwise`
with the same fake-override pattern as matmul_bias. The single SD call
site was GeGLU's `values * gelu(gate)` step in `ff.gate` — 256 calls per
30-step image, with the deepest stage multiplying `[1024, 5120]` =
20 MB tensors. Every call was downloading both operands, multiplying on
host, and uploading the product, *then* forcing the next op
(`ff.proj_out` matmul_bias) to re-upload because the result variant was
`Cpu`.

Wiring a `MUL_ELEMENTWISE_CUDA` shader (mirror of the existing
`ADD_ELEMENTWISE_CUDA` — one character change) and a
`gpu_mul_elementwise_persistent` helper reproduced the matmul_bias
cascade pattern:

  Per-step:    700 ms  →  537 ms   (-23%)
  Per-call:    ff.gate           4.746 ms  →  0.060 ms  (**79× faster**)
               xfblock.ff        7.949 ms  →  2.467 ms  (-69%)
               ff.proj_out       1.250 ms  →  0.757 ms  (-39%)
               ff.proj_values    flat        flat       (already GPU)
               ff.proj_gate      flat        flat       (already GPU)

`ff.gate` went from 9.3% of UNet to 0.17%. The `ff.proj_out` -39% is
the same cascade matmul_bias produced — its previous re-upload of the
host-resident `ff.gate` output is gone.

Same session also lifted the **VAE decoder's local helpers** onto the
GPU. `vae/decoder.rs` had its own `clone_tensor`,
`transpose_chw_to_hwc`, `transpose_hwc_to_chw` that did `B::to_vec` +
scalar loop — the UNet equivalents in `unet/common.rs` were already
lifted during M9c (via `B::scale(_, 1.0)` and `B::transpose_2d`); the
VAE copies got missed. VAE runs once per image but moves an `[512,
4096]` = 8 MB tensor through `transpose_chw_to_hwc` → mid-attention →
`transpose_hwc_to_chw`. M4 parity unchanged at 2.176e-6.

**Generalized lesson.** Round 1 (matmul_bias) found *one* fake-override
on the SD hot path. Round 2 found *another*, plus a parallel set of
host-roundtrip helpers in the VAE that were copy-pasted from a pre-M9c
UNet. Both rounds came after the profile said "near floor". **Run the
fake-override check whenever per-step looks stuck**, and grep
`B::to_vec` across the diffusion crate to catch duplicated helpers
that didn't get the same lifting treatment as their primary copy.

## The to_device cascade win (2026-05-08, third session)

The biggest landmine of the entire perf pass — and one that hid
behind a perfectly normal call stack: **UNet/VAE/CLIP weights were
stored as `Cpu` storage and re-uploaded on every kernel dispatch**.
SD 1.5 UNet alone is 3.4 GB; the pipeline was paying that bandwidth
tax 60× per image (2 CFG × 30 steps).

Two related symptoms once the cause was identified:

1. `as_gpu_buffer(Cpu(v))` in `scry-llm/src/backend/scry_gpu.rs:691-700`
   uploads on every call without caching — `ctx.dev.upload(v)` then
   fresh `Arc::new`, no shared handle across calls. Every weight
   tensor paid the full upload cost on every forward.
2. `as_gpu_buffer_bf16` short-circuits to `None` on `Cpu` storage
   (line 717-719). The `--bf16-matmul` toggle was therefore a **no-op
   for weights** — the bf16 fast path only fired for activation
   tensors that had already been lifted by an earlier op. Three
   sessions of bf16-matmul tuning had been measuring the wrong thing.

`scry-vision`'s ResNet/ViT/BatchNorm have `to_device(&mut self)`
methods that walk Tensor fields and call `B::to_device_in_place`.
`scry-diffusion` had **zero** such cascade — confirmed by `grep`:
`scry-vision` had 11 `to_device` methods, scry-diffusion had 0.

Fix: 14 new `to_device(&mut self)` methods walking the
UNet/VAE/CLIP trees bottom-up — `LayerNormModule` and
`CausalSelfAttention` (in `scry-llm`); `GroupNormParams`, `ResBlock`,
`Downsample`, `Upsample`, `DownBlock`, `MidBlock`, `UpBlock`,
`Attention`, `GeGluFf`, `BasicTransformerBlock`, `SpatialTransformer`,
`TimeEmbedding`, `Unet`, `VaeResnetBlock`, `VaeMidAttention`,
`VaeUpBlock`, `VaeDecoder`, `ClipBlock`, `ClipTextEncoder` (in
`scry-diffusion`). One pipeline-level `Txt2ImgPipeline::to_device`
walks all three. `TextEncoder` trait gets a default-no-op
`to_device` so the call site is generic over CLIP / SDXL.

  Per-step:    537 ms  →  159 ms   (-70%)
  Total/30 steps:  17.0 s  →  5.68 s
  Gap to PyTorch fp16:  16×  →  4.8×

Two effects compounded: (a) host→device upload of weights was
amortized from per-dispatch to once at load time, and (b) the bf16
fast path now actually fires for weight tensors, so Q/K/V/out
projections, FF projections, conv1×1, time-embed Linear, and CLIP MLP
all run as cuBLAS bf16 GemmEx instead of fp32.

**Lesson for future-you.** When a sister crate already has the
pattern (scry-vision's `to_device` cascade), a new crate with no
analogous wiring is the first place to look. The previous M9c
perf table called the gap "round-trip-bound", which was right —
but the dominant round-trip wasn't the activations (M9c rounds 1-2
fixed those). It was the **weights re-uploading**, hidden behind
`as_gpu_buffer`'s implicit upload-on-`Cpu`-variant fallback. The
fallback is a useful affordance (lets `CpuBackend` callers pass
`Cpu` storage everywhere) but silently masks a 3.4 GB/forward leak
when nobody walks the parameter tree at load time.

**The audit pattern that catches this in future crates.** Compare
`grep -rn "fn to_device" <new_crate>` against the same grep on
sister crates. If the new crate has zero matches and the sister
has many, the new crate has the latent leak. Run before declaring
any GPU-resident milestone done.

**Remaining audit items** (8, all currently dormant on SD's hot path
post-batched-attention but landmines for future code):

- `add_inplace` — same-shape elementwise add. Trivial kernel.
- `gather_columns`, `scatter_columns` — worked around for SD by
  batched-attention; new code paths could reintroduce.
- `gather_rows`, `scatter_rows` — Llama-only currently.
- `apply_causal_mask_and_scale`, `apply_batched_causal_mask_and_scale`
  — Llama-only.
- `matmul_i8_f32_bias` — quantized path, unused.

**The remaining levers (F / G / H)** are all multi-week and are the
right next chunks of work after the easy wins. Per-UNet-forward profile
at the post-session steady state (post-mul_elementwise):

```
resblock.forward     31%  cuDNN-bound, near floor
xfblock.self_attn    13%  needs FlashAttention
xfblock.ff            8%  matmul-bound, near floor (was 39% pre-mul_elementwise)
xfblock.cross_attn    3%
unet.mid_block        5%
outer overhead       40%  up_blocks/down_blocks aggregate residual
```

- **F: CFG batching** (UNet at batch=2) — single biggest theoretical
  win, ~50% step. Major refactor: UNet's op call sites all assume
  batch=1, plus all the helpers (`transpose_chw_to_hwc`,
  `concat_channels`) need to handle batched inputs. 1-2 weeks of
  careful changes.
- **G: Fused attention** — collapses Q·Kᵀ → softmax → ·V into one
  streaming kernel. Self-attn drops from 4.1 ms/call to maybe
  1.5-2 ms. Multi-week kernel work; reference HF `xformers` /
  FlashAttention but write in-tree (no Triton dep).
- **H: CUDA graphs** — Phase 1 unblocked this. Records the UNet
  forward into a `cudaGraph_t`, replays it 60×/image. Eliminates
  per-kernel launch overhead. Requires extending `Device::batch()` /
  `Device::run_configured_async` to expose a recordable graph
  primitive.

The session's **legacy "denoise-loop residency" hypothesis** below was
the going-in framing for M9c. It turned out to be directionally right
(transfers ARE expensive) but pointed at the wrong specific transfers
— the latent itself was small (64 KB) and per-step transfers totaled
<1 MB. The actual culprits were the *intermediate-tensor* host
roundtrips inside the UNet forward (gather_columns, transpose_2d,
concat_rows, etc.) which moved tens of MB per call. Profile-driven
work found and fixed them. Keep this section as a record of what the
M9c plan looked like *before* measurement.

**Original M9c framing (pre-profile):** `pipeline.rs:141-158`
incurred **4 GPU↔CPU transfers per step** under `ScryGpuBackend`:
`scheduler.scale_model_input` took/returned `Vec<f32>`, CFG combine ran
on host (`cond_eps.to_vec()` + `uncond_eps.to_vec()`), and
`scheduler.step` took/returned `&[f32]`/`Vec<f32>`. Phase 1 fixed
this; the bigger wins came from the *next* layer of host roundtrips
that profile-driven work surfaced.

**Phased plan.** Land in order; gate each phase with a re-run of
`bench_sd --steps 30 --size 512` so the wins are auditable.

1. **Genericize the scheduler — done 2026-05-08, commit `4408375`.**
   `Scheduler` is now `Scheduler<B: MathBackend>`; `scale_model_input`
   and `step` take/return `&Tensor<B>`. DDIM step rewritten as
   `out = c_in · x + c_pred · model_output` with two host-precomputed
   coefficients — two `B::scale` + one `B::add` on the device per
   step. CFG combine in `pipeline.rs` likewise tensor-typed. The
   default `scale_model_input` uses `B::scale(_, 1.0)` to materialize
   a fresh storage without requiring `B: Clone` on `Tensor<B>`.

   **Reality check on the win:** the audit's "residency-bound"
   framing was directionally right but quantitatively overstated.
   At 512×512, latents are 64 KB; per-step transfers total <1 MB at
   >10 GB/s — under 0.1 ms / step. Post-phase-1 numbers are flat:

   | Backend | Size | Steps | Per-step (pre) | Per-step (post) |
   |---|---|---|---|---|
   | GPU bf16 | 64×64 | 4 | 446 ms | 458 ms |
   | GPU bf16 | 512×512 | 2 | 2066 ms | 2075 ms |

   The 60× gap vs PyTorch fp16 is **dispatch-count bound**, not
   memory-bandwidth bound. SD 1.5's UNet runs hundreds of kernel
   launches per step, each with 10-20 µs of overhead.

   **Phase 1's value is structural** — it's the prerequisite for
   phase 4 (CUDA graphs), which can only record over a pipeline that
   never syncs back to host between dispatches.

2. **Flip bf16 by default for UNet matmuls.** The `scry-gpu-bf16` feature
   plus `set_bf16_matmul(true)` is already wired and benched (row 3 vs
   row 2 above is roughly free on small shapes; ~2× win at SD shapes is
   expected once round-trips are gone). Either set the env var
   `SCRY_GPU_MATMUL_BF16=1` from inside `Txt2ImgPipeline::generate` when
   `scry-gpu-bf16` is enabled, or expose a `bf16` field on
   `GenerationParams` that flips the toggle at the start of `generate`.

3. **Fused attention kernel** (the "ViT round 3 attack surface" callout
   in `project_scry_vision_gpu_residency`). SD 1.5's UNet has 16
   `BasicTransformerBlock`s, each running self-attn + cross-attn. The
   current path goes Q/K/V projections → reshape → matmul (Q·K^T) →
   softmax → matmul (·V) → reshape → out_proj — 4 kernel dispatches
   plus 2 reshape kernels per attention. A single fused-attn kernel
   (FlashAttention-style with online softmax, or just a matmul-softmax-
   matmul fusion via `Device::batch()`) collapses that. Pattern to
   follow: `matmul_then_gelu_batched_for_bench` at
   `scry-llm/src/backend/scry_gpu.rs:1975` is the existing PoC for
   `Device::batch()`-mediated fusion; lift it into a real `MathBackend`
   path or write the fused-attn shader directly.

   **What was tried and DIDN'T work — fused KV projection (2026-05-08).**
   The intuition was: K and V both project from the same `kv_input`,
   so collapse the two `[n_kv, cross_dim] × [cross_dim, inner_dim]`
   matmuls into one `[n_kv, cross_dim] × [cross_dim, 2·inner_dim]`
   call, gather_columns to slice K and V halves per head. M6
   numerical-equivalence gate confirmed the math was correct
   (1.549e-4 max abs vs HF, bit-identical to pre-fuse). **But the
   bench regressed by ~5% at 512×512** (2186 vs 2075 ms/step) and
   was neutral at 64×64 (456 vs 458 ms/step). At SD self-attention
   shapes (n_kv=4096, cross_dim=320, inner_dim=320), cuBLAS GemmEx
   in bf16 picks a measurably worse algorithm for the wider M=640
   output than for two M=320 calls. The "fewer launches" intuition
   loses to cuBLAS algorithm selection. Don't retry this fusion in
   isolation — only as part of a full fused-attn kernel where the
   K/V output gets directly consumed by the next matmul without a
   gather_columns trip through global memory.

4. **CUDA graphs for the UNet step** (largest, do last). Once (1) keeps
   the latent on GPU, the UNet forward is a stable graph of dispatches
   per timestep — the only varying input is `t`'s sinusoidal embedding.
   Record once, replay 30×. Requires extending `Device::batch()` /
   `Device::run_configured_async` to expose a recordable graph
   primitive; cuBLAS dispatches won't capture into a Vulkan-style batch,
   so the bf16 path may need its own cuBLAS-graph recording.

The reference implementation to crib from in cases (3)/(4) is HF
diffusers' `xformers_memory_efficient_attention` and the cuBLAS
`cublasLtMatmulAlgoCapGetAttribute` graph-capture pattern. Don't try to
port FlashAttention's Triton kernel — write the WGSL/CUDA equivalents
in-tree.

**M10 — SDXL** (optional / opportunistic)

- New text encoder: `text_encoder/sdxl.rs` wrapping CLIP-L + OpenCLIP-bigG
  (concat embeddings, take pooled from bigG).
- Switch to `UnetConfig::sdxl_base()` and `VaeDecoderConfig::sdxl()`.
- Wire `Conditioning::extras` through the timestep MLP — the only UNet
  code change is in the time-embed pathway (everything else config-drives).

## Build commands

```bash
# Type-check (default features).
cargo check -p scry-diffusion

# Type-check with all features the next agent will need.
CUDARC_CUDA_VERSION=13010 cargo check -p scry-diffusion \
  --features safetensors,decode,scry-gpu-cuda,scry-gpu-bf16,scry-gpu-cudnn

# Run tests.
CUDARC_CUDA_VERSION=13010 cargo test -p scry-diffusion --features safetensors

# Driver (once M9 lands).
CUDARC_CUDA_VERSION=13010 cargo run -p scry-diffusion --release --example txt2img \
  --features safetensors,decode,scry-gpu-cuda,scry-gpu-bf16,scry-gpu-cudnn
```

`CUDARC_CUDA_VERSION=13010` is required everywhere CUDA features are on —
the system has CUDA 13.2 but cudarc 0.19.3 only knows 13.0.10. Without it,
cudarc panics at `Unsupported cuda toolkit version: 13.2`.

## Reference implementations to consult

- HF `diffusers` (Python): `pip install diffusers transformers safetensors`.
  Path-walk: `models/unet_2d_condition.py`, `models/attention.py`,
  `models/resnet.py`, `models/embeddings.py`, `schedulers/scheduling_ddim.py`,
  `pipelines/stable_diffusion/pipeline_stable_diffusion.py`.
- OpenAI CLIP: `https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py`
  for the BPE tokenizer.
- For numerical references at each milestone, dump intermediate tensors
  from a Python driver (`diffusers` step-by-step) and compare against our
  output. Same pattern `crates/scry-vision/bench_pytorch.py` uses for
  ResNet/ViT.

## Conventions

Inherit from the project's existing CLAUDE.md:

- `cargo check`, `cargo clippy --workspace -- -D warnings`,
  `cargo fmt --all -- --check`, `cargo test --workspace` all clean.
- `thiserror` for errors, `tracing` for logs, `#[repr(C)] + bytemuck::Pod`
  for any GPU-facing struct, `unwrap()` only in tests / `main()`.
- Foundation commit first, then feature commits.
- bf16 / GPU struct changes touch the kernel, the trait method, and the
  test in **the same commit**.

## Project-memory pointers (for the next agent)

Before starting: read these memories. They have load-bearing context the
scaffold doesn't repeat.

- `project_scry_vision_gpu_residency` — full activity log on phase 1
  (M1–M5 GPU residency, ResNet & ViT). Includes the bf16 envelope
  (~5% relative tolerance), the CUDARC_CUDA_VERSION gotcha, and the
  "fused attention is the next attack surface" finding from ViT round 3.
- The `gpu-resident/scry-vision` branch may still hold an open PR
  (`bf16 strided batched matmul`); confirm via `gh pr list`.
