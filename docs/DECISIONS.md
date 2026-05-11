# Architecture decisions

Append-only log. **Newest entries at the top.** Don't edit historical entries — supersede with a newer one and link back. Each entry: ~100-200 words, stating *what* was decided and *why*. The "why" is the load-bearing part — six-month-future-you will read this to remember the constraints, not just the conclusion.

When a decision is later reversed or evolved, the new entry should reference the superseded number (e.g., "supersedes 0004").

---

## 2026-05-10 · 0009 — M9h ships `head_dim ∈ {40, 160}` WMMA kernels; -17.9% per-step

**Decision:** Add WMMA tensor-core kernels for `head_dim = 40` and
`head_dim = 160` (joining the v2 `head_dim = 80` kernel) so the fused
path covers every attention shape in SD 1.5's UNet. The dispatcher
in `ScryGpuBackend::fused_attention` routes by head_dim and falls
through to the cuBLAS cascade only for shapes outside `{40, 80, 160}`.

**Why these two head_dims, in this order, in one milestone.** The
M9g v2 perf-neutral result wasn't a kernel-quality problem — it was a
coverage problem. SD 1.5's UNet stage map at `block_out_channels = [320,
640, 1280, 1280]` ÷ 8 heads gives `head_dim ∈ [40, 80, 160, 160]`. At
512×512 the stage-0 self-attention runs at `n = 4096` (latent 64×64
flattened), making its score matrix `[8, 4096, 4096] ≈ 128 MiB` per
head batch — the largest attention workload in the model by an order
of magnitude. Catching only stage 1 (d80) leaves the two biggest cost
buckets on the cascade. Shipping `d40` *and* `d160` together collapses
all three stages in one milestone rather than two, with shared
dispatcher / test / threshold plumbing.

**Implementation note on `head_dim = 40`.** 40 is not a multiple of
16, which the WMMA fragment loaders require. Two options: a special
"narrow" kernel without WMMA, or pad to `D_PAD = 48` in shared memory
and let WMMA run over the padded layout with zero-fill on the
trailing 8 cols. We picked padding because it preserves the d80 /
d160 kernel structure byte-for-byte (same loop nest, same online
softmax, just a different smem stride). The padded values contribute
0 to both matmuls, so the answer is identical to a strict d=40
kernel. The final store reads stride D_PAD=48 from smem and writes
stride D=40 to global. Smem cost is tiny (~9 KiB).

**Bench result (RTX 5070 Ti, 30 steps × 512×512, bf16-matmul):**
135.12 → 110.97 ms/step (-17.9%); 4834 → 4120 ms wall-clock (-14.8%);
gap vs PyTorch fp16 closes from 4.1× → 3.4×. Profile shows
`xfblock.self_attn` collapsing from 54% (M9f baseline) → 17.2% of
UNet wall-clock. M6 HF parity gate unchanged at 1.549e-4; golden
hash regenerated.

**Doesn't supersede 0008** — extends it. The v2 d80 kernel is byte-
identical; M9h adds two siblings and a shared dispatcher. Tuning
levers (4-warp blocks, larger BC tile, narrower d40 without padding)
remain available for M9i if profiles after CFG batching / CUDA
graphs land justify another attention pass.

---

## 2026-05-10 · 0008 — M9g v2 lands a tensor-core kernel for `head_dim = 80`; perf-neutral first cut, tuning is M9h

**Decision:** Re-enable the `ScryGpuBackend::fused_attention` override
on top of a new WMMA tensor-core kernel
(`FUSED_ATTENTION_TC_D80_CUDA`) specialized for `head_dim = 80`. The
kernel uses bf16 inputs / fp32 fp32 accumulators via `<mma.h>` WMMA
fragments — matches the M9g v0 trait shape, replaces the cuBLAS
strided-batched cascade for SD 1.5's mid-stage self-attention. Other
head dims (`{40, 160}`) still fall through to the cascade.

**Why ship it perf-neutral.** Production bench at 30 steps × 512×512
shows TC: 4926 ms vs cascade: 4986 ms — a 1.2% wall-clock improvement,
within run-to-run noise (3% per-run variance). Numerical correctness is
solid: `gpu_fused_attention_tc_d80_matches_cpu_within_tolerance` covers
4 SD shapes with `max_abs_diff` 2–7e-4 vs the CPU bf16-cascade
reference (5e-3 tolerance). The kernel is the load-bearing
*infrastructure*: it proves the WMMA path on this NVRTC + driver +
hardware combo, and gives a baseline to tune from. Shipping the
scaffolding now keeps the override clean for the M9h tuning passes
without dragging the kernel work into a separate milestone.

**Why first-cut perf is flat.** The kernel uses a single warp per
block (32 threads, each block computes 16 Q rows). That leaves
parallelism on the table: each block finishes faster than its launch
overhead at SD's mid-stage shapes. Plus only `head_dim = 80` is wired
up, which is roughly 1/3 of SD UNet attention layers — the other 2/3
still go through the cascade, so any per-call win is diluted. The
known levers for M9h are 4-warp blocks, larger BC tile, persistent Q
in registers, plus the `head_dim ∈ {40, 160}` specializations. None
require interface changes.

**What this preserves.** The v1 naive online-softmax kernel +
`gpu_fused_attention_persistent` helper + numerical-equivalence test
stay in the tree as `#[allow(dead_code)]` — the test still exercises
that kernel and pins its correctness. The `MathBackend::fused_attention`
trait method and `Attention::forward` integration point are unchanged
from v0. A runtime opt-out (`SCRY_GPU_FUSED_ATTN_TC=0`) reverts the
override to the cascade for clean A/B benchmarking.

**Supersedes 0007**'s "tensor-core variant deferred to a future
milestone" — that future milestone landed here. Tuning is M9h.

---

## 2026-05-10 · 0007 — M9g picks G (fused attention), v1 lands as scaffolding only

**Decision:** M9g (next perf milestone after M9f's profile) tackles **G —
fused attention** in preference to **F — CFG batching** and **H — CUDA
graphs**. A first kernel landed (FlashAttention-1 style online softmax,
correct within 1e-4 abs vs the cascade) but underperformed the cuBLAS
strided-gemm cascade by ~12× at SD production shapes. The
`ScryGpuBackend::fused_attention` override is disabled and routes
through the trait-default cascade; the kernel + helper + numerical
test are preserved as scaffolding. A real win requires rewriting the
matmul portions to use tensor cores via `mma.sync` PTX intrinsics —
deferred to a future milestone.

**Why G over F and H:** Per the M9f profile (`bench/profile_2026-05-10.txt`),
self-attention is 54% of UNet wall-clock and the
softmax + scores + values triple alone is 31.7% of UNet — the largest
single signal in the profile. F (CFG batching) ranked second by
ceiling (~1.6–1.9× UNet) but third by implementation cost: it requires
threading a batch dim through `scry-vision::Conv2d`, `scry-llm::group_norm`,
and every shape-handling layer in `scry-diffusion` (ResBlock, Attention,
GeGluFf, BasicTransformerBlock, SpatialTransformer, Down/Mid/UpBlock,
Unet, Conditioning) — a 1–2 week sister-crate refactor with real
golden-hash drift risk from batched matmul reduction order. H (CUDA
graphs) is small scope (~3–5 days) but only ~7% UNet — a warm-up,
not a milestone. G's contained scope (one kernel + one trait method +
one integration site) was the right size for M9g.

**Why the v1 kernel doesn't beat cuBLAS:** the bandwidth saved by
skipping the `[num_heads · n_q, n_kv]` softmax intermediate (~12.9%
of UNet per the M9f profile) is dwarfed by the matmul slowdown when
the dot-product reductions run on CUDA cores at ~5 TFLOPs fp32
instead of tensor cores at ~100 TFLOPs bf16. The numbers pencil out
cleanly: ~768 attention layers per image × ~80 ms naive cost ≈ 61 s,
matching the observed 59 s vs M9f's ~5 s baseline.

**What this preserves:** the `MathBackend::fused_attention` trait
method and `scry-diffusion`'s `Attention::forward` integration point
remain; re-enabling a tensor-core kernel is a one-line change to the
override body. The decision to fuse attention as a single trait call
(rather than overriding the three sub-ops) is locked in regardless
of the kernel internals.

**Reverses / supersedes:** none. F and H remain on the table for M9h.
Full per-commit notes and the per-attention cost breakdown live at
`crates/scry-diffusion/HANDOFF.md#m9g--fused-attention-2026-05-10`.

---

## 2026-05-10 · 0006 — Path C MVP scope: drop VLM and LoRA from v0.1

**Decision:** Cut LLM-vision (LLaVA-class multimodal) and LoRA loading from the v0.1 MVP. Remaining feature set: SD 1.5 + SDXL + LCM/Turbo + img2img + inpainting + GGUF chat + tool use. Estimated 14-15 weeks solo.

**Why:** Full feature list (VLM + LoRA included) priced out at 18-24 weeks, incompatible with the 15-week target. Cutting these two preserves the "chat + image gen" core while removing the highest-effort items not on the critical path. VLM (image-input chat) and LoRA (community fine-tunes) are both v0.2 candidates — both are additive, neither requires re-architecting the MVP. Tool use is kept because it's the integration story between the LLM and SD halves of the app, not just a feature.

---

## 2026-05-10 · 0005 — App lives at `crates/scry-studio` (clean rewrite)

**Decision:** The new MVP app reuses the `scry-studio` slot. The previously-scaffolded Tauri-based `scry-studio` is scrapped — it was untracked, judged "too elementary," and its framework choice is now superseded by 0004.

`crates/scry-app` (a separate Tauri-based scry-cv workbench) is left untouched as an independent concern. If it becomes abandoned, address in a future entry.

**Why:** `scry-studio` is the right semantic name for the LM-Studio-analog product. Reusing the slot avoids parallel scaffolding. A clean rewrite avoids inheriting Tauri scaffolding that conflicts with decision 0004.

---

## 2026-05-10 · 0004 — UI stack: egui via eframe (not Tauri, not webview)

**Decision:** The app uses [egui](https://github.com/emilk/egui) directly via `eframe` for v0.1. No webview, no JS toolchain, no IPC layer.

**Why:** Solo + 15-week timeline + native-tool feel is acceptable. egui ships as a single statically-linked binary, has no separate frontend build step, and stays idiomatic Rust end-to-end. Tauri's polish ceiling is higher but its complexity tax (separate frontend, IPC, build pipeline, devtools setup) is not justified at MVP scale. Reconsider for v0.2 if a native-feeling Tauri shell becomes a differentiator — by which point we'll know whether the UX gap actually matters to users.

---

## 2026-05-10 · 0003 — CUDA-only backend for app v0.1 (Vulkan stays in crate)

**Decision:** `scry-studio` v0.1 ships with the CUDA backend only. The `scry-gpu` crate continues to support Vulkan and CUDA; the app simply doesn't expose Vulkan in its build.

**Why:** Modern SD inference relies on cuDNN-class fused convs, FlashAttention-style kernels, and bf16 tensor cores — all of which the CUDA path either has or can grow into. The Vulkan path is competitive on raw matmul (28% of advertised peak, per scry-gpu benches) but per-kernel dispatch overhead and the absence of fused ops means an all-Vulkan app would feel sluggish on otherwise-capable hardware. Revisit when fused-attention lands and Vulkan dispatch is graph-recorded — at that point Vulkan-only NVIDIA + AMD + Intel coverage becomes a real story.

---

## 2026-05-10 · 0002 — Monorepo: app lives in this repo

**Decision:** `scry-studio` is added as a workspace member at `crates/scry-studio` rather than living in a separate repo.

**Why:** App and crates evolve together; cross-crate refactors (e.g. `Device` sharing for M20) are atomic in a monorepo. The workspace already builds 11 crates; adding one more is free. The app is internal-use until v0.1 ships, so there's no separate-repo benefit (independent versioning, separate issue tracker, license decoupling) yet. Revisit if the app gains external contributors or a release cadence that diverges sharply from the crates.

---

## 2026-05-10 · 0001 — Product framing: library showcase, not LM Studio competitor

**Decision:** The app exists to dogfood and demonstrate the `scry-*` crates. Marketing, comparisons, and feature priorities should not target parity with LM Studio, Ollama, Jan, GPT4All, or other established chat/LLM apps.

**Why:** Solo + 15 weeks cannot match incumbent UX teams. The defensible story is the full Rust stack — "custom Vulkan compute kernels → HF-parity SD inference → integrated chat + image gen, single binary, no Python or FFI" — which incumbents structurally cannot tell regardless of UX polish. Direct competitor framing would invite judgement against a polish bar we cannot meet, and would distort feature priorities (e.g., toward model marketplaces and plugin systems) that don't serve the core demonstration purpose.
