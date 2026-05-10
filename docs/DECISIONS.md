# Architecture decisions

Append-only log. **Newest entries at the top.** Don't edit historical entries — supersede with a newer one and link back. Each entry: ~100-200 words, stating *what* was decided and *why*. The "why" is the load-bearing part — six-month-future-you will read this to remember the constraints, not just the conclusion.

When a decision is later reversed or evolved, the new entry should reference the superseded number (e.g., "supersedes 0004").

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
