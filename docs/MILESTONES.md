# v0.1 Milestones

19 milestones across 6 phases, ~15 weeks. Each entry: scope, exit criteria, dependencies, risks. Copy into GitHub Project issues; one issue per milestone, sub-tasks as checklist items inside each issue.

This doc is **living** — refine exit criteria as work starts, update dependency chains as reality shifts. Don't rewrite history; supersede if a milestone is materially redefined.

Status legend: ⬜ not started · 🟡 in progress · ✅ done · 🟥 deferred to v0.2

---

## Phase 1 — Foundation (weeks 1-2)

> **Note for next agent:** This phase was rescoped on 2026-05-10 after re-reading
> `crates/scry-diffusion/HANDOFF.md` and walking the recent commit history.
> The original M9d "full bf16 dataflow" framing was based on a stale memory
> snapshot — current state is much further along. Sequence is now: ship the
> two cheap free-money items (M9d), lock in CI gates so subsequent work is
> measurable (M9e), profile the unaccounted "outer 40%" (M9f), then pick
> F/G/H based on what M9f shows (M9g). Don't skip to F/G/H without M9f.

### ⬜ M9d — Quick wins: bf16-default + DPM-Solver++
**Scope:** Two independent low-risk wins, neither touches hot paths.

1. **bf16-default in pipeline.** `Txt2ImgPipeline::generate` does not currently flip the bf16-matmul toggle on by default — users have to set `SCRY_GPU_MATMUL_BF16=1` or pass through the bench harness. When the `scry-gpu-bf16` feature is enabled, call `set_bf16_matmul(true)` at the start of `generate()`. Restore prior state on exit. The toggle is global on `ScryGpuBackend` (see `scry_gpu.rs:292` per HANDOFF), so save/restore semantics matter if the same process generates with multiple settings.

2. **DPM-Solver++ scheduler.** Add `crates/scry-diffusion/src/scheduler/dpm_solver_pp.rs` implementing `Scheduler<B>` with the same trait shape as `DdimScheduler`. Use the **2M (multistep order 2)** variant — most common production choice, what `diffusers.DPMSolverMultistepScheduler(algorithm_type="dpmsolver++")` defaults to. Pure CPU math on the scheduler side; the only `B::*` calls are the existing `B::scale` / `B::add` / clone-via-`B::scale(_, 1.0)` pattern from DDIM. Wire selection through `pipeline.rs` (e.g. a `SchedulerKind` enum on `GenerationParams` or pass an `Arc<dyn Scheduler<B>>` from caller).

**Exit:**
- bf16 default fires in `generate()` without the env var. `cargo run -p scry-diffusion --release --features safetensors,decode,scry-gpu-cuda,scry-gpu-bf16,scry-gpu-cudnn --example txt2img` shows the bf16 path active in the bench output (or a log line confirms toggle state).
- DPM-Solver++ HF parity gate at **1e-4 abs** vs `diffusers.DPMSolverMultistepScheduler(algorithm_type="dpmsolver++", solver_order=2)` on a fixed (latent, conditioning, t-schedule) input pulled from a Python ref dump (mirror M3/M4/M6 pattern: `python/dump_dpmpp_ref.py` + `examples/check_dpmpp.rs`).
- DPM-Solver++ at 20 steps produces a perceptually-equivalent 512×512 image vs DDIM at 30 steps on `"a photo of a cat"` seed=42 (no parity gate — DPM++ paths diverge from DDIM by design; eyeball check + commit a sample image).
- HF parity gates for CLIP / VAE / UNet unchanged (5.7e-6, 2.18e-6, 1.55e-4 max abs).
- `cargo nextest run -p scry-diffusion --features safetensors` green.
- Bench: 30-step 512×512 with default bf16 ≤ current `--bf16-matmul` numbers (~127 ms/step). 20-step DPM++ ≤ ~2.5s total wall-clock at 512×512.

**Depends on:** nothing.

**Files touched:**
- `crates/scry-diffusion/src/pipeline.rs` — toggle bf16 in `generate()`.
- `crates/scry-diffusion/src/scheduler/mod.rs` — re-export DPM++; possibly add `SchedulerKind` enum.
- `crates/scry-diffusion/src/scheduler/dpm_solver_pp.rs` — new file.
- `crates/scry-diffusion/python/dump_dpmpp_ref.py` — new file.
- `crates/scry-diffusion/examples/check_dpmpp.rs` — new file.
- `crates/scry-diffusion/tests/` — unit tests for DPM++ math.

**Risk:** DPM-Solver++ math has a few sign/exp conventions where it's easy to ship a bug that passes one test and fails another. The HF reference dump approach catches this — implement against the dump, not the paper alone. If 2M needs the previous step's noise prediction, the DDIM scheduler trait may need extension (an optional `&mut self` state slot the impl owns); diffusers stores it as `model_outputs` in the scheduler state, mirror that.

**Reference:** `diffusers/schedulers/scheduling_dpmsolver_multistep.py` — the `algorithm_type="dpmsolver++"` branches with `solver_order=2` are the canonical math.

---

### ⬜ M9e — CI gates: perf regression + golden image hash
**Scope:** Add `bench/history.jsonl` (append-only per-PR median step time at fixed config). CI workflow runs `bench_sd --steps 4 --size 64 --runs 1` (bf16 default after M9d), parses a JSON line, fails if >15% slower than the trailing-10 median. Add a golden-image hash test for txt2img (4 steps, fixed seed, 64×64) — hash committed; test fails on any change.

**Exit:**
- CI workflow file (`.github/workflows/perf.yml` or extension of existing CI).
- `bench/history.jsonl` exists with at least one baseline entry.
- Golden hash test in `crates/scry-diffusion/tests/`.
- A deliberate +20% regression PR fails CI; a deliberate hash-changing PR fails CI.

**Depends on:** M9d (baseline is bf16-default; lock in *after* the toggle flip so the trend line is consistent).

**Risk:** GPU runner availability in CI. If GitHub Actions doesn't have CUDA, this needs a self-hosted runner or local-only enforcement via pre-push hook — decision deferred to start of milestone.

---

### ⬜ M9f — Profile the outer 40% (diagnostic only)
**Scope:** The current per-UNet-forward profile (per HANDOFF) accounts for 60% of time in named sections (resblock 31%, self_attn 13%, ff 8%, cross_attn 3%, mid_block 5%) and lumps **40% as "outer overhead"** — up_blocks/down_blocks aggregate residual, helper kernels, launch overhead. We don't know which of those it is. Until we do, choosing F/G/H is partially guessing.

Use the existing `profile.rs` / `--profile` machinery. Add timing sections inside `up_blocks` and `down_blocks` covering: `concat_channels`, `transpose_chw_to_hwc` / `hwc_to_chw`, residual `add_same`, downsampler/upsampler convs, bias broadcasts. Run `bench_sd --profile --steps 4 --size 512 --bf16-matmul` (post-M9d, bf16 default).

**Exit:**
- New profile breakdown checked in as `bench/profile_2026-05-XX.txt` (or whatever date).
- The outer 40% is broken into named segments each ≥1% of UNet, with no remaining unaccounted bucket >5%.
- A short writeup (append to HANDOFF.md, or a new `docs/PROFILE_NOTES.md`) interpreting the breakdown: which segments look like helper-kernel traffic, which look like launch overhead, which are at floor.

**Depends on:** M9d (bf16 changes which sections dominate).

**Risk:** None significant — this is read-only diagnostic work. The output **does not commit us to a specific F/G/H choice**; that's M9g.

---

### ✅ M9g — Pick F/G/H based on M9f — done 2026-05-10
**Picked G (fused attention).** Three commits on `feat/m9g`:

- **v0** (`47cb5bc`) — `fused_attention` trait scaffolding + `Attention::forward` call site.
- **v1** (`4b6b95c`) — Naive CUDA-cores online-softmax kernel. Correct (1e-4 vs cascade) but ~12× slower at SD shapes. Kept as `#[allow(dead_code)]` scaffolding for the equivalence test.
- **v2** (`eb93d1a`) — WMMA tensor-core kernel for `head_dim = 80` only. FlashAttention-2 style, BR=BC=16, 1 warp/block, compute_80 arch. Correctness 2-7e-4 across 4 SD shapes; perf neutral (-1.2%, within noise) because d80 only catches SD 1.5's stage 1, not the wall-clock-dominant stages 0 + 2.

Continued in M9h.

### ✅ M9h — Full head_dim coverage for fused attention — done 2026-05-10
**Scope:** Add WMMA specializations for `head_dim ∈ {40, 160}` so the M9g v2 fused kernel covers every attention shape in SD 1.5's UNet. Stage 0 (head_dim=40, n=4096 at 512×512) was the wall-clock dominant shape that v2 missed.

**Implementation:**
- `FUSED_ATTENTION_TC_D160_CUDA` — structurally identical to d80 since 160 = 10 × 16; only `#define D 160` differs. Static shared memory grows to ~26.5 KiB.
- `FUSED_ATTENTION_TC_D40_CUDA` — pads the WMMA K-loop to `D_PAD = 48` in shared memory (40 is not a multiple of 16). Q/K/V loaders zero-fill cols 40..47; the padded contributions are 0 across both matmuls so the answer is identical to a strict d=40 kernel. Final store reads stride D_PAD=48 from smem and writes stride D=40 to global.
- `ScryGpuBackend::fused_attention` now routes through `gpu_fused_attention_tc_dispatch` which picks the matching kernel slot by head_dim. Old `_d80_persistent` helper was folded into a generic `_tc_persistent` body so all three head_dims share one dispatch path.

**Validation:**
- Per-head_dim WMMA-vs-cascade tests in `scry-gpu/tests/cuda_compute.rs` (all three pass, max_abs_diff well under the 5e-3 tolerance).
- Per-head_dim equivalence tests through the public override in `scry-llm` (all three pass).
- M6 HF parity gate holds at 1.549e-4 — same number as before M9h. Golden hash regenerated (kernel swap = different bit-level output).

**Perf (RTX 5070 Ti, 30 steps, 512×512, bf16-matmul, A/B via `SCRY_GPU_FUSED_ATTN_TC`):**

| | Cascade (M9g v2 baseline) | M9h (all 3 TC kernels) | Δ |
|---|---|---|---|
| Per-step | 135.12 ms | **110.97 ms** | **-17.9%** |
| 30-step image | 4834 ms | **4120 ms** | **-14.8%** |
| Gap vs PyTorch fp16 (32.9 ms/step) | 4.1× | **3.4×** | |

Profile post-M9h: `xfblock.self_attn` dropped from 54% (M9f) → 17.2% of UNet wall-clock. The fused-attention dispatch (`attn.fused` section, 16.45%) is now the largest single line, but it's bounded by tensor-core throughput rather than memory bandwidth on softmax-intermediate writes. Next frontier is convolutions (resblock 12.4% + up/down blocks containing most of the remaining 38%) or CUDA graphs.

**Open levers for M9i+:** 4-warp blocks (estimated additional 5-10% on the larger shapes where one warp doesn't saturate the SM), `head_dim=40` cleanup (eliminate the 8-col padding by adding a single ragged WMMA tile — micro-optimization, probably <2%), and longer-horizon F (CFG batching) / H (CUDA graphs).

**Failed-experiments callout (do not retry standalone):** Fused KV projection was tried (single `[n_kv, cross_dim] × [cross_dim, 2·inner_dim]` matmul + gather_columns to slice K/V halves). Math was correct (1.549e-4 vs HF, bit-identical to pre-fuse), but bench regressed ~5% at 512×512. Root cause: cuBLAS GemmEx in bf16 picks a worse algorithm for the wider M output. Only retry as part of a full fused-attn kernel where K/V output gets directly consumed by the next matmul without a `gather_columns` round-trip through global memory.

**Remaining F/H levers (deferred):** F (CFG batching) is still the largest theoretical win on the table — UNet at batch=2 for cond + uncond in one pass, HANDOFF estimates ~50% step win, 1-2 week refactor across every UNet op call site. H (CUDA graphs) is orthogonal and layers on top of any other lever; requires extending `Device::batch()` with a recordable graph primitive and a separate cuBLAS-graph recording path for the bf16 dispatches.

---

## Phase 2 — SD features (weeks 3-5)

### ⬜ M10 — img2img pipeline
**Scope:** VAE encoder forward (mirrors existing decoder; 108 keys to load, ~3 weeks of M4's structural work but ~1 week given the decoder's pattern is now well-understood). Pipeline `img2img(image, prompt, strength)` that initializes the latent from the encoded image plus scheduled noise.

**Exit:**
- HF parity vs `StableDiffusionImg2ImgPipeline` at 1e-3 abs on a fixed seed + image + strength.
- `examples/check_img2img.rs` matches HF dump.
- Smoke test in `tests/sd15_assets_smoke.rs`.

**Depends on:** M9d.

**Risk:** VAE encoder's `quant_conv` output sampling (mean+logvar then `randn`-scaled). Use the same dump-fixed-input pattern as M3/M4 to avoid cross-language RNG drift.

---

### ⬜ M11 — inpainting pipeline
**Scope:** SD 1.5 inpainting checkpoint (`runwayml/stable-diffusion-inpainting`) has a 9-channel UNet conv_in (4 latent + 4 masked latent + 1 mask). Loader path adapts. Pipeline takes image + mask + prompt.

**Exit:**
- HF parity vs `StableDiffusionInpaintPipeline` at 1e-3.
- `examples/check_inpaint.rs` passes.
- Manual visual: mask a region of a real photo, confirm regen looks plausible.

**Depends on:** M10 (shared encode path).

**Risk:** Different checkpoint = separate weight management. Document in `ASSETS.md`.

---

### ⬜ M12 — LCM + SDXL-Turbo schedulers
**Scope:** Add `LcmScheduler` and `TurboScheduler` (single-step) implementations of `Scheduler<B>`. Wire scheduler selection through pipeline. No model changes — these are pure CPU scheduler math + sigma schedules.

**Exit:**
- 4-step LCM produces visually-acceptable output on standard prompts (no parity gate — LCM is non-deterministic vs DDIM by design).
- 1-step SDXL-Turbo placeholder ready (will be wired in M13c).
- Bench numbers documented for 4-step LCM 512×512.

**Depends on:** M9d.

**Risk:** None significant — scheduler math is well-documented.

---

## Phase 3 — SDXL (weeks 6-8) — highest-risk phase

### ⬜ M13a — OpenCLIP-bigG text encoder
**Scope:** SDXL uses two text encoders: the existing CLIP-L (already implemented) plus OpenCLIP-bigG (~32 layers, ~1280 width, different tokenizer vocab). Both encodings are concatenated along the channel dim and fed to UNet cross-attention.

**Exit:**
- HF parity vs OpenCLIP-bigG output at 1e-4.
- Dual-encoder concat path tested (bigG + CLIP-L → 2048-d embedding).
- `examples/check_text_encoder_xl.rs` byte-compares against an HF dump.

**Depends on:** M9d (assumes bf16 baseline locked in).

**Risk:** OpenCLIP tokenizer differs slightly from CLIP-L (different merges file, different special tokens). Dump test catches drift but expect 1-2 days of tokenizer debugging.

---

### ⬜ M13b — SDXL UNet
**Scope:** Larger UNet, transformer-layers-per-block varies (10, 10, 0 — not uniform like SD 1.5). Time conditioning has added kwargs: target image size, original image size, crop coordinates. Loader handles ~2.6 GB of weights (vs SD 1.5's ~860 MB).

**Exit:**
- HF parity vs `UNet2DConditionModel` (SDXL config) at 1e-3.
- `examples/check_unet_xl.rs` byte-compares.
- Smoke test for 1024×1024 latent forward (cap at 256×256 for CI to keep runtime bounded).

**Depends on:** M13a.

**Risk:** **Highest in plan.** Three weeks budgeted; if not parity-passing by end of week 9, cut from MVP per slip rule. Watch for: per-block transformer-layer count mismatch silently zeroing output, added cond kwargs ordering, cross-attention dim now 2048 not 768.

---

### ⬜ M13c — SDXL VAE + pipeline wire-up
**Scope:** SDXL VAE has a different config (no scaling factor in some forks; fp16-unsafe in places — fp32 fallback for known problematic ops). Wire the full SDXL txt2img pipeline using M13a + M13b + M13c. Wire SDXL-Turbo (1-step) on top.

**Exit:**
- HF parity vs `AutoencoderKL` (SDXL config) at 1e-3.
- End-to-end SDXL txt2img generates a 1024×1024 image in <30s.
- SDXL-Turbo 1-step generates a 512×512 image in <2s.
- Golden image hashes locked in for SDXL + Turbo.

**Depends on:** M13a, M13b.

**Risk:** SDXL VAE numerical instability is documented in the diffusers ecosystem. Ship with the fp32 fallback for affected ops.

---

## Phase 4 — LLM (weeks 9-10)

### ⬜ M14 — GGUF loader for scry-llm
**Scope:** Read GGUF format (header + metadata + tensor table + tensor data). Support Q4_K_M, Q5_K_M, Q8_0 quant variants. Map to existing `scry-llm` weight types (potentially extending if quantized matmul kernels are needed — most likely they are).

**Exit:**
- Load Llama 3 8B Instruct in Q4_K_M.
- Perplexity on a fixed test prompt within 0.1 of llama.cpp's output on the same quant.
- `examples/run_gguf.rs` runs a chat completion.

**Depends on:** nothing direct (parallel with Phase 3 if time permits, but solo means serial).

**Risk:** Q4_K_M dequant is non-trivial — block-wise scales, mins, K-quant superblock structure. Reference llama.cpp's `ggml-quants.c`. CPU dequant first; CUDA path optional for v0.1.

---

### ⬜ M15 — streaming token API + cancellation
**Scope:** Refactor `scry-llm` generation to expose a token-by-token stream (channel-based or async iterator). Thread a cancellation token through the inference loop so callers can abort cleanly.

**Exit:**
- Token stream observable from caller code; `cargo run` example shows live output.
- Cancellation mid-generation releases all GPU resources, no leaks (verified via repeated start/cancel cycles + memory snapshot).
- Existing non-streaming API still works (kept as `generate_blocking` wrapper).

**Depends on:** M14.

**Risk:** Cancellation in the middle of a kernel batch — needs to drain in-flight work before tearing down. Use ticket/fence pattern from scry-gpu's async dispatch.

---

### ⬜ M16 — tool-use parsing
**Scope:** Constrained sampling or post-hoc parsing for structured output. Define a `generate_image` tool schema; LLM emits JSON; parser dispatches to SD pipeline. JSON-mode (sampling-time grammar enforcement) is preferred but post-hoc parse with retry is acceptable for v0.1.

**Exit:**
- Llama 3 8B Instruct with a tool-prompt reliably emits valid JSON for `generate_image` (>90% on a 20-prompt test set).
- Parser dispatches to `scry-diffusion::txt2img` and returns the result back into the chat context.
- End-to-end: user types "show me a sunset over mountains" → LLM tool-calls SD → image appears in the same chat panel.

**Depends on:** M15. Also indirectly depends on M9d / M10 (SD pipeline must be callable).

**Risk:** Llama 3 8B's tool-calling reliability is decent but not perfect. Build retry-on-parse-fail. Avoid over-engineering — the integration moment is what matters; full grammar enforcement is post-MVP.

---

## Phase 5 — App shell (weeks 11-13)

### ⬜ M17 — scry-studio scaffold (egui + model picker + settings)
**Scope:** Scrap old `crates/scry-studio` Tauri code. Create new `crates/scry-studio` with `eframe`/`egui`. Add to workspace members (remove from excludes). Three top-level windows/tabs: Models, Chat, Image. Settings persisted to TOML in `~/.config/scry-studio/`.

**Exit:**
- `cargo run --release -p scry-studio` opens a window.
- User can browse to a directory and see detected GGUF + safetensors files.
- Selecting a model loads it (chat or image, depending on type) — for now just prints to stdout, no inference.
- Settings TOML round-trips correctly.

**Depends on:** decision 0004/0005 from `docs/DECISIONS.md`.

**Risk:** First time touching egui in earnest. Budget extra time for layout struggles. Use `egui_demo_app` source as reference.

---

### ⬜ M18 — chat panel with streaming
**Scope:** Wire chat panel to scry-llm streaming API (M15). User types prompt, hits send, sees tokens stream in. History scrollback. Cancel button mid-stream. Regenerate button after completion.

**Exit:**
- Smooth streaming at full LLM token rate (no UI stutter).
- Cancel works mid-token without freezing UI.
- Regenerate produces a different completion with the same context.
- History persists to disk on chat close (JSONL).

**Depends on:** M15, M17.

**Risk:** egui's immediate-mode + a streaming background task needs careful channel + repaint-request wiring. Use `ctx.request_repaint_after()` on each token.

---

### ⬜ M19 — image panel with progress
**Scope:** Wire image panel to scry-diffusion pipelines. Inputs: prompt, negative prompt, scheduler dropdown, steps slider, CFG slider, size dropdown. Live progress bar via existing per-step callback. Cancel mid-generation. Generated images shown inline + saved to disk.

**Exit:**
- All schedulers from M12 selectable.
- Progress bar updates per step without stutter.
- Cancel cleans up GPU resources (verified by repeated cancels).
- Image saved with metadata sidecar (prompt, seed, scheduler, etc.).

**Depends on:** M9d, M10, M11, M12, M13c (any subset that's MVP-bound), M17.

**Risk:** Showing per-step preview (decoded latent) is tempting but expensive — defer unless free.

---

### ⬜ M20 — shared GPU context + load/unload
**Scope:** Refactor `scry-llm` and `scry-diffusion` to accept an external `Arc<Device>` instead of constructing their own. App owns the singleton `Device`. Add explicit `unload()` on each model so the app can swap between LLM-mode and image-mode without leaking VRAM.

**Exit:**
- Swap between Llama 3 8B Q4 and SD 1.5 in <5s, no VRAM leak across 10 swap cycles (verified via `nvidia-smi` snapshot).
- Both crates compile with the external-device API; existing `cargo run -p` examples still work (constructed device passed in).
- Clear error if user tries to load both simultaneously without enough VRAM.

**Depends on:** M14, M17.

**Risk:** Multi-crate refactor with public API change. Surface this work *as early as possible* (start scoping during Phase 1 or 2 if any free cycles); leaving it for week 12 is the riskiest sequencing choice in this plan. **Consider promoting to Phase 1.5.**

---

### ⬜ M21 — history persistence + UX polish
**Scope:** Per-conversation history (JSONL or SQLite — pick during M21 start), per-image metadata sidecar files. Export/import. Settings expansion: model directories, default scheduler, default size, theme.

**Exit:**
- Closing and reopening the app restores last conversation + last-used settings.
- Image library view shows all generated images with metadata.
- Settings round-trip correctly across version upgrades (forward-compat in TOML schema).

**Depends on:** M18, M19.

**Risk:** Scope creep — "polish" is unbounded. Hold to the exit criteria; defer everything else to v0.2.

---

## Phase 6 — Release (weeks 14-15)

### ⬜ M22 — packaging + first-run experience + README
**Scope:** Linux + Windows builds (no macOS — see decision 0006-adjacent). Static binary where possible. README walks a clean clone through model download → first chat → first image, with screenshots. First-run experience: detect no models in default dir, link to HF download instructions.

**Exit:**
- Fresh clone on a clean Linux VM produces a working binary in <5min.
- README screenshot validated on actual fresh-clone screenshot (not staged).
- Windows `.exe` builds and runs.
- Binary size <200 MB (without cuDNN bundling — assume system install).

**Depends on:** all prior milestones.

**Risk:** Windows-specific bugs. Plan for 2-3 days of "it worked on Linux but not Windows" debugging.

---

### ⬜ M23 — bug burndown + smoke testing
**Scope:** Self-imposed beta period. File issues for everything found. Fix only the blockers (anything that breaks Definition of Done in PROJECT.md).

**Exit:**
- All P0/P1 issues resolved.
- Non-blocker issues filed for v0.2.
- Manual run-through of Definition of Done passes end-to-end.

**Depends on:** M22.

**Risk:** Scope creep into v0.2. Hold the line.

---

### ⬜ M24 — v0.1 tag + announcement
**Scope:** Tag `scry-studio-v0.1.0` and supporting crate versions. Write announcement post (HN / r/rust / personal blog — your call). Optional release binaries on GitHub Releases.

**Exit:**
- Git tag pushed.
- CHANGELOG entries for all touched crates.
- Announcement written (publishing is your call).

**Depends on:** M23.

**Risk:** None — this is the victory lap.

---

## Tracking

- One GitHub issue per milestone above. Title format: `M<N> — <title>`.
- Use a single GitHub Project (board view) with columns: Backlog / This week / In progress / Blocked / Done.
- Link the issue back to this doc in the issue body so future-you can find the fuller context.
- Update **status legend** at the top of each milestone here as it progresses (don't rewrite the body — the original scope is part of the historical record).
