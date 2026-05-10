# esolearn — project plan

**Status:** active development.
**MVP target:** v0.1, ~15 weeks from 2026-05-10.

This repo hosts the `scry-*` AI/ML inference crates and the `esoc-*` graphics crates. The current product goal is **`scry-studio`**: a single-binary Rust desktop app that loads local LLM and Stable Diffusion checkpoints and lets a user chat and generate images, sharing one GPU context.

The crates are the headline; the app is the dogfooding harness that makes them tangibly useful.

This document is **living**. Update scope, definition of done, and roadmap as reality shifts — but only at phase boundaries, not mid-milestone.

## Vision

A pure-Rust, single-binary desktop app where you load a GGUF LLM and an SD checkpoint, chat in one panel, generate images in another, with the LLM able to invoke image generation as a tool. No Python runtime, no FFI to llama.cpp, no Electron. Shared GPU context across both models.

The differentiator is **not** UX polish — LM Studio, Ollama, Jan, GPT4All have that covered. The differentiator is the technical story: a fully-Rust stack from custom Vulkan compute kernels through HF-parity SD inference up to the chat UI, all in one process.

## In scope (MVP, v0.1)

**Diffusion**
- SD 1.5 + SDXL base (no refiner)
- DDIM, DPM-Solver++, LCM, SDXL-Turbo schedulers
- txt2img, img2img, inpainting
- bf16 throughout the SD inference path (not just matmul)

**LLM**
- GGUF loader (Q4_K_M, Q5_K_M, Q8_0)
- Streaming token API with cancellation
- Tool-use parsing (LLM can call SD via structured output)
- Single-model-loaded-at-a-time (swap, not co-resident)

**App (`crates/scry-studio`)**
- egui via `eframe`, single binary
- Model picker: point at a directory, pick a checkpoint
- Chat panel with streaming, history, cancel, regenerate
- Image panel with prompt/negative/scheduler/steps/CFG/size, progress bar
- Persistent settings (TOML), persistent chat history (JSONL or SQLite)
- CUDA backend only for v0.1 (Vulkan compute remains in the crate for ecosystem)

## Out of scope (post-MVP or never)

| Item | Disposition |
|---|---|
| Vision-capable LLMs (LLaVA-class) | post-MVP (v0.2) |
| LoRA loading | post-MVP (v0.2) |
| SDXL refiner | post-MVP (v0.2) |
| macOS support | post-MVP (no CUDA on Mac) |
| Multi-GPU | never (MVP) |
| Mobile / web build | never |
| Model marketplace / in-app downloads | never — point users at HF |
| Fine-tuning | never — separate project |
| Multi-user / hosted mode | never |

## Definition of done for v0.1

A user with an NVIDIA GPU and 16+ GB VRAM:
1. Clones the repo, runs `cargo run --release -p scry-studio`.
2. Picks a GGUF chat model and an SD or SDXL checkpoint from a local directory.
3. Within 30 seconds is chatting and generating images.
4. Can switch between chat-mode and image-mode (model swap, not co-resident).
5. Can cancel generations cleanly without GPU resource leaks.

The README walks a fresh clone through this end-to-end with screenshots.

## Roadmap (15 weeks, 6 phases)

Detailed milestones tracked in the GitHub Project board (M9d through M24). Phase summaries:

| Phase | Weeks | Focus | Headline milestones |
|---|---|---|---|
| 1 — Foundation | 1-2 | Quick perf wins, CI gates, profile drill-down | M9d (bf16-default + DPM++), M9e (CI gates), M9f (profile outer 40%) |
| 1.5 — Perf lever | 2-4 | Pick F/G/H from M9f findings; implement | M9g (CFG batch *or* fused attn *or* CUDA graphs) |
| 2 — SD features | 3-5 | img2img, inpainting, fast schedulers | M10, M11, M12 |
| 3 — SDXL | 6-8 | Highest-risk phase | M13a-c |
| 4 — LLM | 9-10 | GGUF + streaming + tool use | M14, M15, M16 |
| 5 — App shell | 11-13 | egui, panels, GPU mgmt, persistence | M17-M21 |
| 6 — Release | 14-15 | Packaging, README, polish | M22-M24 |

**Slip rule:** if SDXL is not passing parity by end of week 9, cut it from MVP and defer to v0.2. The timeline does not extend; scope absorbs the slip.

## Working agreements

- **Branching:** short-lived `feat/<milestone>` branches, PR to main, squash merge.
- **Commits:** Conventional Commits (`feat:`, `fix:`, `perf:`, `docs:`, `chore:`).
- **Tests:** `cargo nextest` for unit/integration; `cargo test --doc` periodically.
- **Per-PR quality gates:** fmt, clippy (workspace pedantic), nextest green, perf regression CI gate (≤15% step-time regression on the smoke bench), HF parity gates for any new model component, golden image hash unchanged.
- **Per-milestone documentation:** every milestone closes with a HANDOFF entry — either appended to the relevant per-crate HANDOFF doc or summarized in the GitHub Project issue. Future-self reads these to reload context after a week away.

## Metrics tracked

- **Perf:** `bench/history.jsonl` — append per-PR median step time at fixed config; CI gate fails on >15% regression vs trailing-10 median.
- **Parity:** HF parity test tolerances per component (currently CLIP 1e-4, VAE 1e-3, UNet 1e-3, DDIM 1e-4); regressions fail CI.
- **Output stability:** golden image hash for fixed seed + config; any change requires a manual override commit.

## See also

- `docs/DECISIONS.md` — append-only architecture decision log.
- `CLAUDE.md` — workspace coding conventions and commands.
- Per-crate `CLAUDE.md` and `HANDOFF*.md` files — crate-specific guidance.
