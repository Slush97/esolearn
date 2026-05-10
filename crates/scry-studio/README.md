# scry-studio

Tauri 2 desktop app for `scry-llm`, `scry-vision`, and `scry-diffusion`.

Three tabs: **LLM** (GPT-2 generation + tokenizer + next-token logits), **Vision** (ResNet classify, SCRFD detect when built with `--features onnx`), **Diffusion** (SD-1.5 txt2img).

Excluded from the workspace (parallels `scry-app`, `scry-stt`).

## Prereqs

- Rust 1.85+
- pnpm (frontend)
- For diffusion / GPU paths: CUDA + cuBLAS, Vulkan drivers
- Linux: gtk3, webkit2gtk, libsoup3 (the usual Tauri 2 deps)

## Run

```bash
cd crates/scry-studio/frontend
pnpm install
pnpm tauri dev               # CPU build
pnpm tauri dev -- --features scry-gpu-cuda      # CUDA + cuBLAS
pnpm tauri dev -- --features "scry-gpu-cuda onnx"  # + SCRFD
```

`pnpm tauri dev` runs Vite (port 5173) and `cargo run` against the Rust shell.

## Build

```bash
cd crates/scry-studio/frontend
pnpm build                   # frontend → dist/
cd ..
cargo build --release        # native shell
```

## Layout

- `src/` — Tauri shell, command handlers per domain
- `frontend/` — Vue 3 (`<script setup>` TS) + Pinia + Tailwind v4 + Vite
- `frontend/src/lib/tauri.ts` — typed wrappers around `invoke()`
