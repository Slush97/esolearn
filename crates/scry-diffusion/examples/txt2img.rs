// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end txt2img driver.
//!
//! Loads SD 1.5 weights from a local HF snapshot and renders a prompt to a
//! PNG. Wired up in M9 once the underlying pipeline lands; for now this is
//! the shape of the call site the next agent will fill in.
//!
//! Usage (once implemented):
//! ```bash
//! CUDARC_CUDA_VERSION=13010 cargo run -p scry-diffusion --release \
//!   --example txt2img --features safetensors,decode,scry-gpu-cuda,scry-gpu-bf16,scry-gpu-cudnn \
//!   -- --model /path/to/sd-1.5 --prompt "a photo of a cat" --out cat.png
//! ```

fn main() {
    eprintln!(
        "scry-diffusion txt2img driver — not yet implemented. See \
         crates/scry-diffusion/HANDOFF.md for the milestone plan."
    );
}
