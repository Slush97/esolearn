#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""HuggingFace DDIM reference dump for scry-diffusion M7.

Builds an HF `DDIMScheduler` with the SD 1.5 config (scaled_linear betas,
`steps_offset=1`, `set_alpha_to_one=False`, ε-prediction), sets it for 30
inference steps, and runs the deterministic DDIM update chain over a
fixed initial latent and a fixed sequence of "predicted noise" vectors.
The Rust example `check_ddim.rs` replays the same chain with our
`DdimScheduler` and byte-compares to the HF trajectory.

We cannot share an RNG between PyTorch and Rust, so the noise sequence
is generated here and saved to the safetensors blob alongside the
trajectory. The Rust side reads the same bytes.

Run via the venv shared with scry-vision:

    crates/scry-vision/.venv/bin/python \\
        crates/scry-diffusion/python/dump_ddim_ref.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from safetensors.torch import save_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument(
        "--out",
        default="crates/scry-diffusion/.assets/refs/ddim_seed42.safetensors",
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Match SD 1.5's scheduler_config.json fields. SD 1.5 ships PNDM by
    # default but DDIM with these params is what HF docs recommend as a
    # drop-in deterministic sampler.
    sched = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type="epsilon",
    )
    sched.set_timesteps(args.num_inference_steps)
    # Keep an int64 copy for indexing alphas_cumprod inside `step`, plus a
    # float32 copy for the safetensors dump (Rust reads f32).
    timesteps_long = sched.timesteps.to(torch.int64).clone()
    timesteps = timesteps_long.to(torch.float32)
    print(f"timesteps[0:5]={timesteps[:5].tolist()} ... timesteps[-1]={timesteps[-1].item()}")

    # Deterministic init latent + per-step noise. Two separate seeds so
    # the sequences don't correlate.
    g_latent = torch.Generator().manual_seed(args.seed)
    init_latent = torch.randn(
        1, 4, args.latent_size, args.latent_size, generator=g_latent, dtype=torch.float32
    )

    g_eps = torch.Generator().manual_seed(args.seed + 1)
    eps_seq = torch.randn(
        args.num_inference_steps,
        4,
        args.latent_size,
        args.latent_size,
        generator=g_eps,
        dtype=torch.float32,
    )

    latent = init_latent.clone()
    intermediates = [latent.clone()]
    for step_idx, t in enumerate(timesteps_long):
        eps = eps_seq[step_idx : step_idx + 1]
        out = sched.step(eps, int(t.item()), latent, eta=0.0, return_dict=True)
        latent = out.prev_sample
        intermediates.append(latent.clone())
    final_latent = latent
    print(
        f"final latent: shape={tuple(final_latent.shape)}, "
        f"min={final_latent.min().item():.3f}, max={final_latent.max().item():.3f}, "
        f"mean={final_latent.mean().item():.3e}"
    )

    intermediates_t = torch.cat(intermediates, dim=0).contiguous()  # [steps+1, 4, N, N]
    save_file(
        {
            "init_latent": init_latent.contiguous(),
            "eps_seq": eps_seq.contiguous(),
            "timesteps": timesteps.contiguous(),
            "alphas_cumprod": sched.alphas_cumprod.to(torch.float32).contiguous(),
            "intermediates": intermediates_t,
            "final_latent": final_latent.contiguous(),
        },
        out_path.as_posix(),
        metadata={
            "seed": str(args.seed),
            "num_inference_steps": str(args.num_inference_steps),
            "latent_size": str(args.latent_size),
            "scheduler": "DDIMScheduler",
            "framework": f"pytorch-{torch.__version__}",
        },
    )
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
