#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""HuggingFace DPM-Solver++ (2M) reference dump for scry-diffusion M9d.

Mirrors `dump_ddim_ref.py`: builds an HF `DPMSolverMultistepScheduler`
with the SD 1.5 config (scaled_linear betas, `steps_offset=1`,
`set_alpha_to_one=False`, ε-prediction) plus
`algorithm_type="dpmsolver++"`, `solver_order=2`,
`lower_order_final=True` — the canonical SD 1.5 DPM++ configuration.

Sets it for 30 inference steps, runs the multistep update chain over a
fixed initial latent and a fixed sequence of "predicted noise" vectors,
saves the trajectory to safetensors. The Rust example `check_dpmpp.rs`
replays the same chain with our `DpmSolverPpScheduler` and byte-compares
to the HF trajectory at 1e-4 abs.

Run via the venv shared with scry-vision:

    crates/scry-vision/.venv/bin/python \\
        crates/scry-diffusion/python/dump_dpmpp_ref.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler
from safetensors.torch import save_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument(
        "--out",
        default="crates/scry-diffusion/.assets/refs/dpmpp_seed42.safetensors",
    )
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Match SD 1.5's scheduler_config.json fields plus DPM++ choices.
    # Notes on defaults:
    #   - timestep_spacing="leading" + steps_offset=1 matches our DDIM trajectory.
    #   - final_sigmas_type="zero" makes σ at the boundary 0, which our Rust
    #     impl encodes as set_alpha_to_one=true on the DPM++ config.
    #   - DPM++ in diffusers does NOT accept `set_alpha_to_one` (it's a DDIM
    #     concept); the equivalent is final_sigmas_type.
    sched = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        algorithm_type="dpmsolver++",
        solver_order=2,
        lower_order_final=True,
        steps_offset=1,
        timestep_spacing="leading",
        final_sigmas_type="zero",
        prediction_type="epsilon",
        thresholding=False,
        solver_type="midpoint",
        use_karras_sigmas=False,
    )
    sched.set_timesteps(args.num_inference_steps)
    timesteps_long = sched.timesteps.to(torch.int64).clone()
    timesteps = timesteps_long.to(torch.float32)
    print(f"timesteps[0:5]={timesteps[:5].tolist()} ... timesteps[-1]={timesteps[-1].item()}")

    # Same RNG layout as the DDIM dump so cross-scheduler comparisons share
    # the same init latent / noise sequence (different seed for noise so
    # the two streams don't correlate).
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
        out = sched.step(eps, int(t.item()), latent, return_dict=True)
        latent = out.prev_sample
        intermediates.append(latent.clone())
    final_latent = latent
    print(
        f"final latent: shape={tuple(final_latent.shape)}, "
        f"min={final_latent.min().item():.3f}, max={final_latent.max().item():.3f}, "
        f"mean={final_latent.mean().item():.3e}"
    )

    intermediates_t = torch.cat(intermediates, dim=0).contiguous()
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
            "scheduler": "DPMSolverMultistepScheduler",
            "algorithm_type": "dpmsolver++",
            "solver_order": "2",
            "lower_order_final": "True",
            "framework": f"pytorch-{torch.__version__}",
        },
    )
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
