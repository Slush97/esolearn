#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""HuggingFace img2img reference dump for scry-diffusion M10 v2.

The Rust img2img init-latent path composes three primitives end-to-end:

  1. `VaeEncoder::encode(image) -> (mean, logvar)`     (M10 v1, gated)
  2. Reparameterization: `latent = mean + exp(0.5·logvar) * noise_enc`
  3. `Scheduler::add_noise(scaled_latent, noise_add, t_start)`  (M10 v2)

Step 1 is already byte-parity-gated by `check_vae_encoder.rs`. This dump
exists so `check_img2img.rs` can validate steps 2 + 3 + the
`scaling_factor` multiply against HF in one shot, *without* needing
RNG byte-parity between torch and our SplitMix64 sampler: we dump the
noise tensors HF used and the Rust side feeds them back in verbatim.

What we dump:

* `image_norm` — `[3, H, W]` synthetic image, already normalized to
  `[-1, 1]` (HF's `image_processor.preprocess` does `2·x - 1` outside
  the VAE; we keep that step out of our pipeline too, see
  `SdPipeline::img2img` docstring).
* `noise_enc` — `[4, H/8, W/8]` standard-normal noise consumed by the
  reparameterization.
* `noise_add` — `[4, H/8, W/8]` standard-normal noise consumed by
  `scheduler.add_noise`.
* `init_latent_post_noise` — HF's final init latent for the denoise
  loop. Computed below the way the Rust pipeline computes it (NOT via
  `StableDiffusionImg2ImgPipeline` — that path injects normalization
  + image preprocessing + scheduler-order quirks we don't need to
  replicate). This is the target for the Rust max-abs check.
* `t_start_timestep` — scalar (f32) at which `add_noise` is called.

Scheduler: DDIM, SD 1.5 defaults. `num_inference_steps` and `strength`
together determine `t_start_idx`, mirroring HF's
`get_timesteps`/`prepare_latents` math:

    init_timestep = min(round(num_inference_steps * strength), N)
    t_start_idx   = N - init_timestep
    t_start       = timesteps[t_start_idx]

Run via the venv shared with scry-vision:

    crates/scry-vision/.venv/bin/python \\
        crates/scry-diffusion/python/dump_img2img_ref.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from safetensors.torch import save_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        default="crates/scry-diffusion/.assets/sd-1-5",
        help="SD 1.5 snapshot root containing vae/ and scheduler/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="torch.manual_seed for the image, noise_enc, and noise_add.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Spatial size of the fake input image. Latent is 1/8 of this.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Total scheduler timesteps before strength truncation.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="img2img strength in [0, 1]. HF default is 0.8; 0.6 keeps "
        "the parity check inside the middle of the schedule.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output path. Default: <snapshot>/../refs/img2img_seed<S>.safetensors.",
    )
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    vae_dir = snapshot / "vae"
    if not vae_dir.is_dir():
        print(f"vae dir {vae_dir} not found", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else (
        snapshot.parent / "refs" / f"img2img_seed{args.seed}.safetensors"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading VAE from {vae_dir}")
    vae = AutoencoderKL.from_pretrained(vae_dir.as_posix(), torch_dtype=torch.float32)
    vae.eval()
    scaling_factor = float(vae.config.scaling_factor)
    print(f"scaling_factor: {scaling_factor}")

    # Build the SD 1.5 DDIM scheduler. We construct it from the config
    # next to the vae so the alpha schedule matches what Rust's
    # `DdimConfig::sd_1_5()` builds (1000 train steps, scaled_linear,
    # beta 0.00085 -> 0.012, steps_offset=1, set_alpha_to_one=false).
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(args.num_inference_steps)
    timesteps = scheduler.timesteps  # tensor, descending
    init_timestep = min(int(round(args.num_inference_steps * args.strength)), args.num_inference_steps)
    t_start_idx = args.num_inference_steps - init_timestep
    if t_start_idx >= len(timesteps):
        print(
            f"strength {args.strength} gives empty denoise tail "
            f"(t_start_idx={t_start_idx}, len(timesteps)={len(timesteps)})",
            file=sys.stderr,
        )
        return 2
    t_start = int(timesteps[t_start_idx].item())
    print(
        f"timesteps: N={args.num_inference_steps}, strength={args.strength}, "
        f"init_timestep={init_timestep}, t_start_idx={t_start_idx}, t_start={t_start}"
    )

    # Synthetic image in [-1, 1]. HF's pipeline does `2*x - 1` on PIL
    # input; we mint the post-normalize tensor directly.
    torch.manual_seed(args.seed)
    image_unit = torch.rand(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    image_norm = 2.0 * image_unit - 1.0

    with torch.no_grad():
        dist = vae.encode(image_norm).latent_dist
        mean = dist.mean.detach().contiguous()
        logvar = dist.logvar.detach().contiguous()

    latent_shape = mean.shape  # [1, 4, H/8, W/8]

    # Reparameterization + scaling, mirrored from `SdPipeline::img2img`.
    noise_enc = torch.randn(latent_shape, dtype=torch.float32)
    std = torch.exp(0.5 * logvar)
    sampled = mean + std * noise_enc
    scaled_latent = sampled * scaling_factor

    # add_noise on the scaled latent at t_start.
    noise_add = torch.randn(latent_shape, dtype=torch.float32)
    init_latent_post_noise = scheduler.add_noise(
        scaled_latent, noise_add, torch.tensor([t_start])
    ).contiguous()

    # Strip the batch axis so the dump matches scry-llm's [C, H, W] convention.
    image_out = image_norm[0].contiguous()
    noise_enc_out = noise_enc[0].contiguous()
    noise_add_out = noise_add[0].contiguous()
    init_latent_out = init_latent_post_noise[0].contiguous()

    print(f"image_norm shape:               {tuple(image_out.shape)}")
    print(f"noise_enc shape:                {tuple(noise_enc_out.shape)}")
    print(f"noise_add shape:                {tuple(noise_add_out.shape)}")
    print(f"init_latent_post_noise shape:   {tuple(init_latent_out.shape)}, "
          f"min={init_latent_out.min().item():.3f}, max={init_latent_out.max().item():.3f}")
    print(f"writing reference to {out_path}")
    save_file(
        {
            "image_norm": image_out,
            "noise_enc": noise_enc_out,
            "noise_add": noise_add_out,
            "init_latent_post_noise": init_latent_out,
            "t_start_timestep": torch.tensor([float(t_start)], dtype=torch.float32),
        },
        out_path.as_posix(),
        metadata={
            "seed": str(args.seed),
            "image_size": str(args.image_size),
            "num_inference_steps": str(args.num_inference_steps),
            "strength": str(args.strength),
            "t_start": str(t_start),
            "scaling_factor": str(scaling_factor),
            "scheduler": "DDIMScheduler/sd-1-5",
            "framework": f"pytorch-{torch.__version__}",
        },
    )
    print(f"done; {os.path.getsize(out_path)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
