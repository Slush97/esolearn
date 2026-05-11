#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""HuggingFace VAE encoder reference dump for scry-diffusion M10.

Loads HF `AutoencoderKL` from `vae/`, samples a deterministic fake
image, runs `.encode()`, and saves the input image plus the
`(mean, logvar)` parameters of the resulting `DiagonalGaussianDistribution`
to a safetensors blob under `.assets/refs/`. Rust example
`check_vae_encoder.rs` then loads the same image and byte-compares its
forward against the saved `mean` / `logvar` tensors within 1e-3 abs.

Important details mirrored in the Rust forward:

* `vae.encode(image).latent_dist` is a `DiagonalGaussianDistribution`
  built from the 8-channel output of `quant_conv`. Its `.parameters`
  is the raw 8-ch tensor; `.mean` and `.logvar` are the channel-wise
  chunks, with `logvar` clamped to `[-30.0, 20.0]`. We dump the
  *post-chunk, post-clamp* `mean` and `logvar` because that's what
  the Rust `encode()` returns.

* The scaling factor (0.18215) is *not* applied here. HF's pipeline
  multiplies `latent_dist.sample() * scaling_factor` outside the
  encoder. The Rust side mirrors this — `encode()` returns unscaled
  `(mean, logvar)` and the caller scales after sampling.

* No clamping of pixels to (-1, 1) — the test feeds a deterministic
  `randn`-based fake image so we exercise the conv stack across the
  full dynamic range, not just the visible band.

Run via the venv shared with scry-vision:

    crates/scry-vision/.venv/bin/python \\
        crates/scry-diffusion/python/dump_vae_encoder_ref.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from safetensors.torch import save_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        default="crates/scry-diffusion/.assets/sd-1-5",
        help="SD 1.5 snapshot root containing vae/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="torch.manual_seed for the random input image.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Spatial size of the fake input image. Latent is 1/8 of this.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output path. Default: <snapshot>/../refs/vae_encoder_seed<S>.safetensors.",
    )
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    vae_dir = snapshot / "vae"
    if not vae_dir.is_dir():
        print(f"vae dir {vae_dir} not found", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else (
        snapshot.parent / "refs" / f"vae_encoder_seed{args.seed}.safetensors"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading VAE from {vae_dir}")
    vae = AutoencoderKL.from_pretrained(vae_dir.as_posix(), torch_dtype=torch.float32)
    vae.eval()

    print(
        f"sampling image: torch.manual_seed({args.seed}); "
        f"randn(1, 3, {args.image_size}, {args.image_size})"
    )
    torch.manual_seed(args.seed)
    image = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)

    with torch.no_grad():
        dist = vae.encode(image).latent_dist
        mean = dist.mean.detach().contiguous()
        logvar = dist.logvar.detach().contiguous()

    print(f"image shape:  {tuple(image.shape)}, dtype: {image.dtype}")
    print(
        f"mean shape:   {tuple(mean.shape)}, "
        f"min={mean.min().item():.3f}, max={mean.max().item():.3f}"
    )
    print(
        f"logvar shape: {tuple(logvar.shape)}, "
        f"min={logvar.min().item():.3f}, max={logvar.max().item():.3f} "
        f"(HF clamps to [-30, 20])"
    )
    print(f"writing reference to {out_path}")
    save_file(
        {"image": image, "mean": mean, "logvar": logvar},
        out_path.as_posix(),
        metadata={
            "seed": str(args.seed),
            "image_size": str(args.image_size),
            "model": "runwayml/stable-diffusion-v1-5/vae",
            "framework": f"pytorch-{torch.__version__}",
        },
    )
    print(f"done; {os.path.getsize(out_path)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
