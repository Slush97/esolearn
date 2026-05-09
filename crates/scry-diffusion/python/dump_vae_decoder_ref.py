#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""HuggingFace VAE decoder reference dump for scry-diffusion M4.

Loads HF `AutoencoderKL` from `vae/`, samples a deterministic noise
latent, runs `.decode()`, and saves both the input latent and the
output image tensor (un-clamped, in (-1, 1)) to a safetensors blob
under `.assets/refs/`. Rust example `check_vae_decoder.rs` then loads
the same latent and byte-compares its forward against the saved
`decoded` tensor within 1e-3 abs.

Note on scaling: HF's `AutoencoderKL.decode()` does NOT internally
divide by the scaling factor — the *pipeline* is supposed to scale
the latent by `1 / scaling_factor` before calling decode. Our
`VaeDecoder::decode` includes the scale internally (per the scaffold's
config field), so we pass an unscaled latent on both sides and let
the multiplication land inside the decoder.

The output is *not* clamped to [-1, 1] — that's the pipeline's job.
This file dumps the raw decoder output so the numerical comparison
catches everything the decoder does, not just the visible band.

Run via the venv shared with scry-vision:

    crates/scry-vision/.venv/bin/python \\
        crates/scry-diffusion/python/dump_vae_decoder_ref.py
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
        help="torch.manual_seed for the random latent.",
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=64,
        help="Spatial size of the latent (output is 8× this). 64 → 512×512.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output path. Default: <snapshot>/../refs/vae_decoder_seed<S>.safetensors.",
    )
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    vae_dir = snapshot / "vae"
    if not vae_dir.is_dir():
        print(f"vae dir {vae_dir} not found", file=sys.stderr)
        return 1
    out_path = Path(args.out) if args.out else (
        snapshot.parent / "refs" / f"vae_decoder_seed{args.seed}.safetensors"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading VAE from {vae_dir}")
    vae = AutoencoderKL.from_pretrained(vae_dir.as_posix(), torch_dtype=torch.float32)
    vae.eval()

    print(f"sampling latent: torch.manual_seed({args.seed}); randn(1, 4, {args.latent_size}, {args.latent_size})")
    torch.manual_seed(args.seed)
    latent = torch.randn(1, 4, args.latent_size, args.latent_size, dtype=torch.float32)

    # Match what an SD pipeline would do: scale latent before decode.
    # AutoencoderKL.decode does NOT do this internally; the pipeline does.
    # Our Rust VaeDecoder::decode includes the scaling internally, so we
    # pass the *raw* latent there. To compare apples-to-apples here, we
    # also pass the raw latent to AutoencoderKL.decode AFTER scaling it
    # in the same place — i.e. multiply by 1/0.18215 before .decode().
    scaling = 1.0 / vae.config.scaling_factor
    print(f"  scaling_factor in config: {vae.config.scaling_factor}, "
          f"applying 1/scaling_factor = {scaling:.5f} pre-decode")
    scaled = latent * scaling

    with torch.no_grad():
        decoded = vae.decode(scaled, return_dict=False)[0]
    decoded = decoded.detach().contiguous()

    print(f"latent shape:  {tuple(latent.shape)}, dtype: {latent.dtype}")
    print(f"decoded shape: {tuple(decoded.shape)}, dtype: {decoded.dtype}, "
          f"min={decoded.min().item():.3f}, max={decoded.max().item():.3f}")
    print(f"writing reference to {out_path}")
    save_file(
        {"latent": latent, "decoded": decoded},
        out_path.as_posix(),
        metadata={
            "seed": str(args.seed),
            "latent_size": str(args.latent_size),
            "model": "runwayml/stable-diffusion-v1-5/vae",
            "framework": f"pytorch-{torch.__version__}",
        },
    )
    print(f"done; {os.path.getsize(out_path)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
