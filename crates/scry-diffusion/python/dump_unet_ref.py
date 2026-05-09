#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""HuggingFace UNet reference dump for scry-diffusion M6.

Loads HF `UNet2DConditionModel` from the SD 1.5 snapshot, runs a single
forward at `t=981` against a deterministic noise latent and a fixed
conditioning embedding (sampled from CLIP-L on the same prompt the
parity gate uses), and saves the inputs alongside the output to a
safetensors blob under `.assets/refs/`. The Rust example
`check_unet.rs` then loads the same inputs and byte-compares its
forward against the saved `predicted_noise` within 1e-3 abs.

Run via the venv shared with scry-vision:

    crates/scry-vision/.venv/bin/python \\
        crates/scry-diffusion/python/dump_unet_ref.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        default="crates/scry-diffusion/.assets/sd-1-5",
        help="SD 1.5 snapshot root containing unet/, text_encoder/, tokenizer/.",
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
        default=16,
        help="Spatial size of the latent (output is 8× this). 16 keeps memory "
             "manageable on CPU; 64 mirrors a 512px generation.",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=981,
        help="Diffusion timestep to feed the UNet (DDIM step 0 of 30 ≈ 981).",
    )
    parser.add_argument(
        "--prompt",
        default="a photo of a cat",
        help="Prompt to encode into conditioning embeddings.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output path. Default: <snapshot>/../refs/unet_seed<S>.safetensors.",
    )
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    unet_dir = snapshot / "unet"
    text_dir = snapshot / "text_encoder"
    tok_dir = snapshot / "tokenizer"
    for d in (unet_dir, text_dir, tok_dir):
        if not d.is_dir():
            print(f"missing {d}", file=sys.stderr)
            return 1
    out_path = Path(args.out) if args.out else (
        snapshot.parent / "refs" / f"unet_seed{args.seed}.safetensors"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading UNet from {unet_dir}")
    unet = UNet2DConditionModel.from_pretrained(unet_dir.as_posix(), torch_dtype=torch.float32)
    unet.eval()

    print(f"loading text encoder + tokenizer from {snapshot}")
    tokenizer = CLIPTokenizer.from_pretrained(tok_dir.as_posix())
    text_model = CLIPTextModel.from_pretrained(text_dir.as_posix(), torch_dtype=torch.float32)
    text_model.eval()

    print(f"encoding prompt: {args.prompt!r}")
    # CLIP-L sequence length is 77; transformers reports model_max_length as
    # an enormous sentinel for some snapshots, which trips set_truncation.
    tok = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        cond = text_model(input_ids=tok.input_ids).last_hidden_state  # [1, 77, 768]
    print(f"  conditioning shape: {tuple(cond.shape)}")

    print(f"sampling latent: torch.manual_seed({args.seed}); randn(1, 4, {args.latent_size}, {args.latent_size})")
    torch.manual_seed(args.seed)
    latent = torch.randn(1, 4, args.latent_size, args.latent_size, dtype=torch.float32)

    timestep = torch.tensor([args.timestep], dtype=torch.long)

    with torch.no_grad():
        out = unet(latent, timestep, encoder_hidden_states=cond, return_dict=False)[0]
    out = out.detach().contiguous()
    print(f"output shape: {tuple(out.shape)}, "
          f"min={out.min().item():.3f}, max={out.max().item():.3f}, "
          f"mean={out.mean().item():.3e}")

    print(f"writing reference to {out_path}")
    save_file(
        {
            "latent": latent.contiguous(),
            "conditioning": cond.contiguous(),
            "timestep": torch.tensor([float(args.timestep)], dtype=torch.float32),
            "predicted_noise": out,
        },
        out_path.as_posix(),
        metadata={
            "seed": str(args.seed),
            "latent_size": str(args.latent_size),
            "timestep": str(args.timestep),
            "prompt": args.prompt,
            "model": "runwayml/stable-diffusion-v1-5/unet",
            "framework": f"pytorch-{torch.__version__}",
        },
    )
    print(f"done; {os.path.getsize(out_path)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
