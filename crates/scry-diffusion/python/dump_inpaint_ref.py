#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""HuggingFace inpaint reference dump for scry-diffusion M11.

M11 adds two new pieces on top of the txt2img pipeline:

  1. Mask + masked-image prep:
       masked_image  = image * (1 - mask)
       masked_latent = vae.encode(masked_image).latent_dist.mode * scaling_factor
       mask_latent   = F.interpolate(mask, size=(h/8, w/8), mode='nearest')
  2. 9-channel UNet input:
       conv_in_input = concat([noisy_latent, mask_latent, masked_latent], dim=1)
                       └── 4 ch ──┘└── 1 ch ─┘└── 4 ch ──────────┘
       NB: HF's order is (noisy, mask, masked_latent) — mask comes between.

UNet weights match base SD 1.5 except `conv_in.weight` widens to
[320, 9, 3, 3]; everything downstream is identical. The denoise loop
math is unchanged.

This dump exercises ONE UNet forward (first denoise step, cond branch
only) with the 9-channel input, so `check_inpaint.rs` can parity-gate
the conv_in path without dragging in the full denoise loop, VAE
decode, or CFG combine — those are already gated elsewhere.

What we dump:

  * `image_norm`         [3, H, W]  — synthetic image in [-1, 1]
  * `mask`               [1, H, W]  — synthetic mask in {0, 1}
  * `masked_image`       [3, H, W]  — image * (1 - mask), pre-VAE
  * `masked_latent`      [4, h, w]  — vae.encode mode × scaling_factor
  * `mask_latent`        [1, h, w]  — mask resized nearest to latent res
  * `noise_init`         [4, h, w]  — torch.randn for the initial latent
  * `noisy_latent_t0`    [4, h, w]  — first-step latent (= noise × init_sigma)
  * `cond_embed`         [77, 768]  — CLIP-L text embedding (prompt)
  * `unet_input_t0`      [9, h, w]  — concat fed to UNet at step 0
  * `unet_out_t0`        [4, h, w]  — UNet output for step 0, cond branch
  * `t0`                 scalar     — timestep at step 0

The inpainting UNet config is constructed in-script (mirrors
runwayml/stable-diffusion-inpainting's unet/config.json modulo
in_channels=9). The text encoder + VAE are loaded from the existing
sd-1-5/ snapshot — they're byte-identical between base and inpainting.

Run via the venv shared with scry-vision:

    crates/scry-vision/.venv/bin/python \\
        crates/scry-diffusion/python/dump_inpaint_ref.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from safetensors.torch import load_file, save_file
from transformers import CLIPTextModel, CLIPTokenizer


# Matches runwayml/stable-diffusion-inpainting's unet/config.json
# (identical to SD 1.5 base except `in_channels = 9`).
INPAINT_UNET_CONFIG = dict(
    sample_size=64,
    in_channels=9,
    out_channels=4,
    center_input_sample=False,
    flip_sin_to_cos=True,
    freq_shift=0,
    down_block_types=[
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ],
    up_block_types=[
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ],
    block_out_channels=[320, 640, 1280, 1280],
    layers_per_block=2,
    downsample_padding=1,
    mid_block_scale_factor=1,
    act_fn="silu",
    norm_num_groups=32,
    norm_eps=1e-5,
    cross_attention_dim=768,
    attention_head_dim=8,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
)


def load_inpaint_unet(snapshot: Path) -> UNet2DConditionModel:
    """Build a UNet2DConditionModel with the inpaint config and load the
    safetensors from <snapshot>/unet/diffusion_pytorch_model.fp16.safetensors.
    Done this way to avoid needing unet/config.json on disk.
    """
    weights_path = snapshot / "unet" / "diffusion_pytorch_model.fp16.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing inpaint UNet weights: {weights_path}")
    unet = UNet2DConditionModel(**INPAINT_UNET_CONFIG)
    state = load_file(weights_path.as_posix())
    # Upcast fp16 → fp32 to match the rest of the dump (Rust path is fp32).
    state = {k: v.float() for k, v in state.items()}
    missing, unexpected = unet.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )
    unet.eval()
    return unet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-snapshot",
        default="crates/scry-diffusion/.assets/sd-1-5",
        help="SD 1.5 base snapshot (used for text_encoder/, vae/, tokenizer/).",
    )
    parser.add_argument(
        "--inpaint-snapshot",
        default="crates/scry-diffusion/.assets/sd-1-5-inpainting",
        help="SD 1.5 inpainting snapshot (used for unet/).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Spatial size of the synthetic image. Latent is 1/8 of this.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Scheduler timesteps. We only run step 0 but the schedule "
        "determines which timestep step 0 lands on.",
    )
    parser.add_argument(
        "--prompt",
        default="a photo of a cat",
        help="Conditioning prompt for the CLIP text encoder.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = Path(args.base_snapshot).resolve()
    inpaint = Path(args.inpaint_snapshot).resolve()
    for p, label in [(base / "vae", "vae"), (base / "text_encoder", "text_encoder"),
                     (base / "tokenizer", "tokenizer"), (inpaint / "unet", "unet")]:
        if not p.is_dir():
            print(f"missing {label} dir: {p}", file=sys.stderr)
            return 1

    out_path = Path(args.out) if args.out else (
        base.parent / "refs" / f"inpaint_seed{args.seed}.safetensors"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading text encoder from {base / 'text_encoder'}")
    text_encoder = CLIPTextModel.from_pretrained(
        (base / "text_encoder").as_posix(), torch_dtype=torch.float32
    )
    text_encoder.eval()
    tokenizer = CLIPTokenizer.from_pretrained((base / "tokenizer").as_posix())

    print(f"loading VAE from {base / 'vae'}")
    vae = AutoencoderKL.from_pretrained(
        (base / "vae").as_posix(), torch_dtype=torch.float32
    )
    vae.eval()
    scaling_factor = float(vae.config.scaling_factor)
    print(f"scaling_factor: {scaling_factor}")

    print(f"loading inpaint UNet from {inpaint / 'unet'}")
    unet = load_inpaint_unet(inpaint)

    # Scheduler (matches Rust DdimConfig::sd_1_5()).
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        set_alpha_to_one=False,
    )
    scheduler.set_timesteps(args.num_inference_steps)
    timesteps = scheduler.timesteps  # descending
    t0 = int(timesteps[0].item())
    print(f"timesteps[0] = {t0}")

    # ---- Synthetic image + mask ----------------------------------
    torch.manual_seed(args.seed)
    image_unit = torch.rand(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    image_norm = 2.0 * image_unit - 1.0  # [-1, 1]

    # Centered square mask covering ~25% of the image.
    mask = torch.zeros(1, 1, args.image_size, args.image_size, dtype=torch.float32)
    q = args.image_size // 4
    mask[:, :, q:3 * q, q:3 * q] = 1.0

    masked_image = image_norm * (1.0 - mask)

    # ---- VAE encode masked image (mode, NOT sample) --------------
    with torch.no_grad():
        masked_latent = vae.encode(masked_image).latent_dist.mode()
    masked_latent = masked_latent * scaling_factor

    # ---- Mask downsample to latent res ---------------------------
    h_lat = args.image_size // 8
    mask_latent = F.interpolate(mask, size=(h_lat, h_lat), mode="nearest")

    # ---- Init noisy latent ---------------------------------------
    latent_shape = (1, 4, h_lat, h_lat)
    noise_init = torch.randn(latent_shape, dtype=torch.float32)
    init_sigma = float(scheduler.init_noise_sigma)
    noisy_latent = noise_init * init_sigma

    # ---- Text encode prompt --------------------------------------
    tokens = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=77,  # CLIP-L context length; tokenizer.model_max_length is unset in some SD repos
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        cond_embed = text_encoder(tokens.input_ids).last_hidden_state  # [1, 77, 768]

    # ---- Build 9-channel UNet input ------------------------------
    # HF's pipeline_stable_diffusion_inpaint.py concat order is
    # (noisy_latent, mask, masked_image_latent) — mask is between.
    scaled_input = scheduler.scale_model_input(noisy_latent, t0)
    unet_input = torch.cat([scaled_input, mask_latent, masked_latent], dim=1)
    assert unet_input.shape == (1, 9, h_lat, h_lat), unet_input.shape

    # ---- UNet forward (cond branch) ------------------------------
    with torch.no_grad():
        unet_out = unet(
            unet_input,
            torch.tensor([t0], dtype=torch.float32),
            encoder_hidden_states=cond_embed,
        ).sample

    # ---- Strip batch axis ----------------------------------------
    image_out = image_norm[0].contiguous()
    mask_out = mask[0].contiguous()
    masked_image_out = masked_image[0].contiguous()
    masked_latent_out = masked_latent[0].contiguous()
    mask_latent_out = mask_latent[0].contiguous()
    noise_init_out = noise_init[0].contiguous()
    noisy_latent_out = noisy_latent[0].contiguous()
    cond_embed_out = cond_embed[0].contiguous()
    unet_input_out = unet_input[0].contiguous()
    unet_out_out = unet_out[0].contiguous()

    print(f"image_norm:     {tuple(image_out.shape)}")
    print(f"mask:           {tuple(mask_out.shape)}")
    print(f"masked_latent:  {tuple(masked_latent_out.shape)}  "
          f"min={masked_latent_out.min().item():.3f} max={masked_latent_out.max().item():.3f}")
    print(f"mask_latent:    {tuple(mask_latent_out.shape)}")
    print(f"unet_input:     {tuple(unet_input_out.shape)}")
    print(f"unet_out:       {tuple(unet_out_out.shape)}  "
          f"min={unet_out_out.min().item():.3f} max={unet_out_out.max().item():.3f}")

    print(f"\nwriting reference to {out_path}")
    save_file(
        {
            "image_norm": image_out,
            "mask": mask_out,
            "masked_image": masked_image_out,
            "masked_latent": masked_latent_out,
            "mask_latent": mask_latent_out,
            "noise_init": noise_init_out,
            "noisy_latent_t0": noisy_latent_out,
            "cond_embed": cond_embed_out,
            "unet_input_t0": unet_input_out,
            "unet_out_t0": unet_out_out,
            "t0": torch.tensor([float(t0)], dtype=torch.float32),
        },
        out_path.as_posix(),
        metadata={
            "seed": str(args.seed),
            "image_size": str(args.image_size),
            "num_inference_steps": str(args.num_inference_steps),
            "prompt": args.prompt,
            "scaling_factor": str(scaling_factor),
            "init_sigma": str(init_sigma),
            "t0": str(t0),
            "scheduler": "DDIMScheduler/sd-1-5",
            "concat_order": "noisy,mask,masked_latent",
            "framework": f"pytorch-{torch.__version__}",
        },
    )
    print(f"done; {os.path.getsize(out_path)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
