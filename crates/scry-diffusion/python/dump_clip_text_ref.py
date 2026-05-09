#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""HuggingFace CLIP text encoder reference dump for scry-diffusion M3.

Loads `text_encoder/model.safetensors` from the SD 1.5 snapshot, runs HF's
`CLIPTextModel.forward()` on a fixed prompt, and saves
`last_hidden_state` (shape `[1, 77, 768]`) to a safetensors file under
`.assets/refs/`. The Rust example `check_text_encoder.rs` then loads
that file via the same `SafetensorsCheckpoint` we use everywhere else
and byte-compares it to our own forward output within 1e-4 abs.

No new file format: by reusing safetensors for the reference dump too,
the Rust side doesn't need an `.npy` reader, and the file is
self-describing (dtype + shape in the header).

The token IDs HF produces are also dumped (as int64) so the Rust
example can verify that our DIY CLIP BPE tokenizer agrees with
`transformers.CLIPTokenizer` before blaming the encoder for any drift
— a tokenizer disagreement on `[2..76]` would otherwise look like a
catastrophic encoder failure.

Run via the venv shared with scry-vision (which has CUDA torch):

    crates/scry-vision/.venv/bin/python \\
        crates/scry-diffusion/python/dump_clip_text_ref.py

Defaults assume the workspace root as cwd; flags below let you override
prompt, snapshot path, and output path.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        default="crates/scry-diffusion/.assets/sd-1-5",
        help="Path to the SD 1.5 snapshot root containing tokenizer/ and text_encoder/.",
    )
    parser.add_argument(
        "--prompt",
        default="a photo of a cat",
        help="Prompt to encode. Determines the output filename (slugified).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override the output safetensors path. Default: "
        "<snapshot>/../../refs/clip_text_<slug>.safetensors.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Run the forward pass on this device. CPU keeps the math fp32 "
        "deterministic; CUDA uses cuDNN's default kernels.",
    )
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    if not snapshot.is_dir():
        print(f"snapshot dir {snapshot} not found", file=sys.stderr)
        return 1
    text_encoder_dir = snapshot / "text_encoder"
    tokenizer_dir = snapshot / "tokenizer"
    for d in (text_encoder_dir, tokenizer_dir):
        if not d.is_dir():
            print(f"required subdir {d} not found", file=sys.stderr)
            return 1

    out_path = Path(args.out) if args.out else default_out_path(snapshot, args.prompt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading tokenizer from {tokenizer_dir}")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir.as_posix())

    print(f"loading text encoder from {text_encoder_dir}")
    # fp32 keeps the reference deterministic enough for a 1e-4 tolerance;
    # the SD 1.5 checkpoint is already stored as fp32, so this is a no-op.
    model = CLIPTextModel.from_pretrained(text_encoder_dir.as_posix(), torch_dtype=torch.float32)
    model.eval()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("--device cuda but no CUDA device visible", file=sys.stderr)
            return 1
        model = model.cuda()

    inputs = tokenizer(
        args.prompt,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids
    if args.device == "cuda":
        input_ids = input_ids.cuda()

    print(f"prompt: {args.prompt!r}")
    print(f"token ids: {input_ids.flatten().tolist()}")

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    last_hidden = outputs.last_hidden_state.detach().cpu().contiguous()
    input_ids_cpu = input_ids.detach().cpu().to(torch.int64).contiguous()

    print(f"last_hidden_state shape: {tuple(last_hidden.shape)}, "
          f"dtype: {last_hidden.dtype}, device: {last_hidden.device}")
    print(f"writing reference to {out_path}")
    save_file(
        {"last_hidden_state": last_hidden, "input_ids": input_ids_cpu},
        out_path.as_posix(),
        metadata={
            "prompt": args.prompt,
            "model": "runwayml/stable-diffusion-v1-5/text_encoder",
            "framework": f"pytorch-{torch.__version__}",
            "device": args.device,
        },
    )
    print(f"done; {os.path.getsize(out_path)} bytes")
    return 0


def default_out_path(snapshot: Path, prompt: str) -> Path:
    """Place reference dumps under `<snapshot>/../refs/` so the user's
    `.gitignore` (`.assets/`) keeps them out of git automatically.
    """
    slug = "".join(c if c.isalnum() else "_" for c in prompt.lower()).strip("_")
    refs_dir = snapshot.parent / "refs"
    return refs_dir / f"clip_text_{slug}.safetensors"


if __name__ == "__main__":
    sys.exit(main())
