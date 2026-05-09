#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""PyTorch + diffusers reference numbers for scry-diffusion's `bench_sd`.

Times an end-to-end Stable Diffusion 1.5 txt2img generation with the same
prompt, seed, steps, CFG, and resolution our Rust `bench_sd` example uses,
so the two numbers can be compared directly. Reports per-step latency
(median over the denoising loop, excluding the CFG-doubled batch overhead)
and total wall-clock from `pipe()` entry to PIL image.

Run via the venv shared with scry-vision (already has torch + diffusers
+ transformers + safetensors installed):

    crates/scry-vision/.venv/bin/python \\
        crates/scry-diffusion/python/bench_pytorch.py

Reasonable PyTorch baselines on an RTX 5070 Ti per the M9 handoff:
fp16 ~3-5s, fp32 ~10-15s. The bench just measures kernel dispatch and
denoise time, not weight load or tokenizer construction.
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        default="crates/scry-diffusion/.assets/sd-1-5",
        help="SD 1.5 snapshot root (parent of unet/, vae/, text_encoder/, tokenizer/).",
    )
    parser.add_argument(
        "--prompt",
        default="a photo of an astronaut riding a horse on mars",
        help="Prompt to generate. Match this with `bench_sd --prompt` for parity.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt (empty disables).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--precision",
        choices=list(DTYPES),
        default="fp16",
        help="Weight + compute precision (default: fp16, the standard SD inference baseline).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="How many full generations to time. Median over runs after warmup.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Discarded warmup runs (lets CUDA caching allocator + cudnn pick kernels).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run on. CPU is slow but useful for parity vs scry-diffusion's CPU path.",
    )
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    if not snapshot.exists():
        print(f"snapshot not found: {snapshot}")
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; rerun with --device cpu.")
        return 1

    device = torch.device(args.device)
    dtype = DTYPES[args.precision]

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("=== diffusers SD 1.5 bench ===")
    print(f"PyTorch     : {torch.__version__}")
    if device.type == "cuda":
        print(f"cuDNN       : {torch.backends.cudnn.version()}")
        print(f"Device      : {torch.cuda.get_device_name(0)}")
    else:
        print(f"Device      : cpu")
    print(f"Precision   : {args.precision}")
    print(f"Snapshot    : {snapshot}")
    print(f"Prompt      : {args.prompt!r}")
    if args.negative_prompt:
        print(f"Negative    : {args.negative_prompt!r}")
    print(f"Size        : {args.size}x{args.size}")
    print(f"Steps       : {args.steps}, CFG: {args.cfg}, seed: {args.seed}")
    print(f"Runs        : {args.runs} timed (after {args.warmup} warmup)")
    print()

    # Build the pipeline programmatically from the per-component dirs that
    # already live in the snapshot. The full HF SD 1.5 release also ships a
    # `model_index.json` + `scheduler/scheduler_config.json` so that
    # `StableDiffusionPipeline.from_pretrained(snapshot_root)` works in one
    # call, but we explicitly don't keep those files — the parity-gate work
    # only needs the component dirs (tokenizer/, text_encoder/, unet/, vae/).
    # Constructing the pipeline by hand keeps the snapshot minimal and
    # mirrors what `python/dump_unet_ref.py` already does for the per-step
    # reference dumps.
    tokenizer = CLIPTokenizer.from_pretrained(snapshot / "tokenizer")
    # The snapshot's tokenizer config doesn't pin `model_max_length`, so it
    # defaults to a 64-bit int sentinel that overflows the tokenizers crate
    # under newer transformers. CLIP-L's actual max is 77.
    tokenizer.model_max_length = 77
    text_encoder = CLIPTextModel.from_pretrained(
        snapshot / "text_encoder", torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        snapshot / "unet", torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(snapshot / "vae", torch_dtype=dtype)
    # SD 1.5 DDIM defaults — must match `DdimConfig::sd_1_5()` on the Rust
    # side or the two timing rows aren't comparing the same denoising
    # trajectory. `set_alpha_to_one=False` + `steps_offset=1` are SD 1.5
    # quirks.
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    # `safety_checker=None` + `feature_extractor=None` skip the CLIP-NSFW
    # classifier — diffusers loads it by default even though we never use it.
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # ------- per-step timing via callback -------
    step_times_per_run: list[list[float]] = []
    total_times: list[float] = []

    def make_callback(record: list[float]):
        if device.type == "cuda":
            def cb(_pipe, step: int, _timestep, kwargs):
                # diffusers v0.30+ callback signature: callback_on_step_end.
                # The step is invoked AFTER the unet+sched step but BEFORE the
                # next iteration. Synchronize so the measurement reflects GPU
                # work, not just host-side launch.
                torch.cuda.synchronize()
                record.append(time.perf_counter())
                return kwargs
        else:
            def cb(_pipe, step: int, _timestep, kwargs):
                record.append(time.perf_counter())
                return kwargs
        return cb

    def run_once() -> tuple[float, list[float]]:
        marks: list[float] = []
        if device.type == "cuda":
            torch.cuda.synchronize()
        # Reset the generator so each run is identical.
        gen = torch.Generator(device=device).manual_seed(args.seed)
        t0 = time.perf_counter()
        marks.append(t0)
        with torch.inference_mode():
            _img = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt or None,
                width=args.size,
                height=args.size,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=gen,
                callback_on_step_end=make_callback(marks),
            ).images[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        # `marks` holds [t0, after_step_0, after_step_1, ..., after_step_{N-1}].
        deltas = [marks[i + 1] - marks[i] for i in range(len(marks) - 1)]
        return (t1 - t0) * 1000.0, [d * 1000.0 for d in deltas]

    for w in range(args.warmup):
        wall, _ = run_once()
        print(f"  warmup {w}: {wall:7.1f} ms total")

    for r in range(args.runs):
        wall, steps = run_once()
        total_times.append(wall)
        step_times_per_run.append(steps)
        if steps:
            print(f"  run {r}:    {wall:7.1f} ms total  median-step {median(steps):6.2f} ms")
        else:
            print(f"  run {r}:    {wall:7.1f} ms total  (no per-step samples)")

    print()
    print("=== summary ===")
    print(f"total wall-clock   : median {median(total_times):7.1f} ms over {args.runs} runs")
    if step_times_per_run:
        all_steps = [s for run in step_times_per_run for s in run]
        if all_steps:
            print(f"per-step latency   : median {median(all_steps):6.2f} ms  (n={len(all_steps)})")
            print(f"  step min/max     : {min(all_steps):6.2f} / {max(all_steps):6.2f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
