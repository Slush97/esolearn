#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""PyTorch + cuDNN reference numbers for scry-vision's `bench_gpu`.

Times torchvision's ResNet-18 and ResNet-50 forward passes on CUDA with
random `[1, 3, 224, 224]` input. Mirrors the Rust bench in spirit:
random weights (we only care about kernel dispatch timing, not
classification accuracy), median over runs after warmup.

Run via the venv set up in this directory:

    crates/scry-vision/.venv/bin/python crates/scry-vision/bench_pytorch.py

cuDNN's autotune is enabled (`cudnn.benchmark = True`) so the first
forward pass selects the fastest kernel for the given shape — this is
the standard PyTorch inference baseline.
"""

import argparse
import time

import torch
import torchvision.models as models


DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def time_ms(warmup: int, runs: int, fn) -> float:
    # Warmup also lets cudnn.benchmark pick its kernel and the CUDA caching
    # allocator settle.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def bench_model(name: str, ctor, dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    # `weights=None` keeps the default (random) init — matches our zero/
    # default-init Rust path at the kernel level. Numerics don't matter
    # for forward-pass timing; kernel dispatch does.
    model = ctor(weights=None).to(device=device, dtype=dtype).eval()
    x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

    with torch.inference_mode():
        ms = time_ms(5, 20, lambda: model(x))
    print(f"--- {name} (1000-class, 1×3×224×224) ---")
    print(f"  PyTorch + cuDNN : {ms:7.2f} ms/image")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precision",
        choices=list(DTYPES),
        default="fp32",
        help="Precision for weights, activations, and convolutions (default: fp32).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — aborting.")

    # cuDNN benchmark mode picks the fastest kernel per shape during warmup.
    torch.backends.cudnn.benchmark = True

    dtype = DTYPES[args.precision]

    print("=== PyTorch ResNet bench ===")
    print(f"PyTorch  : {torch.__version__}")
    print(f"cuDNN    : {torch.backends.cudnn.version()}")
    print(f"Device   : {torch.cuda.get_device_name(0)}")
    print(f"Mode     : {args.precision}, eager, inference_mode, cudnn.benchmark=True")
    print()

    bench_model("ResNet-18", models.resnet18, dtype)
    bench_model("ResNet-50", models.resnet50, dtype)


if __name__ == "__main__":
    main()
