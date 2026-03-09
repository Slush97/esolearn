#!/usr/bin/env python3
"""Memory footprint + cold-start comparison: PyTorch vs scry-llm.

Measures:
  - Process RSS before/after model load (cold start memory)
  - GPU VRAM allocated after model load
  - GPU VRAM peak during inference
  - Time to first token (cold start latency)
  - Decode throughput (tok/s)

Usage:
    uv run --with torch --with transformers --with accelerate --with psutil \
        tests/bench_memory_compare.py
"""

import gc
import json
import os
import subprocess
import sys
import time

import psutil
import torch
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "fixtures" / "llama-3.2-1b"
PROMPT = "The capital of France is"
N_TOKENS = 30
WARMUP = 2


def get_rss_mb():
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def gpu_mem_mb():
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def gpu_mem_peak_mb():
    """Peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def bench_pytorch():
    """Benchmark PyTorch/HuggingFace Transformers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    results = {}
    model_path = str(MODEL_PATH) if MODEL_PATH.exists() else "NousResearch/Llama-3.2-1B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Measure cold-start RSS
    gc.collect()
    rss_before = get_rss_mb()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # --- Model load (cold start) ---
    t_load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t_load = time.perf_counter() - t_load_start

    rss_after = get_rss_mb()
    vram_after_load = gpu_mem_mb()

    results["load_time_s"] = round(t_load, 2)
    results["rss_before_mb"] = round(rss_before, 1)
    results["rss_after_mb"] = round(rss_after, 1)
    results["rss_delta_mb"] = round(rss_after - rss_before, 1)
    results["vram_after_load_mb"] = round(vram_after_load, 1)

    # --- Time to first token ---
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t_ttft_start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    t_ttft = time.perf_counter() - t_ttft_start

    vram_peak_inference = gpu_mem_peak_mb()
    results["ttft_s"] = round(t_ttft, 3)
    results["vram_peak_inference_mb"] = round(vram_peak_inference, 1)

    # --- Decode throughput (with warmup) ---
    for _ in range(WARMUP):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=N_TOKENS, do_sample=False, use_cache=True)
        if device == "cuda":
            torch.cuda.synchronize()

    if device == "cuda":
        torch.cuda.synchronize()
    t_gen_start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=N_TOKENS, do_sample=False, use_cache=True)
    if device == "cuda":
        torch.cuda.synchronize()
    t_gen = time.perf_counter() - t_gen_start

    generated_tokens = output.shape[1] - prompt_len
    results["decode_tok_s"] = round(generated_tokens / t_gen, 1)
    results["decode_latency_ms"] = round(t_gen / generated_tokens * 1000, 2)

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    results["output"] = text

    # Process-level peak RSS
    results["rss_peak_mb"] = round(get_rss_mb(), 1)

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    results["n_params"] = n_params
    results["param_bytes_mb"] = round(n_params * 2 / 1e6, 1)  # bf16 = 2 bytes

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9

    print("=" * 70)
    print("  Cold Start + Memory Footprint Comparison")
    print(f"  Model: Llama 3.2 1B  |  GPU: {device_name}")
    if torch.cuda.is_available():
        print(f"  VRAM: {vram_total:.1f} GB")
    print("=" * 70)

    # --- PyTorch ---
    print("\n--- PyTorch (HuggingFace Transformers, BF16) ---")
    pt = bench_pytorch()

    print(f"  Model load:          {pt['load_time_s']:.2f}s")
    print(f"  RSS before load:     {pt['rss_before_mb']:.0f} MB")
    print(f"  RSS after load:      {pt['rss_after_mb']:.0f} MB")
    print(f"  RSS delta:           {pt['rss_delta_mb']:.0f} MB")
    print(f"  RSS peak:            {pt['rss_peak_mb']:.0f} MB")
    print(f"  VRAM after load:     {pt['vram_after_load_mb']:.0f} MB")
    print(f"  VRAM peak inference: {pt['vram_peak_inference_mb']:.0f} MB")
    print(f"  Time to 1st token:   {pt['ttft_s']:.3f}s")
    print(f"  Decode throughput:   {pt['decode_tok_s']:.1f} tok/s")
    print(f"  Decode latency:      {pt['decode_latency_ms']:.2f} ms/tok")
    print(f"  Params:              {pt['n_params']:,} ({pt['param_bytes_mb']:.0f} MB bf16)")
    print(f"  Output:              {pt['output']!r}")

    # Save raw data
    out = {
        "gpu": device_name,
        "model": "Llama-3.2-1B",
        "pytorch_bf16": pt,
    }

    out_path = Path(__file__).parent.parent / "bench_memory_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nRaw data saved to: {out_path}")


if __name__ == "__main__":
    main()
