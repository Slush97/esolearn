#!/usr/bin/env python3
"""HuggingFace baseline benchmark for Llama 3.2 1B on same GPU.

Measures prefill and decode tok/s in both f32 and bf16 for apples-to-apples
comparison with scry-llm on the same hardware.

Usage:
    uv run --with torch --with transformers --with accelerate \
        tests/bench_hf_baseline.py

    # Or with pip:
    python tests/bench_hf_baseline.py
"""

import time
import torch
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "fixtures" / "llama-3.2-1b"
PROMPT = "The capital of France is"
N_TOKENS = 30
WARMUP = 2


def bench_generate(model, tokenizer, prompt, n_tokens, label):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                do_sample=False,
                use_cache=True,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed run
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=False,
            use_cache=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    total_tokens = output.shape[1]
    generated_tokens = total_tokens - prompt_len

    # Decode for sanity check
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Prompt:          {prompt!r}")
    print(f"  Prompt tokens:   {prompt_len}")
    print(f"  Generated:       {generated_tokens} tokens")
    print(f"  Total time:      {elapsed:.3f} s")
    print(f"  Overall tok/s:   {generated_tokens / elapsed:.1f}")
    print(f"  Output:          {text!r}")
    print(f"{'=' * 60}")

    return generated_tokens / elapsed


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = str(MODEL_PATH) if MODEL_PATH.exists() else "NousResearch/Llama-3.2-1B"
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = {}

    # --- BF16 ---
    print("\nLoading model in bf16...")
    model_bf16 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    results["bf16"] = bench_generate(model_bf16, tokenizer, PROMPT, N_TOKENS, "HF Transformers — BF16")
    del model_bf16
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- F32 ---
    print("\nLoading model in f32...")
    model_f32 = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device,
    )
    results["f32"] = bench_generate(model_f32, tokenizer, PROMPT, N_TOKENS, "HF Transformers — F32")
    del model_f32

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  Summary (decode tok/s, greedy, {N_TOKENS} tokens)")
    print(f"{'=' * 60}")
    for dtype, tps in results.items():
        print(f"  {dtype:>6s}: {tps:>8.1f} tok/s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
