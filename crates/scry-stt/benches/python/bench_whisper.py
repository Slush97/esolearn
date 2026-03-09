#!/usr/bin/env python3
"""
Cold-start benchmark for openai-whisper (PyTorch) — whisper-tiny.

Measures the same pipeline as the Rust bench_cold_start example:
  import → model load → audio gen → transcribe

Run:
  cd crates/scry-stt/benches/python
  python3 -m venv .venv && source .venv/bin/activate
  pip install openai-whisper numpy
  python3 bench_whisper.py
"""

import time
import sys
import resource
import os

def rss_mb():
    """Current RSS in MB (Linux/macOS)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux, bytes on macOS
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        openai-whisper Cold-Start Benchmark (whisper-tiny)   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    rss_baseline = rss_mb()

    # ── Stage 1: Import whisper ──────────────────────────────────────────
    t0 = time.perf_counter()
    import whisper
    import numpy as np
    import torch
    import_ms = (time.perf_counter() - t0) * 1000

    # ── Stage 2: Load model ──────────────────────────────────────────────
    t1 = time.perf_counter()
    model = whisper.load_model("tiny", device="cpu")
    model_load_ms = (time.perf_counter() - t1) * 1000
    rss_after_model = rss_mb()

    # ── Stage 3: Generate 2s 440Hz sine wave ─────────────────────────────
    t2 = time.perf_counter()
    sr = 16000
    duration = 2.0
    t_audio = np.arange(0, int(sr * duration)) / sr
    audio = (np.sin(2 * np.pi * 440 * t_audio) * 0.5).astype(np.float32)
    audio_gen_ms = (time.perf_counter() - t2) * 1000

    # ── Stage 4: Transcribe (mel + encode + decode) ──────────────────────
    t3 = time.perf_counter()
    result = model.transcribe(
        audio,
        language="en",
        fp16=False,           # CPU mode
        without_timestamps=True,
    )
    inference_ms = (time.perf_counter() - t3) * 1000
    rss_after_inference = rss_mb()

    # ── Totals ───────────────────────────────────────────────────────────
    total_load_ms = import_ms + model_load_ms
    total_cold_start_ms = total_load_ms + audio_gen_ms + inference_ms

    # ── Results table ────────────────────────────────────────────────────
    print("┌─────────────────────────┬──────────────┐")
    print("│ Stage                   │ Time (ms)    │")
    print("├─────────────────────────┼──────────────┤")
    print(f"│ Import whisper          │ {import_ms:>12.2f} │")
    print(f"│ Model load (tiny)       │ {model_load_ms:>12.2f} │")
    print(f"│ Audio generation (2s)   │ {audio_gen_ms:>12.2f} │")
    print(f"│ Transcribe (full)       │ {inference_ms:>12.2f} │")
    print("├─────────────────────────┼──────────────┤")
    print(f"│ Total load              │ {total_load_ms:>12.2f} │")
    print(f"│ Total inference         │ {inference_ms:>12.2f} │")
    print(f"│ Total cold-start        │ {total_cold_start_ms:>12.2f} │")
    print("└─────────────────────────┴──────────────┘")
    print()

    print("┌─────────────────────────┬──────────────┐")
    print("│ Memory                  │ RSS (MB)     │")
    print("├─────────────────────────┼──────────────┤")
    print(f"│ Baseline                │ {rss_baseline:>12.1f} │")
    print(f"│ After model load        │ {rss_after_model:>12.1f} │")
    print(f"│ After inference         │ {rss_after_inference:>12.1f} │")
    print(f"│ Delta (model)           │ {rss_after_model - rss_baseline:>12.1f} │")
    print(f"│ Delta (total)           │ {rss_after_inference - rss_baseline:>12.1f} │")
    print("└─────────────────────────┴──────────────┘")
    print()

    text = result.get("text", "").strip()
    print(f"Text: '{text}'")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: cpu")
    print()

    # ── Second inference (warm) ──────────────────────────────────────────
    print("── Second inference (warm) ─────────────────────────────────")
    t_warm = time.perf_counter()
    _ = model.transcribe(audio, language="en", fp16=False, without_timestamps=True)
    warm_ms = (time.perf_counter() - t_warm) * 1000
    print(f"  Warm inference: {warm_ms:.2f} ms")

if __name__ == "__main__":
    main()
