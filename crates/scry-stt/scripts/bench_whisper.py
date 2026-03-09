#!/usr/bin/env python3
"""Whisper-tiny benchmark companion for scry-stt bench_vs_python.

Reads a 16kHz mono WAV file, transcribes with openai-whisper, and prints
JSON timing results to stdout.

Usage:
    python3 scripts/bench_whisper.py <path_to_wav>

Dependencies:
    pip install openai-whisper
"""

import json
import sys
import time

import numpy as np
import whisper


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: bench_whisper.py <wav_path> [model_name]"}))
        sys.exit(1)

    wav_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "tiny"

    # ── Load model ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model = whisper.load_model(model_name, device="cpu")
    model_load_ms = (time.perf_counter() - t0) * 1000.0

    # ── Load audio once (not timed — same as Rust which has audio in memory) ──
    audio = whisper.load_audio(wav_path)
    audio = whisper.pad_or_trim(audio)

    # ── Mel spectrogram ───────────────────────────────────────────────────
    t2 = time.perf_counter()
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    mel_ms = (time.perf_counter() - t2) * 1000.0

    # ── Decode ────────────────────────────────────────────────────────────
    t3 = time.perf_counter()
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    result = whisper.decode(model, mel, options)
    decode_ms = (time.perf_counter() - t3) * 1000.0

    total_inference_ms = mel_ms + decode_ms

    # ── Warm runs (reuse in-memory audio, same as Rust) ───────────────────
    warm_times = []
    for _ in range(3):
        tw = time.perf_counter()
        mel_w = whisper.log_mel_spectrogram(audio).to(model.device)
        _ = whisper.decode(model, mel_w, options)
        warm_times.append((time.perf_counter() - tw) * 1000.0)

    output = {
        "text": result.text.strip(),
        "model_load_ms": round(model_load_ms, 2),
        "mel_ms": round(mel_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_inference_ms": round(total_inference_ms, 2),
        "warm_avg_ms": round(sum(warm_times) / len(warm_times), 2),
        "warm_runs_ms": [round(t, 2) for t in warm_times],
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
