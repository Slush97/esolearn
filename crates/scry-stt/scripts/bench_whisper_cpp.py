#!/usr/bin/env python3
"""Benchmark whisper.cpp on a WAV file and output JSON timing to stdout.

Usage:
    python3 scripts/bench_whisper_cpp.py <wav_path> [threads]

Requires whisper.cpp built at /tmp/whisper.cpp/build/bin/whisper-cli
with model at /tmp/whisper.cpp/models/ggml-tiny.bin
"""

import json
import os
import re
import subprocess
import sys

WHISPER_CLI = "/tmp/whisper.cpp/build/bin/whisper-cli"
MODELS_DIR = "/tmp/whisper.cpp/models"
LIB_PATH = "/tmp/whisper.cpp/build/src:/tmp/whisper.cpp/build/ggml/src"


def run_once(wav_path: str, threads: int, model_path: str) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        [
            WHISPER_CLI,
            "-m", model_path,
            "-f", wav_path,
            "-t", str(threads),
            "-bs", "1",
            "-bo", "1",
            "-nt",
            "-nf",
            "-l", "en",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    text = result.stdout.strip()
    stderr = result.stderr

    def parse_ms(label: str) -> float:
        # Match "label =   123.45 ms" patterns
        m = re.search(rf"{label}\s*=\s*([\d.]+)\s*ms", stderr)
        return float(m.group(1)) if m else 0.0

    def parse_ms_per_run(label: str) -> float:
        # Match "label =   123.45 ms /   N runs" — total time
        m = re.search(rf"{label}\s*=\s*([\d.]+)\s*ms\s*/", stderr)
        return float(m.group(1)) if m else 0.0

    load_ms = parse_ms("load time")
    mel_ms = parse_ms("mel time")
    encode_ms = parse_ms_per_run("encode time")
    decode_ms = parse_ms_per_run("decode time")
    batchd_ms = parse_ms_per_run("batchd time")
    total_ms = parse_ms("total time")

    # Subtract model load — whisper.cpp "total time" is wall clock including load.
    # For fair comparison with Rust (model pre-loaded), report inference-only time.
    inference_ms = total_ms - load_ms if total_ms > load_ms else total_ms

    return {
        "text": text,
        "load_ms": load_ms,
        "mel_ms": mel_ms,
        "encode_ms": encode_ms,
        "decode_ms": round(decode_ms + batchd_ms, 2),
        "total_inference_ms": round(inference_ms, 2),
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: bench_whisper_cpp.py <wav_path> [threads]"}))
        sys.exit(1)

    wav_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "tiny"
    # Default to all cores (matching Rust rayon default) for fair comparison
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()

    model_path = f"{MODELS_DIR}/ggml-{model_name}.bin"

    if not os.path.exists(WHISPER_CLI):
        print(json.dumps({"error": f"whisper-cli not found at {WHISPER_CLI}"}))
        sys.exit(1)
    if not os.path.exists(model_path):
        print(json.dumps({"error": f"ggml-{model_name}.bin not found at {model_path}. Run: bash /tmp/whisper.cpp/models/download-ggml-model.sh {model_name}"}))
        sys.exit(1)

    # Cold run
    cold = run_once(wav_path, threads, model_path)

    # Warm runs (3x) — each is a separate process so model reloads each time.
    # Subtract load_ms for fair comparison (Rust model is pre-loaded).
    warm_runs = []
    for _ in range(3):
        w = run_once(wav_path, threads, model_path)
        warm_runs.append(w["total_inference_ms"])

    warm_avg = sum(warm_runs) / len(warm_runs)

    output = {
        "text": cold["text"],
        "mel_ms": cold["mel_ms"],
        "encode_ms": cold["encode_ms"],
        "decode_ms": cold["decode_ms"],
        "total_inference_ms": cold["total_inference_ms"],
        "model_load_ms": cold["load_ms"],
        "warm_avg_ms": round(warm_avg, 2),
        "warm_runs_ms": [round(t, 2) for t in warm_runs],
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
