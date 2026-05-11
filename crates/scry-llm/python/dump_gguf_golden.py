#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Golden-tensor dump for scry-llm GGUF M14 validation.

For a given GGUF file, dequantizes a chosen set of tensors via the
canonical `gguf` Python package (the same code llama.cpp produces and
consumes), slices each to a small head, and writes raw f32 bytes plus a
JSON manifest into `tests/fixtures/gguf_golden/`. The Rust integration
test `tests/gguf_golden.rs` reads the same GGUF + manifest and verifies
our dequant produces bit-identical output.

Each tensor in the manifest has:
  name        — the GGUF tensor name (e.g. "blk.0.attn_k.weight")
  source_dtype — what gguf-py reports for the on-disk dtype
  full_shape  — the tensor's real shape
  head_count  — how many leading elements we kept in the golden bin
  bin         — relative path to the .f32 binary file

Run via the shared venv:

    crates/scry-vision/.venv/bin/python \\
        crates/scry-llm/python/dump_gguf_golden.py \\
        --gguf <path-to-gguf> \\
        --out-dir crates/scry-llm/tests/fixtures/gguf_golden \\
        --tensors output_norm.weight,per_layer_proj_norm.weight,blk.0.attn_k.weight \\
        --head 256

The committed Gemma-4-E4B fixture in the repo was produced by:

    python dump_gguf_golden.py \\
        --gguf $LMSTUDIO/.../Gemma-4-E4B-...-Q4_K_M.gguf \\
        --out-dir crates/scry-llm/tests/fixtures/gguf_golden \\
        --tensors output_norm.weight,per_layer_proj_norm.weight,blk.0.attn_k.weight \\
        --head 256
"""

import argparse
import json
import sys
from pathlib import Path

import gguf
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--tensors",
        type=str,
        required=True,
        help="comma-separated tensor names to dump",
    )
    ap.add_argument(
        "--head",
        type=int,
        default=256,
        help="number of leading elements to keep per tensor (0 = whole tensor)",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    reader = gguf.GGUFReader(args.gguf)
    by_name = {t.name: t for t in reader.tensors}

    manifest = {
        "source_gguf_basename": args.gguf.name,
        "tensors": [],
    }
    for name in args.tensors.split(","):
        name = name.strip()
        if name not in by_name:
            print(f"error: tensor {name!r} not in {args.gguf}", file=sys.stderr)
            return 1
        t = by_name[name]
        # gguf-py exposes `.data` as a numpy view in the tensor's source dtype
        # for unquantized tensors, or as raw bytes for quantized. We always
        # dequantize via gguf.quants to f32 so the golden is a single shape.
        full = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        # Flatten then take the leading window.
        flat = full.reshape(-1)
        n_full = int(flat.size)
        n_head = n_full if args.head == 0 else min(args.head, n_full)
        head = flat[:n_head].copy()
        bin_name = name.replace("/", "_").replace(".", "_") + ".f32"
        (args.out_dir / bin_name).write_bytes(head.tobytes(order="C"))
        # Sanity for the manifest — numpy may have stored as little-endian
        # already; double-check.
        assert head.dtype == np.float32
        assert head.flags["C_CONTIGUOUS"]

        manifest["tensors"].append(
            {
                "name": name,
                "source_dtype": t.tensor_type.name,
                "full_shape": [int(d) for d in t.shape],
                "n_full": n_full,
                "head_count": n_head,
                "bin": bin_name,
            }
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path} ({len(manifest['tensors'])} tensors)")
    for e in manifest["tensors"]:
        print(
            f"  {e['name']:<48} {e['source_dtype']:<6}"
            f" full={e['n_full']:>10} head={e['head_count']} → {e['bin']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
