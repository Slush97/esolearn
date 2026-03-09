#!/usr/bin/env python3
"""Convert OpenAI Whisper .pt checkpoint to HuggingFace safetensors format.

Usage:
    python3 scripts/convert_openai_to_hf.py [model_name]

    model_name: tiny (default), base, small, medium, large-v3
    Downloads the model if not cached, converts to safetensors.
"""

import sys
import torch
from safetensors.torch import save_file, load_file
from collections import OrderedDict
from pathlib import Path

# Model configs: (n_encoder_layers, n_decoder_layers)
CONFIGS = {
    "tiny":     (4,  4),
    "base":     (6,  6),
    "small":    (12, 12),
    "medium":   (24, 24),
    "large-v3": (32, 32),
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "tiny"
if model_name not in CONFIGS:
    print(f"Unknown model: {model_name}. Choose from: {', '.join(CONFIGS)}")
    sys.exit(1)

n_enc_layers, n_dec_layers = CONFIGS[model_name]

SRC = Path.home() / f".cache/whisper/{model_name}.pt"
MODELS_DIR = Path(__file__).resolve().parent.parent / f"models/whisper-{model_name}"
DST = MODELS_DIR / "model.safetensors"

# Download if not cached
if not SRC.exists():
    print(f"Downloading OpenAI whisper {model_name}...")
    import whisper
    whisper.load_model(model_name, device="cpu")
    assert SRC.exists(), f"Expected {SRC} after download"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

checkpoint = torch.load(SRC, map_location="cpu", weights_only=False)
state = checkpoint.get("model_state_dict", checkpoint)
print(f"Source keys: {len(state)}")

mapped = OrderedDict()

def add(src_key, dst_key):
    if src_key not in state:
        print(f"  WARNING: missing {src_key}")
        return
    mapped[dst_key] = state[src_key].float().contiguous()

# --- Encoder ---
add("encoder.conv1.weight", "model.encoder.conv1.weight")
add("encoder.conv1.bias", "model.encoder.conv1.bias")
add("encoder.conv2.weight", "model.encoder.conv2.weight")
add("encoder.conv2.bias", "model.encoder.conv2.bias")
add("encoder.positional_embedding", "model.encoder.embed_positions.weight")
add("encoder.ln_post.weight", "model.encoder.layer_norm.weight")
add("encoder.ln_post.bias", "model.encoder.layer_norm.bias")

for i in range(n_enc_layers):
    sb = f"encoder.blocks.{i}"
    db = f"model.encoder.layers.{i}"
    add(f"{sb}.attn.query.weight", f"{db}.self_attn.q_proj.weight")
    add(f"{sb}.attn.query.bias", f"{db}.self_attn.q_proj.bias")
    add(f"{sb}.attn.key.weight", f"{db}.self_attn.k_proj.weight")
    add(f"{sb}.attn.value.weight", f"{db}.self_attn.v_proj.weight")
    add(f"{sb}.attn.value.bias", f"{db}.self_attn.v_proj.bias")
    add(f"{sb}.attn.out.weight", f"{db}.self_attn.out_proj.weight")
    add(f"{sb}.attn.out.bias", f"{db}.self_attn.out_proj.bias")
    add(f"{sb}.attn_ln.weight", f"{db}.self_attn_layer_norm.weight")
    add(f"{sb}.attn_ln.bias", f"{db}.self_attn_layer_norm.bias")
    add(f"{sb}.mlp.0.weight", f"{db}.fc1.weight")
    add(f"{sb}.mlp.0.bias", f"{db}.fc1.bias")
    add(f"{sb}.mlp.2.weight", f"{db}.fc2.weight")
    add(f"{sb}.mlp.2.bias", f"{db}.fc2.bias")
    add(f"{sb}.mlp_ln.weight", f"{db}.final_layer_norm.weight")
    add(f"{sb}.mlp_ln.bias", f"{db}.final_layer_norm.bias")

# --- Decoder ---
add("decoder.token_embedding.weight", "model.decoder.embed_tokens.weight")
add("decoder.positional_embedding", "model.decoder.embed_positions.weight")
add("decoder.ln.weight", "model.decoder.layer_norm.weight")
add("decoder.ln.bias", "model.decoder.layer_norm.bias")

for i in range(n_dec_layers):
    sb = f"decoder.blocks.{i}"
    db = f"model.decoder.layers.{i}"
    add(f"{sb}.attn.query.weight", f"{db}.self_attn.q_proj.weight")
    add(f"{sb}.attn.query.bias", f"{db}.self_attn.q_proj.bias")
    add(f"{sb}.attn.key.weight", f"{db}.self_attn.k_proj.weight")
    add(f"{sb}.attn.value.weight", f"{db}.self_attn.v_proj.weight")
    add(f"{sb}.attn.value.bias", f"{db}.self_attn.v_proj.bias")
    add(f"{sb}.attn.out.weight", f"{db}.self_attn.out_proj.weight")
    add(f"{sb}.attn.out.bias", f"{db}.self_attn.out_proj.bias")
    add(f"{sb}.attn_ln.weight", f"{db}.self_attn_layer_norm.weight")
    add(f"{sb}.attn_ln.bias", f"{db}.self_attn_layer_norm.bias")
    add(f"{sb}.cross_attn.query.weight", f"{db}.encoder_attn.q_proj.weight")
    add(f"{sb}.cross_attn.query.bias", f"{db}.encoder_attn.q_proj.bias")
    add(f"{sb}.cross_attn.key.weight", f"{db}.encoder_attn.k_proj.weight")
    add(f"{sb}.cross_attn.value.weight", f"{db}.encoder_attn.v_proj.weight")
    add(f"{sb}.cross_attn.value.bias", f"{db}.encoder_attn.v_proj.bias")
    add(f"{sb}.cross_attn.out.weight", f"{db}.encoder_attn.out_proj.weight")
    add(f"{sb}.cross_attn.out.bias", f"{db}.encoder_attn.out_proj.bias")
    add(f"{sb}.cross_attn_ln.weight", f"{db}.encoder_attn_layer_norm.weight")
    add(f"{sb}.cross_attn_ln.bias", f"{db}.encoder_attn_layer_norm.bias")
    add(f"{sb}.mlp.0.weight", f"{db}.fc1.weight")
    add(f"{sb}.mlp.0.bias", f"{db}.fc1.bias")
    add(f"{sb}.mlp.2.weight", f"{db}.fc2.weight")
    add(f"{sb}.mlp.2.bias", f"{db}.fc2.bias")
    add(f"{sb}.mlp_ln.weight", f"{db}.final_layer_norm.weight")
    add(f"{sb}.mlp_ln.bias", f"{db}.final_layer_norm.bias")

print(f"\nTotal tensors converted: {len(mapped)}")

save_file(mapped, DST)
print(f"Saved to {DST}")

# --- Verify ---
print("\n--- Verification ---")
loaded = load_file(str(DST))
print(f"Loaded {len(loaded)} tensors from safetensors")

orig = state["encoder.conv1.weight"].float().flatten()[:3]
conv = loaded["model.encoder.conv1.weight"].flatten()[:3]
print(f"encoder.conv1.weight first 3 (original): {orig.tolist()}")
print(f"model.encoder.conv1.weight first 3 (saved): {conv.tolist()}")
assert torch.allclose(orig, conv), "MISMATCH!"

orig_d = state["decoder.token_embedding.weight"].float().flatten()[:3]
conv_d = loaded["model.decoder.embed_tokens.weight"].flatten()[:3]
print(f"decoder.token_embedding first 3 (original): {orig_d.tolist()}")
print(f"model.decoder.embed_tokens first 3 (saved): {conv_d.tolist()}")
assert torch.allclose(orig_d, conv_d), "MISMATCH!"

for k, v in loaded.items():
    assert v.dtype == torch.float32, f"{k} is {v.dtype}, expected float32"
print(f"\nAll tensors verified as float32. Conversion of '{model_name}' complete.")
