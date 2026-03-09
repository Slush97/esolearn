#!/usr/bin/env python3
"""Trace Whisper-tiny encoder intermediate values for a 2s 440Hz sine wave."""

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import whisper
import math

torch.set_grad_enabled(False)

# Load model weights
f = safe_open('/home/esoc/scry/crates/scry-stt/models/whisper-tiny/model.safetensors', framework='pt')
state = {k: f.get_tensor(k) for k in f.keys()}

# Generate audio: 2s 440Hz sine, amplitude 0.5
sr = 16000
t = np.arange(2 * sr) / sr
audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
audio_padded = whisper.pad_or_trim(audio)

# 1. Mel spectrogram
mel = whisper.log_mel_spectrogram(audio_padded)  # [80, 3000]
print("=" * 70)
print("1. MEL SPECTROGRAM")
print(f"   shape: {list(mel.shape)}")
print(f"   row 0 first 10: {mel[0, :10].tolist()}")
print(f"   min={mel.min().item():.6f}, max={mel.max().item():.6f}")

# Prepare input
x = mel.unsqueeze(0).float()  # [1, 80, 3000]

# 2. Conv1 (before GELU)
conv1_w = state['model.encoder.conv1.weight']
conv1_b = state['model.encoder.conv1.bias']
x = F.conv1d(x, conv1_w, conv1_b, padding=1)
print("\n" + "=" * 70)
print("2. AFTER CONV1 (before GELU)")
print(f"   shape: {list(x.shape)}")
print(f"   channel 0 first 5: {x[0, 0, :5].tolist()}")

# 3. Conv1 + GELU
x = F.gelu(x)
print("\n" + "=" * 70)
print("3. AFTER CONV1 + GELU")
print(f"   channel 0 first 5: {x[0, 0, :5].tolist()}")

# 4. Conv2 + GELU
conv2_w = state['model.encoder.conv2.weight']
conv2_b = state['model.encoder.conv2.bias']
x = F.conv1d(x, conv2_w, conv2_b, stride=2, padding=1)
x = F.gelu(x)
print("\n" + "=" * 70)
print("4. AFTER CONV2 + GELU")
print(f"   shape: {list(x.shape)}")
print(f"   channel 0 first 5: {x[0, 0, :5].tolist()}")

# 5. Transpose to [seq, d_model]
x = x.permute(0, 2, 1)  # [1, 1500, 384]
print("\n" + "=" * 70)
print("5. AFTER TRANSPOSE to [batch, seq, d_model]")
print(f"   shape: {list(x.shape)}")
print(f"   first 5 values (x[0,0,:5]): {x[0, 0, :5].tolist()}")

# 6. Positional embedding
pos_emb = state['model.encoder.embed_positions.weight']  # [1500, 384]
print("\n" + "=" * 70)
print("6. POSITIONAL EMBEDDING")
print(f"   shape: {list(pos_emb.shape)}")
print(f"   first 5 values (pos[0,:5]): {pos_emb[0, :5].tolist()}")

# 7. After adding positional embedding
x = x + pos_emb
print("\n" + "=" * 70)
print("7. AFTER ADDING POSITIONAL EMBEDDING")
print(f"   first 5 values (x[0,0,:5]): {x[0, 0, :5].tolist()}")

# 8. Encoder block 0
prefix = 'model.encoder.layers.0'
n_heads = 6
d_model = 384
head_dim = d_model // n_heads  # 64
scale = 1.0 / math.sqrt(head_dim)

# Self-attention layer norm
ln1_w = state[f'{prefix}.self_attn_layer_norm.weight']
ln1_b = state[f'{prefix}.self_attn_layer_norm.bias']
residual = x
x_ln = F.layer_norm(x, [d_model], ln1_w, ln1_b)
print("\n" + "=" * 70)
print("8. ENCODER BLOCK 0 -- internals")
print(f"   After self_attn_layer_norm first 5: {x_ln[0, 0, :5].tolist()}")

# Q, K, V projections
Q = F.linear(x_ln, state[f'{prefix}.self_attn.q_proj.weight'], state[f'{prefix}.self_attn.q_proj.bias'])
K = F.linear(x_ln, state[f'{prefix}.self_attn.k_proj.weight'])
V = F.linear(x_ln, state[f'{prefix}.self_attn.v_proj.weight'], state[f'{prefix}.self_attn.v_proj.bias'])

print(f"   Q first 5: {Q[0, 0, :5].tolist()}")
print(f"   K first 5: {K[0, 0, :5].tolist()}")
print(f"   V first 5: {V[0, 0, :5].tolist()}")

# Reshape for multi-head attention
bsz, seq_len = 1, x.shape[1]
Q = Q.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
K = K.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
V = V.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)

# Scaled dot-product attention
attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
attn_weights = F.softmax(attn_weights, dim=-1)
print(f"   Attn weights [head=0, q=0, :5]: {attn_weights[0, 0, 0, :5].tolist()}")

attn_out = torch.matmul(attn_weights, V)
attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
print(f"   Attn output first 5: {attn_out[0, 0, :5].tolist()}")

# Output projection
attn_out = F.linear(attn_out, state[f'{prefix}.self_attn.out_proj.weight'], state[f'{prefix}.self_attn.out_proj.bias'])
print(f"   After out_proj first 5: {attn_out[0, 0, :5].tolist()}")

# Residual
x = residual + attn_out
print(f"   After attn residual first 5: {x[0, 0, :5].tolist()}")

# MLP
ln2_w = state[f'{prefix}.final_layer_norm.weight']
ln2_b = state[f'{prefix}.final_layer_norm.bias']
residual = x
x_ln2 = F.layer_norm(x, [d_model], ln2_w, ln2_b)
print(f"   After final_layer_norm first 5: {x_ln2[0, 0, :5].tolist()}")

h = F.linear(x_ln2, state[f'{prefix}.fc1.weight'], state[f'{prefix}.fc1.bias'])
print(f"   After fc1 first 5: {h[0, 0, :5].tolist()}")
h = F.gelu(h)
print(f"   After fc1 + GELU first 5: {h[0, 0, :5].tolist()}")
h = F.linear(h, state[f'{prefix}.fc2.weight'], state[f'{prefix}.fc2.bias'])
print(f"   After fc2 first 5: {h[0, 0, :5].tolist()}")

x = residual + h
print(f"\n   ENCODER BLOCK 0 OUTPUT first 5: {x[0, 0, :5].tolist()}")

# 9. Full encoder output (all layers + final layer norm)
x_full = mel.unsqueeze(0).float()
x_full = F.conv1d(x_full, conv1_w, conv1_b, padding=1)
x_full = F.gelu(x_full)
x_full = F.conv1d(x_full, conv2_w, conv2_b, stride=2, padding=1)
x_full = F.gelu(x_full)
x_full = x_full.permute(0, 2, 1)
x_full = x_full + pos_emb

n_layers = 0
while f'model.encoder.layers.{n_layers}.self_attn.q_proj.weight' in state:
    n_layers += 1

print(f"\n" + "=" * 70)
print(f"9. FINAL ENCODER OUTPUT (through {n_layers} layers + layer_norm)")

for layer_idx in range(n_layers):
    p = f'model.encoder.layers.{layer_idx}'

    # Self-attention
    res = x_full
    xn = F.layer_norm(x_full, [d_model],
                       state[f'{p}.self_attn_layer_norm.weight'],
                       state[f'{p}.self_attn_layer_norm.bias'])

    Q = F.linear(xn, state[f'{p}.self_attn.q_proj.weight'], state[f'{p}.self_attn.q_proj.bias'])
    K = F.linear(xn, state[f'{p}.self_attn.k_proj.weight'])
    V = F.linear(xn, state[f'{p}.self_attn.v_proj.weight'], state[f'{p}.self_attn.v_proj.bias'])

    sl = x_full.shape[1]
    Q = Q.view(1, sl, n_heads, head_dim).transpose(1, 2)
    K = K.view(1, sl, n_heads, head_dim).transpose(1, 2)
    V = V.view(1, sl, n_heads, head_dim).transpose(1, 2)

    aw = torch.matmul(Q, K.transpose(-2, -1)) * scale
    aw = F.softmax(aw, dim=-1)
    ao = torch.matmul(aw, V)
    ao = ao.transpose(1, 2).contiguous().view(1, sl, d_model)
    ao = F.linear(ao, state[f'{p}.self_attn.out_proj.weight'], state[f'{p}.self_attn.out_proj.bias'])
    x_full = res + ao

    # MLP
    res = x_full
    xn = F.layer_norm(x_full, [d_model],
                       state[f'{p}.final_layer_norm.weight'],
                       state[f'{p}.final_layer_norm.bias'])
    h = F.gelu(F.linear(xn, state[f'{p}.fc1.weight'], state[f'{p}.fc1.bias']))
    h = F.linear(h, state[f'{p}.fc2.weight'], state[f'{p}.fc2.bias'])
    x_full = res + h

    print(f"   After layer {layer_idx} first 5: {x_full[0, 0, :5].tolist()}")

# Final layer norm
x_full = F.layer_norm(x_full, [d_model],
                       state['model.encoder.layer_norm.weight'],
                       state['model.encoder.layer_norm.bias'])
print(f"\n   FINAL OUTPUT (after layer_norm) first 5: {x_full[0, 0, :5].tolist()}")
print(f"   shape: {list(x_full.shape)}")
print(f"   min={x_full.min().item():.6f}, max={x_full.max().item():.6f}")
print("=" * 70)
