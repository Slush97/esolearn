"""
Debug comparison: generate 440Hz sine, run through whisper-tiny encoder+decoder,
print diagnostics to compare with Rust implementation.
"""

import numpy as np
import torch
import whisper

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. Generate 2-second 440Hz sine wave, pad to 30s
SAMPLE_RATE = 16000
duration = 2.0
t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False, dtype=np.float32)
sine_wave = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)

audio = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)
audio[:len(sine_wave)] = sine_wave

print(f"Audio: {len(audio)} samples, min={audio.min():.6f}, max={audio.max():.6f}")
print(f"First 10 audio samples: {audio[:10]}")
print()

# 2. Compute log-mel spectrogram
mel = whisper.log_mel_spectrogram(torch.from_numpy(audio)).unsqueeze(0).to(device)
print(f"Mel shape: {mel.shape}")
print(f"Mel stats: min={mel.min().item():.6f}, max={mel.max().item():.6f}, "
      f"mean={mel.mean().item():.6f}, std={mel.std().item():.6f}")
print(f"Mel [0,0,:10]: {mel[0, 0, :10].cpu().tolist()}")
print(f"Mel [0,0,-5:]: {mel[0, 0, -5:].cpu().tolist()}")
print()

# 3. Load model and run encoder
model = whisper.load_model("tiny", device=device)
model.eval()

with torch.no_grad():
    encoder_output = model.encoder(mel)

eo = encoder_output.cpu()
print(f"Encoder output shape: {encoder_output.shape}")
print(f"Encoder stats: min={eo.min().item():.6f}, max={eo.max().item():.6f}, "
      f"mean={eo.mean().item():.6f}, std={eo.std().item():.6f}")
print(f"Encoder first 5 values (flat): {eo[0, 0, :5].tolist()}")
print(f"Encoder [0,0,:20]: {eo[0, 0, :20].tolist()}")
print()

# Compare with Rust values
rust_first5 = [0.019999716, -0.19482848, 0.5101801, -1.0496588, 0.7465381]
py_first5 = eo[0, 0, :5].tolist()
print("=== ENCODER COMPARISON ===")
print(f"  Rust first 5:   {rust_first5}")
print(f"  Python first 5: {py_first5}")
diffs = [abs(r - p) for r, p in zip(rust_first5, py_first5)]
print(f"  Abs diffs:      {diffs}")
print(f"  Max diff:       {max(diffs):.8f}")

rust_stats = {"min": -12.7288, "max": 9.8038, "mean": -0.033249, "std": 1.0867}
py_stats = {"min": eo.min().item(), "max": eo.max().item(), "mean": eo.mean().item(), "std": eo.std().item()}
print(f"\n  Rust stats: {rust_stats}")
print(f"  Py   stats: { {k: round(v, 6) for k, v in py_stats.items()} }")
print()

# 4. First decode step
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)

sot_sequence = list(tokenizer.sot_sequence)
print(f"Tokenizer SOT: {tokenizer.sot}")
print(f"SOT sequence: {sot_sequence}")

notimestamps = tokenizer.no_timestamps
sot_sequence_full = sot_sequence + [notimestamps]
print(f"Full prompt (SOT + lang + task + notimestamps): {sot_sequence_full}")
print()

tokens = torch.tensor([sot_sequence_full], dtype=torch.long, device=device)

with torch.no_grad():
    logits = model.decoder(tokens, encoder_output)

last_logits = logits[0, -1, :].cpu()

print(f"Logits shape: {logits.shape}")
print(f"Last-position logits stats: min={last_logits.min().item():.6f}, max={last_logits.max().item():.6f}, "
      f"mean={last_logits.mean().item():.6f}, std={last_logits.std().item():.6f}")

# Top-10 tokens
top10 = torch.topk(last_logits, 10)
print("\n=== TOP-10 TOKENS (first decode step) ===")
for i in range(10):
    tid = top10.indices[i].item()
    score = top10.values[i].item()
    try:
        text = tokenizer.decode([tid])
    except:
        text = "<special>"
    print(f"  #{i+1}: token={tid:6d}  score={score:+10.4f}  text={repr(text)}")

# Timestamp token diagnostics
print("\n=== TIMESTAMP TOKEN LOGITS ===")
ts_start = tokenizer.timestamp_begin
print(f"Timestamp token range starts at: {ts_start}")
ts_logits = last_logits[ts_start:ts_start+10]
print(f"First 10 timestamp logits: {ts_logits.tolist()}")
print(f"Max timestamp logit: {last_logits[ts_start:].max().item():.4f} at offset {last_logits[ts_start:].argmax().item()}")

eot = tokenizer.eot
print(f"\nEOT token ({eot}) logit: {last_logits[eot].item():.4f}")

for word in ["the", "a", "hello", " the", " a"]:
    enc = tokenizer.encode(word)
    if enc:
        tid = enc[0]
        print(f"Token for {repr(word):10s} = {tid:6d}, logit = {last_logits[tid].item():.4f}")

print("\n=== SOFTMAX PROBABILITIES (top 10) ===")
probs = torch.softmax(last_logits, dim=-1)
top10p = torch.topk(probs, 10)
for i in range(10):
    tid = top10p.indices[i].item()
    prob = top10p.values[i].item()
    try:
        text = tokenizer.decode([tid])
    except:
        text = "<special>"
    print(f"  #{i+1}: token={tid:6d}  prob={prob:.6f}  text={repr(text)}")

# Also print logits at each position for decoder debugging
print("\n=== LOGITS AT EACH DECODER POSITION ===")
for pos in range(logits.shape[1]):
    l = logits[0, pos, :].cpu()
    top3 = torch.topk(l, 3)
    print(f"  pos {pos}: min={l.min().item():.4f} max={l.max().item():.4f} mean={l.mean().item():.4f} | top3: ", end="")
    for j in range(3):
        tid = top3.indices[j].item()
        print(f"{tid}({top3.values[j].item():.3f}) ", end="")
    print()
