# Learning Guide: scry-stt (Whisper Speech-to-Text)

This guide teaches you how your own code works — from raw audio to transcribed
text. Study each section, then open the referenced source file and read it
yourself. By the end you should be able to whiteboard the full pipeline.

---

## Prerequisites — Concepts You Need First

Before diving into the code, make sure you're comfortable with:

1. **Linear algebra basics**: matrix multiplication, transpose, dot product
2. **What a neural network layer does**: weight matrix × input + bias
3. **What "attention" means at a high level**: a mechanism that lets one
   sequence look at another sequence and decide what's relevant

You don't need deep ML expertise. The code is concrete — when in doubt,
read the CPU backend implementation, it's just loops and arithmetic.

---

## Part 1: The Tensor Engine (scry-llm)

**Start here.** Everything else builds on this.

### 1.1 What is a Tensor?

Open: `crates/scry-llm/src/tensor/mod.rs`

```rust
pub struct Tensor<B: DeviceBackend> {
    pub id: TensorId,       // unique ID (for future autograd)
    pub data: B::Storage,   // the actual numbers — Vec<f32> on CPU, GPU buffer on CUDA
    pub shape: Shape,       // dimensions, e.g. [80, 3000]
}
```

A tensor is just a flat array of f32 numbers + a shape that tells you how to
interpret them. A `[3, 4]` tensor is 12 floats laid out row-major:

```
row 0: [a b c d]
row 1: [e f g h]
row 2: [i j k l]
memory: [a b c d e f g h i j k l]
```

The `B` generic parameter is the backend — CPU, CUDA, or WGPU. The same
tensor code works on all of them because `B::Storage` is different for each:
- CPU: `Vec<f32>` (just a Rust vector)
- CUDA: `GpuStorage` (a buffer living on the GPU)

Open: `crates/scry-llm/src/tensor/shape.rs`

Shape is stack-allocated (max 6 dims). Key method: `strides()` computes how
many elements to skip per dimension. Shape `[2,3,4]` has strides `[12,4,1]` —
to move one step in dim 0 you skip 12 elements, in dim 1 you skip 4, etc.

### 1.2 The Backend Trait

Open: `crates/scry-llm/src/backend/mod.rs`

This is the core abstraction. `MathBackend` defines every math operation:

| Method | What it does |
|--------|-------------|
| `matmul(a, b, m, k, n)` | Matrix multiply: `[m,k] × [k,n] → [m,n]` |
| `matmul_bias(a, b, bias)` | Fused `a × b + bias` (one allocation instead of two) |
| `softmax(input, shape)` | exp(x) / sum(exp(x)) along last dimension |
| `layernorm(input, gamma, beta)` | Normalize to mean=0, var=1, then scale+shift |
| `gelu(input)` | Activation function (smooth ReLU variant) |
| `embedding(weight, indices)` | Look up rows by index (like a dictionary) |

**Why this design matters:** The Whisper model code is written once, generic
over `<B: MathBackend>`. At compile time, Rust monomorphizes it — you get a
CPU version and a CUDA version from the same source code, with zero runtime
dispatch overhead.

### 1.3 The CPU Backend — Read This Carefully

Open: `crates/scry-llm/src/backend/cpu.rs`

This is the reference implementation. Every operation is plain Rust loops.
When someone asks "how does softmax work in your code?", point here.

**Softmax** (~line 295):
```
For each row:
  1. Find max value (for numerical stability)
  2. Subtract max, compute exp() of each element
  3. Divide by sum of exps
```
Uses f64 intermediates to reduce floating-point error.

**LayerNorm** (~line 361):
```
For each row of size d:
  1. Compute mean = sum(x) / d
  2. Compute variance = sum((x - mean)²) / d
  3. Normalize: (x - mean) / sqrt(variance + eps)
  4. Scale and shift: normalized * gamma + beta
```

**GELU** (~line 473):
```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```
A smooth approximation of ReLU used by GPT-2 and Whisper.

**MatMul** (~line 876): Tiled 32×32 blocks for cache locality, accumulates
in f64, parallelized with rayon for matrices with >128 rows. For single-row
multiplies (m=1), uses a dedicated GEMV path that avoids BLAS dispatch
overhead (~3µs vs ~50-100µs).

### 1.4 The ops Layer and nn Modules

Open: `crates/scry-llm/src/ops.rs`

Thin wrappers that call the backend and wrap results in `Tensor`:
```rust
pub fn matmul<B>(a: &Tensor<B>, b: &Tensor<B>, ...) -> Tensor<B> {
    let data = B::matmul(&a.data, &b.data, m, k, n, ...);
    Tensor::new(data, Shape::new(&[m, n]))
}
```

Open: `crates/scry-llm/src/nn/linear.rs`

A `Linear` layer is just a weight matrix + bias:
```rust
pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
    ops::matmul_bias(input, &self.weight, &self.bias, ...)
}
```

Open: `crates/scry-llm/src/nn/layernorm.rs`

Same pattern — wraps `ops::layernorm_inference()`.

**Exercise:** Trace a single `Linear::forward()` call from nn → ops → cpu
backend. Follow the data through each function. This is the fundamental
building block of the entire model.

---

## Part 2: Audio Processing (mel.rs)

Open: `crates/scry-stt/src/mel.rs`

This converts raw audio into the format Whisper understands.

### 2.1 Why Mel Spectrograms?

Raw audio is a 1D waveform — amplitude over time. Neural networks work better
with a 2D frequency representation: "what frequencies are present at each
moment in time?" That's a spectrogram.

"Mel" means the frequency axis is warped to match human hearing — we're more
sensitive to differences between 200Hz and 400Hz than between 8000Hz and
8200Hz.

### 2.2 The Pipeline

```
Raw audio (480,000 samples = 30s @ 16kHz)
    ↓
Step 1: Reflect-pad edges (avoid boundary artifacts)
    ↓
Step 2: STFT (Short-Time Fourier Transform)
    For each overlapping 25ms window (every 10ms):
    ├─ Multiply by Hann window (smooth taper to avoid spectral leakage)
    ├─ FFT → complex frequency bins
    └─ Take |X[f]|² → power spectrum
    ↓
Step 3: Apply mel filterbank (80 triangular filters)
    Weighted sum of power spectrum → 80 energy values per frame
    ↓
Step 4: Log scale + normalize
    ↓
Output: [80 mel bands, 3000 frames]
```

**Key constants** (line 8-14):
- `N_FFT = 400` → 25ms window at 16kHz
- `HOP_LENGTH = 160` → 10ms between frames
- `N_MELS = 80` → 80 frequency bands
- `CHUNK_SAMPLES = 480,000` → 30 seconds max

**Parallelism:** Both the STFT (per-frame) and mel filterbank (per-band)
stages run in parallel via rayon. Each thread gets its own FFT planner.

**Exercise:** Run the `transcribe_wav` example with the debug prints enabled.
Compare the mel stats output against these expected ranges:
- min ≈ -0.54, max ≈ 1.46, mean ≈ -0.30

---

## Part 3: The Whisper Model

### 3.1 Architecture Overview

Whisper is an encoder-decoder transformer:

```
                    ENCODER                          DECODER

Audio [80, 3000]                          Token IDs (one at a time)
      ↓                                         ↓
  Conv1D ×2                               Token + Position Embedding
      ↓                                         ↓
  + Positional Emb                    ┌─→ Causal Self-Attention (sees past only)
      ↓                               │         ↓
  Transformer Blocks ×N               │   Cross-Attention ──────── reads encoder output
      ↓                               │         ↓
  LayerNorm                           │   MLP (feed-forward)
      ↓                               │         ↓
  [1500, d_model]  ──────────────────►│   × N decoder blocks
                                      │         ↓
                                      │   Logits → argmax → next token
                                      └── feed back ──┘
```

The encoder runs once per audio clip. The decoder runs once per output token,
attending to the encoder output through cross-attention.

### 3.2 Model Configuration

Open: `crates/scry-stt/src/model/config.rs`

| Model | d_model | Layers | Heads | d_head | Params |
|-------|---------|--------|-------|--------|--------|
| tiny | 384 | 4+4 | 6 | 64 | 39M |
| base | 512 | 6+6 | 8 | 64 | 74M |
| small | 768 | 12+12 | 12 | 64 | 244M |
| medium | 1024 | 24+24 | 16 | 64 | 769M |

d_head = d_model / n_heads. Always 64 for standard models.

### 3.3 Encoder

Open: `crates/scry-stt/src/model/encoder.rs`

**forward()** (~line 104):

```
Input: mel [80, 3000]
  ↓
Conv1D(80 → d_model, kernel=3, stride=1, pad=1) + GELU
  → [d_model, 3000]     (project frequency bands to model dimension)
  ↓
Conv1D(d_model → d_model, kernel=3, stride=2, pad=1) + GELU
  → [d_model, 1500]     (stride=2 halves the time dimension)
  ↓
Transpose [d_model, 1500] → [1500, d_model] + add positional embeddings
  (fused in one loop for efficiency)
  ↓
N encoder blocks, each:
  ├─ LayerNorm → Self-Attention → + residual
  └─ LayerNorm → MLP (d_model → 4×d_model → d_model) → + residual
  ↓
Final LayerNorm
  → [1500, d_model]
```

The Conv1D layers (see `conv1d.rs`) use im2col: reshape the convolution into a
matrix multiply. This is the standard trick — convolution = unroll + matmul.

**Encoder self-attention is bidirectional** — every position can attend to every
other position. No masking.

### 3.4 Decoder

Open: `crates/scry-stt/src/model/decoder.rs`

**forward_step()** (~line 220): Processes one token at a time during inference.

```
token_id + position
  ↓
token_embedding[token_id] + positional_embedding[position]
  → [1, d_model]
  ↓
N decoder blocks, each:
  ├─ LayerNorm → Causal Self-Attention (with KV cache) → + residual
  ├─ LayerNorm → Cross-Attention (queries encoder) → + residual
  └─ LayerNorm → MLP → + residual
  ↓
Final LayerNorm
  ↓
Logit projection: [1, d_model] × [d_model, vocab_size] → [1, 51865]
  ↓
argmax → predicted token ID
```

**Causal self-attention** means each position can only attend to itself and
earlier positions. This is enforced by the KV cache — at step t, the cache
contains keys/values for positions 0..t, so the query at position t naturally
only sees the past.

### 3.5 Attention — The Core Mechanism

Open: `crates/scry-stt/src/model/attention.rs`

Attention computes: `Attention(Q, K, V) = softmax(Q × K^T / √d_head) × V`

In words:
1. **Q** (query) = "what am I looking for?"
2. **K** (key) = "what do I contain?"
3. **V** (value) = "what information do I carry?"
4. **Q × K^T** = similarity scores between query and all keys
5. **softmax** = normalize scores to sum to 1 (attention weights)
6. **× V** = weighted sum of values

**Multi-head attention** splits Q, K, V into `n_heads` independent groups of
size `d_head`. Each head can attend to different things. Results are
concatenated and projected back to `d_model`.

**Three types of attention in Whisper:**

| Type | Q from | K,V from | Masking | Where |
|------|--------|----------|---------|-------|
| Encoder self-attn | encoder | encoder | None (bidirectional) | encoder.rs |
| Decoder self-attn | decoder | decoder (cached) | Causal (past only) | decoder.rs |
| Cross-attention | decoder | encoder (cached) | None | attention.rs |

### 3.6 KV Cache — Why It Matters

Without cache: to generate token 50, you'd recompute attention over all 50
previous tokens. That's O(n²) total work for n tokens.

With cache: store K and V from previous steps. At step t, only compute Q for
the new token, look up cached K,V for positions 0..t-1. Each step is O(n)
instead of O(n²).

Open: `crates/scry-stt/src/model/decoder.rs` (~line 83)

```rust
pub struct DecoderLayerKv<B> {
    pub k: Vec<f32>,     // [n_heads, max_seq, d_head] pre-allocated
    pub v: Vec<f32>,     // same layout
    pub seq_len: usize,  // how many positions are filled
}
```

Layout is head-major: all of head 0's cached keys are contiguous, then head
1's, etc. This means reading "all keys for head h up to position t" is a
single contiguous memory read.

### 3.7 Cross-Attention Optimization — Pre-Transposed K

Open: `crates/scry-stt/src/model/attention.rs` (~line 92)

The key optimization: encoder K heads are transposed once after encoding.

Normal attention: `scores = Q × K^T` requires transposing K every decode step.
Our code: transpose K once, store `K_transposed`. Then `scores = Q × K_t`
with no transpose needed.

```
Standard:  Q[1, d_head] × K[audio_len, d_head]^T  (needs transpose per step)
Ours:      Q[1, d_head] × K_t[d_head, audio_len]   (pre-transposed, hits fast GEMV path)
```

This is a ~6× speedup on each decode step because it enables the contiguous-
memory GEMV path instead of strided row-wise dot products.

**Exercise:** In `attention.rs`, find `compute_kv_cache()`. Identify where the
transpose happens and where the result is stored. Then find `forward()` and
see how `k_heads_t` is used with `trans_b=false`.

---

## Part 4: Decoding — Turning Model Output Into Text

### 4.1 The Decode Loop

Open: `crates/scry-stt/src/decode.rs`

```
Phase 1: Force-feed prompt tokens
  [SOT, <|en|>, <|transcribe|>, <|notimestamps|>]
  These condition the model: "transcribe English without timestamps"
  Optimization: skip logit projection for all but the last prompt token

Phase 2: Autoregressive generation
  Loop:
    1. Run decoder on current token → logits [1, 51865]
    2. Suppress special tokens (set to -infinity)
    3. argmax → next token ID
    4. If EOT (end of text) → stop
    5. Otherwise → feed token back as next input
```

**Token suppression** (~line 71): Whisper's vocabulary includes special tokens
(language tags, timestamps, start-of-transcript). During generation, these
are suppressed so only real text tokens and EOT can be selected.

### 4.2 The Tokenizer

Open: `crates/scry-stt/src/tokenizer/mod.rs`

Whisper uses byte-level BPE (Byte Pair Encoding). Each token ID maps to a
sequence of bytes. To decode:

```rust
pub fn decode(&self, token_ids: &[usize]) -> String {
    let mut bytes = Vec::new();
    for &id in token_ids {
        if self.is_special(id) { continue; }
        bytes.extend_from_slice(&self.id_to_bytes[id]);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
```

Just concatenate the bytes for each non-special token and interpret as UTF-8.

Two tokenizer formats are supported:
- `vocab.bin` — binary format with correct Whisper token IDs (preferred)
- `tokenizer.json` — HuggingFace JSON format (fallback)

---

## Part 5: Checkpoint Loading

Open: `crates/scry-stt/src/checkpoint/mod.rs`

### 5.1 Safetensors Format

HuggingFace's safetensors is a simple format: a JSON header describing tensor
names, shapes, and byte offsets, followed by raw tensor data. It supports
memory-mapping — the OS maps the file into virtual memory, and tensors are
read directly from disk without copying into a separate buffer.

```rust
let mmap = unsafe { memmap2::Mmap::map(&file) }?;
let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
```

### 5.2 Weight Layout Conversion

HuggingFace stores linear weights as `[out_features, in_features]`.
Our code expects `[in_features, out_features]`. So every linear weight is
transposed during loading:

```rust
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
```

### 5.3 Fused QKV Loading

HuggingFace has separate `q_proj`, `k_proj`, `v_proj` weights. We fuse them
into a single `[d_model, 3*d_model]` weight matrix during loading. This means
one matmul instead of three during inference.

### 5.4 Dtype Handling

Models may be stored in F16, BF16, or F32. The loader auto-detects and
converts to F32 for inference:

```rust
match tensor.dtype() {
    Dtype::F16  => f16_bytes_to_f32(data),
    Dtype::BF16 => bf16_bytes_to_f32(data),
    Dtype::F32  => f32_bytes_to_f32(data),
}
```

---

## Part 6: The Full Pipeline — Trace It End to End

Open the `transcribe_wav` example and trace the entire flow:

```
1. load_wav_pcm_16k()        → Vec<f32> raw samples
2. pad_or_trim_audio()       → [480000] samples
3. log_mel_spectrogram()     → MelSpectrogram { data: [80×3000], n_mels, n_frames }
4. Tensor::from_vec(mel)     → Tensor [80, 3000]
5. model.encode(&mel)        → Tensor [1500, d_model]  (encoder output)
6. greedy_decode(&model, &encoder_output, &config)
   a. compute_cross_kv_caches()  → pre-computed K,V for cross-attention
   b. force-feed prompt [SOT, en, transcribe, notimestamps]
   c. loop: decode_step → logits → argmax → token
   d. stop at EOT
   → Vec<usize> token IDs
7. tokenizer.decode(&tokens)  → String
```

**Exercise:** Add timing to each step. Which is slowest? (Hint: encoding is
heavy but runs once. Decoding runs per-token but each step is lighter.)

---

## Part 7: Interview Prep — Questions You Should Be Able to Answer

### Architecture

- "Walk me through how audio becomes text in your system."
  → Parts 2-4 of this guide. Hit: mel spectrogram, encoder, cross-attention,
    autoregressive decode, tokenizer.

- "What's the difference between encoder and decoder attention?"
  → Encoder: bidirectional, sees all positions. Decoder self-attention: causal,
    sees only past. Cross-attention: decoder queries encoder output.

- "Why is attention O(n²)?"
  → Q×K^T produces an [n,n] matrix of scores. With KV cache, each step is O(n)
    but total over n steps is still O(n²).

### Optimizations

- "Why pre-transpose the encoder K-heads?"
  → Avoids transposing K on every decode step. Enables contiguous GEMV path
    (~3µs vs ~19µs). One-time cost ~0.1ms, saves 6× per decode step.

- "Why fuse Q, K, V into one weight matrix?"
  → One matmul instead of three. Reduces dispatch overhead and memory traffic.

- "Why skip logit projection for prompt tokens?"
  → Logit projection is [d_model] × [d_model, vocab_size] — the largest matmul
    in decoding. Prompt tokens have known IDs, so computing logits is wasted work.
    Saves 1-2ms per prompt token.

### Design Decisions

- "Why compile-time backend polymorphism instead of runtime dispatch?"
  → Zero overhead. `Tensor<CpuBackend>` and `Tensor<CudaBackend>` are
    monomorphized at compile time. No vtable, no dynamic dispatch, no
    boxing. The optimizer can inline everything.

- "Why safetensors instead of GGUF?"
  → HuggingFace ecosystem compatibility. Memory-mapped, zero-copy loading.
    Auto-detects F16/BF16/F32. No custom format to maintain.

- "Why accumulate in f64 for matmul?"
  → Float32 has ~7 digits of precision. When summing thousands of products
    (e.g., 384-dim dot products), accumulated error becomes significant.
    f64 has ~15 digits, so the final f32 result is more accurate.

### Knowing Your Limits

- "Does your system support beam search?" → No, greedy only (with temperature).
- "Word-level timestamps?" → Not yet.
- "Language detection?" → Hardcoded English.
- "Streaming?" → Processes up to 30s chunks, not continuous streaming.

Being honest about what's missing shows you actually understand the scope.

---

## Study Plan

**Week 1:** Parts 1-2 (tensor engine + mel spectrogram)
- Read cpu.rs top to bottom
- Run transcribe_wav with debug prints, understand each number

**Week 2:** Parts 3-4 (model architecture + decoding)
- Trace one full encode call through encoder.rs
- Trace one decode step through decoder.rs + attention.rs
- Draw the data flow on paper with tensor shapes at each step

**Week 3:** Parts 5-6 (checkpoint loading + end-to-end)
- Read checkpoint/mod.rs, understand the HF → our layout conversion
- Modify the example to print intermediate tensors at each layer
- Time each pipeline stage

**Week 4:** Part 7 (interview prep)
- Practice explaining each section out loud without looking at code
- Have someone ask you the questions from Part 7
- If you can't answer one, go back to the relevant source file
