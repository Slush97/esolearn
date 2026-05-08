# GPU Breakdown — Where Does Time Go?

Findings from `cargo bench -p scry-llm --features scry-gpu --bench gpu_breakdown` on `gpu-resident/scry-vision`.

The bench times each stage of `C = GELU(A @ B)` separately at three sizes: upload, matmul, GELU, download. Reports median + p95 per stage.

> **Update 2026-05-08:** cuBLAS + CUDA-compiled transpose/GELU are now plumbed
> into the persistent matmul path (`gpu_matmul_persistent` no longer
> early-returns on the cuBLAS branch). The numbers below reflect the
> Vulkan/WGSL chain that motivated the work; the post-cuBLAS section at the
> bottom captures the new state on CUDA.

## Reproduction

```bash
cargo bench -p scry-llm --features scry-gpu --bench gpu_breakdown
```

Hardware in this run: RTX 5070 Ti (f32 peak ≈ 55 TFLOPS).

## Headline

**Below ~1024³, dispatch+fence overhead dominates compute. The matmul kernel is not the bottleneck at small/medium sizes.**

| size | matmul TFLOPS | % of 55 TF peak |
|---|---:|---:|
| small  (128, 256, 128) | 0.18 | 0.3% |
| medium (256, 512, 256) | 1.26 | 2.3% |
| large  (1024, 1024, 1024) | 10.69 | 19.4% |

TFLOPS climb ~60× across the sweep. The kernel itself only approaches the prior scry-gpu memo's quoted ~28–30% peak at the large size — at smaller sizes a fixed per-dispatch cost (record + queue submit + fence wait) eats the budget.

## Stage data

Medians (p95 in parens). All times in µs. "Work" is element count for transfers/elementwise, FLOPs for matmul.

```
stage                              small (128/256/128)        medium (256/512/256)        large (1024/1024/1024)
                                   median       p95           median       p95            median        p95
upload B (cpu→gpu)                  29.75      41.89           45.31      57.69            196.29      258.94
matmul (persistent)                 46.60      54.35           53.13      65.51            200.87      213.39
gelu (persistent)                   27.59      41.40           26.63      39.18             37.23       47.61
download C (gpu→cpu)                29.57      36.36           33.34      48.42            917.97     1306.73
chained total                      138.71     182.75          157.58     209.17           1372.57     1928.30
stage-sum vs chained                +5.20                       -0.83                       +20.21
```

Stage-sum tracks chained total within ~20 µs across all sizes — the per-stage isolation is sound.

## Interpretation

**Three converging signals say "overhead-dominated" at small/medium:**

1. **Compute floor math.** A matmul running at 28% peak (15.4 TFLOPS) would do 33.5M FMAs in 4.4 µs. We measured 53 µs at medium. The other ~48 µs is dispatch + fence wait.
2. **GELU corroborates.** 27 µs at small (16K elements) vs 37 µs at large (1M elements) — element-count climbs 64× but time barely moves. The work is sub-microsecond on a modern GPU; what we're measuring is almost entirely fixed per-call overhead.
3. **TFLOPS scales nearly linearly with size up to 1024³.** Classic signature of fixed cost amortizing into more compute.

**At large (1024³) we're transfer-bound, not compute-bound.**

- Download alone: 917 µs (67% of the 1372 µs chain).
- 1M f32s = 4 MB. PCIe 4.0 ×16 ideal = ~125 µs. We're at ~7× that.
- Upload of the same 4 MB takes 196 µs (~1.5× ideal). Asymmetric — possibly a staging-buffer copy + fence on the download side.

**The matmul kernel itself looks roughly competitive at large sizes** — 19.4% of f32 peak is below the prior memo's 28–30% but in the same ballpark. Larger workloads (2048³, 4096³) likely close the rest of that gap.

## Implications for next investment

In priority order:

1. **Port batched command buffers to `ScryGpuBackend`.** scry-gpu already exposes `Device::batch()`; scry-learn's Phase 4 already uses it (memo `project_gpu_persistent_tensors.md`). Recording matmul + GELU into one batch with a single fence wait should eliminate one full per-op overhead. **Validated by `cargo bench -p scry-llm --features scry-gpu --bench gpu_batched_poc`** — see "POC validation" below.

2. **Avoid downloads when possible.** The largest single cost at the large size is `to_vec`. The M1 work already keeps tensors on-GPU between matmul and GELU; the same discipline needs to apply across whole forward passes. M3/M4 (ResNet, ViT GPU residency) are what this unlocks.

3. **Kernel tiling/vec4 loads** is third priority. Until #1 is shipped, faster compute just makes the dispatch overhead a larger fraction of total time. Once batching lands, kernel work becomes the bottleneck at large sizes and is worth optimizing further.

## POC validation

`cargo bench -p scry-llm --features scry-gpu --bench gpu_batched_poc` records `matmul + barrier + gelu` into a single `Device::batch()` (one fence wait) and compares to the current path (two `run_configured` calls, two fence waits). Same workload sizes as the breakdown bench.

| size | unbatched median | batched median | saved | saved % |
|---|---:|---:|---:|---:|
| small  (128, 256, 128) | 55.49 µs | 39.46 µs | 16.03 µs | 28.9% |
| medium (256, 512, 256) | 69.43 µs | 53.61 µs | 15.82 µs | 22.8% |
| large  (1024, 1024, 1024) | 220.55 µs | 199.79 µs | 20.76 µs | 9.4% |

Findings vs. breakdown predictions:

- **Per-fence cost is ~17 µs, not ~40 µs.** The breakdown's compute-floor math overestimated. Fence wait partially overlaps GPU execution — when the GPU is still finishing the kernel, the wait doesn't cost a full roundtrip. The savings come from eliminating the *second* fence's submission + scheduling round-trip, not the wait itself.
- **Absolute savings are nearly constant across sizes** (~16–21 µs). Confirms it's a fixed per-op overhead being eliminated, exactly the predicted shape.
- **% savings tracks the regime**: 29% at small (overhead-dominated) → 9% at large (compute-dominated). Same pattern the breakdown showed for TFLOPS.

**Decision: port is justified.** At LLM-typical hidden dims (256–1024), 22–29% wall-clock savings per matmul+activation chain. A transformer layer has ~6–10 such chains; per-layer savings compound through long generations. Keep "expect 20–30% at common sizes" as the realistic target — the original "30–40%" estimate was optimistic.

The POC also serves as the API smoke test for the port: `ScryGpuBackend::matmul_then_gelu_batched_for_bench` is the prototype, and produces numerically equivalent output to the unbatched path (1e-3 relative tolerance).

## Competitive comparison (added after POC)

How does our path compare to alternatives on the **same hardware** (RTX 5070 Ti)? Three reference points:

- **scry-gpu cuBLAS** (in-Rust, via `scry-gpu/src/backend/cuda.rs`) — `bench_cuda_compare` example.
- **PyTorch 2.11 + CUDA 13** — standalone Python script (see below).
- **Our paths** — gpu_breakdown bench (WGSL chain) and gpu_batched_poc (WGSL batched).

### End-to-end `C = GELU(A @ B)` chain median

| size | scry-llm WGSL chain | scry-llm batched POC | PyTorch chain |
|---|---:|---:|---:|
| small  (128/256/128) | ~74 µs | 39.46 µs | **12.26 µs** |
| medium (256/512/256) | ~80 µs | 53.61 µs | **14.47 µs** |
| large  (1024³) | ~238 µs | 199.79 µs | **90.53 µs** |

Gap to PyTorch shrinks with size (6× → 5.5× → 2.6×) — the same overhead-vs-compute curve we observed internally.

### Matmul TFLOPS at 1024³

| backend | TFLOPS | % of 55 TF peak |
|---|---:|---:|
| scry-llm WGSL coarse 4×4 | 10.69 | 19% |
| scry-gpu cuBLAS (in-Rust) | 24.54 | 45% |
| PyTorch matmul | 24.76 | 45% |

PyTorch matmul ≡ scry-gpu cuBLAS within 1% — confirms PyTorch is just wrapping cuBLAS underneath. **The kernel-quality gap is exactly 2.3× and is independent of substrate, language, or batching.**

### GELU stays flat across sizes

| backend | small | medium | large |
|---|---:|---:|---:|
| scry-llm WGSL | 27.6 µs | 26.6 µs | 37.2 µs |
| PyTorch | 6.5 µs | 6.5 µs | 9.2 µs |

PyTorch's elementwise dispatch + execution is consistently 4–5× faster. The existing `bench_cuda_compare` shows 3–6× CUDA-vs-Vulkan throughput on memory-bound kernels at all sizes, so this is largely a substrate gap.

### Per-launch overhead estimates

PyTorch's matmul at 1024³ (compute-dominated): 86.7 µs ≈ scry-gpu cuBLAS 87.5 µs — they agree.
PyTorch's matmul at small (4M FMAs): 11.4 µs. Compute floor at 24 TFLOPS would be ~0.17 µs, so ~11 µs is launch overhead. Comparable to our ~17 µs Vulkan fence wait — within a factor of 2.

### Sources of the gap (large size, decomposed)

```
scry-llm WGSL chain:        ~238 µs
  scry-llm WGSL coarse 4×4 matmul:                     ~201 µs
    (cuBLAS would be):                                  ~87 µs   (-114 µs)
  scry-llm WGSL gelu:                                   ~37 µs
    (PyTorch elementwise would be):                     ~9 µs    (-28 µs)

scry-llm batched POC:       ~200 µs   (-38 µs from one fence saved)
PyTorch chain:              ~90 µs    (mostly cuBLAS + tiny GELU + ~10 µs overlap)
```

If we had cuBLAS matmul + a competitive GELU shader, even *without* batching, our chain at 1024³ would land near 100–120 µs — close to PyTorch territory. Batching gets us another 10–20%.

### Path forward, ranked by leverage

1. **Plumb cuBLAS into the persistent matmul path on CUDA.** Memory entry already noted "cuBLAS keeps legacy materialize-and-download until CUDA transpose/GELU shaders are wired" — that's the gating work. Closes ~80% of the gap on NVIDIA hardware. **Highest single uplift available.**
2. **Better WGSL elementwise kernels (GELU first, then softmax/layernorm when added).** Closes the 4–5× elementwise gap that hurts both Vulkan users and the GELU portion of our chain. Less work than fixing matmul because the kernels are simpler (vec4 loads, larger workgroups).
3. **Batched dispatch port** (validated by gpu_batched_poc, 22–29% wins). Biggest leverage on Vulkan platforms; marginal on CUDA where launch overhead is already low.

### Reproducing PyTorch numbers

Saved as `/tmp/torch-bench/bench.py` during this session; equivalent script:

```python
import time, torch
assert torch.cuda.is_available()
dev = torch.device("cuda")
SHAPES = [(128,256,128), (256,512,256), (1024,1024,1024)]
for m,k,n in SHAPES:
    a = torch.randn(m,k,device=dev,dtype=torch.float32)
    b = torch.randn(k,n,device=dev,dtype=torch.float32)
    for _ in range(10):  # warmup
        c = a @ b; g = torch.nn.functional.gelu(c); torch.cuda.synchronize()
    times = []
    for _ in range(100):
        torch.cuda.synchronize(); t = time.perf_counter()
        c = a @ b; g = torch.nn.functional.gelu(c)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    times.sort()
    print(f"{m}x{k}x{n}: median {times[50]*1e6:.2f} µs")
```

Setup: `uv venv && source .venv/bin/activate && uv pip install torch && python bench.py`.

### Reproducing scry-gpu cuBLAS numbers

```bash
CUDARC_CUDA_VERSION=13010 cargo run -p scry-gpu \
  --example bench_cuda_compare --features "cuda vulkan" --release
```

(Env var is needed because `cudarc 0.19` is pinned to `cuda-13010` features but local CUDA is 13.2.)

## Out of scope here

- **Splitting per-op time into "kernel compute" vs "fence wait" vs "command-buffer record"** would need GPU timestamp queries (`vkCmdWriteTimestamp`, `vkCmdResetQueryPool`). That's a scry-gpu-side instrumentation effort, not a scry-llm bench. The compute-floor argument above pins down compute time well enough to justify the next investment without it.
- **Comparing to candle-core** — still not wired. Self-vs-self proves architecture; self-vs-industry proves competitiveness. Worth doing once batching lands and we have a number worth defending.
- **Sweeping buffer-pool / threshold heuristics** (`GPU_MIN_ELEMENTS = 65_536`, `GPU_ELEMENTWISE_MIN = 16_384`). These were picked on intuition; the breakdown shows the small workload (4M FMAs, just over the matmul threshold) gets a 0.18 TFLOPS effective rate — the threshold may be too aggressive. Empirical curves would be a follow-up bench.

## Post-cuBLAS results (2026-05-08)

After plumbing cuBLAS into `gpu_matmul_persistent` and compiling the
transpose / GELU CUDA kernels (`scry-gpu/src/shaders.rs::TRANSPOSE_CUDA`,
`GELU_CUDA`), re-running the breakdown bench with `--features scry-gpu-cuda`:

```bash
CUDARC_CUDA_VERSION=13010 cargo bench -p scry-llm \
    --features scry-gpu-cuda --bench gpu_breakdown
```

| stage (median µs)    | small | medium | large (1024³) |
|---|---:|---:|---:|
| upload B             |   5.97 |  19.06 |  146.02 |
| matmul (cuBLAS)      |  11.98 |  31.83 |  115.18 |
| gelu (CUDA)          |   6.48 |   6.53 |   13.29 |
| download C           |   9.81 |  27.29 |  981.13 |
| **chained total**    | **34.09** | **85.27** | **1255.41** |

| size | WGSL chain (before) | cuBLAS chain (now) | Speedup |
|---|---:|---:|---:|
| small  (128, 256, 128)    | 138.71 µs |  34.09 µs | **4.07×** |
| medium (256, 512, 256)    | 157.58 µs |  85.27 µs | **1.85×** |
| large  (1024³)            | 1372.57 µs | 1255.41 µs | 1.09× |

### Matmul-only TFLOPS — closing the kernel-quality gap

| backend | small | medium | large (1024³) |
|---|---:|---:|---:|
| WGSL coarse 4×4 (before) | 0.18 TF |  1.26 TF | 10.69 TF |
| cuBLAS persistent (now)  | 0.70 TF |  2.11 TF | 18.64 TF |
| cuBLAS standalone (`bench_cuda_compare`) | — | — | 22.99 TF |

cuBLAS standalone hits 22.99 TFLOPS at 1024³; the persistent path measures
18.64 TFLOPS on the same hardware. The ~22 µs delta per call is allocation
+ Rust-side timer overhead (each `Device::cublas_matmul` already
`stream.synchronize()`s internally). At 2048³ standalone hits 31.7 TFLOPS,
so the kernel itself is no longer the ceiling.

### Matmul + GELU only (vs PyTorch — fair comparison)

PyTorch's chain measurement excludes upload/download (matrices live on
GPU), so the apples-to-apples line is `matmul + gelu`:

| size | scry-llm WGSL | scry-llm cuBLAS | PyTorch | gap remaining |
|---|---:|---:|---:|---:|
| small  | 74.19 µs |  18.46 µs | 12.26 µs | 1.5× |
| medium | 79.76 µs |  38.36 µs | 14.47 µs | 2.7× |
| large  | 238.10 µs | 128.47 µs | 90.53 µs | 1.4× |

That's the "~80% of the gap" the path-forward analysis predicted, in one move.

### What now dominates

At small/medium: still per-launch overhead, but cuBLAS' launch is much
tighter than WGSL's — each stage drops by roughly a factor of 2–3.

At 1024³: **download is 78% of the chain** (981 µs of 1255 µs). Real
models don't `to_vec` after every operator, so this isn't the bottleneck
end-to-end — but it caps how much faster the breakdown bench can get.
Anything that fixes it (page-locked staging, async copy) is a separate
investment from the kernel work.

GELU at large dropped 37 → 13 µs (2.8×) — the CUDA elementwise substrate
gap is largely closed. PyTorch's 9 µs is still ~30% faster, presumably
from launch + tighter kernel; small further wins available with vec4
loads but not where to spend cycles next.

### Where the remaining gap to PyTorch lives

- **Sync overhead per dispatch.** Each cuBLAS / CUDA-kernel call inside
  scry-gpu calls `stream.synchronize()` to mirror Vulkan fence semantics.
  Two syncs per chain (matmul + gelu) ≈ ~20–30 µs. PyTorch only syncs
  when the user calls `torch.cuda.synchronize()`. Eliminating these
  internal syncs (or using `Device::batch()` on CUDA — the API already
  exists) is the next 20–30% at small sizes.
- **Allocation per call.** Both stages `alloc::<f32>(m*n)` fresh. A
  small workspace pool would erase a few µs per stage.

### Path forward, updated (CUDA-first as of 2026-05-08)

Strategic call: optimization investment goes into CUDA. PyTorch is also
CUDA-first, so that's where parity is reachable. WGSL kernels stay
correct and supported but aren't the focus. The Vulkan batched-dispatch
port (which gpu_batched_poc validated for 22–29% wins) is **off the
active backlog**.

1. **(Done)** Plumb cuBLAS + CUDA helpers into the persistent path.
2. **(Done, 2026-05-09)** Drop internal sync from cuBLAS / `dispatch_cuda`.
   See "Async dispatch results" below.
3. **(Done, 2026-05-09)** Skip wasted zero-init on output buffers via
   `Backend::alloc_uninit`. Cudarc's stream allocator already pools, so
   the actual win was eliminating the unnecessary `cuMemsetD8Async`
   dispatch per `dev.alloc::<f32>()` call. A full Drop-wrapper buffer pool
   (the original "workspace pool" plan) was deemed too invasive vs the
   marginal alloc-churn-on-top-of-pool win. See "alloc_uninit results"
   below.
4. **Tune `GPU_MIN_ELEMENTS`** on CUDA. Cutoff was set when small-matmul
   was 0.18 TFLOPS; cuBLAS is now 0.70. Threshold likely too conservative.
5. **M2 CUDA kernels** — softmax, layernorm, batchnorm, conv2d (hard).
   Each one unblocks a class of vision/transformer workloads.
6. **End-to-end model bench.** Operator wins don't count until they
   show up in a real forward pass. Pick BERT-base / ViT-small / ResNet-18
   and time vs PyTorch on the same GPU.

## Async dispatch results (2026-05-09)

`CudaBackend::dispatch_cuda` and `cublas_matmul` no longer sync internally;
new `*_async` variants are wired into `gpu_matmul_persistent`,
`gpu_transpose`, and `gpu_gelu_persistent`. The host sync moved to the
storage boundary — `CudaBuffer::read_back` (called from `Buffer::download`
and `ScryGpuBackend::to_vec`) issues a single `stream.synchronize()` once
the host actually needs the data. Re-running the breakdown bench:

```bash
CUDARC_CUDA_VERSION=13010 cargo bench -p scry-llm \
    --features scry-gpu-cuda --bench gpu_breakdown
```

| stage (median µs)  | small | medium | large (1024³) |
|---|---:|---:|---:|
| upload B           |   5.94 |  18.84 |  147.44 |
| matmul (queue)     |   7.32 |   6.95 |    9.92 |
| gelu (queue)       |   3.27 |   3.31 |    3.63 |
| download C         |  10.86 |  57.13 | 1074.21 |
| **chained total**  | **27.46** | **86.96** | **1243.68** |

| size  | sync chain (prev) | async chain (now) | delta |
|---|---:|---:|---:|
| small  (128, 256, 128) |   34.09 µs |  27.46 µs | **−6.6 µs (−19%)** |
| medium (256, 512, 256) |   85.27 µs |  86.96 µs |   ≈ noise          |
| large  (1024³)         | 1255.41 µs | 1243.68 µs |   ≈ noise          |

Small matches the doc's "10–15 µs at small" prediction in shape (a hair
under in magnitude). Medium and large are already download-bound, so
async vs sync doesn't move the chained total — the savings get absorbed
into download because the GPU is still finishing kernels when the host
asks for the result.

**Caveat on per-stage numbers under async dispatch.** The matmul/gelu
medians above only reflect launch+queue time (216 TFLOPS at 1024³ is not
a real number — it's the launch returning before the GPU is done). Only
chained totals are meaningful for direct comparison. Per-stage values
still help diagnose *where* dispatch overhead lives, but don't compare
them to the pre-async numbers stage-by-stage.

### What now dominates (post-async)

At small/medium: per-launch overhead, but cuBLAS' launch is tight enough
that the chain is approaching the irreducible floor. Further gains here
are mostly allocation pool (#3 above).

At 1024³: download is **86%** of the chain (1074 µs of 1244 µs). Real
forward passes don't materialize after every operator, so the bench's
download cost overstates what models actually pay. The real story for
end-to-end perf shows up in #6 (model-level bench).

### Validation

- All 51 scry-gpu lib tests + 13 cuda_compute integration tests pass.
- All 31 scry-llm lib tests pass, including
  `chained_matmuls_stay_on_gpu`, `matmul_through_gpu_resident_inputs_matches_cpu`,
  and `gpu_gelu_matches_cpu_within_tolerance` — these exercise the
  exact paths that changed and check numerical equivalence with CPU
  within 1e-3.
- Clippy clean on `--lib --benches` for both crates.

### Sync semantics — important context for future kernel work

cudarc's `clone_dtoh` for plain `Vec<T>` destinations issues a stream-ordered
async memcpy, but its `SyncOnDrop::Sync(None)` is a **no-op**. The host
data is not guaranteed visible on return. `CudaBuffer::read_back` now
calls `stream.synchronize()` explicitly after the copy. Any new kernel
or dispatch path added on the CUDA backend should follow the same
pattern: queue async, sync at storage boundaries.

## alloc_uninit results (2026-05-09)

`cudarc::Stream::alloc_zeros` calls `cuMemAllocAsync` (cheap — hits CUDA's
stream-ordered pool) followed by `cuMemsetD8Async` (a separate kernel
dispatch that writes zeros across the buffer). For output buffers that the
next kernel fully overwrites — matmul C, GELU output, transpose output —
the zero-fill is wasted work.

Added `Backend::alloc_uninit` (default impl falls back to `alloc`, CUDA
overrides to call `Stream::alloc::<u8>` and skip the memset). Mirrored on
`Device::alloc_uninit<T>`. scry-llm's persistent path now uses it for all
output buffers (matmul, GELU, transpose, plus the legacy `gpu_matmul` and
the bench-only batched helper).

Re-running the breakdown bench after the change:

| stage (median µs)  | small | medium | large (1024³) |
|---|---:|---:|---:|
| upload B           |   5.98 |  19.21 |  146.66 |
| matmul (queue)     |   5.85 |   5.68 |    8.04 |
| gelu (queue)       |   1.80 |   1.84 |    1.99 |
| download C         |  11.69 |  59.01 | 1065.23 |
| **chained total**  | **25.39** | **85.98** | **1225.48** |

| size  | post-async chain | post-uninit chain | delta |
|---|---:|---:|---:|
| small  (128, 256, 128) |  27.46 µs |  25.39 µs | **−2.1 µs (−7.5%)** |
| medium (256, 512, 256) |  86.96 µs |  85.98 µs |   ≈ noise          |
| large  (1024³)         | 1243.68 µs | 1225.48 µs |  −18 µs (−1.5%)    |

Per-stage GELU at small dropped 3.27 → 1.80 µs (−45%) — the zero-init
dispatch was nearly half the stage time at this scale. Per-stage matmul
dropped 7.32 → 5.85 µs (−20%) for the same reason.

Cumulative wins this session (cuBLAS already in place at start, then
async dispatch + alloc_uninit on top): **34.09 → 25.39 µs at small,
−25.5%**.

### Note on comparing to PyTorch under async dispatch

The earlier "Matmul + GELU only (vs PyTorch — fair comparison)" table
was meaningful when each stage synced internally, because matmul/gelu
medians measured actual GPU work. Under async dispatch the per-stage
values are launch+queue time only — the GPU is still running when the
timer stops, with the actual compute charged to the next stage that
syncs (download). So per-stage TFLOPS and per-stage µs are no longer
apples-to-apples vs PyTorch.

The chained total *is* still apples-to-apples (both ends synced). For
small/medium that's the meaningful number. For 1024³, the bench's
download cost dominates — real forward passes don't materialize after
every operator, so it overstates what models actually pay. Real-model
perf comparison (item #6 above) is where the cumulative wins get
validated end-to-end. Per-op wins don't count until they show up in a
full forward pass.

## Source

- Bench: `crates/scry-llm/benches/gpu_breakdown.rs`
- Companion: `HANDOFF_gpu_resident.md` (next-experiment selection rationale)
- Memory: `project_scry_vision_gpu_residency.md`
