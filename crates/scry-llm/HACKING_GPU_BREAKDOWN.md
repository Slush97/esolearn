# GPU Breakdown — Where Does Time Go?

Findings from `cargo bench -p scry-llm --features scry-gpu --bench gpu_breakdown` on `gpu-resident/scry-vision`.

The bench times each stage of `C = GELU(A @ B)` separately at three sizes: upload, matmul, GELU, download. Reports median + p95 per stage.

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

## Out of scope here

- **Splitting per-op time into "kernel compute" vs "fence wait" vs "command-buffer record"** would need GPU timestamp queries (`vkCmdWriteTimestamp`, `vkCmdResetQueryPool`). That's a scry-gpu-side instrumentation effort, not a scry-llm bench. The compute-floor argument above pins down compute time well enough to justify the next investment without it.
- **Comparing to candle-core** — still not wired. Self-vs-self proves architecture; self-vs-industry proves competitiveness. Worth doing once batching lands and we have a number worth defending.
- **Sweeping buffer-pool / threshold heuristics** (`GPU_MIN_ELEMENTS = 65_536`, `GPU_ELEMENTWISE_MIN = 16_384`). These were picked on intuition; the breakdown shows the small workload (4M FMAs, just over the matmul threshold) gets a 0.18 TFLOPS effective rate — the threshold may be too aggressive. Empirical curves would be a follow-up bench.

## Source

- Bench: `crates/scry-llm/benches/gpu_breakdown.rs`
- Companion: `HANDOFF_gpu_resident.md` (next-experiment selection rationale)
- Memory: `project_scry_vision_gpu_residency.md`
