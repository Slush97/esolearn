# GPU-Resident scry-vision — Session Handoff

You are continuing work on branch `gpu-resident/scry-vision`. This doc is your starter — read it first, then check the memory entry `project_scry_vision_gpu_residency.md` for full project context.

## What landed in the previous session (4 commits)

```
93ef9b8 bench(scry-llm): GPU residency microbench validates M1 architecture
59d9725 feat(scry-gpu, scry-llm): GPU-resident GELU shader
697ae66 feat(scry-llm): GPU-resident matmul keeps results on device
10f904f feat(scry-llm): introduce ScryGpuStorage enum for GPU-resident tensors
```

**M1 is done.** ScryGpuStorage is an enum (Cpu/Gpu); matmul and GELU keep results on-device on the Wgsl path. The microbench shows persistent path beats CPU-round-trip by 1.2–1.6× at medium+ sizes.

## The next experiment (recommended)

**Profile the existing bench to break down where time goes.** This is the highest-information-per-context experiment we can run. It tells us whether to invest next in: more kernels, better matmul tiling, batched dispatch, or buffer pool tuning.

### Why this and not something else

- **vs. adding candle as comparator** — multi-hour project (deps, API, equivalent code), yields one comparison number.
- **vs. conv2d (the obvious "biggest win")** — week of focused engineering. Premature without knowing if our matmul kernel is competitive in the first place. If 60% of our time is dispatch overhead, optimizing the kernel itself is the wrong move.
- **vs. softmax/layernorm** — useful for ViT/CLIP but doesn't teach us anything about the substrate's perf characteristics.

The breakdown answers a real question; the alternatives all assume we already know the answer.

### Concrete steps

1. **Read the existing bench** at `crates/scry-llm/benches/gpu_residency.rs`. It runs matmul→GELU on three sizes.

2. **Add a third bench file** `crates/scry-llm/benches/gpu_breakdown.rs` (or modify the existing one — your call). Don't reuse criterion's iteration framework for this — instead write a manual timing loop that reports a stage-by-stage breakdown for ONE workload size. Pseudocode:

```rust
let m = 256; let k = 512; let n = 256;
let a = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(make_random(m*k, 0xA))).unwrap();
let b = ScryGpuBackend::to_gpu(&ScryGpuStorage::Cpu(make_random(k*n, 0xB))).unwrap();

// Warmup
for _ in 0..10 {
    let c = ScryGpuBackend::matmul(&a, &b, m, k, n, false, false);
    let _ = ScryGpuBackend::gelu(&c);
}

// Measure 100 iterations, recording per-stage times
let mut t_matmul = vec![]; let mut t_gelu = vec![]; let mut t_download = vec![];
for _ in 0..100 {
    let t0 = Instant::now();
    let c = ScryGpuBackend::matmul(&a, &b, m, k, n, false, false);
    t_matmul.push(t0.elapsed());

    let t1 = Instant::now();
    let g = ScryGpuBackend::gelu(&c);
    t_gelu.push(t1.elapsed());

    let t2 = Instant::now();
    let _ = ScryGpuBackend::to_vec(&g);
    t_download.push(t2.elapsed());
}
// Print median + p95 for each stage
```

3. **Crucially, add timing for the upload too:** the bench pre-uploads inputs once, but in real workloads inputs are uploaded too. Measure that separately:

```rust
let t = Instant::now();
let _ = ScryGpuBackend::to_gpu(&cpu_storage).unwrap();
let upload_time = t.elapsed();
```

4. **Compute TFLOPS for the matmul kernel.** For (M, K, N) at ~750µs:
   - FMAs = M × K × N = 256 × 512 × 256 = 33.5M
   - Each FMA is 2 FLOPs → 67M FLOPs
   - Time per matmul (subtract GELU, download from total) → TFLOPS
   - Compare to RTX 5070 Ti advertised peak (~55 TFLOPS) — what % are we hitting? scry-gpu status memo says ~28–30% on standalone matmul; if we're getting that here, kernel optimization isn't the bottleneck.

5. **Run it, capture the numbers, write findings into `crates/scry-llm/HACKING_GPU_BREAKDOWN.md` (or extend this doc).**

### Expected findings (your hypothesis to test)

Roughly, if our matmul kernel is at ~30% peak and the workload at 256/512/256 takes ~750µs total, then:
- Matmul compute: ~150–200µs (the actual kernel)
- GELU compute: tiny (~10µs — element-wise, memory-bound)
- Dispatch overhead per op: ~50–100µs (kernel launch, fence wait)
- Download: ~50–200µs depending on output size

If total ≈ 750µs and compute ≈ 200µs, **dispatch is dominant** → invest in batched command buffers (the scry-learn Phase 4 work — there's already a `Device::batch()` API in scry-gpu).

If total ≈ 750µs and compute ≈ 600µs, **kernel is the bottleneck** → optimize matmul shader (try COARSE_8X8, vec4 loads).

You won't know which until you measure.

## Open research questions (background)

1. **Threshold heuristic** is 65K elements for matmul, 16K for elementwise — picked on intuition. Real workloads may want different curves. Empirical curves would be a follow-up bench.
2. **Candle comparator** still not wired. Eventually needed for M5.
3. **Conv2d** is the biggest M2 kernel and gates ResNet-50 going GPU-resident.

## Build commands you'll need

```bash
# Format / clippy / test
cargo fmt -p scry-llm -p scry-gpu
cargo clippy -p scry-llm -p scry-gpu --features scry-gpu --lib
cargo test -p scry-llm --features scry-gpu --lib backend::scry_gpu

# Run the existing bench
cargo bench -p scry-llm --features scry-gpu --bench gpu_residency -- --warm-up-time 1 --measurement-time 3

# scry-vision sweep (catches accidental breakage of Storage type)
cargo test -p scry-vision --features scry-gpu --lib
```

## Gotchas you'll hit

- **scry-stt missing binary** breaks workspace-wide builds. Use `-p` targeting.
- **GPG signing** may prompt for passphrase mid-commit. If it fails, write commit message to `/tmp/commit_msg` and the user can run `! git commit -F /tmp/commit_msg` themselves.
- **rustfmt is required** before commit; the workspace has a `cargo fmt --check` style policy.
- **fp32 tolerance** for chained ops needs to be relative (`1e-4 * value.abs().max(1.0)`), not absolute. Got bitten by this on `chained_matmuls_stay_on_gpu`.
- **CuBLAS path is unaffected** by all M1/M2 work currently — the persistent matmul only kicks in on the Wgsl path. Don't try to "fix" cuBLAS without explicit go-ahead; CUDA transpose/GELU shaders need wiring first.

## Stop conditions

Stop and ask the user before:
- Adding a heavy dep (candle-core, candle-nn, ort) — these are multi-hour decisions.
- Touching scry-vision model files (resnet.rs, vit.rs, clip.rs) — that's M3/M4 territory and needs M2 kernels first.
- Modifying the MathBackend trait surface — affects scry-llm's own LLM inference path.

Otherwise: commit small, run tests, stage commit messages in `/tmp/commit_msg`.
