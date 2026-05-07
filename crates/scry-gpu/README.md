# scry-gpu

[![Crates.io](https://img.shields.io/crates/v/scry-gpu.svg)](https://crates.io/crates/scry-gpu)
[![docs.rs](https://img.shields.io/docsrs/scry-gpu)](https://docs.rs/scry-gpu)
[![License](https://img.shields.io/crates/l/scry-gpu.svg)](https://github.com/Slush97/esolearn/blob/main/LICENSE-MIT)

Compute-only GPU dispatch for Rust. No render passes, no swapchains, no framebuffers — upload data, run a WGSL shader, read results back.

```rust
use scry_gpu::Device;

let gpu = Device::auto()?;

let input  = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0])?;
let output = gpu.alloc::<f32>(4)?;

let shader = "
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * 2.0;
    }
}";

gpu.dispatch(shader, &[&input, &output], 4)?;

let result: Vec<f32> = output.download()?;
assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
```

> **Status:** 0.x, pre-1.0 — the API may break between minor versions. The Vulkan backend is the most exercised path. CUDA is supported via cudarc + NVRTC for sites that prefer it; Metal is on the roadmap.

---

## Why this exists

`wgpu` is a great GPU library, but its dependency tree is built around graphics: window surfaces, swapchains, render pipelines. For pure compute workloads (machine learning, data-parallel math, simulation) most of that is unused weight on compile time and binary size. `scry-gpu` skips the graphics surface and goes directly to ash (Vulkan) and naga (WGSL → SPIR-V), with optional cudarc for CUDA.

The result is a smaller dependency footprint, faster builds, and an API shaped around dispatching kernels rather than drawing pixels.

---

## Design

- **Compute only.** No graphics state. The public type surface is `Device`, `Buffer<T>`, `Kernel`, `Batch` — and `dispatch`.
- **Auto-dispatch.** Workgroup counts are computed from the invocation count and the shader's `@workgroup_size`. No manual `ceil(n / 64)` at every call site.
- **Typed buffers.** `Buffer<f32>::upload(&[f32])` and `Buffer<f32>::download() -> Vec<f32>`. Staging, alignment, and host-device sync are internal.
- **Pipeline cache.** SPIR-V module hashes are cached at `~/.cache/scry-gpu/<vendor>-<device>.bin`, so kernel compilation pays once across runs.
- **Thread-safe.** `Device` is `Send + Sync`; submission is internally serialized.

---

## Backends

| Backend | Feature | Status |
|---------|---------|--------|
| Vulkan  | `vulkan` (default) | Primary. Tested on AMD, NVIDIA, Intel iGPU. |
| CUDA    | `cuda`  | Optional. cudarc + NVRTC + cuBLAS for matmul. |
| Metal   | —       | Planned. |

```toml
[dependencies]
scry-gpu = "0.1"                              # vulkan only (default)
scry-gpu = { version = "0.1", features = ["cuda"] }  # vulkan + cuda
```

`Device::auto()` picks the fastest available backend. Use `Device::with_backend(BackendKind::Vulkan)` to pin one explicitly.

---

## Built-in shaders

`scry_gpu::shaders` exposes WGSL strings for the kernels that ML workloads hit most often:

- **Tiled matrix multiply** — 16×16 shared-memory tiles, plus a coarse 64×64 variant for large matrices
- **Pairwise squared Euclidean distance** — `n_q × n_t` distance matrix from row-major query and reference matrices

When the `cuda` feature is on, the same kernels are exposed as CUDA C strings for NVRTC compilation. For matmul on CUDA, prefer `Device::cublas_matmul` over custom kernels — cuBLAS reaches >80 % of peak immediately.

---

## Multi-dispatch batches

For workloads with many small kernel launches (e.g. neural network forward/backward passes), `Device::batch()` records dispatches into a single command buffer with one fence wait at the end:

```rust
let mut batch = gpu.batch()?;
batch.run(&kernel_a, &[&x, &w1, &z1], n)?;
batch.run(&kernel_b, &[&z1, &w2, &z2], n)?;
batch.submit()?;  // single fence
```

This avoids the per-dispatch submission overhead that dominates small-kernel timelines.

---

## Requirements

- **Vulkan 1.2+** drivers (default backend). On Linux, Mesa 22+ for AMD/Intel; NVIDIA proprietary or open-kernel drivers.
- **CUDA 12.0+** if using the `cuda` feature. cudarc detects the toolkit at build time.
- **Rust 1.85+** (edition 2021, workspace MSRV).

Tests fail gracefully (`GpuError::NoDevice`) on systems without a usable GPU rather than hanging or panicking.

---

## Benchmarks

```bash
cargo run -p scry-gpu --example bench_compute --release
cargo run -p scry-gpu --example bench_kernel  --release

# Backend comparisons
cargo run -p scry-gpu --example bench_wgpu_compare --release --features bench-wgpu
cargo run -p scry-gpu --example bench_cuda_compare --release --features cuda
```

The `bench_compute` example covers SAXPY, reduction, and matmul under steady-state dispatch with cached kernels — the regime that matters for sustained ML training, not first-call latency.

---

## License

Dual-licensed under MIT or Apache-2.0, at your option.
