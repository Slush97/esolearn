// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared WGSL shader sources for reuse across crates.
//!
//! Each constant is a complete WGSL shader string ready to pass to
//! [`Device::compile`](crate::Device::compile). Push constant layouts
//! and workgroup sizes are documented per shader.

/// Matrix multiplication shaders.
///
/// Each shader is available as a WGSL constant (for Vulkan dispatch) and,
/// when the `cuda` feature is enabled, as a CUDA C constant (for NVRTC
/// compilation via [`Device::compile_cuda`](crate::Device::compile_cuda)).
///
/// For CUDA matmul, prefer [`Device::cublas_matmul`](crate::Device::cublas_matmul)
/// over custom kernels — cuBLAS reaches 80%+ peak throughput immediately.
pub mod matmul {
    /// Tiled matmul: 16x16 shared-memory tiles, 1 element per thread.
    ///
    /// **Push constants:** `struct Dims { M: u32, N: u32, K: u32 }` (12 bytes)
    /// **Workgroup size:** `(16, 16)` — dispatch `[N.div_ceil(16), M.div_ceil(16), 1]`
    /// **Shared memory:** 2 x 256 floats (2 KB)
    pub const TILED_16X16: &str = "\
struct Dims { M: u32, N: u32, K: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.y * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    let lr = lid.y;
    let lc = lid.x;

    var sum = 0.0;
    let num_tiles = (dims.K + 15u) / 16u;

    for (var t = 0u; t < num_tiles; t++) {
        let a_col = t * 16u + lc;
        if row < dims.M && a_col < dims.K {
            tile_a[lr * 16u + lc] = A[row * dims.K + a_col];
        } else {
            tile_a[lr * 16u + lc] = 0.0;
        }

        let b_row = t * 16u + lr;
        if b_row < dims.K && col < dims.N {
            tile_b[lr * 16u + lc] = B[b_row * dims.N + col];
        } else {
            tile_b[lr * 16u + lc] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < 16u; k++) {
            sum += tile_a[lr * 16u + k] * tile_b[k * 16u + lc];
        }

        workgroupBarrier();
    }

    if row < dims.M && col < dims.N {
        C[row * dims.N + col] = sum;
    }
}";

    /// CUDA C equivalent of [`TILED_16X16`].
    ///
    /// **Kernel signature:** `matmul_tiled_16x16(const float* A, const float* B, float* C, unsigned int M, unsigned int N, unsigned int K)`
    /// **Block size:** `(16, 16)` — dispatch `[N.div_ceil(16), M.div_ceil(16), 1]`
    #[cfg(feature = "cuda")]
    pub const TILED_16X16_CUDA: &str = "\
extern \"C\" __global__ void matmul_tiled_16x16(
    const float* A, const float* B, float* C,
    unsigned int M, unsigned int N, unsigned int K
) {
    __shared__ float tile_a[256];
    __shared__ float tile_b[256];

    unsigned int row = blockIdx.y * 16 + threadIdx.y;
    unsigned int col = blockIdx.x * 16 + threadIdx.x;
    unsigned int lr = threadIdx.y;
    unsigned int lc = threadIdx.x;

    float sum = 0.0f;
    unsigned int num_tiles = (K + 15) / 16;

    for (unsigned int t = 0; t < num_tiles; t++) {
        unsigned int a_col = t * 16 + lc;
        tile_a[lr * 16 + lc] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        unsigned int b_row = t * 16 + lr;
        tile_b[lr * 16 + lc] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (unsigned int k = 0; k < 16; k++) {
            sum += tile_a[lr * 16 + k] * tile_b[k * 16 + lc];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}";

    /// Thread-coarsened matmul: 64x64 output tile, each thread computes 4x4.
    ///
    /// **Push constants:** `struct Dims { M: u32, N: u32, K: u32 }` (12 bytes)
    /// **Workgroup size:** `(16, 16)` = 256 threads, each owns a 4x4 output block.
    /// **Dispatch:** `[N.div_ceil(64), M.div_ceil(64), 1]`
    /// **Shared memory:** A\[64x(16+1)\] + B\[16x64\] = ~8.5 KB (A padded to stride 17
    /// to eliminate bank conflicts).
    /// **Arithmetic intensity:** 16 FLOP/byte (4x over the simple tiled kernel).
    pub const COARSE_64X64: &str = "\
struct Dims { M: u32, N: u32, K: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

var<workgroup> sa: array<f32, 1088>;
var<workgroup> sb: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let block_row = wid.y * 64u;
    let block_col = wid.x * 64u;
    let tr = lid.y * 4u;
    let tc = lid.x * 4u;

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

    let num_k_tiles = (dims.K + 15u) / 16u;

    for (var kt = 0u; kt < num_k_tiles; kt++) {
        // Load A tile [64x16] into padded layout (stride 17)
        for (var x = 0u; x < 4u; x++) {
            let flat = li * 4u + x;
            let r = flat / 16u;
            let c = flat % 16u;
            let gr = block_row + r;
            let gc = kt * 16u + c;
            if gr < dims.M && gc < dims.K {
                sa[r * 17u + c] = A[gr * dims.K + gc];
            } else {
                sa[r * 17u + c] = 0.0;
            }
        }

        // Load B tile [16x64]
        for (var x = 0u; x < 4u; x++) {
            let flat = li * 4u + x;
            let r = flat / 64u;
            let c = flat % 64u;
            let gr = kt * 16u + r;
            let gc = block_col + c;
            if gr < dims.K && gc < dims.N {
                sb[flat] = B[gr * dims.N + gc];
            } else {
                sb[flat] = 0.0;
            }
        }

        workgroupBarrier();

        for (var k = 0u; k < 16u; k++) {
            for (var i = 0u; i < 4u; i++) {
                let a_val = sa[(tr + i) * 17u + k];
                for (var j = 0u; j < 4u; j++) {
                    acc[i * 4u + j] += a_val * sb[k * 64u + tc + j];
                }
            }
        }

        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let gr = block_row + tr + i;
            let gc = block_col + tc + j;
            if gr < dims.M && gc < dims.N {
                C[gr * dims.N + gc] = acc[i * 4u + j];
            }
        }
    }
}";
    /// Thread-coarsened matmul: 128x128 tile, 8x8 per thread with vec4 accumulators.
    ///
    /// Uses 16 named `vec4<f32>` accumulator variables instead of `array<f32, 64>`
    /// to avoid NVIDIA SPIR-V register spill (which triggers at `array<f32, 32+>`).
    /// Vec4 loads from the B shared-memory tile halve load instruction count.
    ///
    /// **Push constants:** `struct Dims { M: u32, N: u32, K: u32 }` (12 bytes)
    /// **Workgroup size:** `(16, 16)` = 256 threads, each owns an 8×8 output block.
    /// **Dispatch:** `[N.div_ceil(128), M.div_ceil(128), 1]`
    /// **Shared memory:** A\[128×(16+1)\] + B\[16×128\] ≈ 16.6 KB
    /// **Arithmetic intensity:** 64 FLOP per (8+2) loads ≈ 6.4 FMA/load (3.2× over 4×4).
    pub const COARSE_8X8: &str = "\
struct Dims { M: u32, N: u32, K: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

var<workgroup> sa: array<f32, 2176>;
var<workgroup> sb: array<f32, 2048>;

fn store_row(gr: u32, gc: u32, lo: vec4<f32>, hi: vec4<f32>) {
    if gr >= dims.M { return; }
    let base = gr * dims.N + gc;
    if gc < dims.N { C[base] = lo.x; }
    if gc + 1u < dims.N { C[base + 1u] = lo.y; }
    if gc + 2u < dims.N { C[base + 2u] = lo.z; }
    if gc + 3u < dims.N { C[base + 3u] = lo.w; }
    if gc + 4u < dims.N { C[base + 4u] = hi.x; }
    if gc + 5u < dims.N { C[base + 5u] = hi.y; }
    if gc + 6u < dims.N { C[base + 6u] = hi.z; }
    if gc + 7u < dims.N { C[base + 7u] = hi.w; }
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let block_row = wid.y * 128u;
    let block_col = wid.x * 128u;
    let tr = lid.y * 8u;
    let tc = lid.x * 8u;

    // 16 named vec4 accumulators — avoids array-based register spill.
    var r0l = vec4<f32>(0.0); var r0h = vec4<f32>(0.0);
    var r1l = vec4<f32>(0.0); var r1h = vec4<f32>(0.0);
    var r2l = vec4<f32>(0.0); var r2h = vec4<f32>(0.0);
    var r3l = vec4<f32>(0.0); var r3h = vec4<f32>(0.0);
    var r4l = vec4<f32>(0.0); var r4h = vec4<f32>(0.0);
    var r5l = vec4<f32>(0.0); var r5h = vec4<f32>(0.0);
    var r6l = vec4<f32>(0.0); var r6h = vec4<f32>(0.0);
    var r7l = vec4<f32>(0.0); var r7h = vec4<f32>(0.0);

    let num_k_tiles = (dims.K + 15u) / 16u;

    for (var kt = 0u; kt < num_k_tiles; kt++) {
        // Load A tile [128x16] — 2048 elements, 8 per thread, padded stride 17
        for (var x = 0u; x < 8u; x++) {
            let flat = li * 8u + x;
            let r = flat / 16u;
            let c = flat % 16u;
            let gr = block_row + r;
            let gc = kt * 16u + c;
            if gr < dims.M && gc < dims.K {
                sa[r * 17u + c] = A[gr * dims.K + gc];
            } else {
                sa[r * 17u + c] = 0.0;
            }
        }

        // Load B tile [16x128] — 2048 elements, 8 per thread
        for (var x = 0u; x < 8u; x++) {
            let flat = li * 8u + x;
            let r = flat / 128u;
            let c = flat % 128u;
            let gr = kt * 16u + r;
            let gc = block_col + c;
            if gr < dims.K && gc < dims.N {
                sb[flat] = B[gr * dims.N + gc];
            } else {
                sb[flat] = 0.0;
            }
        }

        workgroupBarrier();

        // Inner loop: 8 a-scalar loads + 2 vec4 b-loads + 16 vec4 FMAs per k
        for (var k = 0u; k < 16u; k++) {
            let bk = k * 128u + tc;
            let bl = vec4<f32>(sb[bk], sb[bk+1u], sb[bk+2u], sb[bk+3u]);
            let bh = vec4<f32>(sb[bk+4u], sb[bk+5u], sb[bk+6u], sb[bk+7u]);

            let a0 = sa[(tr    ) * 17u + k]; r0l += a0 * bl; r0h += a0 * bh;
            let a1 = sa[(tr+1u) * 17u + k]; r1l += a1 * bl; r1h += a1 * bh;
            let a2 = sa[(tr+2u) * 17u + k]; r2l += a2 * bl; r2h += a2 * bh;
            let a3 = sa[(tr+3u) * 17u + k]; r3l += a3 * bl; r3h += a3 * bh;
            let a4 = sa[(tr+4u) * 17u + k]; r4l += a4 * bl; r4h += a4 * bh;
            let a5 = sa[(tr+5u) * 17u + k]; r5l += a5 * bl; r5h += a5 * bh;
            let a6 = sa[(tr+6u) * 17u + k]; r6l += a6 * bl; r6h += a6 * bh;
            let a7 = sa[(tr+7u) * 17u + k]; r7l += a7 * bl; r7h += a7 * bh;
        }

        workgroupBarrier();
    }

    let gc = block_col + tc;
    store_row(block_row + tr,      gc, r0l, r0h);
    store_row(block_row + tr + 1u, gc, r1l, r1h);
    store_row(block_row + tr + 2u, gc, r2l, r2h);
    store_row(block_row + tr + 3u, gc, r3l, r3h);
    store_row(block_row + tr + 4u, gc, r4l, r4h);
    store_row(block_row + tr + 5u, gc, r5l, r5h);
    store_row(block_row + tr + 6u, gc, r6l, r6h);
    store_row(block_row + tr + 7u, gc, r7l, r7h);
}";
}

/// Element-wise activation and bias shaders.
///
/// All shaders use workgroup size 256 (1D) and take a push constant `N: u32`
/// for bounds checking. Each thread processes one element.
pub mod elementwise {
    /// Bias add: `out[i] = z[i] + bias[i % cols]`.
    ///
    /// **Push constants:** `struct Dims { N: u32, cols: u32 }` (8 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations (N = rows * cols)
    /// **Bindings:**
    ///   - `@binding(0)` `z: array<f32>` (read) — input matrix `[rows, cols]`
    ///   - `@binding(1)` `bias: array<f32>` (read) — bias vector `[cols]`
    ///   - `@binding(2)` `out: array<f32>` (`read_write`) — output `[rows, cols]`
    pub const BIAS_ADD: &str = "\
struct Dims { N: u32, cols: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    out[i] = z[i] + bias[i % dims.cols];
}";

    /// CUDA C equivalent of [`BIAS_ADD`].
    #[cfg(feature = "cuda")]
    pub const BIAS_ADD_CUDA: &str = "\
extern \"C\" __global__ void bias_add(
    const float* z, const float* bias, float* out,
    unsigned int N, unsigned int cols
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = z[i] + bias[i % cols];
}";

    /// `ReLU` activation: `out[i] = max(0, in[i])`.
    ///
    /// **Push constants:** `struct Dims { N: u32 }` (4 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `input: array<f32>` (read)
    ///   - `@binding(1)` `out: array<f32>` (`read_write`)
    pub const RELU: &str = "\
struct Dims { N: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    out[i] = max(0.0, input[i]);
}";

    /// CUDA C equivalent of [`RELU`].
    #[cfg(feature = "cuda")]
    pub const RELU_CUDA: &str = "\
extern \"C\" __global__ void relu(
    const float* input, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = fmaxf(0.0f, input[i]);
}";

    /// `SiLU` / Swish activation: `out[i] = x * sigmoid(x) = x / (1 + exp(-x))`.
    ///
    /// Used in every Stable-Diffusion `UNet` `ResBlock` (and elsewhere in the
    /// SD family). Same dispatch shape as [`GELU_CUDA`]; the only arithmetic
    /// difference is the sigmoid instead of the tanh-approx polynomial.
    ///
    /// **Kernel signature:** `silu(const float* input, float* out, unsigned int N)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[N.div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const SILU_CUDA: &str = "\
extern \"C\" __global__ void silu(
    const float* input, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = input[i];
    out[i] = x / (1.0f + expf(-x));
}";

    /// Exact (erf-based) GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    ///
    /// Distinct from [`GELU_CUDA`] (the tanh approximation). Used by
    /// PyTorch's `F.gelu(approximate=\"none\")` and HF's GeGLU MLP — the
    /// SD UNet's GeGLU layer uses exactly this form, and the tanh
    /// approximation drifts ~3e-4 per block, accumulating across SD
    /// 1.5's 16 transformer blocks to break the 1e-3 parity gate.
    ///
    /// CUDA's `erff` intrinsic is the right primitive — it's the same
    /// erf PyTorch uses on device, so post-kernel results match
    /// PyTorch's bit-for-bit on identical inputs (within fp32
    /// rounding).
    ///
    /// **Kernel signature:** `gelu_exact(const float* input, float* out, unsigned int N)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[N.div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const GELU_EXACT_CUDA: &str = "\
extern \"C\" __global__ void gelu_exact(
    const float* input, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = input[i];
    // 1 / sqrt(2) = 0.70710678118654752440f
    out[i] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}";

    /// Tanh activation: `out[i] = tanh(in[i])`.
    ///
    /// **Push constants:** `struct Dims { N: u32 }` (4 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `input: array<f32>` (read)
    ///   - `@binding(1)` `out: array<f32>` (`read_write`)
    pub const TANH: &str = "\
struct Dims { N: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    out[i] = tanh(input[i]);
}";

    /// CUDA C equivalent of [`TANH`].
    #[cfg(feature = "cuda")]
    pub const TANH_CUDA: &str = "\
extern \"C\" __global__ void tanh_fwd(
    const float* input, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = tanhf(input[i]);
}";

    /// Sigmoid activation: `out[i] = 1 / (1 + exp(-in[i]))`.
    ///
    /// **Push constants:** `struct Dims { N: u32 }` (4 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `input: array<f32>` (read)
    ///   - `@binding(1)` `out: array<f32>` (`read_write`)
    pub const SIGMOID: &str = "\
struct Dims { N: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    out[i] = 1.0 / (1.0 + exp(-input[i]));
}";

    /// CUDA C equivalent of [`SIGMOID`].
    #[cfg(feature = "cuda")]
    pub const SIGMOID_CUDA: &str = "\
extern \"C\" __global__ void sigmoid(
    const float* input, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = 1.0f / (1.0f + expf(-input[i]));
}";

    /// GELU activation (tanh approximation):
    /// `out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    ///
    /// **Push constants:** `struct Dims { N: u32 }` (4 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `input: array<f32>` (read)
    ///   - `@binding(1)` `out: array<f32>` (`read_write`)
    pub const GELU: &str = "\
struct Dims { N: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

const SQRT_2_OVER_PI: f32 = 0.7978845608028654;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    let x = input[i];
    let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
    out[i] = 0.5 * x * (1.0 + tanh(inner));
}";

    /// CUDA C equivalent of [`GELU`].
    #[cfg(feature = "cuda")]
    pub const GELU_CUDA: &str = "\
extern \"C\" __global__ void gelu(
    const float* input, float* out,
    unsigned int N
) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = input[i];
    float inner = SQRT_2_OVER_PI * (x + 0.044715f * x * x * x);
    out[i] = 0.5f * x * (1.0f + tanhf(inner));
}";

    /// Same-shape elementwise add: `out[i] = a[i] + b[i]`.
    ///
    /// Distinct from [`BIAS_ADD`] (column-broadcast over a row vector) and
    /// [`ADD_ROW_BIAS_CUDA`] (column-broadcast over a column vector). Use this
    /// when both operands have identical shape — e.g. `ResNet` residual adds.
    ///
    /// **Push constants:** `struct Dims { N: u32 }` (4 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `a: array<f32>` (read)
    ///   - `@binding(1)` `b: array<f32>` (read)
    ///   - `@binding(2)` `out: array<f32>` (`read_write`)
    pub const ADD_ELEMENTWISE: &str = "\
struct Dims { N: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    out[i] = a[i] + b[i];
}";

    /// CUDA C equivalent of [`ADD_ELEMENTWISE`].
    #[cfg(feature = "cuda")]
    pub const ADD_ELEMENTWISE_CUDA: &str = "\
extern \"C\" __global__ void add_elementwise(
    const float* a, const float* b, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = a[i] + b[i];
}";

    /// Same-shape elementwise multiply: `out[i] = a[i] * b[i]`.
    ///
    /// Mirror of [`ADD_ELEMENTWISE`]. Used by `GeGLU`'s `values * gelu(gate)`
    /// gating step in scry-diffusion's `UNet` feed-forward, where the deepest
    /// stage multiplies `[1024, 5120]` tensors.
    ///
    /// **Push constants:** `struct Dims { N: u32 }` (4 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `a: array<f32>` (read)
    ///   - `@binding(1)` `b: array<f32>` (read)
    ///   - `@binding(2)` `out: array<f32>` (`read_write`)
    pub const MUL_ELEMENTWISE: &str = "\
struct Dims { N: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    out[i] = a[i] * b[i];
}";

    /// CUDA C equivalent of [`MUL_ELEMENTWISE`].
    #[cfg(feature = "cuda")]
    pub const MUL_ELEMENTWISE_CUDA: &str = "\
extern \"C\" __global__ void mul_elementwise(
    const float* a, const float* b, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = a[i] * b[i];
}";

    /// Row-wise softmax over the last dimension (numerically stable).
    ///
    /// For an input tensor reshaped as `[n_rows, d]`, computes
    /// `out[r, j] = exp(in[r, j] - max(in[r, *])) / sum_k exp(in[r, k] - max(in[r, *]))`.
    /// Three passes per row: max-reduce, exp+sum-reduce, normalize.
    /// Threads in a block cooperate via static shared memory for both reductions,
    /// so each block processes one row independently and any `d > blockDim.x` is
    /// handled by a strided per-thread loop.
    ///
    /// **Kernel signature:** `softmax_rowwise(const float* input, float* out, unsigned int n_rows, unsigned int d)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[n_rows, 1, 1]` blocks.
    /// **Shared memory:** static, 256 floats (1 KiB).
    #[cfg(feature = "cuda")]
    pub const SOFTMAX_ROWWISE_CUDA: &str = "\
extern \"C\" __global__ void softmax_rowwise(
    const float* input, float* out,
    unsigned int n_rows, unsigned int d
) {
    __shared__ float smem[256];

    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* row_in = input + row * d;
    float* row_out = out + row * d;
    unsigned int tid = threadIdx.x;
    unsigned int bs = blockDim.x;

    // Pass 1: per-thread partial max, then block-wide reduction.
    // Use -FLT_MAX as the sentinel; NVRTC does not include <math.h> by
    // default, so INFINITY / FLT_MAX macros aren't in scope.
    float local_max = -3.402823466e38f;
    for (unsigned int i = tid; i < d; i += bs) {
        float v = row_in[i];
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = smem[tid];
            float b = smem[tid + s];
            smem[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: write exp(x - max), accumulate per-thread sum, block-reduce.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < d; i += bs) {
        float e = expf(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float row_sum = smem[0];

    // Pass 3: normalize. row_sum > 0 by construction (at least one exp(0) = 1).
    float inv = 1.0f / row_sum;
    for (unsigned int i = tid; i < d; i += bs) {
        row_out[i] *= inv;
    }
}";

    /// Row-wise fused scaled softmax: `softmax(scale · x)` along the last
    /// axis in a single kernel. Mathematically identical to dispatching
    /// [`SCALE_CUDA`](super::backward::SCALE_CUDA) followed by
    /// [`SOFTMAX_ROWWISE_CUDA`], but reads the input once instead of
    /// writing-and-rereading a scaled intermediate — eliminates the
    /// standalone scale kernel's `(read d + write d)` per row of bandwidth.
    /// At SD's deepest self-attn stage (`[B=8, n=4096, n=4096]` = 537 MB
    /// scores), this saves ~1.07 GB of memory traffic per call.
    ///
    /// Numerically stable via the standard max-shift trick — but the max
    /// is computed over the *scaled* values, so any `scale` (positive,
    /// negative, or zero) produces the same result as the unfused
    /// dispatch pair.
    ///
    /// **Kernel signature:** `scaled_softmax_rowwise(const float* input, float* out, unsigned int n_rows, unsigned int d, float scale)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[n_rows, 1, 1]` blocks.
    /// **Shared memory:** static, 256 floats (1 KiB).
    #[cfg(feature = "cuda")]
    pub const SCALED_SOFTMAX_ROWWISE_CUDA: &str = "\
extern \"C\" __global__ void scaled_softmax_rowwise(
    const float* input, float* out,
    unsigned int n_rows, unsigned int d, float scale
) {
    __shared__ float smem[256];

    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* row_in = input + row * d;
    float* row_out = out + row * d;
    unsigned int tid = threadIdx.x;
    unsigned int bs = blockDim.x;

    // Pass 1: per-thread partial max OF THE SCALED VALUES, then block reduce.
    float local_max = -3.402823466e38f;
    for (unsigned int i = tid; i < d; i += bs) {
        float v = row_in[i] * scale;
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = smem[tid];
            float b = smem[tid + s];
            smem[tid] = a > b ? a : b;
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: write exp(x*scale - max), accumulate per-thread sum, block-reduce.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < d; i += bs) {
        float e = expf(row_in[i] * scale - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float row_sum = smem[0];

    // Pass 3: normalize. row_sum > 0 by construction (at least one exp(0) = 1).
    float inv = 1.0f / row_sum;
    for (unsigned int i = tid; i < d; i += bs) {
        row_out[i] *= inv;
    }
}";

    /// Fused multi-head scaled dot-product attention — single kernel for the
    /// `scores → softmax → values` cascade.
    ///
    /// Mathematically equivalent to dispatching
    /// [`super::matmul::TILED_16X16_CUDA`] (scores = Q · Kᵀ · scale),
    /// [`SCALED_SOFTMAX_ROWWISE_CUDA`], and a second matmul (out = attn · V)
    /// — but never materializes the `[num_heads · n_q, n_kv]` softmax
    /// intermediate. At Stable Diffusion 1.5's deepest self-attn stage
    /// (`num_heads=8, n_q=n_kv=4096`), that intermediate is ~537 MB of
    /// device traffic per attention layer; this kernel turns those reads
    /// + writes into per-block accumulator updates in shared memory.
    ///
    /// FlashAttention-1 style online softmax: each block maintains a
    /// running max `m`, running denom `l`, and `head_dim`-wide accumulator
    /// `o`. For each K/V row visited, the softmax statistics are
    /// rescaled by `exp(m_old - m_new)` so they remain numerically
    /// equivalent to a single-pass softmax over the full row of scores.
    /// Reference: Dao et al., "FlashAttention" (2022) §3.1.
    ///
    /// **Block layout:** `(128, 1, 1)` threads. Each block computes one
    /// `(head, q_row)` output vector — strided over the head_dim
    /// elements when `head_dim > 128`. Block reductions use the lower
    /// 64 threads for the dot-product sum tree (standard pattern; the
    /// upper 64 hold partials in `red_smem` then drop out of the
    /// per-step `if (tid < s)` halving).
    ///
    /// **Grid:** `(n_q, num_heads, 1)`. One block per output query row
    /// per head — gives `num_heads × n_q` blocks total. Avoids
    /// per-warp work imbalance: every block does the same amount of
    /// `n_kv`-loop work.
    ///
    /// **Shared memory:** static, 3·256 + 4 floats (~3.1 KiB):
    /// `q_smem[256]` (the Q row, loaded once and reused across the
    /// K/V loop), `o_smem[256]` (the running fp32 accumulator),
    /// `red_smem[128]` (the dot-product reduction scratch), and the
    /// four scalars (`m`, `l`, `alpha`, `p`) broadcast from thread 0.
    /// `head_dim` up to 256 is supported in-place — SD 1.5 uses 40 /
    /// 80 / 160; SDXL adds 64 / 128.
    ///
    /// **Numerical correctness:** matches the unfused cascade within
    /// `1e-4` abs at fp32 inputs (online-softmax rescale is
    /// algebraically exact; the fp32 accumulator order differs from
    /// the cuBLAS strided-batched gemm so individual elements may
    /// differ by 1 ulp on the order of `1e-7`). The CPU-side cascade
    /// reference and an equivalence test live in
    /// `scry_llm::backend::scry_gpu::tests::gpu_fused_attention_*`.
    ///
    /// **Kernel signature:** `fused_attention(const float* q, const float* k, const float* v, float* out, unsigned int n_q, unsigned int n_kv, unsigned int d, float scale)`
    /// where each tensor is `[num_heads, *, d]` row-major (stride
    /// derived from `n_q` / `n_kv` and `d` inside the kernel).
    /// **Block size:** `(128, 1, 1)`.
    #[cfg(feature = "cuda")]
    pub const FUSED_ATTENTION_CUDA: &str = "\
extern \"C\" __global__ void fused_attention(
    const float* q, const float* k, const float* v, float* out,
    unsigned int n_q, unsigned int n_kv, unsigned int d, float scale
) {
    __shared__ float q_smem[256];
    __shared__ float o_smem[256];
    __shared__ float red_smem[128];
    __shared__ float m_sh;
    __shared__ float l_sh;
    __shared__ float alpha_sh;
    __shared__ float p_sh;

    unsigned int head = blockIdx.y;
    unsigned int q_row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bs = blockDim.x;

    unsigned int qstride_per_head = n_q * d;
    unsigned int kvstride_per_head = n_kv * d;
    const float* q_ptr = q + head * qstride_per_head + q_row * d;
    const float* k_head = k + head * kvstride_per_head;
    const float* v_head = v + head * kvstride_per_head;
    float* o_ptr = out + head * qstride_per_head + q_row * d;

    // Load Q row into shared; init the fp32 accumulator to zero.
    for (unsigned int i = tid; i < d; i += bs) {
        q_smem[i] = q_ptr[i];
        o_smem[i] = 0.0f;
    }
    if (tid == 0) {
        m_sh = -3.402823466e38f;
        l_sh = 0.0f;
    }
    __syncthreads();

    // Online-softmax sweep over K/V rows.
    for (unsigned int ki = 0; ki < n_kv; ki++) {
        const float* k_ptr = k_head + ki * d;
        const float* v_ptr = v_head + ki * d;

        // Block reduction: dot(q_smem, k_ptr) into red_smem[0].
        float partial = 0.0f;
        for (unsigned int i = tid; i < d; i += bs) {
            partial += q_smem[i] * k_ptr[i];
        }
        red_smem[tid] = partial;
        __syncthreads();
        for (unsigned int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) red_smem[tid] += red_smem[tid + s];
            __syncthreads();
        }
        float score = red_smem[0] * scale;

        // Online-softmax update — done by tid 0, broadcast to the rest
        // of the block via `alpha_sh` / `p_sh`.
        if (tid == 0) {
            float m_old = m_sh;
            float m_new = fmaxf(m_old, score);
            alpha_sh = expf(m_old - m_new);
            p_sh = expf(score - m_new);
            l_sh = l_sh * alpha_sh + p_sh;
            m_sh = m_new;
        }
        __syncthreads();

        // Rescale the running output and add the new V row weighted by p.
        float alpha = alpha_sh;
        float p = p_sh;
        for (unsigned int i = tid; i < d; i += bs) {
            o_smem[i] = o_smem[i] * alpha + p * v_ptr[i];
        }
        __syncthreads();
    }

    // Final normalize and write back.
    float inv_l = 1.0f / l_sh;
    for (unsigned int i = tid; i < d; i += bs) {
        o_ptr[i] = o_smem[i] * inv_l;
    }
}";

    /// Row-wise layer normalization with affine gamma/beta.
    ///
    /// For an input tensor reshaped as `[n_rows, d]`, computes
    /// `out[r, j] = ((in[r, j] - mean_r) * rstd_r) * gamma[j] + beta[j]`,
    /// where `mean_r = sum(in[r, *]) / d`, `var_r = sum((in[r, *] - mean_r)^2) / d`,
    /// and `rstd_r = 1 / sqrt(var_r + eps)`. Per-row `mean_r` and `rstd_r` are
    /// also written to the output `means` and `rstds` buffers (one entry per
    /// row) so the backward pass can reuse them.
    ///
    /// One block per row, 256 threads, two block-wide reductions (sum → mean,
    /// then sum-of-squares → variance) sharing a single static-shared-memory
    /// scratchpad. `d > blockDim.x` is handled by a strided per-thread loop, so
    /// any last-dim length works without recompile.
    ///
    /// **Kernel signature:** `layernorm_rowwise(const float* input, const float* gamma, const float* beta, float* out, float* means, float* rstds, unsigned int n_rows, unsigned int d, float eps)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[n_rows, 1, 1]` blocks.
    /// **Shared memory:** static, 256 floats (1 KiB).
    #[cfg(feature = "cuda")]
    pub const LAYERNORM_ROWWISE_CUDA: &str = "\
extern \"C\" __global__ void layernorm_rowwise(
    const float* input, const float* gamma, const float* beta,
    float* out, float* means, float* rstds,
    unsigned int n_rows, unsigned int d, float eps
) {
    __shared__ float smem[256];

    unsigned int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* row_in = input + row * d;
    float* row_out = out + row * d;
    unsigned int tid = threadIdx.x;
    unsigned int bs = blockDim.x;

    // Pass 1: per-thread partial sum, block-wide reduction → mean.
    float local_sum = 0.0f;
    for (unsigned int i = tid; i < d; i += bs) {
        local_sum += row_in[i];
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float inv_d = 1.0f / (float)d;
    float mean = smem[0] * inv_d;
    __syncthreads();

    // Pass 2: per-thread partial sum of squared deviations, block-reduce → var.
    float local_sq = 0.0f;
    for (unsigned int i = tid; i < d; i += bs) {
        float diff = row_in[i] - mean;
        local_sq += diff * diff;
    }
    smem[tid] = local_sq;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float var = smem[0] * inv_d;
    float rstd = rsqrtf(var + eps);

    // Thread 0 writes the per-row stats; other threads can race ahead to pass 3
    // since pass 3 doesn't depend on smem and each thread writes its own
    // output indices.
    if (tid == 0) {
        means[row] = mean;
        rstds[row] = rstd;
    }

    // Pass 3: normalize, scale, shift.
    for (unsigned int i = tid; i < d; i += bs) {
        float norm = (row_in[i] - mean) * rstd;
        row_out[i] = norm * gamma[i] + beta[i];
    }
}";

    /// `im2col` lowering for a 2D convolution input `[C_in, H_in, W_in]`.
    ///
    /// Produces a `[C_in*kH*kW, H_out*W_out]` matrix in row-major layout where
    /// row `c*kH*kW + kh*kW + kw` and column `oh*W_out + ow` is the input
    /// pixel at channel `c`, position `(oh*stride + kh - padding,
    /// ow*stride + kw - padding)`, zero where that position lies outside the
    /// input. Lowering convolution to a GEMM lets the cuBLAS path do the
    /// arithmetic; the kernel here is the lowering itself.
    ///
    /// One thread per output element; each thread does one bounds check + one
    /// load + one store. Memory-bound; the input read pattern is strided per
    /// `(kh, kw)` but the working set is small (one channel plane fits in L2
    /// for ResNet-class layers), so cache hit rate is high.
    ///
    /// **Kernel signature:** `im2col_nchw(const float* input, float* col,
    /// unsigned int c_in, unsigned int h_in, unsigned int w_in,
    /// unsigned int kh, unsigned int kw,
    /// unsigned int stride, unsigned int padding,
    /// unsigned int h_out, unsigned int w_out)`
    /// **Block size:** `(256, 1, 1)` — dispatch
    ///   `[(c_in*kh*kw*h_out*w_out).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const IM2COL_NCHW_CUDA: &str = "\
extern \"C\" __global__ void im2col_nchw(
    const float* input, float* col,
    unsigned int c_in, unsigned int h_in, unsigned int w_in,
    unsigned int kh, unsigned int kw,
    unsigned int stride, unsigned int padding,
    unsigned int h_out, unsigned int w_out
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int spatial_out = h_out * w_out;
    unsigned int khw = kh * kw;
    unsigned int col_rows = c_in * khw;
    unsigned int total = col_rows * spatial_out;
    if (idx >= total) return;

    unsigned int row = idx / spatial_out;
    unsigned int out_col = idx - row * spatial_out;

    unsigned int c = row / khw;
    unsigned int rem = row - c * khw;
    unsigned int khi = rem / kw;
    unsigned int kwi = rem - khi * kw;

    unsigned int oh = out_col / w_out;
    unsigned int ow = out_col - oh * w_out;

    int ih = (int)(oh * stride + khi) - (int)padding;
    int iw = (int)(ow * stride + kwi) - (int)padding;

    float v = 0.0f;
    if (ih >= 0 && ih < (int)h_in && iw >= 0 && iw < (int)w_in) {
        v = input[(c * h_in + (unsigned int)ih) * w_in + (unsigned int)iw];
    }
    col[idx] = v;
}";

    /// Column-broadcast bias add: `out[r*cols + c] = a[r*cols + c] + bias[r]`.
    ///
    /// Mirrors the `[N, M] + [N, 1]` row-bias broadcast that
    /// `CpuBackend::add` already handles — used by Conv2d to add a
    /// `[C_out]` bias to a `[C_out, H_out*W_out]` matmul output without
    /// rounding through the CPU. One thread per output element; the
    /// per-row `bias[r]` load is L1-broadcast across the block.
    ///
    /// **Kernel signature:** `add_row_bias(const float* a, const float* bias,
    /// float* out, unsigned int rows, unsigned int cols)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[(rows*cols).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const ADD_ROW_BIAS_CUDA: &str = "\
extern \"C\" __global__ void add_row_bias(
    const float* a, const float* bias, float* out,
    unsigned int rows, unsigned int cols
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * cols;
    if (idx >= total) return;
    unsigned int r = idx / cols;
    out[idx] = a[idx] + bias[r];
}";

    /// Split a fused-QKV tensor and reshape to per-head layout.
    ///
    /// Reads `[seq, 3 * d_model]` row-major where columns `[0, d_model)` are
    /// Q, `[d_model, 2*d_model)` are K, `[2*d_model, 3*d_model)` are V.
    /// Writes three `[n_heads, seq, d_head]` row-major tensors with the head
    /// dimension folded out:
    ///
    /// ```text
    /// q[h, s, d] = qkv[s, h*d_head + d]
    /// k[h, s, d] = qkv[s, d_model + h*d_head + d]
    /// v[h, s, d] = qkv[s, 2*d_model + h*d_head + d]
    /// ```
    ///
    /// One thread per `(h, s, d)` triple — each thread reads three values
    /// from the same row of `qkv` and writes one each to `q`, `k`, `v`.
    /// Replaces the trait default's full host round-trip in transformer
    /// attention.
    ///
    /// **Kernel signature:** `split_qkv_reshape_heads(const float* qkv,
    /// float* q, float* k, float* v,
    /// unsigned int seq, unsigned int n_heads, unsigned int d_head)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[(n_heads*seq*d_head).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const SPLIT_QKV_RESHAPE_HEADS_CUDA: &str = "\
extern \"C\" __global__ void split_qkv_reshape_heads(
    const float* qkv,
    float* q, float* k, float* v,
    unsigned int seq, unsigned int n_heads, unsigned int d_head
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n_heads * seq * d_head;
    if (idx >= total) return;
    unsigned int d_model = n_heads * d_head;
    // Decode (h, s, d) from the output index in [n_heads, seq, d_head]
    // row-major: idx = (h*seq + s)*d_head + d.
    unsigned int d = idx % d_head;
    unsigned int rest = idx / d_head;
    unsigned int s = rest % seq;
    unsigned int h = rest / seq;
    unsigned int row_base = s * 3u * d_model + h * d_head + d;
    q[idx] = qkv[row_base];
    k[idx] = qkv[row_base + d_model];
    v[idx] = qkv[row_base + 2u * d_model];
}";

    /// Reverse of [`SPLIT_QKV_RESHAPE_HEADS_CUDA`]: pack `[n_heads, seq,
    /// d_head]` back to `[seq, n_heads*d_head]` row-major.
    ///
    /// `out[s, h*d_head + d] = in[h, s, d]`. One thread per output element;
    /// each thread does one strided load + one coalesced store.
    ///
    /// **Kernel signature:** `reshape_from_heads(const float* in, float* out,
    /// unsigned int seq, unsigned int n_heads, unsigned int d_head)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[(seq*n_heads*d_head).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const RESHAPE_FROM_HEADS_CUDA: &str = "\
extern \"C\" __global__ void reshape_from_heads(
    const float* input, float* out,
    unsigned int seq, unsigned int n_heads, unsigned int d_head
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int d_model = n_heads * d_head;
    unsigned int total = seq * d_model;
    if (idx >= total) return;
    // Decode (s, h, d) from output index in [seq, n_heads*d_head] row-major:
    // idx = s*d_model + h*d_head + d.
    unsigned int s = idx / d_model;
    unsigned int rem = idx - s * d_model;
    unsigned int h = rem / d_head;
    unsigned int d = rem - h * d_head;
    out[idx] = input[(h * seq + s) * d_head + d];
}";

    /// Concatenate two row-major matrices along axis 0 (rows). For
    /// `a: [a_rows, cols]` and `b: [b_rows, cols]`, the output is
    /// `[a_rows + b_rows, cols]`. Because both inputs are row-major
    /// with the same `cols`, the operation is a flat append: `a`'s
    /// `a_total` elements followed by `b`'s `b_total` elements.
    ///
    /// One thread per output element. The branch on `i < a_total` is
    /// uniform across each warp until exactly one warp crosses the
    /// boundary, so divergence is bounded.
    ///
    /// Used by the SD UNet's UpBlock skip-concat (12×/forward, on
    /// tensors up to `[2560, 64, 64]` = 10 MB at the shallowest
    /// up-stage) and by the Llama KV cache. Replaces a host-roundtrip
    /// default that this trait method had on `ScryGpuBackend`.
    ///
    /// **Kernel signature:** `concat_rows(const float* a, const float* b, float* out, uint a_total, uint b_total)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[(a_total+b_total).div_ceil(256), 1, 1]`.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const CONCAT_ROWS_CUDA: &str = "\
extern \"C\" __global__ void concat_rows(
    const float* a, const float* b, float* out,
    unsigned int a_total, unsigned int b_total
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = a_total + b_total;
    if (i >= total) return;
    if (i < a_total) {
        out[i] = a[i];
    } else {
        out[i] = b[i - a_total];
    }
}";

    /// Forward permute for multi-head attention: `[seq, n_heads*d_head]` →
    /// `[n_heads, seq, d_head]`. Inverse of [`RESHAPE_FROM_HEADS_CUDA`].
    ///
    /// Used by SD UNet attention to lay out Q/K/V projection outputs into
    /// per-head buffers in one dispatch, replacing a `num_heads`-deep loop
    /// of `gather_columns` calls (each of which is itself a host
    /// roundtrip on `ScryGpuBackend` because there's no kernel override).
    ///
    /// **Kernel signature:** `reshape_to_heads(const float* input, float* out, uint seq, uint n_heads, uint d_head)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[total.div_ceil(256), 1, 1]`.
    /// **Shared memory:** none. Strided gather pattern but each thread
    /// reads / writes one element, so memory access is fully coalescable.
    #[cfg(feature = "cuda")]
    pub const RESHAPE_TO_HEADS_CUDA: &str = "\
extern \"C\" __global__ void reshape_to_heads(
    const float* input, float* out,
    unsigned int seq, unsigned int n_heads, unsigned int d_head
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int d_model = n_heads * d_head;
    unsigned int total = seq * d_model;
    if (idx >= total) return;
    // Decode (h, s, d) from output index in [n_heads, seq, d_head] row-major:
    // idx = (h * seq + s) * d_head + d.
    unsigned int hs = idx / d_head;
    unsigned int d = idx - hs * d_head;
    unsigned int h = hs / seq;
    unsigned int s = hs - h * seq;
    out[idx] = input[s * d_model + h * d_head + d];
}";

    /// 2D batch normalization (inference) with stored running stats.
    ///
    /// For an input shaped `[batch, channels, spatial]` (where `spatial = H*W`),
    /// computes `out[n, c, i] = (in[n, c, i] - mean[c]) * rsqrt(var[c] + eps) * weight[c] + bias[c]`
    /// using fused per-channel `scale = weight[c] * rsqrt(var[c] + eps)` and
    /// `shift = bias[c] - mean[c] * scale`. No reductions — purely elementwise
    /// per `(channel, batch_index)` plane.
    ///
    /// One block per `(channel, batch)` plane via a 2D grid; threads stride
    /// over the spatial dimension. The four per-channel constants
    /// (`weight[c]`, `bias[c]`, `running_mean[c]`, `running_var[c]`) are L1-cached
    /// since every thread in the block reads the same address — no shared
    /// memory needed.
    ///
    /// `BatchNorm2d` modules with the standard `[channels, h, w]` layout pass
    /// `batch = 1` and `spatial = h*w`.
    ///
    /// **Kernel signature:** `batchnorm_inference(const float* input, const float* weight, const float* bias, const float* running_mean, const float* running_var, float* out, unsigned int channels, unsigned int spatial, float eps)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[channels, batch, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const BATCHNORM_INFERENCE_CUDA: &str = "\
extern \"C\" __global__ void batchnorm_inference(
    const float* input, const float* weight, const float* bias,
    const float* running_mean, const float* running_var,
    float* out,
    unsigned int channels, unsigned int spatial, float eps
) {
    unsigned int c = blockIdx.x;
    unsigned int n = blockIdx.y;
    if (c >= channels) return;

    // All threads in this block read the same per-channel constants — the
    // four loads are broadcast and L1-cached, so the kernel runs at memory
    // bandwidth on the input/output streams.
    float w = weight[c];
    float b = bias[c];
    float m = running_mean[c];
    float v = running_var[c];
    float scale = w * rsqrtf(v + eps);
    float shift = b - m * scale;

    unsigned int plane = (n * channels + c) * spatial;
    const float* in_plane = input + plane;
    float* out_plane = out + plane;
    unsigned int tid = threadIdx.x;
    unsigned int bs = blockDim.x;

    for (unsigned int i = tid; i < spatial; i += bs) {
        out_plane[i] = in_plane[i] * scale + shift;
    }
}";

    /// 2D max-pooling over a `[channels, h_in, w_in]` input with fixed kernel
    /// size, stride, and zero padding (PyTorch / ResNet semantics — windows
    /// that would draw exclusively from padding produce 0.0 rather than
    /// `-inf`).
    ///
    /// One thread per output element across the full `(channel, oh, ow)`
    /// grid; each thread scans the `kh*kw` window with bounds checks. The
    /// per-channel input plane stays in L2 for ResNet-class layers, so the
    /// kernel runs memory-bound.
    ///
    /// **Kernel signature:** `maxpool_2d(const float* input, float* out,
    /// unsigned int channels, unsigned int h_in, unsigned int w_in,
    /// unsigned int kh, unsigned int kw,
    /// unsigned int stride, unsigned int padding,
    /// unsigned int h_out, unsigned int w_out)`
    /// **Block size:** `(256, 1, 1)` — dispatch
    ///   `[(channels*h_out*w_out).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const MAXPOOL_2D_CUDA: &str = "\
extern \"C\" __global__ void maxpool_2d(
    const float* input, float* out,
    unsigned int channels, unsigned int h_in, unsigned int w_in,
    unsigned int kh, unsigned int kw,
    unsigned int stride, unsigned int padding,
    unsigned int h_out, unsigned int w_out
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int spatial_out = h_out * w_out;
    unsigned int total = channels * spatial_out;
    if (idx >= total) return;

    unsigned int c = idx / spatial_out;
    unsigned int rem = idx - c * spatial_out;
    unsigned int oh = rem / w_out;
    unsigned int ow = rem - oh * w_out;

    const float* plane = input + c * h_in * w_in;
    float m = -3.402823466e38f;
    bool any = false;
    for (unsigned int khi = 0; khi < kh; khi++) {
        int ih = (int)(oh * stride + khi) - (int)padding;
        if (ih < 0 || ih >= (int)h_in) continue;
        for (unsigned int kwi = 0; kwi < kw; kwi++) {
            int iw = (int)(ow * stride + kwi) - (int)padding;
            if (iw < 0 || iw >= (int)w_in) continue;
            float v = plane[(unsigned int)ih * w_in + (unsigned int)iw];
            if (!any || v > m) m = v;
            any = true;
        }
    }
    out[idx] = any ? m : 0.0f;
}";

    /// Adaptive 2D average pooling: `[channels, h_in, w_in]` →
    /// `[channels, h_out, w_out]` with per-output regions
    /// `h_start = oh*h_in/h_out`, `h_end = (oh+1)*h_in/h_out` (and likewise
    /// for w). Matches PyTorch's `AdaptiveAvgPool2d` integer-rounded
    /// regions; global average pooling is the `h_out=w_out=1` special
    /// case.
    ///
    /// One thread per output element. For global pooling each thread
    /// reduces `h_in*w_in` inputs serially — fine for ResNet-class
    /// channels (≤2048) where the SM count gives plenty of parallelism;
    /// no shared-memory reduction needed.
    ///
    /// **Kernel signature:** `adaptive_avg_pool_2d(const float* input,
    /// float* out, unsigned int channels, unsigned int h_in,
    /// unsigned int w_in, unsigned int h_out, unsigned int w_out)`
    /// **Block size:** `(256, 1, 1)` — dispatch
    ///   `[(channels*h_out*w_out).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const ADAPTIVE_AVG_POOL_2D_CUDA: &str = "\
extern \"C\" __global__ void adaptive_avg_pool_2d(
    const float* input, float* out,
    unsigned int channels, unsigned int h_in, unsigned int w_in,
    unsigned int h_out, unsigned int w_out
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int spatial_out = h_out * w_out;
    unsigned int total = channels * spatial_out;
    if (idx >= total) return;

    unsigned int c = idx / spatial_out;
    unsigned int rem = idx - c * spatial_out;
    unsigned int oh = rem / w_out;
    unsigned int ow = rem - oh * w_out;

    unsigned int h_start = oh * h_in / h_out;
    unsigned int h_end = (oh + 1) * h_in / h_out;
    unsigned int w_start = ow * w_in / w_out;
    unsigned int w_end = (ow + 1) * w_in / w_out;

    const float* plane = input + c * h_in * w_in;
    float sum = 0.0f;
    for (unsigned int h = h_start; h < h_end; h++) {
        for (unsigned int w = w_start; w < w_end; w++) {
            sum += plane[h * w_in + w];
        }
    }
    unsigned int count = (h_end - h_start) * (w_end - w_start);
    out[idx] = sum / (float)count;
}";

    /// Group normalization with per-channel affine, inference path.
    ///
    /// For input shaped `[batch, channels, spatial]` (where `spatial = H*W`)
    /// with `channels` evenly divisible by `num_groups` (so
    /// `channels_per_group = channels / num_groups`), normalizes each
    /// `(batch, group)` plane independently:
    ///
    /// ```text
    /// mean[n, g]  = sum  over c in [g*cpg, (g+1)*cpg), i in [0, spatial) { in[n, c, i] } / (cpg * spatial)
    /// var[n, g]   = sum  over c in [g*cpg, (g+1)*cpg), i in [0, spatial) { (in[n, c, i] - mean[n, g])^2 } / (cpg * spatial)
    /// out[n, c, i] = ((in[n, c, i] - mean[n, g_of_c]) * rsqrt(var[n, g_of_c] + eps)) * weight[c] + bias[c]
    /// ```
    ///
    /// where `g_of_c = c / cpg`. Stable Diffusion uses `num_groups = 32`
    /// everywhere it appears (`UNet` `ResBlocks` + `VAE` decoder).
    ///
    /// 2D grid `[num_groups, batch, 1]` — one block per `(batch, group)`
    /// plane, 256 threads. Two block-wide reductions (sum → mean,
    /// sum-of-squared-deviations → variance) sharing one 256-float static
    /// shared-memory scratchpad. The per-block work is
    /// `cpg * spatial` elements; threads stride over them.
    ///
    /// **Kernel signature:** `group_norm(const float* input,
    /// const float* weight, const float* bias, float* out,
    /// unsigned int channels, unsigned int spatial,
    /// unsigned int num_groups, float eps)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[num_groups, batch, 1]` blocks.
    /// **Shared memory:** static, 256 floats (1 KiB).
    #[cfg(feature = "cuda")]
    pub const GROUP_NORM_CUDA: &str = "\
extern \"C\" __global__ void group_norm(
    const float* input, const float* weight, const float* bias, float* out,
    unsigned int channels, unsigned int spatial,
    unsigned int num_groups, float eps
) {
    __shared__ float smem[256];

    unsigned int g = blockIdx.x;
    unsigned int n = blockIdx.y;
    if (g >= num_groups) return;

    unsigned int cpg = channels / num_groups;
    unsigned int gsize = cpg * spatial;
    unsigned int c_start = g * cpg;
    // Base offset of this (n, group)'s first element in the flat
    // [batch, channels, spatial] input.
    unsigned int base = (n * channels + c_start) * spatial;
    const float* in_block = input + base;
    float* out_block = out + base;

    unsigned int tid = threadIdx.x;
    unsigned int bs = blockDim.x;

    // Pass 1: per-thread partial sum, block-wide reduction → mean.
    float local_sum = 0.0f;
    for (unsigned int j = tid; j < gsize; j += bs) {
        local_sum += in_block[j];
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float inv_g = 1.0f / (float)gsize;
    float mean = smem[0] * inv_g;
    __syncthreads();

    // Pass 2: per-thread partial sum of squared deviations, block-reduce → var.
    float local_sq = 0.0f;
    for (unsigned int j = tid; j < gsize; j += bs) {
        float diff = in_block[j] - mean;
        local_sq += diff * diff;
    }
    smem[tid] = local_sq;
    __syncthreads();
    for (unsigned int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    float var = smem[0] * inv_g;
    float rstd = rsqrtf(var + eps);

    // Pass 3: normalize, scale, shift. weight[c] / bias[c] vary across the
    // group, so each thread looks them up by channel index. Both lookups
    // are L1-cached: cpg consecutive channels share `weight`/`bias` reads
    // across many threads in the same block.
    for (unsigned int j = tid; j < gsize; j += bs) {
        unsigned int local_c = j / spatial;
        unsigned int c = c_start + local_c;
        float norm = (in_block[j] - mean) * rstd;
        out_block[j] = norm * weight[c] + bias[c];
    }
}";

    /// 2D nearest-neighbor upsample by an integer factor, NCHW layout.
    ///
    /// For input `[channels, h_in, w_in]` and integer `scale`, writes
    /// `[channels, h_in*scale, w_in*scale]` where
    /// `out[c, oh, ow] = in[c, oh/scale, ow/scale]` (integer divide). Used in
    /// SD `UNet` `UpBlocks` and the `VAE` decoder. `PyTorch`'s
    /// `F.interpolate(mode="nearest")` and `nn.Upsample(mode="nearest")`
    /// match this byte-for-byte.
    ///
    /// One thread per output element; each thread does one integer-divide
    /// per axis + one load + one store. The input plane stays in L2 for
    /// SD-class layers (largest is 1280 × 64 × 64 ≈ 5 MiB), so the kernel
    /// runs memory-bound.
    ///
    /// **Kernel signature:** `upsample_2d_nearest(const float* input,
    /// float* out, unsigned int channels, unsigned int h_in,
    /// unsigned int w_in, unsigned int scale)`
    /// **Block size:** `(256, 1, 1)` — dispatch
    ///   `[(channels*h_in*scale*w_in*scale).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const UPSAMPLE_2D_NEAREST_CUDA: &str = "\
extern \"C\" __global__ void upsample_2d_nearest(
    const float* input, float* out,
    unsigned int channels, unsigned int h_in, unsigned int w_in,
    unsigned int scale
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int h_out = h_in * scale;
    unsigned int w_out = w_in * scale;
    unsigned int spatial_out = h_out * w_out;
    unsigned int total = channels * spatial_out;
    if (idx >= total) return;

    unsigned int c = idx / spatial_out;
    unsigned int rem = idx - c * spatial_out;
    unsigned int oh = rem / w_out;
    unsigned int ow = rem - oh * w_out;

    unsigned int ih = oh / scale;
    unsigned int iw = ow / scale;
    out[idx] = input[(c * h_in + ih) * w_in + iw];
}";

    /// 2D nearest-neighbor upsample by an integer factor, NCHW layout.
    ///
    /// WGSL equivalent of [`UPSAMPLE_2D_NEAREST_CUDA`]. Currently unused by
    /// the dispatcher (CUDA-first per project memory) but compiles for
    /// future Vulkan-path use.
    ///
    /// **Push constants:** `struct Dims { channels: u32, h_in: u32, w_in: u32, scale: u32 }` (16 bytes)
    /// **Workgroup size:** 256
    pub const UPSAMPLE_2D_NEAREST: &str = "\
struct Dims { channels: u32, h_in: u32, w_in: u32, scale: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let h_out = dims.h_in * dims.scale;
    let w_out = dims.w_in * dims.scale;
    let spatial_out = h_out * w_out;
    let total = dims.channels * spatial_out;
    if idx >= total { return; }
    let c = idx / spatial_out;
    let rem = idx - c * spatial_out;
    let oh = rem / w_out;
    let ow = rem - oh * w_out;
    let ih = oh / dims.scale;
    let iw = ow / dims.scale;
    out[idx] = input[(c * dims.h_in + ih) * dims.w_in + iw];
}";

    /// f32 → bf16 elementwise cast: `out[i] = (bf16) in[i]` with RNE rounding.
    ///
    /// bf16 is the high 16 bits of fp32 with round-to-nearest-even — no
    /// header required. We avoid `#include <cuda_bf16.h>` because NVRTC's
    /// default include path does not cover CUDA toolkit headers, and the
    /// cast is a one-liner in raw bits anyway. The buffer type on the device
    /// side is `unsigned short` (16 bits, matching `half::bf16`'s layout).
    ///
    /// RNE bias: `+0x7FFF + (mantissa_lsb_after_truncate ? 1 : 0)`. NaN
    /// inputs propagate as bf16 NaN provided the high mantissa bits are set
    /// (which they are for canonical fp32 NaNs); subnormal NaNs are not
    /// handled specially. Activation tensors should never carry NaN/Inf in
    /// practice, so the naive form suffices.
    ///
    /// Gather a contiguous column range from a `[rows, total_cols]` matrix.
    ///
    /// `out[r, c] = input[r, col_start + c]` for `r ∈ [0, rows)`,
    /// `c ∈ [0, col_count)`. One thread per output element. Used by
    /// `MathBackend::gather_columns`, which the per-head transformer
    /// attention loop calls 3× per head — at SD 1.5 CLIP that's 432
    /// calls per text-encode and the trait default round-trips the full
    /// `[77, 2304]` qkv tensor through host on every one.
    ///
    /// **Kernel signature:** `gather_columns(const float* input, float* out, unsigned int rows, unsigned int total_cols, unsigned int col_start, unsigned int col_count)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[(rows*col_count).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const GATHER_COLUMNS_CUDA: &str = "\
extern \"C\" __global__ void gather_columns(
    const float* input, float* out,
    unsigned int rows, unsigned int total_cols,
    unsigned int col_start, unsigned int col_count
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * col_count;
    if (idx >= total) return;
    unsigned int r = idx / col_count;
    unsigned int c = idx - r * col_count;
    out[idx] = input[r * total_cols + col_start + c];
}";

    /// Additive scatter: write `src[r, c]` into `dst[r, col_start + c]`,
    /// adding to whatever's already there.
    ///
    /// `dst[r, col_start + c] += src[r, c]` for `r ∈ [0, rows)`,
    /// `c ∈ [0, col_count)`. One thread per source element. The kernel
    /// accumulates into `dst` rather than overwriting because the
    /// `MathBackend::scatter_columns` contract is additive (so the same
    /// destination row can be the target of multiple per-head writes).
    ///
    /// Caller is expected to have zeroed `dst` if a fresh accumulator is
    /// wanted — same convention as the trait default.
    ///
    /// **Kernel signature:** `scatter_columns_add(const float* src, float* dst, unsigned int rows, unsigned int total_cols, unsigned int col_start, unsigned int col_count)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[(rows*col_count).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none. Disjoint per-head column ranges → no
    /// atomics needed; each `(r, col_start + c)` written by exactly one
    /// thread per dispatch.
    #[cfg(feature = "cuda")]
    pub const SCATTER_COLUMNS_ADD_CUDA: &str = "\
extern \"C\" __global__ void scatter_columns_add(
    const float* src, float* dst,
    unsigned int rows, unsigned int total_cols,
    unsigned int col_start, unsigned int col_count
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * col_count;
    if (idx >= total) return;
    unsigned int r = idx / col_count;
    unsigned int c = idx - r * col_count;
    dst[r * total_cols + col_start + c] += src[idx];
}";

    /// Apply a strict-upper-triangle causal mask and scale to a `[seq, seq]`
    /// score matrix in-place.
    ///
    /// `out[s, t] = (t > s) ? mask_value : in[s, t] * scale`. One thread
    /// per `(s, t)` cell. The mask sentinel is typically `-INF` so the
    /// subsequent softmax row driver zeros the masked positions.
    ///
    /// Used by transformer attention's pre-softmax step. Trait default
    /// runs `to_vec` → CPU loop → `from_vec` per call — at SD 1.5 CLIP
    /// that's 144 launches per encode, each round-tripping the small
    /// `[77, 77]` score block through host.
    ///
    /// **Kernel signature:** `apply_causal_mask_and_scale(float* scores, unsigned int seq, float scale, float mask_value)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[(seq*seq).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const APPLY_CAUSAL_MASK_AND_SCALE_CUDA: &str = "\
extern \"C\" __global__ void apply_causal_mask_and_scale(
    float* scores,
    unsigned int seq, float scale, float mask_value
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = seq * seq;
    if (idx >= total) return;
    unsigned int s = idx / seq;
    unsigned int t = idx - s * seq;
    if (t > s) {
        scores[idx] = mask_value;
    } else {
        scores[idx] = scores[idx] * scale;
    }
}";

    /// Embedding lookup: gather rows of `weight[vocab, dim]` by index.
    ///
    /// `out[i, d] = weight[indices[i], d]` for `i ∈ [0, n_indices)`,
    /// `d ∈ [0, dim)`. One thread per output element; each thread does one
    /// load from the index buffer (broadcast across a warp when threads share
    /// `i`) and one strided gather from the weight table. Replaces the
    /// `ScryGpuBackend::embedding` trait override that downloaded the entire
    /// device-resident weight table to host before doing the gather on CPU
    /// — at SD 1.5 CLIP that's a ~145 MiB round-trip per text-encode.
    ///
    /// No bounds-check on `indices[i] < vocab` — caller validates (the
    /// trait-level [`crate::backend::MathBackend::embedding`] contract).
    ///
    /// **Kernel signature:** `embedding_fwd(const float* weight, const unsigned int* indices, float* out, unsigned int n_indices, unsigned int dim)`
    /// **Block size:** `(256, 1, 1)` — dispatch `[(n_indices*dim).div_ceil(256), 1, 1]` blocks.
    /// **Shared memory:** none.
    #[cfg(feature = "cuda")]
    pub const EMBEDDING_FWD_CUDA: &str = "\
extern \"C\" __global__ void embedding_fwd(
    const float* weight,
    const unsigned int* indices,
    float* out,
    unsigned int n_indices, unsigned int dim
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n_indices * dim;
    if (idx >= total) return;
    unsigned int i = idx / dim;
    unsigned int d = idx - i * dim;
    unsigned int row = indices[i];
    out[idx] = weight[row * dim + d];
}";

    /// **Kernel signature:** `cast_f32_bf16(const float* input, unsigned short* out, unsigned int N)`
    /// **Block size:** `(256, 1, 1)` — dispatch `ceil(N / 256)` blocks.
    #[cfg(feature = "bf16")]
    pub const CAST_F32_BF16_CUDA: &str = "\
extern \"C\" __global__ void cast_f32_bf16(
    const float* input, unsigned short* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned int u = __float_as_uint(input[i]);
    unsigned int lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    out[i] = (unsigned short)(u >> 16);
}";

    /// bf16 → f32 elementwise cast: `out[i] = (float) in[i]`.
    ///
    /// Lossless: bf16 is a strict subset of fp32, so we just shift the 16
    /// bits up into the high half. No header needed.
    ///
    /// **Kernel signature:** `cast_bf16_f32(const unsigned short* input, float* out, unsigned int N)`
    /// **Block size:** `(256, 1, 1)` — dispatch `ceil(N / 256)` blocks.
    #[cfg(feature = "bf16")]
    pub const CAST_BF16_F32_CUDA: &str = "\
extern \"C\" __global__ void cast_bf16_f32(
    const unsigned short* input, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = __uint_as_float(((unsigned int)input[i]) << 16);
}";
}

/// Backward activation and utility shaders for backpropagation.
///
/// All shaders use workgroup size 256 (1D) and follow the same dispatch
/// pattern as the [`elementwise`] forward shaders.
pub mod backward {
    /// `ReLU` backward: `out[i] = grad[i] * (z[i] > 0 ? 1 : 0)`.
    ///
    /// Uses the pre-activation value `z` (not the activated output).
    ///
    /// **Push constants:** `struct Dims { N: u32 }` (4 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `grad: array<f32>` (read) — upstream gradient
    ///   - `@binding(1)` `z: array<f32>` (read) — pre-activation values
    ///   - `@binding(2)` `out: array<f32>` (`read_write`) — output gradient
    pub const RELU_BACKWARD: &str = "\
struct Dims { N: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read> z: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    out[i] = select(0.0, grad[i], z[i] > 0.0);
}";

    /// CUDA C equivalent of [`RELU_BACKWARD`].
    #[cfg(feature = "cuda")]
    pub const RELU_BACKWARD_CUDA: &str = "\
extern \"C\" __global__ void relu_backward(
    const float* grad, const float* z, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = z[i] > 0.0f ? grad[i] : 0.0f;
}";

    /// Sigmoid backward: `out[i] = grad[i] * a[i] * (1 - a[i])`.
    ///
    /// Uses the post-activation value `a = sigmoid(z)`.
    ///
    /// **Push constants:** `struct Dims { N: u32 }` (4 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `grad: array<f32>` (read) — upstream gradient
    ///   - `@binding(1)` `activated: array<f32>` (read) — post-activation values
    ///   - `@binding(2)` `out: array<f32>` (`read_write`) — output gradient
    pub const SIGMOID_BACKWARD: &str = "\
struct Dims { N: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read> activated: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    let a = activated[i];
    out[i] = grad[i] * a * (1.0 - a);
}";

    /// CUDA C equivalent of [`SIGMOID_BACKWARD`].
    #[cfg(feature = "cuda")]
    pub const SIGMOID_BACKWARD_CUDA: &str = "\
extern \"C\" __global__ void sigmoid_backward(
    const float* grad, const float* activated, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float a = activated[i];
    out[i] = grad[i] * a * (1.0f - a);
}";

    /// Tanh backward: `out[i] = grad[i] * (1 - a[i]^2)`.
    ///
    /// Uses the post-activation value `a = tanh(z)`.
    ///
    /// **Push constants:** `struct Dims { N: u32 }` (4 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `grad: array<f32>` (read) — upstream gradient
    ///   - `@binding(1)` `activated: array<f32>` (read) — post-activation values
    ///   - `@binding(2)` `out: array<f32>` (`read_write`) — output gradient
    pub const TANH_BACKWARD: &str = "\
struct Dims { N: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read> activated: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    let a = activated[i];
    out[i] = grad[i] * (1.0 - a * a);
}";

    /// CUDA C equivalent of [`TANH_BACKWARD`].
    #[cfg(feature = "cuda")]
    pub const TANH_BACKWARD_CUDA: &str = "\
extern \"C\" __global__ void tanh_backward(
    const float* grad, const float* activated, float* out,
    unsigned int N
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float a = activated[i];
    out[i] = grad[i] * (1.0f - a * a);
}";

    /// Matrix transpose: `out[col * rows + row] = in[row * cols + col]`.
    ///
    /// Transposes a row-major `[rows, cols]` matrix to `[cols, rows]`.
    /// Each thread handles one element.
    ///
    /// **Push constants:** `struct Dims { rows: u32, cols: u32 }` (8 bytes)
    /// **Workgroup size:** 256 — dispatch `rows * cols` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `input: array<f32>` (read)
    ///   - `@binding(1)` `out: array<f32>` (`read_write`)
    pub const TRANSPOSE: &str = "\
struct Dims { rows: u32, cols: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = dims.rows * dims.cols;
    if i >= n { return; }
    let row = i / dims.cols;
    let col = i % dims.cols;
    out[col * dims.rows + row] = input[i];
}";

    /// CUDA C equivalent of [`TRANSPOSE`].
    #[cfg(feature = "cuda")]
    pub const TRANSPOSE_CUDA: &str = "\
extern \"C\" __global__ void transpose_2d(
    const float* input, float* out,
    unsigned int rows, unsigned int cols
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    unsigned int row = i / cols;
    unsigned int col = i % cols;
    out[col * rows + row] = input[i];
}";

    /// Element-wise scale: `out[i] = in[i] * alpha`.
    ///
    /// **Push constants:** `struct Dims { N: u32, alpha: f32 }` (8 bytes)
    /// **Workgroup size:** 256 — dispatch `N` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `input: array<f32>` (read)
    ///   - `@binding(1)` `out: array<f32>` (`read_write`)
    pub const SCALE: &str = "\
struct Dims { N: u32, alpha: f32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= dims.N { return; }
    out[i] = input[i] * dims.alpha;
}";

    /// CUDA C equivalent of [`SCALE`].
    #[cfg(feature = "cuda")]
    pub const SCALE_CUDA: &str = "\
extern \"C\" __global__ void scale_fwd(
    const float* input, float* out,
    unsigned int N, float alpha
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = input[i] * alpha;
}";

    /// Column-wise reduction: `out[j] = sum_i(in[i * cols + j]) * scale`.
    ///
    /// Sums over the row (batch) dimension for each column, then scales.
    /// Used for bias gradient computation: `db = reduce_cols(delta, 1/batch)`.
    ///
    /// **Push constants:** `struct Dims { rows: u32, cols: u32, scale: f32 }` (12 bytes)
    /// **Workgroup size:** 256 — dispatch `cols` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `input: array<f32>` (read) — `[rows, cols]` matrix
    ///   - `@binding(1)` `out: array<f32>` (`read_write`) — `[cols]` vector
    pub const REDUCE_COLS: &str = "\
struct Dims { rows: u32, cols: u32, scale: f32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= dims.cols { return; }
    var sum = 0.0;
    for (var i = 0u; i < dims.rows; i++) {
        sum += input[i * dims.cols + j];
    }
    out[j] = sum * dims.scale;
}";

    /// CUDA C equivalent of [`REDUCE_COLS`].
    #[cfg(feature = "cuda")]
    pub const REDUCE_COLS_CUDA: &str = "\
extern \"C\" __global__ void reduce_cols(
    const float* input, float* out,
    unsigned int rows, unsigned int cols, float scale
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < rows; i++) {
        sum += input[i * cols + j];
    }
    out[j] = sum * scale;
}";
}

/// Pairwise distance shaders.
pub mod distance {
    /// Pairwise squared Euclidean distance.
    ///
    /// For `n_q` query points and `n_t` training points in `dim` dimensions,
    /// computes the `n_q x n_t` distance matrix where:
    ///   `D[i][j] = sum_d (Q[i*dim+d] - T[j*dim+d])^2`
    ///
    /// Each thread computes one (query, train) pair.
    ///
    /// **Push constants:** `struct Dims { n_q: u32, n_t: u32, dim: u32 }` (12 bytes)
    /// **Workgroup size:** 256 (1D) — dispatch `n_q * n_t` invocations
    /// **Bindings:**
    ///   - `@binding(0)` `queries: array<f32>` (read)
    ///   - `@binding(1)` `train: array<f32>` (read)
    ///   - `@binding(2)` `dists: array<f32>` (`read_write`)
    pub const PAIRWISE_EUCLIDEAN: &str = "\
struct Dims { n_q: u32, n_t: u32, dim: u32 }
var<push_constant> dims: Dims;

@group(0) @binding(0) var<storage, read> queries: array<f32>;
@group(0) @binding(1) var<storage, read> train: array<f32>;
@group(0) @binding(2) var<storage, read_write> dists: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.n_q * dims.n_t;
    if (idx >= total) {
        return;
    }

    let i = idx / dims.n_t;
    let j = idx % dims.n_t;

    var sum: f32 = 0.0;
    let q_base = i * dims.dim;
    let t_base = j * dims.dim;

    for (var d: u32 = 0u; d < dims.dim; d = d + 1u) {
        let diff = queries[q_base + d] - train[t_base + d];
        sum = sum + diff * diff;
    }

    dists[idx] = sum;
}";

    /// CUDA C equivalent of [`PAIRWISE_EUCLIDEAN`].
    ///
    /// **Kernel signature:** `pairwise_euclidean(const float* queries, const float* train, float* dists, unsigned int n_q, unsigned int n_t, unsigned int dim)`
    /// **Block size:** `(256, 1, 1)` — dispatch `n_q * n_t` invocations
    #[cfg(feature = "cuda")]
    pub const PAIRWISE_EUCLIDEAN_CUDA: &str = "\
extern \"C\" __global__ void pairwise_euclidean(
    const float* queries, const float* train, float* dists,
    unsigned int n_q, unsigned int n_t, unsigned int dim
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = n_q * n_t;
    if (idx >= total) return;

    unsigned int i = idx / n_t;
    unsigned int j = idx % n_t;

    float sum = 0.0f;
    unsigned int q_base = i * dim;
    unsigned int t_base = j * dim;

    for (unsigned int d = 0; d < dim; d++) {
        float diff = queries[q_base + d] - train[t_base + d];
        sum += diff * diff;
    }

    dists[idx] = sum;
}";
}
