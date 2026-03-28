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
