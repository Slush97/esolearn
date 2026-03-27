//! Shared WGSL shader sources for reuse across crates.
//!
//! Each constant is a complete WGSL shader string ready to pass to
//! [`Device::compile`](crate::Device::compile). Push constant layouts
//! and workgroup sizes are documented per shader.

/// Matrix multiplication shaders.
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
    ///   - `@binding(2)` `dists: array<f32>` (read_write)
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
}
