// Pairwise squared Euclidean distance compute shader.
//
// For n_q query points and n_t training points in `dim` dimensions,
// computes the n_q × n_t distance matrix where:
//   D[i][j] = Σ_d (Q[i*dim+d] - T[j*dim+d])²
//
// Each thread computes one (query, train) pair.

struct Dimensions {
    n_q: u32,
    n_t: u32,
    dim: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> train: array<f32>;
@group(0) @binding(3) var<storage, read_write> dists: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = dims.n_q * dims.n_t;
    if (idx >= total) {
        return;
    }

    let i = idx / dims.n_t; // query index
    let j = idx % dims.n_t; // train index

    var sum: f32 = 0.0;
    let q_base = i * dims.dim;
    let t_base = j * dims.dim;

    for (var d: u32 = 0u; d < dims.dim; d = d + 1u) {
        let diff = queries[q_base + d] - train[t_base + d];
        sum = sum + diff * diff;
    }

    dists[idx] = sum;
}
