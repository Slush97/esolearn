// Histogram construction for gradient boosting.
// 1 workgroup per feature, 256 threads per workgroup (one per bin).
// Each thread owns one histogram bin and accumulates grad/hess/count
// for all samples that map to that bin.

struct Params {
    n_samples: u32,
    n_features: u32,
    n_bins: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;

// Binned features: column-major [n_features][n_samples] as u32.
@group(0) @binding(1) var<storage, read> binned: array<u32>;

// Gradients: [n_samples] as f32.
@group(0) @binding(2) var<storage, read> gradients: array<f32>;

// Hessians: [n_samples] as f32.
@group(0) @binding(3) var<storage, read> hessians: array<f32>;

// Sample indices: [n_samples] as u32 (active sample subset).
@group(0) @binding(4) var<storage, read> sample_indices: array<u32>;

// Output: [n_features][n_bins][3] as f32 (grad_sum, hess_sum, count).
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let feature_idx = gid.y;
    let bin_idx = gid.x;

    if feature_idx >= params.n_features || bin_idx >= params.n_bins {
        return;
    }

    var grad_sum: f32 = 0.0;
    var hess_sum: f32 = 0.0;
    var count: f32 = 0.0;

    let col_offset = feature_idx * params.n_samples;

    for (var s: u32 = 0u; s < params.n_samples; s = s + 1u) {
        let sample_idx = sample_indices[s];
        let bin = binned[col_offset + sample_idx];
        if bin == bin_idx {
            grad_sum = grad_sum + gradients[sample_idx];
            hess_sum = hess_sum + hessians[sample_idx];
            count = count + 1.0;
        }
    }

    let out_idx = (feature_idx * params.n_bins + bin_idx) * 3u;
    output[out_idx] = grad_sum;
    output[out_idx + 1u] = hess_sum;
    output[out_idx + 2u] = count;
}
