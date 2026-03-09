// Tiled matrix multiply compute shader.
//
// C = A × B where:
//   A is M×K (row-major)
//   B is K×N (row-major)
//   C is M×N (row-major)

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

// Workgroup tile size — 16×16 threads × 16 K-steps
const TILE: u32 = 16u;

var<workgroup> tileA: array<array<f32, TILE>, TILE>;
var<workgroup> tileB: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    var sum: f32 = 0.0;
    let numTiles = (dims.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        // Load tile of A
        let a_col = t * TILE + lid.x;
        if (row < dims.M && a_col < dims.K) {
            tileA[lid.y][lid.x] = A[row * dims.K + a_col];
        } else {
            tileA[lid.y][lid.x] = 0.0;
        }

        // Load tile of B
        let b_row = t * TILE + lid.y;
        if (b_row < dims.K && col < dims.N) {
            tileB[lid.y][lid.x] = B[b_row * dims.N + col];
        } else {
            tileB[lid.y][lid.x] = 0.0;
        }

        workgroupBarrier();

        // Accumulate
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tileA[lid.y][k] * tileB[k][lid.x];
        }

        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = sum;
    }
}
