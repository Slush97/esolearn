//! CUDA-specific integration tests for scry-gpu.
//!
//! These tests require an NVIDIA GPU with CUDA drivers. They are gated
//! behind `#[cfg(feature = "cuda")]` so they're skipped without CUDA.

#![cfg(feature = "cuda")]

use scry_gpu::{BackendKind, Device};

fn cuda_gpu() -> Device {
    Device::with_backend(BackendKind::Cuda).expect("no CUDA-capable GPU found — skipping test")
}

// ── Upload / download roundtrip ──

#[test]
fn cuda_upload_download_f32() {
    let gpu = cuda_gpu();

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let buf = gpu.upload(&data).unwrap();
    let result: Vec<f32> = buf.download().unwrap();
    assert_eq!(result, data);
}

#[test]
fn cuda_upload_download_u32() {
    let gpu = cuda_gpu();

    let data = vec![10u32, 20, 30, 40];
    let buf = gpu.upload(&data).unwrap();
    let result: Vec<u32> = buf.download().unwrap();
    assert_eq!(result, data);
}

#[test]
fn cuda_alloc_zeros() {
    let gpu = cuda_gpu();

    let buf = gpu.alloc::<f32>(8).unwrap();
    let result: Vec<f32> = buf.download().unwrap();
    assert_eq!(result, vec![0.0; 8]);
}

// ── Device info ──

#[test]
fn cuda_device_reports_name_and_memory() {
    let gpu = cuda_gpu();
    assert!(!gpu.name().is_empty());
    assert!(gpu.memory() > 0);
    assert_eq!(gpu.subgroup_size(), 32); // NVIDIA warp size
}

// ── Custom CUDA kernel compile + dispatch ──

#[test]
fn cuda_custom_kernel_vector_double() {
    let gpu = cuda_gpu();

    let source = r#"
extern "C" __global__ void vector_double(const float* input, float* output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * 2.0f;
    }
}
"#;

    // binding_count=2 (input + output), workgroup_size=[256,1,1]
    let kernel = gpu
        .compile_cuda(source, "vector_double", 2, [256, 1, 1])
        .unwrap();

    let input = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let output = gpu.alloc::<f32>(4).unwrap();

    // Push constants: n=4
    let n: u32 = 4;
    let push_constants = bytemuck::bytes_of(&n);
    gpu.run_with_push_constants(&kernel, &[&input, &output], 4, push_constants)
        .unwrap();

    let result: Vec<f32> = output.download().unwrap();
    assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn cuda_custom_kernel_vector_add() {
    let gpu = cuda_gpu();

    let source = r#"
extern "C" __global__ void vector_add(
    const float* a, const float* b, float* out, unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

    let kernel = gpu
        .compile_cuda(source, "vector_add", 3, [256, 1, 1])
        .unwrap();

    let a = gpu.upload(&[10.0f32, 20.0, 30.0]).unwrap();
    let b = gpu.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let out = gpu.alloc::<f32>(3).unwrap();

    let n: u32 = 3;
    gpu.run_with_push_constants(&kernel, &[&a, &b, &out], 3, bytemuck::bytes_of(&n))
        .unwrap();

    let result: Vec<f32> = out.download().unwrap();
    assert_eq!(result, vec![11.0, 22.0, 33.0]);
}

// ── cuBLAS matmul ──

#[test]
fn cuda_cublas_matmul_identity() {
    let gpu = cuda_gpu();

    // 2x2 identity: I × A = A
    let identity = gpu.upload(&[1.0f32, 0.0, 0.0, 1.0]).unwrap();
    let a = gpu.upload(&[5.0f32, 6.0, 7.0, 8.0]).unwrap();
    let mut c = gpu.alloc::<f32>(4).unwrap();

    gpu.cublas_matmul(&identity, &a, &mut c, 2, 2, 2).unwrap();

    let result: Vec<f32> = c.download().unwrap();
    assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn cuda_cublas_matmul_2x3_times_3x2() {
    let gpu = cuda_gpu();

    // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
    // B = [[7, 8], [9, 10], [11, 12]]  (3x2)
    // C = A * B = [[58, 64], [139, 154]]  (2x2)
    let a = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = gpu.upload(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let mut c = gpu.alloc::<f32>(4).unwrap();

    gpu.cublas_matmul(&a, &b, &mut c, 2, 2, 3).unwrap();

    let result: Vec<f32> = c.download().unwrap();
    assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn cuda_cublas_matmul_vs_cpu_reference() {
    let gpu = cuda_gpu();

    // Larger test: 4x4 matmul vs CPU reference
    let m = 4u32;
    let k = 4u32;
    let n = 4u32;

    let a_data: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
    let b_data: Vec<f32> = (0..16).map(|i| ((i % 4) * 3 + i / 4) as f32).collect();

    // CPU reference
    let mut expected = vec![0.0f32; 16];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut sum = 0.0f32;
            for kk in 0..k as usize {
                sum += a_data[i * k as usize + kk] * b_data[kk * n as usize + j];
            }
            expected[i * n as usize + j] = sum;
        }
    }

    let a = gpu.upload(&a_data).unwrap();
    let b = gpu.upload(&b_data).unwrap();
    let mut c = gpu.alloc::<f32>(16).unwrap();

    gpu.cublas_matmul(&a, &b, &mut c, m, n, k).unwrap();

    let result: Vec<f32> = c.download().unwrap();
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "mismatch at index {i}: got {got}, want {want}"
        );
    }
}

// ── Built-in CUDA shader: tiled matmul ──

#[test]
fn cuda_builtin_tiled_matmul_16x16() {
    let gpu = cuda_gpu();

    let source = scry_gpu::shaders::matmul::TILED_16X16_CUDA;
    let kernel = gpu
        .compile_cuda(source, "matmul_tiled_16x16", 3, [16, 16, 1])
        .unwrap();

    // 4x4 matmul
    let a = gpu
        .upload(&[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ])
        .unwrap();
    let b = gpu
        .upload(&[
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ])
        .unwrap(); // identity
    let c = gpu.alloc::<f32>(16).unwrap();

    // Push constants: M=4, N=4, K=4
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Dims {
        m: u32,
        n: u32,
        k: u32,
    }

    let dims = Dims { m: 4, n: 4, k: 4 };
    gpu.run_configured(
        &kernel,
        &[&a, &b, &c],
        [1, 1, 1], // ceil(4/16) = 1
        Some(bytemuck::bytes_of(&dims)),
    )
    .unwrap();

    let result: Vec<f32> = c.download().unwrap();
    let expected: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "mismatch at index {i}: got {got}, want {want}"
        );
    }
}

// ── Built-in CUDA shader: pairwise distance ──

#[test]
fn cuda_builtin_pairwise_euclidean() {
    let gpu = cuda_gpu();

    let source = scry_gpu::shaders::distance::PAIRWISE_EUCLIDEAN_CUDA;
    let kernel = gpu
        .compile_cuda(source, "pairwise_euclidean", 3, [256, 1, 1])
        .unwrap();

    // 2 query points, 3 training points, 2 dimensions
    let queries = gpu.upload(&[0.0f32, 0.0, 1.0, 1.0]).unwrap();
    let train = gpu.upload(&[0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
    let dists = gpu.alloc::<f32>(6).unwrap(); // 2x3

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Dims {
        n_q: u32,
        n_t: u32,
        dim: u32,
    }

    let dims = Dims {
        n_q: 2,
        n_t: 3,
        dim: 2,
    };

    let total = dims.n_q * dims.n_t;
    gpu.run_with_push_constants(
        &kernel,
        &[&queries, &train, &dists],
        total,
        bytemuck::bytes_of(&dims),
    )
    .unwrap();

    let result: Vec<f32> = dists.download().unwrap();
    // Q[0]=(0,0): dist to (0,0)=0, (1,0)=1, (0,1)=1
    // Q[1]=(1,1): dist to (0,0)=2, (1,0)=1, (0,1)=1
    let expected = [0.0f32, 1.0, 1.0, 2.0, 1.0, 1.0];
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "mismatch at index {i}: got {got}, want {want}"
        );
    }
}

// ── Batch dispatch ──

#[test]
fn cuda_batch_dispatch() {
    let gpu = cuda_gpu();

    let source = r#"
extern "C" __global__ void scale(const float* input, float* output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] * 3.0f;
    }
}
"#;

    let kernel = gpu.compile_cuda(source, "scale", 2, [256, 1, 1]).unwrap();

    let input = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let pass1 = gpu.alloc::<f32>(4).unwrap();
    let pass2 = gpu.alloc::<f32>(4).unwrap();

    let n: u32 = 4;
    let pc = bytemuck::bytes_of(&n);

    let mut batch = gpu.batch().unwrap();
    batch
        .run_with_push_constants(&kernel, &[&input, &pass1], 4, pc)
        .unwrap();
    batch.barrier();
    batch
        .run_with_push_constants(&kernel, &[&pass1, &pass2], 4, pc)
        .unwrap();
    batch.submit().unwrap();

    let result: Vec<f32> = pass2.download().unwrap();
    // 3x scaling twice: [1,2,3,4] → [3,6,9,12] → [9,18,27,36]
    assert_eq!(result, vec![9.0, 18.0, 27.0, 36.0]);
}

// ── WGSL dispatch on CUDA should fail gracefully ──

#[test]
fn cuda_wgsl_dispatch_returns_error() {
    let gpu = cuda_gpu();

    let input = gpu.upload(&[1.0f32, 2.0]).unwrap();
    let output = gpu.alloc::<f32>(2).unwrap();

    let shader = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * 2.0;
    }
}";

    let result = gpu.dispatch(shader, &[&input, &output], 2);
    assert!(result.is_err(), "WGSL dispatch on CUDA should fail");
}
