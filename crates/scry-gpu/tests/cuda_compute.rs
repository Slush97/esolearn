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

// ── cuBLAS GemmEx (bf16 / fp32-accumulate) ──

#[cfg(feature = "bf16")]
#[test]
fn cuda_cublas_matmul_bf16_identity() {
    use half::bf16;

    let gpu = cuda_gpu();

    // 2x2 identity: I × A = A
    let identity_f = [1.0f32, 0.0, 0.0, 1.0];
    let a_f = [5.0f32, 6.0, 7.0, 8.0];
    let identity: Vec<bf16> = identity_f.iter().map(|x| bf16::from_f32(*x)).collect();
    let a: Vec<bf16> = a_f.iter().map(|x| bf16::from_f32(*x)).collect();

    let id_buf = gpu.upload(&identity).unwrap();
    let a_buf = gpu.upload(&a).unwrap();
    let mut c_buf = gpu.alloc::<bf16>(4).unwrap();

    gpu.cublas_matmul_bf16(&id_buf, &a_buf, &mut c_buf, 2, 2, 2)
        .unwrap();

    let result: Vec<bf16> = c_buf.download().unwrap();
    let result_f: Vec<f32> = result.iter().copied().map(half::bf16::to_f32).collect();
    // bf16 represents these small integers exactly, so check equality.
    assert_eq!(result_f, vec![5.0, 6.0, 7.0, 8.0]);
}

#[cfg(feature = "bf16")]
#[test]
fn cuda_cublas_matmul_bf16_2x3_times_3x2() {
    use half::bf16;

    let gpu = cuda_gpu();

    // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
    // B = [[7, 8], [9, 10], [11, 12]]  (3x2)
    // C = A * B = [[58, 64], [139, 154]]  (2x2) — all exact in bf16.
    let a_f = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_f = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let a: Vec<bf16> = a_f.iter().map(|x| bf16::from_f32(*x)).collect();
    let b: Vec<bf16> = b_f.iter().map(|x| bf16::from_f32(*x)).collect();

    let a_buf = gpu.upload(&a).unwrap();
    let b_buf = gpu.upload(&b).unwrap();
    let mut c_buf = gpu.alloc::<bf16>(4).unwrap();

    gpu.cublas_matmul_bf16(&a_buf, &b_buf, &mut c_buf, 2, 2, 3)
        .unwrap();

    let result: Vec<bf16> = c_buf.download().unwrap();
    let result_f: Vec<f32> = result.iter().copied().map(half::bf16::to_f32).collect();
    assert_eq!(result_f, vec![58.0, 64.0, 139.0, 154.0]);
}

#[cfg(feature = "bf16")]
#[test]
fn cuda_cublas_matmul_bf16_64x64_vs_fp32_reference() {
    use half::bf16;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let gpu = cuda_gpu();

    let m = 64u32;
    let k = 64u32;
    let n = 64u32;

    // Bound inputs in [-1, 1] so the bf16 7-bit mantissa carries the rounding
    // error and we don't need to model fp32-accumulate dominance over a wide
    // exponent range.
    let mut rng = StdRng::seed_from_u64(0xb_f16_d_15);
    let a_f: Vec<f32> = (0..(m * k)).map(|_| rng.random_range(-1.0..1.0)).collect();
    let b_f: Vec<f32> = (0..(k * n)).map(|_| rng.random_range(-1.0..1.0)).collect();

    // bf16 inputs round once before the GEMM.
    let a_bf: Vec<bf16> = a_f.iter().map(|x| bf16::from_f32(*x)).collect();
    let b_bf: Vec<bf16> = b_f.iter().map(|x| bf16::from_f32(*x)).collect();

    // fp32 reference is computed against the *bf16-rounded* inputs so the
    // tolerance only has to absorb the per-multiply rounding error inside
    // the GEMM, not the input-cast error.
    let a_ref: Vec<f32> = a_bf.iter().copied().map(half::bf16::to_f32).collect();
    let b_ref: Vec<f32> = b_bf.iter().copied().map(half::bf16::to_f32).collect();
    let mut expected = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut sum = 0.0f32;
            for kk in 0..k as usize {
                sum += a_ref[i * k as usize + kk] * b_ref[kk * n as usize + j];
            }
            expected[i * n as usize + j] = sum;
        }
    }

    let a_buf = gpu.upload(&a_bf).unwrap();
    let b_buf = gpu.upload(&b_bf).unwrap();
    let mut c_buf = gpu.alloc::<bf16>((m * n) as usize).unwrap();

    gpu.cublas_matmul_bf16(&a_buf, &b_buf, &mut c_buf, m, n, k)
        .unwrap();

    let result: Vec<bf16> = c_buf.download().unwrap();
    let result_f: Vec<f32> = result.iter().copied().map(half::bf16::to_f32).collect();

    // Each output sums k=64 products of values in [-1, 1]; output magnitude
    // is typically O(sqrt(k)) ≈ 8 by random-walk argument. bf16 has ~3
    // decimal digits of mantissa; per-element absolute error scales with
    // output magnitude, so 5e-2 absolute is the realistic envelope after
    // the final cast back to bf16.
    let tol = 5e-2_f32;
    let mut max_err = 0.0_f32;
    for (i, (got, want)) in result_f.iter().zip(expected.iter()).enumerate() {
        let err = (got - want).abs();
        max_err = max_err.max(err);
        assert!(
            err < tol,
            "mismatch at index {i}: got {got}, want {want}, err {err} (tol {tol})"
        );
    }
    eprintln!("bf16 64x64 GemmEx max abs err: {max_err:.5}");
}

// ── bf16 cast kernels ──

#[cfg(feature = "bf16")]
#[test]
fn cuda_cast_f32_bf16_roundtrip() {
    use half::bf16;

    let gpu = cuda_gpu();

    // Compile both directions of the cast.
    let to_bf16 = gpu
        .compile_cuda(
            scry_gpu::shaders::elementwise::CAST_F32_BF16_CUDA,
            "cast_f32_bf16",
            2,
            [256, 1, 1],
        )
        .unwrap();
    let to_f32 = gpu
        .compile_cuda(
            scry_gpu::shaders::elementwise::CAST_BF16_F32_CUDA,
            "cast_bf16_f32",
            2,
            [256, 1, 1],
        )
        .unwrap();

    // Inputs that exercise both representable values (small ints) and ones
    // that round (1/3, π, 1e-5).
    let input = vec![
        0.0_f32,
        1.0,
        -1.0,
        2.0,
        0.5,
        -0.5,
        1.0 / 3.0,
        std::f32::consts::PI,
        1e-5,
        -1e-5,
        12345.0,
        -12345.0,
    ];
    let n = input.len() as u32;

    let in_buf = gpu.upload(&input).unwrap();
    let bf_buf = gpu.alloc::<bf16>(input.len()).unwrap();
    let out_buf = gpu.alloc::<f32>(input.len()).unwrap();

    gpu.run_with_push_constants(&to_bf16, &[&in_buf, &bf_buf], n, bytemuck::bytes_of(&n))
        .unwrap();
    gpu.run_with_push_constants(&to_f32, &[&bf_buf, &out_buf], n, bytemuck::bytes_of(&n))
        .unwrap();

    let result: Vec<f32> = out_buf.download().unwrap();
    // Reference: round each input through `bf16::from_f32` on the host. The
    // kernel must match this bit-for-bit.
    let expected: Vec<f32> = input.iter().map(|x| bf16::from_f32(*x).to_f32()).collect();
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "round-trip mismatch at index {i}: got {got}, want {want}"
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

// ── cuDNN conv2d forward ──

#[cfg(feature = "cudnn")]
#[test]
fn cuda_cudnn_conv2d_matches_cpu_reference() {
    let gpu = cuda_gpu();

    // Small ResNet-shape conv: 1×3×8×8 input, 4×3×3×3 filter, stride 1, pad 1.
    let n: u32 = 1;
    let c_in: u32 = 3;
    let h_in: u32 = 8;
    let w_in: u32 = 8;
    let c_out: u32 = 4;
    let k_h: u32 = 3;
    let k_w: u32 = 3;
    let pad: u32 = 1;
    let stride: u32 = 1;
    let h_out = (h_in + 2 * pad - k_h) / stride + 1;
    let w_out = (w_in + 2 * pad - k_w) / stride + 1;

    let input_len = (n * c_in * h_in * w_in) as usize;
    let filter_len = (c_out * c_in * k_h * k_w) as usize;
    let output_len = (n * c_out * h_out * w_out) as usize;

    // Deterministic small floats so bit-exact compare against the CPU
    // reference is practical (cuDNN's implicit-GEMM and our naive loop both
    // sum in fp32, so single rounding is identical at this size).
    let input: Vec<f32> = (0..input_len)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.5)
        .collect();
    let filter: Vec<f32> = (0..filter_len)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.25)
        .collect();

    // CPU reference: NCHW conv with zero padding, stride 1, no bias.
    let mut expected = vec![0.0f32; output_len];
    let h_in_i = h_in as i32;
    let w_in_i = w_in as i32;
    let pad_i = pad as i32;
    for oc in 0..c_out as usize {
        for oh in 0..h_out as usize {
            for ow in 0..w_out as usize {
                let mut acc = 0.0f32;
                for ic in 0..c_in as usize {
                    for kh in 0..k_h as usize {
                        for kw in 0..k_w as usize {
                            let ih = oh as i32 + kh as i32 - pad_i;
                            let iw = ow as i32 + kw as i32 - pad_i;
                            if ih >= 0 && ih < h_in_i && iw >= 0 && iw < w_in_i {
                                let i_idx = ((ic * h_in as usize) + ih as usize) * w_in as usize
                                    + iw as usize;
                                let f_idx = (((oc * c_in as usize) + ic) * k_h as usize + kh)
                                    * k_w as usize
                                    + kw;
                                acc += input[i_idx] * filter[f_idx];
                            }
                        }
                    }
                }
                let o_idx = (oc * h_out as usize + oh) * w_out as usize + ow;
                expected[o_idx] = acc;
            }
        }
    }

    let in_buf = gpu.upload(&input).unwrap();
    let filt_buf = gpu.upload(&filter).unwrap();
    let mut out_buf = gpu.alloc::<f32>(output_len).unwrap();

    let (h_got, w_got) = gpu
        .cudnn_conv2d_forward_async(
            &in_buf, &filt_buf, &mut out_buf, n, c_in, h_in, w_in, c_out, k_h, k_w, pad, pad,
            stride, stride,
        )
        .unwrap();
    assert_eq!((h_got, w_got), (h_out, w_out));

    let result: Vec<f32> = out_buf.download().unwrap();
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "mismatch at {i}: got {got}, want {want}"
        );
    }
}

#[cfg(feature = "cudnn")]
#[test]
fn cuda_cudnn_conv2d_caches_repeat_calls() {
    // Issue the same conv shape twice; both should succeed (second hits the
    // descriptor cache, no re-pick).
    let gpu = cuda_gpu();

    let n: u32 = 1;
    let c_in: u32 = 4;
    let h_in: u32 = 16;
    let w_in: u32 = 16;
    let c_out: u32 = 8;
    let k: u32 = 3;
    let pad: u32 = 1;
    let stride: u32 = 1;

    let input: Vec<f32> = (0..(c_in * h_in * w_in) as usize)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let filter: Vec<f32> = vec![0.1f32; (c_out * c_in * k * k) as usize];

    let in_buf = gpu.upload(&input).unwrap();
    let filt_buf = gpu.upload(&filter).unwrap();
    let mut out_buf = gpu
        .alloc::<f32>((c_out * h_in * w_in) as usize)
        .unwrap();

    for _ in 0..3 {
        gpu.cudnn_conv2d_forward_async(
            &in_buf, &filt_buf, &mut out_buf, n, c_in, h_in, w_in, c_out, k, k, pad, pad, stride,
            stride,
        )
        .unwrap();
    }

    let result: Vec<f32> = out_buf.download().unwrap();
    assert!(result.iter().all(|v| v.is_finite()));
    assert!(result.iter().any(|v| *v != 0.0));
}
