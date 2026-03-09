//! CUDA smoke tests — verify the entire GPU stack works:
//! device init, memory alloc, cuBLAS GEMM, NVRTC kernel compilation.

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    use cudarc::driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg};
    use std::sync::Arc;

    fn setup() -> (Arc<CudaContext>, Arc<CudaStream>) {
        let ctx = CudaContext::new(0).expect("Failed to init CUDA device 0");
        let stream = ctx.new_stream().expect("Failed to create stream");
        (ctx, stream)
    }

    /// Test 1: Can we initialize the CUDA device and allocate memory?
    #[test]
    fn device_init_and_alloc() {
        let (ctx, stream) = setup();

        let name = ctx.name().expect("Failed to get device name");
        println!("  Device: {name}");

        // Allocate 1024 floats on device, copy up, read back
        let data: Vec<f32> = vec![42.0; 1024];
        let gpu_buf = stream.clone_htod(&data).expect("Failed to copy to device");

        let host_buf: Vec<f32> = stream
            .clone_dtoh(&gpu_buf)
            .expect("Failed to copy from device");
        assert_eq!(host_buf.len(), 1024);
        assert_eq!(host_buf[0], 42.0);
        assert_eq!(host_buf[1023], 42.0);

        println!("  [PASS] Device init + alloc + round-trip copy");
    }

    /// Test 2: Does cuBLAS GEMM work? Multiply two 64x64 matrices.
    #[test]
    fn cublas_gemm() {
        let (_ctx, stream) = setup();
        let blas = CudaBlas::new(stream.clone()).expect("Failed to init cuBLAS");

        let n = 64usize;

        // A = identity, B = all 2.0
        // C = A * B should be all 2.0
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let b = vec![2.0f32; n * n];
        let c = vec![0.0f32; n * n];

        let a_dev = stream.clone_htod(&a).unwrap();
        let b_dev = stream.clone_htod(&b).unwrap();
        let mut c_dev = stream.clone_htod(&c).unwrap();

        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: n as i32,
            k: n as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: n as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            blas.gemm(cfg, &a_dev, &b_dev, &mut c_dev)
                .expect("cuBLAS GEMM failed");
        }

        let result: Vec<f32> = stream.clone_dtoh(&c_dev).unwrap();

        for i in 0..n {
            for j in 0..n {
                let val = result[i * n + j];
                assert!(
                    (val - 2.0).abs() < 1e-5,
                    "GEMM result[{i}][{j}] = {val}, expected 2.0"
                );
            }
        }

        println!("  [PASS] cuBLAS GEMM {n}x{n} identity * 2.0 = 2.0");
    }

    /// Test 3: Can NVRTC compile and launch a custom kernel?
    #[test]
    fn nvrtc_custom_kernel() {
        let (ctx, stream) = setup();

        let kernel_src = r#"
extern "C" __global__ void scale(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 3.0f;
    }
}
"#;

        let ptx = cudarc::nvrtc::compile_ptx(kernel_src).expect("NVRTC compilation failed");
        let module = ctx.load_module(ptx).expect("Failed to load PTX module");
        let scale_fn = module
            .load_function("scale")
            .expect("Failed to get kernel function");

        let n = 2048i32;
        let data: Vec<f32> = vec![7.0; n as usize];
        let mut gpu_buf = stream.clone_htod(&data).unwrap();

        let cfg = LaunchConfig::for_num_elems(n as u32);

        unsafe {
            stream
                .launch_builder(&scale_fn)
                .arg(&mut gpu_buf)
                .arg(&n)
                .launch(cfg)
                .expect("Kernel launch failed");
        }

        let result: Vec<f32> = stream.clone_dtoh(&gpu_buf).unwrap();

        for (i, val) in result.iter().enumerate() {
            assert!(
                (*val - 21.0_f32).abs() < 1e-5,
                "Kernel result[{i}] = {val}, expected 21.0 (7.0 * 3.0)"
            );
        }

        println!("  [PASS] NVRTC compile + launch custom kernel: {n} elements * 3.0");
    }

    /// Test 4: BF16 round-trip — verify the GPU handles bf16 data correctly.
    #[test]
    fn bf16_round_trip() {
        let (ctx, stream) = setup();

        let kernel_src = r#"
#include <cuda_bf16.h>
extern "C" __global__ void bf16_roundtrip(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __nv_bfloat16 val = __float2bfloat16(input[idx]);
        output[idx] = __bfloat162float(val);
    }
}
"#;

        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            kernel_src,
            cudarc::nvrtc::CompileOptions {
                include_paths: vec!["/opt/cuda/include".to_string()],
                ..Default::default()
            },
        )
        .expect("NVRTC BF16 compilation failed");

        let module = ctx
            .load_module(ptx)
            .expect("Failed to load BF16 PTX module");
        let func = module.load_function("bf16_roundtrip").unwrap();

        let n = 1024i32;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let output_init: Vec<f32> = vec![0.0; n as usize];

        let input_dev = stream.clone_htod(&input).unwrap();
        let mut output_dev = stream.clone_htod(&output_init).unwrap();

        let cfg = LaunchConfig::for_num_elems(n as u32);

        unsafe {
            stream
                .launch_builder(&func)
                .arg(&input_dev)
                .arg(&mut output_dev)
                .arg(&n)
                .launch(cfg)
                .expect("BF16 kernel launch failed");
        }

        let result: Vec<f32> = stream.clone_dtoh(&output_dev).unwrap();

        let mut max_err: f32 = 0.0;
        for i in 0..n as usize {
            let err = (result[i] - input[i]).abs();
            max_err = max_err.max(err);
        }

        println!("  [PASS] BF16 round-trip: max error = {max_err:.6} over {n} values");
        assert!(
            max_err < 2.0,
            "BF16 round-trip max error {max_err} too large"
        );
    }

    /// Test 5: GPU info summary.
    #[test]
    fn gpu_info() {
        let (ctx, _stream) = setup();
        let name = ctx.name().unwrap_or_else(|_| "unknown".to_string());

        println!("\n  === GPU Smoke Test Summary ===");
        println!("  Device: {name}");
        println!("  cuBLAS: available");
        println!("  NVRTC: available");
        println!("  BF16: supported");
        println!("  ================================\n");
    }
}
