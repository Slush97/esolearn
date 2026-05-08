// SPDX-License-Identifier: MIT OR Apache-2.0
//! End-to-end ResNet forward-pass benchmark across `CpuBackend` and
//! `ScryGpuBackend`.
//!
//! Models are zero-init (no checkpoint needed) — we only care about kernel
//! dispatch time, not classification accuracy. BatchNorm defaults
//! (`weight=1`, `bias=0`, `running_mean=0`, `running_var=1`) keep activations
//! finite so every kernel actually runs through to completion.
//!
//! Run with:
//!
//! ```text
//! CUDARC_CUDA_VERSION=13010 cargo run -p scry-vision \
//!     --features scry-gpu-cuda --example bench_gpu --release
//! ```
//!
//! The CUDA features pull `scry-gpu-cuda` through the dependency chain so
//! `ScryGpuBackend` actually engages the GPU; without them every op falls
//! back to CPU and the two columns will report identical numbers.

use std::time::Instant;

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::scry_gpu::ScryGpuBackend;
use scry_llm::backend::DeviceBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_vision::models::{ResNet, ResNetConfig};

fn time_ms<F: FnMut()>(warmup: usize, runs: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[runs / 2]
}

/// Cheap LCG so the bench doesn't pull in fastrand as a dependency.
fn random_input(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            // Map 32 high bits to [-1, 1).
            ((state >> 32) as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

fn run_resnet18(input: &[f32]) {
    println!("--- ResNet-18 (1000-class, 3×224×224) ---");

    let cpu_model = ResNet::<CpuBackend>::new(ResNetConfig::resnet18(1000));
    let cpu_input = Tensor::<CpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    let cpu_ms = time_ms(1, 3, || {
        let _ = std::hint::black_box(cpu_model.forward(&cpu_input));
    });
    println!("  CpuBackend                 : {cpu_ms:7.1} ms/image");

    // Without weight pre-upload: every forward pays ~50 host→device uploads.
    let gpu_model_lazy = ResNet::<ScryGpuBackend>::new(ResNetConfig::resnet18(1000));
    let gpu_input = Tensor::<ScryGpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    let gpu_lazy_ms = time_ms(2, 5, || {
        let _ = std::hint::black_box(gpu_model_lazy.forward(&gpu_input));
        // Persistent path uses async dispatch on CUDA — sync the stream
        // before stopping the timer so we measure GPU execution, not host
        // launch latency. Mirrors PyTorch's `torch.cuda.synchronize()`.
        ScryGpuBackend::synchronize().expect("synchronize");
    });
    println!("  ScryGpuBackend (lazy)      : {gpu_lazy_ms:7.1} ms/image  ({:.2}× CPU)", cpu_ms / gpu_lazy_ms);

    // With weight + input pre-upload: weights live in ScryGpuStorage::Gpu for
    // every forward, so as_gpu_buffer hits the Arc fast path instead of
    // re-uploading.
    let mut gpu_model = ResNet::<ScryGpuBackend>::new(ResNetConfig::resnet18(1000));
    gpu_model.to_device();
    let mut input_storage = ScryGpuBackend::from_vec(input.to_vec(), &Shape::new(&[3, 224, 224]));
    ScryGpuBackend::to_device_in_place(&mut input_storage);
    let gpu_input_resident =
        Tensor::<ScryGpuBackend>::new(input_storage, Shape::new(&[3, 224, 224]));
    let gpu_resident_ms = time_ms(2, 5, || {
        let _ = std::hint::black_box(gpu_model.forward(&gpu_input_resident));
        ScryGpuBackend::synchronize().expect("synchronize");
    });
    println!("  ScryGpuBackend (pre-upload): {gpu_resident_ms:7.1} ms/image  ({:.2}× CPU, {:.2}× lazy)",
        cpu_ms / gpu_resident_ms, gpu_lazy_ms / gpu_resident_ms);

    // Fold BN into preceding conv before upload — eliminates ~21 BN kernel
    // launches and HBM passes per forward (one per BasicBlock + stem).
    let mut gpu_model_fused = ResNet::<ScryGpuBackend>::new(ResNetConfig::resnet18(1000));
    gpu_model_fused.fuse_batchnorms();
    gpu_model_fused.to_device();
    let gpu_fused_ms = time_ms(2, 5, || {
        let _ = std::hint::black_box(gpu_model_fused.forward(&gpu_input_resident));
        ScryGpuBackend::synchronize().expect("synchronize");
    });
    println!("  ScryGpuBackend (BN-fused)  : {gpu_fused_ms:7.1} ms/image  ({:.2}× CPU, {:.2}× pre-upload)",
        cpu_ms / gpu_fused_ms, gpu_resident_ms / gpu_fused_ms);

    // The BN-fused row above was measured with cuDNN enabled by default when
    // the `scry-gpu-cudnn` feature is on. Flip it off and re-run to attribute
    // wins to cuDNN specifically vs the legacy im2col + cuBLAS chain.
    #[cfg(feature = "scry-gpu-cudnn")]
    {
        ScryGpuBackend::set_cudnn_conv(false).expect("toggle cudnn off");
        let gpu_im2col_ms = time_ms(2, 5, || {
            let _ = std::hint::black_box(gpu_model_fused.forward(&gpu_input_resident));
            ScryGpuBackend::synchronize().expect("synchronize");
        });
        ScryGpuBackend::set_cudnn_conv(true).expect("toggle cudnn back on");
        println!(
            "  ScryGpuBackend (im2col conv): {gpu_im2col_ms:7.1} ms/image  ({:.2}× CPU, baseline)",
            cpu_ms / gpu_im2col_ms,
        );
        println!(
            "  ScryGpuBackend (cuDNN conv) : {gpu_fused_ms:7.1} ms/image  ({:.2}× CPU, {:.2}× im2col)",
            cpu_ms / gpu_fused_ms,
            gpu_im2col_ms / gpu_fused_ms,
        );
    }

    #[cfg(feature = "scry-gpu-bf16")]
    {
        // bf16 / fp32-accumulate matmul via cuBLAS GemmEx (tensor cores).
        // Same fused, pre-uploaded model — only the matmul routing changes.
        // cuDNN runs in fp32 either way; this row stacks bf16 on top of
        // whichever conv path is currently active (cuDNN if the feature is on).
        ScryGpuBackend::set_bf16_matmul(true).expect("toggle bf16");
        let gpu_bf16_ms = time_ms(2, 5, || {
            let _ = std::hint::black_box(gpu_model_fused.forward(&gpu_input_resident));
            ScryGpuBackend::synchronize().expect("synchronize");
        });
        ScryGpuBackend::set_bf16_matmul(false).expect("toggle bf16");
        println!(
            "  ScryGpuBackend (bf16 matmul): {gpu_bf16_ms:7.1} ms/image  ({:.2}× CPU, {:.2}× BN-fused)",
            cpu_ms / gpu_bf16_ms,
            gpu_fused_ms / gpu_bf16_ms
        );
    }
}

fn run_resnet50(input: &[f32]) {
    println!("\n--- ResNet-50 (1000-class, 3×224×224) ---");

    let cpu_model = ResNet::<CpuBackend>::new(ResNetConfig::resnet50(1000));
    let cpu_input = Tensor::<CpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    let cpu_ms = time_ms(1, 3, || {
        let _ = std::hint::black_box(cpu_model.forward(&cpu_input));
    });
    println!("  CpuBackend                 : {cpu_ms:7.1} ms/image");

    let gpu_model_lazy = ResNet::<ScryGpuBackend>::new(ResNetConfig::resnet50(1000));
    let gpu_input = Tensor::<ScryGpuBackend>::from_vec(input.to_vec(), Shape::new(&[3, 224, 224]));
    let gpu_lazy_ms = time_ms(2, 5, || {
        let _ = std::hint::black_box(gpu_model_lazy.forward(&gpu_input));
        ScryGpuBackend::synchronize().expect("synchronize");
    });
    println!("  ScryGpuBackend (lazy)      : {gpu_lazy_ms:7.1} ms/image  ({:.2}× CPU)", cpu_ms / gpu_lazy_ms);

    let mut gpu_model = ResNet::<ScryGpuBackend>::new(ResNetConfig::resnet50(1000));
    gpu_model.to_device();
    let mut input_storage = ScryGpuBackend::from_vec(input.to_vec(), &Shape::new(&[3, 224, 224]));
    ScryGpuBackend::to_device_in_place(&mut input_storage);
    let gpu_input_resident =
        Tensor::<ScryGpuBackend>::new(input_storage, Shape::new(&[3, 224, 224]));
    let gpu_resident_ms = time_ms(2, 5, || {
        let _ = std::hint::black_box(gpu_model.forward(&gpu_input_resident));
        ScryGpuBackend::synchronize().expect("synchronize");
    });
    println!("  ScryGpuBackend (pre-upload): {gpu_resident_ms:7.1} ms/image  ({:.2}× CPU, {:.2}× lazy)",
        cpu_ms / gpu_resident_ms, gpu_lazy_ms / gpu_resident_ms);

    // Fold BN into preceding conv before upload. Bottleneck has 3 conv-bn
    // pairs per block; ResNet-50 has 16 blocks, so ~49 BN launches vanish.
    let mut gpu_model_fused = ResNet::<ScryGpuBackend>::new(ResNetConfig::resnet50(1000));
    gpu_model_fused.fuse_batchnorms();
    gpu_model_fused.to_device();
    let gpu_fused_ms = time_ms(2, 5, || {
        let _ = std::hint::black_box(gpu_model_fused.forward(&gpu_input_resident));
        ScryGpuBackend::synchronize().expect("synchronize");
    });
    println!("  ScryGpuBackend (BN-fused)  : {gpu_fused_ms:7.1} ms/image  ({:.2}× CPU, {:.2}× pre-upload)",
        cpu_ms / gpu_fused_ms, gpu_resident_ms / gpu_fused_ms);

    #[cfg(feature = "scry-gpu-cudnn")]
    {
        ScryGpuBackend::set_cudnn_conv(false).expect("toggle cudnn off");
        let gpu_im2col_ms = time_ms(2, 5, || {
            let _ = std::hint::black_box(gpu_model_fused.forward(&gpu_input_resident));
            ScryGpuBackend::synchronize().expect("synchronize");
        });
        ScryGpuBackend::set_cudnn_conv(true).expect("toggle cudnn back on");
        println!(
            "  ScryGpuBackend (im2col conv): {gpu_im2col_ms:7.1} ms/image  ({:.2}× CPU, baseline)",
            cpu_ms / gpu_im2col_ms,
        );
        println!(
            "  ScryGpuBackend (cuDNN conv) : {gpu_fused_ms:7.1} ms/image  ({:.2}× CPU, {:.2}× im2col)",
            cpu_ms / gpu_fused_ms,
            gpu_im2col_ms / gpu_fused_ms,
        );
    }

    #[cfg(feature = "scry-gpu-bf16")]
    {
        ScryGpuBackend::set_bf16_matmul(true).expect("toggle bf16");
        let gpu_bf16_ms = time_ms(2, 5, || {
            let _ = std::hint::black_box(gpu_model_fused.forward(&gpu_input_resident));
            ScryGpuBackend::synchronize().expect("synchronize");
        });
        ScryGpuBackend::set_bf16_matmul(false).expect("toggle bf16");
        println!(
            "  ScryGpuBackend (bf16 matmul): {gpu_bf16_ms:7.1} ms/image  ({:.2}× CPU, {:.2}× BN-fused)",
            cpu_ms / gpu_bf16_ms,
            gpu_fused_ms / gpu_bf16_ms
        );
    }
}

fn main() {
    println!("=== scry-vision end-to-end ResNet bench ===");
    println!("(zero-init weights — kernel dispatch timing only)\n");

    let scry_gpu_cuda = cfg!(feature = "scry-gpu-cuda");
    let scry_gpu_bf16 = cfg!(feature = "scry-gpu-bf16");
    let scry_gpu_cudnn = cfg!(feature = "scry-gpu-cudnn");
    println!("Features: scry-gpu-cuda={scry_gpu_cuda}, scry-gpu-bf16={scry_gpu_bf16}, scry-gpu-cudnn={scry_gpu_cudnn}");
    println!(
        "Threads:  RAYON_NUM_THREADS={}",
        rayon::current_num_threads()
    );
    println!();

    let input = random_input(0xdead_beef, 3 * 224 * 224);

    run_resnet18(&input);
    run_resnet50(&input);
}
