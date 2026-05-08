// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-op breakdown of `ResNet::forward` on `ScryGpuBackend`.
//!
//! Hand-rolls the same forward as `ResNet::forward` but inserts
//! `ScryGpuBackend::synchronize()` between each op so we can attribute wall
//! time to op categories. Output is a per-bucket table:
//!
//! ```text
//! conv_winograd       :   16× 14.20 ms (78.0%)  avg 0.888 ms
//! relu                :   17×  1.70 ms ( 9.3%)  avg 0.100 ms
//! ...
//! ```
//!
//! Run with:
//!
//! ```text
//! CUDARC_CUDA_VERSION=13010 cargo run -p scry-vision \
//!     --features scry-gpu-cuda --example bench_profile --release
//! ```
//!
//! Note: synchronize-between-ops adds ~30–50 µs per op of host wait, so
//! absolute times are slightly inflated vs. the un-instrumented bench. The
//! relative breakdown is still informative — we're attributing share, not
//! competing on absolute throughput.

use std::collections::HashMap;
use std::time::Instant;

use scry_llm::backend::scry_gpu::ScryGpuBackend;
use scry_llm::backend::{DeviceBackend, MathBackend};
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;
use scry_vision::models::resnet::{BasicBlock, Bottleneck, ResNetStage};
use scry_vision::models::{ResNet, ResNetConfig};
use scry_vision::nn::conv2d::Conv2d;
use scry_vision::nn::relu;

type B = ScryGpuBackend;

#[derive(Default)]
struct Profile {
    buckets: HashMap<&'static str, (usize, f64)>,
}

impl Profile {
    fn time<T, F: FnOnce() -> T>(&mut self, bucket: &'static str, f: F) -> T {
        let t0 = Instant::now();
        let result = f();
        ScryGpuBackend::synchronize().expect("sync");
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        let entry = self.buckets.entry(bucket).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += elapsed;
        result
    }

    fn print(&self, total_ms: f64, label: &str) {
        println!("\n  Per-op breakdown for {label} (total {total_ms:.2} ms):");
        let mut entries: Vec<_> = self.buckets.iter().collect();
        entries.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        let sum: f64 = entries.iter().map(|(_, (_, ms))| *ms).sum();
        println!("    {:24} {:>5}   {:>9}   {:>7}   {:>9}", "bucket", "count", "total_ms", "share", "avg_ms");
        for (name, (count, ms)) in &entries {
            let pct = ms / total_ms * 100.0;
            #[allow(clippy::cast_precision_loss)]
            let avg = ms / *count as f64;
            println!("    {name:24} {count:>5}   {ms:>9.2}   {pct:>6.1}%   {avg:>9.4}");
        }
        let overhead = total_ms - sum;
        println!(
            "    {:24} {:>5}   {sum:>9.2}   sum    (timer overhead: {overhead:+.2} ms)",
            "TOTAL", "",
        );
    }
}

fn conv_bucket(c: &Conv2d<B>) -> &'static str {
    // Note: bucket names reflect *kernel shape*, not the algorithm taken at
    // runtime. With `PREFERS_IM2COL_OVER_WINOGRAD = true` on `ScryGpuBackend`,
    // `conv_winograd_3x3s1` actually runs im2col + matmul. The 1×1-s1-p0
    // bucket runs through the pointwise fast path (`forward_1x1`), which
    // skips im2col entirely.
    if c.kernel_h == 3 && c.kernel_w == 3 && c.stride == 1 && c.padding == 1 {
        "conv_winograd_3x3s1"
    } else if c.kernel_h == 1 && c.kernel_w == 1 {
        if c.stride == 1 && c.padding == 0 {
            "conv_pointwise_1x1s1"
        } else {
            "conv_im2col_1x1s2_ds"
        }
    } else if c.kernel_h == 7 {
        "conv_im2col_7x7s2_stem"
    } else if c.kernel_h == 3 && c.stride == 2 {
        "conv_im2col_3x3s2"
    } else {
        "conv_im2col_other"
    }
}

fn forward_basic_block(p: &mut Profile, blk: &BasicBlock<B>, input: &Tensor<B>) -> Tensor<B> {
    let identity = match &blk.downsample {
        Some((dc, db)) => {
            let bucket = conv_bucket(dc);
            let c = p.time(bucket, || dc.forward(input));
            p.time("bn", || db.forward(&c))
        }
        None => Tensor::<B>::new(B::clone_storage(&input.data), input.shape.clone()),
    };

    let bucket1 = conv_bucket(&blk.conv1);
    let c1 = p.time(bucket1, || blk.conv1.forward(input));
    let n1 = p.time("bn", || blk.bn1.forward(&c1));
    let r1 = p.time("relu", || relu(&n1));
    let bucket2 = conv_bucket(&blk.conv2);
    let c2 = p.time(bucket2, || blk.conv2.forward(&r1));
    let n2 = p.time("bn", || blk.bn2.forward(&c2));

    let added = p.time("residual_add", || {
        let out_data = B::add(
            &n2.data,
            &identity.data,
            &n2.shape,
            &identity.shape,
            &n2.shape,
        );
        Tensor::<B>::new(out_data, n2.shape.clone())
    });
    p.time("relu", || relu(&added))
}

fn forward_bottleneck(p: &mut Profile, blk: &Bottleneck<B>, input: &Tensor<B>) -> Tensor<B> {
    let identity = match &blk.downsample {
        Some((dc, db)) => {
            let bucket = conv_bucket(dc);
            let c = p.time(bucket, || dc.forward(input));
            p.time("bn", || db.forward(&c))
        }
        None => Tensor::<B>::new(B::clone_storage(&input.data), input.shape.clone()),
    };

    let bucket1 = conv_bucket(&blk.conv1);
    let c1 = p.time(bucket1, || blk.conv1.forward(input));
    let n1 = p.time("bn", || blk.bn1.forward(&c1));
    let r1 = p.time("relu", || relu(&n1));
    let bucket2 = conv_bucket(&blk.conv2);
    let c2 = p.time(bucket2, || blk.conv2.forward(&r1));
    let n2 = p.time("bn", || blk.bn2.forward(&c2));
    let r2 = p.time("relu", || relu(&n2));
    let bucket3 = conv_bucket(&blk.conv3);
    let c3 = p.time(bucket3, || blk.conv3.forward(&r2));
    let n3 = p.time("bn", || blk.bn3.forward(&c3));

    let added = p.time("residual_add", || {
        let out_data = B::add(
            &n3.data,
            &identity.data,
            &n3.shape,
            &identity.shape,
            &n3.shape,
        );
        Tensor::<B>::new(out_data, n3.shape.clone())
    });
    p.time("relu", || relu(&added))
}

fn forward_instrumented(model: &ResNet<B>, input: &Tensor<B>) -> Profile {
    let mut p = Profile::default();
    let bucket = conv_bucket(&model.conv1);
    let c = p.time(bucket, || model.conv1.forward(input));
    let n = p.time("bn", || model.bn1.forward(&c));
    let r = p.time("relu", || relu(&n));
    let mut x = p.time("maxpool", || model.maxpool.forward(&r));

    for stage in &model.stages {
        match stage {
            ResNetStage::Basic(blocks) => {
                for blk in blocks {
                    x = forward_basic_block(&mut p, blk, &x);
                }
            }
            ResNetStage::Bottleneck(blocks) => {
                for blk in blocks {
                    x = forward_bottleneck(&mut p, blk, &x);
                }
            }
        }
    }

    let pooled = p.time("avgpool", || model.avgpool.forward(&x));
    let feature_dim = pooled.shape.dims()[0];

    if let (Some(w), Some(b)) = (&model.fc_weight, &model.fc_bias) {
        let _ = p.time("fc", || {
            let logits = B::matmul(
                &pooled.data,
                &w.data,
                1,
                feature_dim,
                model.config.num_classes,
                false,
                false,
            );
            let out_shape = Shape::new(&[model.config.num_classes]);
            let bias_shape = Shape::new(&[model.config.num_classes]);
            let result = B::add(&logits, &b.data, &out_shape, &bias_shape, &out_shape);
            Tensor::<B>::new(result, out_shape)
        });
    }
    p
}

/// Cheap LCG, mirrors bench_gpu.rs.
fn random_input(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let v = ((state >> 32) as i32 as f32) / (i32::MAX as f32);
            v
        })
        .collect()
}

fn run(label: &str, config: ResNetConfig, input: &[f32]) {
    println!("\n=== {label} ===");

    let mut model = ResNet::<B>::new(config);
    model.to_device();

    let mut input_storage = ScryGpuBackend::from_vec(input.to_vec(), &Shape::new(&[3, 224, 224]));
    ScryGpuBackend::to_device_in_place(&mut input_storage);
    let input_tensor = Tensor::<B>::new(input_storage, Shape::new(&[3, 224, 224]));

    // Warmup — JIT, allocator, Winograd cache.
    for _ in 0..2 {
        let _ = forward_instrumented(&model, &input_tensor);
    }

    // Best-of-3 to keep noise out without averaging away the signal.
    let mut best: Option<(f64, Profile)> = None;
    for _ in 0..3 {
        let t0 = Instant::now();
        let p = forward_instrumented(&model, &input_tensor);
        ScryGpuBackend::synchronize().expect("sync");
        let total = t0.elapsed().as_secs_f64() * 1000.0;
        if best.as_ref().is_none_or(|(t, _)| total < *t) {
            best = Some((total, p));
        }
    }
    let (total, prof) = best.unwrap();
    prof.print(total, label);
}

fn main() {
    println!("=== scry-vision per-op profile ===");
    println!("Features: scry-gpu-cuda={}", cfg!(feature = "scry-gpu-cuda"));

    let input = random_input(0xdead_beef, 3 * 224 * 224);

    run("ResNet-18", ResNetConfig::resnet18(1000), &input);
    run("ResNet-50", ResNetConfig::resnet50(1000), &input);
}
