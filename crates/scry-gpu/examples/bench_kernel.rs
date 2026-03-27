//! Benchmark: one-shot `dispatch()` vs cached `compile()` + `run()`.
//!
//! Run with: `cargo run --example bench_kernel --release`

use std::time::Instant;

use scry_gpu::Device;

const SHADER: &str = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * 2.0;
    }
}";

fn main() {
    let gpu = Device::auto().expect("no GPU found");
    println!("Device: {} ({} MB)", gpu.name(), gpu.memory() / (1024 * 1024));
    println!();

    for &n in &[1_000, 10_000, 100_000] {
        bench_at_size(&gpu, n);
    }
}

fn bench_at_size(gpu: &Device, n: usize) {
    let iterations = 200;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();

    println!("── n = {n} ({iterations} iterations) ──");

    // Warmup: one dispatch to prime the driver.
    {
        let input = gpu.upload(&data).expect("upload failed");
        let output = gpu.alloc::<f32>(n).expect("alloc failed");
        gpu.dispatch(SHADER, &[&input, &output], n as u32)
            .expect("warmup dispatch failed");
    }

    // Benchmark one-shot dispatch (compile + pipeline + dispatch every time).
    let oneshot_start = Instant::now();
    for _ in 0..iterations {
        let input = gpu.upload(&data).expect("upload failed");
        let output = gpu.alloc::<f32>(n).expect("alloc failed");
        gpu.dispatch(SHADER, &[&input, &output], n as u32)
            .expect("dispatch failed");
    }
    let oneshot_total = oneshot_start.elapsed();
    let oneshot_per = oneshot_total / iterations;

    // Benchmark cached kernel (compile once, run many).
    let compile_start = Instant::now();
    let kernel = gpu.compile(SHADER).expect("compile failed");
    let compile_time = compile_start.elapsed();

    let cached_start = Instant::now();
    for _ in 0..iterations {
        let input = gpu.upload(&data).expect("upload failed");
        let output = gpu.alloc::<f32>(n).expect("alloc failed");
        gpu.run(&kernel, &[&input, &output], n as u32)
            .expect("run failed");
    }
    let cached_total = cached_start.elapsed();
    let cached_per = cached_total / iterations;

    let speedup = oneshot_per.as_nanos() as f64 / cached_per.as_nanos() as f64;

    println!("  one-shot dispatch():  {:>8.2?} / iter  (total: {oneshot_total:.2?})", oneshot_per);
    println!("  compile() once:       {:>8.2?}", compile_time);
    println!("  cached run():         {:>8.2?} / iter  (total: {cached_total:.2?})", cached_per);
    println!("  speedup:              {speedup:.2}x");
    println!();
}
