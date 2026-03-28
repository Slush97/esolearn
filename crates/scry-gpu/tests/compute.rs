//! End-to-end integration tests for scry-gpu.
//!
//! These tests require a Vulkan-capable GPU. They will fail (not hang)
//! on systems without one — the `Device::auto()` call returns `NoDevice`.
//!
//! All tests in this file dispatch WGSL shaders and therefore require the
//! Vulkan backend. When both `vulkan` and `cuda` features are enabled,
//! `Device::auto()` prefers CUDA (which cannot execute SPIR-V), so we
//! explicitly request the Vulkan backend.

use scry_gpu::{BackendKind, Device, DispatchConfig};

fn gpu() -> Device {
    Device::with_backend(BackendKind::Vulkan).expect("no Vulkan-capable GPU found — skipping test")
}

#[test]
fn device_auto_reports_name_and_memory() {
    let gpu = gpu();
    assert!(!gpu.name().is_empty());
    assert!(gpu.memory() > 0);
}

#[test]
fn vector_double() {
    let gpu = gpu();

    let input = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let output = gpu.alloc::<f32>(4).unwrap();

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

    gpu.dispatch(shader, &[&input, &output], 4).unwrap();

    let result: Vec<f32> = output.download().unwrap();
    assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn vector_add() {
    let gpu = gpu();

    let a = gpu.upload(&[10.0f32, 20.0, 30.0]).unwrap();
    let b = gpu.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let out = gpu.alloc::<f32>(3).unwrap();

    let shader = "\
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&a) {
        out[i] = a[i] + b[i];
    }
}";

    gpu.dispatch(shader, &[&a, &b, &out], 3).unwrap();

    let result: Vec<f32> = out.download().unwrap();
    assert_eq!(result, vec![11.0, 22.0, 33.0]);
}

#[test]
fn u32_square() {
    let gpu = gpu();

    let input = gpu.upload(&[2u32, 3, 5, 7, 11]).unwrap();
    let output = gpu.alloc::<u32>(5).unwrap();

    let shader = "\
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * input[i];
    }
}";

    gpu.dispatch(shader, &[&input, &output], 5).unwrap();

    let result: Vec<u32> = output.download().unwrap();
    assert_eq!(result, vec![4, 9, 25, 49, 121]);
}

#[test]
fn dispatch_configured_with_explicit_workgroups() {
    let gpu = gpu();

    let input = gpu.upload(&[1.0f32, 2.0]).unwrap();
    let output = gpu.alloc::<f32>(2).unwrap();

    let shader = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(2)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] + 100.0;
    }
}";

    let config = DispatchConfig::new(shader, 2).workgroups([1, 1, 1]);
    gpu.dispatch_configured(&config, &[&input, &output])
        .unwrap();

    let result: Vec<f32> = output.download().unwrap();
    assert_eq!(result, vec![101.0, 102.0]);
}

#[test]
fn binding_mismatch_is_caught() {
    let gpu = gpu();

    let buf = gpu.alloc::<f32>(4).unwrap();

    // Shader expects 2 bindings, we provide 1.
    let shader = "\
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&a) {
        b[i] = a[i];
    }
}";

    let err = gpu.dispatch(shader, &[&buf], 4).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("mismatch"),
        "expected binding mismatch error, got: {msg}"
    );
}

// ── Kernel (compile-once, dispatch-many) tests ──

const DOUBLE_SHADER: &str = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * 2.0;
    }
}";

#[test]
fn kernel_reuse_vector_double() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    // Run the same kernel three times with different data.
    for data in &[
        vec![1.0f32, 2.0, 3.0, 4.0],
        vec![10.0, 20.0, 30.0, 40.0],
        vec![0.5, -1.0, 100.0, 0.0],
    ] {
        let input = gpu.upload(data).expect("upload failed");
        let output = gpu.alloc::<f32>(data.len()).expect("alloc failed");

        gpu.run(&kernel, &[&input, &output], data.len() as u32)
            .expect("run failed");

        let result: Vec<f32> = output.download().expect("download failed");
        let expected: Vec<f32> = data.iter().map(|x| x * 2.0).collect();
        assert_eq!(result, expected);
    }
}

#[test]
fn kernel_reuse_different_sizes() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    // Same kernel, different buffer sizes.
    for n in [4, 100, 1000] {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let input = gpu.upload(&data).expect("upload failed");
        let output = gpu.alloc::<f32>(n).expect("alloc failed");

        gpu.run(&kernel, &[&input, &output], n as u32)
            .expect("run failed");

        let result: Vec<f32> = output.download().expect("download failed");
        for (i, (&got, &src)) in result.iter().zip(data.iter()).enumerate() {
            assert!(
                src.mul_add(-2.0, got).abs() < f32::EPSILON,
                "mismatch at index {i}: got {got}, expected {}",
                src * 2.0
            );
        }
    }
}

#[test]
fn kernel_binding_mismatch() {
    let gpu = gpu();

    // Kernel expects 2 bindings.
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");
    assert_eq!(kernel.binding_count(), 2);

    // Provide only 1 buffer.
    let buf = gpu.alloc::<f32>(4).expect("alloc failed");
    let err = gpu.run(&kernel, &[&buf], 4).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("mismatch"),
        "expected binding mismatch, got: {msg}"
    );
}

#[test]
fn kernel_debug_format() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let debug = format!("{kernel:?}");
    assert!(
        debug.contains("main"),
        "Debug output should contain entry point: {debug}"
    );
    assert!(
        debug.contains("Kernel"),
        "Debug output should contain type name: {debug}"
    );
}

#[test]
fn kernel_u32_square() {
    let gpu = gpu();

    let shader = "\
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * input[i];
    }
}";

    let kernel = gpu.compile(shader).expect("compile failed");

    for data in &[vec![2u32, 3, 5, 7, 11], vec![0, 1, 100, 255]] {
        let input = gpu.upload(data).expect("upload failed");
        let output = gpu.alloc::<u32>(data.len()).expect("alloc failed");

        gpu.run(&kernel, &[&input, &output], data.len() as u32)
            .expect("run failed");

        let result: Vec<u32> = output.download().expect("download failed");
        let expected: Vec<u32> = data.iter().map(|x| x * x).collect();
        assert_eq!(result, expected);
    }
}

#[test]
fn kernel_metadata() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    assert_eq!(kernel.entry_point(), "main");
    assert_eq!(kernel.binding_count(), 2);
    assert_eq!(kernel.workgroup_size(), [64, 1, 1]);
}

// ── Scale + edge-case tests ──

#[test]
fn scale_100k_elements() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let n = 100_000;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input = gpu.upload(&data).expect("upload failed");
    let output = gpu.alloc::<f32>(n).expect("alloc failed");

    gpu.run(&kernel, &[&input, &output], n as u32)
        .expect("run failed");

    let result: Vec<f32> = output.download().expect("download failed");
    assert_eq!(result.len(), n);

    // Spot-check a handful of indices rather than comparing 100K floats.
    for &i in &[0, 1, 999, 50_000, 99_999] {
        let expected = data[i] * 2.0;
        assert!(
            (result[i] - expected).abs() < f32::EPSILON,
            "mismatch at {i}: got {}, expected {expected}",
            result[i]
        );
    }
}

#[test]
fn scale_500k_elements() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let n = 500_000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let input = gpu.upload(&data).expect("upload failed");
    let output = gpu.alloc::<f32>(n).expect("alloc failed");

    gpu.run(&kernel, &[&input, &output], n as u32)
        .expect("run failed");

    let result: Vec<f32> = output.download().expect("download failed");
    assert_eq!(result.len(), n);

    for &i in &[0, 1, n / 2, n - 1] {
        let expected = data[i] * 2.0;
        assert!(
            (result[i] - expected).abs() < 0.001,
            "mismatch at {i}: got {}, expected {expected}",
            result[i]
        );
    }
}

#[test]
fn workgroup_boundary_alignment() {
    // Test sizes that are NOT multiples of workgroup_size (64).
    // Ensures the bounds check in the shader + dispatch ceiling work correctly.
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    for n in [1, 2, 63, 65, 127, 128, 129, 255, 257] {
        let data: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let input = gpu.upload(&data).expect("upload failed");
        let output = gpu.alloc::<f32>(n).expect("alloc failed");

        gpu.run(&kernel, &[&input, &output], n as u32)
            .unwrap_or_else(|e| panic!("run failed for n={n}: {e}"));

        let result: Vec<f32> = output.download().expect("download failed");
        assert_eq!(result.len(), n, "wrong length for n={n}");

        // Check first and last element.
        assert!(
            data[0].mul_add(-2.0, result[0]).abs() < f32::EPSILON,
            "first element wrong for n={n}"
        );
        assert!(
            data[n - 1].mul_add(-2.0, result[n - 1]).abs() < f32::EPSILON,
            "last element wrong for n={n}"
        );
    }
}

#[test]
fn single_element_buffer() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let input = gpu.upload(&[42.0f32]).expect("upload failed");
    let output = gpu.alloc::<f32>(1).expect("alloc failed");

    gpu.run(&kernel, &[&input, &output], 1).expect("run failed");

    let result: Vec<f32> = output.download().expect("download failed");
    assert_eq!(result, vec![84.0]);
}

// ── Batch dispatch ──

const ADD_ONE_SHADER: &str = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] + 1.0;
    }
}";

#[test]
fn batch_single_dispatch() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let input = gpu.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let output = gpu.alloc::<f32>(4).unwrap();

    let mut batch = gpu.batch().unwrap();
    batch.run(&kernel, &[&input, &output], 4).unwrap();
    batch.submit().unwrap();

    let result: Vec<f32> = output.download().unwrap();
    assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn batch_multiple_independent_dispatches() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let in1 = gpu.upload(&[1.0f32, 2.0]).unwrap();
    let out1 = gpu.alloc::<f32>(2).unwrap();
    let in2 = gpu.upload(&[10.0f32, 20.0, 30.0]).unwrap();
    let out2 = gpu.alloc::<f32>(3).unwrap();

    let mut batch = gpu.batch().unwrap();
    batch.run(&kernel, &[&in1, &out1], 2).unwrap();
    batch.run(&kernel, &[&in2, &out2], 3).unwrap();
    batch.submit().unwrap();

    assert_eq!(out1.download().unwrap(), vec![2.0, 4.0]);
    assert_eq!(out2.download().unwrap(), vec![20.0, 40.0, 60.0]);
}

#[test]
fn batch_chained_with_barrier() {
    // Dispatch A writes intermediate, dispatch B reads it.
    // Without a barrier this would be a data race.
    let gpu = gpu();
    let double = gpu.compile(DOUBLE_SHADER).expect("compile double");
    let add_one = gpu.compile(ADD_ONE_SHADER).expect("compile add_one");

    let input = gpu.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let mid = gpu.alloc::<f32>(3).unwrap();
    let output = gpu.alloc::<f32>(3).unwrap();

    let mut batch = gpu.batch().unwrap();
    batch.run(&double, &[&input, &mid], 3).unwrap();
    batch.barrier();
    batch.run(&add_one, &[&mid, &output], 3).unwrap();
    batch.submit().unwrap();

    // input * 2 + 1 = [3.0, 5.0, 7.0]
    let result: Vec<f32> = output.download().unwrap();
    assert_eq!(result, vec![3.0, 5.0, 7.0]);
}

#[test]
fn batch_three_stage_pipeline() {
    // Three chained dispatches: double → double → add_one
    let gpu = gpu();
    let double = gpu.compile(DOUBLE_SHADER).expect("compile double");
    let add_one = gpu.compile(ADD_ONE_SHADER).expect("compile add_one");

    let input = gpu.upload(&[1.0f32, 5.0, 10.0]).unwrap();
    let mid1 = gpu.alloc::<f32>(3).unwrap();
    let mid2 = gpu.alloc::<f32>(3).unwrap();
    let output = gpu.alloc::<f32>(3).unwrap();

    let mut batch = gpu.batch().unwrap();
    batch.run(&double, &[&input, &mid1], 3).unwrap();
    batch.barrier();
    batch.run(&double, &[&mid1, &mid2], 3).unwrap();
    batch.barrier();
    batch.run(&add_one, &[&mid2, &output], 3).unwrap();
    batch.submit().unwrap();

    // (input * 2) * 2 + 1 = [5.0, 21.0, 41.0]
    let result: Vec<f32> = output.download().unwrap();
    assert_eq!(result, vec![5.0, 21.0, 41.0]);
}

#[test]
fn batch_reduction_correctness() {
    // Multi-pass reduction with barriers: sum of 1024 ones should be 1024.
    let gpu = gpu();

    let reduce_shader = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    scratch[lid.x] = select(0.0, input[i], i < arrayLength(&input));
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if lid.x < stride {
            scratch[lid.x] += scratch[lid.x + stride];
        }
        workgroupBarrier();
    }

    if lid.x == 0u {
        output[wid.x] = scratch[0];
    }
}";

    let kernel = gpu.compile(reduce_shader).expect("compile reduce");

    let n: u32 = 1024;
    let data: Vec<f32> = vec![1.0; n as usize];
    let input = gpu.upload(&data).unwrap();

    // 1024 / 256 = 4 workgroups → 4 partial sums
    let pass1 = gpu.alloc::<f32>(4).unwrap();
    // 4 / 256 → 1 workgroup → 1 result
    let pass2 = gpu.alloc::<f32>(1).unwrap();

    let mut batch = gpu.batch().unwrap();
    batch.run(&kernel, &[&input, &pass1], n).unwrap();
    batch.barrier();
    batch.run(&kernel, &[&pass1, &pass2], 4).unwrap();
    batch.submit().unwrap();

    let result = pass2.download().unwrap();
    assert!(
        (result[0] - 1024.0).abs() < 0.01,
        "expected 1024.0, got {}",
        result[0]
    );
}

#[test]
fn batch_with_push_constants() {
    let gpu = gpu();

    let shader = "\
struct Params { scale: f32 }
var<push_constant> params: Params;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * params.scale;
    }
}";

    let kernel = gpu.compile(shader).unwrap();
    let input = gpu.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    let out1 = gpu.alloc::<f32>(3).unwrap();
    let out2 = gpu.alloc::<f32>(3).unwrap();

    let mut batch = gpu.batch().unwrap();
    batch
        .run_with_push_constants(&kernel, &[&input, &out1], 3, &10.0f32.to_ne_bytes())
        .unwrap();
    batch
        .run_with_push_constants(&kernel, &[&input, &out2], 3, &0.5f32.to_ne_bytes())
        .unwrap();
    batch.submit().unwrap();

    assert_eq!(out1.download().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(out2.download().unwrap(), vec![0.5, 1.0, 1.5]);
}

#[test]
fn batch_with_run_configured() {
    let gpu = gpu();

    let shader = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(2)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] + 100.0;
    }
}";

    let kernel = gpu.compile(shader).unwrap();
    let input = gpu.upload(&[1.0f32, 2.0]).unwrap();
    let output = gpu.alloc::<f32>(2).unwrap();

    let mut batch = gpu.batch().unwrap();
    batch
        .run_configured(&kernel, &[&input, &output], [1, 1, 1], None)
        .unwrap();
    batch.submit().unwrap();

    assert_eq!(output.download().unwrap(), vec![101.0, 102.0]);
}

#[test]
fn batch_binding_mismatch() {
    let gpu = gpu();

    // DOUBLE_SHADER expects 2 bindings.
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");
    let buf = gpu.alloc::<f32>(4).unwrap();

    let mut batch = gpu.batch().unwrap();
    let result = batch.run(&kernel, &[&buf], 4);
    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("mismatch"),
                "expected binding mismatch error, got: {msg}"
            );
        }
        Ok(_) => panic!("expected binding mismatch error, but got Ok"),
    }
}

#[test]
fn batch_drop_without_submit() {
    // Creating a batch, recording dispatches, and dropping without submit
    // must not panic or leak. This is a safety/hygiene check.
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let input = gpu.upload(&[1.0f32, 2.0]).unwrap();
    let output = gpu.alloc::<f32>(2).unwrap();

    let mut batch = gpu.batch().unwrap();
    batch.run(&kernel, &[&input, &output], 2).unwrap();
    drop(batch); // intentional: no submit()

    // If we got here without panicking, the test passes.
    // Verify the GPU is still usable after the dropped batch.
    let out2 = gpu.alloc::<f32>(2).unwrap();
    gpu.run(&kernel, &[&input, &out2], 2).unwrap();
    assert_eq!(out2.download().unwrap(), vec![2.0, 4.0]);
}

#[test]
fn batch_fluent_api() {
    // Verify the builder-style chaining works.
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let input = gpu.upload(&[5.0f32]).unwrap();
    let mid = gpu.alloc::<f32>(1).unwrap();
    let output = gpu.alloc::<f32>(1).unwrap();

    let mut batch = gpu.batch().unwrap();
    batch
        .run(&kernel, &[&input, &mid], 1)
        .unwrap()
        .barrier()
        .run(&kernel, &[&mid, &output], 1)
        .unwrap();
    batch.submit().unwrap();

    assert_eq!(output.download().unwrap(), vec![20.0]); // 5 * 2 * 2
}

// ── Subgroup operations ──

#[test]
fn subgroup_size_is_nonzero() {
    let gpu = gpu();
    let sg = gpu.subgroup_size();
    assert!(sg > 0, "subgroup_size should be > 0, got {sg}");
    assert!(
        sg.is_power_of_two(),
        "subgroup_size should be power of 2, got {sg}"
    );
}

#[test]
fn subgroup_reduction_sum() {
    // Single-pass reduction using subgroupAdd.
    // Each workgroup: subgroupAdd within warps → shared memory collect → subgroupAdd again.
    let gpu = gpu();

    let shader = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> partials: array<f32, 32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    // Load (0 for out-of-bounds threads)
    let val = select(0.0, input[gid.x], gid.x < arrayLength(&input));

    // Warp-level reduction
    let warp_sum = subgroupAdd(val);

    // Lane 0 of each subgroup writes its partial sum to shared memory
    if lane == 0u {
        partials[sg_id] = warp_sum;
    }
    workgroupBarrier();

    // First subgroup reduces the partial sums
    let num_subgroups = 256u / sg_size;
    if sg_id == 0u {
        let p = select(0.0, partials[lane], lane < num_subgroups);
        let total = subgroupAdd(p);
        if lane == 0u {
            output[wid.x] = total;
        }
    }
}";

    let kernel = gpu.compile(shader).expect("compile subgroup reduce");

    // Test: sum of 1024 ones = 1024
    let n: u32 = 1024;
    let data: Vec<f32> = vec![1.0; n as usize];
    let input = gpu.upload(&data).unwrap();
    let output = gpu.alloc::<f32>((n.div_ceil(256)) as usize).unwrap();

    gpu.run(&kernel, &[&input, &output], n).unwrap();

    let result = output.download().unwrap();
    let total: f32 = result.iter().sum();
    assert!(
        (total - 1024.0).abs() < 0.01,
        "expected 1024.0, got {total} (partials: {result:?})"
    );

    // Test: sum of 0..256 = 256*255/2 = 32640
    let data2: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let input2 = gpu.upload(&data2).unwrap();
    let output2 = gpu.alloc::<f32>(1).unwrap();

    gpu.run(&kernel, &[&input2, &output2], 256).unwrap();

    let result2 = output2.download().unwrap();
    assert!(
        (result2[0] - 32640.0).abs() < 1.0,
        "expected 32640.0, got {}",
        result2[0]
    );
}

// ── Push constants ──

#[test]
fn push_constants_scale_factor() {
    let gpu = gpu();

    let shader = "\
struct Params {
    scale: f32,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
var<push_constant> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * params.scale;
    }
}";

    let kernel = gpu.compile(shader).expect("compile failed");
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let input = gpu.upload(&data).expect("upload failed");

    // Run with scale = 10.0
    let output = gpu.alloc::<f32>(4).expect("alloc failed");
    gpu.run_with_push_constants(&kernel, &[&input, &output], 4, &10.0f32.to_ne_bytes())
        .expect("run failed");

    let result: Vec<f32> = output.download().expect("download failed");
    assert_eq!(result, vec![10.0, 20.0, 30.0, 40.0]);

    // Same kernel, different scale = 0.5
    let output2 = gpu.alloc::<f32>(4).expect("alloc failed");
    gpu.run_with_push_constants(&kernel, &[&input, &output2], 4, &0.5f32.to_ne_bytes())
        .expect("run failed");

    let result2: Vec<f32> = output2.download().expect("download failed");
    assert_eq!(result2, vec![0.5, 1.0, 1.5, 2.0]);
}

// ── Built-in shader tests: matmul ──

/// CPU reference matmul for verification.
#[allow(clippy::many_single_char_names)]
fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

fn assert_matmul_close(got: &[f32], expected: &[f32], label: &str, tol: f32) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let cols = (expected.len() as f64).sqrt() as usize;
    let cols = cols.max(1);
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "{label}: mismatch at [{}, {}]: got {g}, expected {e}",
            i / cols,
            i % cols,
        );
    }
}

#[test]
#[allow(clippy::many_single_char_names)]
fn matmul_tiled_16x16_square() {
    let gpu = gpu();
    let kernel = gpu
        .compile(scry_gpu::shaders::matmul::TILED_16X16)
        .expect("compile tiled matmul");

    // 32x32 matmul (2 tiles per dimension)
    let m: u32 = 32;
    let n: u32 = 32;
    let k: u32 = 32;

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 - 5.0).collect();
    let expected = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    let buf_a = gpu.upload(&a).unwrap();
    let buf_b = gpu.upload(&b).unwrap();
    let buf_c = gpu.alloc::<f32>((m * n) as usize).unwrap();

    let dims = [m, n, k];
    let push = bytemuck::cast_slice::<u32, u8>(&dims);
    let workgroups = [n.div_ceil(16), m.div_ceil(16), 1];

    gpu.run_configured(&kernel, &[&buf_a, &buf_b, &buf_c], workgroups, Some(push))
        .unwrap();

    let result = buf_c.download().unwrap();
    assert_matmul_close(&result, &expected, "tiled_16x16 32x32", 0.1);
}

#[test]
#[allow(clippy::many_single_char_names)]
fn matmul_tiled_16x16_non_square() {
    let gpu = gpu();
    let kernel = gpu
        .compile(scry_gpu::shaders::matmul::TILED_16X16)
        .expect("compile tiled matmul");

    // Non-square: 20x48 * 48x36 = 20x36 (non-multiple-of-16 dims)
    let m: u32 = 20;
    let n: u32 = 36;
    let k: u32 = 48;

    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| (i as f32).mul_add(-0.005, 1.0))
        .collect();
    let expected = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    let buf_a = gpu.upload(&a).unwrap();
    let buf_b = gpu.upload(&b).unwrap();
    let buf_c = gpu.alloc::<f32>((m * n) as usize).unwrap();

    let dims = [m, n, k];
    let push = bytemuck::cast_slice::<u32, u8>(&dims);
    let workgroups = [n.div_ceil(16), m.div_ceil(16), 1];

    gpu.run_configured(&kernel, &[&buf_a, &buf_b, &buf_c], workgroups, Some(push))
        .unwrap();

    let result = buf_c.download().unwrap();
    assert_matmul_close(&result, &expected, "tiled_16x16 20x36", 0.5);
}

#[test]
#[allow(clippy::many_single_char_names)]
fn matmul_coarse_64x64_square() {
    let gpu = gpu();
    let kernel = gpu
        .compile(scry_gpu::shaders::matmul::COARSE_64X64)
        .expect("compile coarse matmul");

    // 128x128 matmul (2 tiles per dimension at 64x64)
    let m: u32 = 128;
    let n: u32 = 128;
    let k: u32 = 128;

    let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 - 6.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 9) as f32 - 4.0).collect();
    let expected = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    let buf_a = gpu.upload(&a).unwrap();
    let buf_b = gpu.upload(&b).unwrap();
    let buf_c = gpu.alloc::<f32>((m * n) as usize).unwrap();

    let dims = [m, n, k];
    let push = bytemuck::cast_slice::<u32, u8>(&dims);
    let workgroups = [n.div_ceil(64), m.div_ceil(64), 1];

    gpu.run_configured(&kernel, &[&buf_a, &buf_b, &buf_c], workgroups, Some(push))
        .unwrap();

    let result = buf_c.download().unwrap();
    assert_matmul_close(&result, &expected, "coarse_64x64 128x128", 1.0);
}

#[test]
#[allow(clippy::many_single_char_names)]
fn matmul_coarse_64x64_non_square() {
    let gpu = gpu();
    let kernel = gpu
        .compile(scry_gpu::shaders::matmul::COARSE_64X64)
        .expect("compile coarse matmul");

    // Non-square: 50x100 * 100x70 = 50x70
    let m: u32 = 50;
    let n: u32 = 70;
    let k: u32 = 100;

    let a: Vec<f32> = (0..m * k)
        .map(|i| (i as f32).mul_add(0.01, -25.0))
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| (i as f32).mul_add(0.01, -35.0))
        .collect();
    let expected = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    let buf_a = gpu.upload(&a).unwrap();
    let buf_b = gpu.upload(&b).unwrap();
    let buf_c = gpu.alloc::<f32>((m * n) as usize).unwrap();

    let dims = [m, n, k];
    let push = bytemuck::cast_slice::<u32, u8>(&dims);
    let workgroups = [n.div_ceil(64), m.div_ceil(64), 1];

    gpu.run_configured(&kernel, &[&buf_a, &buf_b, &buf_c], workgroups, Some(push))
        .unwrap();

    let result = buf_c.download().unwrap();
    assert_matmul_close(&result, &expected, "coarse_64x64 50x70", 1.0);
}

#[test]
fn matmul_identity() {
    // A * I = A — sanity check with identity matrix
    let gpu = gpu();
    let kernel = gpu
        .compile(scry_gpu::shaders::matmul::TILED_16X16)
        .expect("compile tiled matmul");

    let m: u32 = 16;
    let n: u32 = 16;
    let k: u32 = 16;

    let a: Vec<f32> = (0..m * k).map(|i| i as f32 + 1.0).collect();
    let mut identity = vec![0.0f32; (k * n) as usize];
    for i in 0..n {
        identity[(i * n + i) as usize] = 1.0;
    }

    let buf_a = gpu.upload(&a).unwrap();
    let buf_b = gpu.upload(&identity).unwrap();
    let buf_c = gpu.alloc::<f32>((m * n) as usize).unwrap();

    let dims = [m, n, k];
    let push = bytemuck::cast_slice::<u32, u8>(&dims);
    let workgroups = [n.div_ceil(16), m.div_ceil(16), 1];

    gpu.run_configured(&kernel, &[&buf_a, &buf_b, &buf_c], workgroups, Some(push))
        .unwrap();

    let result = buf_c.download().unwrap();
    assert_matmul_close(&result, &a, "identity", 0.001);
}

#[test]
#[allow(clippy::many_single_char_names)]
fn matmul_coarse_8x8_square() {
    let gpu = gpu();
    let kernel = gpu
        .compile(scry_gpu::shaders::matmul::COARSE_8X8)
        .expect("compile 8x8 matmul");

    let m: u32 = 128;
    let n: u32 = 128;
    let k: u32 = 128;

    let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 - 6.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 9) as f32 - 4.0).collect();
    let expected = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    let buf_a = gpu.upload(&a).unwrap();
    let buf_b = gpu.upload(&b).unwrap();
    let buf_c = gpu.alloc::<f32>((m * n) as usize).unwrap();

    let dims = [m, n, k];
    let push = bytemuck::cast_slice::<u32, u8>(&dims);
    let workgroups = [n.div_ceil(128), m.div_ceil(128), 1];

    gpu.run_configured(&kernel, &[&buf_a, &buf_b, &buf_c], workgroups, Some(push))
        .unwrap();

    let result = buf_c.download().unwrap();
    assert_matmul_close(&result, &expected, "coarse_8x8 128x128", 1.0);
}

#[test]
#[allow(clippy::many_single_char_names)]
fn matmul_coarse_8x8_non_square() {
    let gpu = gpu();
    let kernel = gpu
        .compile(scry_gpu::shaders::matmul::COARSE_8X8)
        .expect("compile 8x8 matmul");

    // Non-square: 200x300 * 300x250 = 200x250
    let m: u32 = 200;
    let n: u32 = 250;
    let k: u32 = 300;

    let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 - 8.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 - 5.0).collect();
    let expected = cpu_matmul(&a, &b, m as usize, n as usize, k as usize);

    let buf_a = gpu.upload(&a).unwrap();
    let buf_b = gpu.upload(&b).unwrap();
    let buf_c = gpu.alloc::<f32>((m * n) as usize).unwrap();

    let dims = [m, n, k];
    let push = bytemuck::cast_slice::<u32, u8>(&dims);
    let workgroups = [n.div_ceil(128), m.div_ceil(128), 1];

    gpu.run_configured(&kernel, &[&buf_a, &buf_b, &buf_c], workgroups, Some(push))
        .unwrap();

    let result = buf_c.download().unwrap();
    assert_matmul_close(&result, &expected, "coarse_8x8 200x250", 2.0);
}

// ── Built-in shader tests: pairwise distance ──

#[test]
fn pairwise_euclidean_basic() {
    let gpu = gpu();
    let kernel = gpu
        .compile(scry_gpu::shaders::distance::PAIRWISE_EUCLIDEAN)
        .expect("compile pairwise euclidean");

    // 2 query points, 3 train points, 3 dimensions
    let n_q: u32 = 2;
    let n_t: u32 = 3;
    let dim: u32 = 3;

    let queries = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // q0=(1,0,0), q1=(0,1,0)
    let train = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // t0=(1,0,0), t1=(0,1,0), t2=(0,0,1)

    let buf_q = gpu.upload(&queries).unwrap();
    let buf_t = gpu.upload(&train).unwrap();
    let buf_d = gpu.alloc::<f32>((n_q * n_t) as usize).unwrap();

    let dims = [n_q, n_t, dim];
    let push = bytemuck::cast_slice::<u32, u8>(&dims);

    gpu.run_configured(
        &kernel,
        &[&buf_q, &buf_t, &buf_d],
        [(n_q * n_t).div_ceil(256), 1, 1],
        Some(push),
    )
    .unwrap();

    let result = buf_d.download().unwrap();
    // D[0][0] = dist(q0, t0) = 0.0 (same point)
    // D[0][1] = dist(q0, t1) = 1+1+0 = 2.0
    // D[0][2] = dist(q0, t2) = 1+0+1 = 2.0
    // D[1][0] = dist(q1, t0) = 1+1+0 = 2.0
    // D[1][1] = dist(q1, t1) = 0.0
    // D[1][2] = dist(q1, t2) = 0+1+1 = 2.0
    let expected = [0.0, 2.0, 2.0, 2.0, 0.0, 2.0];

    for (i, (&g, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 0.001, "dist[{i}] = {g}, expected {e}",);
    }
}

#[test]
fn pairwise_euclidean_scaled() {
    // Verify with known distances: points along a line
    let gpu = gpu();
    let kernel = gpu
        .compile(scry_gpu::shaders::distance::PAIRWISE_EUCLIDEAN)
        .expect("compile pairwise euclidean");

    let n_q: u32 = 3;
    let n_t: u32 = 3;
    let dim: u32 = 1;

    // Points at 0, 3, 7 on a 1D line
    let queries = [0.0f32, 3.0, 7.0];
    let train = [0.0f32, 3.0, 7.0];

    let buf_q = gpu.upload(&queries).unwrap();
    let buf_t = gpu.upload(&train).unwrap();
    let buf_d = gpu.alloc::<f32>((n_q * n_t) as usize).unwrap();

    let dims = [n_q, n_t, dim];
    let push = bytemuck::cast_slice::<u32, u8>(&dims);

    gpu.run_configured(
        &kernel,
        &[&buf_q, &buf_t, &buf_d],
        [(n_q * n_t).div_ceil(256), 1, 1],
        Some(push),
    )
    .unwrap();

    let result = buf_d.download().unwrap();
    // Squared distances: |qi - tj|^2
    let expected = [
        0.0, 9.0, 49.0, // q=0 vs t=0,3,7
        9.0, 0.0, 16.0, // q=3 vs t=0,3,7
        49.0, 16.0, 0.0, // q=7 vs t=0,3,7
    ];

    for (i, (&g, &e)) in result.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 0.001, "dist[{i}] = {g}, expected {e}",);
    }
}

// ── Error path tests ──

#[test]
fn compile_invalid_wgsl_returns_shader_error() {
    let gpu = gpu();

    let bad_shader = "this is not valid WGSL at all!!!";
    let err = gpu.compile(bad_shader).unwrap_err();

    match err {
        scry_gpu::GpuError::ShaderCompilation(_) => {} // expected
        other => panic!("expected ShaderCompilation, got: {other}"),
    }
}

#[test]
fn compile_missing_entry_point() {
    let gpu = gpu();

    // Valid WGSL but entry point is "process", not "main"
    let shader = "\
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn process(@builtin(global_invocation_id) gid: vec3<u32>) {
    buf[gid.x] = 0.0;
}";

    let err = gpu.compile(shader).unwrap_err(); // default looks for "main"

    match err {
        scry_gpu::GpuError::MissingEntryPoint { ref name } => {
            assert_eq!(name, "main");
        }
        other => panic!("expected MissingEntryPoint, got: {other}"),
    }

    // But compiling with the correct name succeeds
    let kernel = gpu.compile_named(shader, "process").unwrap();
    assert_eq!(kernel.entry_point(), "process");
}

#[test]
fn compile_named_entry_point() {
    let gpu = gpu();

    let shader = "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn double_it(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * 2.0;
    }
}";

    let kernel = gpu.compile_named(shader, "double_it").unwrap();
    assert_eq!(kernel.entry_point(), "double_it");
    assert_eq!(kernel.binding_count(), 2);

    let input = gpu.upload(&[5.0f32, 10.0]).unwrap();
    let output = gpu.alloc::<f32>(2).unwrap();
    gpu.run(&kernel, &[&input, &output], 2).unwrap();

    assert_eq!(output.download().unwrap(), vec![10.0, 20.0]);
}

// ── Large batch (descriptor pool overflow) ──

#[test]
fn batch_large_descriptor_pool_overflow() {
    // >64 dispatches in a single batch to exercise pool overflow logic
    // (VulkanBatch uses pools of 64 sets each)
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let n = 100; // well over the 64-set pool limit
    let input = gpu.upload(&[1.0f32]).unwrap();
    let mut bufs: Vec<scry_gpu::Buffer<f32>> = Vec::with_capacity(n);
    for _ in 0..n {
        bufs.push(gpu.alloc::<f32>(1).unwrap());
    }

    let mut batch = gpu.batch().unwrap();
    // First dispatch: input → bufs[0]
    batch.run(&kernel, &[&input, &bufs[0]], 1).unwrap();
    batch.barrier();

    // Chain: bufs[i-1] → bufs[i], doubling each time
    for i in 1..n {
        batch.run(&kernel, &[&bufs[i - 1], &bufs[i]], 1).unwrap();
        batch.barrier();
    }
    batch.submit().unwrap();

    // After 100 doublings: 1.0 * 2^100 — but f32 can only hold 2^128.
    // Check a few milestones.
    let r0 = bufs[0].download().unwrap();
    assert!(
        (r0[0] - 2.0).abs() < 0.001,
        "buf[0] = {}, expected 2.0",
        r0[0]
    );

    let r9 = bufs[9].download().unwrap();
    assert!(
        (r9[0] - 1024.0).abs() < 0.01,
        "buf[9] = {}, expected 1024.0 (2^10)",
        r9[0],
    );

    // buf[20] = 2^21 = 2097152
    let r20 = bufs[20].download().unwrap();
    assert!(
        (r20[0] - 2_097_152.0).abs() < 1.0,
        "buf[20] = {}, expected 2097152.0 (2^21)",
        r20[0],
    );
}

// ── Buffer edge cases ──

#[test]
fn buffer_len_and_byte_size() {
    let gpu = gpu();
    let buf = gpu.upload(&[1.0f32, 2.0, 3.0]).unwrap();
    assert_eq!(buf.len(), 3);
    assert_eq!(buf.byte_size(), 12); // 3 * 4 bytes
    assert!(!buf.is_empty());

    let buf_u32 = gpu.upload(&[0u32; 100]).unwrap();
    assert_eq!(buf_u32.len(), 100);
    assert_eq!(buf_u32.byte_size(), 400);
}

#[test]
fn upload_download_repeated_reuses_pool() {
    // Repeated upload+dispatch+download cycles should reuse staging buffers.
    // This test verifies correctness (not timing), but exercises the pool path.
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    for i in 0..20u32 {
        let data: Vec<f32> = (0..1000).map(|j| (i * 1000 + j) as f32).collect();
        let input = gpu.upload(&data).unwrap();
        let output = gpu.alloc::<f32>(1000).unwrap();
        gpu.run(&kernel, &[&input, &output], 1000).unwrap();

        let result = output.download().unwrap();
        assert!(
            data[0].mul_add(-2.0, result[0]).abs() < f32::EPSILON,
            "iter {i}: got {}, expected {}",
            result[0],
            data[0] * 2.0,
        );
    }
}

#[test]
fn alloc_download_is_zeroed_or_deterministic() {
    // GPU alloc doesn't guarantee zeroed memory, but download should succeed
    let gpu = gpu();
    let buf = gpu.alloc::<f32>(64).unwrap();
    let data = buf.download().unwrap();
    assert_eq!(data.len(), 64);
    // We don't assert values — uninitialized GPU memory is undefined.
    // This test verifies the alloc→download path doesn't crash.
}
