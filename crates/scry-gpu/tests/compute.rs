//! End-to-end integration tests for scry-gpu.
//!
//! These tests require a Vulkan-capable GPU. They will fail (not hang)
//! on systems without one — the `Device::auto()` call returns `NoDevice`.

use scry_gpu::{Device, DispatchConfig};

fn gpu() -> Device {
    Device::auto().expect("no Vulkan-capable GPU found — skipping test")
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
    gpu.dispatch_configured(&config, &[&input, &output]).unwrap();

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
    assert!(msg.contains("mismatch"), "expected binding mismatch error, got: {msg}");
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
                (got - src * 2.0).abs() < f32::EPSILON,
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
    assert!(msg.contains("mismatch"), "expected binding mismatch, got: {msg}");
}

#[test]
fn kernel_debug_format() {
    let gpu = gpu();
    let kernel = gpu.compile(DOUBLE_SHADER).expect("compile failed");

    let debug = format!("{kernel:?}");
    assert!(debug.contains("main"), "Debug output should contain entry point: {debug}");
    assert!(debug.contains("Kernel"), "Debug output should contain type name: {debug}");
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
            (result[0] - data[0] * 2.0).abs() < f32::EPSILON,
            "first element wrong for n={n}"
        );
        assert!(
            (result[n - 1] - data[n - 1] * 2.0).abs() < f32::EPSILON,
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
    gpu.run_with_push_constants(
        &kernel,
        &[&input, &output],
        4,
        &10.0f32.to_ne_bytes(),
    )
    .expect("run failed");

    let result: Vec<f32> = output.download().expect("download failed");
    assert_eq!(result, vec![10.0, 20.0, 30.0, 40.0]);

    // Same kernel, different scale = 0.5
    let output2 = gpu.alloc::<f32>(4).expect("alloc failed");
    gpu.run_with_push_constants(
        &kernel,
        &[&input, &output2],
        4,
        &0.5f32.to_ne_bytes(),
    )
    .expect("run failed");

    let result2: Vec<f32> = output2.download().expect("download failed");
    assert_eq!(result2, vec![0.5, 1.0, 1.5, 2.0]);
}
