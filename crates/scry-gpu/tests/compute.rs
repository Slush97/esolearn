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
