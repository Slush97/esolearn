//! Fuzz `Shape::broadcast` and `Shape::broadcast_strides` with arbitrary dimensions.
//! Must never panic on valid inputs, must return `Err` on incompatible shapes.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::tensor::shape::Shape;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Parse two shapes from fuzz bytes.
    // First byte: ndim_a (1..=4), second byte: ndim_b (1..=4)
    let ndim_a = ((data[0] % 4) + 1) as usize;
    let ndim_b = ((data[1] % 4) + 1) as usize;

    if data.len() < 2 + ndim_a + ndim_b {
        return;
    }

    let dims_a: Vec<usize> = data[2..2 + ndim_a]
        .iter()
        .map(|&b| (b % 5) as usize + 1) // dims 1..=5
        .collect();
    let dims_b: Vec<usize> = data[2 + ndim_a..2 + ndim_a + ndim_b]
        .iter()
        .map(|&b| (b % 5) as usize + 1)
        .collect();

    let a = Shape::new(&dims_a);
    let b = Shape::new(&dims_b);

    // broadcast may succeed or fail — either is fine, must not panic
    if let Ok(out) = Shape::broadcast(&a, &b) {
        // strides must not panic
        let _ = a.broadcast_strides(&out);
        let _ = b.broadcast_strides(&out);
        let _ = out.strides();
        assert!(out.numel() > 0);
    }
});
