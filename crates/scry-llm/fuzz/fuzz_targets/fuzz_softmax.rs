//! Fuzz softmax forward. Output must sum to 1, be non-negative,
//! and be finite for any finite input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let batch = (data[0] % 4) as usize + 1; // 1..=4
    let dim = (data[1] % 8) as usize + 1; // 1..=8
    let numel = batch * dim;

    // Build input from fuzz bytes or deterministic fallback
    let input: Vec<f32> = if data.len() >= 2 + numel * 4 {
        let mut cursor = 2;
        (0..numel)
            .map(|_| {
                let bytes = [data[cursor], data[cursor + 1], data[cursor + 2], data[cursor + 3]];
                cursor += 4;
                let v = f32::from_le_bytes(bytes);
                if v.is_finite() { v } else { 0.0 }
            })
            .collect()
    } else {
        (0..numel).map(|i| ((i % 11) as f32 - 5.0) * 0.5).collect()
    };

    let shape = Shape::new(&[batch, dim]);
    let output = CpuBackend::softmax(&input, &shape);

    assert_eq!(output.len(), numel);
    assert!(output.iter().all(|&v| v.is_finite() && v >= 0.0));

    // Each row should sum to ~1.0
    for b in 0..batch {
        let row_sum: f64 = output[b * dim..(b + 1) * dim]
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "softmax row {b} sum = {row_sum}, expected ~1.0"
        );
    }
});
