//! Fuzz matmul forward with arbitrary (small) dimensions and transpose combos.
//! Must never panic, NaN, or Inf for finite inputs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::MathBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let m = (data[0] % 8) as usize + 1; // 1..=8
    let k = (data[1] % 8) as usize + 1;
    let n = (data[2] % 8) as usize + 1;

    let needed = 3 + (m * k + k * n) * 4;
    if data.len() < needed {
        // Not enough fuzz bytes for the matrices, use deterministic data
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();

        for (trans_a, trans_b) in [(false, false), (true, false), (false, true), (true, true)] {
            let c = CpuBackend::matmul(&a, &b, m, k, n, trans_a, trans_b);
            assert_eq!(c.len(), m * n);
            assert!(c.iter().all(|v| v.is_finite()));
        }
        return;
    }

    // Parse fuzz-controlled f32 data
    let mut cursor = 3;
    let parse_f32s = |cursor: &mut usize, count: usize, data: &[u8]| -> Vec<f32> {
        (0..count)
            .map(|_| {
                let bytes = [data[*cursor], data[*cursor + 1], data[*cursor + 2], data[*cursor + 3]];
                *cursor += 4;
                let v = f32::from_le_bytes(bytes);
                if v.is_finite() { v } else { 0.0 }
            })
            .collect()
    };

    let a = parse_f32s(&mut cursor, m * k, data);
    let b = parse_f32s(&mut cursor, k * n, data);

    let c = CpuBackend::matmul(&a, &b, m, k, n, false, false);
    assert_eq!(c.len(), m * n);
    // Finite inputs should give finite outputs for small dims
    assert!(c.iter().all(|v| v.is_finite()));
});
