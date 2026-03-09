//! Fuzz the attention forward op with random dimensions and weights.
//! Must never panic, produce NaN, or Inf.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::nn::attention::CausalSelfAttention;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

type Cpu = CpuBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }

    let seq_len = (data[0] % 4) as usize + 1; // 1..=4
    let d_model = if data[1] % 2 == 0 { 2 } else { 4 };
    let n_heads = if d_model == 2 { 1 } else { (data[2] % 2) as usize + 1 }; // 1 or 2

    // Build small input from fuzz bytes, scaled to prevent overflow
    let make_data = |len: usize, offset: usize| -> Vec<f32> {
        (0..len)
            .map(|i| {
                let byte = data.get(offset + i).copied().unwrap_or(128);
                (byte as f32 - 128.0) * 0.001
            })
            .collect()
    };

    let input_data = make_data(seq_len * d_model, 6);
    let input = Tensor::<Cpu>::from_vec(input_data, Shape::new(&[seq_len, d_model]));

    let mut rng = fastrand::Rng::with_seed(data[3] as u64);
    let attn = CausalSelfAttention::<Cpu>::new(d_model, n_heads, &mut rng);

    let out = attn.forward(&input);
    let out_vec = out.to_vec();
    assert!(out_vec.iter().all(|v| v.is_finite()), "attention output contains NaN/Inf");
});
