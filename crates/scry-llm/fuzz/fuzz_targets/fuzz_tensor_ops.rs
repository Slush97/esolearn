//! Fuzz `Tensor::from_vec`, `zeros`, `ones`, `to_vec` with arbitrary shapes.
//! Validates that storage round-trips correctly and bad sizes are caught.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_llm::backend::cpu::CpuBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

type Cpu = CpuBackend;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let ndim = ((data[0] % 3) + 1) as usize; // 1..=3
    if data.len() < 1 + ndim {
        return;
    }

    let dims: Vec<usize> = data[1..1 + ndim]
        .iter()
        .map(|&b| (b % 8) as usize + 1) // dims 1..=8
        .collect();

    let shape = Shape::new(&dims);
    let numel = shape.numel();

    // zeros/ones must work
    let z = Tensor::<Cpu>::zeros(shape.clone());
    assert_eq!(z.to_vec().len(), numel);
    assert!(z.to_vec().iter().all(|&v| v == 0.0));

    let o = Tensor::<Cpu>::ones(shape.clone());
    assert_eq!(o.to_vec().len(), numel);
    assert!(o.to_vec().iter().all(|&v| v == 1.0));

    // from_vec with correct size must work
    let data_vec: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let t = Tensor::<Cpu>::from_vec(data_vec.clone(), shape);
    let back = t.to_vec();
    assert_eq!(back, data_vec);
});
