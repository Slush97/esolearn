//! Tests designed to run under Miri for undefined behavior detection.
//! Keep dimensions tiny (Miri is ~100x slower than normal execution).
//!
//! Run with: `cargo +nightly miri test -p scry-llm --no-default-features --test miri_safe`

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

type Cpu = CpuBackend;

// ============================================================
// Tensor basics
// ============================================================

#[test]
fn miri_tensor_zeros_ones() {
    let z = Tensor::<Cpu>::zeros(Shape::new(&[2, 3]));
    assert_eq!(z.to_vec(), vec![0.0; 6]);
    assert_eq!(z.numel(), 6);

    let o = Tensor::<Cpu>::ones(Shape::new(&[3]));
    assert_eq!(o.to_vec(), vec![1.0; 3]);
}

#[test]
fn miri_tensor_from_vec_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t = Tensor::<Cpu>::from_vec(data.clone(), Shape::new(&[2, 3]));
    assert_eq!(t.to_vec(), data);
}

// ============================================================
// Shape operations
// ============================================================

#[test]
fn miri_shape_broadcast() {
    let a = Shape::new(&[2, 1]);
    let b = Shape::new(&[1, 3]);
    let c = Shape::broadcast(&a, &b).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    assert_eq!(c.numel(), 6);
}

#[test]
fn miri_shape_strides() {
    let s = Shape::new(&[2, 3, 4]);
    let strides = s.strides();
    assert_eq!(&strides[..], &[12, 4, 1]);
}

#[test]
fn miri_shape_broadcast_strides() {
    let a = Shape::new(&[1, 3]);
    let target = Shape::new(&[2, 3]);
    let bs = a.broadcast_strides(&target);
    assert_eq!(&bs[..], &[0, 1]);
}

// ============================================================
// Backend ops (tiny sizes)
// ============================================================

#[test]
fn miri_matmul_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = CpuBackend::matmul(&a, &b, 2, 2, 2, false, false);
    assert_eq!(c.len(), 4);
    // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn miri_matmul_transposed() {
    let a = vec![1.0, 3.0, 2.0, 4.0]; // [K=2, M=2] for trans_a
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = CpuBackend::matmul(&a, &b, 2, 2, 2, true, false);
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn miri_add_broadcast() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![10.0, 20.0, 30.0];
    let a_shape = Shape::new(&[2, 3]);
    let b_shape = Shape::new(&[1, 3]);
    let out_shape = Shape::new(&[2, 3]);
    let c = CpuBackend::add(&a, &b, &a_shape, &b_shape, &out_shape);
    assert_eq!(c, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn miri_softmax() {
    let input = vec![1.0, 2.0, 3.0];
    let shape = Shape::new(&[1, 3]);
    let output = CpuBackend::softmax(&input, &shape);
    let sum: f64 = output.iter().map(|&v| f64::from(v)).sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn miri_layernorm() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0];
    let beta = vec![0.0, 0.0];
    let shape = Shape::new(&[2, 2]);
    let (out, mean, rstd) = CpuBackend::layernorm(&input, &gamma, &beta, &shape, 1e-5);
    assert_eq!(out.len(), 4);
    assert_eq!(mean.len(), 2);
    assert_eq!(rstd.len(), 2);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn miri_gelu() {
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let output = CpuBackend::gelu(&input);
    assert_eq!(output.len(), 4);
    assert!((output[1]).abs() < 1e-6); // gelu(0) = 0
}

#[test]
fn miri_embedding() {
    let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [3, 2]
    let indices = vec![0usize, 2, 1];
    let out = CpuBackend::embedding(&weight, &indices, 3, 2);
    assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0]);
}

#[test]
fn miri_sum() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let s = CpuBackend::sum(&input);
    assert!((s - 10.0).abs() < 1e-6);
}

// ============================================================
// Module forward passes (tiny sizes)
// ============================================================

#[test]
fn miri_gpt2_tiny_forward() {
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 4,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config, &mut rng);

    let token_ids = &[0, 2, 4];
    let logits = model.forward(token_ids);
    assert_eq!(logits.shape.dims(), &[3, 5]);
    assert!(logits.to_vec().iter().all(|v| v.is_finite()));
}

#[test]
fn miri_generate_tiny() {
    use scry_llm::generate::{generate, SamplingConfig};
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};

    let config = Gpt2Config {
        vocab_size: 5,
        max_seq_len: 16,
        d_model: 4,
        n_heads: 2,
        n_layers: 1,
        d_ff: 8,
    };
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let sampling = SamplingConfig {
        temperature: 0.0, // greedy for determinism
        top_k: 0,
        top_p: 1.0,
        max_tokens: 3,
    };
    let mut gen_rng = fastrand::Rng::with_seed(99);
    let tokens = generate(&model, &[0, 1], &sampling, &mut gen_rng);
    assert_eq!(tokens.len(), 3);
    for &t in &tokens {
        assert!(t < config.vocab_size);
    }
}
