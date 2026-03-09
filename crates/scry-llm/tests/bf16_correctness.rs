#![cfg(feature = "bf16")]

//! Correctness tests: compare BF16-mode CudaBackend against CpuBackend.
//! Relaxed tolerances account for bf16 precision (~3 decimal digits).

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::cuda::{init_gpu_bf16, CudaBackend};
use scry_llm::backend::{DeviceBackend, MathBackend};
use scry_llm::tensor::shape::Shape;

fn init() {
    init_gpu_bf16(0);
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1e-6);
        assert!(
            diff / scale < tol || diff < tol,
            "{label}[{i}]: cpu={av}, gpu_bf16={bv}, diff={diff}"
        );
    }
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n).map(|_| (rng.f32() - 0.5) * 2.0).collect()
}

#[test]
fn bf16_matmul_nn() {
    init();
    let m = 64;
    let k = 48;
    let n = 32;
    let a = rand_vec(m * k, 1);
    let b = rand_vec(k * n, 2);

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, false, false);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[m, k]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[k, n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, false, false));

    assert_close(&cpu, &gpu, 2e-2, "bf16_matmul_nn");
}

#[test]
fn bf16_matmul_tn() {
    init();
    let m = 32;
    let k = 64;
    let n = 16;
    let a = rand_vec(k * m, 3);
    let b = rand_vec(k * n, 4);

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, true, false);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[k, m]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[k, n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, true, false));

    assert_close(&cpu, &gpu, 2e-2, "bf16_matmul_tn");
}

#[test]
fn bf16_matmul_nt() {
    init();
    let m = 32;
    let k = 64;
    let n = 16;
    let a = rand_vec(m * k, 5);
    let b = rand_vec(n * k, 6);

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, false, true);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[m, k]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[n, k]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, false, true));

    assert_close(&cpu, &gpu, 2e-2, "bf16_matmul_nt");
}

#[test]
fn bf16_softmax_forward() {
    init();
    let rows = 8;
    let cols = 64;
    let input = rand_vec(rows * cols, 10);
    let shape = Shape::new(&[rows, cols]);

    let cpu = CpuBackend::softmax(&input, &shape);
    let gi = CudaBackend::from_vec(input, &shape);
    let gpu = CudaBackend::to_vec(&CudaBackend::softmax(&gi, &shape));

    assert_close(&cpu, &gpu, 1e-2, "bf16_softmax_fwd");
}

#[test]
fn bf16_layernorm_forward() {
    init();
    let rows = 16;
    let d = 64;
    let input = rand_vec(rows * d, 20);
    let gamma = rand_vec(d, 21);
    let beta = rand_vec(d, 22);
    let shape = Shape::new(&[rows, d]);

    let (cpu_out, cpu_mean, cpu_rstd) =
        CpuBackend::layernorm(&input, &gamma, &beta, &shape, 1e-5);
    let gi = CudaBackend::from_vec(input, &shape);
    let gg = CudaBackend::from_vec(gamma, &Shape::new(&[d]));
    let gb = CudaBackend::from_vec(beta, &Shape::new(&[d]));
    let (go, gm, gr) = CudaBackend::layernorm(&gi, &gg, &gb, &shape, 1e-5);

    assert_close(&cpu_out, &CudaBackend::to_vec(&go), 2e-2, "bf16_layernorm_fwd_out");
    assert_close(&cpu_mean, &CudaBackend::to_vec(&gm), 1e-2, "bf16_layernorm_fwd_mean");
    assert_close(&cpu_rstd, &CudaBackend::to_vec(&gr), 2e-2, "bf16_layernorm_fwd_rstd");
}

#[test]
fn bf16_gelu_forward() {
    init();
    let n = 256;
    let input = rand_vec(n, 30);

    let cpu = CpuBackend::gelu(&input);
    let gi = CudaBackend::from_vec(input, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::gelu(&gi));

    assert_close(&cpu, &gpu, 1e-2, "bf16_gelu_fwd");
}

#[test]
fn bf16_embedding_forward() {
    init();
    let vocab = 16;
    let dim = 32;
    let weight = rand_vec(vocab * dim, 50);
    let indices = vec![0usize, 3, 7, 15, 2];

    let cpu = CpuBackend::embedding(&weight, &indices, vocab, dim);
    let gw = CudaBackend::from_vec(weight, &Shape::new(&[vocab, dim]));
    let gpu = CudaBackend::to_vec(&CudaBackend::embedding(&gw, &indices, vocab, dim));

    assert_close(&cpu, &gpu, 1e-2, "bf16_embedding_fwd");
}

#[test]
fn bf16_mul_elementwise() {
    init();
    let n = 128;
    let a = rand_vec(n, 60);
    let b = rand_vec(n, 61);

    let cpu = CpuBackend::mul_elementwise(&a, &b);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[n]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::mul_elementwise(&ga, &gb));

    assert_close(&cpu, &gpu, 1e-2, "bf16_mul_elementwise");
}

#[test]
fn bf16_scale() {
    init();
    let n = 128;
    let a = rand_vec(n, 62);
    let scalar = 0.123f32;

    let cpu = CpuBackend::scale(&a, scalar);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::scale(&ga, scalar));

    assert_close(&cpu, &gpu, 1e-2, "bf16_scale");
}

#[test]
fn bf16_add_broadcast_2d() {
    init();
    let rows = 16;
    let cols = 32;
    let a = rand_vec(rows * cols, 90);
    let b = rand_vec(cols, 91);
    let a_shape = Shape::new(&[rows, cols]);
    let b_shape = Shape::new(&[1, cols]);
    let out_shape = Shape::new(&[rows, cols]);

    let cpu = CpuBackend::add(&a, &b, &a_shape, &b_shape, &out_shape);
    let ga = CudaBackend::from_vec(a, &a_shape);
    let gb = CudaBackend::from_vec(b, &b_shape);
    let gpu = CudaBackend::to_vec(&CudaBackend::add(&ga, &gb, &a_shape, &b_shape, &out_shape));

    assert_close(&cpu, &gpu, 1e-2, "bf16_add_broadcast_2d");
}

#[test]
fn bf16_add_same_shape() {
    init();
    let n = 128;
    let a = rand_vec(n, 92);
    let b = rand_vec(n, 93);
    let shape = Shape::new(&[n]);

    let cpu = CpuBackend::add(&a, &b, &shape, &shape, &shape);
    let ga = CudaBackend::from_vec(a, &shape);
    let gb = CudaBackend::from_vec(b, &shape);
    let gpu = CudaBackend::to_vec(&CudaBackend::add(&ga, &gb, &shape, &shape, &shape));

    assert_close(&cpu, &gpu, 1e-2, "bf16_add_same_shape");
}

#[test]
fn bf16_sum() {
    init();
    let n = 256;
    let data = rand_vec(n, 160);

    let cpu_sum = CpuBackend::sum(&data);
    let gd = CudaBackend::from_vec(data, &Shape::new(&[n]));
    let gpu_sum = CudaBackend::sum(&gd);

    let diff = (cpu_sum - gpu_sum).abs();
    assert!(
        diff < 0.5,
        "bf16_sum: cpu={cpu_sum}, gpu={gpu_sum}, diff={diff}"
    );
}
