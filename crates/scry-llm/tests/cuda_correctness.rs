#![cfg(feature = "cuda")]

//! Correctness tests: compare CudaBackend against CpuBackend for forward-only MathBackend methods.
//! All results must match within tolerance (1e-4 for reductions, 1e-3 for softmax/layernorm).

use scry_llm::backend::cpu::CpuBackend;
use scry_llm::backend::cuda::{init_gpu, CudaBackend};
use scry_llm::backend::{DeviceBackend, MathBackend};
use scry_llm::tensor::shape::Shape;

fn init() {
    init_gpu(0);
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1e-6);
        assert!(
            diff / scale < tol || diff < tol,
            "{label}[{i}]: cpu={av}, gpu={bv}, diff={diff}"
        );
    }
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = fastrand::Rng::with_seed(seed);
    (0..n).map(|_| (rng.f32() - 0.5) * 2.0).collect()
}

// ---- Individual op tests ----

#[test]
fn matmul_nn() {
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

    assert_close(&cpu, &gpu, 1e-4, "matmul_nn");
}

#[test]
fn matmul_tn() {
    init();
    let m = 32;
    let k = 64;
    let n = 16;
    let a = rand_vec(k * m, 3); // [K, M] stored
    let b = rand_vec(k * n, 4);

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, true, false);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[k, m]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[k, n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, true, false));

    assert_close(&cpu, &gpu, 1e-4, "matmul_tn");
}

#[test]
fn matmul_nt() {
    init();
    let m = 32;
    let k = 64;
    let n = 16;
    let a = rand_vec(m * k, 5);
    let b = rand_vec(n * k, 6); // [N, K] stored

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, false, true);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[m, k]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[n, k]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, false, true));

    assert_close(&cpu, &gpu, 1e-4, "matmul_nt");
}

#[test]
fn matmul_tt() {
    init();
    let m = 16;
    let k = 32;
    let n = 24;
    let a = rand_vec(k * m, 7);
    let b = rand_vec(n * k, 8);

    let cpu = CpuBackend::matmul(&a, &b, m, k, n, true, true);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[k, m]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[n, k]));
    let gpu = CudaBackend::to_vec(&CudaBackend::matmul(&ga, &gb, m, k, n, true, true));

    assert_close(&cpu, &gpu, 1e-4, "matmul_tt");
}

#[test]
fn softmax_forward() {
    init();
    let rows = 8;
    let cols = 64;
    let input = rand_vec(rows * cols, 10);
    let shape = Shape::new(&[rows, cols]);

    let cpu = CpuBackend::softmax(&input, &shape);
    let gi = CudaBackend::from_vec(input, &shape);
    let gpu = CudaBackend::to_vec(&CudaBackend::softmax(&gi, &shape));

    assert_close(&cpu, &gpu, 1e-4, "softmax_fwd");
}

#[test]
fn layernorm_forward() {
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
    let gpu_out = CudaBackend::to_vec(&go);
    let gpu_mean = CudaBackend::to_vec(&gm);
    let gpu_rstd = CudaBackend::to_vec(&gr);

    assert_close(&cpu_out, &gpu_out, 1e-3, "layernorm_fwd_out");
    assert_close(&cpu_mean, &gpu_mean, 1e-4, "layernorm_fwd_mean");
    assert_close(&cpu_rstd, &gpu_rstd, 1e-3, "layernorm_fwd_rstd");
}

#[test]
fn gelu_forward() {
    init();
    let n = 256;
    let input = rand_vec(n, 30);

    let cpu = CpuBackend::gelu(&input);
    let gi = CudaBackend::from_vec(input, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::gelu(&gi));

    assert_close(&cpu, &gpu, 1e-4, "gelu_fwd");
}

#[test]
fn embedding_forward() {
    init();
    let vocab = 16;
    let dim = 32;
    let weight = rand_vec(vocab * dim, 50);
    let indices = vec![0usize, 3, 7, 15, 2];

    let cpu = CpuBackend::embedding(&weight, &indices, vocab, dim);
    let gw = CudaBackend::from_vec(weight, &Shape::new(&[vocab, dim]));
    let gpu = CudaBackend::to_vec(&CudaBackend::embedding(&gw, &indices, vocab, dim));

    assert_close(&cpu, &gpu, 1e-6, "embedding_fwd");
}

#[test]
fn mul_elementwise() {
    init();
    let n = 128;
    let a = rand_vec(n, 60);
    let b = rand_vec(n, 61);

    let cpu = CpuBackend::mul_elementwise(&a, &b);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[n]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::mul_elementwise(&ga, &gb));

    assert_close(&cpu, &gpu, 1e-6, "mul_elementwise");
}

#[test]
fn scale_op() {
    init();
    let n = 128;
    let a = rand_vec(n, 62);
    let scalar = 0.123f32;

    let cpu = CpuBackend::scale(&a, scalar);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[n]));
    let gpu = CudaBackend::to_vec(&CudaBackend::scale(&ga, scalar));

    assert_close(&cpu, &gpu, 1e-6, "scale");
}

#[test]
fn add_broadcast_2d() {
    init();
    let rows = 16;
    let cols = 32;
    let a = rand_vec(rows * cols, 90);
    let b = rand_vec(cols, 91); // [1, cols] broadcast
    let a_shape = Shape::new(&[rows, cols]);
    let b_shape = Shape::new(&[1, cols]);
    let out_shape = Shape::new(&[rows, cols]);

    let cpu = CpuBackend::add(&a, &b, &a_shape, &b_shape, &out_shape);
    let ga = CudaBackend::from_vec(a, &a_shape);
    let gb = CudaBackend::from_vec(b, &b_shape);
    let gpu = CudaBackend::to_vec(&CudaBackend::add(&ga, &gb, &a_shape, &b_shape, &out_shape));

    assert_close(&cpu, &gpu, 1e-6, "add_broadcast_2d");
}

#[test]
fn add_same_shape() {
    init();
    let n = 128;
    let a = rand_vec(n, 92);
    let b = rand_vec(n, 93);
    let shape = Shape::new(&[n]);

    let cpu = CpuBackend::add(&a, &b, &shape, &shape, &shape);
    let ga = CudaBackend::from_vec(a, &shape);
    let gb = CudaBackend::from_vec(b, &shape);
    let gpu = CudaBackend::to_vec(&CudaBackend::add(&ga, &gb, &shape, &shape, &shape));

    assert_close(&cpu, &gpu, 1e-6, "add_same_shape");
}

#[test]
fn concat_rows_op() {
    init();
    let a_rows = 4;
    let b_rows = 6;
    let cols = 16;
    let a = rand_vec(a_rows * cols, 100);
    let b = rand_vec(b_rows * cols, 101);

    let cpu = CpuBackend::concat_rows(&a, &b, a_rows, b_rows, cols);
    let ga = CudaBackend::from_vec(a, &Shape::new(&[a_rows, cols]));
    let gb = CudaBackend::from_vec(b, &Shape::new(&[b_rows, cols]));
    let gpu =
        CudaBackend::to_vec(&CudaBackend::concat_rows(&ga, &gb, a_rows, b_rows, cols));

    assert_close(&cpu, &gpu, 1e-6, "concat_rows");
}

#[test]
fn gather_columns_op() {
    init();
    let rows = 8;
    let total_cols = 48;
    let col_start = 16;
    let col_count = 16;
    let data = rand_vec(rows * total_cols, 110);

    let cpu = CpuBackend::gather_columns(&data, rows, total_cols, col_start, col_count);
    let gd = CudaBackend::from_vec(data, &Shape::new(&[rows, total_cols]));
    let gpu = CudaBackend::to_vec(&CudaBackend::gather_columns(
        &gd,
        rows,
        total_cols,
        col_start,
        col_count,
    ));

    assert_close(&cpu, &gpu, 1e-6, "gather_columns");
}

#[test]
fn scatter_columns_op() {
    init();
    let rows = 8;
    let total_cols = 48;
    let col_start = 16;
    let col_count = 16;
    let dst_data = rand_vec(rows * total_cols, 120);
    let src_data = rand_vec(rows * col_count, 121);

    let mut cpu_dst = dst_data.clone();
    CpuBackend::scatter_columns(&mut cpu_dst, &src_data, rows, total_cols, col_start, col_count);

    let mut gpu_dst = CudaBackend::from_vec(dst_data, &Shape::new(&[rows, total_cols]));
    let gpu_src = CudaBackend::from_vec(src_data, &Shape::new(&[rows, col_count]));
    CudaBackend::scatter_columns(&mut gpu_dst, &gpu_src, rows, total_cols, col_start, col_count);

    assert_close(
        &cpu_dst,
        &CudaBackend::to_vec(&gpu_dst),
        1e-6,
        "scatter_columns",
    );
}

#[test]
fn causal_mask_and_scale_op() {
    init();
    let seq_len = 8;
    let mut cpu_scores = rand_vec(seq_len * seq_len, 130);
    let scale = 0.125f32;

    let mut gpu_scores =
        CudaBackend::from_vec(cpu_scores.clone(), &Shape::new(&[seq_len, seq_len]));

    CpuBackend::apply_causal_mask_and_scale(&mut cpu_scores, seq_len, scale, f32::NEG_INFINITY);
    CudaBackend::apply_causal_mask_and_scale(&mut gpu_scores, seq_len, scale, f32::NEG_INFINITY);

    let gpu_vec = CudaBackend::to_vec(&gpu_scores);

    // Check lower triangle (scaled values)
    for s in 0..seq_len {
        for t in 0..=s {
            let idx = s * seq_len + t;
            let diff = (cpu_scores[idx] - gpu_vec[idx]).abs();
            assert!(
                diff < 1e-6,
                "causal_mask[{s},{t}]: cpu={}, gpu={}, diff={diff}",
                cpu_scores[idx],
                gpu_vec[idx]
            );
        }
        // Upper triangle should be -inf
        for t in (s + 1)..seq_len {
            let idx = s * seq_len + t;
            assert!(
                gpu_vec[idx].is_infinite() && gpu_vec[idx].is_sign_negative(),
                "causal_mask[{s},{t}] should be -inf, got {}",
                gpu_vec[idx]
            );
        }
    }
}

#[test]
fn sum_op() {
    init();
    let n = 256;
    let data = rand_vec(n, 160);

    let cpu_sum = CpuBackend::sum(&data);
    let gd = CudaBackend::from_vec(data, &Shape::new(&[n]));
    let gpu_sum = CudaBackend::sum(&gd);

    let diff = (cpu_sum - gpu_sum).abs();
    assert!(diff < 1e-3, "sum: cpu={cpu_sum}, gpu={gpu_sum}, diff={diff}");
}

// ---- Full forward pass comparison ----

#[test]
fn full_forward_pass() {
    use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
    init();

    let config = Gpt2Config {
        vocab_size: 64,
        max_seq_len: 16,
        d_model: 32,
        n_heads: 4,
        n_layers: 1,
        d_ff: 64,
    };

    let mut rng = fastrand::Rng::with_seed(42);
    let cpu_model = Gpt2Model::<CpuBackend>::new(config.clone(), &mut rng);

    let mut rng2 = fastrand::Rng::with_seed(42);
    let gpu_model = Gpt2Model::<CudaBackend>::new(config.clone(), &mut rng2);

    let tokens: Vec<usize> = (0..8).map(|i| i % 64).collect();

    let cpu_logits = cpu_model.forward(&tokens);
    let cpu_vec = CpuBackend::to_vec(&cpu_logits.data);

    let gpu_logits = gpu_model.forward(&tokens);
    let gpu_vec = CudaBackend::to_vec(&gpu_logits.data);

    assert_close(&cpu_vec, &gpu_vec, 1e-2, "full_forward");
}
