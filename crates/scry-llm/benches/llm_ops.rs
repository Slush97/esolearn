use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scry_llm::backend::cpu::CpuBackend;
#[cfg(feature = "cuda")]
use scry_llm::backend::DeviceBackend;
use scry_llm::backend::MathBackend;
use scry_llm::nn::gpt2::{Gpt2Config, Gpt2Model};
use scry_llm::tensor::shape::Shape;

type Cpu = CpuBackend;

fn bench_config() -> Gpt2Config {
    Gpt2Config {
        vocab_size: 256,
        max_seq_len: 64,
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
    }
}

// ============================================================
// Group 1: Inference forward pass
// ============================================================

fn forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_pass");
    let config = bench_config();
    let mut rng = fastrand::Rng::with_seed(42);
    let model = Gpt2Model::<Cpu>::new(config.clone(), &mut rng);

    let seq = 32;
    let tokens: Vec<usize> = (0..seq).map(|i| i % config.vocab_size).collect();

    // Warmup
    for _ in 0..2 {
        let _ = black_box(model.forward(&tokens));
    }

    group.bench_function("forward_seq32", |b| {
        b.iter(|| black_box(model.forward(black_box(&tokens))));
    });

    group.finish();
}

// ============================================================
// Group 2: Micro ops
// ============================================================

fn ops_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_matmul");

    // matmul 64x64
    {
        let a = vec![0.01f32; 64 * 64];
        let b = vec![0.01f32; 64 * 64];
        for _ in 0..2 {
            let _ = black_box(CpuBackend::matmul(&a, &b, 64, 64, 64, false, false));
        }
        group.bench_function("matmul_64x64", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::matmul(
                    black_box(&a),
                    black_box(&b),
                    64, 64, 64, false, false,
                ))
            });
        });
    }

    // matmul 768x768 (GPT-2 small scale)
    {
        let a = vec![0.01f32; 768 * 768];
        let b = vec![0.01f32; 768 * 768];
        for _ in 0..2 {
            let _ = black_box(CpuBackend::matmul(&a, &b, 768, 768, 768, false, false));
        }
        group.bench_function("matmul_768x768", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::matmul(
                    black_box(&a),
                    black_box(&b),
                    768, 768, 768, false, false,
                ))
            });
        });
    }

    group.finish();
}

fn ops_micro(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_micro");

    // layernorm [32, 32]
    {
        let input = vec![0.5f32; 32 * 32];
        let gamma = vec![1.0f32; 32];
        let beta = vec![0.0f32; 32];
        let shape = Shape::new(&[32, 32]);
        for _ in 0..2 {
            let _ = black_box(CpuBackend::layernorm(&input, &gamma, &beta, &shape, 1e-5));
        }
        group.bench_function("layernorm_1024", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::layernorm(
                    black_box(&input),
                    black_box(&gamma),
                    black_box(&beta),
                    black_box(&shape),
                    1e-5,
                ))
            });
        });
    }

    // softmax [4, 32]
    {
        let input = vec![0.1f32; 4 * 32];
        let shape = Shape::new(&[4, 32]);
        for _ in 0..2 {
            let _ = black_box(CpuBackend::softmax(&input, &shape));
        }
        group.bench_function("softmax_128", |bench| {
            bench.iter(|| black_box(CpuBackend::softmax(black_box(&input), black_box(&shape))));
        });
    }

    // rmsnorm [32, 32]
    {
        let input = vec![0.5f32; 32 * 32];
        let weight = vec![1.0f32; 32];
        let shape = Shape::new(&[32, 32]);
        for _ in 0..2 {
            let _ = black_box(CpuBackend::rmsnorm(&input, &weight, &shape, 1e-5));
        }
        group.bench_function("rmsnorm_1024", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::rmsnorm(
                    black_box(&input),
                    black_box(&weight),
                    black_box(&shape),
                    1e-5,
                ))
            });
        });
    }

    // swiglu [1024]
    {
        let gate = vec![0.5f32; 1024];
        let up = vec![0.5f32; 1024];
        for _ in 0..2 {
            let _ = black_box(CpuBackend::swiglu(&gate, &up));
        }
        group.bench_function("swiglu_1024", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::swiglu(black_box(&gate), black_box(&up)))
            });
        });
    }

    group.finish();
}

// ============================================================
// Group 3: GEMV benchmarks (decode-relevant dimensions)
// ============================================================

fn gemv_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_bench");

    // Self-attn QKV: [1, 384] @ [384, 1152] — gemv_f32
    {
        let a = vec![0.01f32; 384];
        let b = vec![0.01f32; 384 * 1152];
        for _ in 0..2 {
            let _ = black_box(CpuBackend::matmul(&a, &b, 1, 384, 1152, false, false));
        }
        group.bench_function("qkv_384x1152", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::matmul(
                    black_box(&a), black_box(&b), 1, 384, 1152, false, false,
                ))
            });
        });
    }

    // Cross-attn Q@K^T with trans_b=true: [1, 64] @ [1500, 64]^T — gemv_trans_b (current)
    {
        let q = vec![0.01f32; 64];
        let k = vec![0.01f32; 1500 * 64];
        for _ in 0..2 {
            let _ = black_box(CpuBackend::matmul(&q, &k, 1, 64, 1500, false, true));
        }
        group.bench_function("cross_qkt_trans_64x1500", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::matmul(
                    black_box(&q), black_box(&k), 1, 64, 1500, false, true,
                ))
            });
        });
    }

    // Cross-attn Q@K_t with trans_b=false: [1, 64] @ [64, 1500] — gemv_f32 (proposed)
    {
        let q = vec![0.01f32; 64];
        let kt = vec![0.01f32; 64 * 1500];
        for _ in 0..2 {
            let _ = black_box(CpuBackend::matmul(&q, &kt, 1, 64, 1500, false, false));
        }
        group.bench_function("cross_qkt_notrans_64x1500", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::matmul(
                    black_box(&q), black_box(&kt), 1, 64, 1500, false, false,
                ))
            });
        });
    }

    // Logit projection: [1, 384] @ [384, 51865] — gemv_f32
    {
        let a = vec![0.01f32; 384];
        let b = vec![0.01f32; 384 * 51865];
        for _ in 0..2 {
            let _ = black_box(CpuBackend::matmul(&a, &b, 1, 384, 51865, false, false));
        }
        group.bench_function("logit_384x51865", |bench| {
            bench.iter(|| {
                black_box(CpuBackend::matmul(
                    black_box(&a), black_box(&b), 1, 384, 51865, false, false,
                ))
            });
        });
    }

    group.finish();
}

// ============================================================
// Group 4: GPU ops (only with cuda feature)
// ============================================================

#[cfg(feature = "cuda")]
fn gpu_ops(c: &mut Criterion) {
    use scry_llm::backend::cuda::{init_gpu, CudaBackend};

    init_gpu(0);
    type Gpu = CudaBackend;

    let mut group = c.benchmark_group("gpu_ops");

    // matmul 768x768
    {
        let a_data = vec![0.01f32; 768 * 768];
        let b_data = vec![0.01f32; 768 * 768];
        let a = Gpu::from_vec(a_data, &Shape::new(&[768, 768]));
        let b = Gpu::from_vec(b_data, &Shape::new(&[768, 768]));
        CudaBackend::synchronize();

        for _ in 0..2 {
            let _ = black_box(Gpu::matmul(&a, &b, 768, 768, 768, false, false));
        }
        CudaBackend::synchronize();

        group.bench_function("matmul_768x768", |bench| {
            bench.iter(|| {
                let r = black_box(Gpu::matmul(
                    black_box(&a),
                    black_box(&b),
                    768, 768, 768, false, false,
                ));
                CudaBackend::synchronize();
                r
            });
        });
    }

    // Forward pass
    {
        let config = Gpt2Config {
            vocab_size: 256,
            max_seq_len: 128,
            d_model: 128,
            n_heads: 4,
            n_layers: 2,
            d_ff: 256,
        };
        let mut rng = fastrand::Rng::with_seed(42);
        let model = Gpt2Model::<Gpu>::new(config.clone(), &mut rng);
        let tokens: Vec<usize> = (0..64).map(|i| i % config.vocab_size).collect();

        for _ in 0..2 {
            let _ = black_box(model.forward(&tokens));
            CudaBackend::synchronize();
        }

        group.bench_function("forward_seq64_d128", |b| {
            b.iter(|| {
                let r = black_box(model.forward(black_box(&tokens)));
                CudaBackend::synchronize();
                r
            });
        });
    }

    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(benches, forward_pass, ops_matmul, ops_micro, gemv_bench, gpu_ops);
#[cfg(not(feature = "cuda"))]
criterion_group!(benches, forward_pass, ops_matmul, ops_micro, gemv_bench);
criterion_main!(benches);
