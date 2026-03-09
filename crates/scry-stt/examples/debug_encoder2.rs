//! Debug full encoder pipeline against Python reference.
//! Run: cargo run --release -p scry-stt --features safetensors --example debug_encoder2

#[cfg(not(feature = "safetensors"))]
compile_error!("Requires --features safetensors");

use std::path::PathBuf;

#[cfg(not(feature = "wgpu"))]
use scry_llm::backend::cpu::CpuBackend as Backend;
#[cfg(feature = "wgpu")]
use scry_llm::backend::wgpu::WgpuBackend as Backend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use scry_stt::checkpoint::load_whisper_checkpoint;
use scry_stt::model::config::WhisperConfig;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/whisper-tiny")
}

fn main() {
    let config = WhisperConfig::tiny();
    let model = load_whisper_checkpoint::<Backend>(&model_dir().join("model.safetensors"), &config)
        .expect("Failed to load model");

    let d = config.d_model; // 384

    // Load test mel [80, 32] from Python
    let mel_bytes = std::fs::read("/tmp/scry_enc_test_mel.bin").expect("Run Python first");
    let mel_data: Vec<f32> = mel_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(mel_data.len(), 80 * 32);
    let mel = Tensor::<Backend>::from_vec(mel_data, Shape::new(&[80, 32]));

    // Conv1 + GELU
    let x = model.encoder.conv1.forward(&mel);
    let xv = x.to_vec();
    eprintln!("After conv1: [{}, {}]", x.shape.dims()[0], x.shape.dims()[1]);
    eprintln!("  [:3,:3] = {:?}", &[xv[0], xv[1], xv[2], xv[32], xv[33], xv[34], xv[64], xv[65], xv[66]]);
    // Python: [-0.9480, -0.5467, -0.3502, -0.1536, -0.3674, -0.7334, 0.2979, 0.4962, 0.4146]

    let x = scry_llm::ops::gelu(&x);
    let xv = x.to_vec();
    eprintln!("After gelu1: [:3,:3] = {:?}", &[xv[0], xv[1], xv[2], xv[32], xv[33], xv[34], xv[64], xv[65], xv[66]]);
    // Python: [-0.1628, -0.1598, -0.1272, -0.0674, -0.1310, -0.1700, 0.1839, 0.3424, 0.2740]

    // Conv2 + GELU
    let x = model.encoder.conv2.forward(&x);
    let xv = x.to_vec();
    let out_len = x.shape.dims()[1];
    eprintln!("After conv2: [{}, {}]", x.shape.dims()[0], out_len);
    eprintln!("  [:3,:3] = {:?}", &[xv[0], xv[1], xv[2], xv[out_len], xv[out_len+1], xv[out_len+2], xv[2*out_len], xv[2*out_len+1], xv[2*out_len+2]]);
    // Python: [-0.4080, -0.1258, -0.1397, -3.3824, -3.7932, -3.7847, -0.2754, -0.2360, -0.3715]

    let x = scry_llm::ops::gelu(&x);
    let xv = x.to_vec();
    eprintln!("After gelu2: [:3,:3] = {:?}", &[xv[0], xv[1], xv[2], xv[out_len], xv[out_len+1], xv[out_len+2], xv[2*out_len], xv[2*out_len+1], xv[2*out_len+2]]);
    // Python: [-0.1394, -0.0566, -0.0621, -0.0010, -0.0002, -0.0002, -0.1078, -0.0960, -0.1319]

    // Fused transpose + positional embedding (same logic as encoder.forward)
    let pos_data = &model.encoder.pos_data;
    let mut fused = vec![0.0f32; out_len * d];
    for t in 0..out_len {
        let pos_row = t * d;
        for c in 0..d {
            fused[pos_row + c] = xv[c * out_len + t] + pos_data[pos_row + c];
        }
    }
    eprintln!("After transpose+pos: [{}, {}]", out_len, d);
    eprintln!("  [0,:5] = {:?}", &fused[..5]);
    eprintln!("  [1,:5] = {:?}", &fused[d..d+5]);
    // Python [0,:5]: [-0.1394, -0.00097, -0.1078, -0.1026, -0.00661]
    // Python [1,:5]: [0.7847, 0.8148, 0.6921, 0.6376, 0.8736]

    // Run through encoder block 0
    let x_tensor = Tensor::<Backend>::from_vec(fused.clone(), Shape::new(&[out_len, d]));

    // SA: layer_norm → QKV → attention → out_proj → residual
    let block = &model.encoder.blocks[0];
    let ln_out = block.attn_ln.forward(&x_tensor);

    // QKV
    let qkv = scry_llm::ops::matmul_bias(
        &ln_out, &block.attn.qkv_weight, &block.attn.qkv_bias,
        out_len, d, 3 * d, false, false,
    );
    let qkv_data = qkv.to_vec();
    // Check Q values match Python
    eprintln!("\nQ[0,:5]: {:?}", &qkv_data[..5]);

    // Split QKV and reshape heads
    let n_heads = config.n_encoder_heads;
    let d_head = d / n_heads;
    let (q_heads, k_heads, v_heads) = <Backend as scry_llm::backend::MathBackend>::split_qkv_reshape_heads(
        &qkv.data, out_len, n_heads, d_head,
    );

    // Batched attention
    let scores = <Backend as scry_llm::backend::MathBackend>::matmul_strided_batched(
        &q_heads, &k_heads, n_heads, out_len, d_head, out_len, false, true,
    );
    let scale = 1.0 / (d_head as f32).sqrt();
    let attn = <Backend as scry_llm::backend::MathBackend>::scaled_softmax(
        &scores, scale, &Shape::new(&[n_heads * out_len, out_len]),
    );
    let out_heads = <Backend as scry_llm::backend::MathBackend>::matmul_strided_batched(
        &attn, &v_heads, n_heads, out_len, out_len, d_head, false, false,
    );
    let head_concat = <Backend as scry_llm::backend::MathBackend>::reshape_from_heads(
        &out_heads, 1, out_len, n_heads, d_head,
    );

    // Output projection
    let hc = Tensor::<Backend>::from_vec(head_concat, Shape::new(&[out_len, d]));
    let sa_out = scry_llm::ops::matmul_bias(
        &hc, &block.attn.out_weight, &block.attn.out_bias,
        out_len, d, d, false, false,
    );

    // Residual
    let x_after_sa = scry_llm::ops::add(&x_tensor, &sa_out);
    let xv = x_after_sa.to_vec();
    eprintln!("\nAfter SA residual [0,:5]: {:?}", &xv[..5]);
    eprintln!("After SA residual [1,:5]: {:?}", &xv[d..d+5]);
    // Python: [0.3183, 0.8382, 1.0204, 0.5580, 1.0283]
    // Python: [0.5844, 0.6063, 0.6261, 0.4107, 0.8387]

    // MLP
    let mlp_in = block.mlp_ln.forward(&x_after_sa);
    let h = block.mlp_fc1.forward(&mlp_in);
    let h = scry_llm::ops::gelu(&h);
    let mlp_out = block.mlp_fc2.forward(&h);
    let x_after_block0 = scry_llm::ops::add(&x_after_sa, &mlp_out);
    let xv = x_after_block0.to_vec();
    eprintln!("\nAfter block 0 [0,:5]: {:?}", &xv[..5]);
    eprintln!("After block 0 [1,:5]: {:?}", &xv[d..d+5]);
    // Python: [0.4410, 1.0877, 0.9660, 0.4487, 1.1349]
    // Python: [0.4632, 0.9942, 0.7328, 0.2320, 0.7885]

    // Now run the FULL encoder forward and compare
    eprintln!("\n=== Full encoder.forward() ===");
    let mel_input = {
        let mel_bytes2 = std::fs::read("/tmp/scry_enc_test_mel.bin").unwrap();
        let mel_data2: Vec<f32> = mel_bytes2.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Tensor::<Backend>::from_vec(mel_data2, Shape::new(&[80, 32]))
    };
    let enc_out = model.encoder.forward(&mel_input);
    let enc_data = enc_out.to_vec();
    eprintln!("Encoder output: [{}, {}]", enc_out.shape.dims()[0], enc_out.shape.dims()[1]);
    eprintln!("  [0,:5] = {:?}", &enc_data[..5]);
    eprintln!("  [1,:5] = {:?}", &enc_data[d..d+5]);

    // Compare first block output (before block 1-3 and final LN)
    // If full encoder output differs from manual block-by-block, the bug is in encoder.forward()
    eprintln!("\n=== manual vs full comparison ===");
    eprintln!("Manual block0 [0,:5]: {:?}", &x_after_block0.to_vec()[..5]);
    eprintln!("(Full encoder runs all 4 blocks + final LN, so won't match block0 directly)");
}
