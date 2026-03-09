//! Debug encoder components against Python reference values.
//! Run: cargo run --release -p scry-stt --features safetensors --example debug_encoder

#[cfg(not(feature = "safetensors"))]
compile_error!("Requires --features safetensors");

use std::path::PathBuf;

#[cfg(not(feature = "wgpu"))]
use scry_llm::backend::cpu::CpuBackend as Backend;
#[cfg(feature = "wgpu")]
use scry_llm::backend::wgpu::WgpuBackend as Backend;
use scry_llm::backend::MathBackend;
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
    let n_heads = config.n_encoder_heads; // 6
    let d_head = d / n_heads; // 64

    // ========================================================================
    // TEST 1: Conv1D
    // ========================================================================
    // Load test mel from Python (numpy seed 42, randn(80,16) * 0.1)
    let mel_bytes = std::fs::read("/tmp/scry_test_mel.bin").expect("Run Python script first");
    let mel_data: Vec<f32> = mel_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(mel_data.len(), 80 * 16);
    let mel_tensor = Tensor::<Backend>::from_vec(mel_data.clone(), Shape::new(&[80, 16]));

    eprintln!("=== Conv1D test ===");
    eprintln!("Input[:3,:5]: {:?}", &mel_data[..5]);

    let conv1_out = model.encoder.conv1.forward(&mel_tensor);
    let conv1_data = conv1_out.to_vec();
    eprintln!("Conv1 out[0,:5]: {:?}", &conv1_data[..5]);
    eprintln!("Conv1 out[1,:5]: {:?}", &conv1_data[16..21]);
    let c1_min = conv1_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let c1_max = conv1_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let c1_mean = conv1_data.iter().sum::<f32>() / conv1_data.len() as f32;
    eprintln!("Conv1 out stats: min={c1_min:.6}, max={c1_max:.6}, mean={c1_mean:.6}");

    // Python reference:
    // Conv1 out[0,:5]: [-0.6196, -0.7429, -0.7979, -0.7521, -0.6620]
    // Conv1 out[1,:5]: [-0.7416, -0.7474, -0.7536, -0.6622, -0.7020]
    // stats: min=-6.369, max=1.101, mean=-0.344

    // ========================================================================
    // TEST 2: Encoder self-attention (seq=4)
    // ========================================================================
    eprintln!("\n=== Encoder Self-Attention test (seq=4) ===");

    // Load test input from Python (numpy seed 123, randn(4,384) * 0.1)
    let sa_bytes = std::fs::read("/tmp/scry_test_sa_input.bin").expect("Need sa input");
    let sa_data: Vec<f32> = sa_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(sa_data.len(), 4 * 384);

    eprintln!("Input[:2,:5]: {:?}", &sa_data[..5]);
    eprintln!("Input[1,:5]:  {:?}", &sa_data[384..389]);

    let sa_tensor = Tensor::<Backend>::from_vec(sa_data.clone(), Shape::new(&[4, d]));

    // Layer norm
    let ln = &model.encoder.blocks[0].attn_ln;
    let normed = ln.forward(&sa_tensor);
    let normed_data = normed.to_vec();
    eprintln!("After LN [:2,:5]: {:?}", &normed_data[..5]);
    eprintln!("After LN [1,:5]:  {:?}", &normed_data[384..389]);
    // Python: [[-1.0699, 11.256, 1.498, -1.966, -0.349], [0.0522, 3.708, -0.614, 2.438, 1.138]]

    // Fused QKV
    let attn = &model.encoder.blocks[0].attn;
    let qkv = scry_llm::ops::matmul_bias(
        &normed, &attn.qkv_weight, &attn.qkv_bias,
        4, d, 3 * d, false, false,
    );
    let qkv_data = qkv.to_vec();

    // Extract Q, K, V from fused QKV [4, 3*384] = [4, 1152]
    // Layout: each row is [Q_d0..Q_d383, K_d0..K_d383, V_d0..V_d383]
    eprintln!("Q[0,:5] (from fused): {:?}", &qkv_data[..5]);
    eprintln!("Q[1,:5] (from fused): {:?}", &qkv_data[3*d..3*d+5]);
    // Python Q[:2,:5]: [[-4.857, 1.944, 0.022, -0.973, -3.110], [-4.871, -3.188, -0.135, 2.119, -1.606]]

    eprintln!("K[0,:5] (from fused): {:?}", &qkv_data[d..d+5]);
    eprintln!("K[1,:5] (from fused): {:?}", &qkv_data[3*d+d..3*d+d+5]);
    // Python K[:2,:5]: [[-1.272, 2.494, -0.824, 0.169, -1.969], [-1.397, -1.134, -2.779, 3.529, 0.350]]

    // Now test split_qkv_reshape_heads
    let (q_heads, k_heads, v_heads) = Backend::split_qkv_reshape_heads(&qkv.data, 4, n_heads, d_head);

    // Python Q_h[0,0,:5] (head0, pos0): [-4.857, 1.944, 0.022, -0.973, -3.110]
    // Python Q_h[1,0,:5] (head1, pos0): [1.118, 0.939, -3.743, -0.018, -2.685]
    eprintln!("\nQ_h[head0,pos0,:5]: {:?}", &q_heads[..5]);
    eprintln!("Q_h[head1,pos0,:5]: {:?}", &q_heads[4*d_head..4*d_head+5]);
    // head1 starts at offset: head1 * seq * d_head = 1 * 4 * 64 = 256
    eprintln!("Q_h layout check - head0,pos1,:5: {:?}", &q_heads[d_head..d_head+5]);

    // Compute attention scores manually for head 0
    // scores[0] = Q_h[0] @ K_h[0].T  (both [4, 64])
    // Q_h[0] is at q_heads[0..4*64], K_h[0] is at k_heads[0..4*64]
    let q_h0 = &q_heads[..4 * d_head]; // [4, 64]
    let k_h0 = &k_heads[..4 * d_head]; // [4, 64]

    // scores = Q @ K^T → [4, 4]
    let scores_h0 = Backend::matmul(&q_h0.to_vec(), &k_h0.to_vec(), 4, d_head, 4, false, true);
    let scale = 1.0 / (d_head as f32).sqrt();
    let scores_scaled: Vec<f32> = scores_h0.iter().map(|x| x * scale).collect();
    eprintln!("\nScores[head0] (scaled):");
    for row in 0..4 {
        eprintln!("  {:?}", &scores_scaled[row*4..(row+1)*4]);
    }
    // Python scores[0]:
    // [[21.537,  6.253,  4.517, 12.824]
    //  [-4.508, 15.921, 11.590, -8.306]
    //  [-9.820,  2.432, 14.960, -8.719]
    //  [ 6.138, -4.618, -0.203,  5.750]]

    // Now run the full encoder block forward (block.forward is also private, so use the whole encoder)
    // Instead, test via matmul_strided_batched which is what the encoder attention uses
    let scores_batched = Backend::matmul_strided_batched(
        &q_heads, &k_heads, n_heads, 4, d_head, 4, false, true,
    );
    eprintln!("\nBatched scores[head0] (unscaled):");
    for row in 0..4 {
        let s: Vec<f32> = scores_batched[row*4..(row+1)*4].iter().map(|x| x * scale).collect();
        eprintln!("  {:?}", &s);
    }

    // Full softmax
    let attn_weights = Backend::scaled_softmax(
        &scores_batched, scale, &Shape::new(&[n_heads * 4, 4]),
    );
    eprintln!("\nAttn weights[head0]:");
    for row in 0..4 {
        eprintln!("  {:?}", &attn_weights[row*4..(row+1)*4]);
    }

    // attn @ V
    let out_heads = Backend::matmul_strided_batched(
        &attn_weights, &v_heads, n_heads, 4, 4, d_head, false, false,
    );

    // Reshape from heads
    let head_concat = Backend::reshape_from_heads(&out_heads, 1, 4, n_heads, d_head);
    eprintln!("\nAttn concat[:2,:5]: {:?}", &head_concat[..5]);
    eprintln!("Attn concat[1,:5]:  {:?}", &head_concat[384..389]);
    // Python: [[1.344, -0.840, -0.806, 4.591, -0.475], [1.465, 0.494, -0.840, 3.638, -1.190]]

    // Output projection
    let out_w = &model.encoder.blocks[0].attn.out_weight;
    let out_b = &model.encoder.blocks[0].attn.out_bias;
    let hc_tensor = Tensor::<Backend>::from_vec(head_concat, Shape::new(&[4, d]));
    let sa_out = scry_llm::ops::matmul_bias(&hc_tensor, out_w, out_b, 4, d, d, false, false);
    let sa_out_data = sa_out.to_vec();
    eprintln!("\nSA output[:2,:5]: {:?}", &sa_out_data[..5]);
    eprintln!("SA output[1,:5]:  {:?}", &sa_out_data[384..389]);
    // Python: [[0.5834, 0.1185, 0.1157, 2.9716, 1.2789], [0.2145, 1.1394, 1.3094, -0.2159, 0.5910]]
}
