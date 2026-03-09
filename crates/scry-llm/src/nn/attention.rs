use crate::backend::MathBackend;
use crate::nn::init;
use crate::nn::kv_cache::LayerKvCache;
use crate::nn::Module;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Multi-head causal self-attention.
pub struct CausalSelfAttention<B: MathBackend> {
    pub qkv_weight: Tensor<B>,
    pub qkv_bias: Tensor<B>,
    pub proj_weight: Tensor<B>,
    pub proj_bias: Tensor<B>,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_head: usize,
}

impl<B: MathBackend> CausalSelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, rng: &mut fastrand::Rng) -> Self {
        let d_head = d_model / n_heads;
        let qkv_w = init::normal_vec(rng, d_model * 3 * d_model, 0.0, 0.02);
        let qkv_b = vec![0.0f32; 3 * d_model];
        let proj_w = init::normal_vec(rng, d_model * d_model, 0.0, 0.02);
        let proj_b = vec![0.0f32; d_model];
        Self {
            qkv_weight: Tensor::from_vec(qkv_w, Shape::new(&[d_model, 3 * d_model])),
            qkv_bias: Tensor::from_vec(qkv_b, Shape::new(&[3 * d_model])),
            proj_weight: Tensor::from_vec(proj_w, Shape::new(&[d_model, d_model])),
            proj_bias: Tensor::from_vec(proj_b, Shape::new(&[d_model])),
            n_heads,
            d_model,
            d_head,
        }
    }

    /// Full-sequence forward for prefill (no KV cache).
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let seq_len = input.shape.dims()[0];
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let d_head = self.d_head;

        // QKV = input @ W_qkv + b_qkv  => [seq, 3*d_model]
        let qkv_raw = B::matmul(
            &input.data,
            &self.qkv_weight.data,
            seq_len,
            d_model,
            3 * d_model,
            false,
            false,
        );
        let qkv_shape = Shape::new(&[seq_len, 3 * d_model]);
        let bias_shape = Shape::new(&[1, 3 * d_model]);
        let qkv = B::add(
            &qkv_raw,
            &self.qkv_bias.data,
            &qkv_shape,
            &bias_shape,
            &qkv_shape,
        );

        let mut head_concat_storage = B::zeros(&Shape::new(&[seq_len, d_model]));
        let scale = 1.0 / (d_head as f64).sqrt();

        for h in 0..n_heads {
            let q_h = B::gather_columns(&qkv, seq_len, 3 * d_model, h * d_head, d_head);
            let k_h = B::gather_columns(&qkv, seq_len, 3 * d_model, d_model + h * d_head, d_head);
            let v_h =
                B::gather_columns(&qkv, seq_len, 3 * d_model, 2 * d_model + h * d_head, d_head);

            let mut scores = B::matmul(&q_h, &k_h, seq_len, d_head, seq_len, false, true);
            B::apply_causal_mask_and_scale(&mut scores, seq_len, scale as f32, f32::NEG_INFINITY);

            let attn = B::softmax(&scores, &Shape::new(&[seq_len, seq_len]));
            let out_h = B::matmul(&attn, &v_h, seq_len, seq_len, d_head, false, false);

            B::scatter_columns(
                &mut head_concat_storage,
                &out_h,
                seq_len,
                d_model,
                h * d_head,
                d_head,
            );
        }

        // Output projection
        let proj_raw = B::matmul(
            &head_concat_storage,
            &self.proj_weight.data,
            seq_len,
            d_model,
            d_model,
            false,
            false,
        );
        let proj_shape = Shape::new(&[seq_len, d_model]);
        let pbias_shape = Shape::new(&[1, d_model]);
        let output_data = B::add(
            &proj_raw,
            &self.proj_bias.data,
            &proj_shape,
            &pbias_shape,
            &proj_shape,
        );

        Tensor::new(output_data, Shape::new(&[seq_len, d_model]))
    }

    /// Single-token forward with KV cache for autoregressive inference.
    #[allow(clippy::too_many_lines)]
    pub fn forward_with_cache(&self, input: &Tensor<B>, cache: &mut LayerKvCache<B>) -> Tensor<B> {
        let d_model = self.d_model;
        let n_heads = self.n_heads;
        let d_head = self.d_head;

        // QKV = input @ W_qkv + b_qkv => [1, 3*d_model]
        let qkv_raw = B::matmul(
            &input.data,
            &self.qkv_weight.data,
            1,
            d_model,
            3 * d_model,
            false,
            false,
        );
        let qkv_shape = Shape::new(&[1, 3 * d_model]);
        let bias_shape = Shape::new(&[1, 3 * d_model]);
        let qkv = B::add(
            &qkv_raw,
            &self.qkv_bias.data,
            &qkv_shape,
            &bias_shape,
            &qkv_shape,
        );
        let qkv_vec = B::to_vec(&qkv);

        let mut new_k: Vec<B::Storage> = Vec::with_capacity(n_heads);
        let mut new_v: Vec<B::Storage> = Vec::with_capacity(n_heads);
        let mut q_heads: Vec<Vec<f32>> = Vec::with_capacity(n_heads);

        for h in 0..n_heads {
            let q_offset = h * d_head;
            let k_offset = d_model + h * d_head;
            let v_offset = 2 * d_model + h * d_head;

            let q_h = qkv_vec[q_offset..q_offset + d_head].to_vec();
            let k_h = qkv_vec[k_offset..k_offset + d_head].to_vec();
            let v_h = qkv_vec[v_offset..v_offset + d_head].to_vec();

            q_heads.push(q_h);
            let head_shape = Shape::new(&[1, d_head]);
            new_k.push(B::from_vec(k_h, &head_shape));
            new_v.push(B::from_vec(v_h, &head_shape));
        }

        cache.append(new_k, new_v);
        let cached_len = cache.seq_len;

        let scale = 1.0 / (d_head as f64).sqrt();
        let mut head_concat = vec![0.0f32; d_model];

        for h in 0..n_heads {
            let q_h = &q_heads[h];
            let k_cached = &cache.k_per_head[h];
            let v_cached = &cache.v_per_head[h];

            let scores_raw = B::matmul(
                &B::from_vec(q_h.clone(), &Shape::new(&[1, d_head])),
                k_cached,
                1,
                d_head,
                cached_len,
                false,
                true,
            );
            let mut scores = B::to_vec(&scores_raw);
            for t in 0..cached_len {
                scores[t] = (f64::from(scores[t]) * scale) as f32;
            }

            let scores_storage = B::from_vec(scores, &Shape::new(&[1, cached_len]));
            let attn = B::softmax(&scores_storage, &Shape::new(&[1, cached_len]));

            let out_h = B::matmul(&attn, v_cached, 1, cached_len, d_head, false, false);
            let out_h_vec = B::to_vec(&out_h);

            for d in 0..d_head {
                head_concat[h * d_head + d] = out_h_vec[d];
            }
        }

        let hc_storage = B::from_vec(head_concat, &Shape::new(&[1, d_model]));
        let proj_raw = B::matmul(
            &hc_storage,
            &self.proj_weight.data,
            1,
            d_model,
            d_model,
            false,
            false,
        );
        let proj_shape = Shape::new(&[1, d_model]);
        let pbias_shape = Shape::new(&[1, d_model]);
        let output_data = B::add(
            &proj_raw,
            &self.proj_bias.data,
            &proj_shape,
            &pbias_shape,
            &proj_shape,
        );

        Tensor::new(output_data, Shape::new(&[1, d_model]))
    }
}

impl<B: MathBackend> Module<B> for CausalSelfAttention<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![
            &self.qkv_weight,
            &self.qkv_bias,
            &self.proj_weight,
            &self.proj_bias,
        ]
    }
}
