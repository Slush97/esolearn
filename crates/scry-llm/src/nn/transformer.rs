use crate::backend::MathBackend;
use crate::nn::attention::CausalSelfAttention;
use crate::nn::kv_cache::LayerKvCache;
use crate::nn::layernorm::LayerNormModule;
use crate::nn::mlp::Mlp;
use crate::nn::Module;
use crate::ops;
use crate::tensor::Tensor;

/// A single transformer block: LN → attention → residual, LN → MLP → residual.
pub struct TransformerBlock<B: MathBackend> {
    pub ln1: LayerNormModule<B>,
    pub attn: CausalSelfAttention<B>,
    pub ln2: LayerNormModule<B>,
    pub mlp: Mlp<B>,
}

impl<B: MathBackend> TransformerBlock<B> {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, rng: &mut fastrand::Rng) -> Self {
        Self {
            ln1: LayerNormModule::new(d_model),
            attn: CausalSelfAttention::new(d_model, n_heads, rng),
            ln2: LayerNormModule::new(d_model),
            mlp: Mlp::new(d_model, d_ff, rng),
        }
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let ln1_out = self.ln1.forward(input);
        let attn_out = self.attn.forward(&ln1_out);
        let x = ops::add(input, &attn_out);

        let ln2_out = self.ln2.forward(&x);
        let mlp_out = self.mlp.forward(&ln2_out);
        ops::add(&x, &mlp_out)
    }

    /// Single-token forward with KV cache.
    pub fn forward_with_cache(&self, input: &Tensor<B>, cache: &mut LayerKvCache<B>) -> Tensor<B> {
        let ln1_out = self.ln1.forward(input);
        let attn_out = self.attn.forward_with_cache(&ln1_out, cache);
        let x = ops::add(input, &attn_out);

        let ln2_out = self.ln2.forward(&x);
        let mlp_out = self.mlp.forward(&ln2_out);
        ops::add(&x, &mlp_out)
    }
}

impl<B: MathBackend> Module<B> for TransformerBlock<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.ln1.parameters();
        params.extend(self.attn.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.mlp.parameters());
        params
    }
}
