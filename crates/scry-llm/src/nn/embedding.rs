use crate::backend::MathBackend;
use crate::nn::init;
use crate::nn::Module;
use crate::ops;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Token + position embedding layer.
pub struct EmbeddingLayer<B: MathBackend> {
    pub token_embedding: Tensor<B>,
    pub position_embedding: Tensor<B>,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
}

impl<B: MathBackend> EmbeddingLayer<B> {
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        rng: &mut fastrand::Rng,
    ) -> Self {
        let tok_data = init::normal_vec(rng, vocab_size * d_model, 0.0, 0.02);
        let pos_data = init::normal_vec(rng, max_seq_len * d_model, 0.0, 0.02);
        Self {
            token_embedding: Tensor::from_vec(tok_data, Shape::new(&[vocab_size, d_model])),
            position_embedding: Tensor::from_vec(pos_data, Shape::new(&[max_seq_len, d_model])),
            vocab_size,
            max_seq_len,
            d_model,
        }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Tensor<B> {
        let seq_len = token_ids.len();
        let positions: Vec<usize> = (0..seq_len).collect();

        let tok_emb = ops::embedding(
            &self.token_embedding,
            token_ids,
            self.vocab_size,
            self.d_model,
        );
        let pos_emb = ops::embedding(
            &self.position_embedding,
            &positions,
            self.max_seq_len,
            self.d_model,
        );
        ops::add(&tok_emb, &pos_emb)
    }
}

impl<B: MathBackend> Module<B> for EmbeddingLayer<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.token_embedding, &self.position_embedding]
    }
}
