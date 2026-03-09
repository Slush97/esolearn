pub mod attention;
pub mod config;
pub mod conv1d;
pub mod decoder;
pub mod encoder;

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::Tensor;

use self::attention::CrossKvCache;
use self::config::WhisperConfig;
use self::decoder::{DecoderKvCache, WhisperDecoder};
use self::encoder::WhisperEncoder;

/// Top-level Whisper model combining encoder and decoder.
pub struct WhisperModel<B: MathBackend> {
    /// Audio encoder.
    pub encoder: WhisperEncoder<B>,
    /// Text decoder.
    pub decoder: WhisperDecoder<B>,
    /// Model configuration.
    pub config: WhisperConfig,
}

impl<B: MathBackend> WhisperModel<B> {
    /// Create a new model with random initialization.
    pub fn new(config: WhisperConfig) -> Self {
        let mut rng = fastrand::Rng::with_seed(42);

        let encoder = WhisperEncoder::new(
            config.n_mels,
            config.d_model,
            config.n_encoder_layers,
            config.n_encoder_heads,
            config.n_audio_ctx,
            &mut rng,
        );

        let decoder = WhisperDecoder::new(
            config.n_vocab,
            config.d_model,
            config.n_decoder_layers,
            config.n_decoder_heads,
            config.n_text_ctx,
            &mut rng,
        );

        Self { encoder, decoder, config }
    }

    /// Encode a mel spectrogram into encoder hidden states.
    ///
    /// `mel`: `[n_mels, n_frames]` → returns `[n_audio_ctx, d_model]`.
    pub fn encode(&self, mel: &Tensor<B>) -> Tensor<B> {
        self.encoder.forward(mel)
    }

    /// Compute cross-attention KV caches from encoder output.
    ///
    /// Returns one `CrossKvCache` per decoder layer.
    pub fn compute_cross_kv_caches(&self, encoder_output: &Tensor<B>) -> Vec<CrossKvCache<B>> {
        self.decoder
            .blocks
            .iter()
            .map(|block| block.cross_attn.compute_kv_cache(encoder_output))
            .collect()
    }

    /// Create a new pre-allocated decoder KV cache.
    pub fn new_decoder_kv_cache(&self) -> DecoderKvCache<B> {
        DecoderKvCache::new(
            self.config.n_decoder_layers,
            self.config.n_decoder_heads,
            self.config.d_head_decoder(),
            self.config.n_text_ctx,
        )
    }

    /// Decode a single token, returning logits `[1, vocab_size]`.
    pub fn decode_step(
        &self,
        token_id: usize,
        position: usize,
        self_kv_cache: &mut DecoderKvCache<B>,
        cross_kv_caches: &[CrossKvCache<B>],
    ) -> Tensor<B> {
        self.decoder
            .forward_step(token_id, position, self_kv_cache, cross_kv_caches)
    }

    /// Decode a single token with optional sub-block profiling.
    pub fn decode_step_profiled(
        &self,
        token_id: usize,
        position: usize,
        self_kv_cache: &mut DecoderKvCache<B>,
        cross_kv_caches: &[CrossKvCache<B>],
        profile: bool,
    ) -> Tensor<B> {
        self.decoder
            .forward_step_profiled(token_id, position, self_kv_cache, cross_kv_caches, profile)
    }

    /// Run decoder blocks only — updates KV caches but skips the logit projection.
    ///
    /// Use for prompt tokens whose logits are discarded (saves ~1-2ms per call
    /// by skipping the `[1, d_model] × [vocab_size, d_model]^T` GEMV).
    pub fn decode_step_no_logits(
        &self,
        token_id: usize,
        position: usize,
        self_kv_cache: &mut DecoderKvCache<B>,
        cross_kv_caches: &[CrossKvCache<B>],
    ) {
        self.decoder
            .forward_step_blocks_only(token_id, position, self_kv_cache, cross_kv_caches);
    }
}

impl<B: MathBackend> Module<B> for WhisperModel<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }
}
