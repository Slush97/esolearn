/// Whisper model configuration.
///
/// Defines the architecture hyperparameters for each model size variant.
#[derive(Clone, Debug)]
pub struct WhisperConfig {
    /// Model variant name (e.g. "tiny", "base", "small", "medium", "large-v3").
    pub name: String,
    /// Hidden dimension of the model.
    pub d_model: usize,
    /// Number of encoder layers.
    pub n_encoder_layers: usize,
    /// Number of encoder attention heads.
    pub n_encoder_heads: usize,
    /// Number of decoder layers.
    pub n_decoder_layers: usize,
    /// Number of decoder attention heads.
    pub n_decoder_heads: usize,
    /// Number of mel frequency bands (input feature dimension).
    pub n_mels: usize,
    /// Maximum audio context length in frames (1500 = 30s).
    pub n_audio_ctx: usize,
    /// Maximum text context length in tokens.
    pub n_text_ctx: usize,
    /// Vocabulary size.
    pub n_vocab: usize,
}

impl WhisperConfig {
    /// Whisper tiny (39M parameters).
    pub fn tiny() -> Self {
        Self {
            name: "tiny".into(),
            d_model: 384,
            n_encoder_layers: 4,
            n_encoder_heads: 6,
            n_decoder_layers: 4,
            n_decoder_heads: 6,
            n_mels: 80,
            n_audio_ctx: 1500,
            n_text_ctx: 448,
            n_vocab: 51865,
        }
    }

    /// Whisper base (74M parameters).
    pub fn base() -> Self {
        Self {
            name: "base".into(),
            d_model: 512,
            n_encoder_layers: 6,
            n_encoder_heads: 8,
            n_decoder_layers: 6,
            n_decoder_heads: 8,
            n_mels: 80,
            n_audio_ctx: 1500,
            n_text_ctx: 448,
            n_vocab: 51865,
        }
    }

    /// Whisper small (244M parameters).
    pub fn small() -> Self {
        Self {
            name: "small".into(),
            d_model: 768,
            n_encoder_layers: 12,
            n_encoder_heads: 12,
            n_decoder_layers: 12,
            n_decoder_heads: 12,
            n_mels: 80,
            n_audio_ctx: 1500,
            n_text_ctx: 448,
            n_vocab: 51865,
        }
    }

    /// Whisper medium (769M parameters).
    pub fn medium() -> Self {
        Self {
            name: "medium".into(),
            d_model: 1024,
            n_encoder_layers: 24,
            n_encoder_heads: 16,
            n_decoder_layers: 24,
            n_decoder_heads: 16,
            n_mels: 80,
            n_audio_ctx: 1500,
            n_text_ctx: 448,
            n_vocab: 51865,
        }
    }

    /// Whisper large-v3 (1.5B parameters).
    pub fn large_v3() -> Self {
        Self {
            name: "large-v3".into(),
            d_model: 1280,
            n_encoder_layers: 32,
            n_encoder_heads: 20,
            n_decoder_layers: 32,
            n_decoder_heads: 20,
            n_mels: 128, // large-v3 uses 128 mel bands
            n_audio_ctx: 1500,
            n_text_ctx: 448,
            n_vocab: 51866, // large-v3 has 1 extra token
        }
    }

    /// Whisper large-v3-turbo (809M parameters, distilled).
    pub fn large_v3_turbo() -> Self {
        Self {
            name: "large-v3-turbo".into(),
            d_model: 1280,
            n_encoder_layers: 32,
            n_encoder_heads: 20,
            n_decoder_layers: 4, // turbo: only 4 decoder layers
            n_decoder_heads: 20,
            n_mels: 128,
            n_audio_ctx: 1500,
            n_text_ctx: 448,
            n_vocab: 51866,
        }
    }

    /// Head dimension (`d_model` / `n_heads`).
    pub fn d_head_encoder(&self) -> usize {
        self.d_model / self.n_encoder_heads
    }

    /// Head dimension for decoder.
    pub fn d_head_decoder(&self) -> usize {
        self.d_model / self.n_decoder_heads
    }
}
