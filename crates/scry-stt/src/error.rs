/// Result type for scry-stt operations.
pub type Result<T> = std::result::Result<T, SttError>;

/// Top-level error type for speech-to-text operations.
#[derive(Debug, thiserror::Error)]
pub enum SttError {
    /// Audio loading or processing error.
    #[error("audio: {0}")]
    Audio(#[from] AudioError),

    /// Model loading or inference error.
    #[error("model: {0}")]
    Model(#[from] ModelError),

    /// Decoding error.
    #[error("decode: {0}")]
    Decode(#[from] DecodeError),

    /// Tokenizer error.
    #[error("tokenizer: {0}")]
    Tokenizer(String),

    /// I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Audio-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// Audio file has an unsupported format.
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Sample rate mismatch (Whisper requires 16kHz).
    #[error("sample rate {got}Hz, expected {expected}Hz")]
    SampleRateMismatch {
        /// The actual sample rate.
        got: u32,
        /// The expected sample rate.
        expected: u32,
    },

    /// Audio is too short for meaningful transcription.
    #[error("audio too short: {duration_ms}ms < minimum {min_ms}ms")]
    TooShort {
        /// Actual duration in milliseconds.
        duration_ms: u64,
        /// Minimum required duration.
        min_ms: u64,
    },
}

/// Model-specific errors.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// Checkpoint file could not be loaded.
    #[error("checkpoint: {0}")]
    Checkpoint(String),

    /// Weight tensor has unexpected shape.
    #[error("shape mismatch for '{name}': expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Parameter name.
        name: String,
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Required weight tensor is missing from checkpoint.
    #[error("missing weight: {0}")]
    MissingWeight(String),

    /// Model configuration is invalid.
    #[error("config: {0}")]
    Config(String),
}

/// Decoding errors.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Maximum decode length exceeded.
    #[error("max length {max} exceeded")]
    MaxLength {
        /// Maximum allowed tokens.
        max: usize,
    },

    /// Unexpected token encountered during decode.
    #[error("unexpected token {token_id} at position {position}")]
    UnexpectedToken {
        /// The token ID.
        token_id: usize,
        /// Position in the output sequence.
        position: usize,
    },
}
