use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ScryLlmError {
    #[error("matmul dimension mismatch: [{m}x{k1}] @ [{k2}x{n}]")]
    MatmulMismatch {
        m: usize,
        k1: usize,
        k2: usize,
        n: usize,
    },

    #[error("broadcast incompatible: {a:?} vs {b:?}")]
    BroadcastIncompatible { a: Vec<usize>, b: Vec<usize> },

    #[error("weight load error: {0}")]
    WeightLoadError(String),

    #[error("data error: {0}")]
    DataError(String),

    #[error("checkpoint error: {0}")]
    CheckpointError(String),
}

pub type Result<T> = std::result::Result<T, ScryLlmError>;
