pub mod backward;
pub mod ops;

use crate::backend::DeviceBackend;
use crate::tensor::shape::Shape;
use crate::tensor::TensorId;

/// What data each operation saves for backward.
pub enum SavedData<B: DeviceBackend> {
    Matmul {
        a: B::Storage,
        b: B::Storage,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    },
    Add {
        a_shape: Shape,
        b_shape: Shape,
        out_shape: Shape,
    },
    Softmax {
        output: B::Storage,
        shape: Shape,
    },
    LayerNorm {
        input: B::Storage,
        gamma: B::Storage,
        mean: B::Storage,
        rstd: B::Storage,
        shape: Shape,
        gamma_id: TensorId,
        beta_id: TensorId,
    },
    Gelu {
        input: B::Storage,
    },
    CrossEntropy {
        logits: B::Storage,
        targets: Vec<usize>,
        batch: usize,
        vocab: usize,
    },
    Embedding {
        indices: Vec<usize>,
        vocab: usize,
        dim: usize,
        weight_id: TensorId,
    },
    Sum {
        input_shape: Shape,
    },
    Attention {
        input: B::Storage,
        qkv_weight: B::Storage,
        proj_weight: B::Storage,
        attn_weights: Vec<B::Storage>,
        q_per_head: Vec<B::Storage>,
        k_per_head: Vec<B::Storage>,
        v_per_head: Vec<B::Storage>,
        attn_dropout_masks: Vec<B::Storage>,
        head_concat: B::Storage,
        n_heads: usize,
        d_model: usize,
        d_head: usize,
        seq_len: usize,
        qkv_weight_id: TensorId,
        qkv_bias_id: TensorId,
        proj_weight_id: TensorId,
        proj_bias_id: TensorId,
    },
    Dropout {
        mask: B::Storage,
    },
    /// Placeholder for a checkpointed segment of transformer blocks.
    /// Stores the input data/shape needed to recompute the segment's forward pass.
    Checkpoint {
        input_data: B::Storage,
        input_shape: Shape,
        /// Indices of the transformer blocks in this segment (start..end).
        block_start: usize,
        block_end: usize,
        dropout_rate: f32,
        rng_seed: u64,
        /// If set, this checkpoint was from a batched forward pass.
        batch_size: Option<usize>,
        seq_len: Option<usize>,
    },
    /// Batched attention: contiguous tensors for all batch×heads (no per-item vectors).
    AttentionBatched {
        input: B::Storage,
        qkv_weight: B::Storage,
        proj_weight: B::Storage,
        /// `[B*H, S, S]` — attention weights (pre-dropout softmax output)
        attn_weights: B::Storage,
        /// `[B*H, S, d_head]` — Q, K, V in head-first layout
        q_heads: B::Storage,
        k_heads: B::Storage,
        v_heads: B::Storage,
        /// `[B*H*S*S]` — dropout mask (scale values or 1.0 if no dropout)
        attn_dropout_mask: B::Storage,
        /// `[B*S, D]` — concatenated head outputs before projection
        head_concat: B::Storage,
        n_heads: usize,
        d_model: usize,
        d_head: usize,
        batch_size: usize,
        seq_len: usize,
        qkv_weight_id: TensorId,
        qkv_bias_id: TensorId,
        proj_weight_id: TensorId,
        proj_bias_id: TensorId,
    },
}

/// Which operation produced a tensor.
pub enum Operation {
    Matmul,
    Add,
    Softmax,
    LayerNorm,
    Gelu,
    CrossEntropy,
    Embedding,
    Sum,
    Attention,
    Dropout,
    Checkpoint,
    AttentionBatched,
}

/// A node on the autograd tape.
pub struct TapeNode<B: DeviceBackend> {
    pub output_id: TensorId,
    pub input_ids: Vec<TensorId>,
    pub op: Operation,
    pub saved: SavedData<B>,
}

/// Arena-based gradient tape. Records operations for backward pass.
pub struct GradTape<B: DeviceBackend> {
    pub nodes: Vec<TapeNode<B>>,
}

impl<B: DeviceBackend> GradTape<B> {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn record(&mut self, node: TapeNode<B>) {
        self.nodes.push(node);
    }
}

impl<B: DeviceBackend> Default for GradTape<B> {
    fn default() -> Self {
        Self::new()
    }
}
