use crate::backend::MathBackend;
use crate::tensor::shape::Shape;

/// Per-layer KV cache for autoregressive inference (GPT-2 style).
///
/// Stores accumulated K and V projections for each attention head.
pub struct LayerKvCache<B: MathBackend> {
    /// `k_per_head[h]`: `[cached_seq, d_head]`
    pub k_per_head: Vec<B::Storage>,
    /// `v_per_head[h]`: `[cached_seq, d_head]`
    pub v_per_head: Vec<B::Storage>,
    /// Number of cached tokens so far.
    pub seq_len: usize,
    pub n_heads: usize,
    pub d_head: usize,
}

impl<B: MathBackend> LayerKvCache<B> {
    pub fn new(n_heads: usize, d_head: usize) -> Self {
        Self {
            k_per_head: Vec::new(),
            v_per_head: Vec::new(),
            seq_len: 0,
            n_heads,
            d_head,
        }
    }

    /// Append new K and V for a single token to each head.
    /// `new_k[h]`: `[1, d_head]`, `new_v[h]`: `[1, d_head]`.
    pub fn append(&mut self, new_k: Vec<B::Storage>, new_v: Vec<B::Storage>) {
        if self.seq_len == 0 {
            self.k_per_head = new_k;
            self.v_per_head = new_v;
        } else {
            for h in 0..self.n_heads {
                self.k_per_head[h] =
                    B::concat_rows(&self.k_per_head[h], &new_k[h], self.seq_len, 1, self.d_head);
                self.v_per_head[h] =
                    B::concat_rows(&self.v_per_head[h], &new_v[h], self.seq_len, 1, self.d_head);
            }
        }
        self.seq_len += 1;
    }
}

/// Full model KV cache: one [`LayerKvCache`] per transformer block.
pub struct KvCache<B: MathBackend> {
    pub layers: Vec<LayerKvCache<B>>,
}

impl<B: MathBackend> KvCache<B> {
    pub fn new(n_layers: usize, n_heads: usize, d_head: usize) -> Self {
        let layers = (0..n_layers)
            .map(|_| LayerKvCache::new(n_heads, d_head))
            .collect();
        Self { layers }
    }
}

// ============================================================
// Llama-optimized KV cache: pre-allocated contiguous storage
// ============================================================

/// Per-layer KV cache with pre-allocated contiguous storage for Llama.
///
/// Instead of per-head `Vec<B::Storage>` with repeated concat_rows,
/// uses two contiguous buffers `[max_seq, n_kv_heads * head_dim]` and
/// writes/reads with single `scatter_rows`/`gather_rows` calls.
pub struct LlamaLayerKvCache<B: MathBackend> {
    /// `[max_seq, n_kv_heads * head_dim]` — pre-allocated K buffer.
    pub k_cache: B::Storage,
    /// `[max_seq, n_kv_heads * head_dim]` — pre-allocated V buffer.
    pub v_cache: B::Storage,
    /// Number of cached tokens so far.
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: MathBackend> LlamaLayerKvCache<B> {
    pub fn new(max_seq_len: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        Self {
            k_cache: B::zeros(&Shape::new(&[max_seq_len, kv_dim])),
            v_cache: B::zeros(&Shape::new(&[max_seq_len, kv_dim])),
            seq_len: 0,
            max_seq_len,
            n_kv_heads,
            head_dim,
        }
    }

    /// Append K and V for a single token.
    ///
    /// `k_row`: `[1, n_kv_heads * head_dim]`, `v_row`: `[1, n_kv_heads * head_dim]`.
    /// Writes into row `seq_len` of the pre-allocated buffers.
    pub fn append(&mut self, k_row: &B::Storage, v_row: &B::Storage) {
        let kv_dim = self.n_kv_heads * self.head_dim;
        B::scatter_rows(
            &mut self.k_cache, k_row,
            self.max_seq_len, kv_dim, self.seq_len, 1,
        );
        B::scatter_rows(
            &mut self.v_cache, v_row,
            self.max_seq_len, kv_dim, self.seq_len, 1,
        );
        self.seq_len += 1;
    }

    /// Read cached K values: `[seq_len, n_kv_heads * head_dim]`.
    pub fn k(&self) -> B::Storage {
        let kv_dim = self.n_kv_heads * self.head_dim;
        B::gather_rows(&self.k_cache, self.max_seq_len, kv_dim, 0, self.seq_len)
    }

    /// Read cached V values: `[seq_len, n_kv_heads * head_dim]`.
    pub fn v(&self) -> B::Storage {
        let kv_dim = self.n_kv_heads * self.head_dim;
        B::gather_rows(&self.v_cache, self.max_seq_len, kv_dim, 0, self.seq_len)
    }
}

/// Full Llama KV cache: one [`LlamaLayerKvCache`] per transformer block.
pub struct LlamaKvCache<B: MathBackend> {
    pub layers: Vec<LlamaLayerKvCache<B>>,
}

impl<B: MathBackend> LlamaKvCache<B> {
    pub fn new(
        n_layers: usize,
        max_seq_len: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let layers = (0..n_layers)
            .map(|_| LlamaLayerKvCache::new(max_seq_len, n_kv_heads, head_dim))
            .collect();
        Self { layers }
    }
}
