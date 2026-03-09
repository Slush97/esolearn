use std::path::{Path, PathBuf};

use crate::error::ScryLlmError;

/// A batch of token sequences for language model training.
pub struct Batch {
    /// Input token IDs, flat `[batch_size * seq_len]`.
    pub input_ids: Vec<usize>,
    /// Target token IDs (shifted by 1), flat `[batch_size * seq_len]`.
    pub targets: Vec<usize>,
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Loads pre-tokenized binary shards (packed `u16` little-endian) for LM training.
///
/// Shard format: contiguous `u16` LE token IDs, documents separated by `<|endoftext|>` (50256).
/// Compatible with llm.c / nanoGPT shard format.
pub struct DataLoader {
    shard_paths: Vec<PathBuf>,
    tokens: Vec<u16>,
    seq_len: usize,
    batch_size: usize,
    current_shard: usize,
    current_pos: usize,
    rng: fastrand::Rng,
    /// Shuffled starting positions within the current shard.
    positions: Vec<usize>,
    /// Current index into `positions`.
    pos_idx: usize,
}

impl DataLoader {
    /// Create a new data loader from shard files.
    ///
    /// # Errors
    ///
    /// Returns an error if no shard files are found or the first shard can't be loaded.
    pub fn new(
        shard_dir: &Path,
        pattern: &str,
        seq_len: usize,
        batch_size: usize,
        seed: u64,
    ) -> crate::error::Result<Self> {
        let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(shard_dir)
            .map_err(|e| ScryLlmError::DataError(format!("cannot read shard dir: {e}")))?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| {
                        n.contains(pattern)
                            && Path::new(n)
                                .extension()
                                .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
                    })
            })
            .collect();

        if shard_paths.is_empty() {
            return Err(ScryLlmError::DataError(format!(
                "no shards matching '{pattern}' in {}",
                shard_dir.display()
            )));
        }

        shard_paths.sort();

        let tokens = load_shard(&shard_paths[0])?;
        let mut rng = fastrand::Rng::with_seed(seed);
        let positions = Self::compute_positions(tokens.len(), seq_len, batch_size, &mut rng);

        Ok(Self {
            shard_paths,
            tokens,
            seq_len,
            batch_size,
            current_shard: 0,
            current_pos: 0,
            rng,
            positions,
            pos_idx: 0,
        })
    }

    /// Create a data loader from an already-loaded token buffer (for testing).
    pub fn from_tokens(tokens: Vec<u16>, seq_len: usize, batch_size: usize, seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let positions = Self::compute_positions(tokens.len(), seq_len, batch_size, &mut rng);
        Self {
            shard_paths: Vec::new(),
            tokens,
            seq_len,
            batch_size,
            current_shard: 0,
            current_pos: 0,
            rng,
            positions,
            pos_idx: 0,
        }
    }

    /// Compute and shuffle valid starting positions for batches within a shard.
    fn compute_positions(
        n_tokens: usize,
        seq_len: usize,
        batch_size: usize,
        rng: &mut fastrand::Rng,
    ) -> Vec<usize> {
        let tokens_per_item = seq_len + 1;
        let tokens_per_batch = batch_size * tokens_per_item;
        if tokens_per_batch == 0 || n_tokens < tokens_per_batch {
            return Vec::new();
        }
        let n_batches = (n_tokens - tokens_per_item) / tokens_per_item;
        if n_batches == 0 {
            return Vec::new();
        }
        // Generate starting positions for individual sequence slots
        let mut positions: Vec<usize> = (0..n_batches)
            .map(|i| i * tokens_per_item)
            .filter(|&pos| pos + tokens_per_item <= n_tokens)
            .collect();
        rng.shuffle(&mut positions);
        positions
    }

    /// Get the next batch, using shuffled positions within the shard.
    ///
    /// # Errors
    ///
    /// Returns an error if shard advancement fails (e.g., missing shard file).
    pub fn next_batch(&mut self) -> crate::error::Result<Batch> {
        // If not enough shuffled positions remain, advance shard
        if self.pos_idx + self.batch_size > self.positions.len() {
            self.advance_shard()?;
        }

        let mut input_ids = Vec::with_capacity(self.batch_size * self.seq_len);
        let mut targets = Vec::with_capacity(self.batch_size * self.seq_len);

        for _ in 0..self.batch_size {
            let start = self.positions[self.pos_idx];
            self.pos_idx += 1;
            let end = start + self.seq_len + 1;
            let chunk = &self.tokens[start..end];

            for i in 0..self.seq_len {
                input_ids.push(chunk[i] as usize);
                targets.push(chunk[i + 1] as usize);
            }
        }

        Ok(Batch {
            input_ids,
            targets,
            batch_size: self.batch_size,
            seq_len: self.seq_len,
        })
    }

    /// Reset to beginning of first shard.
    pub fn reset(&mut self) {
        self.current_shard = 0;
        self.current_pos = 0;
        if let Some(path) = self.shard_paths.first() {
            if let Ok(tokens) = load_shard(path) {
                self.tokens = tokens;
            }
        }
        self.positions = Self::compute_positions(
            self.tokens.len(),
            self.seq_len,
            self.batch_size,
            &mut self.rng,
        );
        self.pos_idx = 0;
    }

    /// Shuffle shard order for epoch boundary.
    pub fn shuffle_shards(&mut self) {
        self.rng.shuffle(&mut self.shard_paths);
    }

    fn advance_shard(&mut self) -> crate::error::Result<()> {
        if self.shard_paths.is_empty() {
            // In-memory mode: reshuffle positions and wrap around
            self.positions = Self::compute_positions(
                self.tokens.len(),
                self.seq_len,
                self.batch_size,
                &mut self.rng,
            );
            self.pos_idx = 0;
            self.current_pos = 0;
            return Ok(());
        }

        self.current_shard = (self.current_shard + 1) % self.shard_paths.len();
        if self.current_shard == 0 {
            self.shuffle_shards();
        }
        self.tokens = load_shard(&self.shard_paths[self.current_shard])?;
        self.current_pos = 0;
        self.positions = Self::compute_positions(
            self.tokens.len(),
            self.seq_len,
            self.batch_size,
            &mut self.rng,
        );
        self.pos_idx = 0;
        Ok(())
    }
}

fn load_shard(path: &Path) -> crate::error::Result<Vec<u16>> {
    let data = std::fs::read(path)
        .map_err(|e| ScryLlmError::DataError(format!("failed to read shard {}: {e}", path.display())))?;

    if data.len() % 2 != 0 {
        return Err(ScryLlmError::DataError(format!(
            "shard {} has odd byte count ({})",
            path.display(),
            data.len()
        )));
    }

    let tokens: Vec<u16> = data
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    Ok(tokens)
}
