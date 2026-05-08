// SPDX-License-Identifier: MIT OR Apache-2.0
//! CLIP BPE tokenizer.
//!
//! SD 1.5, SD 2.x, and SDXL all use the same OpenAI CLIP byte-pair encoder
//! with a 49,408-token vocabulary, BOS = 49406, EOS = 49407, and a fixed
//! sequence length of 77 (truncated or padded with EOS). The same tokenizer
//! file can be loaded from any HF SD checkpoint — they are byte-identical.
//!
//! Two reasonable implementation paths (M2 in HANDOFF):
//!   1. **Pull `tokenizers`** (the HF Rust crate). Drop-in for
//!      `tokenizer.json`, but adds a non-trivial dep tree (`onig`).
//!   2. **DIY BPE** — load `vocab.json` + `merges.txt`, byte-level
//!      pre-tokenize, then BPE merge. ~300 LOC; matches OpenAI's CLIP impl
//!      exactly. Recommended for zero new C deps.
//!
//! Either way the public surface stays as below.

use crate::error::Result;

/// CLIP BPE tokenizer. Wraps either the HF `tokenizers` crate or a DIY
/// byte-level BPE — see module doc.
pub struct Tokenizer {
    // M2: backing impl (vocab map, merges, special tokens). Left empty so
    // the scaffold compiles; pick one of the two paths above.
    _placeholder: (),
}

impl Tokenizer {
    /// Maximum token sequence length used by SD's text encoders. Always 77
    /// for the CLIP family; longer prompts get chunked into windows by the
    /// pipeline (HF diffusers' `chunk_text_with_clip` pattern).
    pub const MAX_SEQ_LEN: usize = 77;

    /// Beginning-of-sequence token (`<|startoftext|>`).
    pub const BOS_TOKEN: u32 = 49_406;

    /// End-of-sequence token (`<|endoftext|>`). Also used as the pad token,
    /// matching OpenAI CLIP behavior.
    pub const EOS_TOKEN: u32 = 49_407;

    /// Load a tokenizer from a directory containing `vocab.json` +
    /// `merges.txt` (or a `tokenizer.json` if going the HF route).
    pub fn from_dir(dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let _ = dir;
        todo!("M2: load CLIP BPE vocab + merges")
    }

    /// Encode a prompt into exactly [`Self::MAX_SEQ_LEN`] tokens, prefixed
    /// with `BOS` and padded with `EOS`. Truncates if the prompt overflows.
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let _ = prompt;
        todo!("M2: byte-level pre-tokenize + BPE merges + BOS/EOS framing")
    }
}
