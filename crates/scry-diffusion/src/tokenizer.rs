// SPDX-License-Identifier: MIT OR Apache-2.0
//! CLIP BPE tokenizer.
//!
//! SD 1.5, SD 2.x, and SDXL all use the same OpenAI CLIP byte-pair encoder
//! with a 49,408-token vocabulary, BOS = 49406, EOS = 49407, and a fixed
//! sequence length of 77 (truncated or padded with EOS). The same tokenizer
//! files (`vocab.json` + `merges.txt`) ship inside every HF SD checkpoint's
//! `tokenizer/` subdirectory and are byte-identical across them.
//!
//! Three things distinguish CLIP's BPE from the GPT-2 byte-level BPE that
//! `scry_llm::tokenizer::BpeTokenizer` implements:
//!
//! 1. The input is **lowercased and whitespace-collapsed** before
//!    pre-tokenization (HF's `whitespace_clean` + `.lower()`).
//! 2. The pre-tokenization regex is the CLIP-specific
//!    `<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`
//!    — letters greedy, digits singleton, punctuation greedy.
//! 3. Each pre-token's last char is suffixed with `</w>` before the BPE
//!    merge loop runs. This makes CLIP's BPE behave like word-level BPE
//!    even though the vocab keys are byte-level on the inside.
//!
//! For ASCII English prompts (the only thing SD really supports anyway),
//! this matches HF's `CLIPTokenizer.encode(prompt, padding="max_length",
//! truncation=True, max_length=77)` exactly. Non-ASCII input that HF would
//! ftfy-clean (mojibake repair) is passed through as-is — diffusion prompts
//! that exercise that path are vanishingly rare; revisit if M8 finds drift.
//!
//! Reference: `https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py`.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Error, Result};

/// CLIP BPE tokenizer.
pub struct Tokenizer {
    /// `token_string -> token_id`. Includes the `</w>` end-of-word variants.
    encoder: HashMap<String, u32>,
    /// Reverse of `encoder` — kept for `decode` and debugging.
    decoder: HashMap<u32, String>,
    /// `(left, right) -> merge_rank`. Lower rank means earlier merge.
    bpe_ranks: HashMap<(String, String), u32>,
    /// GPT-2's bijection from each of the 256 byte values to a printable
    /// Unicode code point, so BPE can operate on a `String` without losing
    /// byte-level granularity.
    byte_encoder: [char; 256],
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

    /// Load a tokenizer from a directory containing `vocab.json` and
    /// `merges.txt`. The HF SD 1.5 snapshot lays these out under
    /// `tokenizer/` inside the checkpoint root.
    ///
    /// # Errors
    /// Returns `Error::Llm` if either file is missing or unparseable.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");
        let vocab_json = std::fs::read_to_string(&vocab_path)
            .map_err(|e| Error::Llm(format!("read {}: {e}", vocab_path.display())))?;
        let merges_txt = std::fs::read_to_string(&merges_path)
            .map_err(|e| Error::Llm(format!("read {}: {e}", merges_path.display())))?;
        Self::from_vocab_and_merges(&vocab_json, &merges_txt)
    }

    /// Build a tokenizer directly from in-memory `vocab.json` and
    /// `merges.txt` contents. Useful for testing with embedded fixtures.
    ///
    /// # Errors
    /// Returns `Error::Llm` on JSON parse failure or malformed merges.
    pub fn from_vocab_and_merges(vocab_json: &str, merges_txt: &str) -> Result<Self> {
        let encoder: HashMap<String, u32> = serde_json::from_str(vocab_json)
            .map_err(|e| Error::Llm(format!("parse vocab.json: {e}")))?;
        let decoder: HashMap<u32, String> = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        let mut bpe_ranks = HashMap::new();
        let mut rank: u32 = 0;
        for line in merges_txt.lines() {
            // First line of CLIP's merges.txt is `#version: 0.2`. Skip any
            // comment line and any blank line; only count real merges in
            // the rank counter so the indexing is stable.
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let mut parts = trimmed.splitn(2, ' ');
            let first = parts
                .next()
                .ok_or_else(|| Error::Llm(format!("merges.txt rank {rank}: missing left half")))?;
            let second = parts
                .next()
                .ok_or_else(|| Error::Llm(format!("merges.txt rank {rank}: missing right half")))?;
            bpe_ranks.insert((first.to_string(), second.to_string()), rank);
            rank += 1;
        }

        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder: bytes_to_unicode(),
        })
    }

    /// Vocabulary size including the two special tokens. CLIP is 49_408.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Encode a prompt into exactly [`Self::MAX_SEQ_LEN`] tokens, prefixed
    /// with `BOS` and padded with `EOS`. Truncates from the right if the
    /// prompt's BPE expansion exceeds `MAX_SEQ_LEN - 2` tokens.
    ///
    /// # Errors
    /// Returns `Error::Llm` if a BPE piece is missing from the encoder
    /// (indicates a vocab/merges mismatch in the loaded files).
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        let cleaned = whitespace_clean(prompt).to_lowercase();
        let mut ids = Vec::with_capacity(Self::MAX_SEQ_LEN);
        ids.push(Self::BOS_TOKEN);
        let cap = Self::MAX_SEQ_LEN - 1;

        'outer: for chunk in pre_tokenize(&cleaned) {
            // GPT-2 byte-level encode: every UTF-8 byte becomes one Unicode
            // char from the byte_encoder map. BPE then operates on chars.
            let mapped: String = chunk
                .bytes()
                .map(|b| self.byte_encoder[b as usize])
                .collect();
            for piece in self.bpe(&mapped) {
                let id = self.encoder.get(&piece).copied().ok_or_else(|| {
                    Error::Llm(format!(
                        "tokenizer: BPE piece '{piece}' missing from vocab — \
                         vocab.json / merges.txt mismatch?"
                    ))
                })?;
                ids.push(id);
                if ids.len() >= cap {
                    break 'outer;
                }
            }
        }

        ids.push(Self::EOS_TOKEN);
        ids.resize(Self::MAX_SEQ_LEN, Self::EOS_TOKEN);
        Ok(ids)
    }

    /// Decode a token ID stream back to text. Drops `BOS` / `EOS` and any
    /// IDs not present in the vocab. For debugging only — pipelines never
    /// need this.
    pub fn decode(&self, ids: &[u32]) -> String {
        // Reverse map: byte_encoder unicode char -> raw byte. CLIP's
        // pieces include `</w>`, which we render as a trailing space.
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, &c) in self.byte_encoder.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            byte_decoder.insert(c, b as u8);
        }
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if id == Self::BOS_TOKEN || id == Self::EOS_TOKEN {
                continue;
            }
            let Some(piece) = self.decoder.get(&id) else {
                continue;
            };
            // Replace the `</w>` end-of-word marker with a space.
            let piece = piece.replace("</w>", " ");
            for c in piece.chars() {
                if let Some(&b) = byte_decoder.get(&c) {
                    bytes.push(b);
                } else {
                    // Shouldn't happen for well-formed pieces, but don't panic.
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// CLIP's word-level BPE: append `</w>` to the last char, then iterate
    /// merging the lowest-rank adjacent pair until none remain.
    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        if chars.len() == 1 {
            return vec![format!("{}</w>", chars[0])];
        }
        let mut word: Vec<String> = chars[..chars.len() - 1]
            .iter()
            .map(char::to_string)
            .collect();
        word.push(format!("{}</w>", chars[chars.len() - 1]));

        loop {
            // Find the adjacent pair with the lowest merge rank.
            let mut best_pair: Option<(String, String)> = None;
            let mut best_rank = u32::MAX;
            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(pair);
                    }
                }
            }
            let Some((first, second)) = best_pair else {
                break;
            };
            let merged = format!("{first}{second}");
            let mut new_word: Vec<String> = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    new_word.push(merged.clone());
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;
            if word.len() == 1 {
                break;
            }
        }
        word
    }
}

/// Build GPT-2's `bytes_to_unicode()` mapping: a bijection from each of the
/// 256 byte values to a Unicode code point that's printable and not
/// whitespace, so BPE can operate on `String` without losing byte fidelity.
fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u8> = Vec::with_capacity(256);
    bs.extend(b'!'..=b'~');
    bs.extend(0xa1u8..=0xac);
    bs.extend(0xaeu8..=0xff);
    let mut cs: Vec<u32> = bs.iter().map(|&b| u32::from(b)).collect();
    let mut n: u32 = 0;
    for b in 0u16..=255 {
        #[allow(clippy::cast_possible_truncation)]
        let b = b as u8;
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        out[b as usize] = char::from_u32(c).expect("bytes_to_unicode: invalid codepoint");
    }
    out
}

/// HF `whitespace_clean`: collapse any run of Unicode whitespace into a
/// single ASCII space, then strip leading/trailing whitespace.
fn whitespace_clean(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                out.push(' ');
                prev_ws = true;
            }
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

/// Hand-rolled CLIP pre-tokenization, equivalent to:
///
/// `<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`
///
/// applied with `re.IGNORECASE`. Input is assumed to be lowercased + whitespace-cleaned.
fn pre_tokenize(text: &str) -> Vec<String> {
    const SOT: &str = "<|startoftext|>";
    const EOT: &str = "<|endoftext|>";

    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        // Whitespace runs match nothing in the CLIP pattern — they act as
        // separators between pre-tokens and produce no output of their own.
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        // Special tokens first.
        if chars[i] == '<' {
            if substr_eq(&chars, i, SOT) {
                out.push(SOT.to_string());
                i += SOT.chars().count();
                continue;
            }
            if substr_eq(&chars, i, EOT) {
                out.push(EOT.to_string());
                i += EOT.chars().count();
                continue;
            }
        }

        // Contractions: `'s|'t|'re|'ve|'m|'ll|'d`. Input is lowercased so
        // we don't need a case-insensitive match.
        if chars[i] == '\'' && i + 1 < n {
            let next = chars[i + 1];
            let extra = match next {
                's' | 't' | 'm' | 'd' => Some(2),
                'r' | 'v' if i + 2 < n && chars[i + 2] == 'e' => Some(3),
                'l' if i + 2 < n && chars[i + 2] == 'l' => Some(3),
                _ => None,
            };
            if let Some(len) = extra {
                out.push(chars[i..i + len].iter().collect());
                i += len;
                continue;
            }
        }

        // `[\p{L}]+` — run of letters.
        if chars[i].is_alphabetic() {
            let start = i;
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        // `[\p{N}]` — single-character digit run. Note the original CLIP
        // regex has no `+`, so each digit becomes its own pre-token.
        if chars[i].is_numeric() {
            out.push(chars[i].to_string());
            i += 1;
            continue;
        }

        // `[^\s\p{L}\p{N}]+` — punctuation / symbol run.
        let start = i;
        while i < n
            && !chars[i].is_whitespace()
            && !chars[i].is_alphabetic()
            && !chars[i].is_numeric()
        {
            i += 1;
        }
        // Defensive: if we somehow didn't advance, force progress so the
        // loop can't spin. Shouldn't happen — every char hits one of the
        // arms above — but cheap insurance against a future regression.
        if i == start {
            i += 1;
        }
        out.push(chars[start..i].iter().collect());
    }
    out
}

/// Compare `chars[start..]` with the chars of `needle`, returning true on
/// full match. Cheaper than building a `String` slice on every iteration.
fn substr_eq(chars: &[char], start: usize, needle: &str) -> bool {
    let mut it = needle.chars();
    let mut k = start;
    loop {
        match (it.next(), chars.get(k)) {
            (None, _) => return true,
            (Some(_), None) => return false,
            (Some(a), Some(&b)) => {
                if a != b {
                    return false;
                }
            }
        }
        k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny hand-built vocab covering enough merges to exercise the BPE
    /// loop without needing the real 49,408-entry CLIP vocab on disk.
    ///
    /// Word: "low" + EOW. Char-level pieces are `l`, `o`, `w</w>`.
    /// Merges (rank order): `l o`, `lo w</w>`. So `low` should merge
    /// to a single `low</w>` piece.
    fn fixture() -> (String, String) {
        let vocab = serde_json::json!({
            "<|startoftext|>": 0,
            "<|endoftext|>":   1,
            "l":   2,
            "o":   3,
            "w":   4,
            "l</w>": 5,
            "o</w>": 6,
            "w</w>": 7,
            "lo":  8,
            "low</w>": 9,
            "lo</w>": 10
        })
        .to_string();
        let merges = "#version: 0.2\nl o\nlo w</w>\n";
        (vocab, merges.to_string())
    }

    #[test]
    fn bytes_to_unicode_is_a_bijection() {
        let map = bytes_to_unicode();
        // All 256 entries non-null, all distinct.
        let mut seen = std::collections::HashSet::new();
        for (b, &c) in map.iter().enumerate() {
            assert_ne!(c, '\0', "byte {b}: unset");
            assert!(seen.insert(c), "byte {b}: duplicate char {c:?}");
        }
        assert_eq!(seen.len(), 256);
        // Spot-check well-known anchors. Printable ASCII passes through
        // as-is; bytes outside the printable ranges get pushed into the
        // 256+ block in order, so byte 0x00 -> 256, 0x01 -> 257, etc.
        assert_eq!(map[b'!' as usize], '!');
        assert_eq!(map[b'~' as usize], '~');
        assert_eq!(map[0x00], char::from_u32(256).unwrap());
        // Space (0x20) is the 33rd byte not in the printable bs ranges
        // (bytes 0x00..=0x20), so it lands at 256 + 32 = 288.
        assert_eq!(map[b' ' as usize] as u32, 288);
    }

    #[test]
    fn whitespace_clean_collapses_and_strips() {
        assert_eq!(whitespace_clean("  hello   world  "), "hello world");
        assert_eq!(whitespace_clean("a\tb\nc"), "a b c");
        assert_eq!(whitespace_clean(""), "");
    }

    #[test]
    fn pre_tokenize_letters_and_digits_and_punct() {
        // Letters greedy, digits singleton, punct greedy.
        assert_eq!(pre_tokenize("hello"), vec!["hello"]);
        assert_eq!(pre_tokenize("123"), vec!["1", "2", "3"]);
        assert_eq!(pre_tokenize("!?!"), vec!["!?!"]);
        assert_eq!(pre_tokenize("hello, world"), vec!["hello", ",", "world"]);
    }

    #[test]
    fn pre_tokenize_contractions() {
        // Apostrophe contractions get split off as their own pre-token.
        assert_eq!(pre_tokenize("don't"), vec!["don", "'t"]);
        assert_eq!(pre_tokenize("we'll"), vec!["we", "'ll"]);
        assert_eq!(pre_tokenize("they're"), vec!["they", "'re"]);
    }

    #[test]
    fn pre_tokenize_special_tokens() {
        assert_eq!(
            pre_tokenize("<|startoftext|>x<|endoftext|>"),
            vec!["<|startoftext|>", "x", "<|endoftext|>"]
        );
    }

    #[test]
    fn bpe_merges_low_to_single_piece() {
        let (vocab, merges) = fixture();
        let tok = Tokenizer::from_vocab_and_merges(&vocab, &merges).unwrap();
        // "l", "o", "w</w>" -> merge "l o" -> "lo", "w</w>" -> merge
        // "lo w</w>" -> "low</w>".
        assert_eq!(tok.bpe("low"), vec!["low</w>"]);
    }

    #[test]
    fn bpe_singleton_appends_eow() {
        let (vocab, merges) = fixture();
        let tok = Tokenizer::from_vocab_and_merges(&vocab, &merges).unwrap();
        assert_eq!(tok.bpe("o"), vec!["o</w>"]);
    }

    #[test]
    fn encode_frames_with_bos_eos_and_pads_to_max_len() {
        let (vocab, merges) = fixture();
        let tok = Tokenizer::from_vocab_and_merges(&vocab, &merges).unwrap();
        let ids = tok.encode("low").unwrap();
        assert_eq!(ids.len(), Tokenizer::MAX_SEQ_LEN);
        assert_eq!(ids[0], Tokenizer::BOS_TOKEN);
        // Single piece "low</w>" -> id 9 in the fixture.
        assert_eq!(ids[1], 9);
        assert_eq!(ids[2], Tokenizer::EOS_TOKEN);
        // Trailing positions are EOS pad.
        assert!(ids.iter().skip(2).all(|&id| id == Tokenizer::EOS_TOKEN));
    }

    #[test]
    fn encode_truncates_overflow_keeping_bos_and_eos() {
        // Build a fixture where every char is its own piece so we can
        // overflow MAX_SEQ_LEN cleanly. 80 'o' chars -> 80 BPE pieces.
        let mut vocab = serde_json::Map::new();
        vocab.insert("<|startoftext|>".into(), 0.into());
        vocab.insert("<|endoftext|>".into(), 1.into());
        vocab.insert("o".into(), 2.into());
        vocab.insert("o</w>".into(), 3.into());
        let vocab_json = serde_json::Value::Object(vocab).to_string();
        let merges = "#version: 0.2\n";

        let tok = Tokenizer::from_vocab_and_merges(&vocab_json, merges).unwrap();
        let ids = tok.encode(&"o".repeat(80)).unwrap();
        assert_eq!(ids.len(), Tokenizer::MAX_SEQ_LEN);
        assert_eq!(ids[0], Tokenizer::BOS_TOKEN);
        // Last position must be EOS even when the prompt overflows.
        assert_eq!(ids[Tokenizer::MAX_SEQ_LEN - 1], Tokenizer::EOS_TOKEN);
        // The 75 inner positions are the truncated pre-tokens. Pre-tokenize
        // splits "oooo...o" into one letter run, which the BPE turns into
        // ["o", "o", ..., "o</w>"] — only the last char carries </w>.
        for &id in &ids[1..Tokenizer::MAX_SEQ_LEN - 1] {
            assert!(id == 2 || id == 3, "unexpected id {id}");
        }
    }

    #[test]
    fn whitespace_clean_runs_before_tokenize() {
        // Encoding "  LOW  " should match encoding "low" — `whitespace_clean`
        // strips edges, then `to_lowercase` normalizes case.
        let (vocab, merges) = fixture();
        let tok = Tokenizer::from_vocab_and_merges(&vocab, &merges).unwrap();
        assert_eq!(tok.encode("  LOW  ").unwrap(), tok.encode("low").unwrap());
    }

    #[test]
    fn malformed_merges_line_returns_error() {
        let (vocab, _) = fixture();
        // Missing right half on the merges row.
        let bad_merges = "#version: 0.2\nlo\n";
        assert!(Tokenizer::from_vocab_and_merges(&vocab, bad_merges).is_err());
    }
}
