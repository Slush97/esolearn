//! Whisper tokenizer — decode-only tokenizer for converting token IDs to text.
//!
//! Supports two vocab formats:
//!   - `vocab.bin`: binary format with correct Whisper token IDs (preferred)
//!   - `tokenizer.json`: HuggingFace format (uses GPT-2 token ordering)

use std::collections::HashMap;
use std::path::Path;

use crate::error::SttError;

/// Whisper tokenizer for converting token IDs to text.
///
/// This is a decode-only tokenizer — encoding (text to tokens) is not
/// needed for inference since Whisper generates tokens autoregressively.
pub struct WhisperTokenizer {
    /// Reverse vocab: token ID to raw byte sequence.
    id_to_bytes: Vec<Vec<u8>>,
    /// Total vocabulary size (including special tokens).
    vocab_size: usize,
}

/// Special token IDs for Whisper.
pub const EOT_TOKEN: usize = 50257;
pub const SOT_TOKEN: usize = 50258;
pub const SOT_PREV_TOKEN: usize = 50360;
pub const NO_SPEECH_TOKEN: usize = 50362;
pub const NO_TIMESTAMPS_TOKEN: usize = 50363;
/// First timestamp token (represents 0.00 seconds).
pub const TIMESTAMP_BEGIN: usize = 50364;

impl WhisperTokenizer {
    /// Load a tokenizer from a file.
    ///
    /// Supports both `vocab.bin` (binary, correct Whisper IDs) and
    /// `tokenizer.json` (HuggingFace format). The format is detected
    /// by file extension.
    ///
    /// **Prefer `vocab.bin`** — the HuggingFace `tokenizer.json` uses GPT-2
    /// token ordering which differs from Whisper's actual token IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: &Path) -> crate::error::Result<Self> {
        // Try vocab.bin sibling first (same directory as the given path)
        if let Some(parent) = path.parent() {
            let vocab_bin = parent.join("vocab.bin");
            if vocab_bin.exists() {
                return Self::from_vocab_bin(&vocab_bin);
            }
        }

        // Fall back to tokenizer.json parsing
        if path.extension().and_then(|e| e.to_str()) == Some("bin") {
            Self::from_vocab_bin(path)
        } else {
            Self::from_tokenizer_json(path)
        }
    }

    /// Load from `vocab.bin` — binary format with correct Whisper token IDs.
    ///
    /// Format: sequence of `(id: u32-LE, len: u16-LE, bytes: [u8; len])`.
    fn from_vocab_bin(path: &Path) -> crate::error::Result<Self> {
        let data = std::fs::read(path).map_err(SttError::Io)?;
        let mut pos = 0;
        let mut max_id = 0usize;
        let mut entries = Vec::new();

        while pos + 6 <= data.len() {
            let id = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            let len = u16::from_le_bytes(data[pos + 4..pos + 6].try_into().unwrap()) as usize;
            pos += 6;
            if pos + len > data.len() {
                return Err(SttError::Tokenizer(format!(
                    "vocab.bin truncated at id={id}, need {len} bytes at offset {pos}"
                )));
            }
            let bytes = data[pos..pos + len].to_vec();
            pos += len;
            if id > max_id {
                max_id = id;
            }
            entries.push((id, bytes));
        }

        let vocab_size = max_id + 1;
        let mut id_to_bytes = vec![Vec::new(); vocab_size];
        for (id, bytes) in entries {
            id_to_bytes[id] = bytes;
        }

        Ok(Self {
            id_to_bytes,
            vocab_size,
        })
    }

    /// Load from HuggingFace `tokenizer.json`.
    ///
    /// **Warning:** uses GPT-2 token ordering, not Whisper's. Prefer `vocab.bin`.
    fn from_tokenizer_json(path: &Path) -> crate::error::Result<Self> {
        let contents = std::fs::read_to_string(path)
            .map_err(SttError::Io)?;
        let json: serde_json::Value = serde_json::from_str(&contents)
            .map_err(|e| SttError::Tokenizer(format!("parse error: {e}")))?;

        // Extract vocab from model.vocab
        let vocab_obj = json
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| SttError::Tokenizer("missing model.vocab".into()))?;

        // Build token string to id map
        let mut token_to_id: HashMap<String, usize> = HashMap::new();
        for (token, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                token_to_id.insert(token.clone(), id as usize);
            }
        }

        // Also load added_tokens (special tokens like EOT, SOT, etc.)
        if let Some(added) = json.get("added_tokens").and_then(|a| a.as_array()) {
            for entry in added {
                if let (Some(content), Some(id)) = (
                    entry.get("content").and_then(|c| c.as_str()),
                    entry.get("id").and_then(serde_json::Value::as_u64),
                ) {
                    token_to_id.insert(content.to_string(), id as usize);
                }
            }
        }

        // Find max ID to size our lookup table
        let max_id = token_to_id.values().copied().max().unwrap_or(0);
        let vocab_size = max_id + 1;

        // Build the reverse unicode mapping for GPT-2 byte-level BPE
        let unicode_to_byte = build_unicode_to_byte_map();

        // Build id_to_bytes lookup
        let mut id_to_bytes = vec![Vec::new(); vocab_size];
        for (token_str, id) in &token_to_id {
            // For special tokens (ID >= 50257), store the raw string as UTF-8
            if *id >= EOT_TOKEN {
                id_to_bytes[*id] = token_str.as_bytes().to_vec();
            } else {
                // For regular tokens, reverse the GPT-2 bytes_to_unicode mapping
                let bytes: Vec<u8> = token_str
                    .chars()
                    .map(|c| {
                        unicode_to_byte
                            .get(&c)
                            .copied()
                            .unwrap_or(b'?')
                    })
                    .collect();
                id_to_bytes[*id] = bytes;
            }
        }

        Ok(Self {
            id_to_bytes,
            vocab_size,
        })
    }

    /// Decode a sequence of token IDs to a string.
    ///
    /// Special tokens (EOT, SOT, timestamps, etc.) are filtered out.
    pub fn decode(&self, token_ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            if self.is_special(id) {
                continue;
            }
            if id < self.vocab_size {
                bytes.extend_from_slice(&self.id_to_bytes[id]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Check if a token ID is a special token.
    pub fn is_special(&self, token_id: usize) -> bool {
        token_id >= EOT_TOKEN
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// Build the reverse mapping from GPT-2 unicode characters back to bytes.
///
/// GPT-2 uses a byte-level BPE where each byte value (0-255) is mapped to
/// a printable Unicode character. This function builds the reverse map.
fn build_unicode_to_byte_map() -> HashMap<char, u8> {
    let mut byte_to_unicode: HashMap<u8, char> = HashMap::new();

    // Printable ASCII ranges that map to themselves
    // 33..=126 ('!' to '~')
    for b in 33u16..=126 {
        byte_to_unicode.insert(b as u8, char::from(b as u8));
    }
    // 161..=172
    for b in 161u16..=172 {
        byte_to_unicode.insert(b as u8, char::from_u32(u32::from(b)).unwrap());
    }
    // 174..=255
    for b in 174u16..=255 {
        byte_to_unicode.insert(b as u8, char::from_u32(u32::from(b)).unwrap());
    }

    // Remaining bytes get mapped to Unicode chars starting at U+0100 (256)
    let mut n = 256u32;
    for b in 0u16..=255 {
        let byte = b as u8;
        byte_to_unicode.entry(byte).or_insert_with(|| {
            let c = char::from_u32(n).unwrap_or('?');
            n += 1;
            c
        });
    }

    // Reverse the map
    byte_to_unicode
        .into_iter()
        .map(|(b, c)| (c, b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_to_byte_roundtrip() {
        let map = build_unicode_to_byte_map();
        // Should have exactly 256 entries (one per byte value)
        assert_eq!(map.len(), 256);
        // All values should be unique
        let mut seen = [false; 256];
        for &b in map.values() {
            assert!(!seen[b as usize], "duplicate byte value {b}");
            seen[b as usize] = true;
        }
    }

    #[test]
    fn special_token_check() {
        // We can't create a WhisperTokenizer without a file, but we can test the constant
        assert!(EOT_TOKEN >= 50257);
        assert!(SOT_TOKEN >= 50257);
        assert!(TIMESTAMP_BEGIN >= 50257);
    }

    #[test]
    fn ascii_maps_to_self() {
        let map = build_unicode_to_byte_map();
        // Printable ASCII characters should map to their own byte value
        for b in 33u8..=126 {
            let c = char::from(b);
            assert_eq!(map[&c], b, "ASCII char '{c}' should map to byte {b}");
        }
    }
}
