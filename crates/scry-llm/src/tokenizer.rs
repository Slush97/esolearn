use std::collections::HashMap;
use std::path::Path;

/// GPT-2 byte-level BPE tokenizer.
///
/// Implements the same algorithm as the original OpenAI GPT-2 tokenizer:
/// byte-to-unicode mapping, pre-tokenization regex, then iterative BPE merges.
pub struct BpeTokenizer {
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
}

/// HuggingFace `tokenizer.json` tokenizer.
///
/// Loads from a single `tokenizer.json` file (as used by Llama 3, Mistral, etc.).
/// Supports special token handling for chat templates.
pub struct HfTokenizer {
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    /// Special tokens: content → id (e.g. "<|begin_of_text|>" → 128000).
    special_encoder: HashMap<String, usize>,
    /// Reverse special tokens: id → content.
    special_decoder: HashMap<usize, String>,
}

/// Build the GPT-2 `bytes_to_unicode()` mapping.
///
/// This creates a bijection from all 256 byte values to Unicode characters,
/// avoiding control characters and whitespace that cause issues in BPE.
fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    // printable ASCII ranges
    bs.extend(b'!'..=b'~');
    bs.extend(0xa1u8..=0xac);
    bs.extend(0xaeu8..=0xff);

    let mut cs: Vec<u32> = bs.iter().map(|&b| u32::from(b)).collect();
    let mut n: u32 = 0;
    for b in 0u16..=255 {
        let b = b as u8;
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).expect("invalid unicode codepoint in bytes_to_unicode");
        encoder[b as usize] = ch;
        decoder.insert(ch, b);
    }

    (encoder, decoder)
}

/// GPT-2 pre-tokenization: split text into chunks that BPE processes independently.
///
/// Matches the pattern: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
///
/// Hand-rolled to avoid a `regex` dependency.
fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < n {
        // Try contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if chars[i] == '\'' && i + 1 < n {
            let next = chars[i + 1].to_ascii_lowercase();
            match next {
                's' | 't' | 'm' | 'd' => {
                    tokens.push(chars[i..i + 2].iter().collect());
                    i += 2;
                    continue;
                }
                'r' if i + 2 < n && chars[i + 2].to_ascii_lowercase() == 'e' => {
                    tokens.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
                'v' if i + 2 < n && chars[i + 2].to_ascii_lowercase() == 'e' => {
                    tokens.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
                'l' if i + 2 < n && chars[i + 2].to_ascii_lowercase() == 'l' => {
                    tokens.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
                _ => {}
            }
        }

        // ` ?\p{L}+` — optional space then letters
        if (chars[i] == ' ' && i + 1 < n && chars[i + 1].is_alphabetic())
            || chars[i].is_alphabetic()
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // ` ?\p{N}+` — optional space then digits
        if (chars[i] == ' ' && i + 1 < n && chars[i + 1].is_ascii_digit())
            || chars[i].is_ascii_digit()
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+` — optional space then non-alnum-non-whitespace
        if (chars[i] == ' '
            && i + 1 < n
            && !chars[i + 1].is_whitespace()
            && !chars[i + 1].is_alphabetic()
            && !chars[i + 1].is_ascii_digit())
            || (!chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_ascii_digit())
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_ascii_digit()
            {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // `\s+(?!\S)|\s+` — whitespace (trailing whitespace without non-ws lookahead)
        if chars[i].is_whitespace() {
            let start = i;
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            // If there's a non-whitespace char following, keep last ws char separate
            // (it'll be consumed as the optional space in the next token)
            if i < n && (i - start) > 1 {
                // Emit all but last whitespace
                tokens.push(chars[start..i - 1].iter().collect());
                i -= 1; // re-process the last whitespace
            } else {
                tokens.push(chars[start..i].iter().collect());
            }
            continue;
        }

        // Fallback: single character
        tokens.push(chars[i].to_string());
        i += 1;
    }

    tokens
}

impl BpeTokenizer {
    /// Load a GPT-2 tokenizer from `vocab.json` and `merges.txt` files.
    ///
    /// # Errors
    ///
    /// Returns an error if files cannot be read or parsed.
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> Result<Self, String> {
        let vocab_data = std::fs::read_to_string(vocab_path)
            .map_err(|e| format!("failed to read vocab file: {e}"))?;
        let merges_data = std::fs::read_to_string(merges_path)
            .map_err(|e| format!("failed to read merges file: {e}"))?;

        let encoder: HashMap<String, usize> = serde_json::from_str(&vocab_data)
            .map_err(|e| format!("failed to parse vocab.json: {e}"))?;

        let decoder: HashMap<usize, String> = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        let mut bpe_ranks = HashMap::new();
        for (rank, line) in merges_data.lines().enumerate() {
            // Skip header line (starts with #)
            if line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split(' ').collect();
            if parts.len() == 2 {
                let actual_rank = if merges_data.lines().next().map_or(false, |l| l.starts_with('#')) {
                    rank - 1
                } else {
                    rank
                };
                bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), actual_rank);
            }
        }

        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut token_ids = Vec::new();

        for chunk in pre_tokenize(text) {
            // Convert bytes to unicode representation
            let bpe_input: String = chunk.bytes().map(|b| self.byte_encoder[b as usize]).collect();

            // Run BPE on this word
            let bpe_tokens = self.bpe(&bpe_input);

            for tok in bpe_tokens {
                if let Some(&id) = self.encoder.get(&tok) {
                    token_ids.push(id);
                }
            }
        }

        token_ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let text: String = tokens
            .iter()
            .filter_map(|&id| self.decoder.get(&id))
            .cloned()
            .collect();

        // Convert from unicode representation back to raw bytes, then interpret as UTF-8
        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size (typically 50257 for GPT-2).
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// The BPE merge loop for a single word (already converted to unicode chars).
    fn bpe(&self, token: &str) -> Vec<String> {
        bpe(token, &self.bpe_ranks)
    }
}

/// Llama 3 pre-tokenization: split text into chunks using the tiktoken-style pattern.
///
/// Pattern: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
///
/// Hand-rolled to avoid a `regex` dependency.
fn pre_tokenize_llama(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < n {
        // Try contractions (case-insensitive): 's, 't, 're, 've, 'm, 'll, 'd
        if chars[i] == '\'' && i + 1 < n {
            let next = chars[i + 1].to_ascii_lowercase();
            match next {
                's' | 't' | 'm' | 'd' => {
                    tokens.push(chars[i..i + 2].iter().collect());
                    i += 2;
                    continue;
                }
                'r' if i + 2 < n && chars[i + 2].to_ascii_lowercase() == 'e' => {
                    tokens.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
                'v' if i + 2 < n && chars[i + 2].to_ascii_lowercase() == 'e' => {
                    tokens.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
                'l' if i + 2 < n && chars[i + 2].to_ascii_lowercase() == 'l' => {
                    tokens.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
                _ => {}
            }
        }

        // `\s*[\r\n]+` — optional whitespace then newlines
        if chars[i] == '\r' || chars[i] == '\n' || (chars[i].is_whitespace() && {
            // Look ahead for a newline
            let mut j = i;
            while j < n && chars[j].is_whitespace() && chars[j] != '\r' && chars[j] != '\n' {
                j += 1;
            }
            j < n && (chars[j] == '\r' || chars[j] == '\n')
        }) {
            let start = i;
            // Consume optional whitespace
            while i < n && chars[i].is_whitespace() && chars[i] != '\r' && chars[i] != '\n' {
                i += 1;
            }
            // Consume newlines
            while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // `[^\r\n\p{L}\p{N}]?\p{L}+` — optional non-letter/non-digit/non-newline, then letters
        if chars[i].is_alphabetic() || (!chars[i].is_alphanumeric() && !is_newline(chars[i]) && i + 1 < n && chars[i + 1].is_alphabetic()) {
            let start = i;
            // Optional leading non-letter/non-digit/non-newline
            if !chars[i].is_alphabetic() {
                i += 1;
            }
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // `\p{N}{1,3}` — 1 to 3 digits
        if chars[i].is_ascii_digit() {
            let start = i;
            let mut count = 0;
            while i < n && chars[i].is_ascii_digit() && count < 3 {
                i += 1;
                count += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+[\r\n]*` — optional space, non-alnum-non-ws, optional trailing newlines
        if (chars[i] == ' ' && i + 1 < n && is_punct_char(chars[i + 1]))
            || is_punct_char(chars[i])
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && is_punct_char(chars[i]) {
                i += 1;
            }
            // Trailing \r\n
            while i < n && (chars[i] == '\r' || chars[i] == '\n') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // `\s+(?!\S)|\s+` — whitespace
        if chars[i].is_whitespace() {
            let start = i;
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // Fallback
        tokens.push(chars[i].to_string());
        i += 1;
    }

    tokens
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

fn is_punct_char(c: char) -> bool {
    !c.is_whitespace() && !c.is_alphabetic() && !c.is_ascii_digit()
}

impl HfTokenizer {
    /// Load from a HuggingFace `tokenizer.json` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read tokenizer.json: {e}"))?;
        Self::from_json(&data)
    }

    /// Load from a `tokenizer.json` JSON string.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let root: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| format!("failed to parse JSON: {e}"))?;

        let model = root
            .get("model")
            .ok_or("missing 'model' key in tokenizer.json")?;

        // Parse vocab: { "token": id, ... }
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or("missing or invalid 'model.vocab'")?;

        let mut encoder = HashMap::with_capacity(vocab_obj.len());
        let mut decoder = HashMap::with_capacity(vocab_obj.len());
        for (token, id_val) in vocab_obj {
            let id = id_val
                .as_u64()
                .ok_or_else(|| format!("invalid vocab id for '{token}'"))?
                as usize;
            encoder.insert(token.clone(), id);
            decoder.insert(id, token.clone());
        }

        // Parse merges: ["token1 token2", ...]
        let merges_arr = model
            .get("merges")
            .and_then(|v| v.as_array())
            .ok_or("missing or invalid 'model.merges'")?;

        let mut bpe_ranks = HashMap::with_capacity(merges_arr.len());
        for (rank, merge_val) in merges_arr.iter().enumerate() {
            let merge_str = merge_val
                .as_str()
                .ok_or_else(|| format!("invalid merge at rank {rank}"))?;
            let mut parts = merge_str.splitn(2, ' ');
            let first = parts
                .next()
                .ok_or_else(|| format!("invalid merge format at rank {rank}"))?;
            let second = parts
                .next()
                .ok_or_else(|| format!("invalid merge format at rank {rank}"))?;
            bpe_ranks.insert((first.to_string(), second.to_string()), rank);
        }

        // Parse added_tokens for special tokens
        let mut special_encoder = HashMap::new();
        let mut special_decoder = HashMap::new();
        if let Some(added) = root.get("added_tokens").and_then(|v| v.as_array()) {
            for tok in added {
                let is_special = tok.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                if !is_special {
                    continue;
                }
                if let (Some(content), Some(id)) = (
                    tok.get("content").and_then(|v| v.as_str()),
                    tok.get("id").and_then(|v| v.as_u64()),
                ) {
                    let id = id as usize;
                    special_encoder.insert(content.to_string(), id);
                    special_decoder.insert(id, content.to_string());
                }
            }
        }

        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            special_encoder,
            special_decoder,
        })
    }

    /// Encode text into token IDs.
    ///
    /// Special tokens in the text (e.g. `<|begin_of_text|>`) are recognized and
    /// mapped to their IDs. Regular text is processed with BPE.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        // Split on special tokens first, then BPE-encode non-special segments
        let segments = self.split_on_special_tokens(text);
        let mut token_ids = Vec::new();

        for segment in segments {
            if let Some(&id) = self.special_encoder.get(&segment) {
                token_ids.push(id);
            } else {
                self.encode_segment(&segment, &mut token_ids);
            }
        }

        token_ids
    }

    /// Encode without inserting special tokens — treats all text as regular BPE input.
    pub fn encode_ordinary(&self, text: &str) -> Vec<usize> {
        let mut token_ids = Vec::new();
        for chunk in pre_tokenize_llama(text) {
            let bpe_input: String = chunk.bytes().map(|b| self.byte_encoder[b as usize]).collect();
            for tok in bpe(&bpe_input, &self.bpe_ranks) {
                if let Some(&id) = self.encoder.get(&tok) {
                    token_ids.push(id);
                }
            }
        }
        token_ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut pieces = Vec::with_capacity(tokens.len());

        for &id in tokens {
            if let Some(special) = self.special_decoder.get(&id) {
                pieces.push(special.clone());
            } else if let Some(bpe_tok) = self.decoder.get(&id) {
                // Convert byte-level unicode back to raw bytes
                let bytes: Vec<u8> = bpe_tok
                    .chars()
                    .filter_map(|c| self.byte_decoder.get(&c).copied())
                    .collect();
                pieces.push(String::from_utf8_lossy(&bytes).into_owned());
            }
        }

        pieces.join("")
    }

    /// Vocabulary size including special tokens.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len() + self.special_encoder.len()
    }

    /// Look up a special token ID by name.
    pub fn special_token_id(&self, name: &str) -> Option<usize> {
        self.special_encoder.get(name).copied()
    }

    /// BOS token id (`<|begin_of_text|>`).
    pub fn bos_id(&self) -> Option<usize> {
        self.special_token_id("<|begin_of_text|>")
    }

    /// EOS token id (`<|end_of_text|>`).
    pub fn eos_id(&self) -> Option<usize> {
        self.special_token_id("<|end_of_text|>")
    }

    /// EOT (end-of-turn) token id (`<|eot_id|>`).
    pub fn eot_id(&self) -> Option<usize> {
        self.special_token_id("<|eot_id|>")
    }

    /// Encode a chat prompt with Llama 3 chat template.
    ///
    /// Each message is a `(role, content)` pair. Roles are typically
    /// "system", "user", or "assistant".
    pub fn apply_chat_template(&self, messages: &[(&str, &str)]) -> Vec<usize> {
        let mut ids = Vec::new();

        // <|begin_of_text|>
        if let Some(bos) = self.bos_id() {
            ids.push(bos);
        }

        let start_header = self.special_token_id("<|start_header_id|>");
        let end_header = self.special_token_id("<|end_header_id|>");
        let eot = self.eot_id();

        for &(role, content) in messages {
            // <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
            if let Some(id) = start_header {
                ids.push(id);
            }
            ids.extend(self.encode_ordinary(role));
            if let Some(id) = end_header {
                ids.push(id);
            }
            ids.extend(self.encode_ordinary("\n\n"));
            ids.extend(self.encode_ordinary(content));
            if let Some(id) = eot {
                ids.push(id);
            }
        }

        ids
    }

    /// Split text into segments, separating special token strings from regular text.
    fn split_on_special_tokens(&self, text: &str) -> Vec<String> {
        if self.special_encoder.is_empty() {
            return vec![text.to_string()];
        }

        // Build sorted list of special tokens (longest first for greedy matching)
        let mut specials: Vec<&str> = self.special_encoder.keys().map(|s| s.as_str()).collect();
        specials.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Find the earliest special token occurrence
            let mut earliest_pos = remaining.len();
            let mut earliest_token = "";
            for &special in &specials {
                if let Some(pos) = remaining.find(special) {
                    if pos < earliest_pos {
                        earliest_pos = pos;
                        earliest_token = special;
                    }
                }
            }

            if earliest_pos == remaining.len() {
                // No special token found
                segments.push(remaining.to_string());
                break;
            }

            // Push text before the special token
            if earliest_pos > 0 {
                segments.push(remaining[..earliest_pos].to_string());
            }
            // Push the special token itself
            segments.push(earliest_token.to_string());
            remaining = &remaining[earliest_pos + earliest_token.len()..];
        }

        segments
    }

    /// BPE-encode a single text segment (not a special token).
    fn encode_segment(&self, text: &str, token_ids: &mut Vec<usize>) {
        for chunk in pre_tokenize_llama(text) {
            let bpe_input: String = chunk.bytes().map(|b| self.byte_encoder[b as usize]).collect();
            for tok in bpe(&bpe_input, &self.bpe_ranks) {
                if let Some(&id) = self.encoder.get(&tok) {
                    token_ids.push(id);
                }
            }
        }
    }
}

/// Shared BPE merge loop.
fn bpe(token: &str, bpe_ranks: &HashMap<(String, String), usize>) -> Vec<String> {
    if token.is_empty() {
        return Vec::new();
    }

    let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

    if word.len() == 1 {
        return word;
    }

    loop {
        let mut best_pair = None;
        let mut best_rank = usize::MAX;

        for i in 0..word.len() - 1 {
            let pair = (word[i].clone(), word[i + 1].clone());
            if let Some(&rank) = bpe_ranks.get(&pair) {
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
        let mut new_word = Vec::new();
        let mut i = 0;
        while i < word.len() {
            if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
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
