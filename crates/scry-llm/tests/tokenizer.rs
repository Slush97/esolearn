#![cfg(feature = "tokenizer")]

use scry_llm::tokenizer::{BpeTokenizer, HfTokenizer};
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_tokenizer() -> Option<BpeTokenizer> {
    let vocab = fixture_dir().join("vocab.json");
    let merges = fixture_dir().join("merges.txt");
    if !vocab.exists() || !merges.exists() {
        eprintln!(
            "Skipping tokenizer test: download vocab.json and merges.txt from \
             https://huggingface.co/openai-community/gpt2/tree/main into tests/fixtures/"
        );
        return None;
    }
    Some(BpeTokenizer::from_files(&vocab, &merges).expect("failed to load tokenizer"))
}

#[test]
fn known_tokenizations() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    // "Hello world" -> [15496, 995]
    let ids = tok.encode("Hello world");
    assert_eq!(ids, vec![15496, 995], "Hello world tokenization mismatch: {ids:?}");

    // "Hello" alone
    let ids = tok.encode("Hello");
    assert_eq!(ids, vec![15496], "Hello tokenization mismatch: {ids:?}");
}

#[test]
fn round_trip_basic() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    let texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-2 is a language model.",
        "1234567890",
        "foo bar baz",
    ];

    for text in texts {
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "round trip failed for: {text:?}");
    }
}

#[test]
fn round_trip_unicode() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    let text = "caf\u{00e9} na\u{00ef}ve";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text, "unicode round trip failed");
}

#[test]
fn empty_input() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    let ids = tok.encode("");
    assert!(ids.is_empty(), "empty input should produce no tokens");

    let decoded = tok.decode(&[]);
    assert!(decoded.is_empty(), "decoding empty tokens should produce empty string");
}

#[test]
fn special_token_endoftext() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    // <|endoftext|> = 50256
    let ids = tok.encode("<|endoftext|>");
    // The special token is encoded character-by-character in standard GPT-2 BPE
    // (it's not handled as a single special token in the base tokenizer)
    assert!(!ids.is_empty());

    // But decoding 50256 should give back the special token string
    let decoded = tok.decode(&[50256]);
    assert_eq!(decoded, "<|endoftext|>");
}

#[test]
fn vocab_size_is_50257() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    assert_eq!(tok.vocab_size(), 50257, "GPT-2 vocab size should be 50257");
}

#[test]
fn whitespace_handling() {
    let Some(tok) = load_tokenizer() else {
        return;
    };

    // Space-only
    let ids = tok.encode(" ");
    assert!(!ids.is_empty(), "space should produce tokens");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, " ");

    // Multiple spaces
    let text = "hello  world";
    let ids = tok.encode(text);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text);
}

// ---- HfTokenizer tests ----

/// Build a minimal tokenizer.json for testing.
fn make_test_hf_json() -> String {
    // Build a tiny BPE vocab with bytes-to-unicode chars.
    // We include the byte-level unicode chars for ASCII printable range.
    // For simplicity, map individual bytes + a few merges.
    let mut vocab = serde_json::Map::new();
    // Byte-level unicode chars for ASCII letters (these are identity-mapped in bytes_to_unicode)
    let byte_chars = [
        ("H", 0), ("e", 1), ("l", 2), ("o", 3), ("Ġ", 4), // Ġ = byte 0x20 (space) mapped to U+0120
        ("w", 5), ("r", 6), ("d", 7), ("!", 8), (".", 9),
        ("He", 10), ("ll", 11), ("Hello", 12),
        ("wo", 13), ("rld", 14), ("world", 15),
        ("Ġworld", 16),
    ];
    for (tok, id) in &byte_chars {
        vocab.insert(tok.to_string(), serde_json::Value::Number((*id).into()));
    }

    let merges = vec![
        "H e", "l l", "He ll", "Hell o",
        "w o", "r l", "rl d", "wo rld",
        "Ġ world",
    ];

    let added_tokens = vec![
        serde_json::json!({
            "id": 100, "content": "<|begin_of_text|>",
            "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": true
        }),
        serde_json::json!({
            "id": 101, "content": "<|end_of_text|>",
            "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": true
        }),
        serde_json::json!({
            "id": 102, "content": "<|start_header_id|>",
            "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": true
        }),
        serde_json::json!({
            "id": 103, "content": "<|end_header_id|>",
            "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": true
        }),
        serde_json::json!({
            "id": 104, "content": "<|eot_id|>",
            "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": true
        }),
    ];

    let root = serde_json::json!({
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges,
            "byte_fallback": false,
        },
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": null,
        "decoder": null,
        "post_processor": null,
    });

    serde_json::to_string(&root).unwrap()
}

#[test]
fn hf_tokenizer_parse_json() {
    let json = make_test_hf_json();
    let tok = HfTokenizer::from_json(&json).expect("failed to parse");
    // 17 regular tokens + 5 special = 22
    assert_eq!(tok.vocab_size(), 22);
}

#[test]
fn hf_tokenizer_special_token_ids() {
    let json = make_test_hf_json();
    let tok = HfTokenizer::from_json(&json).expect("failed to parse");
    assert_eq!(tok.bos_id(), Some(100));
    assert_eq!(tok.eos_id(), Some(101));
    assert_eq!(tok.eot_id(), Some(104));
    assert_eq!(tok.special_token_id("<|start_header_id|>"), Some(102));
    assert_eq!(tok.special_token_id("<|end_header_id|>"), Some(103));
}

#[test]
fn hf_tokenizer_encode_special_tokens() {
    let json = make_test_hf_json();
    let tok = HfTokenizer::from_json(&json).expect("failed to parse");

    // Special token in text should be recognized
    let ids = tok.encode("<|begin_of_text|>");
    assert_eq!(ids, vec![100]);

    // Multiple special tokens
    let ids = tok.encode("<|begin_of_text|><|end_of_text|>");
    assert_eq!(ids, vec![100, 101]);
}

#[test]
fn hf_tokenizer_decode_special_tokens() {
    let json = make_test_hf_json();
    let tok = HfTokenizer::from_json(&json).expect("failed to parse");
    let decoded = tok.decode(&[100, 101]);
    assert_eq!(decoded, "<|begin_of_text|><|end_of_text|>");
}

#[test]
fn hf_tokenizer_empty() {
    let json = make_test_hf_json();
    let tok = HfTokenizer::from_json(&json).expect("failed to parse");
    assert!(tok.encode("").is_empty());
    assert!(tok.decode(&[]).is_empty());
}

/// Test with real Llama 3 tokenizer.json if available.
fn load_hf_tokenizer() -> Option<HfTokenizer> {
    let path = fixture_dir().join("tokenizer.json");
    if !path.exists() {
        eprintln!(
            "Skipping HF tokenizer test: download tokenizer.json from \
             https://huggingface.co/meta-llama/Llama-3.2-1B into tests/fixtures/"
        );
        return None;
    }
    Some(HfTokenizer::from_file(&path).expect("failed to load tokenizer.json"))
}

#[test]
fn hf_real_tokenizer_round_trip() {
    let Some(tok) = load_hf_tokenizer() else {
        return;
    };

    let texts = [
        "Hello world",
        "The capital of France is Paris.",
        "1 + 1 = 2",
        "Rust is a systems programming language.",
    ];

    for text in texts {
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "HF round trip failed for: {text:?}");
    }
}

#[test]
fn hf_real_tokenizer_special_tokens() {
    let Some(tok) = load_hf_tokenizer() else {
        return;
    };

    // Llama 3 special token IDs
    assert_eq!(tok.bos_id(), Some(128000));
    assert_eq!(tok.eos_id(), Some(128001));
    assert_eq!(tok.eot_id(), Some(128009));
}

#[test]
fn hf_real_tokenizer_chat_template() {
    let Some(tok) = load_hf_tokenizer() else {
        return;
    };

    let messages = vec![("user", "Hello")];
    let ids = tok.apply_chat_template(&messages);

    // Should start with <|begin_of_text|>
    assert_eq!(ids[0], 128000);

    // Should contain <|start_header_id|> and <|end_header_id|>
    assert!(ids.contains(&128006)); // start_header_id
    assert!(ids.contains(&128007)); // end_header_id

    // Should end with <|eot_id|>
    assert_eq!(*ids.last().unwrap(), 128009);

    // Decode and verify structure
    let decoded = tok.decode(&ids);
    assert!(decoded.contains("user"));
    assert!(decoded.contains("Hello"));
}
