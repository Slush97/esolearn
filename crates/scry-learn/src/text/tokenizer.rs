// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text tokenization utilities.
//!
//! Zero-dependency tokenizer that splits text at whitespace and punctuation
//! boundaries, normalizes to lowercase, and supports n-gram generation.

/// Tokenize text into lowercase words, stripping punctuation.
///
/// Splits on whitespace, removes non-alphanumeric characters from token
/// boundaries, and filters empty tokens.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::text::tokenizer::default_tokenize;
///
/// let tokens = default_tokenize("Hello, World! It's a test.");
/// assert_eq!(tokens, vec!["hello", "world", "it's", "a", "test"]);
/// ```
pub fn default_tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            // Strip leading/trailing punctuation, preserve internal (e.g. apostrophes)
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Generate n-grams from a list of tokens.
///
/// `range` is `(min_n, max_n)` inclusive on both ends.
/// For `(1, 1)` this returns unigrams (the original tokens).
/// For `(1, 2)` this returns both unigrams and bigrams.
///
/// # Examples
///
/// ```ignore
/// use scry_learn::text::tokenizer::ngrams;
///
/// let tokens: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
/// let result = ngrams(&tokens, (1, 2));
/// // ["a", "b", "c", "a b", "b c"]
/// ```
pub fn ngrams(tokens: &[String], range: (usize, usize)) -> Vec<String> {
    let (min_n, max_n) = range;
    let min_n = min_n.max(1);
    let max_n = max_n.max(min_n);

    let mut result = Vec::new();

    for n in min_n..=max_n {
        if n > tokens.len() {
            continue;
        }
        for window in tokens.windows(n) {
            if n == 1 {
                result.push(window[0].clone());
            } else {
                result.push(window.join(" "));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tokenization() {
        let tokens = default_tokenize("Hello, World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn handles_punctuation() {
        let tokens = default_tokenize("It's a well-known fact, indeed!");
        assert_eq!(tokens, vec!["it's", "a", "well-known", "fact", "indeed"]);
    }

    #[test]
    fn handles_empty_string() {
        let tokens = default_tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn handles_only_whitespace() {
        let tokens = default_tokenize("   \t\n  ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn handles_only_punctuation() {
        let tokens = default_tokenize("!!! ??? ...");
        assert!(tokens.is_empty());
    }

    #[test]
    fn unigrams() {
        let tokens = vec!["a".into(), "b".into(), "c".into()];
        let result = ngrams(&tokens, (1, 1));
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn bigrams() {
        let tokens = vec!["a".into(), "b".into(), "c".into()];
        let result = ngrams(&tokens, (2, 2));
        assert_eq!(result, vec!["a b", "b c"]);
    }

    #[test]
    fn unigrams_and_bigrams() {
        let tokens = vec!["a".into(), "b".into(), "c".into()];
        let result = ngrams(&tokens, (1, 2));
        assert_eq!(result, vec!["a", "b", "c", "a b", "b c"]);
    }

    #[test]
    fn ngrams_larger_than_input() {
        let tokens = vec!["a".into(), "b".into()];
        let result = ngrams(&tokens, (3, 3));
        assert!(result.is_empty());
    }

    #[test]
    fn trigrams() {
        let tokens = vec!["the".into(), "cat".into(), "sat".into(), "down".into()];
        let result = ngrams(&tokens, (3, 3));
        assert_eq!(result, vec!["the cat sat", "cat sat down"]);
    }
}
