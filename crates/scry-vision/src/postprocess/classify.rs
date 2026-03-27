// SPDX-License-Identifier: MIT OR Apache-2.0
//! Top-k classification from logit vectors.

/// A single classification result: class index + score.
#[derive(Clone, Debug, PartialEq)]
pub struct Classification {
    pub class_id: u32,
    pub score: f32,
}

/// Return the top-k classes from raw logits, applying softmax first.
///
/// Results are sorted by score descending. If `k > num_classes`, all classes
/// are returned.
pub fn top_k_softmax(logits: &[f32], k: usize) -> Vec<Classification> {
    let probs = softmax(logits);
    top_k_from_scores(&probs, k)
}

/// Return the top-k classes from pre-computed scores (no softmax applied).
///
/// Useful when the model already outputs probabilities or sigmoid scores.
pub fn top_k_from_scores(scores: &[f32], k: usize) -> Vec<Classification> {
    let mut indexed: Vec<(u32, f32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i as u32, s))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed
        .into_iter()
        .map(|(class_id, score)| Classification { class_id, score })
        .collect()
}

/// Numerically stable softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_uniform() {
        let logits = vec![1.0, 1.0, 1.0];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_dominant_class() {
        let logits = vec![0.0, 0.0, 100.0];
        let probs = softmax(&logits);
        assert!(probs[2] > 0.99);
    }

    #[test]
    fn softmax_empty() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn top_k_softmax_basic() {
        let logits = vec![1.0, 5.0, 2.0, 0.5];
        let result = top_k_softmax(&logits, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].class_id, 1); // highest logit
        assert_eq!(result[1].class_id, 2); // second highest
        assert!(result[0].score > result[1].score);
    }

    #[test]
    fn top_k_from_scores_basic() {
        let scores = vec![0.1, 0.7, 0.15, 0.05];
        let result = top_k_from_scores(&scores, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].class_id, 1);
        assert!((result[0].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn top_k_larger_than_classes() {
        let scores = vec![0.3, 0.7];
        let result = top_k_from_scores(&scores, 10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn top_k_single_class() {
        let logits = vec![42.0];
        let result = top_k_softmax(&logits, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].class_id, 0);
        assert!((result[0].score - 1.0).abs() < 1e-6);
    }
}
