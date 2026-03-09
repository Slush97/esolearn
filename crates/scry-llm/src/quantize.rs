//! Symmetric INT8 quantization: W8A32 (8-bit weights, 32-bit activations).
//!
//! Per-tensor symmetric quantization: `scale = absmax / 127`, `q[i] = round(w[i] / scale)`.

/// Metadata for a symmetrically quantized tensor.
#[derive(Clone, Debug)]
pub struct QuantMeta {
    /// Scale factor: `absmax / 127.0`. Multiply `i8` values by this to reconstruct f32.
    pub scale: f32,
}

/// Symmetrically quantize f32 weights to i8.
///
/// Returns `(quantized_weights, metadata)`. The quantized value for each weight is
/// `round(weight / scale)` clamped to `[-127, 127]`.
pub fn quantize_symmetric(weights: &[f32]) -> (Vec<i8>, QuantMeta) {
    let absmax = weights
        .iter()
        .fold(0.0f32, |acc, &w| acc.max(w.abs()));

    if absmax == 0.0 {
        return (vec![0i8; weights.len()], QuantMeta { scale: 0.0 });
    }

    let scale = absmax / 127.0;
    let inv_scale = 1.0 / scale;

    let quantized: Vec<i8> = weights
        .iter()
        .map(|&w| (w * inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect();

    (quantized, QuantMeta { scale })
}

/// Dequantize i8 weights back to f32 (for testing / validation).
pub fn dequantize(quantized: &[i8], meta: &QuantMeta) -> Vec<f32> {
    quantized
        .iter()
        .map(|&q| f32::from(q) * meta.scale)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_accuracy() {
        let weights: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let (quantized, meta) = quantize_symmetric(&weights);
        let recovered = dequantize(&quantized, &meta);

        let max_abs = weights.iter().fold(0.0f32, |a, &w| a.max(w.abs()));
        let max_error = weights
            .iter()
            .zip(recovered.iter())
            .map(|(&w, &r)| (w - r).abs())
            .fold(0.0f32, f32::max);

        // Max error should be less than 1/127 of the range
        let tolerance = max_abs / 127.0;
        assert!(
            max_error <= tolerance + 1e-6,
            "max error {max_error} exceeds tolerance {tolerance}"
        );
    }

    #[test]
    fn zero_weights() {
        let weights = vec![0.0f32; 10];
        let (quantized, meta) = quantize_symmetric(&weights);
        assert!(quantized.iter().all(|&q| q == 0));
        assert_eq!(meta.scale, 0.0);
    }

    #[test]
    fn clamps_to_range() {
        // Even extreme values get clamped to [-127, 127]
        let weights = vec![1000.0, -1000.0, 0.0];
        let (quantized, _) = quantize_symmetric(&weights);
        assert_eq!(quantized[0], 127);
        assert_eq!(quantized[1], -127);
        assert_eq!(quantized[2], 0);
    }
}
