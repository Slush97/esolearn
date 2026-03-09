use crate::backend::MathBackend;
use crate::nn::init;
use crate::nn::Module;
use crate::ops;
use crate::tensor::shape::Shape;
use crate::tensor::Tensor;

/// Quantized weight state for a Linear layer (W8A32).
#[cfg(feature = "quantize")]
pub struct QuantState<B: MathBackend> {
    pub weight_q: B::I8Storage,
    pub scale: f32,
}

/// Linear layer: `output = input @ weight + bias`.
///
/// Weight stored as `[in_features, out_features]` — matches HF `Conv1D` layout.
pub struct Linear<B: MathBackend> {
    pub weight: Tensor<B>,
    pub bias: Tensor<B>,
    pub in_features: usize,
    pub out_features: usize,
    #[cfg(feature = "quantize")]
    pub quant: Option<QuantState<B>>,
}

impl<B: MathBackend> Linear<B> {
    pub fn new(in_features: usize, out_features: usize, rng: &mut fastrand::Rng) -> Self {
        let w_data = init::normal_vec(rng, in_features * out_features, 0.0, 0.02);
        let b_data = vec![0.0f32; out_features];
        Self {
            weight: Tensor::from_vec(w_data, Shape::new(&[in_features, out_features])),
            bias: Tensor::from_vec(b_data, Shape::new(&[out_features])),
            in_features,
            out_features,
            #[cfg(feature = "quantize")]
            quant: None,
        }
    }

    /// Apply residual scaling: multiply weights by `1/sqrt(2*n_layer)`.
    pub fn apply_residual_scaling(&mut self, n_layer: usize) {
        let scale = 1.0 / (2.0 * n_layer as f64).sqrt();
        let mut data = self.weight.to_vec();
        for v in &mut data {
            *v = (f64::from(*v) * scale) as f32;
        }
        self.weight = Tensor::from_vec(data, self.weight.shape.clone());
    }

    /// Quantize weights in-place to INT8 (symmetric per-tensor).
    ///
    /// After calling this, `forward()` will use the quantized matmul path.
    /// The original f32 weights are kept for potential dequantization / export.
    #[cfg(feature = "quantize")]
    pub fn quantize_weights(&mut self) {
        use crate::quantize::quantize_symmetric;

        let w_f32 = B::to_vec(&self.weight.data);
        let (q_data, meta) = quantize_symmetric(&w_f32);
        self.quant = Some(QuantState {
            weight_q: B::i8_from_vec(q_data),
            scale: meta.scale,
        });
    }

    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let seq = input.shape.dims()[0];

        #[cfg(feature = "quantize")]
        if let Some(ref qs) = self.quant {
            let out = B::matmul_i8_f32_bias(
                &input.data,
                &qs.weight_q,
                qs.scale,
                &self.bias.data,
                seq,
                self.in_features,
                self.out_features,
            );
            return Tensor::new(out, Shape::new(&[seq, self.out_features]));
        }

        ops::matmul_bias(
            input,
            &self.weight,
            &self.bias,
            seq,
            self.in_features,
            self.out_features,
            false,
            false,
        )
    }
}

impl<B: MathBackend> Module<B> for Linear<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.weight, &self.bias]
    }
}
