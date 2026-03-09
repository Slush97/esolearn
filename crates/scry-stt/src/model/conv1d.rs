use std::cell::RefCell;

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// 1D convolution layer for Whisper's audio feature extraction stem.
///
/// Whisper uses two `Conv1D` layers to downsample the mel spectrogram:
///   - Conv1D(80, `d_model`, kernel=3, stride=1, padding=1)
///   - `Conv1D(d_model`, `d_model`, kernel=3, stride=2, padding=1)
///
/// Input shape: `[channels_in, length]`
/// Output shape: `[channels_out, output_length]`
pub struct Conv1d<B: MathBackend> {
    /// Weight tensor: `[out_channels, in_channels, kernel_size]`.
    pub weight: Tensor<B>,
    /// Bias tensor: `[out_channels]`.
    pub bias: Tensor<B>,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Convolution kernel size.
    pub kernel_size: usize,
    /// Convolution stride.
    pub stride: usize,
    /// Padding added to both sides of the input.
    pub padding: usize,
    /// Reusable im2col workspace buffer to avoid per-forward allocation.
    pub workspace: RefCell<Vec<f32>>,
}

impl<B: MathBackend> Conv1d<B> {
    /// Create a new `Conv1D` layer with random initialization.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        rng: &mut fastrand::Rng,
    ) -> Self {
        let n = in_channels * out_channels * kernel_size;
        let std_dev = (2.0 / (in_channels * kernel_size) as f64).sqrt();
        let w_data: Vec<f32> = (0..n)
            .map(|_| (rng.f64() * 2.0 - 1.0) * std_dev)
            .map(|v| v as f32)
            .collect();
        let b_data = vec![0.0f32; out_channels];

        // Pre-allocate workspace for the expected input length (3000 frames for 30s audio).
        let expected_len = 3000;
        let expected_out = (expected_len + 2 * padding - kernel_size) / stride + 1;
        let col_rows = kernel_size * in_channels;
        let workspace = RefCell::new(vec![0.0f32; col_rows * expected_out]);

        Self {
            weight: Tensor::from_vec(
                w_data,
                Shape::new(&[out_channels, in_channels, kernel_size]),
            ),
            bias: Tensor::from_vec(b_data, Shape::new(&[out_channels])),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            workspace,
        }
    }

    /// Forward pass: `input` is `[in_channels, length]` → `[out_channels, out_length]`.
    ///
    /// Performs convolution by unrolling to matrix multiplication for efficiency.
    /// Reuses an internal workspace buffer for the im2col matrix to avoid
    /// allocating on every call.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let length = input.shape.dims()[1];
        let padded_len = length + 2 * self.padding;
        let out_length = (padded_len - self.kernel_size) / self.stride + 1;

        let input_vec = input.to_vec();

        // Build im2col matrix: [kernel_size * in_channels, out_length]
        let col_rows = self.kernel_size * self.in_channels;
        let needed = col_rows * out_length;
        let mut col = self.workspace.borrow_mut();
        if col.len() < needed {
            col.resize(needed, 0.0);
        }

        for out_pos in 0..out_length {
            let in_start = out_pos * self.stride;
            for c in 0..self.in_channels {
                for k in 0..self.kernel_size {
                    let in_pos = in_start + k;
                    let val = if in_pos >= self.padding && in_pos < self.padding + length {
                        input_vec[c * length + (in_pos - self.padding)]
                    } else {
                        0.0 // zero padding
                    };
                    col[(c * self.kernel_size + k) * out_length + out_pos] = val;
                }
            }
        }

        let col_storage = B::from_vec(col[..needed].to_vec(), &Shape::new(&[col_rows, out_length]));

        // Weight [out_channels, in_channels, kernel_size] has the same flat layout as
        // [out_channels, in_channels * kernel_size] in row-major — use directly, no clone.
        let out_data = B::matmul(
            &self.weight.data,
            &col_storage,
            self.out_channels,
            col_rows,
            out_length,
            false,
            false,
        );

        // Column broadcast bias add: bias[c] added to every element of row c.
        // Reshape bias [out_channels] → [out_channels, 1] to hit the column broadcast
        // fast path in CpuBackend::add, avoiding a to_vec() clone of the matmul output.
        let bias_col = B::from_vec(self.bias.to_vec(), &Shape::new(&[self.out_channels, 1]));
        let out_shape = Shape::new(&[self.out_channels, out_length]);
        let result = B::add(
            &out_data,
            &bias_col,
            &out_shape,
            &Shape::new(&[self.out_channels, 1]),
            &out_shape,
        );
        Tensor::<B>::new(result, out_shape)
    }
}

impl<B: MathBackend> Module<B> for Conv1d<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.weight, &self.bias]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn conv1d_output_shape_stride1() {
        let mut rng = fastrand::Rng::with_seed(42);
        let conv = Conv1d::<CpuBackend>::new(80, 512, 3, 1, 1, &mut rng);
        let input = Tensor::<CpuBackend>::from_vec(
            vec![0.0f32; 80 * 3000],
            Shape::new(&[80, 3000]),
        );
        let output = conv.forward(&input);
        assert_eq!(output.shape.dims(), &[512, 3000]);
    }

    #[test]
    fn conv1d_output_shape_stride2() {
        let mut rng = fastrand::Rng::with_seed(42);
        let conv = Conv1d::<CpuBackend>::new(512, 512, 3, 2, 1, &mut rng);
        let input = Tensor::<CpuBackend>::from_vec(
            vec![0.0f32; 512 * 3000],
            Shape::new(&[512, 3000]),
        );
        let output = conv.forward(&input);
        assert_eq!(output.shape.dims(), &[512, 1500]);
    }
}
