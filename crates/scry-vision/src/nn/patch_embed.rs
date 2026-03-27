// SPDX-License-Identifier: MIT OR Apache-2.0
//! Patch embedding for Vision Transformer (ViT).
//!
//! Splits an image into non-overlapping patches, flattens each patch,
//! and projects to the embedding dimension.
//!
//! Input shape: `[C, H, W]` (H and W must be divisible by `patch_size`)
//! Output shape: `[num_patches, embed_dim]`

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

/// Patch embedding: image → sequence of patch embeddings.
///
/// Equivalent to `nn.Conv2d(C, embed_dim, kernel_size=P, stride=P)` followed
/// by reshape, but implemented directly for efficiency (no im2col overhead
/// since patches are non-overlapping).
pub struct PatchEmbedding<B: MathBackend> {
    /// Projection weight: `[embed_dim, in_channels * patch_size * patch_size]`.
    pub proj: Tensor<B>,
    /// Projection bias: `[embed_dim]`.
    pub bias: Tensor<B>,
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
}

impl<B: MathBackend> PatchEmbedding<B> {
    /// Create a zero-initialized PatchEmbedding (for testing; real usage loads from checkpoint).
    pub fn new(in_channels: usize, embed_dim: usize, patch_size: usize) -> Self {
        let patch_len = in_channels * patch_size * patch_size;
        Self {
            proj: Tensor::from_vec(
                vec![0.0f32; embed_dim * patch_len],
                Shape::new(&[embed_dim, patch_len]),
            ),
            bias: Tensor::from_vec(vec![0.0f32; embed_dim], Shape::new(&[embed_dim])),
            patch_size,
            in_channels,
            embed_dim,
        }
    }

    /// Forward pass: `[C, H, W]` → `[num_patches, embed_dim]`.
    ///
    /// Panics if H or W is not divisible by `patch_size`.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let dims = input.shape.dims();
        let c = dims[0];
        let h = dims[1];
        let w = dims[2];
        let p = self.patch_size;

        assert_eq!(h % p, 0, "height {h} not divisible by patch_size {p}");
        assert_eq!(w % p, 0, "width {w} not divisible by patch_size {p}");

        let patches_h = h / p;
        let patches_w = w / p;
        let num_patches = patches_h * patches_w;
        let patch_len = c * p * p;

        let input_vec = input.to_vec();

        // Extract patches: [num_patches, C * P * P]
        let mut patches = vec![0.0f32; num_patches * patch_len];
        for ph in 0..patches_h {
            for pw in 0..patches_w {
                let patch_idx = ph * patches_w + pw;
                for ch in 0..c {
                    for i in 0..p {
                        for j in 0..p {
                            let src = ch * h * w + (ph * p + i) * w + (pw * p + j);
                            let dst = patch_idx * patch_len + ch * p * p + i * p + j;
                            patches[dst] = input_vec[src];
                        }
                    }
                }
            }
        }

        // Linear projection: patches[num_patches, patch_len] @ proj^T → [num_patches, embed_dim]
        let patches_storage =
            B::from_vec(patches, &Shape::new(&[num_patches, patch_len]));
        let out_data = B::matmul(
            &patches_storage,
            &self.proj.data,
            num_patches,
            patch_len,
            self.embed_dim,
            false,
            true,
        );

        // Add bias[embed_dim] broadcast across patches
        let bias_row = B::from_vec(self.bias.to_vec(), &Shape::new(&[1, self.embed_dim]));
        let out_shape = Shape::new(&[num_patches, self.embed_dim]);
        let result = B::add(
            &out_data,
            &bias_row,
            &out_shape,
            &Shape::new(&[1, self.embed_dim]),
            &out_shape,
        );

        Tensor::<B>::new(result, out_shape)
    }
}

impl<B: MathBackend> Module<B> for PatchEmbedding<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.proj, &self.bias]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn output_shape() {
        let pe = PatchEmbedding::<CpuBackend>::new(3, 768, 16);
        let input = Tensor::from_vec(vec![0.0; 3 * 224 * 224], Shape::new(&[3, 224, 224]));
        let output = pe.forward(&input);
        // 224/16 = 14, 14×14 = 196 patches
        assert_eq!(output.shape.dims(), &[196, 768]);
    }

    #[test]
    fn small_patch_identity_proj() {
        // 1 channel, 2×2 image, patch_size=2 → 1 patch of len 4
        // proj = identity-like: embed_dim=4, proj = I(4×4)
        let mut pe = PatchEmbedding::<CpuBackend>::new(1, 4, 2);
        pe.proj = Tensor::from_vec(
            vec![
                1.0, 0.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, 0.0, // row 1
                0.0, 0.0, 1.0, 0.0, // row 2
                0.0, 0.0, 0.0, 1.0, // row 3
            ],
            Shape::new(&[4, 4]),
        );

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(&[1, 2, 2]));
        let output = pe.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 4]);
        let data = output.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn multiple_patches() {
        // 1 channel, 4×4 image, patch_size=2 → 4 patches
        let pe = PatchEmbedding::<CpuBackend>::new(1, 2, 2);
        let input = Tensor::from_vec(vec![0.0; 16], Shape::new(&[1, 4, 4]));
        let output = pe.forward(&input);
        assert_eq!(output.shape.dims(), &[4, 2]);
    }

    #[test]
    fn with_bias() {
        let mut pe = PatchEmbedding::<CpuBackend>::new(1, 1, 1);
        // proj = [1.0], bias = [10.0]
        pe.proj = Tensor::from_vec(vec![1.0], Shape::new(&[1, 1]));
        pe.bias = Tensor::from_vec(vec![10.0], Shape::new(&[1]));

        let input = Tensor::from_vec(vec![5.0], Shape::new(&[1, 1, 1]));
        let output = pe.forward(&input);
        assert!((output.to_vec()[0] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn multi_channel_patches() {
        // 3 channels, 2×2 image, patch_size=2 → 1 patch of len 12
        let pe = PatchEmbedding::<CpuBackend>::new(3, 8, 2);
        let input = Tensor::from_vec(vec![1.0; 12], Shape::new(&[3, 2, 2]));
        let output = pe.forward(&input);
        assert_eq!(output.shape.dims(), &[1, 8]);
    }

    #[test]
    fn parameters() {
        let pe = PatchEmbedding::<CpuBackend>::new(3, 768, 16);
        let params = pe.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].numel(), 768 * 3 * 16 * 16); // proj
        assert_eq!(params[1].numel(), 768); // bias
    }

    #[test]
    #[should_panic(expected = "not divisible")]
    fn panics_on_indivisible_height() {
        let pe = PatchEmbedding::<CpuBackend>::new(3, 768, 16);
        let input = Tensor::from_vec(vec![0.0; 3 * 225 * 224], Shape::new(&[3, 225, 224]));
        pe.forward(&input);
    }
}
