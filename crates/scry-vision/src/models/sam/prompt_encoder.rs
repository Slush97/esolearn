// SPDX-License-Identifier: MIT OR Apache-2.0
//! SAM prompt encoder — converts point/box/mask prompts into embeddings.
//!
//! Outputs sparse embeddings (from points/boxes) and dense embeddings
//! (from mask input or a learned "no mask" embedding broadcast over spatial dims).

use scry_llm::backend::MathBackend;
use scry_llm::nn::Module;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::nn::{Conv2d, LayerNorm2d};
use crate::pipeline::SegmentPrompt;

use super::SamConfig;

// ── Positional encoding (Fourier features) ──────────────────────────────────

/// Random Fourier feature positional encoding.
///
/// Maps 2D coordinates to a `[embed_dim]` vector via:
/// `[sin(2π · x · B), cos(2π · x · B), sin(2π · y · B), cos(2π · y · B)]`
/// where `B` is a learned Gaussian matrix of shape `[2, num_pos_feats]`.
pub struct PositionalEncoding<B: MathBackend> {
    /// Gaussian random matrix: `[2, num_pos_feats]`.
    pub gaussian_matrix: Tensor<B>,
    pub num_pos_feats: usize,
}

impl<B: MathBackend> PositionalEncoding<B> {
    /// Create with zero-initialized Gaussian matrix.
    pub fn new(num_pos_feats: usize) -> Self {
        Self {
            gaussian_matrix: Tensor::from_vec(
                vec![0.0; 2 * num_pos_feats],
                Shape::new(&[2, num_pos_feats]),
            ),
            num_pos_feats,
        }
    }

    /// Encode a normalized `(x, y)` point (in `[0, 1]`) to `[embed_dim]`.
    ///
    /// Output dimension is `num_pos_feats * 4` = `embed_dim` (typically 256).
    pub fn encode_point(&self, x: f32, y: f32) -> Vec<f32> {
        let gauss = self.gaussian_matrix.to_vec();
        let npf = self.num_pos_feats;
        let mut out = vec![0.0f32; 4 * npf]; // sin_x, cos_x, sin_y, cos_y

        for i in 0..npf {
            // B[0, i] * x, B[1, i] * y
            let val_x = 2.0 * std::f32::consts::PI * x * gauss[i];
            let val_y = 2.0 * std::f32::consts::PI * y * gauss[npf + i];
            out[i] = val_x.sin();
            out[npf + i] = val_x.cos();
            out[2 * npf + i] = val_y.sin();
            out[3 * npf + i] = val_y.cos();
        }
        out
    }

    /// Generate the dense positional encoding grid for the image embedding.
    ///
    /// Returns `[embed_dim, grid_h, grid_w]` where each spatial position gets
    /// a Fourier-encoded positional embedding.
    pub fn encode_grid(&self, grid_h: usize, grid_w: usize) -> Vec<f32> {
        let embed_dim = 4 * self.num_pos_feats;
        let mut grid = vec![0.0f32; embed_dim * grid_h * grid_w];

        for r in 0..grid_h {
            let y = (r as f32 + 0.5) / grid_h as f32;
            for c in 0..grid_w {
                let x = (c as f32 + 0.5) / grid_w as f32;
                let pe = self.encode_point(x, y);
                for d in 0..embed_dim {
                    grid[d * grid_h * grid_w + r * grid_w + c] = pe[d];
                }
            }
        }

        grid
    }
}

impl<B: MathBackend> Module<B> for PositionalEncoding<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.gaussian_matrix]
    }
}

// ── Prompt encoder ──────────────────────────────────────────────────────────

/// SAM prompt encoder.
///
/// Encodes point, box, and mask prompts into sparse and dense embeddings
/// consumed by the mask decoder.
pub struct PromptEncoder<B: MathBackend> {
    pub pe: PositionalEncoding<B>,
    /// Learned embeddings for point types: `[4][embed_dim]`.
    /// Index: 0=background, 1=foreground, 2=box top-left, 3=box bottom-right.
    pub point_embeddings: [Tensor<B>; 4],
    /// "Not a point" embedding: `[embed_dim]`.
    pub not_a_point_embed: Tensor<B>,
    /// Mask downscaling convolutions (for mask-prompt input).
    pub mask_conv1: Conv2d<B>,
    pub mask_ln1: LayerNorm2d<B>,
    pub mask_conv2: Conv2d<B>,
    pub mask_ln2: LayerNorm2d<B>,
    pub mask_conv3: Conv2d<B>,
    /// "No mask" embedding: `[embed_dim]`.
    pub no_mask_embed: Tensor<B>,
    pub embed_dim: usize,
    pub image_embedding_size: usize,
}

impl<B: MathBackend> PromptEncoder<B> {
    /// Create a zero-initialized prompt encoder from config.
    pub fn new(config: &SamConfig) -> Self {
        let embed_dim = config.out_channels; // 256
        let grid = config.num_patches_per_side();
        let npf = embed_dim / 4; // 64

        let make_embed = || Tensor::from_vec(vec![0.0; embed_dim], Shape::new(&[embed_dim]));

        Self {
            pe: PositionalEncoding::new(npf),
            point_embeddings: [make_embed(), make_embed(), make_embed(), make_embed()],
            not_a_point_embed: make_embed(),
            mask_conv1: Conv2d::square(1, embed_dim / 64, 2, 2, 0), // 4 channels
            mask_ln1: LayerNorm2d::new(embed_dim / 64, 1e-6),
            mask_conv2: Conv2d::square(embed_dim / 64, embed_dim / 16, 2, 2, 0), // 16 channels
            mask_ln2: LayerNorm2d::new(embed_dim / 16, 1e-6),
            mask_conv3: Conv2d::square(embed_dim / 16, embed_dim, 1, 1, 0),
            no_mask_embed: make_embed(),
            embed_dim,
            image_embedding_size: grid,
        }
    }

    /// Encode a prompt into sparse and dense embeddings.
    ///
    /// Returns `(sparse_embeddings, dense_embeddings)`:
    /// - sparse: `[num_points, embed_dim]` — one embedding per prompt point
    /// - dense: `[embed_dim, grid_h, grid_w]` — spatial dense embedding
    pub fn forward(&self, prompt: &SegmentPrompt) -> (Tensor<B>, Tensor<B>) {
        let grid = self.image_embedding_size;
        let ed = self.embed_dim;

        // Dense embedding: broadcast no_mask_embed to [embed_dim, grid, grid]
        let no_mask = self.no_mask_embed.to_vec();
        let mut dense = vec![0.0f32; ed * grid * grid];
        for d in 0..ed {
            let val = no_mask[d];
            for i in 0..grid * grid {
                dense[d * grid * grid + i] = val;
            }
        }
        let dense_tensor = Tensor::from_vec(dense, Shape::new(&[ed, grid, grid]));

        // Sparse embeddings depend on prompt type
        let sparse_tensor = match prompt {
            SegmentPrompt::Point { x, y } => self.encode_points(&[(*x, *y, true)], grid),
            SegmentPrompt::PointWithLabel { x, y, foreground } => {
                self.encode_points(&[(*x, *y, *foreground)], grid)
            }
            SegmentPrompt::Points(points) => self.encode_points(points, grid),
            SegmentPrompt::Box(bbox) => {
                // Box is encoded as two points: top-left and bottom-right
                let tl_pe = self.pe.encode_point(
                    bbox.x1 / (self.image_embedding_size as f32 * 16.0),
                    bbox.y1 / (self.image_embedding_size as f32 * 16.0),
                );
                let br_pe = self.pe.encode_point(
                    bbox.x2 / (self.image_embedding_size as f32 * 16.0),
                    bbox.y2 / (self.image_embedding_size as f32 * 16.0),
                );

                let tl_embed = self.point_embeddings[2].to_vec();
                let br_embed = self.point_embeddings[3].to_vec();

                let mut sparse = vec![0.0f32; 2 * ed];
                for d in 0..ed {
                    sparse[d] = tl_pe[d] + tl_embed[d];
                    sparse[ed + d] = br_pe[d] + br_embed[d];
                }

                Tensor::from_vec(sparse, Shape::new(&[2, ed]))
            }
        };

        (sparse_tensor, dense_tensor)
    }

    fn encode_points(&self, points: &[(f32, f32, bool)], grid: usize) -> Tensor<B> {
        let ed = self.embed_dim;
        let image_size = grid as f32 * 16.0; // grid * patch_size
        let mut sparse = vec![0.0f32; points.len() * ed];

        for (i, &(x, y, fg)) in points.iter().enumerate() {
            let pe = self.pe.encode_point(x / image_size, y / image_size);
            let type_embed = if fg {
                self.point_embeddings[1].to_vec() // foreground
            } else {
                self.point_embeddings[0].to_vec() // background
            };

            for d in 0..ed {
                sparse[i * ed + d] = pe[d] + type_embed[d];
            }
        }

        Tensor::from_vec(sparse, Shape::new(&[points.len(), ed]))
    }

    /// Get the dense positional encoding for the image grid.
    ///
    /// Returns `[embed_dim, grid_h, grid_w]`.
    pub fn get_dense_pe(&self) -> Tensor<B> {
        let grid = self.image_embedding_size;
        let pe_data = self.pe.encode_grid(grid, grid);
        Tensor::from_vec(pe_data, Shape::new(&[self.embed_dim, grid, grid]))
    }
}

impl<B: MathBackend> Module<B> for PromptEncoder<B> {
    fn parameters(&self) -> Vec<&Tensor<B>> {
        let mut params = self.pe.parameters();
        for emb in &self.point_embeddings {
            params.push(emb);
        }
        params.push(&self.not_a_point_embed);
        params.extend(self.mask_conv1.parameters());
        params.extend(self.mask_ln1.parameters());
        params.extend(self.mask_conv2.parameters());
        params.extend(self.mask_ln2.parameters());
        params.extend(self.mask_conv3.parameters());
        params.push(&self.no_mask_embed);
        params
    }
}

// ── Safetensors loading ─────────────────────────────────────────────────────

#[cfg(feature = "safetensors")]
impl<B: MathBackend> PromptEncoder<B> {
    /// Load a SAM prompt encoder from safetensors.
    ///
    /// `prefix` is typically `"prompt_encoder."`.
    pub fn from_safetensors(
        config: &SamConfig,
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
    ) -> crate::error::Result<Self> {
        use crate::checkpoint::{load_conv2d_with_bias, load_layernorm2d, load_tensor};

        let ed = config.out_channels;
        let grid = config.num_patches_per_side();
        let npf = ed / 4;

        let gaussian_matrix = load_tensor(
            tensors,
            &format!("{prefix}pe_layer.positional_encoding_gaussian_matrix"),
            &[2, npf],
        )?;

        let point_embeddings = [
            load_tensor(
                tensors,
                &format!("{prefix}point_embeddings.0.weight"),
                &[ed],
            )?,
            load_tensor(
                tensors,
                &format!("{prefix}point_embeddings.1.weight"),
                &[ed],
            )?,
            load_tensor(
                tensors,
                &format!("{prefix}point_embeddings.2.weight"),
                &[ed],
            )?,
            load_tensor(
                tensors,
                &format!("{prefix}point_embeddings.3.weight"),
                &[ed],
            )?,
        ];

        let not_a_point_embed =
            load_tensor(tensors, &format!("{prefix}not_a_point_embed.weight"), &[ed])?;

        // Mask downscaling: Sequential(Conv2d, LN, GELU, Conv2d, LN, GELU, Conv2d)
        // Keys: mask_downscaling.0 (conv), .1 (LN), .3 (conv), .4 (LN), .6 (conv)
        let ch1 = ed / 64; // 4
        let ch2 = ed / 16; // 16
        let mask_conv1 = load_conv2d_with_bias(
            tensors,
            &format!("{prefix}mask_downscaling.0"),
            1,
            ch1,
            2,
            2,
            2,
            0,
        )?;
        let mask_ln1 =
            load_layernorm2d(tensors, &format!("{prefix}mask_downscaling.1"), ch1, 1e-6)?;
        let mask_conv2 = load_conv2d_with_bias(
            tensors,
            &format!("{prefix}mask_downscaling.3"),
            ch1,
            ch2,
            2,
            2,
            2,
            0,
        )?;
        let mask_ln2 =
            load_layernorm2d(tensors, &format!("{prefix}mask_downscaling.4"), ch2, 1e-6)?;
        let mask_conv3 = load_conv2d_with_bias(
            tensors,
            &format!("{prefix}mask_downscaling.6"),
            ch2,
            ed,
            1,
            1,
            1,
            0,
        )?;

        let no_mask_embed = load_tensor(tensors, &format!("{prefix}no_mask_embed.weight"), &[ed])?;

        Ok(Self {
            pe: PositionalEncoding {
                gaussian_matrix,
                num_pos_feats: npf,
            },
            point_embeddings,
            not_a_point_embed,
            mask_conv1,
            mask_ln1,
            mask_conv2,
            mask_ln2,
            mask_conv3,
            no_mask_embed,
            embed_dim: ed,
            image_embedding_size: grid,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    fn tiny_config() -> SamConfig {
        SamConfig {
            image_size: 64,
            patch_size: 16,
            embed_dim: 32,
            depth: 1,
            num_heads: 2,
            mlp_ratio: 2.0,
            global_attn_indices: vec![0],
            window_size: 2,
            out_channels: 16,
            num_multimask_outputs: 3,
            iou_head_depth: 2,
            iou_head_hidden_dim: 16,
            decoder_depth: 1,
            decoder_num_heads: 2,
        }
    }

    #[test]
    fn point_encoding_shape() {
        let config = tiny_config();
        let pe = PromptEncoder::<CpuBackend>::new(&config);
        let prompt = SegmentPrompt::Point { x: 32.0, y: 32.0 };
        let (sparse, dense) = pe.forward(&prompt);
        assert_eq!(sparse.shape.dims(), &[1, 16]); // 1 point, embed_dim=16 (out_channels)
        assert_eq!(dense.shape.dims(), &[16, 4, 4]); // embed_dim, grid, grid
    }

    #[test]
    fn box_encoding_shape() {
        let config = tiny_config();
        let pe = PromptEncoder::<CpuBackend>::new(&config);
        let bbox = crate::postprocess::nms::BBox::new(10.0, 10.0, 50.0, 50.0);
        let prompt = SegmentPrompt::Box(bbox);
        let (sparse, dense) = pe.forward(&prompt);
        assert_eq!(sparse.shape.dims(), &[2, 16]); // 2 points (corners)
        assert_eq!(dense.shape.dims(), &[16, 4, 4]);
    }

    #[test]
    fn multi_point_encoding_shape() {
        let config = tiny_config();
        let pe = PromptEncoder::<CpuBackend>::new(&config);
        let prompt = SegmentPrompt::Points(vec![
            (10.0, 10.0, true),
            (20.0, 20.0, false),
            (30.0, 30.0, true),
        ]);
        let (sparse, _) = pe.forward(&prompt);
        assert_eq!(sparse.shape.dims(), &[3, 16]);
    }

    #[test]
    fn fourier_features_bounded() {
        let pe = PositionalEncoding::<CpuBackend>::new(64);
        // Even with zero Gaussian matrix, sin/cos outputs are bounded
        let enc = pe.encode_point(0.5, 0.5);
        assert_eq!(enc.len(), 256);
        for &v in &enc {
            assert!(v >= -1.0 && v <= 1.0, "Fourier feature out of bounds: {v}");
        }
    }

    #[test]
    fn dense_pe_grid_shape() {
        let config = tiny_config();
        let pe = PromptEncoder::<CpuBackend>::new(&config);
        let dense_pe = pe.get_dense_pe();
        assert_eq!(dense_pe.shape.dims(), &[16, 4, 4]);
    }
}
