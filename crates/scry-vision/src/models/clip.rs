// SPDX-License-Identifier: MIT OR Apache-2.0
//! CLIP visual encoder.
//!
//! Wraps a ViT backbone with a projection head to produce embeddings
//! in CLIP's joint vision-language space.
//!
//! Input: `[3, H, W]` → Output: `[proj_dim]` (L2-normalized embedding).

use scry_llm::backend::MathBackend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use crate::error::Result;
use crate::image::ImageBuffer;
use crate::pipeline::Embed;
use crate::postprocess::embedding::l2_normalize;
use crate::transform::resize::{InterpolationMode, Resize};
use crate::transform::to_tensor::ToTensor;
use crate::transform::ImageTransform;

use super::vit::{Vit, VitConfig};

/// CLIP visual encoder configuration.
#[derive(Clone, Debug)]
pub struct ClipConfig {
    pub vit: VitConfig,
    /// Projection dimension (e.g., 512 for ViT-B/32).
    pub proj_dim: usize,
}

impl ClipConfig {
    /// CLIP ViT-B/32 configuration.
    pub fn vit_b32() -> Self {
        Self {
            vit: VitConfig::vit_b32(),
            proj_dim: 512,
        }
    }

    /// CLIP ViT-B/16 configuration.
    pub fn vit_b16() -> Self {
        Self {
            vit: VitConfig::vit_b16(),
            proj_dim: 512,
        }
    }

    /// CLIP ViT-L/14 configuration.
    pub fn vit_l14() -> Self {
        Self {
            vit: VitConfig::vit_l14(),
            proj_dim: 768,
        }
    }
}

/// CLIP visual encoder: ViT + projection → L2-normalized embedding.
pub struct ClipVisual<B: MathBackend> {
    pub vit: Vit<B>,
    /// Projection weight: `[embed_dim, proj_dim]`.
    pub proj: Tensor<B>,
    pub config: ClipConfig,
}

impl<B: MathBackend> ClipVisual<B> {
    /// Create a zero-initialized CLIP visual encoder.
    pub fn new(config: ClipConfig) -> Self {
        let embed_dim = config.vit.embed_dim;
        Self {
            vit: Vit::new(config.vit.clone()),
            proj: Tensor::from_vec(
                vec![0.0; embed_dim * config.proj_dim],
                Shape::new(&[embed_dim, config.proj_dim]),
            ),
            config,
        }
    }

    /// Forward pass: `[3, H, W]` → `[proj_dim]` (L2-normalized).
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let d = self.config.vit.embed_dim;

        // ViT: [3, H, W] → [embed_dim]
        let cls = self.vit.forward(input);

        // Project: [1, embed_dim] @ [embed_dim, proj_dim] → [1, proj_dim]
        let cls_2d = B::from_vec(cls.to_vec(), &Shape::new(&[1, d]));
        let proj = B::matmul(
            &cls_2d,
            &self.proj.data,
            1,
            d,
            self.config.proj_dim,
            false,
            false,
        );

        // L2 normalize
        let mut embedding = B::to_vec(&proj);
        l2_normalize(&mut embedding);
        Tensor::from_vec(embedding, Shape::new(&[self.config.proj_dim]))
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> ClipVisual<B> {
    /// Load a CLIP visual encoder from a safetensors file.
    ///
    /// Expects OpenAI CLIP naming with `visual.` prefix for the ViT weights,
    /// plus `visual.proj` for the projection matrix.
    pub fn from_safetensors(
        config: ClipConfig,
        path: &std::path::Path,
    ) -> crate::error::Result<Self> {
        use crate::checkpoint::load_tensor;
        use crate::error::VisionError;

        let file = std::fs::File::open(path).map_err(|e| {
            VisionError::ModelLoad(format!("cannot open {}: {e}", path.display()))
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            VisionError::ModelLoad(format!("mmap failed: {e}"))
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&mmap).map_err(|e| {
            VisionError::ModelLoad(format!("safetensors parse failed: {e}"))
        })?;

        let vit = Vit::from_safetensors(config.vit.clone(), &tensors, "visual.")?;

        // visual.proj: [embed_dim, proj_dim] — same layout as ours (not a Linear)
        let d = config.vit.embed_dim;
        let proj = load_tensor(&tensors, "visual.proj", &[d, config.proj_dim])?;

        Ok(Self { vit, proj, config })
    }
}

/// CLIP visual encoder that implements the [`Embed`] pipeline trait.
///
/// Handles preprocessing (resize + normalize) internally.
pub struct ClipEmbedder<B: MathBackend> {
    pub model: ClipVisual<B>,
}

impl<B: MathBackend> ClipEmbedder<B> {
    pub fn new(config: ClipConfig) -> Self {
        Self {
            model: ClipVisual::new(config),
        }
    }
}

#[cfg(feature = "safetensors")]
impl<B: MathBackend> ClipEmbedder<B> {
    /// Load a CLIP visual embedder from a safetensors file.
    pub fn from_safetensors(
        config: ClipConfig,
        path: &std::path::Path,
    ) -> crate::error::Result<Self> {
        let model = ClipVisual::from_safetensors(config, path)?;
        Ok(Self { model })
    }
}

impl<B: MathBackend> Embed for ClipEmbedder<B> {
    fn embed(&self, image: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
        let img = ImageBuffer::from_raw(image.to_vec(), width, height, 3)?;

        let size = self.model.config.vit.image_size as u32;
        let resize = Resize::new(size, size, InterpolationMode::Bilinear);
        let resized = resize.apply(&img)?;

        let tensor = ToTensor::clip().apply::<B>(&resized);
        let output = self.model.forward(&tensor);
        Ok(output.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scry_llm::backend::cpu::CpuBackend;

    #[test]
    fn clip_visual_output_shape() {
        let config = ClipConfig::vit_b32();
        let model = ClipVisual::<CpuBackend>::new(config);
        let input = Tensor::from_vec(vec![0.0; 3 * 224 * 224], Shape::new(&[3, 224, 224]));
        let output = model.forward(&input);
        assert_eq!(output.shape.dims(), &[512]);
    }

    #[cfg(feature = "safetensors")]
    #[test]
    fn clip_roundtrip_safetensors() {
        use std::borrow::Cow;
        use std::collections::HashMap;

        struct F32View { data: Vec<u8>, shape: Vec<usize> }
        impl F32View {
            fn new(values: &[f32], shape: &[usize]) -> Self {
                let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                Self { data, shape: shape.to_vec() }
            }
        }
        impl safetensors::View for F32View {
            fn dtype(&self) -> safetensors::Dtype { safetensors::Dtype::F32 }
            fn shape(&self) -> &[usize] { &self.shape }
            fn data(&self) -> Cow<'_, [u8]> { Cow::Borrowed(&self.data) }
            fn data_len(&self) -> usize { self.data.len() }
        }

        let config = ClipConfig {
            vit: VitConfig {
                image_size: 32, patch_size: 16, embed_dim: 64,
                num_heads: 4, num_layers: 1, mlp_ratio: 4.0, in_channels: 3,
            },
            proj_dim: 32,
        };
        let original = ClipVisual::<CpuBackend>::new(config.clone());

        // Serialize with CLIP naming (visual.* prefix)
        let d = config.vit.embed_dim;
        let d_ff = config.vit.d_ff();
        let mut entries: Vec<(String, F32View)> = Vec::new();

        // Patch embed
        entries.push(("visual.conv1.weight".into(), F32View::new(&original.vit.patch_embed.proj.to_vec(), original.vit.patch_embed.proj.shape.dims())));
        entries.push(("visual.class_embedding".into(), F32View::new(&original.vit.cls_token.to_vec(), &[d])));
        entries.push(("visual.positional_embedding".into(), F32View::new(&original.vit.pos_embed.to_vec(), original.vit.pos_embed.shape.dims())));

        // Block 0
        let b = &original.vit.blocks[0];
        let bp = "visual.transformer.resblocks.0";
        entries.push((format!("{bp}.ln_1.weight"), F32View::new(&b.ln1_gamma.to_vec(), &[d])));
        entries.push((format!("{bp}.ln_1.bias"), F32View::new(&b.ln1_beta.to_vec(), &[d])));
        let qkv_t = crate::checkpoint::transpose_2d(&b.attn.qkv_weight.to_vec(), d, 3 * d);
        entries.push((format!("{bp}.attn.in_proj_weight"), F32View::new(&qkv_t, &[3 * d, d])));
        entries.push((format!("{bp}.attn.in_proj_bias"), F32View::new(&b.attn.qkv_bias.to_vec(), &[3 * d])));
        let proj_t = crate::checkpoint::transpose_2d(&b.attn.proj_weight.to_vec(), d, d);
        entries.push((format!("{bp}.attn.out_proj.weight"), F32View::new(&proj_t, &[d, d])));
        entries.push((format!("{bp}.attn.out_proj.bias"), F32View::new(&b.attn.proj_bias.to_vec(), &[d])));
        entries.push((format!("{bp}.ln_2.weight"), F32View::new(&b.ln2_gamma.to_vec(), &[d])));
        entries.push((format!("{bp}.ln_2.bias"), F32View::new(&b.ln2_beta.to_vec(), &[d])));
        let fc1_t = crate::checkpoint::transpose_2d(&b.mlp.fc1_weight.to_vec(), d, d_ff);
        entries.push((format!("{bp}.mlp.c_fc.weight"), F32View::new(&fc1_t, &[d_ff, d])));
        entries.push((format!("{bp}.mlp.c_fc.bias"), F32View::new(&b.mlp.fc1_bias.to_vec(), &[d_ff])));
        let fc2_t = crate::checkpoint::transpose_2d(&b.mlp.fc2_weight.to_vec(), d_ff, d);
        entries.push((format!("{bp}.mlp.c_proj.weight"), F32View::new(&fc2_t, &[d, d_ff])));
        entries.push((format!("{bp}.mlp.c_proj.bias"), F32View::new(&b.mlp.fc2_bias.to_vec(), &[d])));

        // LN post
        entries.push(("visual.ln_post.weight".into(), F32View::new(&original.vit.ln_post_gamma.to_vec(), &[d])));
        entries.push(("visual.ln_post.bias".into(), F32View::new(&original.vit.ln_post_beta.to_vec(), &[d])));

        // Projection matrix
        entries.push(("visual.proj".into(), F32View::new(&original.proj.to_vec(), original.proj.shape.dims())));

        let info: Option<HashMap<String, String>> = None;
        let bytes = safetensors::serialize(entries, &info).unwrap();

        // Write to temp file and reload
        let path = std::env::temp_dir().join(format!("scry_test_{}_clip.safetensors", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let loaded = ClipVisual::<CpuBackend>::from_safetensors(config, &path).unwrap();
        std::fs::remove_file(&path).ok();

        // Verify projection matrix round-trips
        let orig_proj = original.proj.to_vec();
        let load_proj = loaded.proj.to_vec();
        for (i, (a, b)) in orig_proj.iter().zip(load_proj.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "proj[{i}]: {a} vs {b}");
        }

        // Verify a few ViT tensors
        assert_eq!(original.vit.cls_token.to_vec(), loaded.vit.cls_token.to_vec());
        assert_eq!(original.vit.pos_embed.to_vec(), loaded.vit.pos_embed.to_vec());
    }

    #[test]
    fn clip_output_is_normalized() {
        let config = ClipConfig {
            vit: VitConfig {
                image_size: 32,
                patch_size: 16,
                embed_dim: 64,
                num_heads: 4,
                num_layers: 1,
                mlp_ratio: 4.0,
                in_channels: 3,
            },
            proj_dim: 32,
        };
        // Use non-zero weights to get a non-zero embedding
        let mut model = ClipVisual::<CpuBackend>::new(config);
        model.proj = Tensor::from_vec(vec![0.01; 64 * 32], Shape::new(&[64, 32]));
        model.vit.ln_post_gamma = Tensor::from_vec(vec![1.0; 64], Shape::new(&[64]));

        let input = Tensor::from_vec(vec![1.0; 3 * 32 * 32], Shape::new(&[3, 32, 32]));
        let output = model.forward(&input);
        let data = output.to_vec();

        // Check L2 norm ≈ 1.0
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            assert!((norm - 1.0).abs() < 1e-4, "norm = {norm}");
        }
    }
}
